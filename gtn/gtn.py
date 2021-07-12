import math
from typing import Dict, Union

import torch
import torch.nn as nn
from lcn.cost import distances
from lcn.cost.cost_matrix import get_cost_matrix
from lcn.lcn_sinkhorn import LogLCNSinkhorn, LogLCNSinkhornBP
from lcn.nystrom_sinkhorn import LogNystromSinkhorn, LogNystromSinkhornBP
from lcn.sinkhorn import LogSinkhorn
from lcn.sinkhorn_unbalanced import LogSinkhornBP, entropyRegLogSinkhorn
from lcn.sparse_sinkhorn import LogSparseSinkhorn, LogSparseSinkhornBP
from lcn.utils import call_with_filtered_args
from torch_geometric.data import Data

from gtn.model import aggregation


class GTN(nn.Module):
    def __init__(
        self,
        gnn: nn.Module,
        emb_dist_scale: float,
        device: torch.device,
        sinkhorn_reg: float = 0.1,
        sinkhorn_niter: int = 50,
        unbalanced_mode: dict = {
            "name": "bp_matrix"
        },  # balanced, bp_matrix, entropy_reg
        nystrom: dict = None,
        sparse: dict = None,
        extensive: bool = True,
        num_heads: int = 1,
        multihead_scale_basis: float = 1.0,
        similarity: bool = False,
    ):
        super().__init__()
        self.gnn = gnn
        self.device = device
        self.sinkhorn_reg = torch.tensor(
            sinkhorn_reg * emb_dist_scale, dtype=torch.float32, device=self.device
        )
        self.unbalanced_mode = unbalanced_mode
        self.nystrom = nystrom.copy() if nystrom else None
        self.sparse = sparse.copy() if sparse else None
        self.extensive = extensive
        self.num_heads = num_heads
        self.multihead_scale_basis = multihead_scale_basis
        self.similarity = similarity

        if self.sparse:
            self.sparse["num_hashes"] = self.sparse.get("num_hashes", 1)

        if isinstance(gnn.layer_aggregation, aggregation.MLP):
            gnn_output_size = gnn.emb_size
        else:
            gnn_output_size = gnn.emb_size * (len(gnn.layers) + 1)
        self.match_emb_size = gnn_output_size

        # Rescale embeddings so the predicted distance is in the right range at initialization
        if self.similarity:
            emb_dist_scale = -math.log(emb_dist_scale)
        self.emb_scale = emb_dist_scale / math.sqrt(self.match_emb_size)

        # Alpha for determining cost of insertion/deletion
        self.alpha = nn.Parameter(torch.empty(self.match_emb_size))

        if self.num_heads == 1:
            self.output_layer = nn.Linear(1, 1, bias=True)
        else:
            self.dist_transform = nn.Linear(
                self.match_emb_size, self.num_heads * self.match_emb_size, bias=False
            )
            self.output_layer = nn.Sequential(
                nn.Linear(self.num_heads, gnn.emb_size, bias=True),
                nn.LeakyReLU(),
                nn.Linear(gnn.emb_size, 1, bias=True),
            )

        self.sinkhorn_niter = sinkhorn_niter

        self.reset_parameters()
        self.to(device)

    def output_transform(self, x, cost_mat_len):
        if not self.extensive:
            if x.dim() == 1:
                x = x / cost_mat_len
            else:
                x = x / cost_mat_len[:, None]
        if x.dim() == 1:
            x = x[:, None]
        out = self.output_layer(x).squeeze(-1)
        if self.similarity:
            return torch.exp(-out)
        else:
            return out

    def reset_parameters(self):
        nn.init.ones_(self.alpha)
        if self.num_heads == 1:
            nn.init.ones_(self.output_layer.weight)
            nn.init.zeros_(self.output_layer.bias)

    def _compute_node_embeddings(self, graph1, graph2):
        node_embeddings_raw = []
        b_x_nnodes = graph1["attr_matrix"].shape[:2]
        nfeat = graph1["attr_matrix"].shape[2:]
        for graph in [graph1, graph2]:
            x = graph["attr_matrix"].view((-1, *nfeat))
            data = Data(
                x=x, edge_index=graph["adj_idx"], edge_attr=graph["edge_attr_matrix"]
            )
            node_embeddings_raw.append(self.gnn(data).view(*b_x_nnodes, -1))

        # Scale embeddings
        node_embeddings = [embs * self.emb_scale for embs in node_embeddings_raw]

        return node_embeddings

    def _compute_matching(self, cost_mat, cost_mat_len, sparse=None, nystrom=None):
        sinkhorn_input_dict = cost_mat.copy()

        if (sparse or nystrom) and self.unbalanced_mode["name"] not in [
            "bp_matrix",
            "balanced",
        ]:
            raise NotImplementedError(
                "Sinkhorn approximations only implemented for the BP matrix and regular (balanced) Sinkhorn."
            )

        if sparse and nystrom:
            if self.unbalanced_mode["name"] == "bp_matrix":
                sinkhorn_fn = LogLCNSinkhornBP
            else:
                sinkhorn_fn = LogLCNSinkhorn
        elif sparse:
            if self.unbalanced_mode["name"] == "bp_matrix":
                sinkhorn_fn = LogSparseSinkhornBP
            else:
                sinkhorn_fn = LogSparseSinkhorn
        elif nystrom:
            if self.unbalanced_mode["name"] == "bp_matrix":
                sinkhorn_fn = LogNystromSinkhornBP
            else:
                sinkhorn_fn = LogNystromSinkhorn
        else:
            if self.unbalanced_mode["name"] == "bp_matrix":
                if self.unbalanced_mode.get("full_bp", False):
                    sinkhorn_input_dict.num_points = cost_mat_len
                    sinkhorn_input_dict.offset_entropy = True
                    sinkhorn_fn = LogSinkhorn
                else:
                    sinkhorn_fn = LogSinkhornBP
            elif self.unbalanced_mode["name"] == "entropy_reg":
                sinkhorn_input_dict.reg_marginal = (
                    sinkhorn_input_dict.sinkhorn_reg
                    * self.unbalanced_mode["marginal_reg"]
                )
                sinkhorn_input_dict.offset_entropy = True
                sinkhorn_fn = entropyRegLogSinkhorn
            elif self.unbalanced_mode["name"] == "balanced":
                sinkhorn_input_dict.offset_entropy = True
                sinkhorn_fn = LogSinkhorn
            else:
                raise NotImplementedError(
                    f"Unrecognized unbalanced mode '{self.unbalanced_mode['name']}'"
                )

        return call_with_filtered_args(
            sinkhorn_fn, **sinkhorn_input_dict, niter=self.sinkhorn_niter
        )

    def forward(
        self,
        graph1: Dict[str, Union[torch.Tensor, torch.sparse.FloatTensor]],
        graph2: Dict[str, Union[torch.Tensor, torch.sparse.FloatTensor]],
    ):

        num_nodes = torch.stack((graph1["num_nodes"], graph2["num_nodes"]))
        if self.unbalanced_mode["name"] == "bp_matrix" or self.nystrom or self.sparse:
            cost_mat_len = num_nodes.sum(0)
        else:
            cost_mat_len = num_nodes.max(0).values

        node_embeddings = self._compute_node_embeddings(graph1, graph2)

        _, max_nodes, _ = node_embeddings[0].shape

        if self.num_heads > 1:
            node_match_embs = []
            node_match_embs.append(
                torch.cat(
                    self.dist_transform(node_embeddings[0]).split(
                        self.match_emb_size, dim=-1
                    ),
                    dim=0,
                )
            )
            node_match_embs.append(
                torch.cat(
                    self.dist_transform(node_embeddings[1]).split(
                        self.match_emb_size, dim=-1
                    ),
                    dim=0,
                )
            )
            cost_mat_len_rep = cost_mat_len.repeat(self.num_heads)
            num_nodes_rep = num_nodes.repeat(1, self.num_heads)
        else:
            node_match_embs = node_embeddings
            cost_mat_len_rep = cost_mat_len
            num_nodes_rep = num_nodes

        if self.unbalanced_mode["name"] == "bp_matrix":
            sinkhorn_reg_scaled = self.sinkhorn_reg / torch.log(
                num_nodes_rep.min(0).values.float() + 1
            )
        else:
            sinkhorn_reg_scaled = self.sinkhorn_reg / torch.log(
                cost_mat_len_rep.float()
            )

        if self.num_heads > 1 and self.multihead_scale_basis != 1.0:
            factor = torch.repeat_interleave(
                self.multihead_scale_basis
                ** torch.arange(
                    -self.num_heads // 2,
                    self.num_heads // 2,
                    dtype=torch.float,
                    device=self.device,
                ),
                num_nodes.size(1),
            )
            sinkhorn_reg_scaled *= factor

        cost_matrix = get_cost_matrix(
            node_match_embs,
            num_nodes_rep,
            nystrom=self.nystrom,
            sparse=self.sparse,
            sinkhorn_reg=sinkhorn_reg_scaled,
            sinkhorn_niter=self.sinkhorn_niter,
            alpha=self.alpha,
            dist=distances.PNorm(p=2),
            bp_cost_matrix=(self.unbalanced_mode["name"] == "bp_matrix"),
            full_bp_matrix=self.unbalanced_mode.get("full_bp", False),
        )

        output = self._compute_matching(
            cost_matrix, cost_mat_len_rep, sparse=self.sparse, nystrom=self.nystrom
        )

        if self.num_heads > 1:
            output = output.reshape(
                self.num_heads, num_nodes.shape[1], *output.shape[1:]
            ).transpose(0, 1)

        return self.output_transform(output, cost_mat_len)
