from typing import Callable, Type

import torch
import torch.nn as nn

from gtn.model.gcn import GCNConv


class Net(nn.Module):
    def __init__(
        self,
        node_feat_size: int,
        edge_feat_size: int,
        emb_size: int,
        nlayers: int,
        layer_aggregation: Type[nn.Module],
        device: torch.device,
        act_fn: Callable[[torch.Tensor], torch.Tensor] = lambda x: x,
        use_linear: bool = True,
        gcn_aggr: str = "add",
        avg_degree: float = 1.0,
        deg_norm_hidden: bool = False,
    ):
        super().__init__()
        self.emb_size = emb_size
        self.device = device
        self.act_fn = act_fn
        self.layer_aggregation = layer_aggregation
        self.avg_degree = avg_degree

        if edge_feat_size > 0:
            self.bilin = nn.Bilinear(emb_size, edge_feat_size, emb_size, bias=False)
        else:
            self.bilin = None

        # Input layer
        # Sum aggregation ensures that the degree is represented
        self.layers = [
            GCNConv(
                node_feat_size,
                emb_size,
                act_fn,
                self.bilin,
                use_linear=use_linear,
                aggr=gcn_aggr,
                avg_degree=avg_degree,
                deg_norm=False,
            )
        ]

        # Hidden & output layers
        for _ in range(nlayers - 1):
            self.layers.append(
                GCNConv(
                    emb_size,
                    emb_size,
                    act_fn,
                    self.bilin,
                    use_linear=use_linear,
                    aggr=gcn_aggr,
                    avg_degree=avg_degree,
                    deg_norm=deg_norm_hidden,
                )
            )

        # Add fcs-list to module
        for i in range(nlayers):
            self.add_module(f"layer_{i}", self.layers[i])

        self.to(self.device)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        if not self.bilin:
            edge_attr = None

        node_embeddings = []
        for ilayer, layer in enumerate(self.layers):
            y, x = layer(x, edge_index, edge_weight=edge_attr)
            if ilayer == 0:
                node_embeddings.append(y)
            x = x + y
            node_embeddings.append(x)

        return self.layer_aggregation(node_embeddings)
