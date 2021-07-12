import numpy as np
import torch


class GraphBatchCollator(object):
    def __call__(self, data_batch):
        graphs1 = []
        graphs2 = []
        max_nnodes = 0
        dists = torch.empty(len(data_batch))
        for i, (graph1, graph2, dist) in enumerate(data_batch):
            graphs1.append(graph1)
            graphs2.append(graph2)
            dists[i] = dist
            max_nnodes = max(
                max_nnodes,
                graph1["attr_matrix"].shape[0],
                graph2["attr_matrix"].shape[0],
            )
        return (
            self._collate_graphs(graphs1, max_nnodes),
            self._collate_graphs(graphs2, max_nnodes),
            dists,
        )

    def _collate_graphs(self, graph_batch, max_nnodes):
        num_nodes = torch.LongTensor(
            [graph["attr_matrix"].shape[0] for graph in graph_batch]
        )
        if graph_batch[0]["edge_attr_matrix"] is None:
            edge_attr_matrix = torch.zeros(0)
        else:
            edge_attr_matrix = torch.cat(
                [graph["edge_attr_matrix"] for graph in graph_batch], dim=0
            )
        graphsdict = {
            "adj_idx": self._collate_adjidx(
                [graph["adj_idx"] for graph in graph_batch], max_nnodes
            ),
            "attr_matrix": self._pad_matrices(
                [graph["attr_matrix"] for graph in graph_batch], max_nnodes
            ),
            "edge_attr_matrix": edge_attr_matrix,
            "num_nodes": num_nodes,
        }
        return graphsdict

    def _pad_matrices(self, mats, max_len):
        trailing_dims = mats[0].shape[1:]
        out_dims = (len(mats), max_len) + trailing_dims
        out_tensor = mats[0].new_zeros(*out_dims)
        for i, tensor in enumerate(mats):
            length = tensor.size(0)
            out_tensor[i, :length, ...] = tensor
        return out_tensor

    def _collate_adjidx(self, adj_idxs, max_nnodes):
        batch_size = len(adj_idxs)
        nedges = [adj_idx.size(1) for adj_idx in adj_idxs]
        batch_idx_offset = torch.LongTensor(
            np.repeat(np.arange(batch_size) * max_nnodes, nedges)
        )
        # b*e
        batch_edge_idx = torch.cat(adj_idxs, dim=1)
        # 2 x b*e
        batch_adj_idx = batch_edge_idx + batch_idx_offset
        # 2 x b*e
        return batch_adj_idx
