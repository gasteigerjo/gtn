import numpy as np
import torch
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import Dataset


class GraphDistDataset(Dataset):
    def __init__(
        self,
        graph_collection,
        nattrs,
        edge_nattrs,
        node_onehot=True,
        edge_onehot=False,
        pair_idx=None,
    ):
        self.nattrs = nattrs
        self.edge_nattrs = edge_nattrs
        self.graphs = [
            self._sparsegraph_to_dict(graph, node_onehot, edge_onehot)
            for graph in graph_collection
        ]
        self.dists = torch.FloatTensor(graph_collection.dists.A)
        if pair_idx is None:
            self.pair_idx = np.transpose(np.triu_indices(len(graph_collection), k=1))
        else:
            self.pair_idx = pair_idx

    def _sparsegraph_to_dict(self, spgraph, node_onehot, edge_onehot):
        adj_idx_T = np.transpose(spgraph.get_edgeid_to_idx_array())
        if node_onehot:
            attr_onehot = self._to_onehot(spgraph.attr_matrix, self.nattrs)
            node_attr = torch.FloatTensor(attr_onehot)
        else:
            node_attr = torch.FloatTensor(spgraph.attr_matrix)
        if self.edge_nattrs > 0:
            if edge_onehot:
                edge_attr_onehot = self._to_onehot(
                    spgraph.edge_attr_matrix, self.edge_nattrs
                )
                edge_attr = torch.FloatTensor(edge_attr_onehot)
            else:
                edge_attr = torch.LongTensor(spgraph.edge_attr_matrix.flatten())
        else:
            edge_attr = None
        graphdict = {
            "adj_idx": torch.LongTensor(adj_idx_T),
            "attr_matrix": node_attr,
            "edge_attr_matrix": edge_attr,
        }
        return graphdict

    def _to_onehot(self, mat, num_classes):
        enc = OneHotEncoder(
            categories=[np.arange(num_classes)], sparse=False, dtype=np.float32
        )
        enc.fit(mat)
        return enc.transform(mat)

    def __len__(self):
        return len(self.pair_idx)

    def __getitem__(self, idx):
        i1, i2 = self.pair_idx[idx]
        return self.graphs[i1], self.graphs[i2], self.dists[i1, i2]
