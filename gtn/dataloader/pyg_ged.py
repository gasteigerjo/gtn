import numpy as np
import scipy.sparse as sp
from torch_geometric.datasets import GEDDataset

from .graphcollection import GraphCollection
from .sparsegraph import SparseGraph


def pyggraph_to_sparsegraph(graph):
    # LINUX has no node features -> use degrees
    if graph.x is not None:
        attr_matrix = graph.x.numpy()
        raise NotImplementedError("Need to deactivate the one-hot encoder for this.")
    else:
        _, degrees = graph.edge_index[0].unique(return_counts=True)
        attr_matrix = degrees[:, None].numpy()
        # attr_matrix = np.ones([graph.edge_index[0].max() + 1, 1], dtype=np.int32)

    # Assume graphs are undirected (i.e. edge_index symmetric)
    adj_matrix = sp.csr_matrix(
        (np.ones(graph.edge_index.shape[1]), graph.edge_index.numpy())
    )

    return SparseGraph(adj_matrix=adj_matrix, attr_matrix=attr_matrix)


def get_pyg_ged_gcolls(root, name, use_norm_ged=True, use_similarity=True):
    valtrain = GEDDataset(root=root, name=name, train=True)
    test = GEDDataset(root=root, name=name, train=False)

    pyg_graphs = [pyggraph_to_sparsegraph(graph) for graph in valtrain]
    pyg_graphs.extend([pyggraph_to_sparsegraph(graph) for graph in test])

    if use_norm_ged:
        ged_matrix = valtrain.norm_ged.numpy()
    else:
        ged_matrix = valtrain.ged.numpy()

    if use_similarity:
        ged_matrix = np.exp(-ged_matrix)
    np.fill_diagonal(ged_matrix, 0)

    gcolls = {
        split: GraphCollection(pyg_graphs, dists=ged_matrix)
        for split in ["train", "val", "test"]
    }
    # No test-test distances
    pair_idxs = {
        "train": np.transpose(np.triu_indices(600, k=1)),
        "val": np.array([(i, j) for i in range(800) for j in range(600, 800) if j > i]),
        "test": np.array([(i, j) for i in range(800) for j in range(800, 1000)]),
    }

    return gcolls, pair_idxs
