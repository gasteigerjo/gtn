"""
Standard preprocessing of SparseGraph objects before further usage.
"""
import warnings
from numbers import Number
from typing import Union

import networkx as nx
import numpy as np
import scipy.sparse as sp

from .sparsegraph import SparseGraph

__all__ = ["sparsegraph_to_networkx", "networkx_to_sparsegraph"]


def sparsegraph_to_networkx(sp_graph: "SparseGraph") -> Union[nx.Graph, nx.DiGraph]:
    """Convert SparseGraph to NetworkX graph.

    Everything except metadata is preserved.

    Parameters
    ----------
    sp_graph
        Graph to convert.

    Returns
    -------
    nx.Graph or nx.DiGraph
        Converted graph.

    """
    # Basic graph
    if sp_graph.is_directed():
        nx_graph = nx.DiGraph(sp_graph.adj_matrix)
    else:
        nx_graph = nx.Graph(sp_graph.adj_matrix)

    # Node attributes
    if sp.issparse(sp_graph.attr_matrix):
        for inode, node_attrs in enumerate(sp_graph.attr_matrix):
            for iattr in node_attrs.nonzero()[1]:
                if sp_graph.attr_names is None:
                    nx_graph.nodes[inode][iattr] = node_attrs[iattr]
                else:
                    nx_graph.nodes[inode][sp_graph.attr_names[iattr]] = node_attrs[
                        0, iattr
                    ]
    elif isinstance(sp_graph.attr_matrix, np.ndarray):
        for inode, node_attrs in enumerate(sp_graph.attr_matrix):
            for iattr, attr in enumerate(node_attrs):
                if sp_graph.attr_names is None:
                    nx_graph.nodes[inode][iattr] = attr
                else:
                    nx_graph.nodes[inode][sp_graph.attr_names[iattr]] = attr

    # Edge attributes
    if sp.issparse(sp_graph.edge_attr_matrix):
        for edge_id, (i, j) in enumerate(sp_graph.get_edgeid_to_idx_array()):
            row = sp_graph.edge_attr_matrix[edge_id, :]
            for iattr in row.nonzero()[1]:
                if sp_graph.edge_attr_names is None:
                    nx_graph.edges[i, j][iattr] = row[0, iattr]
                else:
                    nx_graph.edges[i, j][sp_graph.edge_attr_names[iattr]] = row[iattr]
    elif isinstance(sp_graph.edge_attr_matrix, np.ndarray):
        for edge_id, (i, j) in enumerate(sp_graph.get_edgeid_to_idx_array()):
            for iattr, attr in enumerate(sp_graph.edge_attr_matrix[edge_id, :]):
                if sp_graph.edge_attr_names is None:
                    nx_graph.edges[i, j][iattr] = attr
                else:
                    nx_graph.edges[i, j][sp_graph.edge_attr_names[iattr]] = attr

    # Labels
    if sp_graph.labels is not None:
        for inode, label in enumerate(sp_graph.labels):
            if sp_graph.class_names is None:
                nx_graph.nodes[inode]["label"] = label
            else:
                nx_graph.nodes[inode]["label"] = sp_graph.class_names[label]

    # Node names
    if sp_graph.node_names is not None:
        mapping = dict(enumerate(sp_graph.node_names))
        if "self" in sp_graph.attr_names:
            nx_graph = nx.relabel_nodes(nx_graph, mapping)
        else:
            nx.relabel_nodes(nx_graph, mapping, copy=False)

    # Metadata
    if sp_graph.metadata is not None:
        warnings.warn(
            "Could not convert Metadata since NetworkX does not support arbitrary Metadata."
        )

    return nx_graph


def networkx_to_sparsegraph(
    nx_graph: Union[nx.Graph, nx.DiGraph],
    label_name: str = None,
    sparse_node_attrs: bool = True,
    sparse_edge_attrs: bool = True,
) -> "SparseGraph":
    """Convert NetworkX graph to SparseGraph.

    Node and edge attributes need to be numeric.
    Missing entries are interpreted as 0.
    Labels can be any object. If non-numeric they are interpreted as
    categorical and enumerated.

    Parameters
    ----------
    nx_graph
        Graph to convert.

    Returns
    -------
    SparseGraph
        Converted graph.

    """
    # Extract node names
    int_names = True
    for node in nx_graph.nodes:
        int_names &= isinstance(node, int)
    if int_names:
        node_names = None
    else:
        node_names = np.array(nx_graph.nodes)
        nx_graph = nx.convert_node_labels_to_integers(nx_graph)

    # Extract adjacency matrix
    adj_matrix = nx.adjacency_matrix(nx_graph)

    # Collect all node attribute names
    attrs = set()
    for _, node_data in nx_graph.nodes().data():
        attrs.update(node_data.keys())

    # Initialize labels and remove them from the attribute names
    if label_name is None:
        labels = None
    else:
        if label_name not in attrs:
            raise ValueError(
                "No attribute with label name '{}' found.".format(label_name)
            )
        attrs.remove(label_name)
        labels = [0 for _ in range(nx_graph.number_of_nodes())]

    if len(attrs) > 0:
        # Save attribute names if not integer
        all_integer = all((isinstance(attr, int) for attr in attrs))
        if all_integer:
            attr_names = None
            attr_mapping = None
        else:
            attr_names = np.array(list(attrs))
            attr_mapping = {k: i for i, k in enumerate(attr_names)}

        # Initialize attribute matrix
        if sparse_node_attrs:
            attr_matrix = sp.lil_matrix(
                (nx_graph.number_of_nodes(), len(attr_names)), dtype=np.float32
            )
        else:
            attr_matrix = np.zeros(
                (nx_graph.number_of_nodes(), len(attr_names)), dtype=np.float32
            )
    else:
        attr_matrix = None
        attr_names = None

    # Fill label and attribute matrices
    for inode, node_attrs in nx_graph.nodes.data():
        for key, val in node_attrs.items():
            if key == label_name:
                labels[inode] = val
            else:
                if not isinstance(val, Number):
                    if node_names is None:
                        raise ValueError(
                            "Node {} has attribute '{}' with value '{}', which is not a number.".format(
                                inode, key, val
                            )
                        )
                    else:
                        raise ValueError(
                            "Node '{}' has attribute '{}' with value '{}', which is not a number.".format(
                                node_names[inode], key, val
                            )
                        )
                if attr_mapping is None:
                    attr_matrix[inode, key] = val
                else:
                    attr_matrix[inode, attr_mapping[key]] = val
    if attr_matrix is not None and sparse_node_attrs:
        attr_matrix = attr_matrix.tocsr()

    # Convert labels to integers
    if labels is None:
        class_names = None
    else:
        try:
            labels = np.array(labels, dtype=np.float32)
            class_names = None
        except ValueError:
            class_names = np.unique(labels)
            class_mapping = {k: i for i, k in enumerate(class_names)}
            labels_int = np.empty(nx_graph.number_of_nodes(), dtype=np.float32)
            for inode, label in enumerate(labels):
                labels_int[inode] = class_mapping[label]
            labels = labels_int

    # Collect all edge attribute names
    edge_attrs = set()
    for _, _, edge_data in nx_graph.edges().data():
        edge_attrs.update(edge_data.keys())
    if "weight" in edge_attrs:
        edge_attrs.remove("weight")

    if len(edge_attrs) > 0:
        # Save edge attribute names if not integer
        all_integer = all((isinstance(attr, int) for attr in edge_attrs))
        if all_integer:
            edge_attr_names = None
            edge_attr_mapping = None
        else:
            edge_attr_names = np.array(list(edge_attrs))
            edge_attr_mapping = {k: i for i, k in enumerate(edge_attr_names)}

        # Initialize edge attribute matrix
        if sparse_edge_attrs:
            edge_attr_matrix = sp.lil_matrix(
                (adj_matrix.nnz, len(edge_attr_names)), dtype=np.float32
            )
        else:
            edge_attr_matrix = np.zeros(
                (adj_matrix.nnz, len(edge_attr_names)), dtype=np.float32
            )
    else:
        edge_attr_matrix = None
        edge_attr_names = None

    # Fill edge attribute matrix
    edgeid_mat = sp.csr_matrix(
        (np.arange(adj_matrix.nnz), adj_matrix.indices, adj_matrix.indptr),
        shape=adj_matrix.shape,
    )
    for i, j, edge_attrs in nx_graph.edges.data():
        for key, val in edge_attrs.items():
            if key != "weight":
                if not isinstance(val, Number):
                    if node_names is None:
                        raise ValueError(
                            "Edge {}->{} has attribute '{}' with value '{}', which is not a number.".format(
                                i, j, key, val
                            )
                        )
                    else:
                        raise ValueError(
                            "Edge '{}'->'{}' has attribute '{}' with value '{}', which is not a number.".format(
                                node_names[i], node_names[j], key, val
                            )
                        )
                new_key = key if attr_mapping is None else edge_attr_mapping[key]
                edge_attr_matrix[edgeid_mat[i, j], new_key] = val
                if not nx_graph.is_directed():
                    edge_attr_matrix[edgeid_mat[j, i], new_key] = val
    if edge_attr_matrix is not None and sparse_edge_attrs:
        edge_attr_matrix = edge_attr_matrix.tocsr()

    return SparseGraph(
        adj_matrix=adj_matrix,
        attr_matrix=attr_matrix,
        edge_attr_matrix=edge_attr_matrix,
        labels=labels,
        node_names=node_names,
        attr_names=attr_names,
        edge_attr_names=edge_attr_names,
        class_names=class_names,
        metadata=None,
    )
