import warnings
from typing import Any, Dict, Tuple, Union

import numpy as np
import scipy.sparse as sp

__all__ = ["SparseGraph"]

sparse_graph_properties = [
    "adj_matrix",
    "attr_matrix",
    "edge_attr_matrix",
    "labels",
    "node_names",
    "attr_names",
    "edge_attr_names",
    "class_names",
    "metadata",
]


class SparseGraph:
    """Attributed labeled graph stored in sparse matrix form.

    All properties are immutable so users don't mess up the
    data format's assumptions (e.g. of edge_attr_matrix).
    Be careful when circumventing this and changing the internal matrices
    regardless (e.g. by exchanging the data array of a sparse matrix).

    Parameters
    ----------
    adj_matrix
        Adjacency matrix in CSR format. Shape [num_nodes, num_nodes]
    attr_matrix
        Attribute matrix in CSR or numpy format. Shape [num_nodes, num_attr]
    edge_attr_matrix
        Edge attribute matrix in CSR or numpy format. Shape [num_edges, num_edge_attr]
    labels
        Array, where each entry represents respective node's label(s). Shape [num_nodes]
        Alternatively, CSR matrix with labels in one-hot format. Shape [num_nodes, num_classes]
    node_names
        Names of nodes (as strings). Shape [num_nodes]
    attr_names
        Names of the attributes (as strings). Shape [num_attr]
    edge_attr_names
        Names of the edge attributes (as strings). Shape [num_edge_attr]
    class_names
        Names of the class labels (as strings). Shape [num_classes]
    metadata
        Additional metadata such as text.

    """

    def __init__(
        self,
        adj_matrix: sp.spmatrix,
        attr_matrix: Union[np.ndarray, sp.spmatrix] = None,
        edge_attr_matrix: Union[np.ndarray, sp.spmatrix] = None,
        labels: Union[np.ndarray, sp.spmatrix] = None,
        node_names: np.ndarray = None,
        attr_names: np.ndarray = None,
        edge_attr_names: np.ndarray = None,
        class_names: np.ndarray = None,
        metadata: Any = None,
    ):
        # Make sure that the dimensions of matrices / arrays all agree
        if sp.isspmatrix(adj_matrix):
            adj_matrix = adj_matrix.tocsr().astype(np.float32)
        else:
            raise ValueError(
                "Adjacency matrix must be in sparse format (got {0} instead).".format(
                    type(adj_matrix)
                )
            )

        if adj_matrix.shape[0] != adj_matrix.shape[1]:
            raise ValueError("Dimensions of the adjacency matrix don't agree.")

        if attr_matrix is not None:
            if sp.isspmatrix(attr_matrix):
                attr_matrix = attr_matrix.tocsr().astype(np.float32)
            elif isinstance(attr_matrix, np.ndarray):
                attr_matrix = attr_matrix.astype(np.float32)
            else:
                raise ValueError(
                    "Attribute matrix must be a sp.spmatrix or a np.ndarray (got {0} instead).".format(
                        type(attr_matrix)
                    )
                )

            if attr_matrix.shape[0] != adj_matrix.shape[0]:
                raise ValueError(
                    "Dimensions of the adjacency and attribute matrices don't agree."
                )

        if edge_attr_matrix is not None:
            if sp.isspmatrix(edge_attr_matrix):
                edge_attr_matrix = edge_attr_matrix.tocsr().astype(np.float32)
            elif isinstance(edge_attr_matrix, np.ndarray):
                edge_attr_matrix = edge_attr_matrix.astype(np.float32)
            else:
                raise ValueError(
                    "Edge attribute matrix must be a sp.spmatrix or a np.ndarray (got {0} instead).".format(
                        type(edge_attr_matrix)
                    )
                )

            if edge_attr_matrix.shape[0] != adj_matrix.count_nonzero():
                raise ValueError(
                    "Number of edges and dimension of the edge attribute matrix don't agree."
                )

        if labels is not None:
            if labels.shape[0] != adj_matrix.shape[0]:
                raise ValueError(
                    "Dimensions of the adjacency matrix and the label vector don't agree."
                )

        if node_names is not None:
            if len(node_names) != adj_matrix.shape[0]:
                raise ValueError(
                    "Dimensions of the adjacency matrix and the node names don't agree."
                )

        if attr_names is not None:
            if len(attr_names) != attr_matrix.shape[1]:
                raise ValueError(
                    "Dimensions of the attribute matrix and the attribute names don't agree."
                )

        if edge_attr_names is not None:
            if len(edge_attr_names) != edge_attr_matrix.shape[1]:
                raise ValueError(
                    "Dimensions of the edge attribute matrix and the edge attribute names don't agree."
                )

        self._adj_matrix = adj_matrix
        self._attr_matrix = attr_matrix
        self._edge_attr_matrix = edge_attr_matrix
        self._labels = labels
        self._node_names = node_names
        self._attr_names = attr_names
        self._edge_attr_names = edge_attr_names
        self._class_names = class_names
        self._metadata = metadata

        self._flag_writeable(False)

    def num_nodes(self) -> int:
        """Get the number of nodes in the graph."""
        return self.adj_matrix.shape[0]

    def num_edges(self, warn: bool = True) -> int:
        """Get the number of edges in the graph.

        For undirected graphs, (i, j) and (j, i) are counted as _two_ edges.

        """
        if warn and not self.is_directed():
            warnings.warn(
                "num_edges always returns the number of directed edges now.",
                FutureWarning,
            )
        return self.adj_matrix.nnz

    def get_neighbors(self, idx: int) -> np.ndarray:
        """Get the indices of neighbors of a given node.

        Parameters
        ----------
        idx
            Index of the node whose neighbors are of interest.

        """
        return self.adj_matrix[idx].indices

    def get_edgeid_to_idx_array(self) -> np.ndarray:
        """Return a Numpy Array that maps edgeids to the indices in the adjacency matrix.

        Returns
        -------
        np.ndarray
            The i'th entry contains the x- and y-coordinates of edge i in the adjacency matrix.
            Shape [num_edges, 2]

        """
        return np.transpose(self.adj_matrix.nonzero())

    def get_idx_to_edgeid_matrix(self) -> sp.csr_matrix:
        """Return a sparse matrix that maps indices in the adjacency matrix to edgeids.

        Caution: This contains one explicit 0 (zero stored as a nonzero),
        which is the index of the first edge.

        Returns
        -------
        sp.csr_matrix
            The entry [x, y] contains the edgeid of the corresponding edge (or 0 for non-edges).
            Shape [num_nodes, num_nodes]

        """
        return sp.csr_matrix(
            (
                np.arange(self.adj_matrix.nnz),
                self.adj_matrix.indices,
                self.adj_matrix.indptr,
            ),
            shape=self.adj_matrix.shape,
        )

    def is_directed(self) -> bool:
        """Check if the graph is directed (adjacency matrix is not symmetric)."""
        return (self.adj_matrix != self.adj_matrix.T).sum() != 0

    def to_undirected(self) -> "SparseGraph":
        """Convert to an undirected graph (make adjacency matrix symmetric)."""

        idx = self.get_edgeid_to_idx_array().T
        ridx = np.ravel_multi_index(idx, self.adj_matrix.shape)
        ridx_rev = np.ravel_multi_index(idx[::-1], self.adj_matrix.shape)

        # Get duplicate edges (self-loops and opposing edges)
        dup_ridx = ridx[np.isin(ridx, ridx_rev)]
        dup_idx = np.unravel_index(dup_ridx, self.adj_matrix.shape)

        # Check if the adjacency matrix weights are symmetric (if nonzero)
        if len(dup_ridx) > 0 and not np.allclose(
            self.adj_matrix[dup_idx], self.adj_matrix[dup_idx[::-1]]
        ):
            raise ValueError("Adjacency matrix weights of opposing edges differ.")

        # Create symmetric matrix
        new_adj_matrix = self.adj_matrix + self.adj_matrix.T
        if len(dup_ridx) > 0:
            new_adj_matrix[dup_idx] = (
                new_adj_matrix[dup_idx] - self.adj_matrix[dup_idx]
            ).A1
        flag_writeable(new_adj_matrix, False)

        if self.edge_attr_matrix is not None:

            # Check if edge attributes are symmetric
            edgeid_mat = self.get_idx_to_edgeid_matrix()
            if len(dup_ridx) > 0:
                dup_edgeids = edgeid_mat[dup_idx].A1
                dup_rev_edgeids = edgeid_mat[dup_idx[::-1]].A1
                if not np.allclose(
                    self.edge_attr_matrix[dup_edgeids],
                    self.edge_attr_matrix[dup_rev_edgeids],
                ):
                    raise ValueError("Edge attributes of opposing edges differ.")

            # Adjust edge attributes to new adjacency matrix
            edgeid_mat.data += 1  # Add 1 so we don't lose the explicit 0 and change the sparsity structure
            new_edgeid_mat = edgeid_mat + edgeid_mat.T
            if len(dup_ridx) > 0:
                new_edgeid_mat[dup_idx] = (
                    new_edgeid_mat[dup_idx] - edgeid_mat[dup_idx]
                ).A1
            new_idx = new_adj_matrix.nonzero()
            edgeids_perm = new_edgeid_mat[new_idx].A1 - 1
            self._edge_attr_matrix = self.edge_attr_matrix[edgeids_perm]
            flag_writeable(self._edge_attr_matrix, False)

        self._adj_matrix = new_adj_matrix
        return self

    def is_weighted(self) -> bool:
        """Check if the graph is weighted (edge weights other than 1)."""
        return np.any(np.unique(self.adj_matrix[self.adj_matrix.nonzero()].A1) != 1)

    def to_unweighted(self) -> "SparseGraph":
        """Convert to an unweighted graph (set all edge weights to 1)."""
        self._adj_matrix.data = np.ones_like(self._adj_matrix.data)
        flag_writeable(self._adj_matrix, False)
        return self

    def is_connected(self) -> bool:
        """Check if the graph is connected."""
        return (
            sp.csgraph.connected_components(self.adj_matrix, return_labels=False) == 1
        )

    def has_self_loops(self) -> bool:
        """Check if the graph has self-loops."""
        return not np.allclose(self.adj_matrix.diagonal(), 0)

    def __repr__(self) -> str:
        props = []
        for prop_name in sparse_graph_properties:
            prop = getattr(self, prop_name)
            if prop is not None:
                if prop_name == "metadata":
                    props.append(prop_name)
                else:
                    shape_string = "x".join([str(x) for x in prop.shape])
                    props.append("{} ({})".format(prop_name, shape_string))
        dir_string = "Directed" if self.is_directed() else "Undirected"
        weight_string = "weighted" if self.is_weighted() else "unweighted"
        conn_string = "connected" if self.is_connected() else "disconnected"
        loop_string = "has self-loops" if self.has_self_loops() else "no self-loops"
        return "<{}, {} and {} SparseGraph with {} edges ({}). Data: {}>".format(
            dir_string,
            weight_string,
            conn_string,
            self.num_edges(warn=False),
            loop_string,
            ", ".join(props),
        )

    def unpack(
        self,
    ) -> Tuple[
        sp.csr_matrix,
        Union[np.ndarray, sp.csr_matrix],
        Union[np.ndarray, sp.csr_matrix],
        Union[np.ndarray, sp.csr_matrix],
    ]:
        """Return the (A, X, E, z) quadruplet."""
        flag_writeable(self._adj_matrix, True)
        flag_writeable(self._attr_matrix, True)
        flag_writeable(self._edge_attr_matrix, True)
        flag_writeable(self._labels, True)
        return self._adj_matrix, self._attr_matrix, self._edge_attr_matrix, self._labels

    def _adopt_graph(self, graph: "SparseGraph"):
        """Copy all properties from the given graph to this graph."""
        for prop in sparse_graph_properties:
            setattr(self, "_{}".format(prop), getattr(graph, prop))
        self._flag_writeable(False)

    def _flag_writeable(self, writeable: bool):
        """Flag all Numpy arrays and sparse matrices as non-writeable."""
        flag_writeable(self._adj_matrix, writeable)
        flag_writeable(self._attr_matrix, writeable)
        flag_writeable(self._edge_attr_matrix, writeable)
        flag_writeable(self._labels, writeable)
        flag_writeable(self._node_names, writeable)
        flag_writeable(self._attr_names, writeable)
        flag_writeable(self._edge_attr_names, writeable)
        flag_writeable(self._class_names, writeable)

    def to_flat_dict(self) -> Dict[str, Any]:
        """Return flat dictionary containing all SparseGraph properties."""
        data_dict = {}
        for key in sparse_graph_properties:
            val = getattr(self, key)
            if sp.isspmatrix(val):
                data_dict["{}.data".format(key)] = val.data
                data_dict["{}.indices".format(key)] = val.indices
                data_dict["{}.indptr".format(key)] = val.indptr
                data_dict["{}.shape".format(key)] = val.shape
            else:
                data_dict[key] = val
        return data_dict

    @staticmethod
    def from_flat_dict(data_dict: Dict[str, Any]) -> "SparseGraph":
        """Initialize SparseGraph from a flat dictionary."""
        init_dict = {}
        del_entries = []

        # Construct sparse matrices
        for key in data_dict.keys():
            if key.endswith("_data") or key.endswith(".data"):
                if key.endswith("_data"):
                    sep = "_"
                    warnings.warn(
                        "The separator used for sparse matrices during export (for .npz files) "
                        "is now '.' instead of '_'. Please update (re-save) your stored graphs.",
                        DeprecationWarning,
                        stacklevel=3,
                    )
                else:
                    sep = "."
                matrix_name = key[:-5]
                mat_data = key
                mat_indices = "{}{}indices".format(matrix_name, sep)
                mat_indptr = "{}{}indptr".format(matrix_name, sep)
                mat_shape = "{}{}shape".format(matrix_name, sep)
                if matrix_name == "adj" or matrix_name == "attr":
                    warnings.warn(
                        "Matrices are exported (for .npz files) with full names now. "
                        "Please update (re-save) your stored graphs.",
                        DeprecationWarning,
                        stacklevel=3,
                    )
                    matrix_name += "_matrix"
                init_dict[matrix_name] = sp.csr_matrix(
                    (
                        data_dict[mat_data],
                        data_dict[mat_indices],
                        data_dict[mat_indptr],
                    ),
                    shape=data_dict[mat_shape],
                )
                del_entries.extend([mat_data, mat_indices, mat_indptr, mat_shape])

        # Delete sparse matrix entries
        for del_entry in del_entries:
            del data_dict[del_entry]

        # Load everything else
        for key, val in data_dict.items():
            if (val is not None) and (None not in val):
                init_dict[key] = val

        # Check if the dictionary contains only entries in sparse_graph_properties
        unknown_keys = [
            key for key in init_dict.keys() if key not in sparse_graph_properties
        ]
        if len(unknown_keys) > 0:
            raise ValueError(
                "Input dictionary contains keys that are not SparseGraph properties ({}).".format(
                    unknown_keys
                )
            )

        return SparseGraph(**init_dict)

    @property
    def adj_matrix(self) -> sp.csr_matrix:
        return self._adj_matrix

    @property
    def attr_matrix(self) -> Union[np.ndarray, sp.csr_matrix]:
        return self._attr_matrix

    @property
    def edge_attr_matrix(self) -> Union[np.ndarray, sp.csr_matrix]:
        return self._edge_attr_matrix

    @property
    def labels(self) -> Union[np.ndarray, sp.csr_matrix]:
        return self._labels

    @property
    def node_names(self) -> np.ndarray:
        return self._node_names

    @property
    def attr_names(self) -> np.ndarray:
        return self._attr_names

    @property
    def edge_attr_names(self) -> np.ndarray:
        return self._edge_attr_names

    @property
    def class_names(self) -> np.ndarray:
        return self._class_names

    @property
    def metadata(self) -> Any:
        return self._metadata


def flag_writeable(matrix: Union[np.ndarray, sp.csr_matrix], writeable: bool):
    if matrix is not None:
        if sp.isspmatrix(matrix):
            matrix.data.flags.writeable = writeable
            matrix.indices.flags.writeable = writeable
            matrix.indptr.flags.writeable = writeable
        elif isinstance(matrix, np.ndarray):
            matrix.flags.writeable = writeable
        else:
            raise ValueError(
                "Matrix must be an sp.spmatrix or an np.ndarray"
                " (got {0} instead).".format(type(matrix))
            )
