import copy
from collections import Iterable, defaultdict
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import scipy.sparse as sp

from .sparsegraph import SparseGraph

__all__ = ["GraphCollection", "DistanceMatrix"]

graph_collection_properties = ["dists", "metadata"]


class GraphCollection:
    """Collection of multiple SparseGraphs.

    Parameters
    ----------
    graphs
        List of SparseGraphs to be initialized for the collection
    dists
        Distances between graphs in CSR or numpy format. Shape [num_graphs, num_graphs]
    metadata
        Additional metadata such as text.

    """

    def __init__(
        self,
        graphs: List[SparseGraph] = None,
        dists: Union[sp.spmatrix, np.ndarray] = None,
        metadata: Any = None,
    ):

        if isinstance(graphs, Iterable):
            self._graphs = graphs
        elif graphs is None:
            self._graphs = []
        else:
            self._graphs = [graphs]

        if dists is None:
            self.dists = DistanceMatrix(len(self))
        else:
            if (dists.shape[0] != len(graphs)) or (dists.shape[1] != len(graphs)):
                raise ValueError(
                    "Dimensions of the distance matrix and number of graphs don't agree."
                )
            self.dists = DistanceMatrix(dists)

        self._metadata = metadata

    def append(self, graph: SparseGraph):
        if not isinstance(graph, SparseGraph):
            raise ValueError(
                "Expected SparseGraph, got {} instead.".format(type(graph))
            )
        self._graphs.append(graph)
        self.dists.extend(1)

    def extend(self, other: "GraphCollection"):
        self._graphs += other._graphs
        self.dists.extend(other.dists)

    def __getitem__(self, key: int) -> SparseGraph:
        if isinstance(key, slice):
            return GraphCollection(
                self.graphs[key], self.dists[key, key], self.metadata
            )
        else:
            return self.graphs[key]

    def __setitem__(self, key: int, value: SparseGraph):
        if not isinstance(value, SparseGraph):
            raise ValueError(
                "Expected SparseGraph, got {} instead.".format(type(value))
            )
        self.graphs[key] = value

    def __delitem__(self, key: int):
        del self._graphs[key]
        del self.dists[key]

    def __add__(self, other: "GraphCollection") -> "GraphCollection":
        new_dist = copy.deepcopy(self.dists)
        new_dist.extend(other.dists)
        if self.dists.issparse and not other.dists.issparse:
            new_dist = new_dist.todense()
        new_graphs = self._graphs + other._graphs
        return GraphCollection(new_graphs, new_dist)

    def __iadd__(self, other: "GraphCollection") -> "GraphCollection":
        self.extend(other)
        return self

    def __len__(self) -> int:
        return len(self.graphs)

    def __repr__(self) -> str:
        return f"GraphCollection({repr(self.graphs)})"

    def to_flat_dict(self) -> Dict[str, np.ndarray]:
        """Return flat dictionary containing the properties of all SparseGraphs."""
        data_dict = {}

        # Export graphs
        for i, graph in enumerate(self.graphs):
            graph_dict = graph.to_flat_dict()
            for key, val in graph_dict.items():
                data_dict["{}.{}".format(i, key)] = val

        # Export dists
        data_dict.update(self.dists.to_flat_dict())
        props_left = [prop for prop in graph_collection_properties if prop != "dists"]

        # Export properties
        for key in props_left:
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
    def from_flat_dict(data_dict: Dict[str, np.ndarray]) -> "GraphCollection":
        """Initialize GraphCollection from a flat dictionary."""
        init_dict = {}
        del_entries = []

        # Construct sparse matrix properties
        for key in data_dict.keys():
            if key.split(".")[0] in graph_collection_properties and key.endswith(
                ".data"
            ):
                matrix_name = key[:-5]
                mat_data = key
                mat_indices = "{}.indices".format(matrix_name)
                mat_indptr = "{}.indptr".format(matrix_name)
                mat_shape = "{}.shape".format(matrix_name)
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

        # Load other properties
        del_entries = []
        for key, val in data_dict.items():
            if key in graph_collection_properties:
                if (val is not None) and (None not in val):
                    init_dict[key] = val
                del_entries.append(key)

        # Delete property entries
        for del_entry in del_entries:
            del data_dict[del_entry]

        # Split entries from different graphs
        graph_dicts = defaultdict(dict)
        for key in data_dict.keys():
            igraph = int(key.split(".")[0])
            inner_key = key[key.find(".") + 1 :]
            graph_dicts[igraph][inner_key] = data_dict[key]
        sorted_graph_dicts = sorted(graph_dicts.items(), key=lambda kv: kv[0])

        # Initialize graphs
        graphs = []
        for _, graph_dict in sorted_graph_dicts:
            graphs.append(SparseGraph.from_flat_dict(graph_dict))

        # Initialize graph collection
        return GraphCollection(graphs, **init_dict)

    @property
    def graphs(self) -> List[SparseGraph]:
        return self._graphs

    @property
    def metadata(self) -> Any:
        return self._metadata


class DistanceMatrix:
    """Class for saving distances and preserving properties like symmetry, non-negativity and zero diagonal.

    Parameters
    ----------
    dists
        Distances as DistanceMatrix, CSR or numpy format
        or one side of the matrix

    """

    def __init__(self, dists: Union["DistanceMatrix", sp.spmatrix, np.ndarray, int]):
        if isinstance(dists, DistanceMatrix):
            self._matrix = dists._matrix
        elif sp.isspmatrix(dists):
            self._matrix = dists.tocsr().astype(np.float32)
        elif isinstance(dists, np.ndarray):
            self._matrix = dists.astype(np.float32)
        elif isinstance(dists, int):
            self._matrix = sp.csr_matrix((dists, dists), dtype=np.float32)
        else:
            raise ValueError(
                "Distances must be a DistanceMatrix, sp.spmatrix,"
                " np.ndarray or an integer (got {0} instead).".format(type(dists))
            )

        if self.shape[0] != self.shape[1]:
            raise ValueError("Distance matrix has to be square.")

        if (self._matrix != self._matrix.T).sum() != 0:
            raise ValueError("Distances are not symmetric.")

        if self.shape[0] != 0 and not np.allclose(self._matrix.diagonal(), 0):
            raise ValueError("Self-distances are not 0.")

        if (self._matrix < 0).sum() != 0:
            raise ValueError("Distances contain negative entries.")

    def __getitem__(self, key: Tuple[int, int]) -> float:
        if isinstance(key[0], slice):
            return DistanceMatrix(self._matrix[key])
        else:
            return self._matrix[key]

    def __setitem__(self, key: Tuple[int, int], value: float):
        if key[0] == key[1] and not np.isclose(value, 0):
            raise ValueError("Self-distance has to be 0.")
        if value < 0:
            raise ValueError("Distances have to be non-negative.")
        self._matrix[key] = value
        self._matrix[key[::-1]] = value

    def __delitem__(self, key: int):
        """Deletes both the associated row and column."""
        ikeeps = [i for i in range(self.shape[0]) if i != key]
        self._matrix = self._matrix[ikeeps][:, ikeeps]

    def __add__(self, other: "DistanceMatrix") -> "DistanceMatrix":
        new_mat = self._matrix + other._matrix
        return DistanceMatrix(new_mat)

    def __iadd__(self, other: "DistanceMatrix") -> "DistanceMatrix":
        self._matrix += other._matrix
        return self

    def __sub__(self, other: "DistanceMatrix") -> "DistanceMatrix":
        if self.issparse and other.issparse:
            if (self < other).nnz > 0:
                raise ValueError("Subtraction would lead to negative distances.")
        else:
            if np.any(self < other):
                raise ValueError("Subtraction would lead to negative distances.")
        new_mat = self._matrix - other._matrix
        return DistanceMatrix(new_mat)

    def __isub__(self, other: "DistanceMatrix") -> "DistanceMatrix":
        if self.issparse and other.issparse:
            if (self < other).nnz > 0:
                raise ValueError("Subtraction would lead to negative distances.")
        else:
            if np.any(self < other):
                raise ValueError("Subtraction would lead to negative distances.")
        self._matrix -= other._matrix
        return self

    def __eq__(self, other: "DistanceMatrix") -> bool:
        return self._matrix == other._matrix

    def __ne__(self, other: "DistanceMatrix") -> bool:
        return self._matrix != other._matrix

    def __ge__(self, other: "DistanceMatrix") -> bool:
        return self._matrix >= other._matrix

    def __gt__(self, other: "DistanceMatrix") -> bool:
        return self._matrix > other._matrix

    def __le__(self, other: "DistanceMatrix") -> bool:
        return self._matrix <= other._matrix

    def __lt__(self, other: "DistanceMatrix") -> bool:
        return self._matrix < other._matrix

    def __repr__(self) -> str:
        return f"DistanceMatrix(\n{str(self._matrix)})"

    @property
    def shape(self) -> Tuple[int, int]:
        return self._matrix.shape

    @property
    def size(self) -> int:
        return self._matrix.size

    @property
    def issparse(self) -> bool:
        return sp.isspmatrix(self._matrix)

    def todense(self) -> "DistanceMatrix":
        return DistanceMatrix(self._matrix.A)

    def tosparse(self) -> "DistanceMatrix":
        return DistanceMatrix(sp.csr_matrix(self._matrix))

    @property
    def A(self) -> np.ndarray:
        if self.issparse:
            return self._matrix.A
        else:
            return self._matrix.copy()

    def extend(self, other_n: Union["DistanceMatrix", sp.spmatrix, np.ndarray, int]):
        """Extend matrix with another matrix in the block-matrix-style [[self, 0], [0, other]].

        Parameters
        ----------
        other_n
            Distances to extend with as DistanceMatrix, CSR or numpy format
            or the number of additional rows and columns

        """
        if isinstance(other_n, int):
            self._matrix = self._extend_mat(self._matrix, other_n)
        elif isinstance(other_n, DistanceMatrix):
            old_len = self.shape[0]
            self._matrix = self._extend_mat(self._matrix, other_n.shape[0])
            other_n_mat = self._extend_mat(other_n._matrix, old_len, mat_first=False)
            self._matrix += other_n_mat
        elif sp.isspmatrix(other_n) or isinstance(other_n, np.ndarray):
            self.extend(DistanceMatrix(other_n))
        else:
            raise ValueError(
                "Distances must be a DistanceMatrix, sp.spmatrix,"
                " np.ndarray or an integer (got {0} instead).".format(type(other_n))
            )

    def _extend_mat(
        self, mat: Union[sp.spmatrix, np.ndarray], n: int, mat_first: bool = True
    ) -> Union[sp.csr_matrix, np.ndarray]:
        if sp.isspmatrix(mat):
            mats = [mat, sp.csr_matrix((mat.shape[0], n), dtype=np.float32)]
            mat = sp.hstack(mats if mat_first else mats[::-1])
            mats = [mat, sp.csr_matrix((n, mat.shape[1]), dtype=np.float32)]
            mat = sp.vstack(mats if mat_first else mats[::-1])
            return mat.tocsr()
        else:
            mats = [mat, np.zeros([mat.shape[0], n])]
            mat = np.hstack(mats if mat_first else mats[::-1])
            mats = [mat, np.zeros([n, mat.shape[1]])]
            return np.vstack(mats if mat_first else mats[::-1])

    def to_flat_dict(self) -> Dict[str, np.ndarray]:
        """Return flat dictionary containing the properties of all SparseGraphs."""
        data_dict = {}
        if sp.isspmatrix(self._matrix):
            data_dict["dists.data"] = self._matrix.data
            data_dict["dists.indices"] = self._matrix.indices
            data_dict["dists.indptr"] = self._matrix.indptr
            data_dict["dists.shape"] = self._matrix.shape
        else:
            data_dict["dists"] = self._matrix
        return data_dict
