"""
Functions for converting, saving and loading SparseGraph to and from different formats.
"""
import warnings
from typing import Union

import numpy as np

from .graphcollection import GraphCollection
from .sparsegraph import SparseGraph

__all__ = ["load_from_npz", "save_to_npz"]


def load_from_npz(file_name: str) -> Union[SparseGraph, GraphCollection]:
    """Load a SparseGraph or GraphCollection from a Numpy binary file.

    Parameters
    ----------
    file_name
        Name of the file to load.

    Returns
    -------
    SparseGraph or GraphCollection
        Graph(s) in sparse matrix format.

    """
    with np.load(file_name, allow_pickle=True) as loader:
        loader = dict(loader)
        if "type" in loader:
            dataset_type = loader["type"]
            del loader["type"]
            if dataset_type == SparseGraph.__name__:
                dataset = SparseGraph.from_flat_dict(loader)
            elif dataset_type == GraphCollection.__name__:
                dataset = GraphCollection.from_flat_dict(loader)
            else:
                raise ValueError(
                    "Type '{}' of loaded npz-file not recognized.".format(dataset_type)
                )
        else:
            warnings.warn(
                "Type of saved dataset not specified, using heuristic instead. "
                "Please update (re-save) your stored graphs.",
                DeprecationWarning,
                stacklevel=2,
            )
            if "dists" in loader.keys():
                dataset = GraphCollection.from_flat_dict(loader)
            else:
                dataset = SparseGraph.from_flat_dict(loader)
    return dataset


def save_to_npz(file_name: str, dataset: Union[SparseGraph, GraphCollection]):
    """Save a SparseGraph or GraphCollection to a Numpy binary file.

    Better (faster) than pickle for single graphs, where disk space is no issue.
    npz doesn't support compression and fails for files larger than 4GiB.

    Parameters
    ----------
    file_name
        Name of the output file.
    dataset
        Graph(s) in sparse matrix format.

    """
    data_dict = dataset.to_flat_dict()
    data_dict["type"] = dataset.__class__.__name__
    np.savez(file_name, **data_dict)
