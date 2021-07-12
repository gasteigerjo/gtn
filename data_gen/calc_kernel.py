import argparse
import logging

import grakel
import numpy as np

from gtn.dataloader.io import load_from_npz

parser = argparse.ArgumentParser(description="Classification tests")
parser.add_argument(
    "npz_file", type=str, help="Filename of npz-file of GraphCollection"
)
parser.add_argument("output_file", type=str, help="Output filename for distance array")
parser.add_argument("ngraph1", type=int, help="Index of first graph")
parser.add_argument("ngraph2", type=int, help="Index of second graph")
args = parser.parse_args()
logging.basicConfig(
    format="%(asctime)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO
)

logging.info(
    f"Comparing graphs {args.ngraph1} and {args.ngraph2} of '{args.npz_file}'."
)

gcoll = load_from_npz(args.npz_file)


def spgraph_to_grakel_graph(graph):
    edge_dict = list(zip(*graph.adj_matrix.nonzero()))
    node_label_dict = dict(
        zip(range(graph.num_nodes()), graph.attr_matrix.flatten().astype(np.int))
    )
    if graph.edge_attr_matrix is not None:
        edge_label_dict = dict(
            zip(
                zip(*graph.adj_matrix.nonzero()),
                graph.edge_attr_matrix.flatten().astype(np.int),
            )
        )
        return grakel.Graph(edge_dict, node_label_dict, edge_label_dict)
    else:
        return grakel.Graph(edge_dict, node_label_dict)


# kernel = grakel.Propagation(n_jobs=10, normalize=True, M='TV', t_max=5, w=0.01)
kernel = grakel.NeighborhoodHash(normalize=True, R=3, bits=16, nh_type="simple")

grakel_graph1 = spgraph_to_grakel_graph(gcoll[args.ngraph1])
grakel_graph2 = spgraph_to_grakel_graph(gcoll[args.ngraph2])
gram_mat_local = kernel.fit_transform([grakel_graph1, grakel_graph2])
gram_mat = np.zeros((len(gcoll), len(gcoll)))

gram_mat[args.ngraph1, args.ngraph1] = gram_mat_local[0, 0]
gram_mat[args.ngraph2, args.ngraph2] = gram_mat_local[1, 1]
gram_mat[args.ngraph1, args.ngraph2] = gram_mat_local[0, 1]

np.savez(args.output_file, gram_mat)
