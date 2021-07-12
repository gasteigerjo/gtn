import argparse
import logging

import numpy as np

from gtn.dataloader.io import load_from_npz
from gtn.dataloader.preprocessing import sparsegraph_to_networkx

from .ged import calc_ged

parser = argparse.ArgumentParser(description="Classification tests")
parser.add_argument(
    "npz_file", type=str, help="Filename of npz-file of GraphCollection"
)
parser.add_argument("output_file", type=str, help="Output filename for distance array")
parser.add_argument("ngraph1", type=int, help="Index of first graph")
parser.add_argument("ngraph2", type=int, help="Index of second graph")
parser.add_argument(
    "--timelimit", type=int, help="Timelimit of GED computation", default=600
)
args = parser.parse_args()
logging.basicConfig(
    format="%(asctime)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO
)

logging.info(
    f"Comparing graphs {args.ngraph1} and {args.ngraph2} of '{args.npz_file}'."
)


def cost_node_subst(node_i, node_j):
    return 0 if node_i == node_j else 1


def cost_edge_subst(edge_i, edge_j):
    return 0 if edge_i == edge_j else 1


costs = {
    "node_subst": cost_node_subst,
    "node_del": 2,
    "node_ins": 2,
    "edge_subst": cost_edge_subst,
    "edge_del": 2,
    "edge_ins": 2,
}

gcoll = load_from_npz(args.npz_file)

nx_graph1 = sparsegraph_to_networkx(gcoll[args.ngraph1])
nx_graph2 = sparsegraph_to_networkx(gcoll[args.ngraph2])
gcoll.dists[args.ngraph1, args.ngraph2], _, _, optimal = calc_ged(
    nx_graph1, nx_graph2, costs, timelimit=args.timelimit
)

if optimal:
    logging.info(
        f"Distance {args.ngraph1}-{args.ngraph2} ('{args.npz_file}'): "
        f"{gcoll.dists[args.ngraph1, args.ngraph2]} (optimal)"
    )
else:
    logging.info(
        f"Distance {args.ngraph1}-{args.ngraph2} ('{args.npz_file}'): "
        f"{gcoll.dists[args.ngraph1, args.ngraph2]} (suboptimal)"
    )

np.savez(args.output_file, gcoll.dists.A)
