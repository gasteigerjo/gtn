import numbers

import networkx as nx
import numpy as np


def initial_attractiveness(n, m, initialatt, rnd_state=np.random.RandomState()):
    assert isinstance(n, numbers.Integral)
    assert isinstance(m, numbers.Integral)
    assert isinstance(initialatt, numbers.Integral)
    # Add m initial nodes (m0 in barabasi-speak)
    G = nx.empty_graph(m)
    # Target nodes for new edges
    targets = list(range(m))
    # List of existing nodes, with nodes repeated once for each adjacent edge
    repeated_nodes = targets * initialatt
    # Start adding the other n-m nodes. The first node is m.
    source = m
    while source < n:
        # Add edges to m nodes from the source.
        G.add_edges_from(zip([source] * m, targets))
        # Add one node to the list for each new edge just created.
        repeated_nodes.extend(targets)
        # And the new node "source" has m edges to add to the list.
        repeated_nodes.extend([source] * (initialatt + m))
        # Now choose m unique nodes from the existing nodes
        # Pick uniformly from repeated_nodes (preferential attachment)
        targets = rnd_state.choice(repeated_nodes, size=m, replace=False)
        source += 1
    return G
