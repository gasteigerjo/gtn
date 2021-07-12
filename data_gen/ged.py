from multiprocessing.pool import Pool

import cvxopt
import numpy as np
from cvxopt.glpk import ilp

from gtn.dataloader.preprocessing import sparsegraph_to_networkx


def calc_ged(graph1, graph2, costs, timelimit=None):
    """
    Calculate the graph edit distance between two NetworkX graphs.

    Parameters
    ----------
    graph1 : NetworkX graph
        Graph, where all nodes and edges have a 'label' attribute, which is an integer.

    graph2 : NetworkX graph
        Graph, where all nodes and edges have a 'label' attribute, which is an integer.

    costs: dict
        Dictionary containing entries for all edit costs:
        'node_subst': Function: [Number, Number] -> Number
        'node_del': float
        'node_ins': float
        'edge_subst': Function: [Number, Number] -> Number
        'edge_del': float
        'edge_ins': float

    timelimit: int
        Timelimit of the glpk solver in seconds

    Returns
    -------
    dist : float
        Graph edit distance between the graphs

    nodes_perm : list(integer tuples)
        Permutation of nodes from graph1 to graph2

    edges_perm : list(tuples of integer tuples)
        Permutation of edges from graph1 to graph2

    optimal : bool
        True if an optimal solution was found
    """

    # Create costs vector c
    costs_nodes = np.zeros([graph1.number_of_nodes(), graph2.number_of_nodes()])
    costs_nodes -= costs["node_del"] + costs["node_ins"]
    for i, data_i in graph1.nodes().data():
        for k, data_k in graph2.nodes().data():
            costs_nodes[i, k] += costs["node_subst"](data_i["label"], data_k["label"])

    costs_edges = np.zeros([graph1.number_of_edges(), graph2.number_of_edges()])
    costs_edges -= costs["edge_del"] + costs["edge_ins"]
    for id1, (i, j, data_ij) in enumerate(graph1.edges().data()):
        for id2, (k, l, data_kl) in enumerate(graph2.edges().data()):
            costs_edges[id1, id2] += costs["edge_subst"](
                data_ij["label"], data_kl["label"]
            )

    c = cvxopt.matrix(np.concatenate([costs_nodes.flatten(), costs_edges.flatten()]))
    x_size = costs_nodes.size + costs_edges.size

    # Create constraints vector h
    h_nodes = np.ones(graph1.number_of_nodes() + graph2.number_of_nodes())
    h_edges = np.zeros(2 * graph2.number_of_nodes() * graph1.number_of_edges())
    h = cvxopt.matrix(np.concatenate([h_nodes, h_edges]))

    # Create constraints matrix G
    G = np.zeros([h_nodes.size + h_edges.size, x_size])
    row = -1
    for i in graph1.nodes():
        row += 1
        for k in graph2.nodes():
            G[row, np.ravel_multi_index([i, k], costs_nodes.shape)] = 1
    for k in graph2.nodes():
        row += 1
        for i in graph1.nodes():
            G[row, np.ravel_multi_index([i, k], costs_nodes.shape)] = 1
    for k in graph2.nodes():
        for id1, (i, j) in enumerate(graph1.edges()):
            row += 1
            G[row, np.ravel_multi_index([i, k], costs_nodes.shape)] = -1
            for id2, (k_edge, l) in enumerate(graph2.edges()):
                if k == k_edge:
                    G[
                        row,
                        costs_nodes.size
                        + np.ravel_multi_index([id1, id2], costs_edges.shape),
                    ] = 1
    for l in graph2.nodes():
        for id1, (i, j) in enumerate(graph1.edges()):
            row += 1
            G[row, np.ravel_multi_index([j, l], costs_nodes.shape)] = -1
            for id2, (k, l_edge) in enumerate(graph2.edges()):
                if l == l_edge:
                    G[
                        row,
                        costs_nodes.size
                        + np.ravel_multi_index([id1, id2], costs_edges.shape),
                    ] = 1
    G_cvxopt = cvxopt.matrix(G)

    # Create indices for integer/binary variables (everything's binary)
    I = set()
    B = set(range(x_size))

    # Solve binary linear problem
    if timelimit is None:
        options = None
    else:
        options = {"tm_lim": timelimit * 1000}
    status, x = ilp(c, G_cvxopt, h, None, None, I, B, options=options)
    # print(status)

    # Transform node indices
    x_ik = np.reshape(x[: costs_nodes.size], costs_nodes.shape)
    assert np.all(
        (0 <= x_ik.sum(axis=0)) & (x_ik.sum(axis=0) <= 1)
    ), "Node assignment column sum outside allowed range (0, 1)"
    assert np.all(
        (0 <= x_ik.sum(axis=1)) & (x_ik.sum(axis=1) <= 1)
    ), "Node assignment row sum outside allowed range (0, 1)"
    nodes_transf = [None, None]
    nodes_transf[0], nodes_transf[1] = np.where(x_ik > 0)
    nodes_transf[0] = list(nodes_transf[0])
    nodes_transf[1] = list(nodes_transf[1])

    # Transform edge indices
    y_id1id2 = np.reshape(x[costs_nodes.size :], costs_edges.shape)
    assert np.all(
        (0 <= y_id1id2.sum(axis=0)) & (y_id1id2.sum(axis=0) <= 1)
    ), "Edge assignment column sum outside allowed range (0, 1)"
    assert np.all(
        (0 <= y_id1id2.sum(axis=1)) & (y_id1id2.sum(axis=1) <= 1)
    ), "Edge assignment row sum outside allowed range (0, 1)"
    edges_id_perm = zip(*np.where(y_id1id2 > 0))
    edges_transf = [[], []]
    edges1 = list(graph1.edges())
    edges2 = list(graph2.edges())
    for id1, id2 in edges_id_perm:
        edges_transf[0].append(edges1[id1])
        edges_transf[1].append(edges2[id2])

    # Calculate graph edit distance
    dist = (
        (x_ik * costs_nodes).sum()
        + (y_id1id2 * costs_edges).sum()
        + graph1.number_of_nodes() * costs["node_del"]
        + graph2.number_of_nodes() * costs["node_ins"]
        + graph1.number_of_edges() * costs["edge_del"]
        + graph2.number_of_edges() * costs["edge_ins"]
    )

    # Add entries for insertions and deletions
    for node in graph1.nodes():
        if node not in nodes_transf[0]:
            nodes_transf[0].append(node)
            nodes_transf[1].append(np.nan)
    for node in graph2.nodes():
        if node not in nodes_transf[1]:
            nodes_transf[0].append(np.nan)
            nodes_transf[1].append(node)
    for edge in graph1.edges():
        if edge not in edges_transf[0]:
            edges_transf[0].append(edge)
            edges_transf[1].append(np.nan)
    for edge in graph2.edges():
        if edge not in edges_transf[1]:
            edges_transf[0].append(np.nan)
            edges_transf[1].append(edge)

    # Transform permutation list
    nodes_perm = list(zip(*nodes_transf))
    edges_perm = list(zip(*edges_transf))

    return dist, nodes_perm, edges_perm, status == "optimal"


def calc_all_geds(graphcoll, costs, nprocesses=4, timelimit=None):
    with Pool(nprocesses) as pool:
        # Queue tasks
        results = []
        for i, sp_graph1 in enumerate(graphcoll):
            nx_graph1 = sparsegraph_to_networkx(sp_graph1)
            for sp_graph2 in graphcoll[i + 1 :]:
                nx_graph2 = sparsegraph_to_networkx(sp_graph2)
                results.append(
                    pool.apply_async(
                        calc_ged,
                        [nx_graph1, nx_graph2, costs],
                        {"timelimit": timelimit},
                    )
                )

        # Collect results
        idx = 0
        for i in range(len(graphcoll)):
            for j in range(i + 1, len(graphcoll)):
                graphcoll.dists[i, j], _, _, _ = results[idx].get()
                idx += 1
