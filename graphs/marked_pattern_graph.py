from graphviz import Digraph
import re

"""
    viz
    ---

    Visualization module.

    Classes available:

    - MarkedPatternGraph
"""

def get_nodes_from_edges(edges):
    set_collection = frozenset({})
    _edges = list(edges)

    for edge in _edges:
        set_collection = set_collection.union(frozenset(edge))

    return set_collection

def get_nodes_adj_to_node(edges, node):
    collection = set({})
    _edges = list(edges)

    for edge in _edges:
        if set(edge).intersection(set({node})) != set({}):
            collection = collection.union(edge)

    return collection - set({node})

def get_common_adj_nodes(edges, node_1, node_2):
    nodes_adj_to_node_1 = get_nodes_adj_to_node(
        edges=edges,
        node=node_1
    )

    nodes_adj_to_node_2 = get_nodes_adj_to_node(
        edges=edges,
        node=node_2,
    )

    # if node_1 and node_2 are adjacent, exit early.
    if set({node_1, node_2})\
        .intersection(
            nodes_adj_to_node_1.union(nodes_adj_to_node_2)
        ) != set({}):
        return []

    return nodes_adj_to_node_1.intersection(nodes_adj_to_node_2)

class MarkedPatternGraph(object):
    """
        A Marked Pattern represents a set of DAGs. This class uses the
        following definition, taken from Causality (Pearl, 2009). There are
        four types of arrows:

        1. marked arrow a -*-> b, signifying a directed path from a to b in
           the underlying model.
        2. an unmarked arrow a ---> b, signifying a directed path from a to b,
           or some latent common cause a <- L -> b in the underlying model.
        3. a bidirected edge a <--> b signifying some latent common cause a <-
           L -> b in the underlying model; and
        4. an undirected edge a ---- b, signifying either a -> b, a <- b, or a
           <- L -> b in the underlying model.

        Parameters:
            nodes: list[str]
                List of names of nodes in the graph.
            marked_arrows: list[tuple[str]]
                E.g. [('a', 'b'), ('b', 'c')] => a -*-> b, b -*-> c
            unmarked_arrows: list[tuple[str]]
                E.g. [('a', 'b'), ('b', 'c')] => a ---> b, b ---> c
            bidirected_edges: list[sets[str]]
                E.g. [set(('a', 'b')), set(('b', 'c'))] => a <--> b, b <--> c
            undirected_edges: list[sets[str]]
                E.g. [set(('a', 'b')), set(('b', 'c'))] => a ---- b, b ---- c

    """
    def __init__(
        self,
        nodes,
        marked_arrows=[],
        unmarked_arrows=[],
        bidirected_edges=[],
        undirected_edges=[],
        missingness_indicator_prefix='MI_'
    ):
        assert len(nodes) > 0

        self.nodes = nodes
        self.marked_arrows = marked_arrows
        self.unmarked_arrows = unmarked_arrows
        self.bidirected_edges = bidirected_edges
        self.undirected_edges = undirected_edges
        self.missingness_indicator_prefix = missingness_indicator_prefix

    def add_nodes(self, nodes):
        self.nodes = list(set(self.nodes).union(set(nodes)))

    def add_marked_arrows(self, marked_arrows):
        self.unmarked_arrows = set(self.unmarked_arrows).union(set(unmarked_arrows))

    def add_marked_arrows(self, marked_arrows):
        self.marked_arrows = set(self.marked_arrows).union(set(marked_arrows))

    def add_unmarked_arrows(self, unmarked_arrows):
        self.marked_arrows = set(self.unmarked_arrows).union(set(unmarked_arrows))

    def remove_undirected_edges(self, edges_to_remove):
        edges = list(edges_to_remove)
        set_of_sets = [frozenset(edge) for edge in edges]
        self.undirected_edges = set(self.undirected_edges) - set(set_of_sets)

    def graphviz(self):
        digraph = Digraph(comment='marked_pattern')

        for node in self.nodes:
            digraph.node(node)

        for from_node, to_node in list(self.marked_arrows):
            digraph.edge(from_node, to_node, label="*")

        for from_node, to_node in list(self.unmarked_arrows):
            digraph.edge(from_node, to_node)

        for edge_set in list(self.bidirected_edges):
            edges = list(edge_set)
            digraph.edge(edges[0], edges[1], _attributes={"dir": "both"})

        for edge_set in list(self.undirected_edges):
            edges = list(edge_set)
            digraph.edge(edges[0], edges[1], _attributes={"dir": "none"})

        return digraph

    def missingness_indicators(self):
        mi = []

        for from_node, to_node in self.marked_arrows:
            if re.search(self.missingness_indicator_prefix, to_node) != None:
                mi.append(to_node)

        return mi

