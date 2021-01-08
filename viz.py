from graphviz import Digraph
import re

"""
    viz
    ---

    Visualization module.

    Classes available:

    - MarkedPatternGraph
"""

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
        self.marked_arrows = list(set(self.marked_arrows).union(set(marked_arrows)))

    def graphviz(self):
        digraph = Digraph(comment='marked_pattern')

        for node in self.nodes:
            digraph.node(node)

        for from_node, to_node in self.marked_arrows:
            digraph.edge(from_node, to_node, label="*")

        for from_node, to_node in self.unmarked_arrows:
            digraph.edge(from_node, to_node)

        for edge_set in self.bidirected_edges:
            edges = list(edge_set)
            digraph.edge(edges[0], edges[1], _attributes={"dir": "both"})

        for edge_set in self.undirected_edges:
            edges = list(edge_set)
            digraph.edge(edges[0], edges[1], _attributes={"dir": "none"})

        return digraph

    def missingness_indicators(self):
        mi = []

        for from_node, to_node in self.marked_arrows:
            if re.search(self.missingness_indicator_prefix, to_node) != None:
                mi.append(to_node)

        return mi
