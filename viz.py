from graphviz import Digraph

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
            bidirected_edges: list[tuple[str]]
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
        undirected_edges=[]
    ):
        assert len(nodes) > 0

        self.nodes = nodes
        self.marked_arrows = marked_arrows
        self.unmarked_arrows = unmarked_arrows
        self.bidirected_edges = bidirected_edges
        self.undirected_edges = undirected_edges

    def graphviz(self):
        digraph = Digraph(comment='marked_pattern')

        for node in self.nodes:
            digraph.node(node)

        for from_node, to_node in self.marked_arrows:
            digraph.edge(from_node, to_node, label="*")

        for from_node, to_node in self.unmarked_arrows:
            digraph.edge(from_node, to_node)

        for node_1, node_2 in self.bidirected_edges:
            digraph.edge(node_1, node_2, _attributes={"dir": "both"})

        for node_1, node_2 in self.undirected_edges:
            digraph.edge(node_1, node_2, _attributes={"dir": "none"})

        return digraph


