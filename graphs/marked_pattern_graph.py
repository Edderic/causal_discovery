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

def get_common_adj_nodes_between_non_adj_nodes(edges, node_1, node_2):
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
    NO_ARROWHEAD = "-"
    UNMARKED_ARROWHEAD = "->"
    MARKED_ARROWHEAD = "*>"
    ARROWHEAD_TYPES = set({
        NO_ARROWHEAD,
        UNMARKED_ARROWHEAD,
        MARKED_ARROWHEAD
    })

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
            bidirectional_edges: list[sets[str]]
                E.g. [set(('a', 'b')), set(('b', 'c'))] => a <--> b, b <--> c
            undirected_edges: list[sets[str]]
                E.g. [set(('a', 'b')), set(('b', 'c'))] => a ---- b, b ---- c

    """
    def __init__(
        self,
        nodes,
        marked_arrows=[],
        unmarked_arrows=[],
        bidirectional_edges=[],
        undirected_edges=[],
        missingness_indicator_prefix='MI_'
    ):
        assert len(nodes) > 0

        self.nodes = nodes
        self.dict = {}

        self.add_bidirectional_edges(bidirectional_edges)

        self.add_unmarked_arrows(unmarked_arrows)
        self.add_undirected_edges(undirected_edges)
        self.add_marked_arrows(marked_arrows)

        self.missingness_indicator_prefix = missingness_indicator_prefix

    def add_nodes(self, nodes):
        self.nodes = list(set(self.nodes).union(set(nodes)))

    def add_marked_arrows(self, marked_arrows):
        """
            Parameters:
                marked_arrows: list or set of tuple[str]
        """

        for marked_arrow in list(marked_arrows):
            self.add_marked_arrow(marked_arrow)

    def add_unmarked_arrows(self, unmarked_arrows):
        """
            Parameters:
                unmarked_arrows: list or set of tuple[str]
        """

        self.add_undirected_edges(unmarked_arrows)
        self.add_arrowheads(unmarked_arrows)

    def add_unmarked_arrow(self, unmarked_arrow):
        """
            Parameters:
                unmarked_arrow: tuple[str]
        """

        self.add_arrowhead(unmarked_arrow)

    def add_marked_arrow(self, marked_arrow):
        """
            Parameters:
                marked_arrow: tuple[str]
                    Ex: ('a','b') means  a *> b
        """
        self.add_undirected_edge(marked_arrow)
        self.add_marked_arrowhead(marked_arrow)

    def add_undirected_edge(self, node_tuple):
        """
            Parameters:
                node_tuple: tuple[str]
                    Ex: ('a','b') means  a - b
        """

        node_1, node_2 = self._instantiate_node_tuple(node_tuple)

        self.dict[node_1][self.NO_ARROWHEAD] = \
            self.dict[node_1][self.NO_ARROWHEAD].union(set({node_2}))

        self.dict[node_2][self.NO_ARROWHEAD] = \
            self.dict[node_2][self.NO_ARROWHEAD].union(set({node_1}))

    def add_undirected_edges(self, undirected_edges):
        """
            Parameters:
                undirected_edges: list or set of tuple[str]
        """

        for undirected_edge in list(undirected_edges):
            self.add_undirected_edge(undirected_edge)

    def remove_undirected_edges(self, undirected_edges):
        """
            Parameters:
                undirected_edges: list or set of tuple[str]
        """

        for undirected_edge in list(undirected_edges):
            self.remove_undirected_edge(undirected_edge)

    def remove_undirected_edge(self, undirected_edge):
        """
            Parameters:
                undirected_edge: tuple[str]
        """

        node_1, node_2 = self._instantiate_node_tuple(undirected_edge)

        self.dict[node_1][self.NO_ARROWHEAD] = \
            self.dict[node_1][self.NO_ARROWHEAD] - set({node_2})

        self.dict[node_2][self.NO_ARROWHEAD] = \
            self.dict[node_2][self.NO_ARROWHEAD] - set({node_1})

    def add_bidirectional_edges(self, bidirectional_edges):
        """
            Parameters:
                node_tuples: list or set of tuple[str]

            Example:
                If a node_tuple is (a,b) then a <--> b.
        """
        for node_tuple in list(bidirectional_edges):
            self.add_bidirectional_edge(node_tuple)

    def add_bidirectional_edge(self, bidirectional_edge):
        """
            Parameters:
                bidirectional_edge: tuple[str]

            Example:
                If a node_tuple is (a,b) then a <--> b.
        """
        node_1, node_2 = tuple(bidirectional_edge)

        self.add_arrowhead((node_1, node_2))
        self.add_arrowhead((node_2, node_1))

    def add_arrowheads(self, node_tuples):
        """
            Parameters:
                node_tuples: list or set of tuple[str]

            Example:
                If a node_tuple is (a,b) then we add an arrow toward b for
                edge a to b.
        """

        for node_tuple in list(node_tuples):
            self.add_arrowhead(node_tuple)

    def add_arrowhead(self, node_tuple):
        """
            Parameters:
                node_tuple: tuple[str]

            Example:
                If node_tuple is (a,b) then we add an arrow toward b for
                edge a to b.
        """
        node_1, node_2 = self._instantiate_node_tuple(node_tuple)

        self._remove_other_arrowheads(
            node_1,
            node_2,
            except_arrowhead_type=self.UNMARKED_ARROWHEAD
        )

        self.dict[node_1][self.UNMARKED_ARROWHEAD] = \
            self.dict[node_1][self.UNMARKED_ARROWHEAD].union(set({node_2}))

    def add_marked_arrowhead(self, node_tuple):
        """
            Parameters:
                node_tuple: tuple[str]
        """
        node_1, node_2 = self._instantiate_node_tuple(node_tuple)

        self._remove_other_arrowheads(
            node_1,
            node_2,
            except_arrowhead_type=self.MARKED_ARROWHEAD
        )

        self.dict[node_1][self.MARKED_ARROWHEAD] = \
            self.dict[node_1][self.MARKED_ARROWHEAD].union(set({node_2}))

    def has_arrowhead(self, node_tuple):
        """
            If the edge has an arrowhead pointing to the second node, return
            True.  Otherwise, return False.

            Parameters:
                node_tuple: tuple
                    First node and second node determine the edge we're
                    interested in.

            Return: boolean
        """
        node_1, node_2 = self._instantiate_node_tuple(node_tuple)

        boolean = False

        for arrowhead in [self.UNMARKED_ARROWHEAD, self.MARKED_ARROWHEAD]:

            boolean = boolean or self\
                .dict[node_1][arrowhead]\
                .intersection(set({node_2})) != set({})

        return boolean

    def has_marked_arrowhead(self, node_tuple):
        """
            If the edge has a marked arrowhead pointing to the second node,
            return True.  Otherwise, return False.

            Parameters:
                node_tuple: tuple
                    First node and second node determine the edge we're
                    interested in.

            Return: boolean
        """
        node_1, node_2 = self._instantiate_node_tuple(node_tuple)

        boolean = False

        for arrowhead in [self.MARKED_ARROWHEAD]:

            boolean = boolean or self\
                .dict[node_1][arrowhead]\
                .intersection(set({node_2})) != set({})

        return boolean

    def has_unmarked_arrowhead(self, node_tuple):
        """
            If the edge has a marked arrowhead pointing to the second node,
            return True.  Otherwise, return False.

            Parameters:
                node_tuple: tuple
                    First node and second node determine the edge we're
                    interested in.

            Return: boolean
        """
        node_1, node_2 = self._instantiate_node_tuple(node_tuple)

        boolean = False

        for arrowhead in [self.UNMARKED_ARROWHEAD]:

            boolean = boolean or self\
                .dict[node_1][arrowhead]\
                .intersection(set({node_2})) != set({})

        return boolean

    def has_marked_path(self, node_tuple):
        node_1, node_2 = self._instantiate_node_tuple(node_tuple)

        return self._is_node_certainly_a_descendant(node_1, node_2)


    def _is_node_certainly_a_descendant(self, node, possibly_a_descendant_node):
        children = list(self.dict[node][self.MARKED_ARROWHEAD])

        for child in children:
            if child == possibly_a_descendant_node:
                return True

            return self._is_node_certainly_a_descendant(child, possibly_a_descendant_node)

        return False

    def get_edges(self):
        edges = set({})

        for node, ends in self.dict.items():
            for arrowhead_type in self.ARROWHEAD_TYPES:
                for other_node in list(ends[arrowhead_type]):
                    edges = edges.union(set({frozenset({node, other_node})}))

        return edges

    def get_neighbors(self, node):
        """
            Parameters:
                node: str
                    Name of a node.

            Returns: set[str]

            Ex: Let's say the graph is the following:
                node_1 -*> node_2 -- node_3

                           node_4

                >>> get_neighbors('node_2')
                set({'node_1', 'node_3'})
        """
        neighbors = set({})

        for arrowhead_type in self.ARROWHEAD_TYPES:
            neighbors = neighbors.union(self.dict[node][arrowhead_type])

        return neighbors


    def get_common_neighbors(self, node_1, node_2):
        """
            Parameters:
                node_1: str
                node_2: str

            Returns: set[str]

            Ex: Let's say the graph is the following:
                node_1 -*> node_2 -- node_3

                >>> get_common_neighbors('node_1', 'node_3')
                set({'node_2'})
        """

        return self.get_neighbors(node_1).intersection(self.get_neighbors(node_2))

    def get_nodes(self):
        return set(self.dict.keys()).union(self.nodes)

    def get_nodes_of_edges(self):
        edges = list(self.get_edges())

        nodes = set({})

        for edge in edges:
            nodes = nodes.union(set(edge))

        return nodes

    def get_undirected_edges(self):
        undirected_edges = set({})

        for node, ends in self.dict.items():
            for other_node in list(ends[self.NO_ARROWHEAD]):
                if self.dict[other_node][self.NO_ARROWHEAD].intersection(set({node})) != set({}):
                    undirected_edges = undirected_edges.union(set({frozenset({node, other_node})}))

        return undirected_edges

    def get_unmarked_arrows(self):
        unmarked_arrows = set({})

        for node, ends in self.dict.items():
            for other_node in list(ends[self.UNMARKED_ARROWHEAD]):
                if self.dict[other_node][self.NO_ARROWHEAD].intersection(set({node})) != set({}):
                    unmarked_arrows = unmarked_arrows.union(set({(node, other_node)}))

        return unmarked_arrows

    def get_marked_arrows(self):
        marked_arrows = set({})

        for node, ends in self.dict.items():
            for other_node in list(ends[self.MARKED_ARROWHEAD]):
                marked_arrows = marked_arrows.union(set({(node, other_node)}))

        return marked_arrows

    def get_bidirectional_edges(self):
        bidirectional_edges = set({})

        for node, ends in self.dict.items():
            for other_node in list(ends[self.UNMARKED_ARROWHEAD]):
                if self.dict[other_node][self.UNMARKED_ARROWHEAD].intersection(set({node})) != set({}):
                    bidirectional_edges = bidirectional_edges.union(
                        set({frozenset({node, other_node})})
                    )

        return bidirectional_edges

    def _instantiate_dict_for_var(self, var):
        self.dict[var] = {
            self.NO_ARROWHEAD:  set(), # no arrowhead from var to the vars in the list
            self.UNMARKED_ARROWHEAD: set(), # arrowhead from var to the vars in the list
            self.MARKED_ARROWHEAD: set()  # marked arrowhead from var to the vars in the list
        }

    def _remove_other_arrowheads(self, node_1, node_2, except_arrowhead_type):
        arrowhead_types = list(self.ARROWHEAD_TYPES - set({except_arrowhead_type}))

        for arrowhead_type in arrowhead_types:
            self.dict[node_1][arrowhead_type] = \
                self.dict[node_1][arrowhead_type] - set({node_2})

    def _instantiate_node_tuple(self, node_tuple):
        _node_tuple = tuple(node_tuple)

        assert len(_node_tuple) == 2

        node_1 = _node_tuple[0]
        node_2 = _node_tuple[1]

        if node_1 not in self.dict.keys():
            self._instantiate_dict_for_var(node_1)

        if node_2 not in self.dict.keys():
            self._instantiate_dict_for_var(node_2)

        return node_1, node_2

    def graphviz(self):
        digraph = Digraph(comment='marked_pattern')

        nodes = list(self.get_nodes())

        for node in nodes:
            digraph.node(node)

        marked_arrows = list(self.get_marked_arrows())

        for from_node, to_node in marked_arrows:
            digraph.edge(from_node, to_node, label="*")

        unmarked_arrows = list(self.get_unmarked_arrows())

        for from_node, to_node in unmarked_arrows:
            digraph.edge(from_node, to_node)

        bidirectional_edges = list(self.get_bidirectional_edges())
        for edge_set in bidirectional_edges:
            edges = list(edge_set)
            digraph.edge(edges[0], edges[1], _attributes={"dir": "both"})

        undirected_edges = list(self.get_undirected_edges())
        for edge_set in undirected_edges:
            edges = list(edge_set)
            digraph.edge(edges[0], edges[1], _attributes={"dir": "none"})

        return digraph

    def missingness_indicators(self):
        mi = []

        for from_node, to_node in self.marked_arrows:
            if re.search(self.missingness_indicator_prefix, to_node) != None:
                mi.append(to_node)

        return mi

    def copy(self):
        return MarkedPatternGraph(
            nodes=self.get_nodes(),
            marked_arrows=self.get_marked_arrows(),
            unmarked_arrows=self.get_unmarked_arrows(),
            undirected_edges=self.get_undirected_edges(),
            bidirectional_edges=self.get_bidirectional_edges(),
        )

    def __eq__(self, other):
        return (self.get_nodes() == other.get_nodes()) \
            and (
                    self.get_bidirectional_edges() \
                        == other.get_bidirectional_edges()
                ) \
            and (
                    self.get_undirected_edges() \
                        == other.get_undirected_edges()
                ) \
            and (
                    self.get_marked_arrows() \
                        == other.get_marked_arrows()
                ) \
            and (
                    self.get_unmarked_arrows() \
                        == other.get_unmarked_arrows()
                ) \


