"""
@pytest.mark.f
    PartialAncestralGraph
"""

from copy import deepcopy
import re

import pandas as pd
import numpy as np

from errors import ArgumentError, NotAncestralError, EdgeTypeNotFound

def parse_nodes_edge(string, possible_edge_list=None):
    """
        Parameters:
            string: str

            Ex: 'Node 1 <-> Node 2'

        Returns: tuple
        >>> parse_nodes_edge('Node 1 <-> Node 2')
        ('Node 1', '<->', 'Node 2')

    """
    if possible_edge_list is None:
        possible_edge_list = \
            PartialAncestralGraph.POSSIBLE_EDGES

    for possible_edge in possible_edge_list:
        nodes_and_edge = string.split(possible_edge)

        if len(nodes_and_edge) < 2:
            continue

        node_1 = nodes_and_edge[0].strip(' ')
        node_2 = nodes_and_edge[1].strip(' ')

        edge = possible_edge

        return node_1, edge, node_2

    raise EdgeTypeNotFound(
        'Possible edges: {}'.format(str(possible_edge_list))
    )

class Edge(tuple):
    """
        A value object representing an edge.
            Has the following attributes:
                - node_1,
                - edge,
                - node_2
    """
    def __new__(cls, string):
        node_1, edge, node_2 = parse_nodes_edge(string)
        return tuple.__new__(Edge, (node_1, edge, node_2))

    def __init__(self, string):
        super().__init__()
        node_1, edge, node_2 = parse_nodes_edge(string)

        self.node_1 = node_1
        self.edge = edge
        self.node_2 = node_2

        self.string = string

    def __repr__(self):
        return "Edge({})".format(self.string)


class PartialAncestralGraph:
    r"""
        PartialAncestralGraph
        =====================

        Used for representing a set of Maximal Ancestral Graphs (MAGs) that have
        the same (conditional) independence relationships across observed
        variables, without having to explicitly reference latent and selection
        variables.

        References
        ----------

        Zhang, 2008: On the completeness of orientation rules for causal discovery
        in the presence of latent confounders and selection bias

        Types of edges
        --------------
            A <-- B: B causes A or some selection variable, and A doesn't cause B
            or any selection variable.

            A <-> B: A doesn't cause B or any selection variable, and B doesn't
            cause A or any selection variable. Along with adjacency between A and
            B, this implies latent common cause(s) between A and B.

            A --- B: A causes B or some selection variable, or B causes A or some
            selection variable (i.e. there's selection bias.)

            A o-o B: A and B are related to each other, but can't say much about
            it.

            A o-- B: B causes A or some selection variable.

            Circles imply that there is a MAG that exists where it is an arrowhead,
            while in another Markov Equivalent MAG, it is a tail.

        What makes a graph maximal?
        ---------------------------

        Two variables have an edge if and only if they are m-connected.

        What is a mixed graph?
        ----------------------

        A mixed graph can have any of the following edges:

            A --> B
            A <-> B
            A --- B

        What is an Ancestral Graph?
        ---------------------------

        An ancestral graph is a mixed graph with the following properties:

        1. No directed cycles
            Example of directed acyclic (allowed):

                A <-- B
                ^     |
                 \    |
                  \   |
                   \  v
                     C


            Example of directed cycle (not allowed):

                A --> B
                ^     |
                 \    |
                  \   |
                   \  v
                     C

        2. No almost-directed cycles

            Example of almost-directed cycle (not allowed):

                A <-> B
                 ^    |
                  \   |
                    \ v
                      C

        3. Undirected edges don't have parents nor spouses.

            Let's say there's an undirected edge A --- B.

            Example of a node of an undirected edge that is a spouse of another
            node (not allowed):

                A --- B
                      ^
                      |
                      v
                      C

            In the graph above, B is a spouse of C since B <-> C.

            Example of a node that is a parent of an undirected edge (not allowed):

                A --- B
                      ^
                      |
                      |
                      C


    """
    UNCERTAIN = 'o'
    TAIL = '-'
    ARROWHEAD = 'a'
    RIGHT_ARROWHEAD = '>'
    LEFT_ARROWHEAD = '<'

    MARKS = [
        UNCERTAIN,
        TAIL,
        ARROWHEAD
    ]
    POSSIBLE_EDGES = [
        '{}-{}'.format(UNCERTAIN, UNCERTAIN), # pylint: disable=duplicate-string-formatting-argument
        '{}-{}'.format(TAIL, UNCERTAIN),
        '{}-{}'.format(UNCERTAIN, TAIL),
        '{}-{}'.format(TAIL, TAIL), # pylint: disable=duplicate-string-formatting-argument
        '{}-{}'.format(LEFT_ARROWHEAD, UNCERTAIN),
        '{}-{}'.format(UNCERTAIN, RIGHT_ARROWHEAD),
        '{}-{}'.format(LEFT_ARROWHEAD, RIGHT_ARROWHEAD),
        '{}-{}'.format(TAIL, RIGHT_ARROWHEAD),
        '{}-{}'.format(LEFT_ARROWHEAD, TAIL),
    ]

    def __init__(self, variables=None, complete=False):
        if variables is None:
            self.variables = []
        else:
            self.variables = variables

        self._init_graph(complete)


    def _init_graph(self, complete):
        num_vars = len(self.variables)

        self.adjacency_matrix = pd.DataFrame(
            np.zeros((num_vars, num_vars)),
            columns=self.variables
        )

        var_names_col = '|var_names|'

        self.adjacency_matrix.loc[:, var_names_col] = deepcopy(self.variables)
        self.adjacency_matrix.set_index(var_names_col, inplace=True)

        if complete is True:
            assert len(self.variables) > 0
            all_cols = list(set(self.adjacency_matrix.columns) \
                - set({var_names_col}))

            self.adjacency_matrix.loc[:,all_cols] = \
                '{}-{}'.format(self.UNCERTAIN, self.UNCERTAIN)

            for variable in self.variables:
                self.adjacency_matrix.loc[variable, variable] = np.nan

    def remove_edge(self, nodes):
        """
            Removes the edge node_1 and node_2.

            Parameters:
                nodes: tuple[str]
                    Ex: ('node_1', 'node_2')
        """
        node_1, node_2 = tuple(nodes)

        self.adjacency_matrix.loc[node_1, node_2] = np.nan
        self.adjacency_matrix.loc[node_2, node_1] = np.nan

    def get_edges(self):
        """
            Returns: set of froz
                set of nodes
        """
        edges = []

        for node in self.adjacency_matrix.columns:
            node_edges = self.adjacency_matrix[node]

            present_node_edges = node_edges[node_edges.notnull()]

            nodes = list(present_node_edges.index)

            for edge, other_node in zip(present_node_edges, nodes):
                edges.append(
                    Edge('{} {} {}'.format(node, edge, other_node))
                )

        return edges

    def get_neighbors(self, node):
        """
            Parameter:
                node: str

            Returns: set
                set of nodes
        """
        edges = self.adjacency_matrix[node]

        neighbors = set(list(edges[edges.notnull()].index))

        return neighbors - set({node})

    def has_adjacency(self, nodes):
        """
            Returns true if there is some sort of edge between
            two nodes.

            Parameters:
                nodes: tuple
        """
        node_1, node_2 = tuple(nodes)

        return not pd.isna(self.adjacency_matrix.loc[node_1, node_2])

    def has_edge(self, string):
        """
            True if an edge exists in the graph.

            Parameter:
                string: str
                    Examples:
                        'A o-o B'
                        'A --o B'
                        'A o-- B'
                        'A --- B'
                        'A <-o B'
                        'A o-> B'
                        'A <-> B'
                        'A --> B'
                        'A <-- B'

            Returns: bool
        """
        node_1, marks, node_2 = \
            self._get_nodes_and_marks(string)

        internal_marks = self._convert_arrowheads(marks)

        try:
            return self.adjacency_matrix.loc[node_1, node_2] \
                == internal_marks
        except KeyError:
            return False

    def add_edge(self, string):
        """
            Parameter:
                string: str
                    Examples:
                        'A o-o B'
                        'A --o B'
                        'A o-- B'
                        'A --- B'
                        'A <-o B'
                        'A o-> B'
                        'A <-> B'
                        'A --> B'
                        'A <-- B'
        """
        node_1, marks, node_2 = \
            self._get_nodes_and_marks(string)

        self.adjacency_matrix.loc[node_1, node_2] = \
            self._convert_arrowheads(marks)
        self.adjacency_matrix.loc[node_2, node_1] = \
            self._convert_arrowheads(marks[::-1])

        if self._has_directed_cycle(
            node_1,
            node_2
        ):
            raise NotAncestralError(
                "There's a directed cycle between {} and {}."\
                .format(node_1, node_2)
            )

        if self._has_almost_directed_cycle():
            raise NotAncestralError(
                "There's an almost-directed cycle between {} and {}."\
                .format(node_1, node_2)
            )

        if self._has_nodes_of_undirected_edges_with_siblings():
            raise NotAncestralError(
                "Nodes of undirected edges can have siblings."
            )

    def _has_nodes_of_undirected_edges_with_siblings(self):
        undirected_edges = self._get_undirected_edges()

        for undirected_edge in undirected_edges:
            node_1, node_2 = tuple(undirected_edge)

            if self._part_of_bidirected_edge(node_1) or self._part_of_bidirected_edge(node_2):
                return True

        return False

    def _part_of_undirected_edge(self, node):
        edges = self.adjacency_matrix[node]
        undirected_edge = '{}-{}'.format(self.TAIL, self.TAIL)

        other_nodes = edges[edges == undirected_edge].index

        return len(other_nodes) > 0

    def _part_of_bidirected_edge(self, node):
        edges = self.adjacency_matrix[node]
        bidirected_edge = '{}-{}'.format(self.ARROWHEAD, self.ARROWHEAD)

        other_nodes = edges[edges == bidirected_edge].index

        return len(other_nodes) > 0

    def _get_undirected_edges(self):
        undirected_edges = []
        undirected_edge = '{}-{}'.format(self.TAIL, self.TAIL)

        for column_name in self.adjacency_matrix.columns:
            edges = self.adjacency_matrix[column_name]
            other_nodes = edges[edges == undirected_edge].index

            for other_node in other_nodes:
                undirected_edges.append(set({column_name, other_node}))

        for column_name in self.adjacency_matrix.T.columns:
            edges = self.adjacency_matrix.T[column_name]
            other_nodes = edges[edges == undirected_edge].index

            for other_node in other_nodes:
                undirected_edges.append(set({column_name, other_node}))

        return undirected_edges


    def _get_bidirected_edges(self):
        bidirected_edges = []

        bidirected = '{}-{}'.format(self.ARROWHEAD, self.ARROWHEAD)

        for column_name in self.adjacency_matrix.columns:
            column = self.adjacency_matrix[column_name]
            nodes = column[column == bidirected].index

            for node in nodes:
                bidirected_edges.append(
                    set({column_name, node})
                )

        return bidirected_edges

    def _has_almost_directed_cycle(self):
        bidirected_edges = self._get_bidirected_edges()

        for bidirected_edge in bidirected_edges:
            node_1, node_2 = tuple(bidirected_edge)

            if self._has_directed_path(
                from_node=node_1,
                to_node=node_2
                ) \
                or \
                self._has_directed_path(
                    from_node=node_2,
                    to_node=node_1
                ):

                return True

        return False

    def _has_directed_path(self, from_node, to_node):
        directed_edge = '{}-{}'.format(self.TAIL, self.ARROWHEAD)
        node_1_edges = self.adjacency_matrix.loc[from_node]
        children = node_1_edges[node_1_edges == directed_edge].index

        for child in children:
            if child == to_node:
                return True

            return self._has_directed_path(
                from_node=child,
                to_node=to_node,
            )

        return False

    def _has_directed_cycle(self, node_1, node_2):
        blah = self._has_directed_path(from_node=node_1, to_node=node_2)

        bleh = self._has_directed_path(
            from_node=node_2,
            to_node=node_1
        )

        return blah and bleh

    def _convert_arrowheads(self, edge):
        return re.sub(
            '[{}]|[{}]'.format(self.LEFT_ARROWHEAD, self.RIGHT_ARROWHEAD)
            ,
            self.ARROWHEAD,
            edge
        )

    def _get_nodes_and_marks(self, string):
        for possible_edge in self.POSSIBLE_EDGES:
            nodes_and_edge = string.split(possible_edge)

            if len(nodes_and_edge) < 2:
                continue

            node_1 = nodes_and_edge[0].strip(' ')
            node_2 = nodes_and_edge[1].strip(' ')

            return node_1, possible_edge, node_2

        raise ArgumentError( \
            "{} has an unrecognized edge. Possible edges: {}"\
            .format(string, self.POSSIBLE_EDGES)
        )
