"""
@pytest.mark.f
    PartialAncestralGraph
"""

from itertools import combinations
import re

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
        _string = str(string)
        nodes_and_edge = _string.split(possible_edge)

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

        self.node_1 = nodeify(node_1)

        self.node_1_mark = edge[0]
        self.node_2_mark = edge[2]

        self.node_1_mark_converted = convert_arrowhead(self.node_1_mark)
        self.node_2_mark_converted = convert_arrowhead(self.node_2_mark)

        self.node_2 = nodeify(node_2)

    def __repr__(self):
        return "{} {} {}".format(
            self.node_1,
            self.get_marks(),
            self.node_2
        )

    def get_node_other_than(self, other_node):
        """
            Parameters:
                other_node: str

            Returns: str
        """
        other = nodeify(other_node)

        if other not in (self.node_1, self.node_2):
            raise ArgumentError(
                "{} not applicable. Doesn't match {} or {}"\
                .format(other, self.node_1, self.node_2)
            )

        if self.node_1 == other:
            return self.node_2

        return self.node_1

    def get_marks(self):
        """
            Ex: If we have edge "A o-> B", this will return "o->".
        """
        return "{}-{}".format(
            self.node_1_mark,
            self.node_2_mark,
        )

    def is_undirected(self):
        """
            If the edge has tails for both of its marks, then return True, and
            return False otherwise.

            Returns: bool
        """
        return self.node_1_mark_converted == PartialAncestralGraph.TAIL \
            and self.node_2_mark_converted == PartialAncestralGraph.TAIL

    def is_bidirected(self):
        """
            If the edge has an arrowhead at both of its sides, then return
            True, and return False otherwise.

            Returns: bool
        """
        return self.node_1_mark_converted == PartialAncestralGraph.ARROWHEAD \
            and self.node_2_mark_converted == PartialAncestralGraph.ARROWHEAD

    def undetermined_of(self, node):
        """
            If the mark next to the node is an uncertain one, then return True.
            Otherwise, return False.

            Parameters:
                node: str
        """
        _node = nodeify(node)

        if _node not in (self.node_1, self.node_2):
            raise ArgumentError('Node {} not found'.format(_node))

        if _node == self.node_1 and self.node_1_mark == PartialAncestralGraph.UNCERTAIN:
            return True
        if _node == self.node_2 and self.node_2_mark == PartialAncestralGraph.UNCERTAIN:
            return True

        return False

    def out_of(self, node):
        """
            If there is a tail next to a node, then return True, and
            False otherwise.

            Parameters:
                node: str
            Returns: bool

            Raises:
                ArgumentError if node not found.
        """
        _node = nodeify(node)

        if _node not in (self.node_1, self.node_2):
            raise ArgumentError('Node {} not found'.format(_node))

        if _node == self.node_1 and self.node_1_mark == PartialAncestralGraph.TAIL:
            return True
        if _node == self.node_2 and self.node_2_mark == PartialAncestralGraph.TAIL:
            return True
        return False

    def into(self, node):
        """
            If there is an arrowhead next to a node, then return True,
            and False otherwise.

            Parameters:
                node: str
            Returns: bool

            Raises:
                ArgumentError if node not found.
        """
        _node = nodeify(node)

        # if _node not in (self.node_1, self.node_2):
            # raise ArgumentError('Node {} not found'.format(_node))

        if _node == self.node_1 \
            and self.node_1_mark_converted == PartialAncestralGraph.ARROWHEAD:
            return True

        if _node == self.node_2 \
            and self.node_2_mark_converted == PartialAncestralGraph.ARROWHEAD:
            return True
        return False

    def set_into(self, node):
        """
            Add the appropriate arrowhead for the given node.

            Parameters:
                node: str

            Raises:
                ArgumentError if node not found.
        """
        _node = nodeify(node)

        if _node not in (self.node_1, self.node_2):
            raise ArgumentError('Node {} not found'.format(_node))

        if _node == self.node_1:
            self.node_1_mark = PartialAncestralGraph.LEFT_ARROWHEAD
            self.node_1_mark_converted = PartialAncestralGraph.ARROWHEAD
        else:
            self.node_2_mark = PartialAncestralGraph.RIGHT_ARROWHEAD
            self.node_2_mark_converted = PartialAncestralGraph.ARROWHEAD

    def set_out_of(self, node):
        """
            Add a tail next to the given node.

            Parameters:
                node: str

            Raises:
                ArgumentError if node not found.
        """
        _node = nodeify(node)

        if _node not in (self.node_1, self.node_2):
            raise ArgumentError('Node {} not found'.format(_node))

        if _node == self.node_1:
            self.node_1_mark = PartialAncestralGraph.TAIL
            self.node_1_mark_converted = PartialAncestralGraph.TAIL
        else:
            self.node_2_mark = PartialAncestralGraph.TAIL
            self.node_2_mark_converted = PartialAncestralGraph.TAIL

    def __eq__(self, other_edge):
        return (
            other_edge.node_1 == self.node_1 \
            and other_edge.node_1_mark_converted == self.node_1_mark_converted \
            and other_edge.node_2 == self.node_2 \
            and convert_arrowhead(other_edge.node_2_mark) == convert_arrowhead(self.node_2_mark)
        ) or (
            other_edge.node_2 == self.node_1 \
            and other_edge.node_2_mark_converted == self.node_1_mark_converted \
            and other_edge.node_1 == self.node_2 \
            and other_edge.node_1_mark_converted == self.node_2_mark_converted
        )

def convert_arrowhead(string):
    """
        Convert a LEFT_ARROWHEAD or RIGHT_ARROWHEAD to an ARROWHEAD
    """
    return re.sub(
        f"[{PartialAncestralGraph.LEFT_ARROWHEAD}]|[{PartialAncestralGraph.RIGHT_ARROWHEAD}]",
        PartialAncestralGraph.ARROWHEAD,
        string
    )

def nodeify(other_node):
    """
        Ensure that we get a Node instance
    """
    if isinstance(other_node, Node):
        return other_node
    if isinstance(other_node, str):
        return Node(other_node)
    return None

class Node:
    """
        A node. Might or might not have edges.
    """
    def __init__(self, variable):
        self.variable = variable
        self.edges = {}

    def __eq__(self, other_node):
        return self.variable == nodeify(other_node).variable


    def add_edge(self, edge):
        """
            Parameters:
                edge: Edge
        """
        self.edges[str(edge.get_node_other_than(self.variable))] = edge

    def remove_edge(self, edge_or_node):
        """
            Parameters:
                edge: Edge or Node
        """
        if isinstance(edge_or_node, Edge):
            self.edges.pop(str(edge_or_node.get_node_other_than(self.variable)), None)
        elif isinstance(edge_or_node, Node):
            self.edges.pop(str(edge_or_node), None)

    def get_edge(self, node):
        """
            Parameters:
                edge: Edge
        """

        return self.edges[str(node)]

    def get_edges(self):
        """
            Returns: list of Edge
        """
        edges = []

        for other_var in self.edges:
            edges.append(self.edges[other_var])

        return edges

    def get_neighbors(self):
        """
            Returns: list of str
        """

        return map(nodeify, list(self.edges.keys()))

    def get_children(self):
        """
            Returns: list[Node]
        """
        children = []

        for adjacent_node, edge in self.edges.items():
            if edge.out_of(self) and edge.into(adjacent_node):
                children.append(Node(adjacent_node))

        return children

    def is_adjacent(self, node):
        """
            Returns: bool
        """
        return str(node) in self.edges.keys()

    def __repr__(self):
        return f"{self.variable}"

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
            variables = []

        self.variables = {}

        for variable in variables:
            self.variables[variable] = Node(variable)

        if complete:
            self._init_complete_graph()

    def __repr__(self):
        return f"Variables: {self.variables}\nEdges: {self.get_edges()}"

    def _init_complete_graph(self):
        assert len(self.variables) > 0

        for var_1_str, var_2_str in combinations(self.variables.keys(), 2):
            var_1 = self.variables[var_1_str]
            var_2 = self.variables[var_2_str]

            self.add_edge(Edge(f"{var_1} o-o {var_2}"))

    def remove_edge(self, nodes):
        """
            Removes the edge node_1 and node_2.

            Parameters:
                nodes: tuple[str]
                    Ex: ('node_1', 'node_2')
        """
        variable_1 = self.variables[str(nodes[0])]
        variable_2 = self.variables[str(nodes[1])]

        variable_1.remove_edge(variable_2)
        variable_2.remove_edge(variable_1)

    def get_edge(self, node_1, node_2):
        """
            Returns the edge associated to the pair of nodes.

            Parameters:
                node_1: str
                node_2: str

            Returns: Edge or None
        """

        try:
            _node_1 = self.variables[str(node_1)]

            return _node_1.get_edge(node_2)
        except KeyError:
            return None

    def get_adjacent_pairs_of_edges(self):
        """
            Returns: tuple
        """

        pairs = []

        for _, middle_node in self.variables.items():
            neighbors = list(middle_node.get_neighbors())
            if len(neighbors) <= 1:
                continue

            for node_1, node_2 in combinations(neighbors, 2):
                node_1 = self.variables[str(node_1)]
                node_2 = self.variables[str(node_2)]

                pairs.append((
                    node_1.get_edge(middle_node),
                    middle_node.get_edge(node_2),
                ))

        return pairs

    def get_edges(self):
        """
            Returns: list of Edges
        """
        edges = []

        for _, node in self.variables.items():
            node_edges = node.get_edges()

            for node_edge in node_edges:
                if node_edge not in edges:
                    edges.append(node_edge)

        return edges

    def get_neighbors(self, node):
        """
            Parameter:
                node: str

            Returns: set
                set of nodes
        """
        _node = self.variables[str(node)]
        return _node.get_neighbors()

    def get_nodes(self):
        """
            Returns all the variables (may or may not be connected to other
            nodes).
        """
        return set(self.variables)

    def has_adjacency(self, nodes):
        """
            Returns true if there is some sort of edge between
            two nodes.

            Parameters:
                nodes: tuple
        """
        node_1 = self.variables[nodes[0]]
        node_2 = self.variables[nodes[1]]

        return node_1.is_adjacent(node_2)

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
        _node_1, _, _node_2 = \
            self._get_nodes_and_marks(string)

        try:
            node_1 = self.variables[_node_1]
            return node_1.get_edge(_node_2) == Edge(string)
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
        edge = Edge(string)

        if str(edge.node_1) not in self.variables.keys():
            self.variables[str(edge.node_1)] = edge.node_1

        if str(edge.node_2) not in self.variables.keys():
            self.variables[str(edge.node_2)] = edge.node_2

        node_1 = self.variables[str(edge.node_1)]
        node_2 = self.variables[str(edge.node_2)]

        node_1.add_edge(edge)
        node_2.add_edge(edge)

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

    def set_edge(self, string):
        """
            Alias for add_edge.

            Parameters:
                string: str
                    Ex: "Some Node A o-> Some Node B"
        """
        self.add_edge(string)

    def _has_nodes_of_undirected_edges_with_siblings(self):
        undirected_edges = self._get_undirected_edges()

        for undirected_edge in undirected_edges:
            if self._part_of_bidirected_edge(undirected_edge.node_1) \
                or self._part_of_bidirected_edge(undirected_edge.node_2):
                return True

        return False

    def _part_of_undirected_edge(self, node):
        _node = self.variables[str(node)]

        edges = _node.get_edges()
        for edge in edges:
            if edge.is_undirected():
                return True

        return False

    def _part_of_bidirected_edge(self, node):
        _node = self.variables[str(node)]

        edges = _node.get_edges()
        for edge in edges:
            if edge.is_bidirected():
                return True

        return False

    def _get_undirected_edges(self):
        undirected_edges = []

        edges = self.get_edges()

        for edge in edges:
            if edge.is_undirected():
                undirected_edges.append(edge)

        return undirected_edges

    def _get_bidirected_edges(self):
        bidirected_edges = []

        for edge in self.get_edges():
            if edge.is_bidirected():
                bidirected_edges.append(edge)

        return bidirected_edges

    def _has_almost_directed_cycle(self):
        bidirected_edges = self._get_bidirected_edges()

        for bidirected_edge in bidirected_edges:
            node_1 = self.variables[str(bidirected_edge.node_1)]
            node_2 = self.variables[str(bidirected_edge.node_2)]

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
        edges = from_node.get_edges()

        for edge in edges:
            if edge.out_of(from_node) and edge.into(to_node):
                return True

            other_node = self.variables[str(edge.get_node_other_than(from_node))]

            children = other_node.get_children()

            for child in children:
                if from_node == child:
                    continue

                return self._has_directed_path(
                    from_node=self.variables[str(child)],
                    to_node=to_node,
                )

        return False

    def _has_directed_cycle(self, node_1, node_2):
        return self._has_directed_path(
            from_node=node_1,
            to_node=node_2
        ) and self._has_directed_path(
            from_node=node_2,
            to_node=node_1
        )

    def _get_nodes_and_marks(self, string_or_edge):
        if isinstance(string_or_edge, Edge):
            return string_or_edge.node_1, \
                string_or_edge.get_marks(), \
                string_or_edge.node_2

        for possible_edge in self.POSSIBLE_EDGES:
            nodes_and_edge = string_or_edge.split(possible_edge)

            if len(nodes_and_edge) < 2:
                continue

            node_1 = nodes_and_edge[0].strip(' ')
            node_2 = nodes_and_edge[1].strip(' ')

            return node_1, possible_edge, node_2

        raise ArgumentError( \
                "{} has an unrecognized edge. Possible edges: {}"\
                .format(string_or_edge, self.POSSIBLE_EDGES)
                )
