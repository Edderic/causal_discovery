"""
@pytest.mark.f
    PartialAncestralGraph
"""

from itertools import combinations

from errors import ArgumentError, NotAncestralError
from constraint_based.misc import key_for_pair

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

        self.edges = {}

        if complete is True:
            assert len(self.variables) > 0

            visited = []

            for var_1, var_2 in combinations(self.variables, 2):
                key = key_for_pair((var_1, var_2))

                if key not in visited and var_1 != var_2:
                    self.add_edge(
                        '{} {}-{} {}'.format(
                            var_1,
                            self.UNCERTAIN,
                            self.UNCERTAIN,
                            var_2
                        )
                    )

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
        node_1, node_2, node_1_mark, node_2_mark = \
            self._get_nodes_and_marks(string)

        self._initialize_edge(node_1, node_2)

        return node_1 in self.edges \
            and node_2 in self.edges \
            and node_2 in self.edges[node_1][node_1_mark] \
            and node_1 in self.edges[node_2][node_2_mark]

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
        node_1, node_2, node_1_mark, node_2_mark = \
            self._get_nodes_and_marks(string)

        self._initialize_edge(node_1, node_2)

        self.edges[node_1][node_1_mark] = \
            self.edges[node_1][node_1_mark]\
            .union(set({node_2}))

        self.edges[node_2][node_2_mark] = \
            self.edges[node_2][node_2_mark]\
            .union(set({node_1}))

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

        if self._has_nodes_of_undirected_edges_with_siblings(
            node_1,
            node_2,
            node_1_mark,
            node_2_mark
        ):
            raise NotAncestralError(
                "Nodes of undirected edges can't have siblings."
            )

    def _has_nodes_of_undirected_edges_with_siblings(
        self,
        node_1,
        node_2,
        node_1_mark,
        node_2_mark
    ):
        return \
            ( \
                node_1_mark == self.TAIL and node_2_mark == self.TAIL \
                and ( \
                        self._part_of_bidirected_edge(node_1) \
                        or \
                        self._part_of_bidirected_edge(node_2)
                    ) \
            ) \
            or \
            ( \
                node_1_mark == self.ARROWHEAD and node_2_mark == self.ARROWHEAD \
                and ( \
                        self._part_of_undirected_edge(node_1) \
                        or \
                        self._part_of_undirected_edge(node_2)
                    ) \
            ) \

    def _part_of_undirected_edge(self, node):
        out_of_nodes = list(self.edges[node][self.TAIL])

        for out_of_node in out_of_nodes:
            if node in self.edges[out_of_node][self.TAIL]:
                return True

        return False

    def _part_of_bidirected_edge(self, node):
        out_of_nodes = list(self.edges[node][self.ARROWHEAD])

        for out_of_node in out_of_nodes:
            if node in self.edges[out_of_node][self.ARROWHEAD]:
                return True

        return False

    def _get_bidirected_edges(self):
        bidirected_edges = []

        for node in self.edges:
            other_nodes = self.edges[node][self.ARROWHEAD]

            for other_node in other_nodes:
                if node in list(self.edges[other_node][self.ARROWHEAD]):
                    bidirected_edges.append(
                        set({node, other_node})
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
        potential_children = list(self.edges[from_node][self.TAIL])

        for potential_child in potential_children:
            if from_node in self.edges[potential_child][self.ARROWHEAD]:
                actual_child = potential_child

                if actual_child is to_node:
                    return True

                return self._has_directed_path(
                    from_node=actual_child,
                    to_node=to_node,
                )

        return False

    def _has_directed_cycle(self, node_1, node_2):
        blah = self._has_directed_path( from_node=node_1, to_node=node_2)

        bleh = self._has_directed_path(
            from_node=node_2,
            to_node=node_1
        )

        return blah and bleh

    def _convert_mark_internally(self, mark):
        if mark in [self.LEFT_ARROWHEAD, self.RIGHT_ARROWHEAD]:
            return self.ARROWHEAD

        return mark

    def _initialize_edge(self, node_1, node_2):
        if node_1 not in self.edges:
            self.edges[node_1] = {
                self.UNCERTAIN: set({}),
                self.TAIL: set({}),
                self.ARROWHEAD: set({}),
            }

        if node_2 not in self.edges:
            self.edges[node_2] = {
                self.UNCERTAIN: set({}),
                self.TAIL: set({}),
                self.ARROWHEAD: set({}),
            }

    def _get_nodes_and_marks(self, string):
        for possible_edge in self.POSSIBLE_EDGES:
            nodes_and_edge = string.split(possible_edge)

            if len(nodes_and_edge) < 2:
                continue

            node_1 = nodes_and_edge[0].strip(' ')
            node_2 = nodes_and_edge[1].strip(' ')

            edge = possible_edge

            node_1_mark = self._convert_mark_internally(edge[0])
            node_2_mark = self._convert_mark_internally(edge[2])

            return node_1, node_2, node_1_mark, node_2_mark

        raise ArgumentError( \
            "{} has an unrecognized edge. Possible edges: {}"\
            .format(string, self.POSSIBLE_EDGES)
        )
