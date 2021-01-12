class PotentiallyExtraneousEdgesFinder(object):
    """
        Find edges that might be extraneous. We do this because conditional
        independence testing X _||_ Y | Z on test-wise deleted data (i.e.
        excluding rows where X,Y and set Z are missing) could lead to
        extraneous edges. If missingness indicators are children / descendants
        of X & Y, it's possible to find that X is dependent of Y given Z even
        though in the real data-generating process, X is independent of Y given
        Z.

        Potential extraneous edges are those whose pair of variables belong to
        edges that have a common variable other than those two.

        Let's say we know the true data-generating process. Let's say we have
        variables X,Y,Z,A and that X and Y are marginally independent (e.g.
        conditionally independent given the empty set).

        E.g. X->Z, Z->Y,      # X causes Z, and Z causes Y.
             X->A, Y->A,      # A is a collider.
             A->Ry            # Ry is a descendant of a collider.

        Once we have a data set produced by the above, running SkeletonFinder
        on it, we might find a graph like so:

        E.g. X-Z, Y-Z         # X and Y are connected to Z
             X-A, Y-A,        # A might be a collider (e.g. X->A, Y->A is
                              #   possible).
             A->Ry            # Ry is a descendant of a collider.
             X-Y              # Conditioning on a collider / descendant of a
                              #   collider could lead to spurious edges.


        Note: This only applies when missingness is MAR or MNAR. If data given
        is MCAR, then there shouldn't be extraneous edges and exit early.

        Parameters:
            data: pandas.DataFrame.
            marked_pattern_graph: MarkedPatternGraph
                responds to:
                    - marked_arrows
                        We assume that if MAR or MNAR, missingness indicators
                        would be a node in directed arrows. If not, then there
                        are no extraneous edges.

                        We assume that missingness indicators in nodes of
                        directed arrows have the same missingness indicator
                        prefix below.

            missingness_indicator_prefix: str. Defaults to "MI_".
    """
    def __init__(
        self,
        marked_pattern_graph,
        missingness_indicator_prefix = "MI_"
    ):
        self.marked_pattern_graph = marked_pattern_graph
        self.missingness_indicator_prefix = missingness_indicator_prefix

    def find(self):
        # missingness indicators are not descendants of other variables, so
        # this is MCAR.
        if len(self.marked_pattern_graph.marked_arrows) == 0:
            return [], self.marked_pattern_graph

        potentially_extraneous_edges = []

        for some_set in self._undirected_edges():
            node_1, node_2 = tuple(some_set)

            common_nodes = self._get_adjacent_nodes(node_1)\
                .intersection(self._get_adjacent_nodes(node_2))

            if len(common_nodes) > 0:
                potentially_extraneous_edges.append(some_set)

        return set(potentially_extraneous_edges)

    def _get_adjacent_nodes(self, node):
        new_set = set([])

        for edge in self._relevant_edges():
            some_set = set(edge)
            if some_set.intersection(set([node])) != set([]):
                new_set = new_set.union(some_set)

        return new_set - set([node])

    def _relevant_edges(self):
        return frozenset(self._undirected_edges()).union(frozenset(self._marked_arrows()))

    def _undirected_edges(self):
        return self.marked_pattern_graph.undirected_edges

    def _marked_arrows(self):
        return self.marked_pattern_graph.marked_arrows

