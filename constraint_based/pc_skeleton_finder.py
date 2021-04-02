"""
    PCSkeletonFinder
"""
from itertools import combinations

from constraint_based.ci_tests.bmd_is_independent import bmd_is_independent
from constraint_based.misc import setup_logging, SepSets

# pylint: disable=too-few-public-methods
class PCSkeletonFinder():
    """
        Finds the set of undirected edges among nodes, along with conditioning
        sets that separate.

        Parameters:
            data [pandas.DataFrame]
                Each column represents a variable.
            graph:
                Something that responds to the following:
                    - get_edges()
                    - get_neighbors(node)
                    - remove_edge((node_1, node_2))
            cond_indep_test: function.
                Defaults to bmd_is_independent
    """
    def __init__(
        self,
        data,
        graph,
        cond_indep_test=bmd_is_independent,
    ):
        self.data = data
        self.graph = graph
        self.orig_cols = list(data.columns)
        self.cond_indep_test = cond_indep_test

    def find(self):
        """
            For each pair of undirected edges, if possible, find a conditioning
            set that renders the two variables independent.

            Returns:
                marked_pattern: MarkedPatternGraph
                    It'll store the skeleton (a set of undirected edges). It
                    can be used for later steps, such as finding immoralities.

                cond_sets: dict

                    key: str.
                        The pair of variables that are conditionally
                        independent, delimited by " _||_ ".  E.g. If "X _||_ Y"
                        is a key, then X and Y are the variables that are
                        conditionally independent.

                    value: list(sets(str)).
                        The conditioning sets that make X and Y conditionally
                        independent.
        """
        cond_sets = SepSets()

        depth = 0

        logging = setup_logging()

        # pylint: disable=too-many-nested-blocks
        while self._depth_not_greater_than_num_adj_nodes_per_var(depth):
            logging.info("Finding skeleton. Depth: {}".format(depth))

            edges = self.graph.get_edges()

            for edge in edges:
                node_1, _, node_2 = tuple(edge)

                for ordered_edge in [(node_1, node_2), (node_2, node_1)]:
                    ordered_node_1, ordered_node_2 = ordered_edge

                    conditionables = list(
                            {str(i) for i in self.graph.get_neighbors(ordered_node_1)}\
                             - set({ordered_node_2})
                    )

                    if len(conditionables) >= depth:
                        for combo in combinations(conditionables, depth):
                            if self.cond_indep_test(
                                self.data,
                                vars_1=[ordered_node_1],
                                vars_2=[ordered_node_2],
                                conditioning_set=list(combo)
                            ):

                                self.graph.remove_edge(
                                    (node_1, node_2)
                                )

                                cond_sets.add(node_1, node_2, combo)

                                break

            depth += 1

        return cond_sets

    def _depth_not_greater_than_num_adj_nodes_per_var(self, depth):
        edges = list(self.graph.get_edges())

        if len(edges) == 0:
            return False

        for edge in edges:
            node_1, _, node_2 = tuple(edge)
            len_1 = len({str(i) for i in self.graph.get_neighbors(node_1)} - set({node_2}))
            len_2 = len({str(i) for i in self.graph.get_neighbors(node_2)} - set({node_1}))

            if len_1 >= depth or len_2 >= depth:
                return True

        return False
