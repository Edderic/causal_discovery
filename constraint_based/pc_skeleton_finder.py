from constraint_based.ci_tests.bmd_is_independent import bmd_is_independent
from constraint_based.ci_tests.sci_is_independent import sci_is_independent
from itertools import combinations
from graphs.marked_pattern_graph import MarkedPatternGraph
from tqdm import tqdm
from constraint_based.misc import setup_logging, ConditioningSets

# TODO: rename since this is inspired
class PCSkeletonFinder():
    """
        Finds the set of undirected edges among nodes, along with conditioning
        sets that separate.

        Parameters:
            data [pandas.DataFrame]
                Each column represents a variable.
            cond_indep_test: function.
                Defaults to bmd_is_independent
    """
    def __init__(
        self,
        data,
        cond_indep_test=bmd_is_independent,
    ):
        self.data = data
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
        undirected_edges = []
        self.cond_sets = ConditioningSets()
        self.graph = self._init_complete_graph()

        depth = 0

        logging = setup_logging()

        while self._depth_not_greater_than_num_adj_nodes_per_var(depth):
            logging.info("Finding skeleton. Depth: {}".format(depth))

            for undirected_edge in self.graph.get_undirected_edges():

                node_1, node_2 = tuple(undirected_edge)

                for ordered_edge in [(node_1, node_2), (node_2, node_1)]:
                    ordered_node_1, ordered_node_2 = ordered_edge

                    conditionables = list(
                            self.graph.get_neighbors(ordered_node_1)\
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

                                self.graph.remove_undirected_edge(
                                    (node_1, node_2)
                                )

                                self.cond_sets.add(node_1, node_2, combo)

                                break

            depth += 1

        return self.graph, self.cond_sets

    def _init_complete_graph(self):
        return MarkedPatternGraph(
            nodes=list(self.data.columns),
            undirected_edges=list(combinations(self.orig_cols, 2))
        )

    def _depth_not_greater_than_num_adj_nodes_per_var(self, depth):
        undirected_edges = self.graph.get_undirected_edges()

        if len(undirected_edges) == 0:
            return False

        for undirected_edge in undirected_edges:
            ordered_edge_1 = tuple(undirected_edge)
            ordered_edge_2 = (ordered_edge_1[1], ordered_edge_1[0])

            for ordered_edge in [ordered_edge_1, ordered_edge_2]:
                node_1, node_2 = tuple(ordered_edge)

                node_1_neighbors_except_node_2 = list(self.graph.get_neighbors(node_1) - set({node_2}))

                if len(node_1_neighbors_except_node_2) >= depth:
                    return True

        return False
