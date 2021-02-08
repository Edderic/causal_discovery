from constraint_based.ci_tests.bmd_is_independent import bmd_is_independent
from constraint_based.ci_tests.sci_is_independent import sci_is_independent
from itertools import combinations
from constraint_based.misc import conditioning_sets_satisfying_conditional_independence, key_for_pair
from graphs.marked_pattern_graph import MarkedPatternGraph
from tqdm import tqdm
from constraint_based.misc import setup_logging, ConditioningSets

# TODO: rename since this is inspired
class PCSkeletonFinder():
    """
        Finds the set of undirected edges among nodes.

        Parameters:
            var_names: [list[str]]
                The names of variables that will be in the Skeleton. e.g.
                ["Race", "Weight"]

            data [pandas.DataFrame]
                Contains data. Each column is a variable. Each column name must
                match one and only of the var_names.
    """
    def __init__(
        self,
        var_names,
        data,
        is_conditionally_independent_func=bmd_is_independent,
        missing_indicator_prefix='MI_',
        only_find_one=False
    ):
        self.var_names = var_names

        self.data = data
        self.orig_cols = list(data.columns)
        self.is_conditionally_independent_func = is_conditionally_independent_func
        self.missing_indicator_prefix = missing_indicator_prefix
        self.only_find_one = only_find_one

    def find(self):
        """
            Go through each pair of variables (in var_names).
            For each pair, find a conditioning set that renders the two variables
            independent.

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

        # make the graph as sparse as it can.
        self._skeleton_find()
        # update conditioning sets that we missed (to assist with helping find immoralities)
        self._find_more_independencies()

        return self.graph, self.cond_sets

    def _skeleton_find(self):
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
                            if self.is_conditionally_independent_func(
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

    def _find_more_independencies(self):
        depth = 0

        logging = setup_logging()

        nodes = self.graph.get_nodes()

        while self._depth_not_greater_than_num_adj_nodes_per_var(depth):
            for node_1, node_2 in combinations(nodes, 2):
                logging.info("Finding more independencies. Depth: {}".format(depth))

                node_1_neighbors = self.graph.get_neighbors(node_1)
                node_2_neighbors = self.graph.get_neighbors(node_2)

                # if no neighbors, then there exists no path to node_2, so
                # independent by default
                if len(node_1_neighbors) == 0 or len(node_2_neighbors) == 0:
                    continue

                # if node_1 and node_2 are neighbors, then there is no
                # conditioning set that separates them, so move on.
                if node_1_neighbors.intersection(set({node_2})) == set({node_2}):
                    continue

                conditionables = list(node_1_neighbors.union(node_2_neighbors) - set({node_1, node_2}))

                for conditionable in combinations(conditionables, depth):
                    if self.is_conditionally_independent_func(
                        self.data,
                        vars_1=[node_1],
                        vars_2=[node_2],
                        conditioning_set=list(conditionable)
                    ):
                        self.cond_sets.add(node_1, node_2, conditionable)
                        break

            depth += 1

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

    def _is_independent(self, var_name_1, var_name_2):
        """
            Goes through possible conditioning sets, including the empty set.
        """

        for cond_set_length in np.arange(len(self.var_names) - 1):
            cond_set_combos = combinations(
                    list(
                        set(self.var_names) - set([var_name_1, var_name_2])
                        ),
                    cond_set_length
                    )


            for cond_set_combo in cond_set_combos:
                if self.is_conditionally_independent_func(
                        data=self.data,
                        vars_1=[var_name_1],
                        vars_2=[var_name_2],
                        conditioning_set=list(cond_set_combo),
                        ):

                    return True

        return False

