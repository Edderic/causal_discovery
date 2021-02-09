from constraint_based.ci_tests.bmd_is_independent import bmd_is_independent
from constraint_based.ci_tests.sci_is_independent import sci_is_independent
from itertools import combinations
from graphs.marked_pattern_graph import MarkedPatternGraph
from tqdm import tqdm
from constraint_based.misc import setup_logging, ConditioningSets, key_for_pair

class FindMoreCondIndeps():
    """
        Meant to be run after PCSkeletonFinder, as the latter might miss some
        independencies when conditional independence tests are
        imperfect. One example of this situation is when we have
        two variables that become independent when a variable is
        conditioned on, but the latter isn't actually adjacent to
        either of those variables.

        Parameters:
            data [pandas.DataFrame]
                Contains data. Each column is a variable.
            graph: Graph
            cond_sets: misc.ConditioningSets
            cond_indep_test: function
                Defaults to bmd_is_independent
    """
    def __init__(
        self,
        data,
        graph,
        cond_sets,
        cond_indep_test=bmd_is_independent,
    ):
        self.data = data
        self.graph = graph
        self.cond_indep_test = cond_indep_test
        self.cond_sets = cond_sets

    def find(self):
        """
            Go through each pair of variables.
            For each pair, find a conditioning set that renders the two variables
            independent.

        """
        depth = 0

        logging = setup_logging()

        nodes = self.graph.get_nodes()

        undirected_edges = self.graph.get_undirected_edges()

        while self._depth_not_greater_than_num_adj_nodes_per_var(depth):
            visited = {}

            for node_1, node_2 in combinations(nodes, 2):
                _neighbors = self.graph.get_neighbors(node_1).union(self.graph.get_neighbors(node_2))
                if key_for_pair((node_1, node_2)) in visited:
                    continue

                if len(_neighbors.intersection(set({node_1, node_2}))) > 0:
                    continue

                if node_1 == node_2:
                    continue

                if not self.graph.has_path((node_1, node_2)):
                    continue

                neighbors = _neighbors - set({node_1, node_2})

                for conditionable in combinations(neighbors, depth):
                    if self.cond_indep_test(
                        self.data,
                        vars_1=[node_1],
                        vars_2=[node_2],
                        conditioning_set=list(conditionable)
                    ):
                        self.cond_sets.add(node_1, node_2, conditionable)

                visited[key_for_pair((node_1, node_2))] = True

            depth += 1

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

