#!
"""
    PCSkeletonFinder
"""
from itertools import combinations
from concurrent.futures import ProcessPoolExecutor, as_completed

from multiprocessing import cpu_count

from constraint_based.ci_tests.bmd_is_independent import bmd_is_independent
from constraint_based.misc import setup_logging, SepSets

# from pc_skeleton_finder import PCSkeletonFinder
from data import dog_example
from graphs.partial_ancestral_graph import PartialAncestralGraph as Graph
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
        self.logging = setup_logging()
#
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
        num_cpus = cpu_count()
        # num_cpus = 1

        depth = 0
        # pylint: disable=too-many-nested-blocks
        while self._depth_not_greater_than_num_adj_nodes_per_var(depth, self.graph):
            self.logging.info("Finding skeleton. Depth: {}".format(depth))

            edges = self.graph.get_edges()

            for edge in edges:
                pairs = [(str(edge.node_1), str(edge.node_2)), (str(edge.node_2), str(edge.node_1))]

                for ordered_edge in pairs:
                    ordered_node_1, ordered_node_2 = ordered_edge

                    conditionables = list(
                        {str(i) for i in self.graph.get_neighbors(ordered_node_1)}\
                         - set({str(ordered_node_2)})
                    )

                    if len(conditionables) >= depth:

                        with ProcessPoolExecutor(max_workers=num_cpus) as executor:
                            futures = [ \
                                executor.submit( \
                                    _test_edge, \
                                    (self.data, \
                                        ordered_node_1, \
                                        ordered_node_2, \
                                        combo, \
                                        self.cond_indep_test, \
                                    )) for combo in combinations(conditionables, depth) \
                            ]

                            for f in as_completed(futures):
                                result = f.result()

                                if result is not None:
                                    self.graph.remove_edge(
                                        (result[0], result[1])
                                    )

                                    cond_sets.add(result[0], result[1], result[2])

            # break


            depth += 1

        return cond_sets

    def _depth_not_greater_than_num_adj_nodes_per_var(self, depth, graph):
        edges = list(self.graph.get_edges())

        if len(edges) == 0:
            return False

        for edge in edges:
            node_1, _, node_2 = tuple(edge)
            len_1 = len({str(i) for i in graph.get_neighbors(node_1)} - set({node_2}))
            len_2 = len({str(i) for i in graph.get_neighbors(node_2)} - set({node_1}))

            if len_1 >= depth or len_2 >= depth:
                return True

        return False

def _test_edge(
    args
):
    data, node_1, node_2, combo, cond_indep_test = args

    if cond_indep_test(
        data,
        vars_1=[node_1],
        vars_2=[node_2],
        conditioning_set=list(combo)
    ):

        return node_1, node_2, combo

    return None

if __name__ == '__main__':
    logging = setup_logging()
    df = dog_example(size=100000)

    grph = Graph(
        variables=list(df.columns),
        complete=True
    )


    skeleton_finder = PCSkeletonFinder(data=df, graph=grph)
    sep_sets = skeleton_finder.find()
