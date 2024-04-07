#!
"""
    PCSkeletonFinder
"""


from itertools import combinations
import time
import dask
from collections import deque

from distributed import Client

from causal_discovery.constraint_based.ci_tests.bmd_is_independent import bmd_is_independent
from causal_discovery.constraint_based.misc import setup_logging, SepSets

# from pc_skeleton_finder import PCSkeletonFinder
from causal_discovery.data import dog_example
from causal_discovery.errors import SubprocessError
from causal_discovery.graphs.partial_ancestral_graph import PartialAncestralGraph as Graph
# pylint: disable=too-few-public-methods

def get_num_workers(client):
    """
    Get the number of workers from a Dask client
    """
    return len(client.scheduler_info()['workers'])


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
        timeout_limit=172_800,
        client=None
    ):
        if client is None:
            self.client = Client()
        else:
            self.client = client

        self.data = data
        self.graph = graph
        self.orig_cols = list(data.columns)
        self.cond_indep_test = cond_indep_test
        self.logging = setup_logging()
        self.timeout_limit = timeout_limit

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
        num_cpus = get_num_workers(client = self.client)

        depth = 0
        # pylint: disable=too-many-nested-blocks
        while self._depth_not_greater_than_num_adj_nodes_per_var(
                depth, self.graph):
            self.logging.debug("Finding skeleton. Depth: {}".format(depth))

            edges = self.graph.get_edges()

            lazy_results = []
            num_batches = min(len(edges), num_cpus)
            batched_edges = batch_lists(num_batches=num_batches, arr=edges)

            for batch in batched_edges:
                lazy_results.append(
                    dask.delayed(
                        process_edges
                    )(
                        batch,
                        self.graph,
                        self.data,
                        depth,
                        self.cond_indep_test
                    )
                )

            computed = list(dask.compute(*lazy_results))

            for item in computed:
                if item is None:
                    continue

                cond_sets.add(item[0], item[1], item[2])
                self.graph.remove_edge((item[0], item[1]))

            depth += 1

        return cond_sets

    def _depth_not_greater_than_num_adj_nodes_per_var(self, depth, graph):
        edges = list(self.graph.get_edges())

        if len(edges) == 0:
            return False

        for edge in edges:
            node_1, _, node_2 = tuple(edge)
            len_1 = len({str(i) for i in graph.get_neighbors(node_1)}
                        - {node_2})
            len_2 = len({str(i) for i in graph.get_neighbors(node_2)}
                        - {node_1})

            if len_1 >= depth or len_2 >= depth:
                return True

        return False


def batch_lists(num_batches, arr):
    """
        Batches a list.

        Parameters:

            num_batches: int
            arr: list

    """
    if num_batches > len(arr):
        return [[arr[i]] for i in range(len(arr))]

    batches = []

    for _ in range(num_batches):
        batches.append([])

    for index, val in enumerate(arr):
        batches[index % num_batches].append(val)

    return batches


def process_edges(edges, graph, data, depth, cond_indep_test):
    """
        Get a list of edges. For each edge, see if it doesn't exist (i.e.
        there's a conditioning set that separates the nodes of the edge). If
        so, add it to the queue, for later removal.

        Parameters:
            edges: list[Edge],
            graph:
                responds to:
                    - get_neighbors(node)
            data: pd.DataFrame
            depth: Value
            cond_indep_test: function
                returns Boolean
    """

    # edge, graph, data, depth, cond_indep_test

    for edge in edges:
        pairs = [
            (str(edge.node_1), str(edge.node_2)),
            (str(edge.node_2), str(edge.node_1))
        ]

        # Note: if one of the pairs passes, then we don't need to do the
        # preceding one.'
        for ordered_edge in pairs:
            ordered_node_1, ordered_node_2 = ordered_edge

            conditionables = list(
                {str(i) for i in graph.get_neighbors(ordered_node_1)}
                - set({str(ordered_node_2)})
            )

            if len(conditionables) >= depth:
                for combo in combinations(conditionables, depth):
                    if cond_indep_test(
                        data,
                        vars_1=[ordered_node_1],
                        vars_2=[ordered_node_2],
                        conditioning_set=list(combo)
                    ):
                        return (ordered_node_1, ordered_node_2, combo)

if __name__ == '__main__':
    logging = setup_logging()
    df = dog_example(size=100000)

    grph = Graph(
        variables=list(df.columns),
        complete=True
    )

    skeleton_finder = PCSkeletonFinder(data=df, graph=grph)
    sep_sets = skeleton_finder.find()
