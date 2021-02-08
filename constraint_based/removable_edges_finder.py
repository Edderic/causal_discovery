from constraint_based.density_ratio_weighted_correction import DensityRatioWeightedCorrection
from constraint_based.ci_tests.bmd_is_independent import bmd_is_independent
from constraint_based.density_ratio_weighted_correction import DensityRatioWeightedCorrection
from itertools import combinations
import re

class RemovableEdgesFinder(object):

    # TODO: improve documentation
    """
        Parameters:
            data: pandas.DataFrame
            graph: Graph
                responds to ....
            data_correction: class
                Some class that responds to "correct" which should
                simulated data that adjusts for missingness. "correct" invocation
                should return a pandas.DataFrame.
            is_conditionally_independent_func: function.
                Defaults to constraint_based.ci_tests.bmd_is_independent

                Some function that tells us whether or not sets of variables
                are independent from each other given a conditioning set.
    """
    def __init__(
        self,
        data,
        graph,
        cond_sets,
        data_correction=DensityRatioWeightedCorrection,
        is_conditionally_independent_func=bmd_is_independent,
        potentially_extraneous_edges=[],
        missingness_indicator_prefix='MI_'
    ):
        self.data = data
        self.potentially_extraneous_edges = potentially_extraneous_edges
        self.graph = graph
        self.cond_sets = cond_sets
        self.data_correction = data_correction
        self.is_conditionally_independent_func = is_conditionally_independent_func
        self.missingness_indicator_prefix = missingness_indicator_prefix

    def find(self):
        if len(self.potentially_extraneous_edges) == 0:
            return []

        extraneous_edges = []

        for potentially_extraneous_edge in self.potentially_extraneous_edges:
            var_name_1, var_name_2 = tuple(potentially_extraneous_edge)

            var_1_neighbors = \
                self.graph.get_neighbors(var_name_1) - set({var_name_1, var_name_2})
            var_2_neighbors = \
                self.graph.get_neighbors(var_name_2) - set({var_name_1, var_name_2})

            if len(var_1_neighbors) > len(var_2_neighbors):
                nbrs = [var_2_neighbors, var_1_neighbors]
            else:
                nbrs = [var_1_neighbors, var_2_neighbors]

            extraneous = False

            for neighbors in nbrs:
                depth = 0

                if extraneous:
                    break

                while len(neighbors) > depth:
                    if extraneous:
                        break

                    for cond_set in combinations(neighbors, depth):
                        _data = self.data_correction(
                            data=self.data,
                            var_names=set(cond_set)\
                                .union(set({var_name_1, var_name_2})),
                            graph=self.graph
                        ).correct()

                        if self.is_conditionally_independent_func(
                               data=_data,
                               vars_1=[var_name_1],
                               vars_2=[var_name_2],
                               conditioning_set=list(cond_set),
                           ):

                           self.cond_sets.add(var_name_1, var_name_2, cond_set)

                           extraneous_edges.append(potentially_extraneous_edge)
                           extraneous = True
                           break

                    depth += 1

        return extraneous_edges

    def _missingness_indicators(self):
        nodes = list(self.graph.get_nodes())

        mi = []

        for node in nodes:
            if re.search(self.missingness_indicator_prefix, node):
                mi.append(node)

        return mi

    def _possible_conditioning_set_vars(self, var_name_1, var_name_2):
        return (
            set(self.data.columns) - set(self._missingness_indicators())
        ) - set({var_name_1, var_name_2})
