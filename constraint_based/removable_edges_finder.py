from .density_ratio_weighted_correction import DensityRatioWeightedCorrection
from .ci_tests.bmd_is_independent import bmd_is_independent
from .density_ratio_weighted_correction import DensityRatioWeightedCorrection
from .misc import conditioning_sets_satisfying_conditional_independence

class RemovableEdgesFinder(object):
    """
        Parameters:
            data: pandas.DataFrame
            marked_pattern_graph: graphs.MarkedPatternGraph
                TODO: do we really need this? What is this being used for?
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
        marked_pattern_graph,
        data_correction=DensityRatioWeightedCorrection,
        is_conditionally_independent_func=bmd_is_independent,
        potentially_extraneous_edges=[],
        missingness_indicator_prefix='MI_'
    ):
        self.data = data
        self.potentially_extraneous_edges = potentially_extraneous_edges
        self.marked_pattern_graph = marked_pattern_graph
        self.data_correction = data_correction
        self.is_conditionally_independent_func = is_conditionally_independent_func
        self.missingness_indicator_prefix = missingness_indicator_prefix

    def find(self):
        if len(self.potentially_extraneous_edges) == 0:
            return []

        extraneous_edges = []

        for potentially_extraneous_edge in self.potentially_extraneous_edges:
            var_name_1, var_name_2 = tuple(potentially_extraneous_edge)

            cond_sets = conditioning_sets_satisfying_conditional_independence(
                self.data,
                var_name_1,
                var_name_2,
                is_conditionally_independent_func=self.is_conditionally_independent_func,
                possible_conditioning_set_vars=self._possible_conditioning_set_vars(
                    var_name_1, var_name_2
                ),
                indegree=8,
                only_find_one=True,
                data_correction=self.data_correction,
                marked_pattern_graph=self.marked_pattern_graph
            )

            if len(cond_sets) > 0:
                extraneous_edges.append(potentially_extraneous_edge)

        return extraneous_edges

    def _missingness_indicators(self):
        return self.data.columns[self.data.columns.str.contains(self.missingness_indicator_prefix)]

    def _possible_conditioning_set_vars(self, var_name_1, var_name_2):
        return (
            set(self.data.columns) - set(self._missingness_indicators())
        ) - set({var_name_1, var_name_2})
