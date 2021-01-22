from .ci_tests.bmd_is_independent import bmd_is_independent
from itertools import combinations
from .misc import conditioning_sets_satisfying_conditional_independence, key_for_pair
from ..graphs.marked_pattern_graph import MarkedPatternGraph

class SkeletonFinder():
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
        indegree=8,
        missing_indicator_prefix='MI_',
        only_find_one=False
    ):
        self.var_names = var_names
        self.data = data.merge(
            data.isnull().add_prefix(missing_indicator_prefix),
            left_index=True,
            right_index=True
        )
        self.orig_cols = list(data.columns)
        self.is_conditionally_independent_func = is_conditionally_independent_func
        self.indegree = indegree
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

                cond_sets_satisfying_cond_indep: dict

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
        cond_sets_satisfying_cond_indep = {}

        for var_name_1, var_name_2 in combinations(self.orig_cols, 2):
            possible_conditioning_set_vars = \
                set(self.orig_cols) \
                - set([var_name_1, var_name_2])

            cond_sets = conditioning_sets_satisfying_conditional_independence(
                data=self.data,
                var_name_1=var_name_1,
                var_name_2=var_name_2,
                is_conditionally_independent_func=self.is_conditionally_independent_func,
                possible_conditioning_set_vars=possible_conditioning_set_vars,
                only_find_one=self.only_find_one
            )

            if len(cond_sets) == 0:
                undirected_edges.append(frozenset((var_name_1, var_name_2)))
            else:
                cond_sets_satisfying_cond_indep[
                    key_for_pair([var_name_1, var_name_2])
                ] = cond_sets

        marked_pattern = MarkedPatternGraph(
            nodes=list(self.data.columns),
            undirected_edges=undirected_edges
        )

        return marked_pattern, cond_sets_satisfying_cond_indep

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

