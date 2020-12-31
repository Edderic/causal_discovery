"""
    causal_discovery
    ----------------

    A module to help with causal discovery tasks

    Classes:
        - SkeletonFinder

    Functions:
        - conditioning_sets_satisfying_conditional_independence
"""
from information_theory import sci_is_independent
from itertools import combinations
from viz import MarkedPatternGraph
import numpy as np


def conditioning_sets_satisfying_conditional_independence(
    data,
    var_name_1,
    var_name_2,
    is_conditionally_independent_func,
    possible_conditioning_set_vars=[],
    indegree=8,
    only_find_one=False
):
    """
        Does pairwise conditional independence testing. Tries to find
        conditioning sets that satisfy X _||_ Y | Z, where Z is the
        conditioning set.

        Parameters:
            data: pandas.DataFrame

            var_name_1: str
                Represented by X in X _||_ Y | Z.

            var_name_2: str
                Represented by Y in X _||_ Y | Z.

            possible_conditioning_set_vars: list.
                The list of variables that we could possibly condition on.
                Represented by Z in X _||_ Y | Z.

            indegree: int. >= 1
                The maximum number of edges a node can have. This limits the search

            only_find_one: bool. Defaults to False.
                If a conditioning set satisfying the conditional independence
                statement is found, then quickly return.

        Returns: list if there's at least one. Otherwise, returns None
    """

    assert indegree >= 1

    assert len(
               set([var_name_1, var_name_2])\
                   .intersection(set(possible_conditioning_set_vars)
               )
           ) == 0

    if indegree < len(possible_conditioning_set_vars):
        combo_length = indegree
    else:
        combo_length = len(possible_conditioning_set_vars)

    cond_set_combo_satisfies_cond_ind = []

    for cond_set_length in np.arange(combo_length + 1):
        cond_set_combos = combinations(
            possible_conditioning_set_vars,
            cond_set_length
        )

        for cond_set_combo in cond_set_combos:

            if is_conditionally_independent_func(
                   data=data,
                   vars_1=[var_name_1],
                   vars_2=[var_name_2],
                   conditioning_set=list(cond_set_combo),
               ):

               cond_set_combo_satisfies_cond_ind.append(set(cond_set_combo))

               if only_find_one:
                   return cond_set_combo_satisfies_cond_ind

    return cond_set_combo_satisfies_cond_ind

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
        is_conditionally_independent_func=sci_is_independent,
        indegree=8
    ):
        self.var_names = var_names
        self.data = data
        self.is_conditionally_independent_func = is_conditionally_independent_func
        self.indegree = indegree

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

                    value: list(tuple).
                        The conditioning sets that make X and Y conditionally
                        independent.
        """
        undirected_edges = []
        cond_sets_satisfying_cond_indep = {}

        for var_name_1, var_name_2 in combinations(self.var_names, 2):
            possible_conditioning_set_vars = list(
                set(self.var_names) - set([var_name_1, var_name_2])
            )

            cond_sets = conditioning_sets_satisfying_conditional_independence(
                data=self.data,
                var_name_1=var_name_1,
                var_name_2=var_name_2,
                is_conditionally_independent_func=self.is_conditionally_independent_func,
                possible_conditioning_set_vars=possible_conditioning_set_vars,
                indegree=self.indegree
            )

            if len(cond_sets) == 0:
                undirected_edges.append(set((var_name_1, var_name_2)))
            else:
                cond_sets_satisfying_cond_indep[
                    var_name_1 + ' _||_ ' + var_name_2
                ] = cond_sets

        marked_pattern = MarkedPatternGraph(
            nodes=self.var_names,
            undirected_edges=undirected_edges
        )

        return marked_pattern, cond_sets_satisfying_cond_indep

    def _is_independent(self, var_name_1, var_name_2):
        """
            Goes through possible conditioning sets, including the empty set.
        """

        # TODO: What about the case where the number of variables is 2?
        # TODO: set in-degree (i.e. the number of possible parents that one can
        # have)

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

# class DirectCausesOfMissingnessFinder(object):
    # def __init__(
        # self,
        # data,
        # marked_pattern_graph,
        # missingness_prefix='missingness_indicator_'
    # ):
        # self.data = data
        # self.marked_pattern_graph = marked_pattern_graph
#
    # def find(self):
        # self.add_missing_vars_to_marked_pattern_graph()
        # test
#
        # for col_with_missingness in self._cols_with_missingness():
            # for var in data.columns:
                # if missingness_prefix + var != col_with_missingness:
#
#
        # return self.marked_pattern_graph
#
    # def _add_missing_vars_to_marked_pattern_graph():
        # prefixed_missing_vars = \
            # [self.missingness_prefix + n for n in self._cols_with_missingness()]
#
        # self.marked_pattern_graph.add_nodes(prefixed_missing_vars)
#
    # def _cols_with_missingness():
        # if self.cols_with_missingness:
            # return self.cols_with_missingness
#
        # cols_missing_count = self.data.isnull().sum()
        # self.cols_with_missingness = \
            # cols_missing_count[cols_missing_count == 1].index.values
#
        # return self.cols_with_missingness
#
