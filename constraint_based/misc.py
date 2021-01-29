import numpy as np
from itertools import combinations

"""
    Misc

    A library of miscellaneous functions.

    - conditioning_sets_satisfying_conditional_independence
"""

def conditioning_sets_satisfying_conditional_independence(
    data,
    var_name_1,
    var_name_2,
    is_conditionally_independent_func,
    possible_conditioning_set_vars=[],
    indegree=8,
    only_find_one=False,
    data_correction=None,
    marked_pattern_graph=None
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

            is_conditionally_independent_func: function
                This is the function that tests for conditional independendence.

                Parameters:
                    data: pandas.DataFrame

                    vars_1: list[str]
                        A set of variables present in data.  Disjoint from vars_2 and
                        conditioning_set.

                    vars_2: list[str]
                        Another set of variables present in data. Disjoint from vars_1
                        and conditioning_set.

                    conditioning_set: list[str]. Defaults to empty list.
                        Disjoint from vars_1 and vars_2.

            data_correction: class. Defaults to None.
                If this exists, the class will be passed in the following:
                    data: pandas.DataFrame
                    var_names: list[str]
                    marked_pattern_graph: MarkedPatternGraph
                    missing_indicator_prefix: str, Defaults to 'MI_'
                The instance responds to "correct", which returns a
                pandas.DataFrame that would hopefully be more representative of
                the underlying distribution.

            possible_conditioning_set_vars: list.
                The list of variables that we could possibly condition on.
                Represented by Z in X _||_ Y | Z.

            indegree: int. >= 1
                The maximum number of edges a node can have. This limits the search

            only_find_one: bool. Defaults to False.
                If a conditioning set satisfying the conditional independence
                statement is found, then quickly return.

            marked_pattern_graph: MarkedPatternGraph. Defaults to None.
                The existing graph.


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
            if data_correction != None:
                _data = data_correction(
                    data=data,
                    var_names=set(cond_set_combo)\
                        .union(set({var_name_1, var_name_2})),
                    marked_pattern_graph=marked_pattern_graph).correct()
            else:
                _data = data

            if is_conditionally_independent_func(
                   data=_data,
                   vars_1=[var_name_1],
                   vars_2=[var_name_2],
                   conditioning_set=list(cond_set_combo),
               ):

               cond_set_combo_satisfies_cond_ind.append(set(cond_set_combo))

               if only_find_one:
                   return cond_set_combo_satisfies_cond_ind

    return cond_set_combo_satisfies_cond_ind

def key_for_pair(var_names):
    """
        Used for accessing the cond_sets_that_satisfy_cond_indep.
    """
    _var_names = list(var_names)
    _var_names.sort()
    return _var_names[0] + ' _||_ ' + _var_names[1]

