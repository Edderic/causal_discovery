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
import re

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

                    value: list(sets(str)).
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

class DirectCausesOfMissingnessFinder(object):
    """
        Finds the direct causes of missingness.

        Assumption: All causes of missingness are observed.

        Assumption: Missingness indicators cannot cause other variables.

        Assumption: No self-masking type of missingness. An example of
        self-masking is rich people being less likely to disclose their
        incomes.

        Assumption: Faithful observability. ???

        Combining the two assumptions: If a missing indicator is not found to
        be conditionally independent with a variable, then the latter must be a
        parent of the former.

        Paramters:
            data: pd.DataFrame

            marked_pattern_graph: MarkedPatternGraph

            missingness_prefix: str. Defaults to "MI_"
                This is the string that gets prefixed to a column that has
                missingness.

            is_conditionally_independent_func: function.
                Defaults to sci_is_independent.

                Takes the following as parameters:
                   data: pd.DataFrame
                   vars_1: list[str]
                   vars_2: list[str]
                   conditioning_set: list[str]

            indegree: int. Defaults to 8.
                The number of nodes that can be associated to another node.
    """
    def __init__(
        self,
        data,
        marked_pattern_graph,
        missingness_prefix='MI_',
        is_conditionally_independent_func=sci_is_independent,
        indegree=8
    ):
        self.data = data.copy()
        self.orig_data_cols = self.data.columns
        self.marked_pattern_graph = marked_pattern_graph
        self.missingness_prefix = missingness_prefix
        self.is_conditionally_independent_func = is_conditionally_independent_func
        self.indegree = indegree

    def find(self):
        """
            Returns a MarkedPatternGraph with direct edges to missingness
            indicators, if applicable.
        """

        self._add_missing_vars_to_marked_pattern_graph()

        for col_with_missingness in self._cols_with_missingness():
            missingness_col_name = self.missingness_prefix + col_with_missingness
            self.data[missingness_col_name] = self.data[col_with_missingness].isnull()

            for potential_parent in self.orig_data_cols:
                # Assumption: no self-masking (i.e. A doesn't cause MI_A)
                if potential_parent != col_with_missingness:
                    cond_sets = conditioning_sets_satisfying_conditional_independence(
                        data=self.data,
                        var_name_1=missingness_col_name,
                        var_name_2=potential_parent,
                        is_conditionally_independent_func=self.is_conditionally_independent_func,
                        possible_conditioning_set_vars=list(
                            set(self.data.columns)\
                            -set([
                                    col_with_missingness,
                                    missingness_col_name,
                                    potential_parent
                            ])
                        ),
                        indegree=self.indegree,
                        only_find_one=True
                    )

                    if len(cond_sets) == 0:
                        self.marked_pattern_graph.add_marked_arrows(
                            [(potential_parent, missingness_col_name)]
                        )

        return self.marked_pattern_graph

    def _add_missing_vars_to_marked_pattern_graph(self):
        prefixed_missing_vars = \
            [self.missingness_prefix + n for n in self._cols_with_missingness()]

        self.marked_pattern_graph.add_nodes(prefixed_missing_vars)

    def _cols_with_missingness(self):
        if len(set(dir(self)).intersection(set(['cols_with_missingness']))) > 0:
            return self.cols_with_missingness

        cols_missing_count = self.data.isnull().sum()
        self.cols_with_missingness = \
            cols_missing_count[cols_missing_count > 0].index.values

        return self.cols_with_missingness

class PotentialExtraneousEdgesFinder():
    """
        Find edges that might be extraneous. We do this because conditional
        independence testing X _||_ Y | Z on test-wise deleted data (i.e.
        excluding rows where X,Y and set Z are missing) could lead to
        extraneous edges. If missingness indicators are children / descendants
        of X & Y, it's possible to find that X is dependent of Y given Z even
        though in the real data-generating process, X is independent of Y given
        Z.

        Potential extraneous edges are those whose pair of variables belong to
        edges that have a common variable other than those two.

        Let's say we know the true data-generating process. Let's say we have
        variables X,Y,Z,A and that X and Y are marginally independent (e.g.
        conditionally independent given the empty set).

        E.g. X->Z, Z->Y,      # X causes Z, and Z causes Y.
             X->A, Y->A,      # A is a collider.
             A->Ry            # Ry is a descendant of a collider.

        Once we have a data set produced by the above, running SkeletonFinder
        on it, we might find a graph like so:

        E.g. X-Z, Y-Z         # X and Y are connected to Z
             X-A, Y-A,        # W might be a collider (e.g. X->W, Y->W is
                              #   possible).
             A->Ry            # Ry is a descendant of a collider.
             X-Y              # Conditioning on a collider / descendant of a
                              #   collider leads to spurious edges.


        Note: This only applies when missingness is MAR or MNAR. If data given
        is MCAR, then there shouldn't be extraneous edges and exit early.
    """
    def __init__(
        self,
        data,
        marked_pattern_graph,
        missingness_indicator_prefix = "MI_"
    ):
        self.marked_pattern_graph = marked_pattern_graph
        self.missingness_indicator_prefix = missingness_indicator_prefix

    def find(self):
        pass
        # TODO: check for MCAR case. If so, exit early
        # for node_1, node_2 in tuple(self.marked_pattern_graph.undirected_edges):
            # find_


    def _not_a_missing_indicator(node):
        return re.search(self.missingness_indicator_prefix, node) != None
