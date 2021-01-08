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
import pandas as pd
import re

def conditioning_sets_satisfying_conditional_independence(
    data,
    var_name_1,
    var_name_2,
    is_conditionally_independent_func,
    possible_conditioning_set_vars=[],
    indegree=8,
    only_find_one=False,
    adjusted_data_generator=None,
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

            adjusted_data_generator: function. Defaults to None.
                If this exists, the function will be used to generate data that
                is meant for validating potentially extraneous edges.

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
            if adjusted_data_generator != None:
                _data = adjusted_data_generator(
                    data=data,
                    var_names=set(cond_set_combo)\
                              .union(set({var_name_1, var_name_2})),
                    marked_pattern_graph=marked_pattern_graph
                )
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
        indegree=8,
        missing_indicator_prefix='MI_'
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
                indegree=self.indegree,
                only_find_one=False
            )

            if len(cond_sets) == 0:
                undirected_edges.append(frozenset((var_name_1, var_name_2)))
            else:
                cond_sets_satisfying_cond_indep[
                    var_name_1 + ' _||_ ' + var_name_2
                ] = cond_sets

        marked_pattern = MarkedPatternGraph(
            nodes=list(self.data.columns),
            undirected_edges=undirected_edges
        )

        return marked_pattern, cond_sets_satisfying_cond_indep, self.data

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
        missingness_indicator_prefix='MI_',
        is_conditionally_independent_func=sci_is_independent,
        indegree=8
    ):
        self.data = data.copy()
        self.orig_data_cols = self.data.columns
        self.marked_pattern_graph = marked_pattern_graph
        self.missingness_indicator_prefix = missingness_indicator_prefix
        self.is_conditionally_independent_func = is_conditionally_independent_func
        self.indegree = indegree

    def find(self):
        """
            Returns a MarkedPatternGraph with direct edges to missingness
            indicators, if applicable.
        """

        for col_with_missingness in self._cols_with_missingness():
            missingness_col_name = self.missingness_indicator_prefix + col_with_missingness

            for potential_parent in self._orig_vars():
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

    def _orig_vars(self):
        return self.data.columns[
            ~self.data.columns.str.contains(self.missingness_indicator_prefix)
        ]

    def _cols_with_missingness(self):
        self.data.columns.str.contains(self.missingness_indicator_prefix)

        if len(set(dir(self)).intersection(set(['cols_with_missingness']))) > 0:
            return self.cols_with_missingness

        cols_missing_count = self.data.isnull().sum()
        self.cols_with_missingness = \
            cols_missing_count[cols_missing_count > 0].index.values

        return self.cols_with_missingness

class PotentiallyExtraneousEdgesFinder(object):
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
             X-A, Y-A,        # A might be a collider (e.g. X->A, Y->A is
                              #   possible).
             A->Ry            # Ry is a descendant of a collider.
             X-Y              # Conditioning on a collider / descendant of a
                              #   collider could lead to spurious edges.


        Note: This only applies when missingness is MAR or MNAR. If data given
        is MCAR, then there shouldn't be extraneous edges and exit early.

        Parameters:
            data: pandas.DataFrame.
            marked_pattern_graph: MarkedPatternGraph
                responds to:
                    - marked_arrows
                        We assume that if MAR or MNAR, missingness indicators
                        would be a node in directed arrows. If not, then there
                        are no extraneous edges.

                        We assume that missingness indicators in nodes of
                        directed arrows have the same missingness indicator
                        prefix below.

            missingness_indicator_prefix: str. Defaults to "MI_".
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
        # missingness indicators are not descendants of other variables, so
        # this is MCAR.
        if len(self.marked_pattern_graph.marked_arrows) == 0:
            return [], self.marked_pattern_graph

        potentially_extraneous_edges = []

        for some_set in self._undirected_edges():
            node_1, node_2 = tuple(some_set)

            common_nodes = self._get_adjacent_nodes(node_1)\
                .intersection(self._get_adjacent_nodes(node_2))

            if len(common_nodes) > 0:
                potentially_extraneous_edges.append(some_set)

        return set(potentially_extraneous_edges)

    def _get_adjacent_nodes(self, node):
        new_set = set([])

        for edge in self._relevant_edges():
            some_set = set(edge)
            if some_set.intersection(set([node])) != set([]):
                new_set = new_set.union(some_set)

        return new_set - set([node])

    def _relevant_edges(self):
        return frozenset(self._undirected_edges()).union(frozenset(self._marked_arrows()))

    def _undirected_edges(self):
        return self.marked_pattern_graph.undirected_edges

    def _marked_arrows(self):
        return self.marked_pattern_graph.marked_arrows

class DensityRatioWeightedCorrection(object):
    """
        Takes in data with missingness, and produces a corrected data set
        without missingness. Makes use of R Factorization as listed in Mohan &
        Pearl (2020).

        Parameters:
            var_names: set
                The name of variables that are being considered to produce
            data: pandas.DataFrame
                A dataframe where var_names is a subset of the columns.



            missingness_indicator_parents: dict
                keys: missingness indicators (e.g. 'MI_x')
                values: set(str)
                    Parents of the missingness indicator (e.g. {'Y', 'Z'})
    """
    def __init__(
        self,
        data,
        var_names,
        marked_pattern_graph,
        missing_indicator_prefix='MI_'
    ):
        self.data = data.copy()
        self.var_names = var_names
        self.marked_pattern_graph = marked_pattern_graph
        self.missing_indicator_prefix = missing_indicator_prefix

    def correct(self):
        """
            Returns: pandas.DataFrame
                a corrected DataFrame with var_names as columns without any
                missingness.
        """

        self.data['tmp_count'] = 0

        # probas = self._fully_observed_relevant_data() \
            # * self._constant() \
            # * self._density_ratios()

        probas = self._numerator() / self._denominator()

        counts = probas / probas.sum() * self.data.shape[0]

        collection = []

        for index, count in counts.iterrows():
            for i in range(int(count)):
                collection.append(index)

        df = pd.DataFrame(collection, columns=counts.index.names)

        return df

    def _numerator(self):
        return self.data.dropna().groupby(self.marked_pattern_graph.nodes).count() \
                / self.data.shape[0]

    def _denominator(self):
        prod = 1.0

        missing_inds = set({})

        for mi in self._missingness_indicators():
            missing_inds = missing_inds.union(set({mi}))

            parents_of_missingness_indicator = self._find_parents_of_missingness_indicator(mi)
            mis_parents_of_mi = [self.missing_indicator_prefix + p for p in parents_of_missingness_indicator]

            missing_inds = missing_inds.union(set(mis_parents_of_mi))

            proba = self._proba(
                var_values=[mi],
                cond_var_values=list(set(parents_of_missingness_indicator).union(set(mis_parents_of_mi)))
            )

            prod *= proba

        return prod.xs([False for i in range(len(missing_inds))], level=list(missing_inds))[['tmp_count']]

    def _missingness_indicators(self):
        if len(set(dir(self)).intersection(set(['missingness_indicators']))) > 0:
            return self.missingness_indicators

        self.missingness_indicators = self.marked_pattern_graph.missingness_indicators()

        return self.missingness_indicators

    def _proba(self, var_values, cond_var_values):
        """
            Parameters:
                var_values: dict
                    key: str
                        name of a variable
                    value: The value of the variable.

                cond_var_values: dict
                    key: str
                        name of a variable in the conditioning set.
                    value: The value of the variable in the conditioning set.

        """
        numerator = self.data.groupby(
            list(set(var_values).union(cond_var_values))
        ).count()

        denominator = self.data.groupby(cond_var_values).count()

        return numerator / denominator

    def _find_parents_of_missingness_indicator(self, missingness_indicator):
        parents = []

        for from_node, to_node in self._marked_arrows():
            if to_node == missingness_indicator:
                parents.append(from_node)

        return parents

    def _mis_of_parents_of_mi(self, missingness_indicator):
        parents = self._find_parents_of_missingness_indicator(
            missingness_indicator
        )

        return [self.missing_indicator_prefix + parent for parent in parents]

    def _marked_arrows(self):
        return self.marked_pattern_graph.marked_arrows

class PotentiallyExtraneousEdgesValidator(object):
    def __init__(
        self,
        data,
        marked_pattern_graph,
        adjusted_data_generator,
        is_conditionally_independent_func=sci_is_independent,
        potentially_extraneous_edges=[],
    ):
        self.data = data
        self.potentially_extraneous_edges = potentially_extraneous_edges
        self.marked_pattern_graph = marked_pattern_graph
        self.adjusted_data_generator = adjusted_data_generator
        self.is_conditionally_independent_func = is_conditionally_independent_func

    def edges_to_remove(self):
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
                possible_conditioning_set_vars=[],
                indegree=8,
                only_find_one=True,
                adjusted_data_generator=self.adjusted_data_generator,
                marked_pattern_graph=self.marked_pattern_graph
            )

            if len(cond_sets) > 0:
                extraneous_edges.append(potentially_extraneous_edge)

        return extraneous_edges

def adjusted_data_generator(data, var_names, marked_pattern_graph):
    correction = DensityRatioWeightedCorrection(
        data=data,
        var_names=var_names,
        marked_pattern_graph=marked_pattern_graph
    )

    return correction.correct()

class MissingICStar(object):
    def __init__(
        self,
        data,
        missingness_indicator_prefix='MI_'
    ):
        self.data = data.merge(
            data.isnull().add_prefix(missingness_indicator_prefix),
            left_index=True,
            right_index=True
        )

        self.orig_columns = data.columns

        self.missingness_indicator_prefix = missingness_indicator_prefix

    def predict(self):
        skeleton_finder = SkeletonFinder(
            var_names=self.orig_columns,
            data=self.data
        )

        graph, cond_sets_satisfying_cond_indep = skeleton_finder.find()

        self._add_missing_vars_to_marked_pattern_graph(graph)

        DirectCausesOfMissingnessFinder(
            data=self.data,
            graph=graph,
        ).find()

        potentially_extraneous_edges_finder = PotentiallyExtraneousEdgesFinder(
            data=self.data,
            marked_pattern_graph=graph,
            missingness_indicator_prefix=self.missingness_indicator_prefix
        )

        potentially_extraneous_edges = potentially_extraneous_edges_finder.find()

        edges_to_remove = PotentiallyExtraneousEdgesValidator(
            data=self.data,
            adjusted_data_generator=adjusted_data_generator,
            potentially_extraneous_edges=potentially_extraneous_edges,
            marked_pattern_graph=graph
        ).edges_to_remove()
