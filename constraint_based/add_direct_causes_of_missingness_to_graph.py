from .ci_tests.sci_is_independent import sci_is_independent
from .misc import conditioning_sets_satisfying_conditional_independence

class AddDirectCausesOfMissingnessToGraph(object):
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

