from constraint_based.ci_tests.bmd_is_independent import bmd_is_independent
from constraint_based.misc import conditioning_sets_satisfying_conditional_independence, setup_logging
from itertools import combinations

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

            missingness_prefix: str. Defaults to "MI_"
                This is the string that gets prefixed to a column that has
                missingness.

            is_conditionally_independent_func: function.
                Defaults to bmd_is_independent.

                Takes the following as parameters:
                   data: pd.DataFrame
                   vars_1: list[str]
                   vars_2: list[str]
                   conditioning_set: list[str]
    """
    def __init__(
        self,
        data,
        graph,
        missingness_indicator_prefix='MI_',
        is_conditionally_independent_func=bmd_is_independent,
    ):
        self.data = data.merge(
            data.isnull().add_prefix(missingness_indicator_prefix),
            left_index=True,
            right_index=True
        )

        self.orig_data_cols = self.data.columns
        self.missingness_indicator_prefix = missingness_indicator_prefix
        self.is_conditionally_independent_func = is_conditionally_independent_func
        self.graph = graph

    def find(self):
        """
            If applicable, returns a list of marked arrows. A marked arrow is a
            tuple with two items. The first one is the from node and the last
            one is the to node.
        """

        marked_arrows = []

        logging = setup_logging()

        for col_with_missingness in self._cols_with_missingness():
            missingness_col_name = self.missingness_indicator_prefix + col_with_missingness

            logging.info('Finding direct parents of {}...'.format(missingness_col_name))

            for potential_parent in self._orig_vars():
                # Assumption: no self-masking (i.e. A doesn't cause MI_A)
                if potential_parent != col_with_missingness:
                    neighbors = self.graph.get_neighbors(potential_parent)

                    if col_with_missingness in neighbors:
                        col_with_miss_neighbors = self.graph.get_neighbors(col_with_missingness)

                        potential_parent_neighbors = neighbors.union(col_with_miss_neighbors) - set({col_with_missingness, potential_parent})
                    else:
                        potential_parent_neighbors = neighbors

                    depth = 0
                    independent = False

                    while depth <= len(potential_parent_neighbors):
                        if independent:
                            break

                        for combo in combinations(potential_parent_neighbors, depth):
                            if self.is_conditionally_independent_func(
                                self.data,
                                vars_1=[potential_parent],
                                vars_2=[missingness_col_name],
                                conditioning_set=list(combo)
                            ):
                                independent = True
                                break
                        depth += 1

                    if not independent:
                        logging.info('Found direct parents of {}: {}'.format(missingness_col_name, potential_parent))

                        marked_arrows.append(
                            (potential_parent, missingness_col_name)
                        )

        return marked_arrows

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

