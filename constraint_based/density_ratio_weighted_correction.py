import pandas as pd

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
        missingness_indicator_prefix='MI_'
    ):
        self.data = data.merge(
            data.isnull().add_prefix(missingness_indicator_prefix),
            left_index=True,
            right_index=True
        )

        self.all_cols = data.columns
        self.var_names = var_names
        self.marked_pattern_graph = marked_pattern_graph
        self.missingness_indicator_prefix = missingness_indicator_prefix

    def correct(self):
        """
            Returns: pandas.DataFrame
                a corrected DataFrame with var_names as columns without any
                missingness.
        """

        self.data['tmp_count'] = 0

        probas = (self._constant() * self._density_ratio()).dropna()

        counts = probas / probas.sum() * self.data.shape[0]

        collection = []

        for index, count in counts.iterrows():
            for i in range(int(count)):
                collection.append(index)

        df = pd.DataFrame(collection, columns=counts.index.names)

        return df

    def _constant(self):
        num_counts = self.data[self._missingness_indicators()].copy()
        num_counts['tmp_count'] = 0
        counts = num_counts.groupby(list(self._missingness_indicators())).count()
        running_vals = counts.xs(
            [False for i in range(len(self._missingness_indicators()))], level=list(self._missingness_indicators())
        )[['tmp_count']] / self.data.shape[0]

        for mi in self._missingness_indicators():
            cond_var_values = self._missingness_indicator_of_parents_of_missingness_indicator(mi)

            proba = self._proba(
                var_values=[mi],
                cond_var_values=cond_var_values
            )

            combo = list(set(cond_var_values).union(set({mi})))

            running_vals /= proba

        return running_vals

    def _density_ratio(self):
        var_values = list((set(self.data.columns) - set(self._missingness_indicators()) - set({'tmp_count'})))
        cond_var_values = self._missingness_indicators()

        running_probas = self._proba(
            var_values=var_values,
            cond_var_values=cond_var_values
        )

        for mi in self._missingness_indicators():
            parents_of_mi = self._find_parents_of_missingness_indicator(mi)

            if len(parents_of_mi) == 0:
                numerator = 1.0
                denominator = 1.0
            else:
                numerator = self._proba(
                    var_values=parents_of_mi,
                    cond_var_values=self._missingness_indicator_of_parents_of_missingness_indicator(mi)
                )

                denominator = self._proba(
                    var_values=parents_of_mi,
                    cond_var_values=list(set(self._missingness_indicator_of_parents_of_missingness_indicator(mi)).union(set({mi})))
                )

            # the denominator has more columns, so we're doing the
            # multiplication in a weird way to take advantage of this fact.
            running_probas = running_probas / (denominator / numerator)

        return running_probas

    def _missingness_indicator_of_parents_of_missingness_indicator(self, mi):
        return [self.missingness_indicator_prefix + p for p in self._find_parents_of_missingness_indicator(mi)]

    def _numerator(self):
        return self.data.groupby(self.all_cols).count() \
                / self.data.shape[0]

    def _denominator(self):
        prod = 1.0

        for index, mi in enumerate(self._missingness_indicators()):
            if index == 0:
                prod = self._numerator()

            missing_inds = set({})

            missing_inds = missing_inds.union(set({mi}))

            parents_of_missingness_indicator = self._find_parents_of_missingness_indicator(mi)
            mis_parents_of_mi = [self.missingness_indicator_prefix + p for p in parents_of_missingness_indicator]

            missing_inds = missing_inds.union(set(mis_parents_of_mi))

            proba = self._proba(
                var_values=[mi],
                cond_var_values=list(set(parents_of_missingness_indicator).union(set(mis_parents_of_mi)))
            )

            prod /= proba.xs([False for i in range(len(missing_inds))], level=list(missing_inds))[['tmp_count']]

        return prod

    def _missingness_indicators(self):
        if len(set(dir(self)).intersection(set(['missingness_indicators']))) > 0:
            return self.missingness_indicators

        self.missingness_indicators = self.data.columns[
            self.data.columns.str.contains(self.missingness_indicator_prefix)
        ]

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
        numerator = self.data.groupby( list(set(var_values).union(cond_var_values))).count()
        if len(cond_var_values) == 0:
            return numerator / self.data.shape[0]

        denominator = self.data.groupby(list(cond_var_values)).count()

        return (numerator / denominator)[['tmp_count']]

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

        return [self.missingness_indicator_prefix + parent for parent in parents]

    def _marked_arrows(self):
        return self.marked_pattern_graph.marked_arrows

