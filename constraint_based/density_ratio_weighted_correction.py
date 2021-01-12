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

        probas = self._numerator() / self._denominator()

        counts = probas / probas.sum() * self.data.shape[0]

        collection = []

        for index, count in counts.iterrows():
            for i in range(int(count['tmp_count'])):
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

