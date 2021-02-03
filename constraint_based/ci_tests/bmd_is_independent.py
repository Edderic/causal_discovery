import numpy as np
import pandas as pd

def bmd_is_independent(
    data,
    vars_1=[],
    vars_2=[],
    conditioning_set=[],
    threshold=0.99
):
    """
        This is a Bayesian Multinomial Dirichlet independence test. We assume
        the data is multinomially distributed, while the priors are Dirichlet
        distributed. The Dirichlet distribution is conjugate with the
        multinomial distribution (i.e. a Dirichlet prior + Multinomial data =>
        Dirichlet posterior).

        For each strata of the conditioning set, we produce the posterior
        distribution of the first variable. Likewise, we produce the posterior
        distribution of the first variable, given the second variable and the
        conditioning set. We then see if the difference is big enough. If at
        least one difference is big enough, we consider the variables in vars_1
        and vars_2 as dependent, given the conditioning set. Otherwise, we
        consider them dependent.

        Parameters:
            data: pd.DataFrame
                A dataframe that has the variables in vars_1, vars_2,
                conditioning_set

            vars_1: list['str']
                We assume there's only one item.

            vars_2: list['str']
                We assume there's only one item.

            conditioning_set: list['str']

            threshold: float. Defaults to 0.99
                This corresponds to the cutoff for deciding dependence vs.
                independence. If one of the comparisons leads us to a higher
                value than the threshold, then we say that vars_1 and vars_2
                are dependent given the conditioning_set. Otherwise, we
                consider the relationship as independent.

        Returns: boolean
            If the difference between at least one posterior vs. another is
            greater than the threshold parameter, then we consider the two
            distributions "dependent" and return False. Otherwise we return
            True.

        Since this is Bayesian, we get the benefit of Bayesian nalyses:

        - easier interpretation of credible intervals (vs. Frequentist
          confidence intervals)

        - any sample size is valid for inference.

    """
    var_1 = vars_1[0]
    var_2 = vars_2[0]
    _data = data.copy()
    _data['tmp_count'] = 0

    classes_for_var_1 = _data\
        .groupby(var_1).count().index

    classes_for_var_2 = _data\
        .groupby(var_2).count().index

    if conditioning_set == []:
        p1, _ = posterior(
            _data,
            variable=var_1
        )

        cond_set_var_2 = {}

        for var_2_val in classes_for_var_2:
            cond_set_var_2[var_2] = var_2_val
            p2, num_rows_2 = posterior(
                _data,
                variable=var_1,
                conditioning_set=cond_set_var_2
            )

            if num_rows_2 == 0:
                continue

            if is_dependent(p1, p2, threshold):
                return False

    else:
        classes_for_cond_set = _data\
            .groupby(conditioning_set).count().index
        cond_set_names = classes_for_cond_set.names

        for cond_set_val in classes_for_cond_set:
            cond_set_p1 = {}

            if len(cond_set_names) == 1:
                cond_set_p1[cond_set_names[0]] = cond_set_val
            else:
                for cond_set_name, cond_val in zip(cond_set_names, cond_set_val):
                    cond_set_p1[cond_set_name] = cond_val

            p1, num_rows_1 = posterior(
                _data,
                variable=var_1,
                conditioning_set=cond_set_p1
            )

            if num_rows_1 == 0:
                continue

            for var_2_val in classes_for_var_2:
                # make a copy
                cond_set_p2 = dict(cond_set_p1)
                cond_set_p2[var_2] = var_2_val

                p2, num_rows_2 = posterior(
                    _data,
                    variable=var_1,
                    conditioning_set=cond_set_p2
                )

                if num_rows_2 == 0:
                    continue

                if is_dependent(p1, p2, threshold):
                    return False

    return True

def posterior(data, variable, conditioning_set={}, size=1000):
    """
        Samples the Dirichlet posterior distribution.

        Parameters:
            data: pd.DataFrame
            variable: name of the variable
            conditioning_set: dict
                key: name of the variable
                value: value of said variable
            size: int. Defaults to 1,000
    """

    if conditioning_set == {}:
        _counts = data.groupby(variable).count()
        bdeu_prior = np.ones(_counts.shape[0]) / _counts.shape[0]

        num_rows = _counts.shape[0]

        return np.random.dirichlet(
            tuple((bdeu_prior + _counts['tmp_count'])),
            size=size
        ), num_rows

    data_copy = data.copy()

    var_num_classes = data.groupby(variable).count().shape[0]

    for key, val in conditioning_set.items():
        data_copy = data_copy[data_copy[key] == val]

    counts = data_copy.groupby(variable).count()
    num_rows = counts.shape[0]

    bdeu_prior = np.ones(var_num_classes) / var_num_classes

    data_count = make_data_counts_same_size_as_num_classes(
        expected_num_classes=var_num_classes,
        counts=counts
    )

    return np.random.dirichlet( tuple((bdeu_prior + data_count)), size=size), num_rows

def is_dependent(p1, p2, proba_threshold, subt_cutoff=0):
    acceptable_1 = (p1 - p2 > subt_cutoff).sum(axis=0) / p1.shape[0] >= proba_threshold
    acceptable_2 = (p2 - p1 > subt_cutoff).sum(axis=0) / p1.shape[0] >= proba_threshold

    return (acceptable_1 + acceptable_2).sum() > 0

def make_data_counts_same_size_as_num_classes(expected_num_classes, counts):
    """
        When the number of classes in the counts is less than the expected
        num_classes, add to it.
    """
    data_count = []

    if counts['tmp_count'].shape[0] != expected_num_classes:
        data_count = list(counts['tmp_count'].values)

        while len(data_count) < expected_num_classes:
            data_count.append(0)

        return np.array(data_count)
    else:
        return counts['tmp_count']


    # for index, data_count in _counts.iterrows():

    # we get the groupby counts. Each row has coordinates that exist in the data
    # However, there might be coordinates that don't exist in the data. We are
    # also interested in those especially for the variable of interest.

    # To get a posterior count, we need the number of possible values per
    # variable in the data set. E.g. If there's a variable called
    # "dominant-hand", the possible values are left-hand and right-hand, so the
    # number of possible values for this variable is 2.

    # To determine the Bdeu prior, we would need to know how many variables are
    # in the conditioning set, and what the number of possible values are for
    # each of those variables. Let's say in our data set, we have the variables
    #   "dominant-hand"
    #     - left-handed, right-handed,
    #   "gender"
    #     - trans, cis
    #   "religion"
    #     - catholic, protestant, muslim, hindu
    #
    # Let's say we're interested in "dominant-hand | gender, religion". There
    # should be 2 Dirichlet priors for each combination of gender and
    # religion. There are 2 x 4 = 8 combinations of gender and religion, so we
    # make, 8 x 2 = 16 Dirichlet priors

    # _counts = _data.groupby(list(set(variables).union(conditioning_set))).count()
#
    # posterior_samples = []
#
    # for index, data_count in _counts.iterrows():
        # bdeu_prior = np.ones(num_classes_variable) / num_classes_variable
#
        # posterior_samples.append(np.random.dirichlet(bdeu_prior + data_count, size=size))
#
    # _counts['posterior_samples'] = posterior_samples
#
    # return _counts


class Posterior(object):
    def __init__(self, indices, posteriors):
        pass
