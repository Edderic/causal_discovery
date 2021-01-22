import numpy as np
import pandas as pd

def bmd_is_independent(
    data,
    vars_1=[],
    vars_2=[],
    conditioning_set=[],
    threshold=0.99
):
    var_1 = vars_1[0]
    var_2 = vars_2[0]
    _data = data.copy()
    _data['tmp_count'] = 0

    classes_for_var_1 = _data\
        .groupby(var_1).count().index

    classes_for_var_2 = _data\
        .groupby(var_2).count().index

    if conditioning_set == []:
        p1 = posterior(
            _data,
            variable=var_1
        )

        cond_set_var_2 = {}

        for var_2_val in classes_for_var_2:
            cond_set_var_2[var_2] = var_2_val
            p2 = posterior(
                _data,
                variable=var_1,
                conditioning_set=cond_set_var_2
            )

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

            p1 = posterior(
                _data,
                variable=var_1,
                conditioning_set=cond_set_p1
            )

            for var_2_val in classes_for_var_2:
                # make a copy
                cond_set_p2 = dict(cond_set_p1)
                cond_set_p2[var_2] = var_2_val

                p2 = posterior(
                    _data,
                    variable=var_1,
                    conditioning_set=cond_set_p2

                )

                if is_dependent(p1, p2, threshold):
                    return False

    return True

def posterior(data, variable, conditioning_set={}, size=1000):
    """
        Parameters:
            data: pd.DataFrame
            variable: name of the variable
            conditioning_set: dict
                key: name of the variable
                value: value of said variable

    """

    if conditioning_set == {}:
        _counts = data.groupby(variable).count()
        bdeu_prior = np.ones(_counts.shape[0]) / _counts.shape[0]

        return np.random.dirichlet(
            tuple((bdeu_prior + _counts['tmp_count'])),
            size=size
        )

    data_copy = data.copy()

    var_num_classes = data.groupby(variable).count().shape[0]

    for key, val in conditioning_set.items():
        data_copy = data_copy[data_copy[key] == val]

    counts = data_copy.groupby(variable).count()

    bdeu_prior = np.ones(var_num_classes) / var_num_classes

    data_count = make_data_counts_same_size_as_num_classes(
        expected_num_classes=var_num_classes,
        counts=counts
    )

    return np.random.dirichlet( tuple((bdeu_prior + data_count)), size=size)

def is_dependent(p1, p2, proba_threshold, subt_cutoff=0.01, div_cutoff=1.5):
    acceptable_1 = (p1 - p2 > subt_cutoff).sum(axis=0) / p1.shape[0] >= proba_threshold
    acceptable_2 = (p2 - p1 > subt_cutoff).sum(axis=0) / p1.shape[0] >= proba_threshold

    # import matplotlib.pyplot as plt

    # fig = plt.figure(figsize=(10,10))
    # ax_1 = fig.add_subplot(311)
    # ax_2 = fig.add_subplot(312)
    # ax_3 = fig.add_subplot(313)

    # pd.DataFrame(p1).plot.hist(bins=100,alpha=0.5, ax=ax_1, xlim=(0,1))
    # pd.DataFrame(p2).plot.hist(bins=100,alpha=0.5, ax=ax_2, xlim=(0,1))
    # pd.DataFrame(p1 - p2).plot.hist(bins=100,alpha=0.5, ax=ax_3)
    # plt.tight_layout()
#
    # fig.savefig('test2.png')

    # import pdb; pdb.set_trace()

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
