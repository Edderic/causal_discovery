"""
    Information Theory
    ------------------

    Provides Information Theory helpers.

    Available functions
    -------------------

    - entropy
    - conditional_entropy
    - conditional_mutual_information
    - multinomial_normalizing_sum
    - regret
    - sci_is_independent

"""

import numpy as np

def entropy(data, variables=[]):
    """
        Computes Shannon entropy.

        Parameters:
            data : pandas.DataFrame
                A dataframe where variables are columns.

            variables: list[str]
                A list of variable names to include in the entropy calculation.

        Examples:
            Say that X is multinomially distributed with 4 classes, and they are
            uniformly distributed. The Shannon entropy is:

            - 4 * (0.25 * np.log2(0.25)) = -1 * np.log2(0.25)
                                         = -1 * -2
                                         = 2

            >>> from pytest import approx
            >>> size = 10000
            >>> multinomials = np.random.multinomial(
            >>>         n=1,
            >>>         pvals=[0.25,
            >>>             0.25,
            >>>             0.25,
            >>>             0.25],
            >>>         size=size)

            >>> x = multinomials[:, 0] \
            >>>         + multinomials[:, 1] * 2 \
            >>>         + multinomials[:, 2] * 3 \
            >>>         + multinomials[:, 3] * 4

            >>> df = pd.DataFrame({'x': x})
            >>> calc = EntropyCalculator(data=df, variables=['x'])

            >>> assert calc.calculate() == approx(2, abs=0.01)
    """

    data = data.copy()
    count_col_name = 'tmp. count'
    data[count_col_name] = 0
    total_count = data.shape[0]

    assert len(variables) > 0

    variable_counts = data.groupby(list(variables)).count()
    probas = variable_counts / total_count

    return -(probas * np.log2(probas)).sum()[count_col_name]


def conditional_entropy(data, conditioning_set=[], variables=[]):
    """
        Computes H(X | Y) = H(X,Y) - H(Y) where Y is the conditioning_set and X
        and Y are the variables.

        Parameters:
            data : pandas.DataFrame
                A dataframe where variables are columns.

            variables: list[str]
                A list of variable names.

            conditioning_set: list[str]. Defaults to empty list.
                A list of variable names that are being conditioned on. If the
                conditionals is not empty, then we'll be computing conditional
                entropy.

                If conditioning_set is an empty list, this function returns the
                entropy for the set of variables (i.e. instead of computing
                H(X|Y), it'll return H(X), the entropy of X).

        Examples:
            Say there's a variable X and Y and they are independent. X and Y are
            multinomial variables with 4 possible values:

            >>> from pytest import approx
            >>> size = 10000
            >>> x = np.random.multinomial(
            >>>         n=1,
            >>>         pvals=[0.25,
            >>>             0.25,
            >>>             0.25,
            >>>             0.25],
            >>>         size=size)
            >>>
            >>> x = x[:, 0] \
            >>>         + x[:, 1] * 2 \
            >>>         + x[:, 2] * 3 \
            >>>         + x[:, 3] * 4
            >>>
            >>> y = np.random.multinomial(
            >>>         n=1,
            >>>         pvals=[0.25,
            >>>             0.25,
            >>>             0.25,
            >>>             0.25],
            >>>         size=size)
            >>>
            >>> y = y[:, 0] \
            >>>         + y[:, 1] * 2 \
            >>>         + y[:, 2] * 3 \
            >>>         + y[:, 3] * 4
            >>>
            >>> df_2_multinomial_indep_RVs = pd.DataFrame({'x': x, 'y': y})


            # Entropy of X, without conditioning on anything, is 2. Conditional
            # entropy of X given Y is still 2. In other words, knowing about Y
            # doesn't change the entropy (i.e. the uncertainty) on X.
            # Therefore X and Y are independent.

            >>> assert conditional_entropy(
            >>>     data=df_2_multinomial_indep_RVs,
            >>>     variables=['x', 'y'],
            >>>     conditioning_set=['y']
            >>> ) == approx(2, abs=0.01)
    """
    assert len(set(variables)) > 0

    if len(conditioning_set) == 0:
        return entropy(data=data, variables=variables)

    vars_and_conditioning_set = \
        list(set(variables).union(set(conditioning_set)))

    return entropy(
               data=data,
               variables=vars_and_conditioning_set
           ) - entropy(
               data=data,
               variables=conditioning_set
           )

def conditional_mutual_information(data, vars_1, vars_2, conditioning_set=[]):
    """
        Computes I(X;Y|Z) = H(X|Z) - H(X|Y,Z). Essentially, this tells us
        whether or not Y tells us something about X, after we've known about
        Z. In the large sample limit, if Y is independent of X given Z, then
        conditional mutual information I(X;Y|Z) is 0. However, with finite
        samples, even if in the true generating process, X and Y are
        independent given Z, it's very possible that the conditional mutual
        information is greater than 0.

        Parameters:
            data: pandas.DataFrame
            vars_1: list[str]
                Represents X in I(X;Y|Z).
            vars_2: list[str]
                Represents Y in I(X;Y|Z).
            conditioning_set: list[str]. Defaults to empty list.
                Represents Z in I(X;Y|Z).

                If conditioning_set is empty, this computes mutual information:
                I(X;Y) = H(X) - H(X|Y).

        Examples:
            Ex 1: Say there's a variable X and Y and they are independent. X
            and Y are multinomial variables with 4 possible values:

            >>> from pytest import approx
            >>> size = 10000
            >>> x = np.random.multinomial(
            >>>         n=1,
            >>>         pvals=[0.25,
            >>>             0.25,
            >>>             0.25,
            >>>             0.25],
            >>>         size=size)
            >>>
            >>> x = x[:, 0] \
            >>>         + x[:, 1] * 2 \
            >>>         + x[:, 2] * 3 \
            >>>         + x[:, 3] * 4
            >>>
            >>> y = np.random.multinomial(
            >>>         n=1,
            >>>         pvals=[0.25,
            >>>             0.25,
            >>>             0.25,
            >>>             0.25],
            >>>         size=size)
            >>>
            >>> y = y[:, 0] \
            >>>         + y[:, 1] * 2 \
            >>>         + y[:, 2] * 3 \
            >>>         + y[:, 3] * 4
            >>>
            >>> df_2_multinomial_indep_RVs = pd.DataFrame({'x': x, 'y': y})
            >>>
            >>> assert conditional_mutual_entropy(
            >>>     data=df_2_multinomial_indep_RVs,
            >>>     variables=['x', 'y'],
            >>>     conditioning_set=[]
            >>> ) == approx(0, abs=0.01)

            Ex 2: Z causes X and Y:

            >>> from pytest import approx
            >>> size = 10000
            >>> z = np.random.multinomial(
            >>>         n=1,
            >>>         pvals=[0.25,
            >>>             0.25,
            >>>             0.25,
            >>>             0.25],
            >>>         size=size)
            >>>
            >>> z = z[:, 0] \
            >>>         + z[:, 1] * 2 \
            >>>         + z[:, 2] * 3 \
            >>>         + z[:, 3] * 4
            >>> y = (z == 2)
            >>> x = (z == 1)
            >>>
            >>> df_2_multinomial_indep_RVs = pd.DataFrame(
            >>>     {'x': x, 'y': y, 'z': z}
            >>> )
            >>>
            >>> assert conditional_mutual_entropy(
            >>>     data=df_2_multinomial_indep_RVs,
            >>>     variables=['x', 'y'],
            >>>     conditioning_set=['z']
            >>> ) == approx(0, abs=0.01)
    """
    return conditional_entropy(
        data=data,
        variables=vars_1,
        conditioning_set=conditioning_set
    ) - conditional_entropy(
        data=data,
        variables=vars_1,
        conditioning_set=list(set(conditioning_set).union(vars_2))
    )


def multinomial_normalizing_sum(num_classes, sample_size):
    """
        Implementation of a fast algorithm for computing the multinomial
        normalizing sum. This is used in the Normalized Maximum Likelihood
        (NML) distribution, which can be used for measuring Stochastic
        Complexity. The latter is a model selection technique and can be used
        for conditional independence testing.

        See "Computing the Multinomial Stochastic Complexity in Sub-Linear
        Time" by Mononen and Myllmäki.

        Parameters:
            num_classes: int
                Number of classes
            sample_size: int
                Sample size
    """
    d = 10
    b = 1
    summation = 1

    bound = int(np.ceil(2 + np.sqrt(-2 * sample_size * np.log2(2 * 10**(-d) - 100**(-d)))))

    for k in range(1, bound):
        b = (num_classes - k + 1) / sample_size * b
        summation = summation + b

    old_sum = 1

    for j in range(3,num_classes):
        new_sum = summation + (sample_size * old_sum) / (j-2)
        old_sum = summation
        summation = new_sum

    return summation

def regret(data, variables=[], conditioning_set=[]):
    assert len(variables) > 0

    data_copy = data.copy()
    data_copy['tmp counts'] = 0

    counts = data_copy.groupby(variables).count()

    if len(conditioning_set) == 0:
        return np.log2(
            multinomial_normalizing_sum(
                num_classes=counts.shape[0],
                sample_size=data.shape[0]
            )
        )

    conditioning_set_counts = \
        data_copy.\
        groupby(conditioning_set).\
        count()[['tmp counts']]

    summation = 0

    for indices, count in conditioning_set_counts.iterrows():
        summation += np.log2(
            multinomial_normalizing_sum(
                num_classes=counts.shape[0],
                sample_size=count
            )
        )

    return summation['tmp counts']

def stochastic_complexity_score(
    testwise_deleted_data,
    vars_1,
    vars_2,
    conditioning_set,
    sample_size
):
    return sample_size \
        * conditional_mutual_information(
            data=testwise_deleted_data,
            vars_1=vars_1,
            vars_2=vars_2,
            conditioning_set=conditioning_set
        ) \
        + regret(
            data=testwise_deleted_data,
            variables=vars_1,
            conditioning_set=conditioning_set
        ) \
        - regret(
            data=testwise_deleted_data,
            variables=vars_1,
            conditioning_set=list(set(conditioning_set).union(vars_2))
        )
