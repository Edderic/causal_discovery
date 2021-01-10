from ...information_theory import stochastic_complexity_score

def sci_is_independent(data, vars_1=[], vars_2=[], conditioning_set=[]):
    """
        This is an implementation of Stochastic Complexity Based
        Conditional Independence Based Test.

        Conditional Independence Testing, which answers the question "Is vars_1
        independent of vars_2 given the conditioning set?" is very important in
        causality, causal discovery, and statistics. One metric that measures
        this is the conditional mutual information (CMI) statistic. In the
        large sample limit, if the CMI is 0, then vars_1 and vars_2 are
        independent given the conditioning set.  However, CMI is usually never
        0 when sample sizes are small, even if the data was sampled from
        variables that are independent from each other. People usually select a
        threshold to decide whether the given set of variables are independent
        from each other. The threshold should depend on sample size and the
        number of possible values (i.e. the domain) for each variable.
        Stochastic Complexity Based Conditional Independence Based Test helps
        us automatically select a threshold for deciding which value of
        conditional mutual information is small enough to consider
        "independent" given sample size and complexity of the distributions.

        Note: This method does test-wise deletion. In other words, this only
        considers rows that have no NAs for the set of columns pertaining to
        this test. Those set of columns are the union of vars_1, vars_2, and
        the conditioning_set.

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

        Returns true if vars_1 is independent from vars_2 given conditioning
        set, false otherwise.

        Examples:
            Ex 1: Z causes X and Y:

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
            >>> assert sci_is_independent(
            >>>     data=df_2_multinomial_indep_RVs,
            >>>     variables=['x', 'y'],
            >>>     conditioning_set=['z']
            >>> ) == True
    """

    testwise_deleted_data = \
        data[
            list(set(conditioning_set).union(set(vars_1)).union(set(vars_2)))
        ].dropna()
    sample_size = testwise_deleted_data.shape[0]

    score_1 = \
        stochastic_complexity_score(
            testwise_deleted_data=testwise_deleted_data,
            vars_1=vars_1,
            vars_2=vars_2,
            conditioning_set=conditioning_set,
            sample_size=sample_size
        )

    score_2 = \
        stochastic_complexity_score(
            testwise_deleted_data=testwise_deleted_data,
            vars_1=vars_2,
            vars_2=vars_1,
            conditioning_set=conditioning_set,
            sample_size=sample_size
        )

    score = min(score_1, score_2)

    return score <= 0

