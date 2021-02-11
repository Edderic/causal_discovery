import pytest
import pandas as pd
import numpy as np
from pytest import approx
from graphs.marked_pattern_graph import MarkedPatternGraph
from constraint_based.density_ratio_weighted_correction import DensityRatioWeightedCorrection

def test_deterministic_cause_of_missingness():
    size = 1000
    x = np.random.binomial(n=1, p=0.6, size=size)
    y = np.random.binomial(n=1, p=0.3, size=size)
    z = np.random.binomial(n=1, p=0.3, size=size)

    missing = np.where(x == 1)[0]

    df = pd.DataFrame({
        'x': x,
        'y': y,
        'z': z,
    })

    df.at[missing, 'z'] = np.nan

    graph = MarkedPatternGraph(
        nodes=['x', 'y', 'z', 'MI_z'],
        marked_arrows=[('x', 'MI_z')]
    )

    corrector = DensityRatioWeightedCorrection(
        data=df,
        var_names=['x', 'y', 'z'],
        graph=graph
    ).correct()

    # no errors thrown
    assert 1

def test_missing_data_because_of_ses():
    size = 10000

    ses = np.random.binomial(n=1, p=0.3, size=size)

    b_1_given_ses_low = np.random.binomial(n=1, p=0.4, size=size)
    b_1_given_ses_high = np.random.binomial(n=1, p=0.9, size=size)

    missing_b_1_given_ses_low = np.random.binomial(n=1, p=0.5, size=size)
    missing_b_1_given_ses_high = np.random.binomial(n=1, p=0.1, size=size)

    b = ses * b_1_given_ses_high + (ses == 0) * b_1_given_ses_low

    missing = ses * missing_b_1_given_ses_high \
        + (ses == 0) * missing_b_1_given_ses_low

    # true mean
    assert b.mean() == approx(0.55, abs=0.015)

    # Those with lower SES are more likely to be missing.
    missing_index = np.where(missing == 1)[0]

    df_with_missing_data = pd.DataFrame({
        'ses': ses,
        'b': b
    })

    df_with_missing_data.loc[missing_index, 'b'] = np.nan

    # A naive analysis leads to an overestimate.
    assert df_with_missing_data['b'].mean() == approx(0.62, abs=0.015)

    graph = MarkedPatternGraph(
        nodes=['ses', 'b', 'MI_b'],
        marked_arrows=[('ses', 'MI_b')],
        undirected_edges=[('ses', 'b')]
    )

    corrector = DensityRatioWeightedCorrection(
        data=df_with_missing_data,
        var_names=['ses', 'b', 'MI_b'],
        graph=graph
    )

    # reweight data before running statistics on it
    reweighted_df = corrector.correct()

    # we're able to recover the true mean
    assert reweighted_df['b'].mean() == approx(0.55, abs=0.015)

def test_long_chains_collider_bias_with_MI(
    df_long_chains_and_collider_with_MI,
    df_long_chains_and_collider_without_MI
):

    size=10000
    var_names = ['a', 'b', 'c', 'd', 'e']

    graph = MarkedPatternGraph(
        nodes=var_names,
        marked_arrows=[('c', 'MI_b')],
        undirected_edges=[
            ('a', 'b'),
            ('b', 'c'),
            ('e', 'd'),
            ('d', 'c'),
            ('b', 'd'), # extraneous edge
        ]
    )

    df_no_missing = df_long_chains_and_collider_without_MI(size=size)
    df_no_missing['count'] = 0

    assert df_no_missing['b'].mean() == approx(0.175, abs=0.01)
    no_missing_counts =  (df_no_missing.groupby(['b', 'd']).count()  / df_no_missing.groupby('d').count())['count']

    # B & D are marginally independent
    assert no_missing_counts.xs([False, False], level=['b', 'd']).values[0] \
            == approx(1 - 0.175, abs=0.02)

    assert no_missing_counts.xs([False, True], level=['b', 'd']).values[0] \
            == approx(1 - 0.175, abs=0.02)

    assert no_missing_counts.xs([True, False], level=['b', 'd']).values[0] \
            == approx(0.175, abs=0.02)

    assert no_missing_counts.xs([True, True], level=['b', 'd']).values[0] \
            == approx(0.175, abs=0.02)

    corrected_df = DensityRatioWeightedCorrection(
        data=df_long_chains_and_collider_with_MI(size=size),
        var_names=['b', 'd'],
        graph=graph
    ).correct()

    corrected_df['count'] = 0

    corrected_df_counts =  (corrected_df.groupby(['b', 'd']).count()  / corrected_df.groupby('d').count())['count']

    # B & D are marginally independent
    assert corrected_df_counts.xs([0, False], level=['b', 'd']).values[0] \
            == approx(1 - 0.175, abs=0.02)

    assert corrected_df_counts.xs([0, True], level=['b', 'd']).values[0] \
            == approx(1 - 0.175, abs=0.02)

    assert corrected_df_counts.xs([1, False], level=['b', 'd']).values[0] \
            == approx(0.175, abs=0.02)

    assert corrected_df_counts.xs([1, True], level=['b', 'd']).values[0] \
            == approx(0.175, abs=0.02)
