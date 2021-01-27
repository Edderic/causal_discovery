import pytest
import pandas as pd
import numpy as np
from pytest import approx
from ..graphs.marked_pattern_graph import MarkedPatternGraph
from .density_ratio_weighted_correction import DensityRatioWeightedCorrection

def test_long_chains_and_collider_with_MI(df_long_chains_and_collider_with_MI):
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
    assert b.mean() == approx(0.55, abs=0.01)

    # Those with lower SES are more likely to be missing.
    missing_index = np.where(missing == 1)[0]

    df_with_missing_data = pd.DataFrame({
        'ses': ses,
        'b': b
    })

    df_with_missing_data.loc[missing_index, 'b'] = np.nan

    # A naive analysis leads to an overestimate.
    assert df_with_missing_data['b'].mean() == approx(0.62, abs=0.01)

    graph = MarkedPatternGraph(
        nodes=['ses', 'b', 'MI_b'],
        marked_arrows=[('ses', 'MI_b')],
        undirected_edges=[('ses', 'b')]
    )

    corrector = DensityRatioWeightedCorrection(
        data=df_with_missing_data,
        var_names=['ses', 'b', 'MI_b'],
        marked_pattern_graph=graph
    )

    reweighted_df = corrector.correct()

    # we're able to recover the true mean
    assert reweighted_df['b'].mean() == approx(0.55, abs=0.01)



