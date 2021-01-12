import pytest
from .removable_edges_finder import RemovableEdgesFinder
from ..graphs.marked_pattern_graph import MarkedPatternGraph
from .density_ratio_weighted_correction import DensityRatioWeightedCorrection

def test_cond_on_collider(df_X_and_Y_cause_Z_and_Z_cause_MI_X):
    df = df_X_and_Y_cause_Z_and_Z_cause_MI_X()

    # extraneous edge x-y
    graph = MarkedPatternGraph(
        nodes=['x', 'y', 'z', 'MI_x'],
        undirected_edges=[set({'x', 'y'}), set({'x', 'z'}), set({'y', 'z'})],
        marked_arrows=[('z', 'MI_x')]
    )

    validator = RemovableEdgesFinder(
        data=df,
        marked_pattern_graph=graph,
        potentially_extraneous_edges=[set({'x', 'y'})],
        data_correction=DensityRatioWeightedCorrection,
    )

    removables = validator.find()

    assert removables == [set({'x', 'y'})]
