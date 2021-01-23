import pytest
from .removable_edges_finder import RemovableEdgesFinder
from ..graphs.marked_pattern_graph import MarkedPatternGraph
from .density_ratio_weighted_correction import DensityRatioWeightedCorrection

def test_cond_on_collider(df_X_and_Y_cause_Z_and_Z_cause_MI_X):
    df = df_X_and_Y_cause_Z_and_Z_cause_MI_X(size=1000)

    # extraneous edge x-y
    graph = MarkedPatternGraph(
        nodes=['x', 'y', 'z', 'MI_x'],
        undirected_edges=[set({'x', 'y'}), set({'x', 'z'}), set({'y', 'z'})],
        marked_arrows=[('z', 'MI_x')]
    )

    finder = RemovableEdgesFinder(
        data=df,
        marked_pattern_graph=graph,
        potentially_extraneous_edges=[set({'x', 'y'})],
        data_correction=DensityRatioWeightedCorrection,
    )

    removables = finder.find()

    assert removables == [set({'x', 'y'})]

def test_long_chains_and_collider_with_MI(df_long_chains_and_collider_with_MI):
    df = df_long_chains_and_collider_with_MI(size=1000, proba_noise=0.6)

    graph = MarkedPatternGraph(
        nodes=list(set(df.columns).union(set({'MI_b'}))),
        undirected_edges=set({
            frozenset({'b', 'a'}),
            frozenset({'d', 'e'}),
            frozenset({'d', 'c'}),
            frozenset({'b', 'c'}),
            frozenset({'d', 'b'})
        }),
        marked_arrows=[('c', 'MI_b')]
    )

    finder = RemovableEdgesFinder(
        data=df,
        marked_pattern_graph=graph,
        potentially_extraneous_edges=set({
            frozenset({'d', 'b'}),
            frozenset({'d', 'c'}),
            frozenset({'b', 'c'})
        }),
        data_correction=DensityRatioWeightedCorrection,
    )

    removables = finder.find()

    assert set(removables) == set({ frozenset({'b', 'd'}) })

def test_3_multinom_RVs_MAR(
    df_Z_causes_X_Y_and_X_Z_causes_MI_Y
):
    size = 1000

    df = df_Z_causes_X_Y_and_X_Z_causes_MI_Y(size=size)

    graph = MarkedPatternGraph(
        nodes=list(set(df.columns).union(set({'MI_y'}))),
        undirected_edges=set({
            frozenset({'x', 'y'}), # extraneous edge
            frozenset({'x', 'z'}),
            frozenset({'z', 'y'}),
        }),
        marked_arrows=[
            ('x', 'MI_y'),
            ('z', 'MI_y')
        ]
    )

    finder = RemovableEdgesFinder(
        data=df,
        marked_pattern_graph=graph,
        potentially_extraneous_edges=set({
            frozenset({'x', 'y'}),
        }),
        data_correction=DensityRatioWeightedCorrection,
    )

    removables = finder.find()

    assert set(removables) == set({ frozenset({'x', 'y'}) })

def test_chain_and_collider_with_MI(
    df_chain_and_collider_with_MI
):
    size = 10000

    df = df_chain_and_collider_with_MI(size=size)

    graph = MarkedPatternGraph(
        nodes=list(set(df.columns).union(set({'MI_y'}))),
        undirected_edges=set({
            frozenset({'a', 'c'}), # extraneous edge
            frozenset({'a', 'b'}),
            frozenset({'b', 'c'}),
            frozenset({'a', 'd'}),
            frozenset({'c', 'd'}),
        }),
        marked_arrows=[
            ('d', 'MI_a')
        ]
    )

    # we expect a-c in this intermediate stage. a-c is spurious, due to
    # collider bias.

    expected_undirected_edges = frozenset({
        frozenset({'a', 'c'}),
    })

    finder = RemovableEdgesFinder(
        data=df,
        marked_pattern_graph=graph,
        potentially_extraneous_edges=set({
            frozenset({'a', 'c'}),
        }),
        data_correction=DensityRatioWeightedCorrection,
    )

    removables = finder.find()

    assert set(removables) == set({ frozenset({'a', 'c'}) })
