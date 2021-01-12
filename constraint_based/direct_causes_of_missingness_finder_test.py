import numpy as np
import pytest
from .skeleton_finder import SkeletonFinder
from .direct_causes_of_missingness_finder import DirectCausesOfMissingnessFinder

def test_2_multinom_RVs_MCAR(
    df_2_multinomial_indep_RVs
):
    size = 1000
    df = df_2_multinomial_indep_RVs(size=size)
    missingness_of_x = np.random.binomial(n=1, p=0.3, size=size)
    missingness_indices = np.where(missingness_of_x == 1)

    df.at[missingness_indices[0], 'x'] = np.nan

    skeleton_finder = SkeletonFinder(
        data=df,
        var_names=df.columns,
    )

    graph, cond_sets_satisfying_cond_indep = \
        skeleton_finder.find()

    direct_causes_of_missingness_finder = DirectCausesOfMissingnessFinder(
        data=df
    )

    marked_arrows = direct_causes_of_missingness_finder.find()

    assert set(graph.nodes).intersection(set(['x', 'y', 'MI_x']))
    assert graph.undirected_edges == []

def test_2_multinom_RVs_MAR(
    df_X_Y_indep_Y_causes_MI_X
):

    size = 1000

    df = df_X_Y_indep_Y_causes_MI_X(size=size)

    skeleton_finder = SkeletonFinder(
        data=df,
        var_names=df.columns,
    )

    graph, cond_sets_satisfying_cond_indep = \
        skeleton_finder.find()

    direct_causes_of_missingness_finder = DirectCausesOfMissingnessFinder(
        data=df
    )

    marked_arrows = direct_causes_of_missingness_finder.find()

    assert set(graph.nodes).intersection(set(['x', 'y', 'MI_x']))
    assert graph.undirected_edges == []
    assert marked_arrows == [('y', 'MI_x')]

def test_3_multinom_RVs_MAR(
    df_Z_causes_X_Y_and_X_Z_causes_MI_Y
):
    size = 1000

    df = df_Z_causes_X_Y_and_X_Z_causes_MI_Y(size=size)

    skeleton_finder = SkeletonFinder(
        data=df,
        var_names=df.columns,
    )

    graph, cond_sets_satisfying_cond_indep = \
        skeleton_finder.find()

    direct_causes_of_missingness_finder = DirectCausesOfMissingnessFinder(
        data=df
    )

    marked_arrows = direct_causes_of_missingness_finder.find()
    assert set(graph.nodes).intersection(set(['x', 'y', 'MI_x']))
    assert graph.undirected_edges == [set(('x', 'z')), set(('z', 'y'))]
    assert set(marked_arrows) == set([('z', 'MI_y'), ('x', 'MI_y')])

def test_long_chains_collider_bias_without_MI(
    df_long_chains_and_collider_without_MI
):
    size = 1000

    df = df_long_chains_and_collider_without_MI(size=size)
    skeleton_finder = SkeletonFinder(
        data=df,
        var_names=df.columns,
    )

    graph, cond_sets_satisfying_cond_indep = skeleton_finder.find()

    # we expect b-d in this intermediate stage. b-d is spurious, due to
    # collider bias.

    expected_undirected_edges = frozenset({
        frozenset({'a', 'b'}),
        frozenset({'d', 'e'}),
        frozenset({'b', 'c'}),
        frozenset({'c', 'd'}),
    })

    assert frozenset(graph.undirected_edges) == expected_undirected_edges

    direct_causes_of_missingness_finder = DirectCausesOfMissingnessFinder(
        data=df
    )

    marked_arrows = direct_causes_of_missingness_finder.find()

    expected_marked_arrows = frozenset({})

    assert frozenset(marked_arrows) == expected_marked_arrows

def test_long_chains_collider_bias_with_MI(
    df_long_chains_and_collider_with_MI
):
    size = 10000

    df = df_long_chains_and_collider_with_MI(size=size)
    skeleton_finder = SkeletonFinder(
        data=df,
        var_names=df.columns,
        only_find_one=True
    )

    graph, cond_sets_satisfying_cond_indep = skeleton_finder.find()

    # we expect b-d in this intermediate stage. b-d is spurious, due to
    # collider bias.

    expected_undirected_edges = frozenset({
        frozenset({'a', 'b'}),
        frozenset({'b', 'd'}),
        frozenset({'d', 'e'}),
        frozenset({'b', 'c'}),
        frozenset({'c', 'd'}),
    })

    assert frozenset(graph.undirected_edges) == expected_undirected_edges

    direct_causes_of_missingness_finder = DirectCausesOfMissingnessFinder(
        data=df
    )

    marked_arrows = direct_causes_of_missingness_finder.find()

    expected_marked_arrows = frozenset({
      ('c', 'MI_a')
    })

    assert frozenset(marked_arrows) == expected_marked_arrows
