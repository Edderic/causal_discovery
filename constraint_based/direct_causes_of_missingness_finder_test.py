import numpy as np
import pytest
from constraint_based.skeleton_finder import SkeletonFinder
from constraint_based.direct_causes_of_missingness_finder import DirectCausesOfMissingnessFinder
from graphs.marked_pattern_graph import MarkedPatternGraph

def test_2_multinom_RVs_MCAR(
    df_2_multinomial_indep_RVs
):
    size = 2000
    df = df_2_multinomial_indep_RVs(size=size)

    missingness_of_x = np.random.binomial(n=1, p=0.3, size=size)
    missingness_indices = np.where(missingness_of_x == 1)

    df.at[missingness_indices[0], 'x'] = np.nan

    graph = MarkedPatternGraph(
        nodes=['x','y']
    )

    direct_causes_of_missingness_finder = DirectCausesOfMissingnessFinder(
        data=df,
        graph=graph
    )

    marked_arrows = direct_causes_of_missingness_finder.find()

    assert marked_arrows == []

def test_2_multinom_RVs_MAR(
    df_X_Y_indep_Y_causes_MI_X
):

    size = 2000

    df = df_X_Y_indep_Y_causes_MI_X(size=size)

    graph = MarkedPatternGraph(
        nodes=['x','y']
    )

    direct_causes_of_missingness_finder = DirectCausesOfMissingnessFinder(
        data=df,
        graph=graph
    )

    marked_arrows = direct_causes_of_missingness_finder.find()

    assert marked_arrows == [('y', 'MI_x')]

def test_3_multinom_RVs_MAR(
    df_Z_causes_X_Y_and_X_Z_causes_MI_Y
):
    size = 1000

    df = df_Z_causes_X_Y_and_X_Z_causes_MI_Y(size=size)
    graph = MarkedPatternGraph(
        nodes=['x','y','z']
    )

    direct_causes_of_missingness_finder = DirectCausesOfMissingnessFinder(
        data=df,
        graph=graph
    )

    marked_arrows = direct_causes_of_missingness_finder.find()
    assert set(marked_arrows) == set([('z', 'MI_y'), ('x', 'MI_y')])

def test_long_chains_collider_bias_without_MI(
    df_long_chains_and_collider_without_MI
):
    size = 1000

    df = df_long_chains_and_collider_without_MI(size=size)

    graph = MarkedPatternGraph(
        nodes=df.columns,
        undirected_edges=[
            set({'a', 'b'}),
            set({'b', 'c'}),
            set({'c', 'd'}),
            set({'d', 'e'}),
        ]
    )

    direct_causes_of_missingness_finder = DirectCausesOfMissingnessFinder(
        data=df,
        graph=graph
    )

    marked_arrows = direct_causes_of_missingness_finder.find()

    expected_marked_arrows = frozenset({})

    assert frozenset(marked_arrows) == expected_marked_arrows

def test_chain_and_collider_with_MI(
    df_chain_and_collider_with_MI
):
    size = 10000

    df = df_chain_and_collider_with_MI(size=size)

    graph = MarkedPatternGraph(
        nodes=df.columns,
        undirected_edges=[
            set({'a', 'b'}),
            set({'b', 'c'}),
            set({'c', 'd'}),
            set({'a', 'd'}),
        ]
    )

    direct_causes_of_missingness_finder = DirectCausesOfMissingnessFinder(
        data=df,
        graph=graph
    )

    marked_arrows = direct_causes_of_missingness_finder.find()

    expected_marked_arrows = frozenset({
      ('d', 'MI_a')
    })

    assert frozenset(marked_arrows) == expected_marked_arrows

def test_long_chains_collider_bias_with_MI(
    df_long_chains_and_collider_with_MI
):
    size = 10000

    df = df_long_chains_and_collider_with_MI(size=size)

    graph = MarkedPatternGraph(
        nodes=df.columns,
        undirected_edges=[
            set({'a', 'b'}),
            set({'b', 'c'}),
            set({'c', 'd'}),
            set({'d', 'e'}),
        ]
    )

    direct_causes_of_missingness_finder = DirectCausesOfMissingnessFinder(
        data=df,
        graph=graph
    )

    marked_arrows = direct_causes_of_missingness_finder.find()

    expected_marked_arrows = frozenset({
      ('c', 'MI_b')
    })

    assert frozenset(marked_arrows) == expected_marked_arrows
