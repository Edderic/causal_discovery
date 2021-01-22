import pytest
import numpy as np
from .skeleton_finder import SkeletonFinder

def test_2_multinom_RVs(df_2_multinomial_indep_RVs):
    skeleton_finder = SkeletonFinder(
        data=df_2_multinomial_indep_RVs(size=10000),
        var_names=['x', 'y'],
    )

    graph, cond_sets_satisfying_cond_indep = skeleton_finder.find()

    assert graph.undirected_edges == []
    assert cond_sets_satisfying_cond_indep['x _||_ y'] == [set()]

def test_skeleton_finder_X_causes_Y(df_X_causes_Y):
    skeleton_finder = SkeletonFinder(
        data=df_X_causes_Y(size=1000),
        var_names=['x', 'y'],
    )

    graph, cond_sets_satisfying_cond_indep = skeleton_finder.find()

    assert graph.undirected_edges == [set(('x', 'y'))]
    assert cond_sets_satisfying_cond_indep == {}

def test_skeleton_finder_Z_causes_X_and_Y(df_Z_causes_X_and_Y):
    var_names = ['x', 'y', 'z', 'MI_x', 'MI_y', 'MI_z']

    skeleton_finder = SkeletonFinder(
        data=df_Z_causes_X_and_Y(size=1000),
        var_names=var_names,
    )

    graph, cond_sets_satisfying_cond_indep = skeleton_finder.find()

    assert graph.nodes == var_names
    assert graph.undirected_edges == [set(('x', 'z')), set(('y', 'z'))]
    assert cond_sets_satisfying_cond_indep == {'x _||_ y': [set(('z'))]}

def test_long_chains_collider_bias_without_MI(
    df_long_chains_and_collider_without_MI
):
    size = 100000

    df = df_long_chains_and_collider_without_MI(size=size)
    skeleton_finder = SkeletonFinder(
        data=df,
        var_names=df.columns,
        only_find_one=True
    )

    graph, cond_sets_satisfying_cond_indep = skeleton_finder.find()

    expected_undirected_edges = frozenset({
        frozenset({'a', 'b'}),
        frozenset({'d', 'e'}),
        frozenset({'b', 'c'}),
        frozenset({'c', 'd'}),
    })

    assert frozenset(graph.undirected_edges) == expected_undirected_edges

@pytest.mark.focus
def test_long_chains_collider_bias_with_MI(
    df_long_chains_and_collider_with_MI
):
    size = 10000

    df = df_long_chains_and_collider_with_MI(size=size, proba_noise=0.7)
    skeleton_finder = SkeletonFinder(
        data=df,
        var_names=df.columns,
        # only_find_one=True
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

def test_chain_and_collider_without_MI(
    df_chain_and_collider_without_MI
):
    size = 100000

    df = df_chain_and_collider_without_MI(size=size)
    skeleton_finder = SkeletonFinder(
        data=df,
        var_names=df.columns,
        only_find_one=True
    )

    graph, cond_sets_satisfying_cond_indep = skeleton_finder.find()

    expected_undirected_edges = frozenset({
        frozenset({'a', 'b'}),
        frozenset({'b', 'c'}),
        frozenset({'a', 'd'}),
        frozenset({'c', 'd'})
    })

    assert frozenset(graph.undirected_edges) == expected_undirected_edges

def test_chain_and_collider_with_MI(
    df_chain_and_collider_with_MI
):
    size = 100000

    df = df_chain_and_collider_with_MI(size=size)
    skeleton_finder = SkeletonFinder(
        data=df,
        var_names=df.columns,
        only_find_one=True
    )

    graph, cond_sets_satisfying_cond_indep = skeleton_finder.find()

    # we expect a-c in this intermediate stage. a-c is spurious, due to
    # collider bias.

    expected_undirected_edges = frozenset({
        frozenset({'a', 'c'}),
        frozenset({'a', 'b'}),
        frozenset({'b', 'c'}),
        frozenset({'a', 'd'}),
        frozenset({'c', 'd'})
    })

    assert frozenset(graph.undirected_edges) == expected_undirected_edges
