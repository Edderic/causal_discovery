import pytest
from causal_discovery import SkeletonFinder

def test_skeleton_finder_2_multinom_RVs(df_2_multinomial_indep_RVs):
    skeleton_finder = SkeletonFinder(
        data=df_2_multinomial_indep_RVs(size=1000),
        var_names=['x', 'y'],
    )

    graph = skeleton_finder.find()

    assert graph.undirected_edges == []

def test_skeleton_finder_X_causes_Y(df_X_causes_Y):
    skeleton_finder = SkeletonFinder(
        data=df_X_causes_Y(size=1000),
        var_names=['x', 'y'],
    )

    graph = skeleton_finder.find()

    assert graph.undirected_edges == [('x', 'y')]

def test_skeleton_finder_Z_causes_X_and_Y(df_Z_causes_X_and_Y):
    var_names = ['x', 'y', 'z']

    skeleton_finder = SkeletonFinder(
        data=df_Z_causes_X_and_Y(size=1000),
        var_names=var_names,
    )

    graph = skeleton_finder.find()

    assert graph.nodes == var_names
    assert graph.undirected_edges == [('x', 'z'), ('y', 'z')]
