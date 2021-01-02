import pytest
import numpy as np
from causal_discovery import SkeletonFinder, DirectCausesOfMissingnessFinder

def test_skeleton_finder_2_multinom_RVs(df_2_multinomial_indep_RVs):
    skeleton_finder = SkeletonFinder(
        data=df_2_multinomial_indep_RVs(size=1000),
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
    var_names = ['x', 'y', 'z']

    skeleton_finder = SkeletonFinder(
        data=df_Z_causes_X_and_Y(size=1000),
        var_names=var_names,
    )

    graph, cond_sets_satisfying_cond_indep = skeleton_finder.find()

    assert graph.nodes == var_names
    assert graph.undirected_edges == [set(('x', 'z')), set(('y', 'z'))]
    assert cond_sets_satisfying_cond_indep == {'x _||_ y': [set(('z'))]}

def test_direct_causes_of_missingness_finder_2_multinom_RVs_MCAR(
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

    graph, cond_sets_satisfying_cond_indep = skeleton_finder.find()

    direct_causes_of_missingness_finder = DirectCausesOfMissingnessFinder(
        data=df,
        marked_pattern_graph=graph,
    )

    new_graph = direct_causes_of_missingness_finder.find()

    assert set(graph.nodes).intersection(set(['x', 'y', 'MI_x']))
    assert graph.undirected_edges == []

def test_direct_causes_of_missingness_finder_2_multinom_RVs_MAR(
    df_X_Y_indep_Y_causes_MI_X
):

    size = 10000

    df = df_X_Y_indep_Y_causes_MI_X(size=size)

    skeleton_finder = SkeletonFinder(
        data=df,
        var_names=df.columns,
    )

    graph, cond_sets_satisfying_cond_indep = skeleton_finder.find()

    direct_causes_of_missingness_finder = DirectCausesOfMissingnessFinder(
        data=df,
        marked_pattern_graph=graph,
    )

    new_graph = direct_causes_of_missingness_finder.find()

    assert set(graph.nodes).intersection(set(['x', 'y', 'MI_x']))
    assert graph.undirected_edges == []
    assert graph.marked_arrows == [('y', 'MI_x')]

@pytest.mark.focus
def test_direct_causes_of_missingness_finder_3_multinom_RVs_MAR(
    df_Z_causes_X_Y_and_X_Z_causes_MI_Y
):
    size = 1000

    df = df_Z_causes_X_Y_and_X_Z_causes_MI_Y(size=size)

    skeleton_finder = SkeletonFinder(
        data=df,
        var_names=df.columns,
    )

    graph, cond_sets_satisfying_cond_indep = skeleton_finder.find()

    direct_causes_of_missingness_finder = DirectCausesOfMissingnessFinder(
        data=df,
        marked_pattern_graph=graph,
    )

    new_graph = direct_causes_of_missingness_finder.find()
    assert set(graph.nodes).intersection(set(['x', 'y', 'MI_x']))
    assert graph.undirected_edges == [set(('x', 'z')), set(('z', 'y'))]
    assert set(graph.marked_arrows) == set([('z', 'MI_y'), ('x', 'MI_y')])
