import pytest
import pandas as pd
import numpy as np
from causal_discovery import SkeletonFinder, DirectCausesOfMissingnessFinder, PotentiallyExtraneousEdgesFinder, DensityRatioWeightedCorrection, adjusted_data_generator, PotentiallyExtraneousEdgesValidator
from information_theory import sci_is_independent
from viz import MarkedPatternGraph

def test_skeleton_finder_2_multinom_RVs(df_2_multinomial_indep_RVs):
    skeleton_finder = SkeletonFinder(
        data=df_2_multinomial_indep_RVs(size=1000),
        var_names=['x', 'y'],
    )

    graph, cond_sets_satisfying_cond_indep, df = skeleton_finder.find()

    assert graph.undirected_edges == []
    assert cond_sets_satisfying_cond_indep['x _||_ y'] == [set()]

def test_skeleton_finder_X_causes_Y(df_X_causes_Y):
    skeleton_finder = SkeletonFinder(
        data=df_X_causes_Y(size=1000),
        var_names=['x', 'y'],
    )

    graph, cond_sets_satisfying_cond_indep, df  = skeleton_finder.find()

    assert graph.undirected_edges == [set(('x', 'y'))]
    assert cond_sets_satisfying_cond_indep == {}

def test_skeleton_finder_Z_causes_X_and_Y(df_Z_causes_X_and_Y):
    var_names = ['x', 'y', 'z', 'MI_x', 'MI_y', 'MI_z']

    skeleton_finder = SkeletonFinder(
        data=df_Z_causes_X_and_Y(size=1000),
        var_names=var_names,
    )

    graph, cond_sets_satisfying_cond_indep, df = skeleton_finder.find()

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

    graph, cond_sets_satisfying_cond_indep, df_with_mi = \
        skeleton_finder.find()

    direct_causes_of_missingness_finder = DirectCausesOfMissingnessFinder(
        data=df_with_mi,
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

    graph, cond_sets_satisfying_cond_indep, df_with_mi = \
        skeleton_finder.find()

    direct_causes_of_missingness_finder = DirectCausesOfMissingnessFinder(
        data=df_with_mi,
        marked_pattern_graph=graph,
    )

    new_graph = direct_causes_of_missingness_finder.find()

    assert set(graph.nodes).intersection(set(['x', 'y', 'MI_x']))
    assert graph.undirected_edges == []
    assert graph.marked_arrows == [('y', 'MI_x')]

def test_direct_causes_of_missingness_finder_3_multinom_RVs_MAR(
    df_Z_causes_X_Y_and_X_Z_causes_MI_Y
):
    size = 1000

    df = df_Z_causes_X_Y_and_X_Z_causes_MI_Y(size=size)

    skeleton_finder = SkeletonFinder(
        data=df,
        var_names=df.columns,
    )

    graph, cond_sets_satisfying_cond_indep, df_with_mi = \
        skeleton_finder.find()

    direct_causes_of_missingness_finder = DirectCausesOfMissingnessFinder(
        data=df_with_mi,
        marked_pattern_graph=graph,
    )

    new_graph = direct_causes_of_missingness_finder.find()
    assert set(graph.nodes).intersection(set(['x', 'y', 'MI_x']))
    assert graph.undirected_edges == [set(('x', 'z')), set(('z', 'y'))]
    assert set(graph.marked_arrows) == set([('z', 'MI_y'), ('x', 'MI_y')])

def test_direct_causes_of_missingness_finder_long_chains_collider_bias_without_MI(
    df_long_chains_and_collider_without_MI
):
    size = 10000

    df = df_long_chains_and_collider_without_MI(size=size)
    skeleton_finder = SkeletonFinder(
        data=df,
        var_names=df.columns,
    )

    graph, cond_sets_satisfying_cond_indep, df_with_mi = skeleton_finder.find()

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
        data=df_with_mi,
        marked_pattern_graph=graph,
    )

    new_graph = direct_causes_of_missingness_finder.find()

    expected_marked_arrows = frozenset({})

    assert frozenset(new_graph.marked_arrows) == expected_marked_arrows

def test_direct_causes_of_missingness_finder_long_chains_collider_bias_with_MI(
    df_long_chains_and_collider_with_MI
):
    size = 10000

    df = df_long_chains_and_collider_with_MI(size=size)
    skeleton_finder = SkeletonFinder(
        data=df,
        var_names=df.columns,
    )

    graph, cond_sets_satisfying_cond_indep, df = skeleton_finder.find()

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
        data=df,
        marked_pattern_graph=graph,
    )

    new_graph = direct_causes_of_missingness_finder.find()


    expected_marked_arrows = frozenset({
      ('c', 'MI_a')
    })

    assert frozenset(new_graph.marked_arrows) == expected_marked_arrows

def test_potentially_extraneous_edges_finder_mcar():
    marked_pattern_graph = MarkedPatternGraph(
        nodes=['X', 'Y', 'MI_x']
    )

    potentially_extraneous_edges_finder = PotentiallyExtraneousEdgesFinder(
        data=pd.DataFrame(),
        marked_pattern_graph=marked_pattern_graph
    )

    potentially_extraneous_edges, marked_pattern = potentially_extraneous_edges_finder.find()
    assert potentially_extraneous_edges == []

def test_potentially_extraneous_edges_finder_mar():
    undirected_edges = [
        frozenset(('x', 'y')),
        frozenset(('y', 'z')),
        frozenset(('x', 'z')),
        frozenset(('x', 'w')),
        frozenset(('y', 'w')),
    ]

    marked_pattern_graph = MarkedPatternGraph(
        nodes=['x', 'y', 'z', 'w', 'MI_x'],
        marked_arrows=[('w', 'MI_x')],
        undirected_edges=undirected_edges
    )

    potentially_extraneous_edges_finder = PotentiallyExtraneousEdgesFinder(
        data=pd.DataFrame(),
        marked_pattern_graph=marked_pattern_graph
    )

    potentially_extraneous_edges = \
        potentially_extraneous_edges_finder.find()


    assert potentially_extraneous_edges == set(undirected_edges)

def test_potentially_extraneous_edges_finder_two_causes_MI_collider():
    undirected_edges = [
        frozenset(('z', 'y'))
    ]

    marked_pattern_graph = MarkedPatternGraph(
        nodes=['x', 'y', 'z', 'MI_x'],
        marked_arrows=[('y', 'MI_x'), ('z', 'MI_x' )],
        undirected_edges=undirected_edges
    )

    potentially_extraneous_edges_finder = PotentiallyExtraneousEdgesFinder(
        data=pd.DataFrame(),
        marked_pattern_graph=marked_pattern_graph
    )

    potentially_extraneous_edges = \
        potentially_extraneous_edges_finder.find()

    assert potentially_extraneous_edges == set(undirected_edges)

def test_potentially_extraneous_edges_finder_marked_arrow_exists_with_no_MI():
    undirected_edges = [
        frozenset(('z', 'y'))
    ]

    marked_pattern_graph = MarkedPatternGraph(
        nodes=['x', 'y', 'z', 'MI_x'],
        marked_arrows=[('x', 'y')],
        undirected_edges=undirected_edges
    )

    potentially_extraneous_edges_finder = PotentiallyExtraneousEdgesFinder(
        data=pd.DataFrame(),
        marked_pattern_graph=marked_pattern_graph,
        adjusted_data_generator=adjusted_data_generator
    )

    potentially_extraneous_edges, marked_pattern = \
        potentially_extraneous_edges_finder.find()

    assert potentially_extraneous_edges == set([])

def test_potentially_extraneous_edges_finder_firing_squad_example():
    undirected_edges = [
        frozenset(('captain', 'rifle_person_1')),
        frozenset(('captain', 'rifle_person_2')),
        frozenset(('rifle_person_1', 'death')),
        frozenset(('rifle_person_2', 'death')),
    ]

    marked_pattern_graph = MarkedPatternGraph(
        nodes=['captain',
               'rifle_person_1',
               'rifle_person_2',
               'death',
               'MI_captain'],
        marked_arrows=[('death', 'MI_captain')],
        undirected_edges=undirected_edges
    )

    potentially_extraneous_edges_finder = PotentiallyExtraneousEdgesFinder(
        data=pd.DataFrame(),
        marked_pattern_graph=marked_pattern_graph
    )

    potentially_extraneous_edges = \
        potentially_extraneous_edges_finder.find()

    assert potentially_extraneous_edges == set([])

def test_potentially_extraneous_edges_validator(df_X_and_Y_cause_Z_and_Z_cause_MI_X):
    df = df_X_and_Y_cause_Z_and_Z_cause_MI_X()

    # extraneous edge x-y
    graph = MarkedPatternGraph(
        nodes=['x', 'y', 'z', 'MI_x', 'MI_y', 'MI_z'],
        undirected_edges=[set({'x', 'y'}), set({'x', 'z'}), set({'y', 'z'})],
        marked_arrows=[('z', 'MI_x')]
    )

    validator = PotentiallyExtraneousEdgesValidator(
        data=df,
        marked_pattern_graph=graph,
        potentially_extraneous_edges=[set({'x', 'y'})],
        adjusted_data_generator=adjusted_data_generator,
    )

    removables = validator.edges_to_remove()

    assert removables == [set({'x', 'y'})]
