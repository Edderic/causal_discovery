import pytest
import numpy as np
import pandas as pd
from constraint_based.pc_skeleton_finder import PCSkeletonFinder
from constraint_based.ci_tests.sci_is_independent import sci_is_independent
from data import dog_example
from constraint_based.misc import key_for_pair

# PCSkeletonFinder is missing an edge because the distribution we used is
# unstable. It has independencies that is incompatible with the true DAG
def test_2_deterministic_and_3rd_var_caused_by_one_of_them(
    df_2_deterministic_and_3rd_var_caused_by_one_of_them
):
    skeleton_finder = PCSkeletonFinder(
        data=df_2_deterministic_and_3rd_var_caused_by_one_of_them(size=1000),
    )

    graph, cond_sets_satisfying_cond_indep = skeleton_finder.find()

    assert graph.get_undirected_edges() == set({frozenset({'x', 'y'})})
    assert graph.get_nodes() == set({'x', 'y', 'z'})

def test_2_multinom_RVs(df_2_multinomial_indep_RVs):
    skeleton_finder = PCSkeletonFinder(
        data=df_2_multinomial_indep_RVs(size=10000),
    )

    graph, cond_sets_satisfying_cond_indep = skeleton_finder.find()

    assert graph.get_undirected_edges() == set({})
    assert cond_sets_satisfying_cond_indep['x _||_ y'] == set({frozenset({})})

def test_skeleton_finder_X_causes_Y(df_X_causes_Y):
    skeleton_finder = PCSkeletonFinder(
        data=df_X_causes_Y(size=1000),
    )

    graph, cond_sets_satisfying_cond_indep = skeleton_finder.find()

    assert graph.get_undirected_edges() == set({frozenset(('x', 'y'))})
    assert cond_sets_satisfying_cond_indep == {}

def test_skeleton_finder_Z_causes_X_and_Y(df_Z_causes_X_and_Y):

    skeleton_finder = PCSkeletonFinder(
        data=df_Z_causes_X_and_Y(size=1000),
    )

    graph, cond_sets_satisfying_cond_indep = skeleton_finder.find()

    assert graph.get_undirected_edges() == set({
        frozenset(('x', 'z')),
        frozenset(('y', 'z'))
    })
    assert cond_sets_satisfying_cond_indep == {'x _||_ y': set({frozenset({'z'})})}

def test_long_chains_collider_bias_without_MI(
    df_long_chains_and_collider_without_MI
):
    size = 100000

    df = df_long_chains_and_collider_without_MI(size=size)
    skeleton_finder = PCSkeletonFinder(
        data=df,
    )

    graph, cond_sets_satisfying_cond_indep = skeleton_finder.find()

    expected_undirected_edges = frozenset({
        frozenset({'a', 'b'}),
        frozenset({'d', 'e'}),
        frozenset({'b', 'c'}),
        frozenset({'c', 'd'}),
    })

    assert frozenset(graph.get_undirected_edges()) == expected_undirected_edges

def test_long_chains_collider_bias_with_MI(
    df_long_chains_and_collider_with_MI
):
    size = 10000

    df = df_long_chains_and_collider_with_MI(size=size, proba_noise=0.7)
    skeleton_finder = PCSkeletonFinder(
        data=df,
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

    assert frozenset(graph.get_undirected_edges()) == expected_undirected_edges

def test_chain_and_collider_without_MI(
    df_chain_and_collider_without_MI
):
    size = 10000

    df = df_chain_and_collider_without_MI(size=size)
    skeleton_finder = PCSkeletonFinder(
        data=df,
    )

    graph, cond_sets_satisfying_cond_indep = skeleton_finder.find()

    expected_undirected_edges = frozenset({
        frozenset({'a', 'b'}),
        frozenset({'b', 'c'}),
        frozenset({'a', 'd'}),
        frozenset({'c', 'd'})
    })

    assert frozenset(graph.get_undirected_edges()) == expected_undirected_edges

def test_chain_and_collider_with_MI(
    df_chain_and_collider_with_MI
):
    size = 20000

    df = df_chain_and_collider_with_MI(size=size)
    skeleton_finder = PCSkeletonFinder(
        data=df,
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

    assert frozenset(graph.get_undirected_edges()) == expected_undirected_edges

def test_3_multinom_RVs_MAR(
    df_Z_causes_X_Y_and_X_Z_causes_MI_Y
):
    size = 70000

    df = df_Z_causes_X_Y_and_X_Z_causes_MI_Y(size=size)

    skeleton_finder = PCSkeletonFinder(
        data=df,
    )

    graph, cond_sets_satisfying_cond_indep = \
        skeleton_finder.find()

    assert set(graph.nodes).intersection(set(['x', 'y', 'MI_x']))
    assert set(graph.get_undirected_edges()) == frozenset({frozenset(('x', 'z')), frozenset(('z', 'y'))})

def test_dog_pee():
    size = 100000

    # Sometimes cloudy
    cloudy = np.random.binomial(n=1, p=0.5, size=size)

    # Cloudyness causes rain, but sometimes it rains even when it's not cloudy.
    rain = cloudy * np.random.binomial(n=1, p=0.7, size=size) + (1 - cloudy) * np.random.binomial(n=1, p=0.1, size=size)

    # Sprinkler generally turns on when it isn't cloudy.
    sprinkler = (cloudy == 0) * np.random.binomial(n=1, p=0.8, size=size) + cloudy * np.random.binomial(n=1, p=0.1, size=size)

    # Grass is generally wet whenever it rained or the sprinkler is on.
    wet_grass = (rain | sprinkler) * np.random.binomial(n=1, p=0.90, size=size)

    # Dog doesn't like to get rained on
    # Dog goes out more frequently when it's not raining
    dog_goes_out_to_pee = rain * np.random.binomial(n=1, p=0.2, size=size) + (1 - rain) * np.random.binomial(n=1, p=0.9, size=size)

    df = pd.DataFrame({
        'cloudy': cloudy,
        'sprinkler': sprinkler,
        'rain': rain,
        'wet_grass': wet_grass,
        'dog_goes_out_to_pee': dog_goes_out_to_pee
    })

    skeleton_finder = PCSkeletonFinder(
        data=df,
    )

    graph, cond_sets_satisfying_cond_indep = \
        skeleton_finder.find()

    assert set(graph.get_undirected_edges()) == frozenset({
        frozenset(('cloudy', 'rain')),
        frozenset(('cloudy', 'sprinkler')),
        frozenset(('rain', 'dog_goes_out_to_pee')),
        frozenset(('rain', 'wet_grass')),
        frozenset(('sprinkler', 'wet_grass')),
    })

def test_dog_example():
    df = dog_example(size=100000)

    skeleton_finder = PCSkeletonFinder(
        data=df,
    )

    graph, cond_sets_satisfying_cond_indep = \
        skeleton_finder.find()

    assert cond_sets_satisfying_cond_indep[
        key_for_pair(('activity', 'dog_tired'))
    ].intersection(set({frozenset({'exercise_levels'})})) == set({frozenset({'exercise_levels'})})

    assert set({frozenset({'best_friends_visit', 'activity'})}) not in \
        cond_sets_satisfying_cond_indep[
            key_for_pair(('weekend', 'mentally_exhausted_before_bed'))
        ]

    assert graph.get_undirected_edges() == frozenset({
        frozenset(('rain', 'best_friends_visit')),
        frozenset(('weekend', 'best_friends_visit')),
        frozenset(('rain', 'activity')),
        frozenset(('exercise_levels', 'best_friends_visit')),
        frozenset(('exercise_levels', 'activity')),
        frozenset(('mentally_exhausted_before_bed', 'activity')),
        frozenset(('exercise_levels', 'dog_tired')),
        frozenset(('best_friends_visit', 'mentally_exhausted_before_bed')),
        frozenset(('mentally_exhausted_before_bed', 'carry_dog_for_last_potty_before_bed')),
        frozenset(('dog_tired', 'carry_dog_for_last_potty_before_bed')),
    })
