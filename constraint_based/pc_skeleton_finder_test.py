#constraint_based/pc_skeleton_finder_test.py pylint: disable=missing-module-docstring,missing-function-docstring
import pytest # pylint: disable=unused-import
import numpy as np
import pandas as pd
from causal_discovery.constraint_based.misc import key_for_pair
from causal_discovery.constraint_based.pc_skeleton_finder import PCSkeletonFinder
from causal_discovery.data import dog_example
from causal_discovery.graphs.partial_ancestral_graph import PartialAncestralGraph as Graph


# PCSkeletonFinder is missing an edge because the distribution we used is
# unstable. It has independencies that is incompatible with the true DAG
@pytest.mark.f
def test_2_deterministic_and_3rd_var_caused_by_one_of_them(
    df_2_deterministic_and_3rd_var_caused_by_one_of_them,
    dask_client
):
    df = df_2_deterministic_and_3rd_var_caused_by_one_of_them(size=1000)
    graph = Graph(variables=list(df.columns), complete=True)

    skeleton_finder = PCSkeletonFinder(
        data=df,
        graph=graph,
        client=dask_client
    )

    skeleton_finder.find()

    assert graph.has_adjacency(('x','y'))
    assert graph.get_nodes() == set({'x', 'y', 'z'}) # pylint: disable='no-member'

def test_2_multinom_RVs(df_2_multinomial_indep_RVs, dask_client):
    df = df_2_multinomial_indep_RVs(size=10000)
    graph = Graph(
        variables=list(df.columns),

        complete=True
    )

    skeleton_finder = PCSkeletonFinder(
        data=df,
        graph=graph,
        client=dask_client
    )

    cond_sets_satisfying_cond_indep = skeleton_finder.find()

    assert graph.get_edges() == []
    assert cond_sets_satisfying_cond_indep['x _||_ y'] == set({frozenset({})})

def test_skeleton_finder_X_causes_Y(df_X_causes_Y, dask_client):
    df = df_X_causes_Y(size=1000)

    graph = Graph(
        variables=list(df.columns),
        complete=True
    )

    skeleton_finder = PCSkeletonFinder(
        data=df,
        graph=graph,
        client=dask_client
    )

    cond_sets_satisfying_cond_indep = skeleton_finder.find()

    assert graph.has_adjacency(('x', 'y'))
    assert cond_sets_satisfying_cond_indep == {}

def test_skeleton_finder_Z_causes_X_and_Y(df_Z_causes_X_and_Y):
    df = df_Z_causes_X_and_Y(size=1000)

    graph = Graph(
        variables=list(df.columns),
        complete=True
    )

    skeleton_finder = PCSkeletonFinder(
        data=df,
        graph=graph,
        client=dask_client
    )

    cond_sets_satisfying_cond_indep = skeleton_finder.find()

    assert graph.has_adjacency(('x', 'z'))
    assert graph.has_adjacency(('y', 'z'))
    assert cond_sets_satisfying_cond_indep == \
        {'x _||_ y': set({frozenset({'z'})})}

def test_long_chains_collider_bias_without_MI(
    df_long_chains_and_collider_without_MI,
    dask_client
):
    size = 100000

    df = df_long_chains_and_collider_without_MI(size=size)

    graph = Graph(
        variables=list(df.columns),
        complete=True
    )

    skeleton_finder = PCSkeletonFinder(
        data=df,
        graph=graph,
        client=dask_client
    )

    skeleton_finder.find()

    assert graph.has_adjacency(('a', 'b'))
    assert graph.has_adjacency(('d', 'e'))
    assert graph.has_adjacency(('b', 'c'))
    assert graph.has_adjacency(('c', 'd'))

def test_long_chains_collider_bias_with_MI(
    df_long_chains_and_collider_with_MI,
    dask_client
):
    size = 10000

    df = df_long_chains_and_collider_with_MI(size=size, proba_noise=0.7)

    graph = Graph(
        variables=list(df.columns),
        complete=True
    )

    skeleton_finder = PCSkeletonFinder(
        data=df,
        graph=graph,
        client=dask_client
    )

    skeleton_finder.find()

    # we expect b-d in this intermediate stage. b-d is spurious, due to
    # collider bias.

    assert graph.has_adjacency(('a', 'b'))
    assert graph.has_adjacency(('b', 'd'))
    assert graph.has_adjacency(('d', 'e'))
    assert graph.has_adjacency(('b', 'c'))
    assert graph.has_adjacency(('c', 'd'))

def test_chain_and_collider_without_MI(
    df_chain_and_collider_without_MI,
    dask_client
):
    size = 10000

    df = df_chain_and_collider_without_MI(size=size)

    graph = Graph(
        variables=list(df.columns),
        complete=True
    )

    skeleton_finder = PCSkeletonFinder(
        data=df,
        graph=graph,
        client=dask_client
    )

    skeleton_finder.find()

    assert graph.has_adjacency(('a', 'b'))
    assert graph.has_adjacency(('b', 'c'))
    assert graph.has_adjacency(('a', 'd'))
    assert graph.has_adjacency(('c', 'd'))

def test_chain_and_collider_with_MI(
    df_chain_and_collider_with_MI,
    dask_client
):
    size = 20000

    df = df_chain_and_collider_with_MI(size=size)
    graph = Graph(
        variables=list(df.columns),
        complete=True
    )

    skeleton_finder = PCSkeletonFinder(
        data=df,
        graph=graph,
        client=dask_client
    )

    skeleton_finder.find()

    # we expect a-c in this intermediate stage. a-c is spurious, due to
    # collider bias.

    assert graph.has_adjacency(('a', 'c'))
    assert graph.has_adjacency(('a', 'b'))
    assert graph.has_adjacency(('b', 'c'))
    assert graph.has_adjacency(('a', 'd'))
    assert graph.has_adjacency(('c', 'd'))

def test_3_multinom_RVs_MAR(
    df_Z_causes_X_Y_and_X_Z_causes_MI_Y,
    dask_client
):
    size = 70000

    df = df_Z_causes_X_Y_and_X_Z_causes_MI_Y(size=size)

    graph = Graph(
        variables=list(df.columns),
        complete=True
    )

    skeleton_finder = PCSkeletonFinder(
        data=df,
        graph=graph,
        client=dask_client
    )

    skeleton_finder.find()

    assert set(graph.get_nodes()).intersection(set(['x', 'y', 'MI_x']))  # pylint: disable='no-member'

    assert graph.has_adjacency(('x', 'z'))
    assert graph.has_adjacency(('y', 'z'))

def test_dog_pee(dask_client):
    size = 100000

    # Sometimes cloudy
    cloudy = np.random.binomial(n=1, p=0.5, size=size)

    # Cloudyness causes rain, but sometimes it rains even when it's not cloudy.
    rain = cloudy * np.random.binomial(n=1, p=0.7, size=size) \
        + (1 - cloudy) * np.random.binomial(n=1, p=0.1, size=size)

    # Sprinkler generally turns on when it isn't cloudy.
    sprinkler = (cloudy == 0) * np.random.binomial(n=1, p=0.8, size=size) \
        + cloudy * np.random.binomial(n=1, p=0.1, size=size)

    # Grass is generally wet whenever it rained or the sprinkler is on.
    wet_grass = (rain | sprinkler) * np.random.binomial(n=1, p=0.90, size=size)

    # Dog doesn't like to get rained on
    # Dog goes out more frequently when it's not raining
    dog_goes_out_to_pee = rain * np.random.binomial(n=1, p=0.2, size=size) \
        + (1 - rain) * np.random.binomial(n=1, p=0.9, size=size)

    df = pd.DataFrame({
        'cloudy': cloudy,
        'sprinkler': sprinkler,
        'rain': rain,
        'wet_grass': wet_grass,
        'dog_goes_out_to_pee': dog_goes_out_to_pee
    })

    graph = Graph(
        variables=list(df.columns),
        complete=True
    )
    skeleton_finder = PCSkeletonFinder(
        data=df,
        graph=graph,
        client=dask_client
    )

    skeleton_finder.find()

    assert graph.has_adjacency(('cloudy', 'rain'))
    assert graph.has_adjacency(('cloudy', 'sprinkler'))
    assert graph.has_adjacency(('rain', 'dog_goes_out_to_pee'))
    assert graph.has_adjacency(('rain', 'wet_grass'))
    assert graph.has_adjacency(('sprinkler', 'wet_grass'))

def test_dog_example():
    df = dog_example(size=100000)

    graph = Graph(
        variables=list(df.columns),
        complete=True
    )

    skeleton_finder = PCSkeletonFinder(
        data=df,
        graph=graph,
        client=dask_client
    )

    cond_sets_satisfying_cond_indep = \
        skeleton_finder.find()

    assert cond_sets_satisfying_cond_indep[
        key_for_pair(('activity', 'dog_tired'))
    ].intersection(set({frozenset({'exercise_levels'})})) == set({frozenset({'exercise_levels'})})

    assert set({frozenset({'best_friends_visit', 'activity'})}) not in \
        cond_sets_satisfying_cond_indep[
            key_for_pair(('weekend', 'mentally_exhausted_before_bed'))
        ]

    assert graph.has_adjacency(('rain', 'best_friends_visit'))
    assert graph.has_adjacency(('weekend', 'best_friends_visit'))
    assert graph.has_adjacency(('rain', 'activity'))
    assert graph.has_adjacency(('exercise_levels', 'best_friends_visit'))
    assert graph.has_adjacency(('exercise_levels', 'activity'))
    assert graph.has_adjacency(('mentally_exhausted_before_bed', 'activity'))
    assert graph.has_adjacency(('exercise_levels', 'dog_tired'))
    assert graph.has_adjacency(('best_friends_visit', 'mentally_exhausted_before_bed'))
    assert graph.has_adjacency(('mentally_exhausted_before_bed', 'dog_teeth_brushed'))
    assert graph.has_adjacency(('dog_tired', 'dog_teeth_brushed'))
