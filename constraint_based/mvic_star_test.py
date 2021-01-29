import pytest
from constraint_based.mvic_star import MVICStar

def test_long_chains_and_collider_without_MI(df_long_chains_and_collider_without_MI):
    df = df_long_chains_and_collider_without_MI(size=50000)

    graph = MVICStar(
        data=df
    ).predict()

    assert graph.get_bidirectional_edges() == set({})
    assert graph.get_undirected_edges() == set({ frozenset({'a', 'b'}), frozenset({'e', 'd'}) })
    assert graph.get_unmarked_arrows() == set({
        ('b', 'c'),
        ('d', 'c')
    })
    assert graph.get_marked_arrows() == set({})

def test_long_chains_and_collider_with_MI(df_long_chains_and_collider_with_MI):
    df = df_long_chains_and_collider_with_MI(size=50000)

    graph = MVICStar(
        data=df
    ).predict()

    assert graph.get_bidirectional_edges() == set({})
    assert graph.get_undirected_edges() == set({ frozenset({'a', 'b'}), frozenset({'e', 'd'}) })
    assert graph.get_unmarked_arrows() == set({
        ('b', 'c'),
        ('d', 'c')
    })
    assert graph.get_marked_arrows() == set({
        ('c', 'MI_b')
    })

def test_chain_and_collider_with_MI(df_chain_and_collider_with_MI):
    df = df_chain_and_collider_with_MI()

    graph = MVICStar(
        data=df
    ).predict()

    assert graph.get_bidirectional_edges() == set({})
    assert graph.get_undirected_edges() == set({
        frozenset({'a', 'b'}), frozenset({'b', 'c'})
    })

    assert graph.get_unmarked_arrows() == set({
        ('a', 'd'),
        ('c', 'd')
    })
    assert graph.get_marked_arrows() == set({
        ('d', 'MI_a')
    })

def test_chain_and_collider_without_MI(df_chain_and_collider_without_MI):
    df = df_chain_and_collider_without_MI()

    graph = MVICStar(
        data=df
    ).predict()

    assert graph.get_bidirectional_edges() == set({})
    assert graph.get_undirected_edges() == set({
        frozenset({'a', 'b'}), frozenset({'b', 'c'})
    })

    assert graph.get_unmarked_arrows() == set({
        ('a', 'd'),
        ('c', 'd')
    })
    assert graph.get_marked_arrows() == set({})

def test_Z_causes_X_Y_and_X_Z_causes_MI_Y(df_Z_causes_X_Y_and_X_Z_causes_MI_Y):
    df = df_Z_causes_X_Y_and_X_Z_causes_MI_Y()

    graph = MVICStar(
        data=df
    ).predict()

    assert graph.get_bidirectional_edges() == set({})
    assert graph.get_undirected_edges() == set({
        frozenset({'x', 'z'}), frozenset({'z', 'y'})
    })
    assert graph.get_marked_arrows() == set({('x', 'MI_y'), ('z', 'MI_y')})
    assert graph.get_unmarked_arrows() == set({})

def test_X_Y_indep_Y_causes_MI_X(df_X_Y_indep_Y_causes_MI_X):
    df = df_X_Y_indep_Y_causes_MI_X()

    graph = MVICStar(
        data=df
    ).predict()

    assert graph.get_bidirectional_edges() == set({})
    assert graph.get_undirected_edges() == set({})
    assert graph.get_marked_arrows() == set({('y', 'MI_x')})
    assert graph.get_unmarked_arrows() == set({})
    assert graph.get_nodes()

def test_X_Y_indep_Y_causes_MI_X(df_X_Y_indep_Y_causes_MI_X):
    df = df_X_Y_indep_Y_causes_MI_X()

    graph = MVICStar(
        data=df
    ).predict()

    assert graph.get_bidirectional_edges() == set({})
    assert graph.get_undirected_edges() == set({})
    assert graph.get_marked_arrows() == set({('y', 'MI_x')})
    assert graph.get_unmarked_arrows() == set({})
    assert graph.get_nodes()
