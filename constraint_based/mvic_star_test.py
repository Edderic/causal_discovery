import pytest
from .mvic_star import MVICStar

def test_long_chains_and_collider_with_MI(df_long_chains_and_collider_with_MI):
    df = df_long_chains_and_collider_with_MI(size=50000)

    graph = MVICStar(
        data=df
    ).predict()

    assert graph.get_undirected_edges() == set({ frozenset({'a', 'b'}), frozenset({'e', 'd'}) })
    assert graph.get_unmarked_arrows() == set({
        ('b', 'c'),
        ('d', 'c')
    })
    assert graph.get_marked_arrows() == set({
        ('c', 'MI_b')
    })
