import pytest
from constraint_based.time_edge_orienter import TimeEdgeOrienter
from graphs.marked_pattern_graph import MarkedPatternGraph

def test_chain_without_no_arguments(simple_chain_graph):
    # True DAG is X_1 -> X_2 -> X_3
    graph = simple_chain_graph
    TimeEdgeOrienter(graph).orient()

    assert graph.get_unmarked_arrows() == set({
        ('X_t=1', 'X_t=2'), ('X_t=2', 'X_t=3')
    })

def test_when_some_vars_are_in_the_same_time_window():
    graph = MarkedPatternGraph(
        nodes=['X_t=2', 'Y_t=2'],
        undirected_edges=[('X_t=2', 'Y_t=2')]
    )

    TimeEdgeOrienter(graph).orient()

    assert graph.get_unmarked_arrows() == set({})

def test_immorality_across_time():
    # X_t=1 --> Y_t=2 <-- X_t=3
    graph = MarkedPatternGraph(
        nodes=['X_t=1', 'Y_t=2', 'X_t=3'],
        unmarked_arrows=[
            ('X_t=1', 'Y_t=2'),
            ('X_t=3', 'Y_t=2')
        ]
    )

    TimeEdgeOrienter(graph).orient()

    # X_t=1 --> Y_t=2 <--> X_t=3
    assert graph.get_unmarked_arrows() == set({
        ('X_t=1', 'Y_t=2')
    })

    assert graph.get_bidirectional_edges() == set({
        frozenset({'X_t=3', 'Y_t=2'})
    })


