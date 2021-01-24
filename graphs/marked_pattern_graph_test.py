import pytest
from .marked_pattern_graph import MarkedPatternGraph

def test_add_undirected_edge():
    graph = MarkedPatternGraph(nodes=['a', 'b'])
    graph.add_undirected_edge(('a', 'b'))

    assert set(graph.get_undirected_edges()) == set({'a', 'b'})

def test_remove_undirected_edge():
    graph = MarkedPatternGraph(nodes=['a', 'b'])
    graph.add_undirected_edge(('a', 'b'))
    graph.remove_undirected_edge(('a', 'b'))

    assert set(graph.get_undirected_edges()) == set({})

def test_remove_undirected_edge_when_not_exist():
    graph = MarkedPatternGraph(nodes=['a', 'b'])
    graph.remove_undirected_edge(('a', 'b'))

    assert set(graph.get_undirected_edges()) == set({})

def test_add_arrowhead():
    graph = MarkedPatternGraph(nodes=['a', 'b'])
    graph.add_undirected_edge(('a', 'b'))
    graph.add_arrowhead(('a', 'b'))

    assert set(graph.get_undirected_edges()) == set({})
    assert set(graph.get_unmarked_arrows()) == set({('a', 'b')})

def test_add_marked_arrowhead():
    graph = MarkedPatternGraph(nodes=['a', 'b'])
    graph.add_undirected_edge(('a', 'b'))
    graph.add_marked_arrowhead(('a', 'b'))

    assert set(graph.get_undirected_edges()) == set({})
    assert set(graph.get_unmarked_arrows()) == set({})
    assert set(graph.get_marked_arrows()) == set({('a', 'b')})
