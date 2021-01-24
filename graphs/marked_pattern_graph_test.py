import pytest
from .marked_pattern_graph import MarkedPatternGraph

def test_add_undirected_edge():
    graph = MarkedPatternGraph(nodes=['a', 'b'])
    graph.add_undirected_edge(('a', 'b'))

    assert graph.get_undirected_edges() == set({
        frozenset({'a', 'b'})
    })

def test_add_undirected_edge_instantiate():
    graph = MarkedPatternGraph(
        nodes=['a', 'b'],
        undirected_edges=[('a', 'b')]
    )

    assert graph.get_undirected_edges() == set({
        frozenset({'a', 'b'})
    })

def test_add_marked_arrows():
    graph = MarkedPatternGraph(
        nodes=['a', 'b'],
        undirected_edges=[('a', 'b')]
    )

    graph.add_marked_arrows(set({('c', 'd')}))

    assert graph.get_marked_arrows() == set({('c', 'd')})
    assert set(graph.get_edges()) == set({frozenset({'a', 'b'}), frozenset({'c', 'd'})})

def test_remove_undirected_edge():
    graph = MarkedPatternGraph(nodes=['a', 'b'])
    graph.add_undirected_edge(('a', 'b'))
    graph.remove_undirected_edge(('a', 'b'))

    assert set(graph.get_undirected_edges()) == set({})
    assert set(graph.get_unmarked_arrows()) == set({})
    assert set(graph.get_marked_arrows()) == set({})
    assert set(graph.get_bidirectional_edges()) == set({})
    assert set(graph.get_edges()) == set({})

def test_remove_undirected_edge_when_not_exist():
    graph = MarkedPatternGraph(nodes=['a', 'b'])
    graph.remove_undirected_edge(('a', 'b'))

    assert set(graph.get_undirected_edges()) == set({})
    assert set(graph.get_unmarked_arrows()) == set({})
    assert set(graph.get_marked_arrows()) == set({})
    assert set(graph.get_bidirectional_edges()) == set({})
    assert set(graph.get_edges()) == set({})

def test_add_arrowhead():
    graph = MarkedPatternGraph(nodes=['a', 'b'])
    graph.add_undirected_edge(('a', 'b'))
    graph.add_arrowhead(('a', 'b'))

    assert set(graph.get_undirected_edges()) == set({})
    assert set(graph.get_unmarked_arrows()) == set({('a', 'b')})
    assert set(graph.get_marked_arrows()) == set({})
    assert set(graph.get_bidirectional_edges()) == set({})
    assert set(graph.get_edges()) == set({frozenset({'a', 'b'})})

def test_add_marked_arrowhead():
    graph = MarkedPatternGraph(nodes=['a', 'b'])
    graph.add_undirected_edge(('a', 'b'))
    graph.add_marked_arrowhead(('a', 'b'))

    assert set(graph.get_undirected_edges()) == set({})
    assert set(graph.get_unmarked_arrows()) == set({})
    assert set(graph.get_marked_arrows()) == set({('a', 'b')})
    assert set(graph.get_bidirectional_edges()) == set({})
    assert set(graph.get_edges()) == set({frozenset({'a', 'b'})})

def test_bidirectional_edges():
    graph = MarkedPatternGraph(nodes=['a', 'b'])
    graph.add_undirected_edge(('a', 'b'))
    graph.add_arrowhead(('a', 'b'))
    graph.add_arrowhead(('b', 'a'))

    assert set(graph.get_undirected_edges()) == set({})
    assert set(graph.get_unmarked_arrows()) == set({})
    assert set(graph.get_marked_arrows()) == set({})
    assert set(graph.get_bidirectional_edges()) == set({frozenset({'a', 'b'})})

def test_has_arrowhead_with_marked_arrowhead():
    graph = MarkedPatternGraph(nodes=['a', 'b'])
    graph.add_undirected_edge(('a', 'b'))
    assert graph.has_arrowhead(('a', 'b')) == False

    graph.add_marked_arrowhead(('a', 'b'))

    assert graph.has_arrowhead(('a', 'b')) == True
    assert graph.has_marked_arrowhead(('a', 'b')) == True

def test_has_arrowhead_with_unmarked_arrowhead():
    graph = MarkedPatternGraph(nodes=['a', 'b'])
    graph.add_undirected_edge(('a', 'b'))
    assert graph.has_arrowhead(('a', 'b')) == False

    graph.add_arrowhead(('a', 'b'))

    assert graph.has_arrowhead(('a', 'b')) == True

def test_has_marked_path():
    graph = MarkedPatternGraph(nodes=['a', 'b'])
    graph.add_undirected_edge(('a', 'b'))
    graph.add_marked_arrowhead(('a', 'b'))

    assert graph.has_marked_arrowhead(('a', 'b')) == True

    graph.add_undirected_edge(('b', 'c'))

    assert graph.has_marked_path(('a', 'c')) == False

    graph.add_marked_arrowhead(('b', 'c'))

    assert graph.has_marked_path(('a', 'c')) == True

    assert graph.get_nodes_of_edges() == set({'a', 'b', 'c'})

def test_has_marked_path_longer():
    #   a -*> b -*> c -*> d
    #    \               /
    #      \           /
    #       \        /
    #         ------
    graph = MarkedPatternGraph(
        nodes=['a', 'b', 'c', 'd']
    )
    graph.add_undirected_edge(('a', 'b'))
    graph.add_undirected_edge(('b', 'c'))
    graph.add_undirected_edge(('c', 'd'))
    graph.add_undirected_edge(('a', 'd'))

    graph.add_marked_arrowhead(('a', 'b'))
    graph.add_marked_arrowhead(('b', 'c'))
    graph.add_marked_arrowhead(('c', 'd'))

    assert graph.has_marked_path(('a', 'd')) == True

