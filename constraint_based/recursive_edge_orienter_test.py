import pytest
from ..graphs.marked_pattern_graph import MarkedPatternGraph
from .recursive_edge_orienter import RecursiveEdgeOrienter

def test_simple():
    #  a     c
    #   \   /
    #    v v
    #     b
    #     |
    #     d
    graph = MarkedPatternGraph(
        nodes=['a', 'b', 'c', 'd']
    )
    graph.add_undirected_edge(('a', 'b'))
    graph.add_undirected_edge(('c', 'b'))
    graph.add_undirected_edge(('b', 'd'))

    graph.add_arrowhead(('a', 'b'))
    graph.add_arrowhead(('c', 'b'))

    RecursiveEdgeOrienter(
        marked_pattern_graph=graph
    ).orient()

    assert graph.get_marked_arrows() == set({('b', 'd')})

def test_when_marked_path_exists():
    #   a -*> b -*> c
    #    \        /
    #      \    /
    #       \ /
    #
    graph = MarkedPatternGraph(
        nodes=['a', 'b', 'c']
    )
    graph.add_undirected_edge(('a', 'b'))
    graph.add_undirected_edge(('b', 'c'))
    graph.add_undirected_edge(('a', 'c'))

    graph.add_marked_arrowhead(('a', 'b'))
    graph.add_marked_arrowhead(('b', 'c'))

    RecursiveEdgeOrienter(
        marked_pattern_graph=graph
    ).orient()

    assert graph.get_unmarked_arrows() == set({('a', 'c')})

@pytest.mark.focus
def test_longer_marked_path_exists():
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

    RecursiveEdgeOrienter(
        marked_pattern_graph=graph
    ).orient()

    assert graph.get_unmarked_arrows() == set({('a', 'd')})

def test_firing_squad():
    undirected_edges = [
        frozenset(('captain', 'rifle_person_1')),
        frozenset(('captain', 'rifle_person_2')),
        frozenset(('rifle_person_1', 'death')),
        frozenset(('rifle_person_2', 'death')),
    ]

    graph = MarkedPatternGraph(
        nodes=['captain',
               'rifle_person_1',
               'rifle_person_2',
               'prisoner shot',
               'prisoner death']
    )

    graph.add_undirected_edge(('captain', 'rifle_person_1'))
    graph.add_undirected_edge(('captain', 'rifle_person_2'))
    graph.add_undirected_edge(('rifle_person_1', 'prisoner shot'))
    graph.add_undirected_edge(('rifle_person_2', 'prisoner shot'))
    graph.add_undirected_edge(('prisoner shot', 'prisoner death'))

    graph.add_arrowhead(('rifle_person_1', 'prisoner shot'))
    graph.add_arrowhead(('rifle_person_2', 'prisoner shot'))

    RecursiveEdgeOrienter(
        marked_pattern_graph=graph
    ).orient()

    assert graph.get_marked_arrows() == set({('prisoner shot', 'prisoner death')})

