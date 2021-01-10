import pytest
import pandas as pd
from ..graphs.marked_pattern_graph import MarkedPatternGraph
from .potentially_extraneous_edges_finder import PotentiallyExtraneousEdgesFinder

def test_mcar():
    marked_pattern_graph = MarkedPatternGraph(
        nodes=['X', 'Y', 'MI_x']
    )

    potentially_extraneous_edges_finder = PotentiallyExtraneousEdgesFinder(
        data=pd.DataFrame(),
        marked_pattern_graph=marked_pattern_graph
    )

    potentially_extraneous_edges, marked_pattern = potentially_extraneous_edges_finder.find()
    assert potentially_extraneous_edges == []

def test_mar():
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

def test_two_causes_MI_collider():
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

def test_marked_arrow_exists_with_no_MI():
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
        marked_pattern_graph=marked_pattern_graph
    )

    potentially_extraneous_edges = \
        potentially_extraneous_edges_finder.find()

    assert potentially_extraneous_edges == set([])

def test_firing_squad_example():
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
