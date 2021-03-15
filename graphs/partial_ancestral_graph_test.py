# pylint: disable=missing-module-docstring
# pylint: disable=missing-function-docstring
import pytest # pylint: disable=unused-import
from graphs.partial_ancestral_graph import PartialAncestralGraph, Edge
from errors import NotAncestralError, ArgumentError

def test_edge_into_and_out_of():
    edge = Edge('A o-o B')
    edge.set_into('A')
    assert edge == Edge('A <-o B')

    with pytest.raises(ArgumentError):
        edge.into('non-existent variable')

    with pytest.raises(ArgumentError):
        edge.out_of('non-existent variable')

    edge.set_out_of('B')
    assert edge == Edge('A <-- B')

    with pytest.raises(ArgumentError):
        edge.set_into('non-existent variable')

    with pytest.raises(ArgumentError):
        edge.set_out_of('non-existent variable')

def test_edge():
    edge = Edge('A --> B')
    assert not edge.into('A')
    assert edge.out_of('A')
    assert edge.into('B')
    assert not edge.out_of('B')

    edge = Edge('A <-> B')
    assert not edge.out_of('A')
    assert edge.into('A')
    assert edge.into('B')
    assert not edge.out_of('B')

    edge = Edge('A o-> B')
    assert not edge.out_of('A')
    assert not edge.into('A')
    assert edge.into('B')
    assert not edge.out_of('B')

def test_get_edge():
    graph = PartialAncestralGraph()
    assert graph.get_edge('A', 'B') is None

    graph.add_edge('A o-o B')

    result = graph.get_edge('A', 'B')

    assert result[0] == 'A'
    assert result[1] == 'o-o'
    assert result[2] == 'B'

def test_add_edge():
    graph = PartialAncestralGraph()

    assert graph.has_edge('A o-o B') is False
    assert graph.has_edge('B o-o A') is False

    graph.add_edge('A o-o B')
    assert graph.has_edge('A o-o B') is True
    assert graph.has_edge('B o-o A') is True

    graph = PartialAncestralGraph()

    assert graph.has_edge('A --o B') is False
    assert graph.has_edge('B o-- A') is False
    graph.add_edge('A --o B')
    assert graph.has_edge('B o-- A') is True
    assert graph.has_edge('A --o B') is True

    graph = PartialAncestralGraph()

    assert graph.has_edge('A o-- B') is False
    assert graph.has_edge('B --o A') is False
    graph.add_edge('A o-- B')
    assert graph.has_edge('B --o A') is True
    assert graph.has_edge('A o-- B') is True

    graph = PartialAncestralGraph()

    assert graph.has_edge('A --- B') is False
    assert graph.has_edge('B --- A') is False
    graph.add_edge('A --- B')
    assert graph.has_edge('B --- A') is True
    assert graph.has_edge('A --- B') is True

    graph = PartialAncestralGraph()

    assert graph.has_edge('A o-> B') is False
    assert graph.has_edge('B <-o A') is False
    graph.add_edge('A o-> B')
    assert graph.has_edge('A o-> B') is True
    assert graph.has_edge('B <-o A') is True

    graph = PartialAncestralGraph()

    assert graph.has_edge('A <-> B') is False
    assert graph.has_edge('B <-> A') is False
    graph.add_edge('A <-> B')
    assert graph.has_edge('A <-> B') is True
    assert graph.has_edge('B <-> A') is True

def test_init_complete_graph():
    graph = PartialAncestralGraph(
        complete=True,
        variables=['A', 'B', 'C']
    )

    assert graph.has_edge('A o-o B')
    assert graph.has_edge('C o-o B')
    assert graph.has_edge('A o-o C')

    assert set({'A', 'C'}) == graph.get_neighbors('B')
    assert set({'B', 'C'}) == graph.get_neighbors('A')

    assert Edge('A o-o B') in graph.get_edges()
    assert Edge('A o-o A') not in graph.get_edges()
    assert Edge('B o-o B') not in graph.get_edges()

def test_init_complete_then_add_edge():
    graph = PartialAncestralGraph(
        complete=True,
        variables=['A', 'B', 'C']
    )
    graph.add_edge('E o-> D')
    assert graph.has_edge('E o-> D')
    assert graph.has_edge('D <-o E')

    assert graph.has_edge('A o-o B')
    assert graph.has_edge('C o-o B')
    assert graph.has_edge('A o-o C')

    assert graph.has_edge('D o-o A') is False

def test_remove_edge():
    graph = PartialAncestralGraph()

    graph.add_edge('A o-o B')
    assert graph.has_adjacency(('A', 'B')) is True

    graph.remove_edge(('A', 'B'))
    assert graph.has_adjacency(('A', 'B')) is False

    graph = PartialAncestralGraph()

    graph.add_edge('A <-o B')
    assert graph.has_adjacency(('A', 'B')) is True

    graph.remove_edge(('A', 'B'))
    assert graph.has_adjacency(('A', 'B')) is False

def test_ancestral_validation():
    # No directed cycles

    graph = PartialAncestralGraph()

    graph.add_edge('A --> B')
    graph.add_edge('B --> C')

    with pytest.raises(NotAncestralError):
        graph.add_edge('C --> A')

    graph = PartialAncestralGraph()

    graph.add_edge('C --> A')
    graph.add_edge('B --> C')

    with pytest.raises(NotAncestralError):
        graph.add_edge('A --> B')

    # No almost-directed cycles

    graph = PartialAncestralGraph()

    graph.add_edge('A --> B')
    graph.add_edge('B --> C')

    with pytest.raises(NotAncestralError):
        graph.add_edge('C <-> A')

    graph = PartialAncestralGraph()

    graph.add_edge('A --> B')
    graph.add_edge('C <-> A')

    with pytest.raises(NotAncestralError):
        graph.add_edge('B --> C')
    # Nodes of undirected edges can't have siblings.

    graph = PartialAncestralGraph()

    graph.add_edge('A --- B')

    with pytest.raises(NotAncestralError):
        graph.add_edge('B <-> C')

    graph = PartialAncestralGraph()

    graph.add_edge('B <-> C')

    with pytest.raises(NotAncestralError):
        graph.add_edge('A --- B')
