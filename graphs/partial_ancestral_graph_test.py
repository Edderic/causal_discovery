# pylint: disable=missing-module-docstring
# pylint: disable=missing-function-docstring
import pytest # pylint: disable=unused-import
from graphs.partial_ancestral_graph import PartialAncestralGraph
from errors import NotAncestralError

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

    assert ('A', 'o-o', 'B') in graph.get_edges()
    assert ('A', 'o-o', 'A') not in graph.get_edges()
    assert ('B', 'o-o', 'B') not in graph.get_edges()

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
