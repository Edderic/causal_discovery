# pylint: disable=missing-module-docstring
# pylint: disable=missing-function-docstring
import pytest # pylint: disable=unused-import
from graphs.partial_ancestral_graph import PartialAncestralGraph, Edge, Node
from errors import NotAncestralError, ArgumentError

def test_get_adjacent_pairs_of_edges():
    pag = PartialAncestralGraph()
    pag.add_edge('A o-> B')
    pag.add_edge('B o-> C')

    results = pag.get_adjacent_pairs_of_edges()
    assert len(results) == 1

    first_result = results[0]

    assert Edge('A o-> B') in first_result
    assert Edge('B o-> C') in first_result

def test_node():
    node = Node('A')
    edge = Edge('A o-> B')

    node.add_edge(edge)

    assert edge in node.get_edges()

    assert ['B'] == list(node.get_neighbors())

    node.remove_edge(Edge('B <-o A'))

    assert node.get_edges() == []

def test_edge_undetermined_of():
    edge = Edge('A o-o B')

    assert edge.undetermined_of('A')
    assert edge.undetermined_of('B')

    with pytest.raises(ArgumentError):
        edge.undetermined_of('non-existent variable')

    edge = Edge('A o-> B')

    assert edge.undetermined_of('A')
    assert not edge.undetermined_of('B')

    edge = Edge('A o-- B')

    assert edge.undetermined_of('A')
    assert not edge.undetermined_of('B')

def test_edge_into_and_out_of():
    edge = Edge('A o-o B')
    assert not edge.into('A')
    edge.set_into('A')
    assert edge.into('A')
    assert edge == Edge('A <-o B')

    edge.set_out_of('B')
    assert edge == Edge('A <-- B')

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

    assert set({'A', 'C'}) == {str(n) for n in graph.get_neighbors('B')}
    assert set({'B', 'C'}) == {str(n) for n in graph.get_neighbors('A')}

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
