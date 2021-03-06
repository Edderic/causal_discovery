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