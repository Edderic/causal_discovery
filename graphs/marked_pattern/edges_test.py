import pytest
from graphs.marked_pattern.edges import MarkedPatternEdge
from graphs.marked_pattern.edges import NoEdge, UndirectedEdge, UnmarkedArrow, MarkedArrow
from errors import NotComparableError

def test_hamming_distance_no_edge_vs_no_edge_diff_vars():
    edge = NoEdge('A', 'B')
    # different Edge
    other_edge = NoEdge('C', 'B')

    with pytest.raises(NotComparableError):
        edge.hamming_distance(other_edge)


def test_hamming_distance_no_edge_vs_no_edge_same_vars():
    edge = NoEdge('A', 'B')
    # same Edge
    other_edge = NoEdge('B', 'A')

    assert edge.hamming_distance(other_edge) == 0


def test_hamming_distance_no_edge_vs_undirected_edge_same_vars():
    edge = NoEdge('A', 'B')
    other_edge = UndirectedEdge('B', 'A')

    assert edge.hamming_distance(other_edge) == 1

    edge = UndirectedEdge('A', 'B')
    other_edge = NoEdge('B', 'A')

    assert edge.hamming_distance(other_edge) == 1

def test_hamming_distance_undirected_edge_vs_undirected_edge_same_vars():
    edge = UndirectedEdge('A', 'B')
    other_edge = UndirectedEdge('B', 'A')

    assert edge.hamming_distance(other_edge) == 0

    # Different ordering of variables doesn't matter
    edge = UndirectedEdge('A', 'B')
    other_edge = UndirectedEdge('A', 'B')

    assert edge.hamming_distance(other_edge) == 0

def test_hamming_distance_unmarked_arrow_vs_unmarked_arrow():
    # Different ordering
    edge = UnmarkedArrow('A', 'B')
    other_edge = UnmarkedArrow('B', 'A')

    assert edge.hamming_distance(other_edge) == 2

    # Same Ordering
    edge = UnmarkedArrow('A', 'B')
    other_edge = UnmarkedArrow('A', 'B')

    assert edge.hamming_distance(other_edge) == 0

def test_hamming_distance_unmarked_arrow_vs_no_edge():
    # ordering 1
    edge = UnmarkedArrow('A', 'B')
    other_edge = NoEdge('B', 'A')

    assert edge.hamming_distance(other_edge) == 2

    # ordering 2
    edge = UnmarkedArrow('B', 'A')
    other_edge = NoEdge('B', 'A')
    assert edge.hamming_distance(other_edge) == 2

    # ordering 3
    edge = UnmarkedArrow('B', 'A')
    other_edge = NoEdge('A', 'B')

    assert edge.hamming_distance(other_edge) == 2
    # Same Ordering
    edge = UnmarkedArrow('A', 'B')
    other_edge = NoEdge('A', 'B')

    assert edge.hamming_distance(other_edge) == 2

def test_hamming_distance_no_edge_vs_marked_arrow():
    # ordering 1
    edge = NoEdge('A', 'B')
    other_edge = MarkedArrow('B', 'A')

    assert edge.hamming_distance(other_edge) == 3

    # ordering 2
    edge = NoEdge('B', 'A')
    other_edge = MarkedArrow('B', 'A')
    assert edge.hamming_distance(other_edge) == 3

    # ordering 3
    edge = NoEdge('B', 'A')
    other_edge = MarkedArrow('A', 'B')

    assert edge.hamming_distance(other_edge) == 3
    # Same Ordering
    edge = NoEdge('A', 'B')
    other_edge = MarkedArrow('A', 'B')

    assert edge.hamming_distance(other_edge) == 3

def test_hamming_distance_no_edge_vs_marked_arrow():
    # ordering 1
    edge = UndirectedEdge('A', 'B')
    other_edge = MarkedArrow('B', 'A')

    assert edge.hamming_distance(other_edge) == 2

    # ordering 2
    edge = UndirectedEdge('B', 'A')
    other_edge = MarkedArrow('B', 'A')
    assert edge.hamming_distance(other_edge) == 2

    # ordering 3
    edge = UndirectedEdge('B', 'A')
    other_edge = MarkedArrow('A', 'B')

    assert edge.hamming_distance(other_edge) == 2
    # Same Ordering
    edge = UndirectedEdge('A', 'B')
    other_edge = MarkedArrow('A', 'B')

    assert edge.hamming_distance(other_edge) == 2

def test_hamming_distance_marked_arrow_vs_undirected_edge():
    # ordering 1
    edge = MarkedArrow('A', 'B')
    other_edge = UndirectedEdge('B', 'A')

    assert edge.hamming_distance(other_edge) == 2

    # ordering 2
    edge = MarkedArrow('B', 'A')
    other_edge = UndirectedEdge('B', 'A')
    assert edge.hamming_distance(other_edge) == 2

    # ordering 3
    edge = MarkedArrow('B', 'A')
    other_edge = UndirectedEdge('A', 'B')

    assert edge.hamming_distance(other_edge) == 2
    # Same Ordering
    edge = MarkedArrow('A', 'B')
    other_edge = UndirectedEdge('A', 'B')

    assert edge.hamming_distance(other_edge) == 2

def test_hamming_distance_marked_arrow_vs_unmarked_arrow():
    # ordering 1
    edge = MarkedArrow('A', 'B')
    other_edge = UnmarkedArrow('B', 'A')

    assert edge.hamming_distance(other_edge) == 3

    # ordering 2
    edge = MarkedArrow('B', 'A')
    other_edge = UnmarkedArrow('B', 'A')
    assert edge.hamming_distance(other_edge) == 1

    # ordering 3
    edge = MarkedArrow('B', 'A')
    other_edge = UnmarkedArrow('A', 'B')

    assert edge.hamming_distance(other_edge) == 3
    # Same Ordering
    edge = MarkedArrow('A', 'B')
    other_edge = UnmarkedArrow('A', 'B')

    assert edge.hamming_distance(other_edge) == 1

def test_hamming_distance_marked_arrow_vs_marked_arrow():
    # ordering 1
    edge = MarkedArrow('A', 'B')
    other_edge = MarkedArrow('B', 'A')

    assert edge.hamming_distance(other_edge) == 4

    # ordering 2
    edge = MarkedArrow('B', 'A')
    other_edge = MarkedArrow('B', 'A')
    assert edge.hamming_distance(other_edge) == 0

    # ordering 3
    edge = MarkedArrow('B', 'A')
    other_edge = MarkedArrow('A', 'B')

    assert edge.hamming_distance(other_edge) == 4
    # Same Ordering
    edge = MarkedArrow('A', 'B')
    other_edge = MarkedArrow('A', 'B')

    assert edge.hamming_distance(other_edge) == 0

def test_hamming_distance_marked_arrow_vs_no_edge():
    # ordering 1
    edge = MarkedArrow('A', 'B')
    other_edge = NoEdge('B', 'A')

    assert edge.hamming_distance(other_edge) == 3

    # ordering 2
    edge = MarkedArrow('B', 'A')
    other_edge = NoEdge('B', 'A')
    assert edge.hamming_distance(other_edge) == 3

    # ordering 3
    edge = MarkedArrow('B', 'A')
    other_edge = NoEdge('A', 'B')

    assert edge.hamming_distance(other_edge) == 3
    # Same Ordering
    edge = MarkedArrow('A', 'B')
    other_edge = NoEdge('A', 'B')

    assert edge.hamming_distance(other_edge) == 3

def test_hamming_distance_no_edge_vs_unmarked_arrow():
    # ordering 1
    edge = NoEdge('A', 'B')
    other_edge = UnmarkedArrow('B', 'A')

    assert edge.hamming_distance(other_edge) == 2

    # ordering 2
    edge = NoEdge('B', 'A')
    other_edge = UnmarkedArrow('B', 'A')
    assert edge.hamming_distance(other_edge) == 2

    # ordering 3
    edge = NoEdge('B', 'A')
    other_edge = UnmarkedArrow('A', 'B')

    assert edge.hamming_distance(other_edge) == 2
    # Same Ordering
    edge = NoEdge('A', 'B')
    other_edge = UnmarkedArrow('A', 'B')

    assert edge.hamming_distance(other_edge) == 2

def test_hamming_distance_unmarked_arrow_vs_undirected_edge():
    edge = UnmarkedArrow('A', 'B')
    other_edge = UndirectedEdge('A', 'B')

    assert edge.hamming_distance(other_edge) == 1

    edge = UnmarkedArrow('A', 'B')
    other_edge = UndirectedEdge('B', 'A')

    assert edge.hamming_distance(other_edge) == 1

    edge = UnmarkedArrow('B', 'A')
    other_edge = UndirectedEdge('B', 'A')

    assert edge.hamming_distance(other_edge) == 1

    edge = UnmarkedArrow('B', 'A')
    other_edge = UndirectedEdge('A', 'B')

    assert edge.hamming_distance(other_edge) == 1


def test_hamming_distance_undirected_edge_vs_unmarked_arrow_same_vars():
    edge = UndirectedEdge('A', 'B')
    other_edge = UnmarkedArrow('A', 'B')

    assert edge.hamming_distance(other_edge) == 1

    edge = UndirectedEdge('A', 'B')
    other_edge = UnmarkedArrow('B', 'A')

    assert edge.hamming_distance(other_edge) == 1

    edge = UndirectedEdge('B', 'A')
    other_edge = UnmarkedArrow('B', 'A')

    assert edge.hamming_distance(other_edge) == 1

    edge = UndirectedEdge('B', 'A')
    other_edge = UnmarkedArrow('A', 'B')

    assert edge.hamming_distance(other_edge) == 1
