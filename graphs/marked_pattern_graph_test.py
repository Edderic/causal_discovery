import pytest
from causal_discovery.graphs.marked_pattern.edges import UndirectedEdge, UnmarkedArrow, MarkedArrow, NoEdge
from causal_discovery.graphs.marked_pattern_graph import MarkedPatternGraph

def test_has_path_when_there_are_none():
    graph = MarkedPatternGraph(
        nodes=['x', 'y'],
        marked_arrows=[],
        undirected_edges=[],
        unmarked_arrows=[],
        bidirectional_edges=[]
    )

    assert graph.has_path(('x', 'y')) == False

def test_has_path_when_there_is_one():
    graph = MarkedPatternGraph(
        nodes=['x', 'y'],
        marked_arrows=[],
        undirected_edges=[('x', 'y')],
        unmarked_arrows=[],
        bidirectional_edges=[]
    )

    assert graph.has_path(('x', 'y')) == True

def test_has_path_when_there_is_a_longer_one():
    graph = MarkedPatternGraph(
        nodes=['x', 'y', 'z'],
        marked_arrows=[],
        undirected_edges=[('x', 'z')],
        unmarked_arrows=[('z', 'y')],
        bidirectional_edges=[]
    )

    assert graph.has_path(('x', 'y')) == True

def test_equals_same():
    var_names = ['a', 'b', 'c', 'd', 'e']

    graph_1 = MarkedPatternGraph(
        nodes=var_names,
        marked_arrows=[('c', 'MI_b')],
        undirected_edges=[
            ('a', 'b'),
            ('b', 'c'),
            ('e', 'd'),
            ('d', 'c'),
            ('b', 'd'), # extraneous edge
        ],
        unmarked_arrows=[
            ('a', 'e')
        ],
        bidirectional_edges=[
            ('a', 'b')
        ]
    )

    graph_2 = MarkedPatternGraph(
        nodes=var_names,
        marked_arrows=[('c', 'MI_b')],
        undirected_edges=[
            ('a', 'b'),
            ('b', 'c'),
            ('e', 'd'),
            ('d', 'c'),
            ('b', 'd'), # extraneous edge
        ],
        unmarked_arrows=[
            ('a', 'e')
        ],
        bidirectional_edges=[
            ('a', 'b')
        ]
    )

    assert graph_1 == graph_2

def test_equals_bidirectional_diff():
    var_names = ['a', 'b', 'c', 'd', 'e']

    graph_1 = MarkedPatternGraph(
        nodes=var_names,
        marked_arrows=[('c', 'MI_b')],
        undirected_edges=[
            ('a', 'b'),
            ('b', 'c'),
            ('e', 'd'),
            ('d', 'c'),
            ('b', 'd'),
        ],
        unmarked_arrows=[
            ('a', 'e')
        ],
        bidirectional_edges=[] # diff
    )

    graph_2 = MarkedPatternGraph(
        nodes=var_names,
        marked_arrows=[('c', 'MI_b')],
        undirected_edges=[
            ('a', 'b'),
            ('b', 'c'),
            ('e', 'd'),
            ('d', 'c'),
            ('b', 'd'),
        ],
        unmarked_arrows=[
            ('a', 'e')
        ],
        bidirectional_edges=[
            ('a', 'b')
        ]
    )

    assert graph_1 != graph_2

def test_equals_undirected_edges_diff():
    var_names = ['a', 'b', 'c', 'd', 'e']

    graph_1 = MarkedPatternGraph(
        nodes=var_names,
        marked_arrows=[('c', 'MI_b')],
        undirected_edges=[
            ('a', 'b'),
            ('b', 'c'),
            ('e', 'd'),
            ('d', 'c'),
            ('b', 'd'),
        ],
        unmarked_arrows=[],
        bidirectional_edges=[]
    )

    graph_2 = MarkedPatternGraph(
        nodes=var_names,
        marked_arrows=[('c', 'MI_b')],
        undirected_edges=[
            ('a', 'b'),
            ('b', 'c'),
            ('e', 'd'),
            ('d', 'c'),
        ],
        unmarked_arrows=[],
        bidirectional_edges=[]
    )

    assert graph_1 != graph_2

def test_copy_equals():
    var_names = ['a', 'b', 'c', 'd', 'e']

    graph_1 = MarkedPatternGraph(
        nodes=var_names,
        marked_arrows=[('c', 'MI_b')],
        undirected_edges=[
            ('a', 'b'),
            ('b', 'c'),
            ('e', 'd'),
            ('d', 'c'),
        ],
        unmarked_arrows=[],
        bidirectional_edges=[]
    )

    graph_2 = graph_1.copy()

    assert graph_1 == graph_2

def test_equals_marked_arrows_diff():
    var_names = ['a', 'b', 'c', 'd', 'e']

    graph_1 = MarkedPatternGraph(
        nodes=var_names,
        marked_arrows=[],
        undirected_edges=[
            ('a', 'b'),
            ('b', 'c'),
            ('e', 'd'),
            ('d', 'c'),
        ],
        unmarked_arrows=[],
        bidirectional_edges=[]
    )

    graph_2 = MarkedPatternGraph(
        nodes=var_names,
        marked_arrows=[('c', 'MI_b')],
        undirected_edges=[
            ('a', 'b'),
            ('b', 'c'),
            ('e', 'd'),
            ('d', 'c'),
        ],
        unmarked_arrows=[],
        bidirectional_edges=[]
    )

    assert graph_1 != graph_2

def test_equals_unmarked_arrows_diff():
    var_names = ['a', 'b', 'c', 'd', 'e']

    graph_1 = MarkedPatternGraph(
        nodes=var_names,
        marked_arrows=[('c', 'MI_b')],
        undirected_edges=[
            ('a', 'b'),
            ('b', 'c'),
            ('e', 'd'),
            ('d', 'c'),
            ('b', 'd'),
        ],
        unmarked_arrows=[
            ('a', 'e')
        ],
        bidirectional_edges=[]
    )

    graph_2 = MarkedPatternGraph(
        nodes=var_names,
        marked_arrows=[('c', 'MI_b')],
        undirected_edges=[
            ('a', 'b'),
            ('b', 'c'),
            ('e', 'd'),
            ('d', 'c'),
            ('b', 'd'),
        ],
        unmarked_arrows=[], # unmarked arrows diff
        bidirectional_edges=[]
    )

    assert graph_1 != graph_2

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

    graph.add_marked_arrow(('c', 'd'))

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

    assert graph.has_marked_path(('d', 'c')) == False

def test_simple():
     # a     c
      # \   /
       # v v
        # b
        # |
        # d
    graph = MarkedPatternGraph(
        nodes=['a', 'b', 'c', 'd']
    )
    graph.add_undirected_edge(('a', 'b'))
    graph.add_undirected_edge(('c', 'b'))
    graph.add_undirected_edge(('b', 'd'))

    graph.add_arrowhead(('a', 'b'))
    graph.add_arrowhead(('c', 'b'))

    assert graph.get_edges() == set({
        frozenset({'a', 'b'}),
        frozenset({'b', 'c'}),
        frozenset({'b', 'd'})
    })

def test_get_edge_no_edge():
    graph = MarkedPatternGraph(
        nodes=['a', 'b', 'c', 'd', 'e'],
        undirected_edges=[('a', 'b')],
        bidirectional_edges=[('b', 'c')],
        unmarked_arrows=[('c', 'd')],
        marked_arrows=[('d', 'a')]
    )

    edge = graph.get_edge('d', 'e')
    assert isinstance(edge, NoEdge)

def test_get_edge_undirected_edge():
    graph = MarkedPatternGraph(
        nodes=['a', 'b', 'c', 'd', 'e'],
        undirected_edges=[('a', 'b')],
        bidirectional_edges=[('b', 'c')],
        unmarked_arrows=[('c', 'd')],
        marked_arrows=[('d', 'a')]
    )

    edge = graph.get_edge('a', 'b')
    assert isinstance(edge, UndirectedEdge)


def test_get_edge_unmarked_arrow_edge():
    graph = MarkedPatternGraph(
        nodes=['a', 'b', 'c', 'd', 'e'],
        undirected_edges=[('a', 'b')],
        bidirectional_edges=[('b', 'c')],
        unmarked_arrows=[('c', 'd')],
        marked_arrows=[('d', 'a')]
    )

    edge = graph.get_edge('c', 'd')
    assert isinstance(edge, UnmarkedArrow)

def test_get_edge_marked_arrow_edge():
    graph = MarkedPatternGraph(
        nodes=['a', 'b', 'c', 'd', 'e'],
        undirected_edges=[('a', 'b')],
        bidirectional_edges=[('b', 'c')],
        unmarked_arrows=[('c', 'd')],
        marked_arrows=[('d', 'a')]
    )

    edge = graph.get_edge('d', 'a')
    assert isinstance(edge, MarkedArrow)

@pytest.mark.f
def test_undirected_complete():
    """
    undirected complete graph = true, meaning each pair of
    variables has an undirected edge between thehm
    """


    graph = MarkedPatternGraph(
        nodes=['a', 'b', 'c'],
        undirected_complete=True
    )

    assert isinstance(graph.get_edge('a', 'b'), UndirectedEdge)
    assert isinstance(graph.get_edge('a', 'c'), UndirectedEdge)
    assert isinstance(graph.get_edge('c', 'b'), UndirectedEdge)
