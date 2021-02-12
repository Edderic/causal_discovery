import pytest
from graphs.marked_pattern_graph import MarkedPatternGraph
from constraint_based.immoralities_finder import ImmoralitiesFinder

def test_simple_chain():
    graph = MarkedPatternGraph(
        nodes=['X', 'Y', 'Z'],
        undirected_edges=[
            set({'X', 'Y'}),
            set({'Y', 'Z'})
        ]
    )

    cond_sets = {
        'X _||_ Z': [ set({'Y'}) ],
        'X _||_ Y': [],
        'Y _||_ Z': [],
    }

    unmarked_arrows = ImmoralitiesFinder(
        marked_pattern_graph=graph,
        cond_sets=cond_sets
    ).find()

    assert set(unmarked_arrows) == set({})

def test_simple_collider():
    graph = MarkedPatternGraph(
        nodes=['Parent 1', 'Parent 2', 'collider'],
        undirected_edges=[
            set({'Parent 1', 'collider'}),
            set({'Parent 2', 'collider'})
        ]
    )

    cond_sets = {
        'Parent 1 _||_ Parent 2': [ set({}) ],
    }

    unmarked_arrows = ImmoralitiesFinder(
        marked_pattern_graph=graph,
        cond_sets=cond_sets
    ).find()

    assert set(unmarked_arrows) == set(
        {
            ('Parent 1', 'collider'),
            ('Parent 2', 'collider')
        }
    )

def test_firing_squad_example():
    graph = MarkedPatternGraph(
        nodes=[
            'Captain ordered to shoot',
            'Rifleman 1 shot',
            'Rifleman 2 shot',
            'Prisoner hit by bullet',
            'Prisoner dead'
        ],
        undirected_edges=[
            set({'Rifleman 1 shot', 'Prisoner hit by bullet'}),
            set({'Rifleman 2 shot', 'Prisoner hit by bullet'}),
            set({'Captain ordered to shoot', 'Rifleman 1 shot'}),
            set({'Captain ordered to shoot', 'Rifleman 2 shot'}),
            set({'Prisoner hit by bullet', 'Prisoner dead'})
        ]
    )

    cond_sets = {
        'Captain ordered to shoot _||_ Prisoner hit by bullet': [
            set({'Rifleman 1 shot'}),
            set({'Rifleman 2 shot'}),
            set({'Rifleman 1 shot', 'Rifleman 2 shot'})
        ],
        'Rifleman 1 shot _||_ Rifleman 2 shot': [set({'Captain ordered to shoot'})],
        'Prisoner dead _||_ Rifleman 1 shot': [set({'Prisoner hit by bullet'})],
        'Prisoner dead _||_ Rifleman 2 shot': [set({'Prisoner hit by bullet'})]
    }

    unmarked_arrows = ImmoralitiesFinder(
        marked_pattern_graph=graph,
        cond_sets=cond_sets
    ).find()

    assert set(unmarked_arrows) == set(
        {
            ('Rifleman 1 shot', 'Prisoner hit by bullet'),
            ('Rifleman 2 shot', 'Prisoner hit by bullet')
        }
    )
