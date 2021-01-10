import pytest
from ..graphs.marked_pattern_graph import MarkedPatternGraph

def test_firing_squad_example():
    graph = MarkedPatternGraph(
        nodes=[
            'Captain ordered to shoot',
            'Rifleman 1 shot',
            'Rifleman 2 shot',
            'Prisoner shot',
            'Prisoner dead'
        ],
        undirected_edges=[
            set({'Captain ordered to shoot', 'Rifleman 1 shot'}),
            set({'Captain ordered to shoot', 'Rifleman 2 shot'}),
            set({'Prisoner shot', 'Prisoner dead'})
        ],
        unmarked_arrows=[
            ('Rifleman 1 shot', 'Prisoner shot'),
            ('Rifleman 2 shot', 'Prisoner shot')
        ]
    )

    IgT

