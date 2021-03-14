# pylint: disable=missing-module-docstring
# pylint: disable=missing-function-docstring
import pytest # pylint: disable=unused-import

from graphs.partial_ancestral_graph import PartialAncestralGraph as PAG
from constraint_based.immoralities_finder import ImmoralitiesFinder
from constraint_based.misc import SepSets

def test_simple_chain():
    graph = PAG(variables=['X', 'Y', 'Z'])

    graph.add_edge('X o-o Y')
    graph.add_edge('Z o-o Y')

    sep_sets = SepSets()
    sep_sets.add(
        node_1='X',
        node_2='Z',
        cond_set=set({'Y'})
    )

    immoralities = ImmoralitiesFinder(
        graph=graph,
        sep_sets=sep_sets
    ).find()

    assert len(immoralities) == 0

def test_simple_collider():
    graph = PAG(
        variables=['Parent 1', 'Parent 2', 'collider'],
    )

    graph.add_edge('Parent 1 o-o collider')
    graph.add_edge('Parent 2 o-o collider')

    sep_sets = SepSets()
    sep_sets.add(
        node_1='Parent 1',
        node_2='Parent_2',
        cond_set=set({'collider'})
    )

    immoralities = ImmoralitiesFinder(
        graph=graph,
        sep_sets=sep_sets
    ).find()

    assert ('Parent 1', 'collider', 'Parent 2') in immoralities

def test_firing_squad_example():
    graph = PAG(
        variables=[
            'Captain ordered to shoot',
            'Rifleman 1 shot',
            'Rifleman 2 shot',
            'Prisoner hit by bullet',
            'Prisoner dead'
        ]
    )

    graph.add_edge('Rifleman 1 shot o-o Prisoner hit by bullet')
    graph.add_edge('Rifleman 2 shot o-o Prisoner hit by bullet')
    graph.add_edge('Captain ordered to shoot o-o Rifleman 1 shot')
    graph.add_edge('Captain ordered to shoot o-o Rifleman 2 shot')
    graph.add_edge('Prisoner hit by bullet o-o Prisoner dead')

    sep_sets = SepSets()
    sep_sets.add(
        node_1='Captain ordered to shoot',
        node_2='Prisoner hit by bullet',
        cond_set=set({'Rifleman 1 shot', 'Rifleman 2 shot'})
    )
    sep_sets.add(
        node_1='Prisoner dead',
        node_2='Rifleman 1 shot',
        cond_set=set({'Prisoner hit by bullet', 'Rifleman 2 shot'})
    )
    sep_sets.add(
        node_1='Prisoner dead',
        node_2='Rifleman 2 shot',
        cond_set=set({'Prisoner hit by bullet', 'Rifleman 1 shot'})
    )
    sep_sets.add(
        node_1='Rifleman 1 shot',
        node_2='Rifleman 2 shot',
        cond_set=set({'Captain ordered to shoot'})
    )

    immoralities = ImmoralitiesFinder(
        graph=graph,
        sep_sets=sep_sets
    ).find()

    assert ('Rifleman 1 shot', 'Prisoner hit by bullet', 'Rifleman 2 shot') \
        in immoralities
