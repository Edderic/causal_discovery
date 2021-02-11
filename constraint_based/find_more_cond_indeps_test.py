import pytest
from constraint_based.pc_skeleton_finder import PCSkeletonFinder
from constraint_based.find_more_cond_indeps import FindMoreCondIndeps
from data import dog_example

def test_dog_example():
    df = dog_example(size=100000)

    skeleton_finder = PCSkeletonFinder(
        data=df
    )

    graph, cond_sets = \
        skeleton_finder.find()

    FindMoreCondIndeps(
        data=df,
        graph=graph,
        cond_sets=cond_sets
    ).find()

    assert len(set({frozenset({'exercise_levels'})})\
        .intersection(
            cond_sets.get('activity', 'dog_tired')
        )) == 1

    assert len(set({frozenset({'best_friends_visit', 'activity'})})\
        .intersection(
            cond_sets.get('weekend', 'mentally_exhausted_before_bed')\
        )) == 1

    assert graph.get_undirected_edges() == frozenset({
        frozenset(('rain', 'best_friends_visit')),
        frozenset(('weekend', 'best_friends_visit')),
        frozenset(('rain', 'activity')),
        frozenset(('exercise_levels', 'best_friends_visit')),
        frozenset(('exercise_levels', 'activity')),
        frozenset(('mentally_exhausted_before_bed', 'activity')),
        frozenset(('exercise_levels', 'dog_tired')),
        frozenset(('best_friends_visit', 'mentally_exhausted_before_bed')),
        frozenset(('mentally_exhausted_before_bed', 'dog_teeth_brushed')),
        frozenset(('dog_tired', 'dog_teeth_brushed')),
    })
