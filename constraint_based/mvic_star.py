"""
    MVICStar: Missing Value Inductive Causation Star
    ------------------------------------------------

    A module to help with causal discovery tasks

    Classes:
        - SkeletonFinder

    Functions:
        - conditioning_sets_satisfying_conditional_independence
"""
from constraint_based.skeleton_finder import SkeletonFinder
from constraint_based.direct_causes_of_missingness_finder import DirectCausesOfMissingnessFinder
from constraint_based.potentially_extraneous_edges_finder import PotentiallyExtraneousEdgesFinder
from constraint_based.removable_edges_finder import RemovableEdgesFinder
from constraint_based.immoralities_finder import ImmoralitiesFinder
from constraint_based.density_ratio_weighted_correction import DensityRatioWeightedCorrection
from constraint_based.recursive_edge_orienter import RecursiveEdgeOrienter
from graphs.marked_pattern_graph import MarkedPatternGraph

class MVICStar(object):
    def __init__(
        self,
        data,
        missingness_indicator_prefix='MI_'
    ):
        self.data = data.copy()
        self.orig_columns = data.columns
        self.missingness_indicator_prefix = missingness_indicator_prefix

    def predict(self, debug=False):
        self.debug_info = []

        skeleton_finder = SkeletonFinder(
            var_names=self.orig_columns,
            data=self.data
        )

        graph, cond_sets_satisfying_cond_indep = skeleton_finder.find()

        self.debug_info.append({
            'name': 'after skeleton finding',
            'graph': graph.copy(),
            'cond_sets_satisfying_cond_indep': cond_sets_satisfying_cond_indep
        })

        marked_arrows = DirectCausesOfMissingnessFinder(
            data=self.data,
            missingness_indicator_prefix=self.missingness_indicator_prefix
        ).find()

        graph.add_marked_arrows(marked_arrows)

        self.debug_info.append({
            'name': 'after adding direct causes of missingness',
            'graph': graph.copy()
        })

        potentially_extraneous_edges = PotentiallyExtraneousEdgesFinder(
            marked_pattern_graph=graph,
        ).find()

        edges_to_remove = RemovableEdgesFinder(
            data=self.data,
            data_correction=DensityRatioWeightedCorrection,
            potentially_extraneous_edges=potentially_extraneous_edges,
            cond_sets_satisfying_cond_indep=cond_sets_satisfying_cond_indep,
            marked_pattern_graph=graph,
            missingness_indicator_prefix=self.missingness_indicator_prefix
        ).find()

        graph.remove_undirected_edges(edges_to_remove)

        self.debug_info.append({
            'name': 'after removing undirected edges',
            'graph': graph.copy()
        })

        immoralities = ImmoralitiesFinder(
            marked_pattern_graph=graph,
            cond_sets_satisfying_cond_indep=cond_sets_satisfying_cond_indep
        ).find()

        graph.add_arrowheads(immoralities)

        self.debug_info.append({
            'name': 'after adding immoralities',
            'graph': graph.copy()
        })

        RecursiveEdgeOrienter(marked_pattern_graph=graph).orient()

        self.debug_info.append({
            'name': 'after recursively orienting edges',
            'graph': graph.copy()
        })

        return graph

