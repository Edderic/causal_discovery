"""
    MVICStar: Missing Value Inductive Causation Star
    ------------------------------------------------

    A module to help with causal discovery tasks

    Classes:
        - SkeletonFinder

    Functions:
        - conditioning_sets_satisfying_conditional_independence
"""
from .skeleton_finder import SkeletonFinder
from .direct_causes_of_missingness_finder import DirectCausesOfMissingnessFinder
from .potentially_extraneous_edges_finder import PotentiallyExtraneousEdgesFinder
from .removable_edges_finder import RemovableEdgesFinder
from .immoralities_finder import ImmoralitiesFinder
from .density_ratio_weighted_correction import DensityRatioWeightedCorrection
from .recursive_edge_orienter import RecursiveEdgeOrienter
from ..graphs.marked_pattern_graph import MarkedPatternGraph

class MVICStar(object):
    def __init__(
        self,
        data,
        missingness_indicator_prefix='MI_'
    ):
        self.data = data.copy()
        self.orig_columns = data.columns
        self.missingness_indicator_prefix = missingness_indicator_prefix

    def predict(self):
        skeleton_finder = SkeletonFinder(
            var_names=self.orig_columns,
            data=self.data
        )

        graph, cond_sets_satisfying_cond_indep = skeleton_finder.find()

        marked_arrows = DirectCausesOfMissingnessFinder(
            data=self.data
        ).find()

        graph.add_marked_arrows(marked_arrows)

        potentially_extraneous_edges = PotentiallyExtraneousEdgesFinder(
            marked_pattern_graph=graph,
            missingness_indicator_prefix=self.missingness_indicator_prefix
        ).find()

        edges_to_remove = RemovableEdgesFinder(
            data=self.data,
            data_correction=DensityRatioWeightedCorrection,
            potentially_extraneous_edges=potentially_extraneous_edges,
            cond_sets_satisfying_cond_indep=cond_sets_satisfying_cond_indep,
            marked_pattern_graph=graph
        ).find()

        graph.remove_undirected_edges(edges_to_remove)

        immoralities = ImmoralitiesFinder(
            marked_pattern_graph=graph,
            cond_sets_satisfying_cond_indep=cond_sets_satisfying_cond_indep
        ).find()

        graph.add_arrowheads(immoralities)

        RecursiveEdgeOrienter(marked_pattern_graph=graph).orient()

        return graph

