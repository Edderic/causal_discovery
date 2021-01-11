from ..graphs.marked_pattern_graph import get_nodes_from_edges
from ..graphs.marked_pattern_graph import get_nodes_adj_to_node
from .misc import key_for_pair

class ImmoralitiesFinder(object):
    """
        Finds immoralities. An immorality consists of two edges made up of
        three nodes. Two nodes are non-adjacent, and the third node is adjacent
        with the other two nodes.

        Parameters:
            marked_pattern_graph: graphs.MarkedPatternGraph
    """
    def __init__(self, marked_pattern_graph, cond_sets_satisfying_cond_indep):
        self.marked_pattern_graph = marked_pattern_graph
        self.cond_sets_satisfying_cond_indep = cond_sets_satisfying_cond_indep

    def find(self):
        undirected_nodes = \
            get_nodes_from_edges(self.marked_pattern_graph.undirected_edges)

        unmarked_arrows = set({})

        for node_1 in undirected_nodes:
            for node_2 in undirected_nodes:
                if node_1 == node_2:
                    continue

                nodes_adj_to_node_1 = get_nodes_adj_to_node(
                    edges=self.marked_pattern_graph.undirected_edges,
                    node=node_1
                )

                nodes_adj_to_node_2 = get_nodes_adj_to_node(
                    edges=self.marked_pattern_graph.undirected_edges,
                    node=node_2,
                )

                # if node_1 and node_2 are adjacent, continue.
                if set({node_1, node_2}).intersection(nodes_adj_to_node_1.union(nodes_adj_to_node_2)) != set({}):
                    continue

                common_adjacent_nodes = \
                    nodes_adj_to_node_1\
                    .intersection(nodes_adj_to_node_2)

                _common_adjacent_nodes = list(common_adjacent_nodes)

                for common_adj_node in _common_adjacent_nodes:
                    try:
                        if self.\
                                _get_cond_set_vars_for_pair(
                                    pair=[node_1, node_2]
                                ).intersection(set({common_adj_node})) == set({}):
                            unmarked_arrows = unmarked_arrows.union(set({(node_1, common_adj_node)}))
                            unmarked_arrows = unmarked_arrows.union(set({(node_2, common_adj_node)}))

                    except KeyError:
                        continue

        return unmarked_arrows

    def _get_cond_set_vars_for_pair(self, pair):
        cond_sets = self.cond_sets_satisfying_cond_indep[
            key_for_pair(pair)
        ]

        return get_nodes_from_edges(cond_sets)


