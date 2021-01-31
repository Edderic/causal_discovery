from constraint_based.misc import key_for_pair

class ImmoralitiesFinder(object):
    """
        Finds immoralities. An immorality consists of two edges made up of
        three nodes. Two nodes are non-adjacent, and the third node is adjacent
        with the other two nodes.

        Parameters:
            marked_pattern_graph: graphs.MarkedPatternGraph
            cond_sets_satisfying_cond_indep: dict
                key: str
                    Two variables that are conditionally independent.
                    Ex: "A _||_ B", which stands for A is independent of B
                values: list of sets.
                    Conditioning sets that make the two pairs of variables
                    independent.
                    Ex: set({}) for the empty set
                    Ex: set({'C'})
    """
    def __init__(self, marked_pattern_graph, cond_sets_satisfying_cond_indep):
        self.marked_pattern_graph = marked_pattern_graph
        self.cond_sets_satisfying_cond_indep = cond_sets_satisfying_cond_indep

    def find(self):
        undirected_edges = self.marked_pattern_graph.get_undirected_edges()
        undirected_nodes = \
            self.marked_pattern_graph.get_nodes_of_edges(undirected_edges)

        unmarked_arrows = set({})


        for node_1 in undirected_nodes:
            for node_2 in undirected_nodes:
                if node_1 == node_2:
                    continue

                edges_for_getting_nodes_adj_to_node = undirected_edges

                common_neighbors = \
                    self.marked_pattern_graph.get_common_neighbors(
                        node_1=node_1,
                        node_2=node_2
                    )

                for common_adj_node in common_neighbors:
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

        return self\
                .marked_pattern_graph\
                .get_nodes_of_edges(cond_sets)


