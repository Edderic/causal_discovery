from ..graphs.marked_pattern_graph import MarkedPatternGraph
from ..graphs.marked_pattern_graph import get_common_adj_nodes_between_non_adj_nodes

class RecursiveEdgeOrienter(object):
    def __init__(self, marked_pattern_graph):
        self.marked_pattern_graph = marked_pattern_graph

    def orient(self):
        applied = True

        nodes = self.marked_pattern_graph.get_nodes_of_edges()
        edges = self.marked_pattern_graph.get_edges()

        while (applied):
            applied = False

            for node_1 in nodes:
                for node_2 in nodes:
                    if node_1 == node_2:
                        continue

                    if set({frozenset({node_1, node_2})}).intersection(edges) == set({}):
                        common_adj_nodes = get_common_adj_nodes_between_non_adj_nodes(
                            edges,
                            node_1,
                            node_2
                        )

                        applied = applied or self._apply_rule_1_to(
                            node_1,
                            node_2,
                            common_adj_nodes
                        )
                    else:
                        applied = applied or self._apply_rule_2_to(node_1, node_2)


    def _apply_rule_1_to(self, node_1, node_2, common_adj_nodes):
        applied = False

        for common_adj_node in common_adj_nodes:
            if self.marked_pattern_graph.has_arrowhead(
                    (node_1, common_adj_node)
                ) \
                and not self.marked_pattern_graph.has_arrowhead((node_2, common_adj_node)) \
                and not self.marked_pattern_graph.has_marked_arrowhead((common_adj_node, node_2)):

                self.marked_pattern_graph.add_marked_arrowhead((common_adj_node, node_2))

                applied = applied | True

        return applied

    def _apply_rule_2_to(self, node_1, node_2):
        if (not self.marked_pattern_graph.has_arrowhead((node_1, node_2))) \
            and self.marked_pattern_graph.has_marked_path((node_1, node_2)):
                self.marked_pattern_graph.add_arrowhead((node_1, node_2))


                print('node_1: {}, node_2: {}, dict: {}'.format(node_1, node_2, self.marked_pattern_graph.dict))

                return True

        return False

    def _nodes_of_edges(self):
        if 'nodes_cache' in dir(self):
            return self.nodes_cache

        self.nodes_cache = list(self.marked_pattern_graph.get_nodes_of_edges())

        return self.nodes_cache