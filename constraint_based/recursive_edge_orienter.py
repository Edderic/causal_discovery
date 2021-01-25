from ..graphs.marked_pattern_graph import MarkedPatternGraph
from ..graphs.marked_pattern_graph import get_common_adj_nodes_between_non_adj_nodes

class RecursiveEdgeOrienter(object):
    def __init__(self, marked_pattern_graph):
        self.marked_pattern_graph = marked_pattern_graph

    def orient(self):
        # import pdb; pdb.set_trace()
        while (True):
            result_1 = self._apply_rule_1()
            result_2 = self._apply_rule_2()

            if result_1 == False and result_2 == False:
                break

    def _apply_rule_1(self):
        applied = False

        nodes = self.marked_pattern_graph.get_nodes_of_edges()
        edges = self.marked_pattern_graph.get_edges()

        for node_1 in nodes:
            for node_2 in nodes:
                if node_1 == node_2 or set({node_1, node_2}).intersection(edges) != set({}):
                    continue

                common_adj_nodes = get_common_adj_nodes_between_non_adj_nodes(
                    edges,
                    node_1,
                    node_2
                )

                if node_1 == 'a' and node_2 == 'd':
                    import pdb; pdb.set_trace()

                applied = \
                    applied \
                    or self._apply_rule_1_to(node_1, node_2, common_adj_nodes)

        return applied

    def _apply_rule_2(self):
        applied = False
        edges = list(self.marked_pattern_graph.get_edges())

        for edge in edges:
            node_1, node_2 = tuple(edge)

            result_1 = self._apply_rule_2_to(node_1, node_2)
            result_2 = self._apply_rule_2_to(node_2, node_1)

            applied = applied or result_1 or result_2

        return applied

    def _apply_rule_1_to(self, node_1, node_2, common_adj_nodes):
        # if node_1 == 'a' and node_2 == 'd':
            # import pdb; pdb.set_trace()
        applied = False

        for common_adj_node in common_adj_nodes:
            if self.marked_pattern_graph.has_arrowhead(
                    (node_1, common_adj_node)
                ) \
                and not self.marked_pattern_graph.has_arrowhead((node_2, common_adj_node)) \
                and not self.marked_pattern_graph.has_marked_arrowhead((common_adj_node, node_2)):

                import pdb; pdb.set_trace()

                self.marked_pattern_graph.add_marked_arrowhead((common_adj_node, node_2))

                applied = applied | True

        return applied

    def _apply_rule_2_to(self, node_1, node_2):
        # if (node_1 == 'd' and node_2 == 'c') \
            # or (node_1 == 'b' and node_2 == 'a') \
            # or (node_1 == 'c' and node_2 == 'b') \
            # or (node_1 == 'a' and node_2 == 'd'):
#
            # import pdb; pdb.set_trace()
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
