from causal_discovery.errors import NotComparableError

class MarkedPatternEdge(object):
    def __init__(self, node_1, node_2):
        self.node_1 = node_1
        self.node_2 = node_2

    def hamming_distance(self, other_edge):
        if set({self.node_1, self.node_2}) \
            != set({other_edge.node_1, other_edge.node_2}):
            raise NotComparableError

    def same_order(self, other_edge):
        return self.node_1 == other_edge.node_1 \
            and self.node_2 == other_edge.node_2

class NoEdge(MarkedPatternEdge):
    def __init__(self, node_1, node_2):
        super().__init__(node_1, node_2)

    def hamming_distance(self, other_edge):
        super().hamming_distance(other_edge)

        if isinstance(other_edge, NoEdge):
            return 0
        elif isinstance(other_edge, UndirectedEdge):
            # 1. Add an undirected edge
            return 1
        elif isinstance(other_edge, UnmarkedArrow):
            # 1. Add an undirected edge
            # 2. Add an arrowhead
            return 2
        elif isinstance(other_edge, MarkedArrow):
            # 1. Add an undirected edge
            # 2. Add an arrowhead
            # 3. Mark the arrow
            return 3

class UndirectedEdge(MarkedPatternEdge):
    def __init__(self, node_1, node_2):
        super().__init__(node_1, node_2)

    def hamming_distance(self, other_edge):
        super().hamming_distance(other_edge)

        if isinstance(other_edge, NoEdge):
            # 1. Remove an undirected edge
            return 1
        elif isinstance(other_edge, UndirectedEdge):
            return 0
        elif isinstance(other_edge, UnmarkedArrow):
            # 1. Add an arrowhead
            return 1
        elif isinstance(other_edge, MarkedArrow):
            # 1. Add an arrowhead
            # 2. Mark the arrow
            return 2

class UnmarkedArrow(MarkedPatternEdge):
    def __init__(self, node_1, node_2):
        super().__init__(node_1, node_2)

    def hamming_distance(self, other_edge):
        super().hamming_distance(other_edge)

        if isinstance(other_edge, NoEdge):
            # 1. Remove the only arrowhead
            # 2. Remove the undirected edge
            return 2
        elif isinstance(other_edge, UndirectedEdge):
            # 1. Remove the only arrowhead
            return 1
        elif isinstance(other_edge, UnmarkedArrow) \
            and super().same_order(other_edge):
            return 0
        elif isinstance(other_edge, UnmarkedArrow) \
            and not super().same_order(other_edge):
            # 1. Remove the only arrowhead
            # 2. Add an arrowhead to the other end
            return 2

class MarkedArrow(MarkedPatternEdge):
    def __init__(self, node_1, node_2):
        super().__init__(node_1, node_2)

    def hamming_distance(self, other_edge):
        super().hamming_distance(other_edge)

        if isinstance(other_edge, NoEdge):
            # 1. Unmark
            # 2. Remove the only arrowhead
            # 3. Remove the undirected edge
            return 3
        elif isinstance(other_edge, UndirectedEdge):
            # 1. Unmark
            # 2. Remove the only arrowhead
            return 2
        elif isinstance(other_edge, UnmarkedArrow) \
            and super().same_order(other_edge):
            # 1. Unmark
            return 1
        elif isinstance(other_edge, UnmarkedArrow) \
            and not super().same_order(other_edge):
            # 1. Unmark
            # 2. Remove the arrowhead
            # 3. Add the arrowhead to the other side
            return 3
        elif isinstance(other_edge, MarkedArrow) \
            and super().same_order(other_edge):
            return 0
        elif isinstance(other_edge, MarkedArrow) \
            and not super().same_order(other_edge):
            # 1. Unmark
            # 2. Remove the arrowhead
            # 3. Add the arrowhead to the other side
            # 4. Mark
            return 4
