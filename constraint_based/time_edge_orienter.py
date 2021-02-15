import re

class TimeEdgeOrienter(object):
    """
        Takes into account knowledge about time with respect to orienting
        edges, and applies the idea that the future cannot cause the past.
        i.e. in the real DAG, we can't have X_1 <-- X_2. In the real DAG, if
        there's a direct edge between X_1 and X_2, it must be that X_1 causes
        X_2 (i.e. X_1 --> X_2) or there's a common cause L (i.e. X_1 <-- L -->
        X_2).

        Parameters:
            graph: graph.MarkedPatternGraph
                This is the graph that might be updated after running the
                'orient' method.

                This is an object that responds to the following methods:
                    - get_undirected_edges
                    - get_unmarked_arrows

            regex_match: str. Defaults to '_t=[0-9]+'
                This is used to figure out what the time index of a variable is

        Examples:

            Example 1: We have three variables X_t=1, X_t=2, X_t=3 where
            there's an edge between X_t=1 and X_t=2 and an edge between X_t=2
            and X_t=3. For each edge, we expect that an arrowhead exists to the
            later node (i.e. we should get X_t=1 -> X_t=2 -> X_t=3).

            >>> from graphs.marked_pattern_graph import MarkedPatternGraph
            >>> from constraint_based.time_edge_orienter import TimeEdgeOrienter
            >>> # X_1 -- X_2 -- X_3
            >>> graph = MarkedPatternGraph(
            >>>     nodes=['X_t=1', 'X_t=2', 'X_t=3'],
            >>>     undirected_edges=[('X_t=1', 'X_t=2'), ('X_t=2', 'X_t=3')]
            >>> )
            >>> TimeEdgeOrienter(graph).orient()
            >>> assert graph.get_unmarked_arrows() == set({
            >>>     ('X_t=1', 'X_t=2'), ('X_t=2', 'X_t=3')
            >>> })

            Example 2: We have three variables X_t=1, Y_t=2, and X_t=3. Let's
            say the three form an immorality X_t=1 -> Y_t=2 <- X_t=3. After
            applying the orient command of TimeEdgeOrienter, we should get
            X_t=1 -> Y_t=2 <-> X_t=3 (i.e. there's a common cause between Y_t=2
            and X_t=3).

            >>> from graphs.marked_pattern_graph import MarkedPatternGraph
            >>> from constraint_based.time_edge_orienter import TimeEdgeOrienter
            >>> # X_t=1 --> Y_t=2 <-- X_t=3
            >>> graph = MarkedPatternGraph(
            >>>     nodes=['X_t=1', 'Y_t=2', 'X_t=3'],
            >>>     unmarked_arrows=[
            >>>         ('X_t=1', 'Y_t=2'),
            >>>         ('X_t=3', 'Y_t=2')
            >>>     ]
            >>> )
            >>> TimeEdgeOrienter(graph).orient()
            >>> # Adds an arrowhead from Y_t=2 to X_t=3:
            >>> # X_t=1 --> Y_t=2 <--> X_t=3
            >>> graph = MarkedPatternGraph(
            >>> assert graph.get_unmarked_arrows() == set({
            >>>     ('X_t=1', 'Y_t=2')
            >>> })
            >>> assert graph.get_bidirectional_edges() == set({
            >>>     frozenset({'X_t=3', 'Y_t=2'})
            >>> })
    """
    def __init__(self, graph, regex_match='_t=[0-9]+'):
        self.graph = graph
        self.regex_match = regex_match

    def orient(self):
        undirected_edges = self.graph.get_undirected_edges()
        unmarked_arrows = self.graph.get_unmarked_arrows()

        edges = undirected_edges.union(unmarked_arrows)

        for edge in edges:
            node_1, node_2 = tuple(edge)

            matches_node_1 = self._matches(node_1)
            matches_node_2 = self._matches(node_2)

            if matches_node_1 == None or matches_node_2 == None:
                continue

            if self._get_time_index(matches_node_1.group()) \
                > self._get_time_index(matches_node_2.group()):

                self.graph.add_arrowhead((node_2, node_1))
            elif self._get_time_index(matches_node_1.group()) \
                < self._get_time_index(matches_node_2.group()):

                self.graph.add_arrowhead((node_1, node_2))


    def _matches(self, node):
        return re.search(
            self.regex_match,
            node
        )

    def _get_time_index(self, string):
        return int(re.search('[0-9]+', string).group())
