"""
    immoralities_finder
"""

class ImmoralitiesFinder: #pylint: disable=too-few-public-methods
    """
        Finds immoralities. An immorality consists of two edges made up of
        three nodes. Two nodes are non-adjacent, and the third node is adjacent
        with the other two nodes.

        Parameters:
            graph: a graph object
                Responds to the following methods:
                    - get_edges()
                    - get_neighbors(node)
                    - has_adjacency(tuple_of_nodes)

            sep_sets: constraint_based.misc.sep_sets
                key: str
                    Two variables that are conditionally independent.
                    Ex: "A _||_ B", which stands for A is independent of B
                values: list of sets.
                    Conditioning sets that make the two pairs of variables
                    independent.
                    Ex: set({}) for the empty set
                    Ex: set({'C'})
    """
    def __init__(self, graph, sep_sets):
        self.graph = graph
        self.sep_sets = sep_sets

    def find(self):
        """
            Finds immoralities (unshielded colliders).

            Returns: tuple
                Ex: ('A', 'B', 'C'), where 'A' and 'C' are not
                adjacent, and 'B' is not in any of the separating sets
                between 'A' and 'C'

        Example:
        >>> graph = PAG(
        >>>     variables=['Parent 1', 'Parent 2', 'collider'],
        >>> )
        >>> graph.add_edge('Parent 1 o-o collider')
        >>> graph.add_edge('Parent 2 o-o collider')
        >>> sep_sets = SepSets()
        >>> sep_sets.add(
        >>>     node_1='Parent 1',
        >>>     node_2='Parent_2',
        >>>     cond_set=set({'collider'})
        >>> )
        >>> immoralities = ImmoralitiesFinder(
        >>>     graph=graph,
        >>>     sep_sets=sep_sets
        >>> ).find()
        >>> assert ('Parent 1', 'collider', 'Parent 2') in immoralities
        """
        edges = self.graph.get_edges()
        immoralities = []

        for edge in edges:
            node_2_neighbors = list(self.graph.get_neighbors(edge.node_2) - set({edge.node_1}))
            var_set = set({edge.node_2})

            for node_2_neighbor in node_2_neighbors:
                if not self.graph.has_adjacency(
                        (edge.node_1, node_2_neighbor)
                    ) \
                    and not self.sep_sets.include(
                        some_set=var_set,
                        node_1=edge.node_1,
                        node_2=node_2_neighbor
                    ):

                    immorality = (edge.node_1, edge.node_2, node_2_neighbor)
                    immoralities.append(immorality)

        return immoralities
