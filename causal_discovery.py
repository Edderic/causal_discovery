from information_theory import sci_is_independent
from itertools import combinations
from viz import MarkedPatternGraph
import numpy as np

class SkeletonFinder():
    """
        Finds the set of undirected edges among nodes.

        Parameters:
            var_names: [list[str]]
                The names of variables that will be in the Skeleton. e.g.
                ["Race", "Weight"]

            data [pandas.DataFrame]
                Contains data. Each column is a variable. Each column name must
                match one and only of the var_names.  e.g. if var_names is
                ["Race", "Weight"], then a valid set of columns for data are:
                ["Race | Black", "Weight"].
    """
    def __init__(self, var_names, data, is_conditionally_independent_func=sci_is_independent):
        self.var_names = var_names
        self.data = data
        self.is_conditionally_independent_func = is_conditionally_independent_func

    def find(self):
        """
            Go through each pair of variables (in var_names).
            For each pair, find a conditioning set that renders the two variables
            independent.
        """
        undirected_edges = []

        for var_name_1, var_name_2 in combinations(self.var_names, 2):
            indep = self._is_independent(
                var_name_1=var_name_1,
                var_name_2=var_name_2
            )

            if not indep:
                undirected_edges.append((var_name_1, var_name_2))

        return MarkedPatternGraph(
                   nodes=self.var_names,
                   undirected_edges=undirected_edges
               )


    def _is_independent(self, var_name_1, var_name_2):
        """
            Goes through possible conditioning sets, including the empty set.
        """

        # TODO: What about the case where the number of variables is 2?
        # TODO: set in-degree (i.e. the number of possible parents that one can
        # have)

        for cond_set_length in np.arange(len(self.var_names) - 1):
            cond_set_combos = combinations(
                list(
                    set(self.var_names) - set([var_name_1, var_name_2])
                ),
                cond_set_length
            )


            for cond_set_combo in cond_set_combos:
                if self.is_conditionally_independent_func(
                       data=self.data,
                       vars_1=[var_name_1],
                       vars_2=[var_name_2],
                       conditioning_set=list(cond_set_combo),
                   ):

                   return True

        return False
