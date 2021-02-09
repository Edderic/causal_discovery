
"""
    MVPCStar: Missing Value PC Star
    -------------------------------
"""
# from constraint_based.skeleton_finder import SkeletonFinder
from constraint_based.pc_skeleton_finder import PCSkeletonFinder
from constraint_based.ci_tests.bmd_is_independent import bmd_is_independent
from constraint_based.direct_causes_of_missingness_finder import DirectCausesOfMissingnessFinder
from constraint_based.potentially_extraneous_edges_finder import PotentiallyExtraneousEdgesFinder
from constraint_based.removable_edges_finder import RemovableEdgesFinder
from constraint_based.immoralities_finder import ImmoralitiesFinder
from constraint_based.density_ratio_weighted_correction import DensityRatioWeightedCorrection
from constraint_based.recursive_edge_orienter import RecursiveEdgeOrienter
from graphs.marked_pattern_graph import MarkedPatternGraph
from information_theory import conditional_mutual_information
from constraint_based.misc import setup_logging

class MVPCStar(object):
    """
        Missing Value Inductive Causation (*), or MVIC* for short. This helps
        with finding the causal structure (i.e. how variables are causally
        related, e.g. Is A a genuine cause of B? Is B a potential cause of C?
        Is D spuriously related to E?

        This combines two things:
            1. the IC* algorithm.
            2. adjustment for missing data.

        The original IC* algorithm (listed in Causality (Pearl, 2009) book)
        assumed that data had no missingness. Running naive causal discovery
        algorithms, without adjusting for missing data, could lead to
        extraneous edges. It has been modified to account for missingness, by
        making use of ideas from the [Causal Discovery in the Presence of
        Missing Data (Tu et. al,
        2019)](http://proceedings.mlr.press/v89/tu19a/tu19a.pdf). Using the
        ideas from the paper, we are able to remove the extraneous edges
        mentioned above. By combining IC* with the ideas from the paper above,
        we relax the causal sufficiency assumption a little by only having it
        apply to missingness indicators (i.e. all causes of missingness are (at
        least partially) observed).

        Parameters:
            data: pd.DataFrame
                Columns are variables, and rows are instances. May have missing
                data, denoted as NaN.

        Returns: graphs.marked_pattern_graph.MarkedPatternGraph

            A Marked Pattern represents a set of DAGs (Pearl, 2009). It has
            four types of arrows:

            1. marked arrow a -*-> b, signifying a directed path from a to b in
               the underlying model (i.e. A genuinely causes B).

            2. an unmarked arrow a ---> b, signifying a directed path from a to
               b, or some latent common cause a <- L -> b in the underlying
               model (i.e. A potentially causes B).

            3. a bidirected edge a <--> b signifying some latent common cause
               a <- L -> b in the underlying model (i.e. A and B are spuriously
               related); and

            4. an undirected edge a ---- b, signifying either a -> b, a <- b,
               or a <- L -> b in the underlying model.

        Assumptions:

        - Causal Markov:
          A variable is independent of non-descendants given its parents. This
          implies that the independencies in the graph are a subset of the
          independencies in the probability distribution (i.e. P factorizes
          according to G).

              Let's say the true DAG is the following:

                             D  -> E
                               \
                                 v
                  G -> A -> B -> C
                                  \
                                   v
                                   H

                         F

              B's parents: { A }
              B's ancestors: { G, A }

              B's children: { C }
              B's descendants: { C, H }

              All other vars: { D, E, F }

              If we know B's parents (A), then B is independent of G (an
              ancestor of B) along with D, E, F.

              In other words B _||_ G, D, E, F | A.

              More generally:
                  Var _||_ Non-Desc(Var) - Parents(Var) | Parents(Var)

              If this condition doesn't hold, then this is a violation of said
              condition.

        - Faithfulness/Stability:
          Any independence in P can be represented by the graph G.

          Faithfulness assumption, along with the Causal Markov assumption,
          imply that any pairwise independencies we find in the probability
          distribution, we can also represent in the graph (i.e. I(P) = I(G)).
          When this is violated, we may get wonky results.

            Example of a stable distribution:
                Let the true DAG G be the chain X -> Y -> Z, where X, Y, and Z
                have a linear relationship.

                i.e. X := U_x       #
                     Y := X + U_y   # Y *isn't* deterministically related to X
                                    # because of the presence of the y error
                                    # term

                     Z := Y + U_z   # Z is Y plus some error term. U_x, U_y,
                                    # and U_z are independent


            Violation example / Example of an unstable distribution:
                Let the true DAG G be the chain X -> Y -> Z, where X, Y, and Z
                have a linear relationship. Let X and Y be deterministic
                relationships of each other, while Z listens to Y (but is not
                determinstically related to Y):

                i.e. X := U_x       #
                     Y := X         # Y is assigned the value of X, without
                                    # error (i.e. Y and X are deterministic)

                     Z := Y + U_z   # Z is Y plus some error term. U_x and U_z
                                    # are independent

                There is only one independence statement implied by the true
                DAG. The DAG implies that X is independent of Z given Y, since
                this is a chain. In other words:

                    I(G) = { X _||_ Z | Y }

                On the other hand, the independence statements in the
                probability distribution says that in addition to X and Z being
                independent given Y, we also have Y is independent of Z given X.
                In other words:

                    I(P) = { X _||_ Z | Y; Y _||_ Z | X }

                There are more pairwise independencies listed in the
                probability distribution P than in G, so P is an unstable
                distribution. Why is this bad from a causal discovery
                perspective? Let's see.

                The algorithm's first stage involves learning the skeleton, a
                set of undirected edges. An undirected edge represents that two
                variables are directly related, relative to other variables we
                have access to. An undirected edge exists between two variables
                if no conditioning set that satisfies the pairwise conditional
                independence exists.

                If we are to use a stable distribution (such as when X and Y
                are not deterministically related), then I(P) = I(G):

                    I(G) = I(P) = { X _||_ Z | Y }

                the SkeletonFinder would be able to have the correct edges:

                     X - Y - Z

                However, if we don't use a stable distribution, like when
                deterministic relationships exist like in the example above, we
                would get the following graph, where the expected edge Y-Z is
                missing:

                     X - Y   Z

                The skeleton right above satisfies the (conditional)
                independence statements in P. There are no edges between X and
                Z because X is independent of Z given Y (i.e. X _||_ Z | Y) in
                the probability distribution P. Likewise, there's no edge
                between Y and Z because Y and Z are independent of each other
                given X (i.e. Y _||_ Z | X). However, there should be an edge
                between Y and Z because Y causes Z. This is problematic because
                later steps won't add the missing edge Y-Z. Therefore, we are
                not able to recover the true DAG.

                When we assume a distribution is faithful / stable, we are
                saying that there are no accidental independencies present in
                the data.

        - Causal Sufficiency with respect to Missing Indicators.
            When we say Causal Sufficiency with respect to Missingness
            Indicators, we are saying that all causes of missingness are at
            least partially observed. For example, if the true DAG has two
            causes X and Z for the missingness of Y, (i.e. X -> MI_Y <- Z),
            then X and Z must be partially observed (not completely missing) in
            the data.

            However, the algorithm allows for causal sufficiency with respect
            to all other variables.

            For example, let's say that during some pandemic, we find an
            association between people who wear pajamas during work and people
            who don't. Let's say the true relationship between the three
            variables is that the Pandemic is a common cause of wearing Pajamas
            and Depression. More people work from home due to the pandemic, and
            while they work from home, they are also more likely to be
            depressed, because of the pandemic:


                      - Pandemic -
                    /             \
                   /               \
                  v                 v
                Pajamas         Depression

            However, let's say we did not collect values on the Pandemic
            variable (i.e. it's unobserved). The algorithm here should in
            theory not discount the possibility of an unobserved confounder
            (i.e. it won't put a marked arrow between the two: no Pajamas -*>
            Depression or no Depression -*> Pajamas).

        - Missingness indicators are not causes of substantive (observed)
          variables.
            A missingness indicator for a variable tells us which parts of the
            variable are observed, and which are missing.

            A substantive variable would be a variable we have (at least
            partial) measurements for. For example, Gender.

            Violation example:
                Let's say the true DAG has the following subgraph:

                MI_SES -> Gender

        - No self-masking missingness.
            This is a scenario where a variable causes its own missingness.

            Violation example:
                For example, in a survey, rich people might be less likely to
                report their income than non-rich people. Part of the graph
                true DAG will have:

                SES -> MI_SES

                Socioeconomic status causes itself to be missing.

        - No causal interactions between missingness indicators.
            A missingness indicator for a variable tells us which parts of the
            variable are observed, and which are missing.

            Violation example:

                Let's say we have missingness indicators for socioeconomic
                status (MI_SES) and gender (MI_G) and the true DAG has this
                subgraph:

                    MI_SES -> MI_G

        - Faithful Observability.
            Any conditional independence relation in the observed data also
            holds in the unobserved data.

        See [Causal Discovery in the Presence of Missing
        Data](http://proceedings.mlr.press/v89/tu19a/tu19a.pdf) for more
        details for the last 4 assumptions.

        Returns: graphs.marked_pattern_graph.MarkedPatternGraph

    """
    def __init__(
        self,
        data,
        cond_indep_test=bmd_is_independent,
        missingness_indicator_prefix='MI_'
    ):
        self.data = data.copy()
        self.orig_columns = data.columns
        self.missingness_indicator_prefix = missingness_indicator_prefix
        self.cond_indep_test=cond_indep_test
        # self.cond_set_num_vars_max=cond_set_num_vars_max

    def predict(self, debug=False):
        logging = setup_logging()

        self.debug_info = []

        logging.info('Finding skeleton...')

        skeleton_finder = PCSkeletonFinder(
            var_names=self.orig_columns,
            data=self.data,
            cond_indep_test=self.cond_indep_test
        )

        graph, cond_sets_satisfying_cond_indep = skeleton_finder.find()

        self.debug_info.append({
            'name': 'after skeleton finding',
            'graph': graph.copy(),
            'cond_sets_satisfying_cond_indep': dict(cond_sets_satisfying_cond_indep.dict)
        })

        logging.info('Done finding skeleton. Now Finding direct causes of missingness...')

        marked_arrows = DirectCausesOfMissingnessFinder(
            data=self.data,
            graph=graph,
            missingness_indicator_prefix=self.missingness_indicator_prefix,
            cond_indep_test=self.cond_indep_test
        ).find()

        graph.add_marked_arrows(marked_arrows)

        self.debug_info.append({
            'name': 'after adding direct causes of missingness',
            'graph': graph.copy()
        })

        logging.info('Done finding direct causes of missingness. Now finding potential edges to remove...')

        potentially_extraneous_edges = PotentiallyExtraneousEdgesFinder(
            marked_pattern_graph=graph,
        ).find()

        logging.info('Done finding potentially extraneous edges to remove. Now attempting to remove edges...')

        edges_to_remove = RemovableEdgesFinder(
            data=self.data,
            data_correction=DensityRatioWeightedCorrection,
            potentially_extraneous_edges=potentially_extraneous_edges,
            cond_sets=cond_sets_satisfying_cond_indep,
            graph=graph,
            missingness_indicator_prefix=self.missingness_indicator_prefix,
            cond_indep_test=self.cond_indep_test
        ).find()

        graph.remove_undirected_edges(edges_to_remove)

        self.debug_info.append({
            'name': 'after removing undirected edges',
            'graph': graph.copy(),
            'cond_sets_satisfying_cond_indep': dict(cond_sets_satisfying_cond_indep.dict)
        })

        logging.info('Done removing extraneous edges. Now finding immoralities...')

        immoralities = ImmoralitiesFinder(
            marked_pattern_graph=graph,
            cond_sets_satisfying_cond_indep=cond_sets_satisfying_cond_indep
        ).find()

        graph.add_arrowheads(immoralities)

        self.debug_info.append({
            'name': 'after adding immoralities',
            'graph': graph.copy()
        })

        logging.info('Done finding immoralities. Now recursively orienting edges...')

        RecursiveEdgeOrienter(marked_pattern_graph=graph).orient()

        self.debug_info.append({
            'name': 'after recursively orienting edges',
            'graph': graph.copy()
        })

        logging.info('Done recursively orienting edges!')

        return graph

