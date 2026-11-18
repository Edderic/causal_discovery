# Causal discovery with missing data

An independent Python research and learning project by Edderic Ugaddan exploring constraint-based causal discovery: using conditional-independence relationships to learn aspects of a causal graph, including when observations are missing.

The repository contains algorithm implementations, graph representations, synthetic-data fixtures, and pytest tests. Its focus is **causal structure discovery**, rather than estimating a numerical treatment effect or analyzing clinical/claims datasets.

## Implementation map

| Component | Source | Purpose |
| --- | --- | --- |
| Discovery workflow | [`MVPCStar`](constraint_based/mvpc_star.py) | Experimental workflow combining IC*-style reasoning with missing-data adjustments |
| Skeleton discovery | [`PCSkeletonFinder`](constraint_based/pc_skeleton_finder.py), [`SkeletonFinder`](constraint_based/skeleton_finder.py) | Search for separating sets and remove edges using conditional-independence tests |
| Bayesian independence test | [`bmd_is_independent`](constraint_based/ci_tests/bmd_is_independent.py) | Compare Dirichlet posterior samples of categorical distributions across conditioning strata |
| Stochastic-complexity test | [`sci_is_independent`](constraint_based/ci_tests/sci_is_independent.py), [`information_theory.py`](information_theory.py) | Conditional-mutual-information and complexity-based independence testing |
| Missingness modeling | [`DirectCausesOfMissingnessFinder`](constraint_based/direct_causes_of_missingness_finder.py) | Identify candidate causes of missingness indicators |
| Missing-data correction | [`DensityRatioWeightedCorrection`](constraint_based/density_ratio_weighted_correction.py), [`RemovableEdgesFinder`](constraint_based/removable_edges_finder.py) | Explore density-ratio correction and removal of potentially spurious edges |
| Graph orientation | [`ImmoralitiesFinder`](constraint_based/immoralities_finder.py), [`RecursiveEdgeOrienter`](constraint_based/recursive_edge_orienter.py) | Identify unshielded colliders and apply orientation rules |
| Graph representations | [`MarkedPatternGraph`](graphs/marked_pattern_graph.py), [`PartialAncestralGraph`](graphs/partial_ancestral_graph.py) | Represent different edge marks and uncertainty about causal structure |

## How the experimental workflow is organized

`MVPCStar.predict()` orchestrates skeleton discovery, identification of causes of missingness, candidate-edge correction/removal, collider detection, and recursive edge orientation. The implementation draws on IC* and ideas from Tu et al. (2019), rather than being an official implementation of that paper.

The [`conftest.py`](conftest.py) fixtures generate categorical examples involving chains, forks, colliders, latent structure, and missingness mechanisms. Tests sit alongside their implementations as `*_test.py`; useful entry points include [Bayesian independence tests](constraint_based/ci_tests/bmd_is_independent_test.py), [missingness correction tests](constraint_based/density_ratio_weighted_correction_test.py), and [workflow tests](constraint_based/mvpc_star_test.py).

The [`missing-value-ic-star.ipynb`](missing-value-ic-star.ipynb) notebook contains earlier exploratory work, including conditional-independence experiments and algorithm sketches.

## Assumptions and interpretation

Discovery results depend on assumptions about the data-generating process, including the causal Markov condition, faithfulness, and restrictions on missingness. The experimental workflow documents assumptions such as observed/partially observed causes of missingness, no self-masking missingness, and no causal links between missingness indicators. These are substantive assumptions, not properties guaranteed by the code.

An inferred graph can leave directions or latent-confounding possibilities unresolved. It is not, by itself, a treatment-effect estimate or proof of causation in a new dataset. The Bayesian test uses sampled posterior comparisons with a decision threshold; small samples, sparsity, and Monte Carlo variability still matter.

## Environment and current status

[`environment.yml`](environment.yml) records a historical Conda environment, including pandas, NumPy, PyMC3, Theano, pytest, and Graphviz. It is a reference for the original development setup:

```bash
conda env create -f environment.yml
conda activate causal_discovery
```

The checked-in source has unresolved integration issues: imports mix `causal_discovery.*` and top-level module paths; graph modules reference an `errors` module absent from this checkout; the PC skeleton implementation imports Dask/distributed, which are not listed in the environment; and the `MVPCStar` call to `PCSkeletonFinder` does not supply its currently required `graph` argument. The exploratory notebook also references a `conditional_independence` module absent from this checkout.

This is a research prototype, not an installable, production-ready package. End-to-end execution and a passing test suite are not claimed. Dependency, import, and interface reconciliation is needed before running the complete workflow; code modernization is outside this documentation update.

## References

- Pearl, J. (2009). *Causality*, including IC* and marked-pattern representations.
- [Tu et al. (2019), *Causal Discovery in the Presence of Missing Data*](https://proceedings.mlr.press/v89/tu19a.html), referenced by the missing-data workflow.
- Mohan and Pearl (2020), missing-data recoverability and R-factorization, referenced in the density-ratio correction implementation.

## Related work

- [Causal inference presentation](https://github.com/Edderic/causal-inference-presentation): simulated treatment effects, DAGs, g-formula adjustment, and Bayesian measurement-error experiments.
- [Bayesian Python examples for *Causal Inference*](https://github.com/Edderic/causal-inference-python-bayesian): textbook-based regression and posterior uncertainty.
