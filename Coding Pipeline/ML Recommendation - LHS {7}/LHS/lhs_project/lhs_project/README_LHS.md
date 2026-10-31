# LHS candidate generation for the PERLA perovskite-solar-cell models

Every candidate produced by this pipeline is a **model proposal**, not an experimentally validated recommendation.

## 1. What the existing project contains (inspection report)

**Inspected (read in full or in the relevant parts):** `run_models.py`, `perla_cv.py`, `perla_encoding_by_claude.py`, `perla_explain_uq.py`, `perla_mapie.py`, `perla_shap.py`, `test_candidates.py` (synthetic model-comparison tests, nothing to do with LHS candidates), the dataset `perla_final_filtered.pkl` (34,862 x 141), `split_assignment.csv`, `cv_cache/`, `results/5fold_seed0_split42_675a0c14/` (run_config.json, model_selection.json/.md, stage1_diagnostics.csv, oof_predictions.csv, metrics_summary.csv, fold checkpoints) and `results_uq/` (`uq_seed0_split42_9777e691` = MAPIE, `uq_seed0_split42_1ef04462` = SHAP, `_prep_cache`). `test_explain_uq.py` and `test_split_pipeline.py` were only listed, not read.

| # | Question | Finding |
|---|---|---|
| 1 | Exact run directory | `results/5fold_seed0_split42_675a0c14` exists and matches the numbers in the brief. |
| 2 | Input features | The encoder reads **38 raw PERLA columns** (nested token lists, numeric sums, three-state booleans) and produces **306 features** (Direct). The Two-Stage Stage-2 model sees **309** = 306 + `Voc_hat`, `Jsc_hat`, `FF_hat`. Full list: `lhs_input_schema_audit.csv`. |
| 3 | Target | `PCE` in **%** (from `JV_default_PCE`). Stage-1 targets: Voc (V), Jsc (mA/cm2), FF (fraction). |
| 4 | Preprocessing | multi-label token one-hot (min_count 30, plus `__other` / `__missing`), architecture one-hot, 8 three-state booleans (True/False/unknown), numeric sums + missing indicators (>5 % missing), annealing summaries (first/last/max temperature, time sum/max), composition site fractions + Magpie descriptors, out-of-fold target encoding of the anti-solvent medium (`quench__te`), then a KNN imputer. All fitted on TRAIN only. |
| 5 | Loading the models | **Nothing is serialised**: no model, encoder, imputer, SHAP explainer or MAPIE object exists on disk (`_prep_cache` holds only the transformed arrays). `run_models.py` fits the finalists on the 70 % train split but does not save them. The finalists are therefore **reconstructed by refitting** with the same code, hyper-parameters and seeds (`lhs_predict.py`). |
| 6 | Two-stage prediction | Stage 1: one HistGradientBoosting per Voc/Jsc/FF; training features use 5-fold paper-grouped **out-of-fold** predictions, inference uses the fully fitted Stage-1 models. Stage 2: `[X | Voc_hat, Jsc_hat, FF_hat]` -> equal-weight mean of RF, XGBoost, HGB and ExtraTrees (`VotingET`). Direct `VotingOriginal` = the same without ExtraTrees and without Stage 1. |
| 7 | MAPIE | `ConformalizedQuantileRegressor(prefit=True)` over quantile bands (QRF leaf weights for RF/ET, native quantile loss for XGB/HGB), conformalised on the validation split (paper-disjoint from train); end-to-end for the two-stage model. Reference development check: coverage 0.907 (Direct) / 0.903 (Two-Stage) at the 0.90 target, mean width 11.1 / 10.6 PCE points, Spearman(width, abs. error) = 0.13. |
| 8 | Physical / fabrication constraints in existing code | **None.** The existing code only enforces leakage control (paper-grouped splits, no target-derived features). Everything in `lhs_feasibility.py` is new. |

**Direct SHAP was never run** in the reported project (only the Two-Stage finalist, flagged `_INEXACT`: additivity gap 0.0011 vs tolerance 0.001).

### Findings that shaped the design
* **There are no spin-speed, antisolvent-volume, or deposition-duration features.** The genuinely numeric fabrication inputs are: anneal temperature/time (summaries), four layer-thickness sums, number of deposition steps, composition fractions, and cell area. Everything else is categorical.
* **The refit does not bit-reproduce the cached training matrix in a different environment.** In this sandbox (scikit-learn 1.8.0) 31,742 of 7.47 M cells differ: all of `quench__te` (out-of-fold target encoding; `GroupKFold` fold assignment differs between versions) and KNN-imputed cells that use it as a distance coordinate. The pipeline therefore (a) always refits preprocessors and models together so candidates and models are consistent, (b) compares against the cached matrix when `--reference-prep-cache` is given and reports the result, and (c) compares reconstructed point predictions with `mapie_intervals_check.csv` when `--reference-uq-dir` is given.
* `band_gap__mean` and `Perovskite_band_gap_estimation_basis` are film properties / measurement metadata, not fabrication choices. They stay **missing** for candidates (the KNN imputer fills the value exactly as for the 22 % of training rows without a band gap; `__missing` flag = 1).
* `Perovskite_composition_inorganic` / `_leadfree` are exactly derivable from the composition (leadfree: 0 mismatches; inorganic: 9) and are derived, never sampled.
* The anti-solvent flag and medium are tied in the data (flag False <=> medium "Unknown"); candidates keep that semantics.
* Multi-step processes carry one anneal value per step, so a single (T, t) anneal is unrepresentative for them. The default design space is therefore **one-step spin-coated** perovskites (69 % of training rows).

## 2. Architecture

```
perla_final_filtered.pkl --load_and_split (70/15/15 by paper)
   TRAIN rows ---> lhs_design_space  catalogues, bounds, support tables  (TRAIN only; asserted)
                      |
                lhs_sampler   scipy LatinHypercube (18 dims) -> genome -> raw PERLA frame (38 columns)
                      |
                lhs_feasibility   invalid / incompatible (reject) / uncommon (flag)   ->  funnel, reasons
                      |
                lhs_distance      Gower support distance to TRAIN genomes -> out_of_support flag
                      |
                lhs_predict       refit preprocessors+models (existing code) -> encode_frame -> X (float32)
                      |
   lhs_uncertainty   MAPIE CQR calibrated on VALIDATION; held-out coverage; fidelity checks
                      |
                lhs_rank          strategies A/B/C per finalist, consensus + diversity, sensitivity
                lhs_explain       optional: TreeSHAP (existing ShapRunner) + group Shapley of the full pipeline
                run_lhs.py        CLI / orchestration, outputs, summary
```

**Mixed-variable LHS.** One scrambled Latin hypercube `U` (N x 18). Continuous columns are scaled to their bounds (log scale for anneal time) and rounded to a practical step; the unrounded `u_*` columns are saved so stratification stays verifiable. Categorical columns use the inverse CDF of the category probabilities `p_k ~ (distinct papers)^gamma` (gamma = 0.5), so category counts equal `N*p_k` +-1 (stratified quotas). This is **not** space-filling in the categorical sense and not equivalent to continuous LHS. Layer recipes and the substrate are conditional on the architecture. Filtering removes rows, so post-filter coverage is reported separately (`lhs_post_filter_coverage.csv`).

**High cardinality.** A "recipe" (stack + deposition procedure) is an observed combination used as one level; levels need >= 10 distinct papers per architecture (44 ETL, 16 HTL, 10 back-contact recipes; 15 composition families; 15 process recipes). Rare recipes are excluded rather than sampled equally.

## 3. Design space

| Variable | Type | Range / levels | Unit | Source / rationale |
|---|---|---|---|---|
| anneal temperature | continuous | 70-160 | degC | config proposal; observed 5-95 %: 70-150, 61 % of values are exactly 100 |
| anneal time | continuous, log | 2-60 | min | proposal; observed 5-95 %: 3-120 (max 5400 = outliers) |
| perovskite thickness | continuous | 200-650 | nm | proposal; observed 5-95 %: 175-650 (68 % missing) |
| ETL / HTL / back-contact thickness | continuous, recipe-conditional | per recipe 10-90 % quantile of >= 20 observed values, clipped to [5,400]/[5,400]/[20,300] | nm | recipes with too few observations (e.g. screen-printed carbon) keep thickness **unspecified** (missing) |
| Cs / FA / Br fraction | continuous, family-conditional | per family 5-95 % quantile, remainder ion = 1 - sum | fraction | A-sum = 1, C-sum = 3 enforced |
| architecture | categorical | nip, pin | | |
| composition family | categorical | 15 Pb families from {Cs,FA,MA} x {I,Br} with >= 15 papers | | |
| ETL / HTL / back-contact recipe | categorical | 44 / 16 / 10 observed (stack, procedure) pairs, conditional on architecture | | empty HTL excluded (ambiguous) |
| substrate | categorical | SLG\|FTO, SLG\|ITO | | rigid glass only |
| perovskite process | categorical | 15 observed (spin-coating, solvents, 1 step, anti-solvent flag, solvent-anneal flag) | | |
| anti-solvent medium | categorical | 12 (only if flag True) | | generic labels excluded |
| atmosphere | categorical | N2, Air, Ar, Dry air, Ambient | | |
| cell area, flexible, semitransparent, single crystal, graded gap | fixed | 0.10 cm2, False x4 | | context, not recipe choices |
| inorganic, leadfree, steps, aggregation state | derived | | | |
| band gap, band-gap basis | neutralised (missing) | | | see above |
| targets, Voc/Jsc/FF, DOI, year, source | **excluded** | | | `FORBIDDEN_RAW_COLUMNS`; schema check fails loudly |

The **bounds above are proposals** that a perovskite domain expert should review; observed quantiles are descriptive. Full table with observed statistics: `lhs_feature_bounds.csv`.

**Feasibility rules** (each independent, with a recorded reason): *invalid* (non-finite, out of bounds, composition inconsistent, anti-solvent flag/medium mismatch, unknown category, derived-flag mismatch); *incompatible* (MA-majority film annealed > 150 degC; Goldschmidt tolerance factor outside 0.80-1.05; layer used with this architecture in < 10 papers); *uncommon* (6 categorical pairs seen in < 3 papers; flagged, **not** rejected); *out of support* (kNN Gower distance beyond the 95th percentile of held-out-paper TRAIN rows; flagged, excluded from ranking). Not implemented because there is no reliable basis: solvent/anti-solvent compatibility tables, energy-level alignment, material-specific annealing windows.

## 4. Running it

```bash
pip install numpy pandas scipy scikit-learn xgboost mapie shap matminer pymatgen joblib pytest
cd /kaggle/working/psc_project        # or your project directory (contains run_models.py, perla_cv.py, ...)
python run_lhs.py --skip-models       # minutes; design space, LHS, feasibility, support - inspect lhs_feature_bounds.csv first
python run_lhs.py \
   --data perla_final_filtered.pkl --out /kaggle/working/perovskite_outputs/lhs \
   --reference-uq-dir results_uq/uq_seed0_split42_9777e691 \
   --reference-prep-cache results_uq/_prep_cache/prep_923a07d6d1d4.joblib      # full run
python run_lhs.py --shap              # + SHAP on the consensus candidates (random forests: slow)
python run_lhs.py --quick --n 2000    # PRELIMINARY smoke test only
pytest -q test_lhs.py ; LHS_SLOW=1 pytest -q test_lhs.py
```
`--out` defaults to `/kaggle/working/perovskite_outputs/lhs` if `/kaggle/working` exists, otherwise `perovskite_outputs/lhs`. All options: `python run_lhs.py -h`, or `--config my.json` (any `LHSConfig` field). The full run refits both finalists (about 2 and 7 minutes respectively on the Kaggle timings recorded in `mapie_summary.csv`) plus the preprocessors, then predicts ~40k candidates in batches. The TEST split is never read unless `--score-test` is given.

## 5. Outputs

| File | Content |
|---|---|
| `lhs_raw_candidates.csv` | all raw candidates: interpretable recipe, integer category ids, fixed context, unrounded LHS unit samples `u_*` |
| `lhs_feasible_candidates.csv` | feasible candidates + `is_uncommon`, `uncommon_flags`, `tolerance_factor`, `support_score_knn_gower`, `support_percentile`, `out_of_support` |
| `lhs_rejected_candidates.csv` | rejected rows with `status` and `rejection_reasons` |
| `lhs_candidate_predictions.csv` | per feasible candidate: `{direct,twostage}_{pce,lower,upper,width,confidence,eligible,nonfinite,pce_out_of_plausible_range,point_outside_interval}`, `twostage_{Voc,Jsc,FF}_hat` (predicted), `twostage_VocJscFF_product_pct`, `model_abs_difference`, support/feasibility flags, `status` |
| `lhs_ranked_direct.csv`, `lhs_ranked_twostage.csv` | top-K eligible per finalist with `score_A/B/C`, `rank_A/B/C` |
| `lhs_consensus_candidates.csv` | diverse consensus list (`consensus_score`, `consensus_rank`) |
| `lhs_model_input_consensus.csv` | the 306 encoded model inputs of the consensus candidates (audit) |
| `lhs_feature_bounds.csv`, `lhs_input_schema_audit.csv`, `lhs_observed_feature_stats.csv` | bounds, units, source, rationale; 38-column input audit; descriptive statistics |
| `lhs_sampling_config.json` | seed, N, dimension order, full config, category tables with ids/counts/probabilities, per-recipe thickness bounds, software versions |
| `lhs_summary.json` | funnel counts, rule counts, runtimes, prediction statistics, coverage checks, reconstruction/fidelity results, validation checks, limitations |
| `lhs_diagnostics.md` + `lhs_category_frequencies.csv`, `lhs_distribution_comparison.csv`, `lhs_post_filter_coverage.csv`, `lhs_stratification.csv`, `lhs_rounding_effect.csv`, `lhs_rule_violation_counts.csv`, `lhs_score_sensitivity.csv` | diagnostics |
| `shap/` (with `--shap`) | final-stage TreeSHAP, Stage-1 SHAP, group-level Shapley of the complete pipeline |

## 6. Ranking

Per finalist, among eligible candidates (feasible, finite, plausible PCE 0-30, inside support): **A** `mu`; **B** calibrated lower bound; **C** `mu - lambda * half_width`, lambda = 0.5 (midpoint of A and B; there is no empirical basis for a specific value). Consensus = `min(score_C_direct, score_C_twostage)` over candidates eligible for both, thinned greedily so that every kept pair has Gower distance >= 0.15 (categorical mismatch 0/1, continuous differences scaled by design range, equal weight per variable). `lhs_score_sensitivity.csv` varies lambda (0-1) and the support threshold (0.90/0.95/0.99) and reports top-K Jaccard overlap and rank correlation.

## 7. Tests

`test_lhs.py` (37 fast + 2 slow tests): exact 38-column schema; forbidden columns; train-only design space; seed reproducibility; exact LHS stratification before filtering and separate post-filter coverage; bounds/rounding; valid categories, explicit booleans, quench semantics; marginal quotas; architecture-conditional layers; missing-value handling; ID uniqueness and genome round trip through the real `genome_keys`; every feasibility rule triggers with a recorded reason; uncommon is flagged not rejected; funnel consistency; tolerance factor; Gower properties; support; diversity constraint; non-finite / out-of-range predictions flagged and not clipped; interval-column assertions; exact additivity of group Shapley; feature-group coverage; output files. Slow tests: encoding through the fitted pipeline with loud failure for unseen materials and missing booleans, and a synthetic two-stage model proving that Stage 2 receives Stage-1 *predictions* and that MAPIE columns are assigned correctly. A passing suite shows the code does what the tests state; it says nothing about scientific validity.

## 8. Verification status (what was actually run)

* Fast test suite: 37 passed, 2 skipped. Slow tests: 2 passed.
* End-to-end smoke runs with `--quick` models (tiny trees; **preliminary, not scientific**): 1,500 candidates (all integrity checks in `lhs_summary.json` were true) and 800 candidates with SHAP enabled (completed; TreeSHAP additivity gaps ~1e-6, group-Shapley additivity error 4e-15).
* **Not run here:** the full-size run (needs several cores and ~10+ minutes of fitting; this sandbox has 1 CPU / 3 GB), and any comparison with the reported run's saved predictions (that needs the Kaggle environment with matching library versions). The fidelity checks are implemented and will report the result when run.

## 9. Limitations that affect scientific claims

1. The models explain a minority of the variance (R2 ~ 0.38, RMSE ~ 3.3 PCE points). Predicted values are compressed toward the mean (best quick-run candidate ~17 %); use rankings and uncertainty, not absolute PCE.
2. The finalists are statistically indistinguishable (RMSE gap 0.005 +- 0.010); neither is declared superior. Their agreement is not validation.
3. Conformal intervals are marginal for exchangeable real devices. LHS candidates are synthetic, and ranking by predicted PCE or lower bound selects favourable errors (winner's curse), so the nominal 90 % does not apply to ranked candidates. Interval width carries little per-device information (Spearman 0.13). Two-stage Stage-1 error is not propagated separately.
4. In the smoke runs, ~74 % of feasible candidates contained at least one rare categorical pairing and ~44 % were beyond the support threshold. Independent categorical sampling produces many unsupported combinations; rely on the support columns.
5. Reporting artefacts: training rows with reported thickness / band gap are a non-random subset. Candidates set thickness (flag 0) and leave band gap missing (flag 1); predictions refer to those sub-populations. A sensitivity check that toggles these flags is a sensible next step but is not implemented.
6. Scope: one-step spin-coated Pb perovskites on glass. Multi-step anneals, other processes, tin / lead-free, flexible devices, and unseen materials are not explored.
7. Bounds, the MA thermal threshold, the tolerance-factor range, gamma, the support percentile, lambda and the diversity distance are proposals needing expert review.
8. SHAP and group Shapley describe the models, not causal effects of fabrication parameters; low global importance is never used to remove features.
