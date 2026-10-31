"""
run_lhs.py - LHS candidate generation, feasibility filtering, dual-finalist prediction, conformal intervals and ranking
=====================================================================================================================
Run from the project directory (the one that contains run_models.py, perla_cv.py, ... and perla_final_filtered.pkl):

    python run_lhs.py --data perla_final_filtered.pkl --out /kaggle/working/perovskite_outputs/lhs \
        --reference-uq-dir results_uq/uq_seed0_split42_9777e691 \
        --reference-prep-cache results_uq/_prep_cache/prep_923a07d6d1d4.joblib

    python run_lhs.py --skip-models                 # design space + LHS + feasibility + support only (minutes, no model training)
    python run_lhs.py --quick --n 2000              # PRELIMINARY smoke test with tiny models; never use for science
    python run_lhs.py --shap                        # + SHAP on the consensus candidates (slow: random forests)

Nothing is trained on candidates, no PCE / Voc / Jsc / FF is read when candidates are built, and the TEST split is not touched unless
--score-test is given.  Every output row is a model proposal, not an experimentally validated recommendation.
"""
from __future__ import annotations

import argparse
import json
import logging
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import lhs_design_space as DS
import lhs_distance as L
import lhs_feasibility as F
import lhs_rank as R
import lhs_sampler as S
from lhs_config import DIRECT_ID, FINALISTS, SHORT, TWOSTAGE_ID, LHSConfig

log = logging.getLogger("lhs")


def versions() -> dict:
    v = dict(python=platform.python_version(), platform=platform.platform())
    for m in ("numpy", "pandas", "scipy", "sklearn", "xgboost", "mapie", "shap", "matminer", "pymatgen", "joblib"):
        try:
            v[m] = __import__(m).__version__
        except Exception:
            v[m] = None
    return v


class Timer:
    def __init__(self):
        self.t, self.times = time.time(), {}

    def lap(self, name):
        now = time.time()
        self.times[name] = round(now - self.t, 1)
        self.t = now


# =========================================================================================
# DIAGNOSTICS
# =========================================================================================
def category_frequencies(space, raw, feasible_ids, eligible_ids, consensus_ids) -> pd.DataFrame:
    rows = []
    tg = space.train_genome
    for var, col, trcol in (("architecture", "cell_architecture", "arch"), ("composition_family", "composition_family", "family"),
                            ("substrate", "substrate", "substrate"), ("atmosphere", "atmosphere", "atmosphere"),
                            ("process_recipe", "process_recipe", None), ("antisolvent_medium", "antisolvent_medium", None),
                            ("etl_recipe", "etl_recipe", None), ("htl_recipe", "htl_recipe", None), ("bc_recipe", "bc_recipe", None)):
        r = raw.set_index("candidate_id")
        sets = {"raw": r[col], "feasible": r.loc[feasible_ids, col], "eligible": r.loc[eligible_ids, col], "consensus": r.loc[consensus_ids, col]}
        tab = pd.DataFrame({k: v.value_counts(normalize=True) for k, v in sets.items()}).fillna(0.0)
        if trcol:
            tab["train_row_share"] = tg[trcol].value_counts(normalize=True).reindex(tab.index).fillna(0.0)
        tab.insert(0, "variable", var)
        rows.append(tab.reset_index().rename(columns={"index": "category", col: "category"}))
    return pd.concat(rows, ignore_index=True)


def distribution_comparison(space, raw, feasible_ids, eligible_ids) -> pd.DataFrame:
    from scipy.stats import kstest
    rows = []
    r = raw.set_index("candidate_id")
    tg = space.train_genome
    for col, tcol in (("anneal_temperature_C", "anneal_T"), ("anneal_time_min", None), ("perovskite_thickness_nm", "pvk_thick"),
                      ("etl_thickness_nm", "etl_thick"), ("htl_thickness_nm", "htl_thick"), ("bc_thickness_nm", "bc_thick"),
                      ("frac_A_Cs", "frac_Cs"), ("frac_A_FA", "frac_FA"), ("frac_C_Br", "frac_Br")):
        tr = (10 ** tg["anneal_logt"] if tcol is None else tg[tcol]).dropna()
        for name, v in (("raw", r[col]), ("feasible", r.loc[feasible_ids, col]), ("eligible", r.loc[eligible_ids, col]), ("train_observed", tr)):
            v = v.dropna()
            q = v.quantile([0.05, 0.25, 0.5, 0.75, 0.95]) if len(v) else pd.Series([np.nan] * 5, index=[.05, .25, .5, .75, .95])
            rows.append(dict(variable=col, set=name, n=len(v), missing_share=float(1 - len(v) / max(len(r) if name == "raw" else (len(feasible_ids) if name == "feasible" else (len(eligible_ids) if name == "eligible" else len(tg))), 1)),
                             q05=q[.05], q25=q[.25], median=q[.5], q75=q[.75], q95=q[.95]))
    return pd.DataFrame(rows)


def post_filter_coverage(raw, feasible_ids, eligible_ids, bins: int = 20) -> pd.DataFrame:
    """Share of 20 equal-width unit-cube bins still populated after filtering (continuous LHS dimensions), separate from the exact
    stratification of the unfiltered sample."""
    r = raw.set_index("candidate_id")
    rows = []
    for c in [c for c in raw.columns if c.startswith("u_")]:
        for name, ids in (("raw", r.index), ("feasible", feasible_ids), ("eligible", eligible_ids)):
            b = np.minimum((r.loc[ids, c].to_numpy() * bins).astype(int), bins - 1)
            cnt = np.bincount(b, minlength=bins)
            rows.append(dict(dimension=c, set=name, n=len(ids), bins=bins, bins_populated=int((cnt > 0).sum()), min_count=int(cnt.min()), max_count=int(cnt.max()),
                             cv_of_counts=float(cnt.std() / cnt.mean()) if cnt.mean() else np.nan))
    return pd.DataFrame(rows)


def write_report(path, summary, cfg, funnel, ranked_head) -> None:
    f = "| stage | remaining | removed |\n|---|---|---|\n" + "\n".join(f"| {r['stage']} | {r['remaining']} | {r['removed']} |" for r in funnel)
    L_ = ["# LHS candidate generation - diagnostic report", "",
          f"Run: seed {cfg.seed}, N = {cfg.n_candidates}, quick models = {cfg.quick}.  **All candidates below are model proposals, not validated recipes.**", "",
          "## Filtering funnel", "", f, "",
          "## Reconstruction of the finalists", "", f"```json\n{json.dumps(summary.get('reconstruction', {}), indent=2, default=str)}\n```", "",
          "## Held-out interval coverage (paper-disjoint split of VALIDATION)", "",
          f"```json\n{json.dumps({k: {kk: vv for kk, vv in v.items() if kk not in ('repeats', 'coverage_by_true_pce_bin_repeat0')} for k, v in summary.get('conformal_development_check', {}).items()}, indent=2, default=str)}\n```", "",
          "## Extrapolation / support concerns", "", f"```json\n{json.dumps(summary.get('support', {}), indent=2, default=str)}\n```", "",
          "## Prediction diagnostics", "", f"```json\n{json.dumps(summary.get('predictions', {}), indent=2, default=str)}\n```", "",
          "## Limitations that affect scientific claims", ""] + [f"- {x}" for x in summary.get("limitations", [])]
    Path(path).write_text("\n".join(L_))


LIMITATIONS = [
    "Predictions are in-distribution-trained regressions (R2 ~0.38, RMSE ~3.3 PCE points in cross-validation): the model explains a minority of the variance, and the reported RMSE gap between the finalists (0.005 +- 0.010) is within noise.",
    "The finalists are reconstructed by refitting (nothing was serialised); check `reconstruction` for whether this environment reproduced the reported run.",
    "Conformal intervals are marginal for exchangeable real devices; LHS candidates are synthetic, and candidates selected for high predicted PCE / lower bound are subject to winner's-curse bias, so nominal coverage does not apply to the ranked lists.",
    "Interval width is only weakly informative about per-device error; Stage-1 error of the two-stage model is not propagated separately.",
    "The design space is restricted to one-step spin-coated Pb perovskites (Cs/FA/MA, I/Br) on glass with observed recipe levels; multi-step anneals, other processes, tin / lead-free, plastic substrates and unseen materials are NOT explored.",
    "Reporting artefacts: training rows with reported thickness / band gap are a non-random subset; candidates set thickness explicitly (missing flag 0) and leave band gap missing (flag 1), so predictions refer to those sub-populations.",
    "Annealing time / temperature and thickness bounds, the MA thermal rule and the lambda of score C are proposals needing expert review; sensitivity to lambda and the support threshold is reported.",
    "Category probabilities (p ~ papers**gamma) encode a prior about what is worth exploring; very rare recipes are excluded by min_papers_category.",
    "Model agreement is not validation; both finalists share data, preprocessing and likely systematic errors. SHAP / group Shapley describe the models, not causal effects.",
]


# =========================================================================================
def run(cfg: LHSConfig, skip_models: bool = False) -> dict:
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    h = logging.FileHandler(out / "lhs_run.log", mode="w")
    h.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    log.addHandler(h)
    log.setLevel(logging.INFO)
    if not log.handlers or not any(isinstance(x, logging.StreamHandler) and not isinstance(x, logging.FileHandler) for x in log.handlers):
        log.addHandler(logging.StreamHandler(sys.stdout))
    np.random.seed(cfg.seed)
    T = Timer()
    summary = dict(config_seed=cfg.seed, n_requested=cfg.n_candidates, versions=versions(), skip_models=skip_models, quick=cfg.quick,
                   validation_checks={}, limitations=LIMITATIONS)

    # ---- 1. design space (TRAIN only) -------------------------------------------------------------------------------------
    from perla_cv import check_three_way_split, load_and_split
    split = load_and_split(cfg.data_path, random_state=cfg.split_seed)
    check_three_way_split(split)
    space = DS.build_design_space(split.train_df, cfg)
    assert not (set(space.train_genome["paper"]) & set(split.g_val.astype(str))), "design space touched validation papers"
    assert not (set(space.train_genome["paper"]) & set(split.g_test.astype(str))), "design space touched test papers"
    log.info(f"design space from {space.n_train_rows} TRAIN rows / {space.n_train_papers} papers: {len(space.families)} families, "
             f"{len(space.layers['etl'])}/{len(space.layers['htl'])}/{len(space.layers['bc'])} ETL/HTL/BC recipes, {len(space.processes)} process recipes")
    T.lap("design_space")

    # ---- 2. LHS sampling ----------------------------------------------------------------------------------------------------
    g, U, raw = S.sample_candidates(space, cfg)
    strat = S.stratification_report(U)
    assert raw["candidate_id"].is_unique, "candidate ids are not unique"
    summary["validation_checks"].update(candidate_ids_unique=True, lhs_exactly_stratified_all_dims=bool(strat["exact_lhs"].all()))
    rnd = S.rounding_effect(U, g, space)
    S.check_frame_schema(S.genome_to_perla_frame(g.iloc[:50].reset_index(drop=True), space))
    summary["validation_checks"]["raw_frame_schema_ok"] = True
    raw.to_csv(out / "lhs_raw_candidates.csv", index=False)
    strat.to_csv(out / "lhs_stratification.csv", index=False)
    rnd.to_csv(out / "lhs_rounding_effect.csv", index=False)
    log.info(f"sampled {len(raw)} raw candidates; exact LHS strata in all dimensions: {bool(strat['exact_lhs'].all())}")
    T.lap("sampling")

    # ---- 3. feasibility -------------------------------------------------------------------------------------------------------
    feas_g, rej_g, rules, funnel = F.filter_candidates(g, space, cfg)
    for fr in funnel:
        log.info(f"funnel: {fr}")
    rule_counts = rules[[r.name for r in F.RULES]].sum().rename("n_violating").to_frame()
    rule_counts["severity"] = [r.severity for r in F.RULES]
    rule_counts["why"] = [r.why for r in F.RULES]
    rule_counts.to_csv(out / "lhs_rule_violation_counts.csv")
    rej = raw.set_index("candidate_id").loc[rej_g["candidate_id"]].reset_index()
    rej["status"], rej["rejection_reasons"] = rej_g["status"].to_numpy(), rej_g["reasons"].to_numpy()
    rej.to_csv(out / "lhs_rejected_candidates.csv", index=False)
    T.lap("feasibility")

    # ---- 4. support (genome distance to real devices) --------------------------------------------------------------------------
    sm = L.SupportModel(space, cfg.support_k, cfg.support_reference_rows, cfg.seed)
    kf = S.genome_key_frame(feas_g, space)
    score = sm.score(kf)
    pct = sm.percentile(score)
    support = pd.DataFrame({"support_score_knn_gower": score, "support_percentile": pct, "out_of_support": pct > cfg.max_support_percentile})
    rules_f = rules.loc[feas_g.index].reset_index(drop=True)
    raw_f = raw.set_index("candidate_id").loc[feas_g["candidate_id"]].reset_index()
    feas_out = raw_f.copy()
    feas_out["is_uncommon"], feas_out["uncommon_flags"] = rules_f["is_uncommon"].to_numpy(), rules_f["uncommon_flags"].to_numpy()
    feas_out["tolerance_factor"] = rules_f["tolerance_factor"].to_numpy()
    for c in support.columns:
        feas_out[c] = support[c].to_numpy()
    feas_out.to_csv(out / "lhs_feasible_candidates.csv", index=False)
    summary["support"] = dict(reference_rows=int(len(sm.reference)), reference_quantiles={k: float(np.quantile(sm.reference, k)) for k in (.5, .9, .95, .99)},
                              n_feasible=int(len(feas_g)), uncommon_share=float(rules_f["is_uncommon"].mean()),
                              out_of_support_share=float(support["out_of_support"].mean()), max_support_percentile=cfg.max_support_percentile,
                              unspecified_thickness_share={l: float(raw_f[f"{l}_thickness_nm"].isna().mean()) for l in ("etl", "htl", "bc")})
    summary["funnel"] = funnel + [dict(stage="inside support threshold (eligible for ranking, before model checks)", remaining=int((~support["out_of_support"]).sum()), removed=int(support["out_of_support"].sum()))]
    T.lap("support")
    DS_tables = dict(bounds=DS.bounds_table(space), audit=DS.schema_audit(space))
    DS_tables["bounds"].to_csv(out / "lhs_feature_bounds.csv", index=False)
    DS_tables["audit"].to_csv(out / "lhs_input_schema_audit.csv", index=False)
    space.observed_stats.to_csv(out / "lhs_observed_feature_stats.csv", index=False)
    cfg_json = dict(config=cfg.to_dict(), dims=S.DIMS, categories=DS.categories_dict(space),
                    continuous=[c.__dict__ for c in space.continuous], versions=versions(),
                    sampling_note="scipy.stats.qmc.LatinHypercube(d=18, scramble=%s, optimization=%s, seed=%d); categorical columns = inverse CDF of tempered category probabilities" % (cfg.lhs_scramble, cfg.lhs_optimization, cfg.seed),
                    reported_run=cfg.reported_run_dir)
    (out / "lhs_sampling_config.json").write_text(json.dumps(cfg_json, indent=2, default=str))

    if skip_models:
        summary["note"] = "--skip-models: no predictions were made"
        (out / "lhs_summary.json").write_text(json.dumps(summary, indent=2, default=str))
        log.info("skip-models: stopping after the feasibility / support stage")
        return summary

    # ---- 5. models: reconstruct, encode, predict, conformalise ------------------------------------------------------------------
    import lhs_predict as P
    import lhs_uncertainty as Q
    art = P.load_artifacts(cfg, log.info)
    T.lap("preprocessing_fit")
    frame = S.genome_to_perla_frame(feas_g.reset_index(drop=True), space)
    S.check_frame_schema(frame)
    X, imputed = P.encode_in_batches(art, frame, cfg.batch_size, log.info)
    summary["validation_checks"].update(schema_exact=True, no_target_columns_in_candidates=True, representable_in_encoder_vocabulary=True,
                                        encoded_columns_equal_training_columns=True)
    summary["encoding"] = dict(n_features=int(X.shape[1]), cells_imputed_by_knn_per_column=imputed)
    T.lap("encoding")
    fins = P.build_finalists(art, log.info)
    T.lap("finalist_fit")
    preds, dev, fid = {}, {}, {}
    for fin in fins:
        Q.calibrate(fin, art, cfg, log.info)
        preds[fin.model_id] = Q.predict_with_intervals(fin, X, cfg.batch_size, log.info)
        fid[fin.model_id] = Q.fidelity_check(fin, art, cfg)
        dev[fin.model_id] = Q.development_check(fin, art, cfg, log.info)
    summary["validation_checks"]["intervals_lower_le_upper"] = True
    summary["validation_checks"]["predictions_finite"] = bool(all(np.isfinite(p[["point", "lower", "upper"]].to_numpy()).all() for p in preds.values()))
    stage1 = P.stage1_predictions(art, X)
    summary["validation_checks"]["twostage_uses_stage1_predictions_only"] = bool(
        np.allclose(art.bank.features("TwoStage", X[:200])[:, -3:], stage1.to_numpy()[:200], atol=1e-6) and not ({"Voc", "Jsc", "FF"} & set(frame.columns)))
    T.lap("prediction")
    summary["reconstruction"] = dict(cache_comparison=art.cache_comparison, fidelity_vs_reported_uq_run=fid,
                                     note="models are refitted, not loaded; see fidelity entries", fit_seconds=round(art.fit_seconds, 1))
    summary["conformal_development_check"] = dev
    summary["conformal_limitations"] = Q.LIMITATIONS
    summary["conformal_intervals_candidates"] = {SHORT[k]: Q.interval_summary(v) for k, v in preds.items()}
    if cfg.score_test and not cfg.quick:
        Xt = P.perla_cv._transform_with(art.interp, art.encoder, art.imputer, art.split.test_df, art.X_train_df.columns).to_numpy(np.float32)
        summary["final_test_coverage"] = {f.model_id: Q.final_test(f, art, cfg, Xt, art.split.T_test["PCE"].to_numpy(float), art.split.g_test.to_numpy().astype(str)) for f in fins}

    # ---- 6. tables, ranking ----------------------------------------------------------------------------------------------------------
    t = R.prediction_table(raw_f, rules_f, support, preds, stage1, cfg)
    t.to_csv(out / "lhs_candidate_predictions.csv", index=False)
    summary["predictions"] = R.prediction_diagnostics(t)
    for s, name in (("direct", "lhs_ranked_direct.csv"), ("twostage", "lhs_ranked_twostage.csv")):
        R.rank_model(t, s, cfg).to_csv(out / name, index=False)
    coder = L.GenomeCoder(space)
    coder.encode(space.train_genome)                       # same category codes as the support model
    cons, cdiag = R.consensus(t, feas_g.reset_index(drop=True), space, coder, cfg)
    cons.to_csv(out / "lhs_consensus_candidates.csv", index=False)
    sens = R.sensitivity(t, cfg)
    sens.to_csv(out / "lhs_score_sensitivity.csv", index=False)
    summary["consensus"] = cdiag
    summary["sensitivity_default_vs_alternatives"] = dict(min_jaccard=float(sens["jaccard_vs_default"].min()), median_jaccard=float(sens["jaccard_vs_default"].median()),
                                                           note="Jaccard overlap of the top-K sets under alternative (lambda, support threshold) vs the default")
    ids = {"feasible": feas_g["candidate_id"], "eligible": t.loc[t["both_eligible"] | t["direct_eligible"] | t["twostage_eligible"], "candidate_id"]}
    category_frequencies(space, raw, ids["feasible"], ids["eligible"], cons["candidate_id"]).to_csv(out / "lhs_category_frequencies.csv", index=False)
    distribution_comparison(space, raw, ids["feasible"], ids["eligible"]).to_csv(out / "lhs_distribution_comparison.csv", index=False)
    post_filter_coverage(raw, ids["feasible"], ids["eligible"]).to_csv(out / "lhs_post_filter_coverage.csv", index=False)
    if len(cons):
        pos = {c: i for i, c in enumerate(t["candidate_id"])}
        Xc = X[[pos[c] for c in cons["candidate_id"]]]
        pd.DataFrame(Xc, columns=art.feature_names).assign(candidate_id=cons["candidate_id"].to_numpy()).to_csv(out / "lhs_model_input_consensus.csv", index=False)
    T.lap("ranking")

    # ---- 7. optional SHAP --------------------------------------------------------------------------------------------------------
    if cfg.shap and len(cons):
        import lhs_explain as EX
        summary["shap"] = EX.run_explanations(art, fins, Xc, cons["candidate_id"].tolist(), out / "shap", cfg, log.info)
        T.lap("shap")
    summary["runtime_seconds"] = T.times
    (out / "lhs_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    write_report(out / "lhs_diagnostics.md", summary, cfg, summary["funnel"], t.head())
    log.info(f"done. outputs in {out.resolve()}")
    return summary


def parse_args(argv=None) -> tuple[LHSConfig, argparse.Namespace]:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", help="JSON file with LHSConfig fields (overridden by explicit flags)")
    ap.add_argument("--data"), ap.add_argument("--out")
    ap.add_argument("--n", type=int, help="number of raw LHS candidates (default 50000)")
    ap.add_argument("--seed", type=int), ap.add_argument("--split-seed", type=int)
    ap.add_argument("--reference-uq-dir"), ap.add_argument("--reference-prep-cache")
    ap.add_argument("--gamma", type=float, help="category tempering exponent")
    ap.add_argument("--n-jobs", type=int), ap.add_argument("--batch-size", type=int)
    ap.add_argument("--quick", action="store_true"), ap.add_argument("--skip-models", action="store_true")
    ap.add_argument("--shap", action="store_true"), ap.add_argument("--score-test", action="store_true")
    ap.add_argument("--lambda", dest="lam", type=float), ap.add_argument("--max-support-percentile", type=float)
    ap.add_argument("--top-k", type=int), ap.add_argument("--consensus-k", type=int), ap.add_argument("--diversity-min-distance", type=float)
    a = ap.parse_args(argv)
    cfg = LHSConfig.from_json(a.config) if a.config else LHSConfig()
    m = dict(data=("data_path", a.data), out=("out_dir", a.out), n=("n_candidates", a.n), seed=("seed", a.seed), split_seed=("split_seed", a.split_seed),
             reference_uq_dir=("reference_uq_dir", a.reference_uq_dir), reference_prep_cache=("reference_prep_cache", a.reference_prep_cache),
             gamma=("tempering_gamma", a.gamma), n_jobs=("n_jobs", a.n_jobs), batch_size=("batch_size", a.batch_size),
             lam=("score_lambda", a.lam), msp=("max_support_percentile", a.max_support_percentile), top_k=("top_k", a.top_k),
             consensus_k=("consensus_k", a.consensus_k), dmin=("diversity_min_distance", a.diversity_min_distance))
    for _, (field, val) in m.items():
        if val is not None:
            setattr(cfg, field, val)
    cfg.quick, cfg.shap, cfg.score_test = cfg.quick or a.quick, cfg.shap or a.shap, cfg.score_test or a.score_test
    if cfg.score_test and cfg.quick:
        raise SystemExit("--score-test is refused for a preliminary (--quick) run")
    return cfg, a


if __name__ == "__main__":
    cfg, args = parse_args()
    run(cfg, skip_models=args.skip_models)
