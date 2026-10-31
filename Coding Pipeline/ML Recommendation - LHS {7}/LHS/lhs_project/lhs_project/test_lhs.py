"""
test_lhs.py - tests of the LHS candidate-generation pipeline

    pytest -q test_lhs.py                 # fast tests (need perla_final_filtered.pkl; ~1 min)
    LHS_SLOW=1 pytest -q test_lhs.py      # + tests that fit the preprocessors (~2 min) and a tiny model bank (~1 min)

A passing run shows that the code does what this file asserts. It does NOT show that the recommended recipes are good or that the
design space is scientifically right: that needs expert review and experiments.
"""
import copy
import json
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

warnings.filterwarnings("ignore")
import lhs_design_space as DS
import lhs_distance as L
import lhs_feasibility as F
import lhs_rank as R
import lhs_sampler as S
import perla_encoding_by_claude as E
from lhs_config import FORBIDDEN_RAW_COLUMNS, LHSConfig

DATA = "perla_final_filtered.pkl"
HAVE_DATA = Path(DATA).exists()
needs_data = pytest.mark.skipif(not HAVE_DATA, reason=f"{DATA} not found (run from the project directory)")
slow = pytest.mark.skipif(not os.environ.get("LHS_SLOW"), reason="set LHS_SLOW=1 to run")
N = 3000


@pytest.fixture(scope="module")
def cfg():
    return LHSConfig(n_candidates=N, out_dir="/tmp/lhs_test_unused")


@pytest.fixture(scope="module")
def split():
    from perla_cv import load_and_split
    return load_and_split(DATA, random_state=42)


@pytest.fixture(scope="module")
def space(split, cfg):
    return DS.build_design_space(split.train_df, cfg)


@pytest.fixture(scope="module")
def sample(space, cfg):
    g, U, raw = S.sample_candidates(space, cfg)
    return g, U, raw


# ------------------------------------------------------------------------------------------------ schema / leakage
@needs_data
def test_exact_input_schema(sample, space):
    g, _, _ = sample
    frame = S.genome_to_perla_frame(g.iloc[:200].reset_index(drop=True), space)
    S.check_frame_schema(frame)                                  # raises on missing / extra / forbidden columns
    assert frame.index.is_unique


@needs_data
@pytest.mark.parametrize("bad", ["Voc", "Jsc", "FF", "JV_default_PCE", "Ref_publication_year"])
def test_no_target_or_measured_output_columns(sample, space, bad):
    g, _, _ = sample
    frame = S.genome_to_perla_frame(g.iloc[:20].reset_index(drop=True), space)
    frame[bad] = 1.0
    with pytest.raises(AssertionError):
        S.check_frame_schema(frame)


@needs_data
def test_forbidden_columns_are_not_model_inputs(space):
    audit = DS.schema_audit(space)
    inputs = set(audit.loc[audit.model_input, "raw_column"])
    assert not inputs & set(FORBIDDEN_RAW_COLUMNS)
    assert not inputs & set(E.TARGET_COLS.values())
    assert len(inputs) == 38


@needs_data
def test_design_space_built_from_train_only(space, split):
    papers = set(space.train_genome["paper"])
    assert not papers & set(split.g_val.astype(str)) and not papers & set(split.g_test.astype(str))
    assert space.n_train_rows == len(split.train_df)


# ------------------------------------------------------------------------------------------------ sampling
@needs_data
def test_reproducible_same_seed_and_different_seed(space, cfg):
    g1, U1, r1 = S.sample_candidates(space, cfg)
    g2, U2, r2 = S.sample_candidates(space, cfg)
    pd.testing.assert_frame_equal(r1, r2)
    cfg2 = copy.deepcopy(cfg)
    cfg2.seed = 7
    _, U3, _ = S.sample_candidates(space, cfg2)
    assert not np.allclose(U1, U3)


@needs_data
def test_lhs_exact_stratification_before_filtering(sample):
    _, U, _ = sample
    rep = S.stratification_report(U)
    assert rep["exact_lhs"].all() and (rep["strata_covered"] == len(U)).all()


@needs_data
def test_post_filter_coverage_reported_separately(sample, space, cfg):
    g, U, raw = sample
    feas, rej, ev, funnel = F.filter_candidates(g, space, cfg)
    s = np.minimum((U[:, S.D["anneal_T"]][feas.index.to_numpy()] * len(U)).astype(int), len(U) - 1)
    # filtering removes rows, so strata are generally NOT all covered any more; this must be visible, not hidden
    assert len(np.unique(s)) <= len(feas) <= len(U)
    from run_lhs import post_filter_coverage
    cov = post_filter_coverage(raw, feas["candidate_id"], feas["candidate_id"])
    assert set(cov["set"]) == {"raw", "feasible", "eligible"}


@needs_data
def test_numeric_bounds_and_rounding(sample, space, cfg):
    g, _, _ = sample
    spec = {c.name: c for c in space.continuous}
    for col, name in (("anneal_T", "anneal_temperature_C"), ("anneal_time_min", "anneal_time_min"), ("pvk_thick", "perovskite_thickness_nm")):
        s = spec[name]
        assert g[col].between(s.lo, s.hi).all()
        assert np.allclose(g[col] / s.round_to, np.round(g[col] / s.round_to))
    for layer in ("etl", "htl", "bc"):
        cat = space.layers[layer]
        lo, hi = cat["thick_lo"].to_numpy()[g[f"{layer}_id"]], cat["thick_hi"].to_numpy()[g[f"{layer}_id"]]
        v = g[f"{layer}_thick"].to_numpy()
        spec_ok = np.isfinite(lo)
        assert ((v[spec_ok] >= lo[spec_ok]) & (v[spec_ok] <= hi[spec_ok])).all()
        assert np.isnan(v[~spec_ok]).all()                       # unspecified recipes stay missing, never invented
    assert ((g[["frac_A_Cs", "frac_A_FA", "frac_A_MA", "frac_C_Br", "frac_C_I"]] >= 0).all()).all()
    assert np.allclose(g[["frac_A_Cs", "frac_A_FA", "frac_A_MA"]].sum(axis=1), 1.0)
    assert np.allclose(g[["frac_C_Br", "frac_C_I"]].sum(axis=1), 1.0)


@needs_data
def test_rounding_effect_reported(sample, space):
    g, U, _ = sample
    r = S.rounding_effect(U, g, space)
    assert (r["max_abs_change_over_range"] < 0.01).all()         # rounding moves values by < 1 % of the range
    assert (r["ks_stat_unrounded_vs_uniform"] < 0.01).all()


@needs_data
def test_categorical_and_boolean_validity(sample, space):
    g, _, _ = sample
    assert g["arch"].isin(["nip", "pin"]).all()
    assert g["substrate"].isin(space.substrates["value"]).all()
    assert g["antisolvent_medium"].isin(set(space.media["value"]) | {"Unknown"}).all()
    assert g["atmosphere"].isin(space.atmospheres["token"]).all()
    for col, n in (("family_id", len(space.families)), ("etl_id", len(space.layers["etl"])), ("htl_id", len(space.layers["htl"])),
                   ("bc_id", len(space.layers["bc"])), ("process_id", len(space.processes))):
        assert g[col].between(0, n - 1).all() and g[col].dtype.kind == "i"       # integer ids: no interpolated codes
    frame = S.genome_to_perla_frame(g.iloc[:300].reset_index(drop=True), space)
    for c in E.BOOLEAN_COLS:
        vals = frame[c]
        assert vals.map(lambda v: isinstance(v, (bool, np.bool_))).all(), c      # explicit True/False, never NA
    # antisolvent flag <=> medium (training-data semantics)
    assert ((frame["Perovskite_deposition_quenching_induced_crystallisation"].astype(bool)) == (frame["Perovskite_deposition_quenching_media"] != "Unknown")).all()


@needs_data
def test_marginal_category_quotas_follow_probabilities(sample, space):
    g, _, _ = sample
    p = space.families["p"].to_numpy()
    counts = g["family_id"].value_counts().reindex(range(len(p))).fillna(0).to_numpy()
    assert np.all(np.abs(counts - len(g) * p) <= 1.0 + 1e-9)       # stratified quota: +-1 of N*p (unconditional dimension)


@needs_data
def test_conditional_layers_respect_architecture(sample, space):
    g, _, _ = sample
    for layer in ("etl", "htl", "bc"):
        for a in ("nip", "pin"):
            ids = g.loc[g["arch"] == a, f"{layer}_id"]
            assert (space.layer_probs[layer][a][ids] > 0).all()


@needs_data
def test_missing_value_handling(sample, space):
    g, _, _ = sample
    frame = S.genome_to_perla_frame(g.iloc[:300].reset_index(drop=True), space)
    assert frame["band_gap__mean"].isna().all() and frame["Perovskite_band_gap_estimation_basis"].isna().all()   # neutralised on purpose
    for c in E.BOOLEAN_COLS:
        assert frame[c].notna().all()                                              # booleans are never missing
    for c in (E.ANNEAL_TEMP_COL, E.ANNEAL_TIME_COL):
        assert frame[c].map(lambda x: np.isfinite(x[0][0][0])).all()


@needs_data
def test_candidate_ids_unique_and_roundtrip(sample, space):
    g, _, raw = sample
    assert raw["candidate_id"].is_unique
    frame = S.genome_to_perla_frame(g.iloc[:500].reset_index(drop=True), space)
    back, ref = DS.genome_keys(frame), S.genome_key_frame(g.iloc[:500].reset_index(drop=True), space)
    for c in DS.CAT_VARS:
        assert (back[c].to_numpy() == ref[c].to_numpy()).all(), c
    for c in DS.CONT_VARS:
        assert np.allclose(back[c].to_numpy(float), ref[c].to_numpy(float), equal_nan=True, atol=1e-6), c


# ------------------------------------------------------------------------------------------------ feasibility
def _bad(g, space, cfg, mut):
    h = g.iloc[:5].copy().reset_index(drop=True)
    mut(h)
    return F.evaluate_rules(h, space, cfg)


@needs_data
@pytest.mark.parametrize("rule,status,mut", [
    ("nonfinite_or_nonpositive_continuous", "invalid", lambda h: h.__setitem__("anneal_T", np.nan)),
    ("continuous_out_of_bounds", "invalid", lambda h: h.__setitem__("anneal_time_min", 9999.0)),
    ("composition_fraction_invalid", "invalid", lambda h: h.__setitem__("frac_A_MA", -0.2)),
    ("quench_inconsistent", "invalid", lambda h: (h.__setitem__("antisolvent", False), h.__setitem__("antisolvent_medium", "Toluene"))),
    ("unknown_category", "invalid", lambda h: h.__setitem__("etl_id", 10 ** 6)),
    ("derived_flag_mismatch", "invalid", lambda h: h.__setitem__("composition_inorganic", ~h["composition_inorganic"])),
    ("ma_thermal_budget", "incompatible", lambda h: (h.__setitem__("frac_A_MA", 1.0), h.__setitem__("frac_A_FA", 0.0), h.__setitem__("frac_A_Cs", 0.0), h.__setitem__("anneal_T", 190.0))),
])
def test_each_rule_triggers_and_records_reason(sample, space, cfg, rule, status, mut):
    g, _, _ = sample
    cfg2 = copy.deepcopy(cfg)
    cfg2.bounds = dict(cfg.bounds, anneal_temperature_C=(70.0, 250.0))      # so only the intended rule is the reason for the thermal case
    ev = _bad(g, space, cfg2, mut)
    assert ev[rule].all()
    assert (ev["status"] == status).all() or status == "incompatible"
    assert ev["reasons"].str.contains(rule).all()
    assert ev["rejected"].all()


@needs_data
def test_layer_unsupported_for_architecture_rule(sample, space, cfg):
    g, _, _ = sample
    pin_unsupported = int(np.where(space.layer_probs["htl"]["pin"] == 0)[0][0])
    ev = _bad(g, space, cfg, lambda h: (h.__setitem__("arch", "pin"), h.__setitem__("htl_id", pin_unsupported)))
    assert ev["layer_unsupported_for_architecture"].all()


@needs_data
def test_uncommon_is_flagged_not_rejected_by_default(sample, space, cfg):
    g, _, _ = sample
    feas, rej, ev, funnel = F.filter_candidates(g, space, cfg)
    assert (ev.loc[feas.index, "is_uncommon"]).any()                     # uncommon candidates survive ...
    assert not ev.loc[feas.index, "rejected"].any()
    cfg2 = copy.deepcopy(cfg)
    cfg2.reject_uncommon = True
    feas2, rej2, ev2, _ = F.filter_candidates(g, space, cfg2)            # ... unless explicitly rejected
    assert len(feas2) < len(feas) and not ev2.loc[feas2.index, "is_uncommon"].any()


@needs_data
def test_funnel_counts_are_consistent(sample, space, cfg):
    g, _, _ = sample
    feas, rej, ev, funnel = F.filter_candidates(g, space, cfg)
    rem = [f["remaining"] for f in funnel]
    assert rem[0] == len(g) and rem == sorted(rem, reverse=True) and rem[-1] == len(feas)
    assert len(feas) + len(rej) == len(g)
    assert (rej["reasons"] != "").all() and rej["candidate_id"].is_unique


@needs_data
def test_tolerance_factor_values_physical(sample):
    g, _, _ = sample
    t = F.tolerance_factor(g)
    assert 0.78 < t.min() and t.max() < 1.05
    assert abs(F.tolerance_factor(pd.DataFrame({"frac_A_Cs": [0.0], "frac_A_FA": [0.0], "frac_A_MA": [1.0], "frac_C_Br": [0.0], "frac_C_I": [1.0]}))[0] - 0.912) < 0.01   # MAPbI3


# ------------------------------------------------------------------------------------------------ distance / support / diversity
def test_gower_properties():
    cat = np.array([[0, 1], [0, 2], [1, 1]], dtype=np.int32)
    x = np.array([[0.0, np.nan], [0.5, 0.2], [1.0, 0.2]], dtype=np.float32)
    D = L.gower_block(cat, x, cat, x)
    assert np.allclose(np.diag(D), 0) and np.allclose(D, D.T) and (D >= 0).all() and (D <= 1).all()
    assert np.isclose(D[0, 1], (0 + 1 + 0.5) / 3)          # missing continuous value on one side is left out of the average


@needs_data
def test_support_percentile_monotone_and_far_candidate(space, sample):
    g, _, _ = sample
    sm = L.SupportModel(space, k=5, reference_rows=300, seed=1)
    kf = S.genome_key_frame(g.iloc[:200], space)
    sc = sm.score(kf)
    assert (sm.percentile(np.sort(sc)) == np.sort(sm.percentile(sc))).all()
    far = kf.copy()
    for c in DS.CAT_VARS:
        far[c] = "never-seen"
    assert sm.percentile(sm.score(far.iloc[:5])).min() >= 0.99


def _fake_predictions(feas, space, cfg, seed=0):
    rng = np.random.RandomState(seed)
    n = len(feas)
    preds = {}
    for mid in cfg.finalists:
        mu = rng.normal(13, 2, n)
        preds[mid] = pd.DataFrame(dict(point=mu, lower=mu - 5.5, upper=mu + 5.5, raw_lower=mu - 5.5, raw_upper=mu + 5.5))
    return preds


@needs_data
def test_diversity_and_distribution_after_filtering(sample, space, cfg):
    g, U, raw = sample
    feas, rej, ev, _ = F.filter_candidates(g, space, cfg)
    feas = feas.iloc[:1500].reset_index(drop=True)
    sm = L.SupportModel(space, k=5, reference_rows=300, seed=1)
    sc = sm.score(S.genome_key_frame(feas, space))
    support = pd.DataFrame({"support_score_knn_gower": sc, "support_percentile": sm.percentile(sc)})
    support["out_of_support"] = support["support_percentile"] > 0.99
    raw_f = raw.set_index("candidate_id").loc[feas["candidate_id"]].reset_index()
    rules_f = ev.set_index("candidate_id").loc[feas["candidate_id"]].reset_index()
    stage1 = pd.DataFrame(np.zeros((len(feas), 3)), columns=["Voc_hat", "Jsc_hat", "FF_hat"])
    t = R.prediction_table(raw_f, rules_f, support, _fake_predictions(feas, space, cfg), stage1, cfg)
    coder = L.GenomeCoder(space)
    coder.encode(space.train_genome)
    c2 = copy.deepcopy(cfg)
    c2.consensus_k, c2.diversity_min_distance = 15, 0.25
    cons, diag = R.consensus(t, feas, space, coder, c2)
    assert 0 < len(cons) <= 15 and diag["min_pairwise_distance"] >= 0.25 - 1e-9
    assert cons["both_eligible"].all() and cons["candidate_id"].is_unique
    assert (cons["status"].str.contains("not experimentally validated")).all()
    # strategies A / B / C all present and ranks are permutations of 1..n
    rk = R.rank_model(t, "direct", c2)
    assert {"score_A", "score_B", "score_C", "rank_A", "rank_B", "rank_C"} <= set(rk.columns)
    assert set(rk["rank_C"]) == set(range(1, len(rk) + 1))
    sens = R.sensitivity(t, c2)
    assert sens["is_default"].sum() == 2 and (sens.loc[sens["is_default"], "jaccard_vs_default"] == 1.0).all()


@needs_data
def test_nonfinite_and_out_of_range_predictions_are_flagged_not_clipped(sample, space, cfg):
    g, U, raw = sample
    feas, rej, ev, _ = F.filter_candidates(g, space, cfg)
    feas = feas.iloc[:50].reset_index(drop=True)
    preds = _fake_predictions(feas, space, cfg)
    p = preds[cfg.finalists[0]]
    p.loc[0, "point"] = np.nan
    p.loc[1, ["point", "lower", "upper"]] = [45.0, 40.0, 50.0]
    support = pd.DataFrame({"support_score_knn_gower": np.zeros(50), "support_percentile": np.zeros(50), "out_of_support": np.zeros(50, bool)})
    raw_f = raw.set_index("candidate_id").loc[feas["candidate_id"]].reset_index()
    rules_f = ev.set_index("candidate_id").loc[feas["candidate_id"]].reset_index()
    t = R.prediction_table(raw_f, rules_f, support, preds, pd.DataFrame(np.zeros((50, 3)), columns=["Voc_hat", "Jsc_hat", "FF_hat"]), cfg)
    assert t.loc[0, "direct_nonfinite"] and not t.loc[0, "direct_eligible"]
    assert t.loc[1, "direct_pce_out_of_plausible_range"] and t.loc[1, "direct_pce"] == 45.0      # flagged, value untouched


# ------------------------------------------------------------------------------------------------ intervals (no model training needed)
def test_interval_column_checks():
    import lhs_uncertainty as Q
    ok = pd.DataFrame(dict(point=[1.0, 2.0], lower=[0.0, 1.0], upper=[2.0, 3.0]))
    Q.check_interval_columns(ok)
    with pytest.raises(ValueError):
        Q.check_interval_columns(ok.rename(columns={"lower": "upper", "upper": "lower"}))          # swapped columns
    with pytest.raises(ValueError):
        Q.check_interval_columns(ok.assign(point=[np.nan, 1.0]))


# ------------------------------------------------------------------------------------------------ SHAP helpers
def test_group_shapley_is_exactly_additive():
    import lhs_explain as EX
    rng = np.random.RandomState(0)
    p = 12
    w = rng.normal(size=p)
    f = lambda Z: Z @ w + 0.7 * Z[:, 0] * Z[:, 5]            # includes an interaction across groups
    groups = {"a": np.arange(0, 4), "b": np.arange(4, 8), "c": np.arange(8, 12)}
    x, bg = rng.normal(size=p), rng.normal(size=(15, p))
    phi, base, fx, gap = EX.group_shapley(f, x, bg, groups, n_perm=6, seed=1)
    assert gap < 1e-9 and np.isclose(base + phi.sum(), f(x[None])[0], atol=1e-4)
    # a group that does not matter gets ~0 effect
    w2 = np.r_[w[:8], np.zeros(4)]
    phi2, *_ = EX.group_shapley(lambda Z: Z @ w2, x, bg, groups, n_perm=6, seed=1)
    assert abs(phi2[2]) < 1e-9


@needs_data
def test_feature_groups_cover_every_feature_once():
    import joblib
    f = Path("results_uq/_prep_cache")
    cache = sorted(f.glob("prep_*.joblib")) if f.exists() else []
    if not cache:
        pytest.skip("no cached feature list available")
    import lhs_explain as EX
    names = joblib.load(cache[0])["feature_names"]
    groups = EX.feature_groups(names)
    assert sum(len(v) for v in groups.values()) == len(names) == len(set(np.concatenate(list(groups.values()))))


# ------------------------------------------------------------------------------------------------ outputs
@needs_data
def test_pipeline_writes_expected_files(tmp_path):
    import run_lhs
    cfg = LHSConfig(n_candidates=1500, out_dir=str(tmp_path))
    s = run_lhs.run(cfg, skip_models=True)
    for f in ("lhs_raw_candidates.csv", "lhs_feasible_candidates.csv", "lhs_rejected_candidates.csv", "lhs_feature_bounds.csv",
              "lhs_sampling_config.json", "lhs_summary.json"):
        assert (tmp_path / f).exists(), f
    raw, feas, rej = (pd.read_csv(tmp_path / f) for f in ("lhs_raw_candidates.csv", "lhs_feasible_candidates.csv", "lhs_rejected_candidates.csv"))
    assert len(raw) == 1500 and len(feas) + len(rej) == 1500 and rej["rejection_reasons"].notna().all()
    conf = json.loads((tmp_path / "lhs_sampling_config.json").read_text())
    assert conf["config"]["seed"] == 42 and "categories" in conf and "versions" in conf
    b = pd.read_csv(tmp_path / "lhs_feature_bounds.csv")
    assert {"variable", "role", "unit", "lower", "upper", "source", "rationale"} <= set(b.columns)
    assert s["validation_checks"]["lhs_exactly_stratified_all_dims"]


# ------------------------------------------------------------------------------------------------ slow: encoder + models
@pytest.fixture(scope="module")
def prep(split):
    from perla_cv import fit_final_preprocessors
    interp, enc, imp, X = fit_final_preprocessors(split.train_df, split.T_train)
    return interp, enc, imp, X


@slow
@needs_data
def test_candidates_encode_with_existing_pipeline(prep, sample, space, split, cfg):
    import lhs_predict as P
    interp, enc, imp, X_tr = prep

    class A:      # minimal stand-in for Artifacts
        pass
    art = A()
    art.interp, art.encoder, art.imputer, art.X_train_df, art.split = interp, enc, imp, X_tr, split
    art.feature_names = list(X_tr.columns)
    g, _, _ = sample
    frame = S.genome_to_perla_frame(g.iloc[:200].reset_index(drop=True), space)
    X, X_enc, imputed = P.encode_frame(art, frame)
    assert X.dtype == np.float32 and X.shape == (200, X_tr.shape[1]) and np.isfinite(X).all()
    assert list(X_enc.columns) == list(X_tr.columns)                       # order and names preserved
    assert (X_enc["num__band_gap__mean__missing"] == 1).all() and (X_enc["bg_basis__missing"] == 1).all()
    assert not any(c.startswith(("Voc", "Jsc", "FF", "JV_")) for c in X_enc.columns)
    # an unseen material must fail loudly, not be silently bucketed
    bad = frame.copy()
    bad["ETL_stack_sequence__parsed"] = [[["TotallyNewMaterial-X"]]] * len(bad)
    with pytest.raises(ValueError, match="cannot be represented"):
        P.encode_frame(art, bad)
    # an explicit missing boolean must fail loudly (missing is never False)
    bad = frame.copy()
    bad["Cell_flexible"] = pd.Series([None] * len(bad), dtype=object)
    with pytest.raises(ValueError):
        P.encode_frame(art, bad)


@slow
def test_twostage_uses_predicted_intermediates_and_mapie_columns():
    """Synthetic data + tiny models: Stage 2 receives Stage-1 PREDICTIONS; calibrated interval columns are correctly assigned."""
    import argparse
    import perla_mapie as pmq
    import run_models as rm
    import lhs_uncertainty as Q
    from lhs_predict import Finalist
    rng = np.random.RandomState(0)
    n, p = 700, 8
    X = rng.normal(size=(n, p)).astype(np.float32)
    aux = pd.DataFrame({"Voc": 1.0 + 0.1 * X[:, 0], "Jsc": 20 + 2 * X[:, 1], "FF": 0.7 + 0.02 * X[:, 2]})
    y = aux["Voc"].to_numpy() * aux["Jsc"].to_numpy() * aux["FF"].to_numpy() + rng.normal(0, 0.3, n)
    groups = np.repeat(np.arange(70), 10).astype(str)
    args = argparse.Namespace(quick=True, seed=0, n_jobs=1, stage1_base="hgb", inner_folds=3, add_product=False, stage1_label="S1=HistGradientBoosting")
    bank = pmq.MemberBank(args, X[:500], y[:500], aux.iloc[:500].reset_index(drop=True), groups[:500], 0.9)
    cand = bank.build("TwoStage[S1=HistGradientBoosting]|VotingET")
    F_ = bank.features("TwoStage", X[500:])
    assert F_.shape[1] == p + 3 and np.allclose(F_[:, :p], X[500:])          # Stage-2 input = [X | predicted Voc, Jsc, FF]
    assert not np.allclose(F_[:, p:], aux.iloc[500:].to_numpy(), atol=1e-6)    # ... and these are predictions, not the measured values
    fin = Finalist("TwoStage[S1=HistGradientBoosting]|VotingET", cand, cand["ctx"])
    class C:
        confidence, cal_repeats, split_seed, cal_fraction, seed = 0.9, 2, 42, 0.5, 42
    class A:
        D = dict(X_val=X[500:], y_val=y[500:], g_val=groups[500:], g_train=groups[:500])
    Q.calibrate(fin, A, C, lambda *a: None)
    out = Q.predict_with_intervals(fin, X[500:], 100, lambda *a: None)
    assert (out["lower"] <= out["upper"]).all() and np.isfinite(out.to_numpy()).all()
    inside = ((y[500:] >= out["lower"]) & (y[500:] <= out["upper"])).mean()
    assert inside > 0.75                                                     # lower/upper are the right way round and sensible
    assert np.allclose(out["point"], cand["point"](X[500:]))                 # point column is the point prediction
