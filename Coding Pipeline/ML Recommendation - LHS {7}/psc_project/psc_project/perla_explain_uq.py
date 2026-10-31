"""
perla_explain_uq.py  -  MAPIE CQR prediction intervals (all 12 candidates) + SHAP explanations
============================================================================================
Separate from run_models.py on purpose: nothing here takes part in model comparison or selection, and nothing here
retrains or alters the models of the comparison pipeline (point models = rm.make_models(), Stage 1 = rm.Stage1Predictor).

NOTHING RUNS UNLESS YOU ASK FOR IT:   --mapie-models <ids...>   runs MAPIE CQR for exactly those candidates;
                                      --shap-models  <ids...>   runs SHAP for exactly those candidates (all 12 supported; see perla_shap.py).
Candidate identifiers (same as run_models.py; bare name = Direct):
    RandomForest XGBoost HistGradientBoosting ExtraTrees VotingOriginal VotingET            (Direct)
    TwoStage:<same six names>   or the full id  "TwoStage[S1=HistGradientBoosting]|VotingET" (Two-Stage; Stage 1 = --stage1-base)
    python perla_explain_uq.py --list-models                  prints every valid identifier

DATA ROLES (same 70/15/15 paper-grouped split as perla_cv.py; preprocessing fitted on the 70% TRAIN set only)
    TRAIN 70%  : fits the preprocessing, the point models, the quantile models / quantile forests, Stage 1 (two-stage).
    VALIDATION : the MAPIE conformal-calibration set.  Never used to fit anything here.
    TEST 15%   : not even transformed unless --score-test.  Then the CQR is calibrated on the WHOLE validation set and its
                 coverage is measured on the test set - the only data that played no role in fitting or calibration.
    A conformalized interval is NEVER evaluated on the rows used to calibrate it:
      * development check (default): validation papers are split (by paper) into a calibration part (--cal-fraction) and
        a check part; coverage is measured on the check part.  --cal-repeats N repeats this with N different paper splits.
      * final (--score-test): calibrate on all validation papers, measure coverage on the test set.

MAPIE CQR   (mapie.regression.ConformalizedQuantileRegressor, prefit=True; MAPIE >= 1.0 API)
    See perla_mapie.py for the per-architecture construction of the [lower, upper, point] functions:
      HistGradientBoosting / XGBoost : native quantile models              -> CQR
      RandomForest / ExtraTrees      : quantile-regression-forest layer on the same fitted forest -> CQR
      VotingOriginal / VotingET      : point = the real ensemble mean; band = equal-weight average of member quantiles -> CQR
      Two-stage (6)                  : Stage 1 -> stack -> Stage 2 inside the three functions; calibrated end-to-end
    Default correction is MAPIE's asymmetric CQR (alpha/2 per tail); coverage guarantee is marginal and finite-sample, for
    rows exchangeable with the calibration rows.  Rows of one paper are correlated, so splits are by paper and coverage
    refers to NEW papers.  Output per model: <out>/<run_id>/mapie/<model>/ (metrics json, intervals csv, coverage-by-bin,
    plot, mapie_config.json with method / assumptions / calibration size / limitations).

    python perla_explain_uq.py --mapie-models VotingET VotingOriginal --out results_uq
    python perla_explain_uq.py --mapie-models TwoStage:VotingET --stage1-base hgb
    python perla_explain_uq.py --mapie-models VotingET --score-test               # final: calibrate on validation, evaluate on TEST
    python perla_explain_uq.py --mapie-models XGBoost --shap-models RandomForest  # both, independently chosen
    python perla_explain_uq.py --mapie-models VotingET --quick                    # smoke test, PRELIMINARY

SHAP (perla_shap.py; shap.TreeExplainer on the SAME fitted members as MAPIE; explained on a random validation subsample)
    All 12 candidates: single models directly, Voting ensembles as the exact weighted mean of member SHAP values, two-stage
    candidates over [X | Stage-1 predictions] (+ Stage-1 models explained separately).  Interventional SHAP by default (exact
    for every model incl. deep ExtraTrees); every explanation is checked for additivity and written with an _INEXACT suffix if
    it fails.  Run only when --shap-models is given.  SHAP values describe the model, not causation.

    python perla_explain_uq.py --shap-models all --out results_uq
    python perla_explain_uq.py --mapie-models all --shap-models all --score-test
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import warnings
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import GroupShuffleSplit

import perla_cv
import perla_encoding_by_claude as enc_module
import perla_mapie as pmq
import perla_shap as psh
import run_models as rm
from perla_cv import SPLIT_SEED, TARGET, _transform_with, fit_final_preprocessors, load_and_split, split_report

warnings.filterwarnings("ignore", category=UserWarning)


class Tee:
    """Copy everything printed to a log file as well."""

    def __init__(self, stream, path):
        self.stream, self.f = stream, open(path, "w", buffering=1)

    def write(self, s):
        self.stream.write(s)
        self.f.write(s)

    def flush(self):
        self.stream.flush()
        self.f.flush()


# =========================================================================================
# DATA (preprocessing is cached: it is deterministic given data file + split seed + preprocessing code)
# =========================================================================================
def _file_sha(path, n=12):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:n]


def prepare_data(a, need_test, cache_root):
    split = load_and_split(a.data, random_state=a.split_seed)
    split_info = split_report(split)
    key = hashlib.sha1(json.dumps(dict(
        data=_file_sha(a.data), split_seed=a.split_seed, knn=perla_cv.KNN_KWARGS, enc=perla_cv.ENCODER_KWARGS,
        vegard=perla_cv.USE_VEGARD_BAND_GAP_INTERPOLATION, cv_code=_file_sha(perla_cv.__file__),
        enc_code=_file_sha(enc_module.__file__)), sort_keys=True, default=str).encode()).hexdigest()[:12]
    cdir = Path(cache_root) / "_prep_cache"
    cdir.mkdir(parents=True, exist_ok=True)
    f_main, f_test = cdir / f"prep_{key}.joblib", cdir / f"prep_{key}_test.joblib"
    interp = enc = imp = X_tr_df = None
    if f_main.exists() and (not need_test or f_test.exists()):
        D = joblib.load(f_main)
        if need_test:
            D.update(joblib.load(f_test))
        print(f"preprocessing restored from cache {f_main.name} ({D['X_train'].shape[1]} features)")
    else:
        t = time.time()
        interp, enc, imp, X_tr_df = fit_final_preprocessors(split.train_df, split.T_train)       # fitted on TRAIN only
        X_val_df = _transform_with(interp, enc, imp, split.val_df, X_tr_df.columns)
        D = dict(X_train=X_tr_df.to_numpy(np.float32), X_val=X_val_df.to_numpy(np.float32),
                 y_train=split.T_train[TARGET].to_numpy(float), y_val=split.T_val[TARGET].to_numpy(float),
                 g_train=split.g_train.to_numpy().astype(str), g_val=split.g_val.to_numpy().astype(str),
                 T_train_aux=split.T_train[list(rm.AUX_TARGETS)].reset_index(drop=True), feature_names=list(X_tr_df.columns))
        joblib.dump(D, f_main)
        print(f"preprocessing done in {time.time() - t:.0f}s: {D['X_train'].shape[1]} features (cached as {f_main.name})")
        if need_test:
            Dt = dict(X_test=_transform_with(interp, enc, imp, split.test_df, X_tr_df.columns).to_numpy(np.float32),
                      y_test=split.T_test[TARGET].to_numpy(float), g_test=split.g_test.to_numpy().astype(str))
            joblib.dump(Dt, f_test)
            D.update(Dt)
    if need_test and "X_test" not in D:
        raise RuntimeError("test data requested but not available")
    D["split"], D["split_info"], D["prep_key"] = split, split_info, key
    return D


# =========================================================================================
# MAPIE CQR
# =========================================================================================
LIMITATIONS_COMMON = [
    "Coverage is MARGINAL (averaged over devices), not guaranteed within PCE ranges, compositions or for individual papers; "
    "coverage conditional on the true PCE is expected to fall short at the extremes (see coverage-by-bin file).",
    "The guarantee needs the calibration rows and the evaluated rows to be exchangeable.  Rows of a paper are correlated, "
    "so all splits are by paper; the guarantee then refers to NEW papers drawn like the calibration papers.",
    "Finite sample: the guarantee holds in expectation over the calibration draw; the realised coverage on one finite "
    "check/test set fluctuates (a paper-level bootstrap interval is reported).",
    "The development check calibrates on only part of the validation set and is a noisy estimate (use --cal-repeats).",
    "Quantile crossing is repaired by sorting (lower <- min, upper <- max) before conformalization.",
]
LIMITATIONS_BY_KIND = {
    "forest": "RandomForest / ExtraTrees have no quantile loss: the band is a quantile-regression-forest band built from the "
              "leaf weights of the SAME fitted forest (training rows only).  Validity does not depend on this choice; width does.",
    "voting": "There is no native quantile of an ensemble mean: the band is the equal-weight AVERAGE of the member quantile "
              "predictions while the point prediction is the exact ensemble mean.  The conformal calibration is applied to the "
              "averaged band, so coverage describes the ensemble band, not any single member.",
    "twostage": "Two-stage: Stage-2 models train on OUT-OF-FOLD Stage-1 predictions but are calibrated and evaluated on rows whose "
                "Stage-1 inputs come from Stage-1 models fitted on all training rows (as at deployment).  Stage-1 error is not "
                "propagated separately; it is absorbed into the calibrated end-to-end residuals.",
}


def _limitations(meta):
    L = list(LIMITATIONS_COMMON)
    ests = meta["members"]
    if any(m in ("RandomForest", "ExtraTrees") for m in ests):
        L.append(LIMITATIONS_BY_KIND["forest"])
    if len(ests) > 1:
        L.append(LIMITATIONS_BY_KIND["voting"])
    if meta["approach"] == "TwoStage":
        L.append(LIMITATIONS_BY_KIND["twostage"])
    return L


def _evaluate(cand, cqr, X, y, g, confidence, boot_seed):
    point, lo, hi, rlo, rhi = pmq.predict_all(cand, cqr, X)
    conf = pmq.interval_metrics(y, lo, hi, point, confidence, groups=g, seed=boot_seed)
    raw = pmq.interval_metrics(y, rlo, rhi, point, confidence, groups=g, seed=boot_seed)
    conf["raw_band_crossing_fraction_before_repair"] = cand["crossing"](X)
    frame = pd.DataFrame(dict(PCE_true=y, point=point, lower=lo, upper=hi, raw_lower=rlo, raw_upper=rhi, paper=g))
    return conf, raw, frame


def _calibrate(cand, confidence, X_cal, y_cal, X_eval, y_eval, g_eval, boot_seed):
    cqr = pmq.mapie_cqr(*pmq.prefit_trio(cand), confidence, X_cal, y_cal)
    conf, raw, frame = _evaluate(cand, cqr, X_eval, y_eval, g_eval, confidence, boot_seed)
    # independent re-computation of the conformal correction (no MAPIE) must reproduce MAPIE's intervals
    b_cal, b_eval = cand["band"](X_cal), cand["band"](X_eval)
    mq = pmq.manual_cqr_check(b_cal[:, 0], b_cal[:, 1], y_cal, confidence)
    (ql, qh), (tl, th) = mq["mapie_convention"], mq["textbook"]
    d_lo = float(np.abs((b_eval[:, 0] - ql) - frame["lower"].to_numpy()).max())
    d_hi = float(np.abs((b_eval[:, 1] + qh) - frame["upper"].to_numpy()).max())
    check = dict(manual_q_lower=ql, manual_q_upper=qh, textbook_q_lower=tl, textbook_q_upper=th,
                 max_abs_diff_lower=d_lo, max_abs_diff_upper=d_hi, mapie_equals_manual=bool(max(d_lo, d_hi) < 1e-6),
                 mapie_at_least_as_conservative_as_textbook=bool(ql >= tl - 1e-12 and qh >= th - 1e-12))
    if not (check["mapie_equals_manual"] and check["mapie_at_least_as_conservative_as_textbook"]):
        print(f"  WARNING: MAPIE intervals differ from the manual CQR re-computation (max diff {max(d_lo, d_hi):.3g})")
    return conf, raw, frame, check


def verify_candidate(bank, cand, X_val, y_val):
    """Cheap per-candidate dispatch / pipeline checks on a validation subsample (labels unused except shape)."""
    ctx, members = cand["ctx"], cand["members"]
    Xs = X_val[:min(300, len(X_val))]
    F = bank.features(ctx, Xs)
    mean_ref = np.mean([m.predict(F) for m in cand["pts"]], axis=0)         # recomputed independently of cand["point"]
    chk = dict(members_fitted=[type(m).__name__ for m in cand["pts"]],
               point_equals_member_mean=bool(np.allclose(cand["point"](Xs), mean_ref, atol=1e-9)),
               feature_width_seen_by_models=int(F.shape[1]),
               expected_width=int(bank.X.shape[1] + (len(rm.AUX_TARGETS) + int(bank.a.add_product) if ctx == "TwoStage" else 0)))
    chk["feature_width_ok"] = chk["feature_width_seen_by_models"] == chk["expected_width"]
    qr = {}
    for m, p in zip(members, cand["pts"]):
        if m in ("RandomForest", "ExtraTrees"):                              # QRF weights must reproduce forest.predict
            fq = pmq.ForestQuantiles(p, bank.train_X(ctx), bank.y)
            _, mean = fq.predict(F, [0.5], return_mean=True)
            qr[m] = float(np.abs(mean - p.predict(F)).max())
    chk["qrf_mean_vs_forest_predict_max_abs_diff"] = qr
    chk["qrf_ok"] = all(v < 1e-6 for v in qr.values())
    b = bank.quantile_fn(ctx, members[0])(Xs)
    chk["quantile_members_finite"] = bool(np.isfinite(b).all())
    chk["ok"] = bool(chk["point_equals_member_mean"] and chk["feature_width_ok"] and chk["qrf_ok"] and chk["quantile_members_finite"])
    return chk


def _plot(frame, cb, confidence, title, path):
    fig, ax = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    s = frame.sample(min(250, len(frame)), random_state=0).sort_values("point")
    ax[0].errorbar(np.arange(len(s)), s["point"], yerr=[(s["point"] - s["lower"]).clip(lower=0), (s["upper"] - s["point"]).clip(lower=0)],
                   fmt="none", ecolor="lightgray", lw=1)
    ax[0].scatter(np.arange(len(s)), s.PCE_true, c=np.where((s.PCE_true >= s.lower) & (s.PCE_true <= s.upper), "tab:green", "tab:red"),
                  s=8, zorder=3)
    ax[0].set_xlabel("250 random devices sorted by predicted PCE")
    ax[0].set_ylabel("PCE (%)")
    ax[0].set_title(f"{title}\n{confidence:.0%} CQR intervals; green = covered", fontsize=9)
    ax[1].bar(cb["bin"].astype(str), cb["coverage"])
    ax[1].axhline(confidence, color="r", ls="--", label="target")
    ax[1].set_ylabel("coverage")
    ax[1].set_xlabel("true PCE bin (conditioning on y: under-coverage at the extremes is expected)")
    ax[1].legend()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def run_mapie(a, D, out, model_ids, bank):
    print(f"\n=== MAPIE CQR for {len(model_ids)} requested candidate(s): {model_ids} ===")
    mdir = out / "mapie"
    mdir.mkdir(exist_ok=True)
    X_tr, y_tr, X_val, y_val, g_val = D["X_train"], D["y_train"], D["X_val"], D["y_val"], D["g_val"]
    # ---- leakage guards -----------------------------------------------------------------------------------------
    g_tr_set, g_val_set = set(D["g_train"]), set(g_val)
    assert not (g_tr_set & g_val_set), "a paper is in both TRAIN and VALIDATION"
    if a.score_test:
        assert not ((g_tr_set | g_val_set) & set(D["g_test"])), "a paper is in TEST and in TRAIN/VALIDATION"
    else:
        assert "X_test" not in D, "test data must not be loaded without --score-test"
    leakage = dict(train_rows=len(y_tr), validation_rows=len(y_val), test_rows=int(len(D["y_test"])) if a.score_test else "not loaded",
                   train_validation_paper_overlap=0, test_paper_overlap=0 if a.score_test else "not applicable",
                   models_fitted_on="TRAIN only (point models, quantile models, quantile forests, Stage 1)",
                   calibration_data="VALIDATION papers only (development: a paper-disjoint part; final: all)",
                   test_data=("used only for the final coverage / width measurement" if a.score_test else "NOT loaded"))
    all_res, summary_rows = {}, []
    for mid in model_ids:
        print(f"\n--- {mid} ---")
        t0 = time.time()
        cand = bank.build(mid)
        tag = pmq.safe_tag(mid)
        odir = mdir / tag
        odir.mkdir(exist_ok=True)
        ver = verify_candidate(bank, cand, X_val, y_val)
        print(f"  dispatch/pipeline checks: {'OK' if ver['ok'] else 'FAILED'} {ver}")
        if not ver["ok"]:
            raise SystemExit(f"{mid}: internal verification failed; refusing to produce intervals")
        res = dict(model_id=mid, meta=cand["meta"], verification=ver)
        frames = {}
        # ---- development check (paper-disjoint split of the VALIDATION set) -----------------------------------------
        reps = []
        for r in range(a.cal_repeats):
            gss = GroupShuffleSplit(n_splits=1, test_size=1 - a.cal_fraction, random_state=a.split_seed + r)
            cal, chk = next(gss.split(X_val, groups=g_val))
            assert not (set(g_val[cal]) & set(g_val[chk])), "a paper is in both the calibration and the check part"
            conf, raw, frame, xc = _calibrate(cand, a.confidence, X_val[cal], y_val[cal], X_val[chk], y_val[chk], g_val[chk], a.seed + r)
            reps.append(dict(repeat=r, calibration_rows=int(len(cal)), calibration_papers=int(len(set(g_val[cal]))),
                             check_rows=int(len(chk)), check_papers=int(len(set(g_val[chk]))), conformalized=conf,
                             raw_quantile_band=raw, mapie_vs_manual_crosscheck=xc))
            if r == 0:
                frames["check"] = frame
                pmq.coverage_by_bin(y_val[chk], frame.lower.to_numpy(), frame.upper.to_numpy()).round(4).to_csv(odir / "mapie_coverage_by_bin_check.csv", index=False)
                pmq.coverage_by_width_tercile(y_val[chk], frame.lower.to_numpy(), frame.upper.to_numpy()).round(4).to_csv(odir / "mapie_coverage_by_width_tercile_check.csv", index=False)
        res["development_check"] = reps[0]
        if a.cal_repeats > 1:
            cov = [x["conformalized"]["coverage"] for x in reps]
            wid = [x["conformalized"]["mean_width"] for x in reps]
            res["development_check_repeats"] = dict(n=len(reps), coverage_mean=float(np.mean(cov)), coverage_sd=float(np.std(cov, ddof=1)),
                                                    coverage_min=float(np.min(cov)), coverage_max=float(np.max(cov)),
                                                    mean_width_mean=float(np.mean(wid)), mean_width_sd=float(np.std(wid, ddof=1)), per_repeat=reps)
        # ---- final: calibrate on ALL validation, evaluate on TEST ------------------------------------------------------
        if a.score_test:
            conf, raw, frame, xc = _calibrate(cand, a.confidence, X_val, y_val, D["X_test"], D["y_test"], D["g_test"], a.seed)
            res["final_test"] = dict(calibration_rows=int(len(y_val)), calibration_papers=int(len(g_val_set)), test_rows=int(len(D["y_test"])),
                                     test_papers=int(len(set(D["g_test"]))), conformalized=conf, raw_quantile_band=raw, mapie_vs_manual_crosscheck=xc)
            frames["test"] = frame
            pmq.coverage_by_bin(D["y_test"], frame.lower.to_numpy(), frame.upper.to_numpy()).round(4).to_csv(odir / "mapie_coverage_by_bin_test.csv", index=False)
            pmq.coverage_by_width_tercile(D["y_test"], frame.lower.to_numpy(), frame.upper.to_numpy()).round(4).to_csv(odir / "mapie_coverage_by_width_tercile_test.csv", index=False)
        for k, f in frames.items():
            f.round(4).to_csv(odir / f"mapie_intervals_{k}.csv", index=False)
        key = "final_test" if a.score_test else "development_check"
        kk = "test" if a.score_test else "check"
        cb = pd.read_csv(odir / f"mapie_coverage_by_bin_{kk}.csv")
        _plot(frames[kk], cb, a.confidence, f"{mid} ({'TEST' if a.score_test else 'validation check part'})", odir / "mapie_cqr_overview.png")
        n_cal_final = int(len(y_val)) if a.score_test else res["development_check"]["calibration_rows"]
        cfgm = dict(selected_via="--mapie-models", confidence_level=a.confidence, correction="asymmetric CQR (MAPIE default; alpha/2 per tail)",
                    mapie_class="mapie.regression.ConformalizedQuantileRegressor(prefit=True)", **cand["meta"],
                    quantile_levels=[bank.q_lo, bank.q_hi], quantile_notes={k: v for k, v in bank.notes.items() if k.split("|")[1] in cand["members"]},
                    calibration=dict(set="VALIDATION papers", development_calibration_rows=res["development_check"]["calibration_rows"],
                                     development_check_rows=res["development_check"]["check_rows"], cal_fraction_of_papers=a.cal_fraction,
                                     cal_repeats=a.cal_repeats, final_calibration_rows=int(len(y_val)) if a.score_test else None,
                                     evaluated_on=("TEST (final)" if a.score_test else "paper-disjoint check part of VALIDATION")),
                    assumptions=["calibration and evaluation rows exchangeable (paper-level split)", "lower/upper/point fitted without the calibration rows",
                                 "one conformalize call per MAPIE object (fresh objects are used for each calibration)"],
                    limitations=_limitations(cand["meta"]), leakage_checks=leakage,
                    fit_seconds={k: v for k, v in bank.timing.items()}, run_seconds_this_candidate=round(time.time() - t0, 1),
                    versions=dict(mapie=__import__("mapie").__version__, sklearn=sklearn.__version__))
        (odir / "mapie_config.json").write_text(json.dumps(cfgm, indent=2, default=str))
        (odir / "mapie_cqr_metrics.json").write_text(json.dumps(res, indent=2, default=str))
        c, r_ = res[key]["conformalized"], res[key]["raw_quantile_band"]
        print(f"  {key}: conformalized coverage {c['coverage']:.3f} (target {a.confidence}), mean width {c['mean_width']:.2f}, "
              f"interval score {c['interval_score']:.2f} | raw band coverage {r_['coverage']:.3f}, width {r_['mean_width']:.2f} | "
              f"MAPIE==manual: {res[key]['mapie_vs_manual_crosscheck']['mapie_equals_manual']}")
        all_res[mid] = res
        summary_rows.append(dict(model_id=mid, evaluated_on=key, n_calibration=n_cal_final, n_eval=c["n"], target=a.confidence,
                                 coverage=c["coverage"], coverage_ci_low=c.get("coverage_paper_bootstrap_95ci", [np.nan, np.nan])[0],
                                 coverage_ci_high=c.get("coverage_paper_bootstrap_95ci", [np.nan, np.nan])[1], mean_width=c["mean_width"],
                                 median_width=c["median_width"], interval_score=c["interval_score"],
                                 spearman_width_vs_abs_error=c["spearman_width_vs_abs_error"], point_RMSE=c["point_RMSE"], point_R2=c["point_R2"],
                                 raw_coverage=r_["coverage"], raw_mean_width=r_["mean_width"], seconds=round(time.time() - t0, 1)))
    pd.DataFrame(summary_rows).round(4).to_csv(mdir / "mapie_summary.csv", index=False)
    print("\nMAPIE summary:\n" + pd.DataFrame(summary_rows).round(3).to_string(index=False))
    return dict(models=list(model_ids), results=all_res, leakage_checks=leakage)


# =========================================================================================
def validate_args(a, ap):
    label = pmq.stage1_label(a.stage1_base)
    if a.list_models:
        print("Supported --mapie-models identifiers (bare name = Direct; 'TwoStage:<name>' or the full id for Two-Stage):")
        for i in pmq.all_model_ids(label):
            print("   ", i)
        print("--shap-models accepts the same identifiers (all 12 candidates).  'all' = all 12 for both flags.")
        raise SystemExit(0)
    def _parse(flag, names):
        ids = []
        for m in (names or []):
            for cid in (pmq.all_model_ids(label) if str(m).strip().lower() == "all" else [pmq.parse_model_id(m, label)]):
                if cid not in ids:
                    ids.append(cid)
        return ids
    try:
        a.mapie_ids = [] if a.no_mapie else _parse("--mapie-models", a.mapie_models)
        a.shap_ids = [] if a.no_shap else _parse("--shap-models", a.shap_models)
    except ValueError as e:
        ap.error(str(e))
    a.run_mapie = bool(a.mapie_ids)
    a.run_shap = bool(a.shap_ids)
    if not (a.run_mapie or a.run_shap):
        ap.error("nothing requested: give --mapie-models <ids> and/or --shap-models <names>  (see --list-models)")
    if a.shap_background < 1 or a.shap_n < 1:
        ap.error("--shap-background and --shap-n must be >= 1")
    if not 0 < a.confidence < 1:
        ap.error("--confidence must be in (0, 1)")
    if not 0 < a.cal_fraction < 1:
        ap.error("--cal-fraction must be in (0, 1)")
    if a.cal_repeats < 1:
        ap.error("--cal-repeats must be >= 1")
    if a.score_test and not a.run_mapie:
        ap.error("--score-test only applies to MAPIE (--mapie-models); SHAP never uses the test set")


def main(a):
    if a.score_test and a.quick:
        raise SystemExit("--score-test is refused for a preliminary (--quick) run.")
    cfg_key = json.dumps({k: v for k, v in vars(a).items() if k not in ("n_jobs", "out", "data", "list_models")}, sort_keys=True, default=str)
    run_id = f"{'PRELIM_' if a.quick else ''}uq_seed{a.seed}_split{a.split_seed}_{hashlib.sha1(cfg_key.encode()).hexdigest()[:8]}"
    out = Path(a.out) / run_id
    out.mkdir(parents=True, exist_ok=True)
    sys.stdout = Tee(sys.__stdout__, out / "run_log.txt")
    print(f"run id: {run_id} -> {out}\nrequested: MAPIE {a.mapie_ids if a.run_mapie else 'no'} | SHAP {a.shap_ids if a.run_shap else 'no'}"
          f" | test set {'USED (final)' if a.score_test else 'not used'}")
    # fail early on missing packages (only the ones that are needed)
    versions = dict(sklearn=sklearn.__version__, pandas=pd.__version__, numpy=np.__version__)
    if a.run_mapie:
        try:
            import mapie
            import xgboost
            versions.update(mapie=mapie.__version__, xgboost=xgboost.__version__)
        except ImportError as e:
            raise SystemExit(f"{e}. Install with: pip install mapie xgboost")
    if a.run_shap:
        try:
            import shap
            versions["shap"] = shap.__version__
        except ImportError as e:
            raise SystemExit(f"{e}. Install SHAP with: pip install shap")
    base1 = {"hgb": "HistGradientBoosting", "xgb": "XGBoost", "rf": "RandomForest"}[a.stage1_base]
    a.stage1_label = f"S1={base1}"
    D = prepare_data(a, need_test=a.score_test and a.run_mapie, cache_root=a.out)
    cfg = dict(run_id=run_id, preliminary=bool(a.quick), args=vars(a), selected_mapie_models=a.mapie_ids if a.run_mapie else [],
               selected_shap_models=a.shap_ids if a.run_shap else [], split=D["split_info"], n_features=int(D["X_train"].shape[1]),
               feature_names=D["feature_names"], grouping_column=enc_module.GROUP_COL, target_column=enc_module.TARGET_COLS[TARGET],
               test_set_used=bool(a.score_test and a.run_mapie), mapie_implemented=a.run_mapie, shap_implemented=a.run_shap,
               preprocessing_cache_key=D["prep_key"], versions=versions)
    # ONE shared bank: a member (e.g. RandomForest) or the Stage-1 models fitted for MAPIE are reused by SHAP and vice versa.
    bank = pmq.MemberBank(a, D["X_train"], D["y_train"], D["T_train_aux"], D["g_train"], a.confidence)
    if a.run_mapie:
        cfg["mapie"] = run_mapie(a, D, out, a.mapie_ids, bank)
    if a.run_shap:
        cfg["shap"] = psh.run_shap(a, D, out, bank, a.shap_ids)
    (out / "run_config.json").write_text(json.dumps(cfg, indent=2, default=str))
    print(f"\nOutputs written to {out.resolve()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="MAPIE CQR intervals and SHAP explanations; runs only the analyses / models you request.")
    ap.add_argument("--data", default="perla_final_filtered.pkl")
    ap.add_argument("--out", default="results_uq")
    ap.add_argument("--mapie-models", nargs="+", default=None, metavar="MODEL",
                    help="candidates for MAPIE CQR, e.g. VotingET  TwoStage:VotingOriginal, or all (= all 12) (see --list-models). Nothing runs if omitted.")
    ap.add_argument("--list-models", action="store_true", help="print the supported model identifiers and exit")
    ap.add_argument("--confidence", type=float, default=0.9, help="CQR target coverage (default 0.9)")
    ap.add_argument("--cal-fraction", type=float, default=0.5,
                    help="development check: share of validation PAPERS used for calibration (rest = check part)")
    ap.add_argument("--cal-repeats", type=int, default=1, help="development check: number of different paper-level calibration/check splits")
    ap.add_argument("--shap-models", nargs="+", default=None, metavar="MODEL",
                    help="candidates to explain with SHAP, same identifiers as --mapie-models (or 'all' = all 12). SHAP runs only if given")
    ap.add_argument("--shap-method", choices=["interventional", "path"], default="interventional",
                    help="interventional (default; exact, needs a background) or path-dependent (no background; NOT reliable for deep ExtraTrees)")
    ap.add_argument("--shap-background", type=int, default=100, help="TRAIN rows used as interventional-SHAP background (cost is linear in this)")
    ap.add_argument("--stage1-base", choices=["hgb", "xgb", "rf"], default="hgb", help="Stage-1 estimator type (as run_models.py)")
    ap.add_argument("--inner-folds", type=int, default=rm.N_INNER_SPLITS, help="inner grouped CV folds for Stage-1 OOF (default 5)")
    ap.add_argument("--add-product", action="store_true", help="two-stage: add Voc_hat*Jsc_hat*FF_hat as a Stage-2 feature")
    ap.add_argument("--shap-n", type=int, default=1000, help="validation rows explained by SHAP")
    ap.add_argument("--no-shap", action="store_true", help="veto SHAP even if --shap-models is given")
    ap.add_argument("--no-mapie", action="store_true", help="veto MAPIE even if --mapie-models is given")
    ap.add_argument("--score-test", action="store_true", help="final MAPIE: calibrate on all validation, evaluate coverage on TEST")
    ap.add_argument("--quick", action="store_true", help="small models -> PRELIMINARY smoke test")
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split-seed", type=int, default=SPLIT_SEED)
    args = ap.parse_args()
    validate_args(args, ap)
    main(args)
