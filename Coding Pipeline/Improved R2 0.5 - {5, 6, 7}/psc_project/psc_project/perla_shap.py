"""
perla_shap.py - SHAP explanations for the 12 candidate models (6 direct + 6 two-stage)
=======================================================================================
Supporting module of perla_explain_uq.py (like perla_mapie.py).  Nothing here takes part in model comparison / selection and
nothing here retrains or changes the models of run_models.py: the point models are the SAME fitted members that
perla_mapie.MemberBank builds (rm.make_models(), tuned --params and paper weights if given; Stage 1 = rm.Stage1Predictor).

IDENTIFIERS: same as perla_mapie / run_models.py (bare name = Direct, "TwoStage:<name>" or the full candidate id).

HOW EVERY CANDIDATE IS EXPLAINED
    single model (RF, XGBoost, HGBT, ExtraTrees) : shap.TreeExplainer on the fitted model.
    VotingOriginal / VotingET                    : the ensemble is the equal-weight mean of its members, and SHAP values are
                                                   linear in the model, so  phi_ensemble = mean_m phi_m  and
                                                   base_ensemble = mean_m base_m  (exact, no approximation).
    Two-stage (all six)                          : the Stage-2 model(s) are explained over the matrix they were trained on,
                                                   [X | Voc_hat Jsc_hat FF_hat (| product)].  Explained rows carry Stage-1
                                                   PREDICTIONS (never measured Voc/Jsc/FF).  Attribution on a *_hat column is
                                                   not pushed back onto X; the Stage-1 models are explained separately
                                                   (shap/stage1) and the share of attribution sitting on the *_hat columns is
                                                   reported per candidate.

WHICH SHAP ALGORITHM  (--shap-method)
    interventional (default) : TreeExplainer(model, data=<--shap-background TRAIN rows>, feature_perturbation="interventional").
                               Exact for every model here and the same definition for all 12 (baseline = mean prediction over
                               the background rows).  Cost is linear in the background size.
                               EXCEPTION (reported, never silent): some shap versions (0.52.0 with XGBoost >= 3.3, where
                               enable_categorical defaults to True) refuse ANY XGBoost model with NotImplementedError "Categorical
                               split is not yet supported", although these models have no categorical features or splits.  For such a
                               member the XGBoost-native TreeSHAP (Booster.predict(pred_contribs=True), path-dependent, exact for
                               XGBoost's shallow trees) is used instead.  Its baseline is the training-cover mean rather than the
                               background mean; both are additive, so the ensemble sum stays exact.  The algorithm used by each member
                               is written to shap_config.json (member_algorithms) and shap_summary.csv.  (shap >= 0.53 needs no fallback.)
    path                     : path-dependent TreeSHAP (needs no background; baseline = training mean).  It is NOT reliable for
                               deep random-split trees: it is numerically unstable beyond ~60 levels.  ExtraTrees trees in this
                               project are 49-72 deep and the path-dependent explanation missed the model prediction by up to
                               ~5.6 PCE points on single trees (correlation of the error with tree depth: 0.96) while RandomForest
                               (depth <= 57), XGBoost and HGBT were exact.  Every explanation is therefore CHECKED: base + sum(phi)
                               must reproduce the candidate's own prediction to within SHAP_TOL, and a candidate that fails is written
                               with an _INEXACT suffix and flagged (never silently).
FOREST THRESHOLD SNAPPING (RandomForest / ExtraTrees): sklearn stores each split threshold as a float64 midpoint between two
adjacent float32 feature values and compares float32(x) <= threshold.  SHAP evaluates the same splits with the thresholds rounded to
float32; when the rounding lands on the upper neighbour, a row exactly on that value goes to the other child.  On fully grown trees this
happens in a few trees out of hundreds, and then base + sum(phi) misses the model prediction (observed: 0.05 PCE points on a quick RF,
single trees off by >1).  Before explaining a forest every threshold is therefore moved down to the largest float32 <= threshold
(identical decisions for float32 inputs, predictions bit-identical), and restored afterwards.  Explained and background rows are
cast to float32 for the same reason.
SHAP values describe what the MODEL does with its inputs (in PCE %-points); they are not causal effects.  Interventional SHAP
treats features as independent when sampling the background, so combinations that never occur together in real devices are
evaluated too.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import perla_mapie as pmq
import run_models as rm

SHAP_TOL = 1e-3                                       # max |base + sum(SHAP) - prediction| in PCE %-points
XGB_NATIVE = "xgboost_native_pred_contribs"


# ---- helpers ---------------------------------------------------------------------------------------------------------
def _f32(A):
    """C-contiguous float32 copy (see INPUT PRECISION in the module docstring)."""
    return np.ascontiguousarray(np.asarray(A), dtype=np.float32)


def _is_xgb(model):
    try:
        import xgboost as xgb
    except ImportError:
        return False
    return isinstance(model, xgb.XGBModel)


def _xgb_contribs(model, F):
    """XGBoost-native TreeSHAP -> (phi (n, f), base).  Handles categorical splits (which shap.TreeExplainer cannot)."""
    import xgboost as xgb
    booster = model.get_booster()
    kw = {}
    try:                                                      # respect early stopping, like model.predict does
        bi = model.best_iteration
        if bi is not None:
            kw["iteration_range"] = (0, int(bi) + 1)
    except Exception:
        pass
    dm = xgb.DMatrix(F, feature_types=booster.feature_types, enable_categorical=True)
    c = booster.predict(dm, pred_contribs=True, validate_features=False, **kw)
    return c[:, :-1], float(c[0, -1])


def _is_sk_forest(model):
    from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
    return isinstance(model, (RandomForestRegressor, ExtraTreesRegressor))


def _snap_thresholds(model):
    """In place: split thresholds -> largest float32 <= threshold (see FOREST THRESHOLD SNAPPING).  Returns the saved originals."""
    saved = []
    for e in model.estimators_:
        thr = e.tree_.threshold                                   # writable view of the node array
        inner = e.tree_.children_left != -1
        t = thr[inner]
        saved.append(t.copy())
        f = t.astype(np.float32)
        f = np.where(f.astype(np.float64) > t, np.nextafter(f, np.float32(-np.inf)), f)
        thr[inner] = f.astype(np.float64)
    return saved


def _restore_thresholds(model, saved):
    for e, t in zip(model.estimators_, saved):
        e.tree_.threshold[e.tree_.children_left != -1] = t


def tree_shap(shap, model, F, background, method):
    """-> (phi (n, features), base_value, algorithm_used).  The algorithm actually used is always returned and reported."""
    snapped = _snap_thresholds(model) if _is_sk_forest(model) else None
    try:
        if method == "interventional":
            expl = shap.TreeExplainer(model, data=background, feature_perturbation="interventional")
        else:
            expl = shap.TreeExplainer(model)
        sv = np.asarray(expl.shap_values(F, check_additivity=False))
        if sv.ndim == 3:                                      # (n, features, 1) in some versions
            sv = sv[..., 0]
        return sv, float(np.ravel(expl.expected_value)[0]), method
    except NotImplementedError as e:
        if not (_is_xgb(model) and "ategorical" in str(e)):
            raise
        sv, bv = _xgb_contribs(model, F)
        return sv, bv, XGB_NATIVE
    finally:
        if snapped is not None:
            _restore_thresholds(model, snapped)


class ShapRunner:
    """Explains candidates on one fixed set of validation rows with one fixed background; members are cached, so a
    member shared by several candidates (e.g. RandomForest in 4 of the 12) is explained only once per context."""

    def __init__(self, a, bank, Xs, feature_names):
        import shap
        self.shap, self.a, self.bank, self.Xs, self.names = shap, a, bank, _f32(Xs), list(feature_names)
        self.method = a.shap_method
        rng = np.random.RandomState(a.seed)
        nb = min(a.shap_background, len(bank.X))
        self.bg_idx = np.sort(rng.choice(len(bank.X), size=nb, replace=False))      # TRAIN rows only
        self.cache, self.info = {}, {}
        self._fm, self._bg = {}, {}

    # ---- feature space of a context ------------------------------------------------------------------------
    def feat_names(self, ctx):
        if ctx == "Direct":
            return self.names
        return self.names + [f"{k}_hat" for k in rm.AUX_TARGETS] + (["VocJscFF_hat_product"] if self.a.add_product else [])

    def explained_matrix(self, ctx):
        if ctx not in self._fm:
            self._fm[ctx] = _f32(self.bank.features(ctx, self.Xs))   # TwoStage: Stage-1 PREDICTIONS for the explained rows
        return self._fm[ctx]

    def background(self, ctx):
        if ctx not in self._bg:
            self._bg[ctx] = _f32(self.bank.train_X(ctx)[self.bg_idx])   # TwoStage: the OUT-OF-FOLD matrix Stage 2 was trained on
        return self._bg[ctx]

    # ---- one member (fitted point model) ---------------------------------------------------------------------
    def member(self, ctx, base):
        k = (ctx, base)
        if k in self.cache:
            return self.cache[k]
        t = time.time()
        model = self.bank.point_model(ctx, base)
        F = self.explained_matrix(ctx)
        sv, bv, algo = tree_shap(self.shap, model, F, self.background(ctx), self.method)
        gap = np.abs(bv + sv.sum(axis=1) - model.predict(F))
        self.info[k] = dict(shap_seconds=round(time.time() - t, 1), algorithm=algo, max_additivity_gap=float(gap.max()),
                            mean_additivity_gap=float(gap.mean()), additive=bool(gap.max() <= SHAP_TOL),
                            expected_value=bv)
        print(f"  [{ctx}] {base}: SHAP ({algo}) {time.time() - t:.0f}s on {len(F)} rows; additivity gap max "
              f"{gap.max():.2g} -> {'OK' if gap.max() <= SHAP_TOL else 'NOT ADDITIVE'}", flush=True)
        self.cache[k] = (sv, bv, algo)
        return self.cache[k]

    # ---- one candidate ----------------------------------------------------------------------------------------------
    def candidate(self, model_id):
        approach, est = pmq.split_model_id(model_id)
        members = rm.members_of(est)
        w = np.array([rm.VOTING_WEIGHTS[m] for m in members], float)
        w = w / w.sum()
        parts = [self.member(approach, m) for m in members]
        sv = sum(wi * p[0] for wi, p in zip(w, parts))
        bv = float(sum(wi * p[1] for wi, p in zip(w, parts)))
        F = self.explained_matrix(approach)
        pred = np.average(np.column_stack([self.bank.point_model(approach, m).predict(F) for m in members]), axis=1, weights=w)
        gap = np.abs(bv + sv.sum(axis=1) - pred)                  # the CANDIDATE's own prediction (what run_models.py scores)
        return dict(model_id=model_id, approach=approach, estimator=est, members=list(members), sv=sv, base=bv, pred=pred,
                    gap=gap, additive=bool(gap.max() <= SHAP_TOL), names=self.feat_names(approach), F=F,
                    algorithms={m: p[2] for m, p in zip(members, parts)})


def _write_candidate(shap, c, out_dir, a, n_bg):
    tag = pmq.safe_tag(c["model_id"])
    suffix = "" if c["additive"] else "_INEXACT"
    d = Path(out_dir) / (tag + suffix)
    d.mkdir(parents=True, exist_ok=True)
    s, names = c["sv"], c["names"]
    mean_abs = pd.Series(np.abs(s).mean(axis=0), index=names).sort_values(ascending=False)
    tab = pd.DataFrame({"mean_abs_shap": mean_abs, "mean_shap": pd.Series(s.mean(axis=0), index=names).reindex(mean_abs.index)})
    tab.to_csv(d / "shap_mean_abs.csv")
    np.savez_compressed(d / "shap_values.npz", shap_values=s.astype(np.float32), base_value=c["base"],
                        prediction=c["pred"], feature_names=np.array(names))
    plt.figure()
    shap.summary_plot(s, pd.DataFrame(c["F"], columns=names), max_display=20, show=False)
    plt.title(f"SHAP {c['model_id']}{' [INEXACT]' if suffix else ''}\n({a.shap_method}; n={len(s)} validation rows; PCE %-points)", fontsize=9)
    plt.savefig(d / "shap_summary.png", dpi=140, bbox_inches="tight")
    plt.close("all")
    hat = [n for n in names if n.endswith("_hat") or n == "VocJscFF_hat_product"]
    share = float(mean_abs[hat].sum() / mean_abs.sum()) if hat else None
    algos = c["algorithms"]
    cfg = dict(model_id=c["model_id"], approach=c["approach"], estimator=c["estimator"], members=c["members"],
               shap_method=a.shap_method, member_algorithms=algos,
               background_rows=n_bg if a.shap_method == "interventional" else None,
               background="random TRAIN rows (seeded)" if a.shap_method == "interventional" else "none (path-dependent: training cover)",
               n_rows_explained=int(len(s)), n_features=len(names), additivity_tolerance=SHAP_TOL,
               candidate_additivity=dict(base_value=c["base"], max_gap=float(c["gap"].max()), mean_gap=float(c["gap"].mean()),
                                         additive=c["additive"]),
               stage1_prediction_share_of_attribution=share,
               explained_rows="validation subsample (see shap_explained_rows.csv); no labels used",
               note="SHAP values describe the model's behaviour, not causal effects.")
    (d / "shap_config.json").write_text(json.dumps(cfg, indent=2))
    top = [f"{n} ({v:.2f})" for n, v in mean_abs.head(5).items()]
    return dict(model_id=c["model_id"], shap_method=a.shap_method,
                member_algorithms=";".join(f"{m}={al}" for m, al in algos.items()), n_rows=len(s), additive=c["additive"],
                max_additivity_gap=float(c["gap"].max()), base_value=c["base"], stage1_share=share,
                top5=" | ".join(top), folder=d.name)


def _stage1_shap(runner, out_dir):
    """What drives the three Stage-1 models (their predictions become the *_hat inputs of the Stage-2 models)."""
    shap, a, bank = runner.shap, runner.a, runner.bank
    s1 = bank._stage1()
    d = Path(out_dir) / "stage1"
    d.mkdir(parents=True, exist_ok=True)
    bg = _f32(bank.X[runner.bg_idx])
    rows = {}
    for aux in rm.AUX_TARGETS:
        m = s1.models_[aux]
        v, bv, algo = tree_shap(shap, m, runner.Xs, bg, runner.method)
        gap = float(np.abs(bv + v.sum(axis=1) - m.predict(runner.Xs)).max())
        pd.DataFrame({"mean_abs_shap": np.abs(v).mean(axis=0), "mean_shap": v.mean(axis=0)}, index=runner.names
                     ).sort_values("mean_abs_shap", ascending=False).to_csv(d / f"shap_stage1_{aux}.csv")
        plt.figure()
        shap.summary_plot(v, pd.DataFrame(runner.Xs, columns=runner.names), max_display=20, show=False)
        plt.title(f"SHAP Stage 1 target {aux} ({algo}; validation subsample)", fontsize=9)
        plt.savefig(d / f"shap_stage1_{aux}.png", dpi=140, bbox_inches="tight")
        plt.close("all")
        rows[aux] = dict(algorithm=algo, max_additivity_gap=gap, additive=bool(gap <= SHAP_TOL))
        print(f"  [stage1] {aux}: SHAP ({algo}) additivity gap max {gap:.2g} -> {'OK' if gap <= SHAP_TOL else 'NOT ADDITIVE'}", flush=True)
    return rows


def run_shap(a, D, out, bank, model_ids):
    """Explain exactly the requested candidates.  `bank` is the (shared) perla_mapie.MemberBank."""
    import shap
    print(f"\n=== SHAP ({a.shap_method}) for {len(model_ids)} requested candidate(s): {model_ids} ===")
    sdir = Path(out) / "shap"
    sdir.mkdir(exist_ok=True)
    X_val = D["X_val"]
    rng = np.random.RandomState(a.seed)
    idx = np.sort(rng.choice(len(X_val), size=min(a.shap_n, len(X_val)), replace=False))
    pd.DataFrame({"validation_row_position": idx}).to_csv(sdir / "shap_explained_rows.csv", index=False)
    assert "X_test" not in D or a.score_test, "test data must not be loaded for SHAP"
    runner = ShapRunner(a, bank, X_val[idx], D["feature_names"])
    rows, results = [], {}
    for mid in model_ids:
        print(f"\n--- {mid} ---", flush=True)
        t0 = time.time()
        c = runner.candidate(mid)
        print(f"  candidate additivity gap max {c['gap'].max():.2g} (tol {SHAP_TOL}) -> {'OK' if c['additive'] else 'NOT ADDITIVE -> written as _INEXACT'}")
        row = _write_candidate(shap, c, sdir, a, len(runner.bg_idx))
        row["seconds_incl_cached_members"] = round(time.time() - t0, 1)
        rows.append(row)
        results[mid] = row
    stage1 = _stage1_shap(runner, sdir) if any(pmq.split_model_id(m)[0] == "TwoStage" for m in model_ids) else None
    summ = pd.DataFrame(rows)
    summ.round(5).to_csv(sdir / "shap_summary.csv", index=False)
    print("\nSHAP summary:\n" + summ[["model_id", "n_rows", "additive", "max_additivity_gap", "stage1_share", "top5"]].round(4).to_string(index=False))
    return dict(method=a.shap_method, background_rows=len(runner.bg_idx), n_rows_explained=int(len(idx)), models=list(model_ids),
                results=results, stage1_models=stage1, member_diagnostics={f"{k[0]}|{k[1]}": v for k, v in runner.info.items()},
                additivity_tolerance=SHAP_TOL,
                twostage_note=("Stage-2 SHAP is over X plus the Stage-1 predictions; attribution on a *_hat column is NOT split "
                               "back onto X (only approximate). Read shap/stage1 for what drives each *_hat."))
