"""
run_models.py
=============
Fits the models on the leakage-free folds produced by perla_cv.py, compares them with the SAME outer grouped folds,
selects the best configurations automatically (mean outer-CV RMSE) and writes every output needed to audit the choice.
Base-model settings use the selected fixed hyperparameters from tuning/tuned_params.json (see make_models()).

DATA FLOW (70/15/15, grouped by paper / Ref_DOI_number - no paper is ever in two sets)
    full dataset -> TRAIN 70% | VALIDATION 15% | TEST 15%
    1. comparison     : GroupKFold(5) INSIDE the 70% TRAIN set.  Every fold fits interpolator / encoder / KNN Imputer on
                        its own fold-train rows and only transforms its held-out rows.  All 12 candidates see the same
                        folds and are scored on the same held-out rows.
    2. selection      : best direct, best Two-Stage and best overall = lowest MEAN outer-CV RMSE (model_selection.json/.md).
    3. --final        : fits the preprocessing + the SELECTED configurations on the whole 70% TRAIN set and scores the
                        15% VALIDATION set (secondary check, never used for fitting or selecting).
    4. --score-test   : (needs --final) also scores the 15% TEST set, selected configurations only.  Refused for
                        preliminary runs.  Use once, at the very end.

SELECTION BIAS (read this): picking the minimum of 12 noisy CV scores makes the winner's outer-CV score optimistic.
Nested CV is NOT implemented (cost).  The validation score is an independent second look, but it does not remove the
selection bias: if the configuration is changed after looking at it, it is no longer independent either.

12 CANDIDATES (all scored on identical rows of identical outer folds)
  Direct      : RandomForest, XGBoost, HistGradientBoosting, ExtraTrees, VotingOriginal, VotingET
  Two-Stage   : the same six as STAGE 2 estimator, with ONE fixed Stage 1 (see --stage1-base).
  VotingOriginal = equal-weight mean of RandomForest + XGBoost + HistGradientBoosting (exactly the previous "Voting").
  VotingET       = equal-weight mean of those three + ExtraTrees (separate candidate; the original is untouched).
  Voting = mean of the already-fitted members' predictions == sklearn VotingRegressor(weights=equal) without refitting.

TWO-STAGE
  Stage 1 predicts Voc, Jsc, FF from X (one estimator type for all three targets, fitted only on rows whose target is known;
  rows with a missing Voc/Jsc/FF are simply not used to fit that Stage 1 model but ARE predicted).  Stage 2 predicts PCE
  from X + the three Stage 1 predictions.  Stage 2 TRAINING rows get out-of-fold Stage 1 predictions (inner
  GroupKFold over papers, --inner-folds, default 5 as before); held-out rows get predictions from Stage 1 models fitted on
  the whole fold-train set.  Measured Voc/Jsc/FF of held-out rows are never used.  Stage 1 is computed once per fold
  and shared by all six Stage 2 candidates, so differences come from the Stage 2 estimator only.

Outputs (folder <--out>/<run_id>/ ; run_id hashes the configuration so different settings never overwrite each other)
  metrics_per_fold.csv, metrics_summary.csv, oof_predictions.csv, error_by_pce_bin.csv, parity_plots.png,
  cv_rmse_comparison.png, feature_importance_*.csv/.png (model-specific importances; NOT SHAP),
  model_selection.json / .md, run_config.json, [final_metrics.csv with --final]

    python run_models.py --data perla_final_filtered.pkl --cache-dir cv_cache --out results
    python run_models.py ... --final                          # + score the selected configs on VALIDATION
    python run_models.py ... --final --score-test             # + TEST (final evaluation, once)
    python run_models.py --final-only --selection-file results/<run_id>/model_selection.json --data ... [--score-test]
    python run_models.py ... --quick --max-folds 2            # smoke test: PRELIMINARY, never selection evidence
    python run_models.py --from-cache --cache-dir cv_cache --out results
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
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
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold

import perla_cv
import perla_encoding_by_claude as enc_module
from perla_cv import (N_SPLITS, SPLIT_SEED, TARGET, _transform_with, fit_final_preprocessors, iter_train_folds,
                      load_and_split, load_cached_folds, split_report)

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

warnings.filterwarnings("ignore", category=UserWarning)

AUX_TARGETS = ("Voc", "Jsc", "FF")                    # stage-1 targets of the two-stage model
N_INNER_SPLITS = 5                                    # inner GroupKFold of the Two-Stage model (unchanged: was hard-coded 5)
BASE_MODELS = ("RandomForest", "XGBoost", "HistGradientBoosting", "ExtraTrees")
VOTING_SETS = {                                       # name -> members; equal weights (see VOTING_WEIGHTS)
    "VotingOriginal": ("RandomForest", "XGBoost", "HistGradientBoosting"),     # == the previous "Voting"
    "VotingET": ("RandomForest", "XGBoost", "HistGradientBoosting", "ExtraTrees"),
}
VOTING_WEIGHTS = {m: 1.0 for m in BASE_MODELS}        # equal weights, NOT tuned on any held-out data
ESTIMATORS = BASE_MODELS + tuple(VOTING_SETS)         # the six estimators used directly and as Stage 2
PCE_BINS = [0, 5, 10, 15, 20, 25, 100]
TIE_RULE = "winner is 'not clearly separated' from a rival if the mean paired per-fold RMSE gap is below 1 paired SE"
OPTIMISM_NOTE = ("Selecting the minimum of several candidates from the same outer-CV scores makes the winner's CV score "
                 "optimistic (selection bias). Nested CV was NOT run (computational cost). The validation-set score is an "
                 "independent secondary check on data never used for fitting or ranking, but it does not remove the "
                 "selection bias and is itself only independent while the configuration is not changed after seeing it.")


# =========================================================================================
# MODELS  (fixed tuned settings; quick mode only reduces model size for smoke tests)
# =========================================================================================
def make_models(quick=False, seed=0, n_jobs=-1):
    """Create base estimators with the selected tuned hyperparameters.

    In quick mode, only estimator size is reduced for smoke tests; quick-mode
    results are preliminary and must not be used as model-selection evidence.
    Stage 1 keeps its separate settings in make_stage1_factory().
    """
    q = quick
    return {
        "RandomForest": lambda: RandomForestRegressor(
            n_estimators=25 if q else 300,
            min_samples_leaf=1,
            max_features=0.2,
            max_samples=None,
            n_jobs=n_jobs,
            random_state=seed,
        ),
        "XGBoost": lambda: XGBRegressor(
            n_estimators=80 if q else 600,
            learning_rate=0.03,
            max_depth=6,
            min_child_weight=1,
            subsample=1.0,
            colsample_bytree=0.5,
            reg_lambda=10,
            tree_method="hist",
            n_jobs=n_jobs,
            random_state=seed,
        ),
        "HistGradientBoosting": lambda: HistGradientBoostingRegressor(
            max_iter=50 if q else 300,
            learning_rate=0.03,
            max_leaf_nodes=31,
            min_samples_leaf=80,
            l2_regularization=1.0,
            max_features=0.5,
            early_stopping=False,
            random_state=seed,
        ),
        "ExtraTrees": lambda: ExtraTreesRegressor(
            n_estimators=25 if q else 300,
            min_samples_leaf=5,
            max_features=0.33,
            n_jobs=n_jobs,
            random_state=seed,
        ),
    }



def make_stage1_factory(base, models, quick, seed):
    """Stage-1 estimator type (the same type for Voc, Jsc and FF).  'hgb' = the previous default (a lighter HGBT)."""
    if base == "HistGradientBoosting":
        return lambda: HistGradientBoostingRegressor(
            max_iter=30 if quick else 300, learning_rate=0.06, max_leaf_nodes=31, early_stopping=False, random_state=seed)
    return models[base]


def stack(X, hat, add_product=False):
    cols = [X, hat]
    if add_product:
        cols.append(np.prod(hat, axis=1, keepdims=True))      # Voc_hat * Jsc_hat * FF_hat (PCE identity)
    return np.hstack(cols)


class Stage1Predictor:
    """Voc, Jsc, FF from X.  `oof_` = out-of-fold predictions for the rows it was fitted on (inner GroupKFold over
    papers); `predict` = predictions of models fitted on ALL known rows, for any new rows.  Rows whose target is NaN
    are not used to fit that target's model (they are still predicted)."""

    def __init__(self, factory, n_inner_splits=N_INNER_SPLITS):
        self.factory = factory
        self.n_inner_splits = n_inner_splits

    def fit(self, X, T, groups):
        groups = np.asarray(groups)
        n = len(X)
        n_splits = min(self.n_inner_splits, len(np.unique(groups)))
        inner = list(GroupKFold(n_splits=n_splits).split(np.zeros(n), groups=groups))
        for tr, va in inner:
            assert not (set(groups[tr]) & set(groups[va])), "a paper is in both sides of an inner Stage-1 fold"
        oof = np.full((n, len(AUX_TARGETS)), np.nan)
        self.models_ = {}
        for j, a in enumerate(AUX_TARGETS):
            ya = T[a].to_numpy(dtype=float)
            known = ~np.isnan(ya)
            for tr, va in inner:
                tr = tr[known[tr]]
                oof[va, j] = self.factory().fit(X[tr], ya[tr]).predict(X[va])
            idx = np.where(known)[0]
            self.models_[a] = self.factory().fit(X[idx], ya[idx])
        assert not np.isnan(oof).any()
        self.oof_ = oof
        self.n_inner_splits_used_ = n_splits
        return self

    def predict(self, X):
        return np.column_stack([self.models_[a].predict(X) for a in AUX_TARGETS])


class TwoStageRegressor:
    """Single Stage-2 estimator version (kept for compatibility): Stage 1 + Stage 2 PCE model."""

    def __init__(self, stage1_factory, stage2_factory, n_inner_splits=N_INNER_SPLITS, add_product=False):
        self.stage1_factory, self.stage2_factory = stage1_factory, stage2_factory
        self.n_inner_splits, self.add_product = n_inner_splits, add_product

    def fit(self, X, y, T, groups):
        self.s1_ = Stage1Predictor(self.stage1_factory, self.n_inner_splits).fit(X, T, groups)
        self.stage2_ = self.stage2_factory().fit(stack(X, self.s1_.oof_, self.add_product), y)
        return self

    def predict(self, X):
        return self.stage2_.predict(stack(X, self.s1_.predict(X), self.add_product))


def fit_bases(models, X, y, names):
    fitted, secs = {}, {}
    for n in names:
        t = time.time()
        fitted[n] = models[n]().fit(X, y)
        secs[n] = time.time() - t
    return fitted, secs


def predict_candidates(fitted, secs, X):
    """-> {estimator: (prediction, training seconds)} for every fitted base model and every voting set whose members
    are all fitted.  Voting = weighted mean of member predictions (== sklearn VotingRegressor, no refitting)."""
    base = {n: m.predict(X) for n, m in fitted.items()}
    out = {n: (base[n], secs[n]) for n in base}
    for v, members in VOTING_SETS.items():
        if all(m in base for m in members):
            w = np.array([VOTING_WEIGHTS[m] for m in members])
            out[v] = (np.average(np.column_stack([base[m] for m in members]), axis=1, weights=w),
                      sum(secs[m] for m in members))
    return out


def members_of(est):
    return VOTING_SETS.get(est, (est,))


def candidate_meta(stage1_label):
    meta = {}
    for e in ESTIMATORS:
        meta[f"Direct|{e}"] = dict(approach="Direct", stage1="-", estimator=e)
    for e in ESTIMATORS:
        meta[f"TwoStage[{stage1_label}]|{e}"] = dict(approach="TwoStage", stage1=stage1_label, estimator=e)
    return meta


# =========================================================================================
# METRICS / OUTPUT HELPERS
# =========================================================================================
def score(y, p):
    """R2 / MAE / RMSE / Pearson.  NaN (never an exception) when a metric is undefined: <2 rows or constant y / p."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    if not np.isfinite(p).all():
        raise ValueError("non-finite predictions")
    ok = len(y) >= 2 and np.var(y) > 0
    pear = float(np.corrcoef(y, p)[0, 1]) if (len(y) >= 2 and np.var(y) > 0 and np.var(p) > 0) else np.nan
    return dict(R2=r2_score(y, p) if ok else np.nan, MAE=mean_absolute_error(y, p),
                RMSE=mean_squared_error(y, p) ** 0.5, Pearson=pear)


def parity_plot(y, preds, path):
    names = list(preds)
    ncols = min(4, len(names))
    nrows = int(np.ceil(len(names) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 4.4 * nrows), squeeze=False, constrained_layout=True)
    lim = float(np.nanpercentile(y, 99.9)) * 1.05
    for ax, n in zip(axes.ravel(), names):
        s = score(y, preds[n])
        hb = ax.hexbin(y, preds[n], gridsize=45, extent=(0, lim, 0, lim), mincnt=1, bins="log", cmap="viridis")
        ax.plot([0, lim], [0, lim], "r--", lw=1)
        ax.set_title(f"{n.replace('|', chr(10))}\nR2={s['R2']:.3f} MAE={s['MAE']:.2f} RMSE={s['RMSE']:.2f}", fontsize=8)
        ax.set_xlabel("true PCE (%)")
        ax.set_ylabel("out-of-fold predicted PCE (%)")
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
    for ax in axes.ravel()[len(names):]:
        ax.axis("off")
    fig.colorbar(hb, ax=axes, shrink=0.6, label="devices (log)")
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def rmse_plot(summary, path):
    s = summary.iloc[::-1].copy()
    s["RMSE_std"] = s["RMSE_std"].fillna(0.0)          # a single fold has no std
    fig, ax = plt.subplots(figsize=(8, 0.42 * len(s) + 1.5))
    col = ["tab:blue" if a == "Direct" else "tab:orange" for a in s["approach"]]
    ax.barh(s.index, s["RMSE_mean"], xerr=s["RMSE_std"], color=col, capsize=3)
    ax.set_xlabel("mean outer-CV RMSE (PCE %-points), error bar = std over folds")
    ax.set_xlim(s["RMSE_mean"].min() * 0.9, (s["RMSE_mean"] + s["RMSE_std"]).max() * 1.03)
    ax.tick_params(axis="y", labelsize=7)
    ax.set_title("blue = Direct, orange = Two-Stage")
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def importance_plot(imp_tables, path, top=20):
    fig, axes = plt.subplots(1, len(imp_tables), figsize=(7 * len(imp_tables), 6.5), squeeze=False)
    for ax, (name, t) in zip(axes[0], imp_tables.items()):
        t = t.head(top)[::-1]
        ax.barh(t.index, t["mean_importance"])
        ax.set_title(f"{name}: top {top} (model-specific importance, mean over folds)")
        ax.tick_params(axis="y", labelsize=8)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


# =========================================================================================
# SELECTION  (mean outer-CV RMSE; aggregate over folds only - never one fold, never the test set)
# =========================================================================================
def _pair_vs_best(per_fold, best, other):
    a = per_fold[per_fold.model_id == best].set_index("fold")["RMSE"]
    b = per_fold[per_fold.model_id == other].set_index("fold")["RMSE"]
    d = (b - a).dropna()                                   # >0: `other` is worse than `best` on that fold
    se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else np.nan
    return dict(model_id=other, mean_rmse_gap=float(d.mean()), paired_se=float(se),
                folds_best_wins=int((d > 0).sum()), n_folds=int(len(d)),
                not_clearly_separated=bool(np.isfinite(se) and d.mean() < se))


def select_models(per_fold, summary, preliminary):
    def pick(ids):
        best = summary.loc[ids, "RMSE_mean"].idxmin()
        rivals = [_pair_vs_best(per_fold, best, o) for o in summary.loc[ids].sort_values("RMSE_mean").index if o != best]
        tied = [r["model_id"] for r in rivals if r["not_clearly_separated"]]
        cheapest = min([best] + tied, key=lambda m: summary.loc[m, "seconds_mean"])
        row = summary.loc[best]
        return dict(model_id=best, RMSE_mean=float(row.RMSE_mean), RMSE_std=float(row.RMSE_std),
                    MAE_mean=float(row.MAE_mean), R2_mean=float(row.R2_mean), Pearson_mean=float(row.Pearson_mean),
                    mean_train_seconds=float(row.seconds_mean), not_clearly_separated_from=tied,
                    cheapest_among_winner_and_tied=cheapest, comparison_with_rivals=rivals,
                    winner_is_clear=(len(tied) == 0) if all(np.isfinite(r["paired_se"]) for r in rivals) else None)

    direct = [i for i in summary.index if summary.loc[i, "approach"] == "Direct"]
    two = [i for i in summary.index if summary.loc[i, "approach"] == "TwoStage"]
    sel = dict(criterion="mean outer-CV RMSE (lower is better)", tie_rule=TIE_RULE,
               best_direct=pick(direct), best_two_stage=pick(two), best_overall=pick(direct + two),
               preliminary=bool(preliminary), caveat=OPTIMISM_NOTE)
    return sel


def selection_markdown(sel, summary, run_id):
    L = [f"# Model selection - run `{run_id}`", ""]
    if sel["preliminary"]:
        L += ["> **PRELIMINARY run (quick models and/or a subset of folds). Not valid as selection evidence.**", ""]
    L += [f"Criterion: {sel['criterion']}.  Tie rule: {sel['tie_rule']}.", ""]
    for key, title in (("best_direct", "Best direct model"), ("best_two_stage", "Best Two-Stage configuration"),
                       ("best_overall", "Best overall")):
        s = sel[key]
        L += [f"## {title}: `{s['model_id']}`",
              f"mean RMSE {s['RMSE_mean']:.4f} (std over folds {s['RMSE_std']:.4f}), MAE {s['MAE_mean']:.4f}, "
              f"R2 {s['R2_mean']:.4f}, Pearson {s['Pearson_mean']:.4f}, mean training time {s['mean_train_seconds']:.1f}s", ""]
        top = s["comparison_with_rivals"][:3]
        for r in top:
            flag = " (**not clearly separated**)" if r["not_clearly_separated"] else ""
            L.append(f"- vs `{r['model_id']}`: RMSE gap {r['mean_rmse_gap']:+.4f} +- {r['paired_se']:.4f} (paired SE), "
                     f"winner better on {r['folds_best_wins']}/{r['n_folds']} folds{flag}")
        if s["not_clearly_separated_from"]:
            L.append(f"- Not a clear winner: within 1 paired SE of {s['not_clearly_separated_from']}. The rule still "
                     f"selects the lowest mean RMSE; the cheapest of the winner and its near-ties is "
                     f"`{s['cheapest_among_winner_and_tied']}` - weigh stability, complexity and cost before preferring it.")
        L.append("")
    L += ["## All candidates (mean +- std over outer folds)", "",
          summary[["RMSE_mean", "RMSE_std", "MAE_mean", "R2_mean", "R2_std", "Pearson_mean", "seconds_mean"]]
          .round(4).to_markdown(), "", "## Limitation: selection bias", "", sel["caveat"], ""]
    return "\n".join(L)


# =========================================================================================
# FINAL STAGE: selected configurations only, fitted on the whole 70% TRAIN set
# =========================================================================================
def final_evaluation(a, split, models, s1_factory, sel, run_dir):
    roles = {}
    for r in ("best_direct", "best_two_stage", "best_overall"):
        roles.setdefault(sel[r]["model_id"], []).append(r)
    ids = list(roles)
    print(f"\n=== FINAL: preprocessing + selected models {ids} fitted on the 70% TRAIN set only ===")
    interp, enc, imp, X_final = fit_final_preprocessors(split.train_df, split.T_train)
    Xf = X_final.to_numpy(np.float32)
    yf = split.T_train[TARGET].to_numpy(float)
    eval_sets = {"validation": (split.val_df, split.T_val)}
    if a.score_test:
        eval_sets["test"] = (split.test_df, split.T_test)
    else:
        print("TEST set NOT scored (use --score-test for the final evaluation).")
    Xe = {k: _transform_with(interp, enc, imp, d, X_final.columns).to_numpy(np.float32) for k, (d, _) in eval_sets.items()}
    need_d = sorted({m for i in ids if i.startswith("Direct|") for m in members_of(i.split("|", 1)[1])})
    need_t = sorted({m for i in ids if i.startswith("TwoStage") for m in members_of(i.split("|", 1)[1])})
    preds = {k: {} for k in eval_sets}
    if need_d:
        fitted, secs = fit_bases(models, Xf, yf, need_d)
        for k in eval_sets:
            for est, (p, _) in predict_candidates(fitted, secs, Xe[k]).items():
                preds[k][f"Direct|{est}"] = p
    if need_t:
        s1 = Stage1Predictor(s1_factory, a.inner_folds).fit(Xf, split.T_train[list(AUX_TARGETS)],
                                                             split.g_train.to_numpy())
        fitted, secs = fit_bases(models, stack(Xf, s1.oof_, a.add_product), yf, need_t)
        for k in eval_sets:
            hat = s1.predict(Xe[k])                         # held-out rows: Stage-1 predictions only, never measured values
            for est, (p, _) in predict_candidates(fitted, secs, stack(Xe[k], hat, a.add_product)).items():
                preds[k][f"TwoStage[{a.stage1_label}]|{est}"] = p
    rows = []
    for k, (d, T) in eval_sets.items():
        for i in ids:
            rows.append(dict(split=k, model_id=i, selected_as="+".join(roles[i]), n=len(d),
                             **score(T[TARGET].to_numpy(float), preds[k][i])))
    fdf = pd.DataFrame(rows).round(4)
    fdf.to_csv(run_dir / "final_metrics.csv", index=False)
    print(fdf.to_string(index=False))
    md = run_dir / "model_selection.md"
    if md.exists():
        with open(md, "a") as f:
            f.write("\n## Final-stage scores of the selected configurations (fitted on the 70% TRAIN set)\n\n"
                    + fdf.to_markdown(index=False) + "\n\nThe validation rows are the secondary check described above; "
                    "they were not used to fit or rank anything.\n")
    return fdf


# =========================================================================================
# MAIN
# =========================================================================================
def _config_id(a, models, n_rows):
    d = dict(folds=a.folds, inner=a.inner_folds, seed=a.seed, split_seed=a.split_seed, quick=a.quick,
             max_folds=a.max_folds, stage1=a.stage1_base, add_product=a.add_product, n_rows=n_rows,
             voting=VOTING_SETS, weights=VOTING_WEIGHTS, from_cache=a.from_cache,
             models={n: repr(m().get_params()) for n, m in models.items()})
    return hashlib.sha1(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()[:8]


def _setup(a):
    base = {"hgb": "HistGradientBoosting", "xgb": "XGBoost", "rf": "RandomForest"}[a.stage1_base]
    models = make_models(a.quick, a.seed, a.n_jobs)
    a.stage1_label = f"S1={base}"
    return models, make_stage1_factory(base, models, a.quick, a.seed), base


def run_final_only(a):
    sel_path = Path(a.selection_file)
    sel = json.loads(sel_path.read_text())
    saved = json.loads((sel_path.parent / "run_config.json").read_text())["args"]
    for k in ("quick", "seed", "split_seed", "folds", "inner_folds", "stage1_base", "add_product"):
        setattr(a, k, saved[k])                                  # use exactly the settings of the selecting run
    print(f"--final-only: settings taken from {sel_path.parent.name}")
    if sel["preliminary"] and a.score_test:
        raise SystemExit("The selection comes from a PRELIMINARY run; --score-test is refused.")
    models, s1_factory, _ = _setup(a)
    split = load_and_split(a.data, random_state=a.split_seed)
    if len(split.train_df) != sel["n_train_rows"]:
        raise SystemExit("The dataset / split differs from the one used for the selection (train row count mismatch).")
    final_evaluation(a, split, models, s1_factory, sel, sel_path.parent)


def run(a):
    if not HAS_XGB:
        raise SystemExit("xgboost is required for the XGBoost candidates (pip install xgboost); "
                         "running without it would silently change the original Voting ensemble.")
    if a.score_test and not (a.final or a.final_only):
        raise SystemExit("--score-test requires --final (or --final-only)")
    if (a.final or a.final_only) and a.from_cache:
        raise SystemExit("--final / --final-only need the dataset (--data); not combinable with --from-cache")
    if a.final_only:
        return run_final_only(a)
    preliminary = bool(a.quick or (a.max_folds and a.max_folds < a.folds))
    if a.score_test and preliminary:
        raise SystemExit("--score-test is refused for a preliminary run (--quick / --max-folds).")
    models, s1_factory, s1_base = _setup(a)
    meta = candidate_meta(a.stage1_label)

    split = split_info = None
    if a.from_cache:
        assign = pd.read_csv(Path(a.cache_dir) / "fold_assignment.csv", dtype={"group": str})
        groups = assign["group"]
        df = assign
        fold_iter = load_cached_folds(a.cache_dir, a.folds)
        print("--from-cache: folds used as stored (internal consistency checked; dataset / code fingerprint NOT verified)")
    else:
        split = load_and_split(a.data, random_state=a.split_seed)
        split_info = split_report(split)
        df, groups = split.train_df, split.g_train
        fold_iter = iter_train_folds(split, a.folds, a.cache_dir)

    run_id = f"{'PRELIM_' if preliminary else ''}{a.folds}fold_seed{a.seed}_split{a.split_seed}_{_config_id(a, models, len(df))}"
    out = Path(a.out) / run_id
    out.mkdir(parents=True, exist_ok=True)
    print(f"run id: {run_id}  ->  {out}\nCV on the {len(df):,} TRAIN rows, {groups.nunique():,} papers"
          f"{'  [PRELIMINARY]' if preliminary else ''}")

    ids = list(meta)
    y_all = np.full(len(df), np.nan)
    oof = {i: np.full(len(df), np.nan) for i in ids}
    fold_of = np.full(len(df), -1)
    rows, s1_rows, importances, n_features = [], [], {}, {}
    feature_names_fold0 = None

    ckpt_dir = out / "fold_checkpoints"                  # a killed run resumes here (same run_id = same configuration)
    ckpt_dir.mkdir(exist_ok=True)
    for fold in itertools.islice(fold_iter, a.max_folds or None):
        ck = ckpt_dir / f"fold{fold.fold}.joblib"
        if ck.exists():
            c = joblib.load(ck)
            rows.extend(c["rows"]); s1_rows.extend(c["s1_rows"])
            for i in ids:
                oof[i][fold.val_idx] = c["oof"][i]
            for n, v in c["imp"].items():
                importances.setdefault(n, []).append(v)
            y_all[fold.val_idx] = c["y"]; fold_of[fold.val_idx] = fold.fold; n_features[fold.fold] = c["n_features"]
            feature_names_fold0 = feature_names_fold0 or c["features"]
            print(f"  fold {fold.fold}: restored from checkpoint")
            continue
        n_rows0, n_s10 = len(rows), len(s1_rows)
        Xtr, Xva = fold.X_fold_train.to_numpy(np.float32), fold.X_fold_val.to_numpy(np.float32)
        ytr, yva = fold.y_fold_train.to_numpy(float), fold.y_fold_val.to_numpy(float)
        gtr = groups.iloc[fold.train_idx].to_numpy()
        fold_of[fold.val_idx] = fold.fold
        y_all[fold.val_idx] = yva
        n_features[fold.fold] = Xtr.shape[1]
        if feature_names_fold0 is None:
            feature_names_fold0 = list(fold.X_fold_train.columns)

        def record(mid, pred, secs, s1_secs=0.0):
            oof[mid][fold.val_idx] = pred
            m = meta[mid]
            rows.append(dict(model_id=mid, approach=m["approach"], stage1=m["stage1"], stage2_or_direct_estimator=m["estimator"],
                             fold=fold.fold, n_eval=len(yva), seconds=round(secs, 2), stage1_seconds=round(s1_secs, 2),
                             **score(yva, pred)))

        # ---- direct ------------------------------------------------------------------------------
        fitted, secs = fit_bases(models, Xtr, ytr, BASE_MODELS)
        for est, (p, sec) in predict_candidates(fitted, secs, Xva).items():
            record(f"Direct|{est}", p, sec)
        for n in ("RandomForest", "XGBoost", "ExtraTrees"):
            importances.setdefault(n, []).append(pd.Series(fitted[n].feature_importances_, index=fold.X_fold_train.columns))

        # ---- two-stage: Stage 1 ONCE per fold, shared by all six Stage-2 candidates ------------------
        t = time.time()
        s1 = Stage1Predictor(s1_factory, a.inner_folds).fit(Xtr, fold.T_fold_train[list(AUX_TARGETS)], gtr)
        s1_sec = time.time() - t
        hat_val = s1.predict(Xva)                       # held-out rows: predictions only, measured Voc/Jsc/FF unused
        for j, aux in enumerate(AUX_TARGETS):           # evaluation-only diagnostic of Stage 1
            ya = fold.T_fold_val[aux].to_numpy(float)
            k = ~np.isnan(ya)
            s1_rows.append(dict(fold=fold.fold, target=aux, n_eval=int(k.sum()), **score(ya[k], hat_val[k, j])))
        fitted2, secs2 = fit_bases(models, stack(Xtr, s1.oof_, a.add_product), ytr, BASE_MODELS)
        for est, (p, sec) in predict_candidates(fitted2, secs2, stack(Xva, hat_val, a.add_product)).items():
            record(f"TwoStage[{a.stage1_label}]|{est}", p, s1_sec + sec, s1_sec)
        joblib.dump(dict(rows=rows[n_rows0:], s1_rows=s1_rows[n_s10:], oof={i: oof[i][fold.val_idx] for i in ids},
                         imp={n: l[-1] for n, l in importances.items()}, y=yva, n_features=Xtr.shape[1],
                         features=list(fold.X_fold_train.columns)), ck)
        best_now = min((r for r in rows if r["fold"] == fold.fold), key=lambda r: r["RMSE"])
        print(f"  fold {fold.fold}: {len(ids)} candidates scored; lowest RMSE this fold {best_now['RMSE']:.3f} "
              f"({best_now['model_id']}); stage1 {s1_sec:.0f}s")

    done = fold_of >= 0
    y = y_all[done]
    for i in ids:                                       # every candidate scored on exactly the same held-out rows
        assert np.isfinite(oof[i][done]).all() and np.isnan(oof[i][~done]).all(), f"{i}: different evaluation rows"

    # ---- metrics ---------------------------------------------------------------------------------
    per_fold = pd.DataFrame(rows)
    per_fold.to_csv(out / "metrics_per_fold.csv", index=False)
    cols = ["R2", "MAE", "RMSE", "Pearson", "seconds"]
    agg = per_fold.groupby("model_id")[cols].agg(["mean", "std"])          # std: ddof=1 over folds, NaN-skipping
    agg.columns = [f"{m}_{s}" for m, s in agg.columns]
    pooled = pd.DataFrame({i: score(y, oof[i][done]) for i in ids}).T.add_prefix("pooled_")
    info = pd.DataFrame({i: {k: meta[i][k] for k in ("approach", "stage1")} | {"estimator": meta[i]["estimator"]}
                         for i in ids}).T
    summary = info.join(agg).join(pooled).loc[ids]
    summary["n_folds"] = per_fold.groupby("model_id")["fold"].nunique()
    summary.round(4).to_csv(out / "metrics_summary.csv")
    pd.DataFrame(s1_rows).round(4).to_csv(out / "stage1_diagnostics.csv", index=False)

    # ---- selection ---------------------------------------------------------------------------------
    sel = select_models(per_fold, summary, preliminary)
    sel.update(run_id=run_id, n_train_rows=len(df), n_folds_run=int(per_fold.fold.nunique()),
               rule="argmin of mean outer-CV RMSE within Direct, within TwoStage, and over both")
    (out / "model_selection.json").write_text(json.dumps(sel, indent=2, default=str))
    (out / "model_selection.md").write_text(selection_markdown(sel, summary, run_id))

    # ---- predictions, error bins, plots, importances ---------------------------------------------------
    pred_df = pd.DataFrame({"row": np.where(done)[0], "fold": fold_of[done], "group": groups.to_numpy()[done], "PCE_true": y})
    if "perla_row_id" in df.columns:
        pred_df.insert(1, "perla_row_id", df["perla_row_id"].to_numpy()[done])
    for i in ids:
        pred_df[i] = oof[i][done]                       # column name = model / configuration identifier
    pred_df.to_csv(out / "oof_predictions.csv", index=False)
    binned = pd.cut(pred_df["PCE_true"], PCE_BINS, right=False)
    eb = []
    for i in ids:
        err = pred_df[i] - pred_df["PCE_true"]
        g = pd.DataFrame({"bin": binned, "abs": err.abs(), "err": err}).groupby("bin", observed=True)
        eb.append(pd.DataFrame({"model_id": i, "n": g.size(), "MAE": g["abs"].mean(), "bias": g["err"].mean()}).reset_index())
    pd.concat(eb).round(3).to_csv(out / "error_by_pce_bin.csv", index=False)
    parity_plot(y, {i: oof[i][done] for i in ids}, out / "parity_plots.png")
    rmse_plot(summary, out / "cv_rmse_comparison.png")
    imp_tables = {}
    kind = {"RandomForest": "impurity decrease (MDI)", "ExtraTrees": "impurity decrease (MDI)",
            "XGBoost": "xgboost feature_importances_ (gain)"}
    for n, lst in importances.items():
        t = pd.concat(lst, axis=1)
        tab = pd.DataFrame({"mean_importance": t.mean(axis=1), "std_importance": t.std(axis=1),
                            "n_folds_present": t.notna().sum(axis=1), "importance_type": kind[n]}
                           ).sort_values("mean_importance", ascending=False)
        tab.to_csv(out / f"feature_importance_{n}.csv")
        imp_tables[n] = tab
    importance_plot(imp_tables, out / "feature_importance_top20.png")

    # ---- run configuration ------------------------------------------------------------------------------
    cfg = dict(run_id=run_id, preliminary=preliminary, args=vars(a), n_rows=len(df), n_papers=int(groups.nunique()),
               folds_run=int(fold_of.max() + 1), outer_folds=a.folds, inner_folds=a.inner_folds,
               seeds=dict(model=a.seed, split=a.split_seed), split_fractions=perla_cv.SPLIT_FRACTIONS, split=split_info,
               grouping_column=enc_module.GROUP_COL, target_column=enc_module.TARGET_COLS[TARGET],
               aux_targets={k: enc_module.TARGET_COLS[k] for k in AUX_TARGETS},
               n_features_per_fold=n_features, features_fold0=feature_names_fold0,
               preprocessing=dict(order="band-gap interpolation -> PerlaEncoder -> PerlaKNNImputer (fit per fold on fold-train rows)",
                                  knn=perla_cv.KNN_KWARGS, encoder=perla_cv.ENCODER_KWARGS,
                                  interpolator=getattr(perla_cv.BAND_GAP_INTERPOLATOR_FACTORY, "__name__", "?"),
                                  vegard_enabled=perla_cv.USE_VEGARD_BAND_GAP_INTERPOLATION,
                                  verified_this_run=not a.from_cache),
               models={n: repr(m().get_params()) for n, m in models.items()},
               voting=dict(sets=VOTING_SETS, weights=VOTING_WEIGHTS, averaging="weighted mean of member predictions "
                           "(== sklearn VotingRegressor), weights equal and not tuned"),
               two_stage=dict(stage1_label=a.stage1_label, stage1_estimator=repr(s1_factory().get_params()),
                              stage1_targets=list(AUX_TARGETS), inner_folds_requested=a.inner_folds,
                              add_product=a.add_product, stage2_candidates=list(ESTIMATORS),
                              missing_aux_targets="rows with NaN Voc/Jsc/FF are excluded from fitting that target's Stage-1 "
                                                  "model only; they are still predicted and still used by Stage 2/direct models"),
               candidates=ids, feature_importance="model-specific (MDI / gain); SHAP is NOT implemented",
               mapie="NOT implemented", nested_cv="NOT implemented", selection_caveat=OPTIMISM_NOTE,
               test_scored=bool(a.score_test),
               versions=dict(sklearn=sklearn.__version__, pandas=pd.__version__, numpy=np.__version__))
    (out / "run_config.json").write_text(json.dumps(cfg, indent=2, default=str))

    print("\n=== mean outer-CV scores (sorted by RMSE) ===")
    print(summary.sort_values("RMSE_mean")[["RMSE_mean", "RMSE_std", "MAE_mean", "R2_mean", "Pearson_mean", "seconds_mean"]]
          .round(3).to_string())
    for r in ("best_direct", "best_two_stage", "best_overall"):
        print(f"{r:15s}: {sel[r]['model_id']}  RMSE {sel[r]['RMSE_mean']:.4f}  clear winner: {sel[r]['winner_is_clear']} (None = needs >=2 folds)")
    print(f"\nOutputs written to {out.resolve()}")

    if a.final:
        final_evaluation(a, split, models, s1_factory, sel, out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="perla_final_filtered.pkl")
    ap.add_argument("--cache-dir", default="cv_cache")
    ap.add_argument("--out", default="results", help="base folder; each configuration gets its own <run_id> subfolder")
    ap.add_argument("--folds", type=int, default=N_SPLITS, help="outer grouped CV folds (default 5)")
    ap.add_argument("--inner-folds", type=int, default=N_INNER_SPLITS, help="inner grouped CV folds for Stage 1 OOF (default 5)")
    ap.add_argument("--max-folds", type=int, default=0, help="stop after this many folds -> PRELIMINARY run")
    ap.add_argument("--quick", action="store_true", help="small models -> PRELIMINARY run (smoke test only)")
    ap.add_argument("--stage1-base", "--two-stage-base", dest="stage1_base", choices=["hgb", "xgb", "rf"], default="hgb",
                    help="estimator TYPE of the Stage-1 models (Voc/Jsc/FF).  It does NOT choose the Stage-2 model: all six "
                         "Stage-2 candidates are always evaluated.")
    ap.add_argument("--add-product", action="store_true", help="two-stage: add Voc_hat*Jsc_hat*FF_hat as a Stage-2 feature")
    ap.add_argument("--from-cache", action="store_true", help="read the folds from --cache-dir only (no --data, no encoder run)")
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--seed", type=int, default=0, help="model random seed")
    ap.add_argument("--split-seed", type=int, default=SPLIT_SEED, help="seed of the 70/15/15 paper split")
    ap.add_argument("--final", action="store_true",
                    help="after CV: fit the SELECTED configurations on the 70%% TRAIN set and score the VALIDATION set")
    ap.add_argument("--final-only", action="store_true", help="skip CV; run the final stage from --selection-file")
    ap.add_argument("--selection-file", default=None, help="model_selection.json of a finished (non-preliminary) run")
    ap.add_argument("--score-test", action="store_true",
                    help="(with --final) also score the 15%% TEST set, selected configurations only.  Use once, at the very end")
    args = ap.parse_args()
    if args.final_only and not args.selection_file:
        ap.error("--final-only needs --selection-file")
    run(args)
