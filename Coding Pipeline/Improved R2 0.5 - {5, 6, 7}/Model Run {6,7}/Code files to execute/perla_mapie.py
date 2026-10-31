"""
perla_mapie.py - MAPIE conformalized quantile regression (CQR) for the 12 candidate models
==========================================================================================
Supporting module of perla_explain_uq.py.  Nothing here takes part in model comparison / selection and nothing here
retrains or changes the models of run_models.py: the point predictors are built with rm.make_models() (same tuned
hyper-parameters), Stage 1 is rm.Stage1Predictor, Stage-2 features are rm.stack().

IDENTIFIERS (same naming as run_models.py candidate ids)
    Direct|RandomForest ... Direct|VotingET                 (bare names such as "VotingET" mean the Direct model)
    TwoStage[S1=<stage1 base>]|RandomForest ... |VotingET  (aliases: "TwoStage|VotingET", "TwoStage:VotingET")

WHAT "CQR" MEANS HERE
    Every candidate is reduced to three prediction functions of the *preprocessed feature matrix X*:
        lower(X)  ~ conditional (1-c)/2 quantile        upper(X) ~ conditional (1+c)/2 quantile        point(X)
    and these three are handed to MAPIE (ConformalizedQuantileRegressor, prefit=True) in MAPIE's required order
    [lower, upper, point].  MAPIE then computes the CQR conformity scores  s_i = max(lower_i - y_i, y_i - upper_i)  on
    the calibration set and widens / narrows [lower, upper] by the finite-sample conformal quantile of the scores.
    The conformal guarantee (marginal coverage >= c for data exchangeable with the calibration data) holds for ANY
    lower/upper functions fitted without the calibration data, so how the band is produced only affects interval
    efficiency (width), not validity.  How the band is produced differs per architecture:

    HistGradientBoosting : native quantile loss (two HGBT(loss="quantile") models) -> real CQR as before.
    XGBoost              : native quantile loss (two XGBRegressor(objective="reg:quantileerror")) -> real CQR.
    RandomForest         : sklearn forests have no quantile loss.  A quantile-regression-forest layer (Meinshausen 2006)
    ExtraTrees             is put on top of the SAME fitted forest used for point prediction: the leaf co-membership weights
                           of the training rows give a weighted empirical conditional distribution of y, whose
                           quantiles are the band.  (In-bag counts are used for bootstrap forests.)  The weighted mean
                           of this distribution reproduces forest.predict exactly; this is verified at run time.
    VotingOriginal/VotingET: the actual ensemble function is the equal-weight mean of the member point predictions
                           (rm.predict_candidates).  point(X) IS that mean.  Its band is the equal-weight average of the
                           members' lower (and of the upper) quantile predictions (a "quantile average"; there is no
                           native quantile of a mean of models).  Conformalization is then applied to this ensemble
                           band, so the calibrated coverage refers to the ensemble band as a whole, not to the members.
    Two-stage (all six)  : the CQR sees the raw feature matrix X.  Inside the three prediction functions the pipeline is
                           X -> Stage 1 (Voc, Jsc, FF predicted from X) -> stack(X, hat) -> Stage-2 point / quantile
                           models.  Stage-2 models are TRAINED on out-of-fold Stage-1 predictions (as in run_models.py)
                           and are calibrated / evaluated on rows whose Stage-1 inputs are Stage-1 PREDICTIONS (measured
                           Voc/Jsc/FF are never used).  Stage-1 uncertainty is therefore absorbed into the calibrated
                           residuals of the full pipeline; it is not propagated separately.

    The reported point prediction is the candidate's own point predictor (the one compared in run_models.py), not a
    median-quantile model.  (The previous HGB-only version reported the 0.5-quantile model; this does not change the band.)
    Quantile crossing (lower > upper) is repaired by swapping (rearrangement) before MAPIE sees the values and the
    fraction of crossed rows is reported.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

import run_models as rm

QUANTILE_TOOLS = {"HistGradientBoosting": "native HGBT quantile loss", "XGBoost": "native XGBoost reg:quantileerror",
                  "RandomForest": "quantile regression forest (leaf weights)", "ExtraTrees": "quantile regression forest (leaf weights)"}


# =========================================================================================
# IDENTIFIERS
# =========================================================================================
def stage1_label(stage1_base_key):
    base = {"hgb": "HistGradientBoosting", "xgb": "XGBoost", "rf": "RandomForest"}[stage1_base_key]
    return f"S1={base}"


def all_model_ids(label):
    return [f"Direct|{e}" for e in rm.ESTIMATORS] + [f"TwoStage[{label}]|{e}" for e in rm.ESTIMATORS]


def parse_model_id(raw, label):
    """User string -> canonical run_models.py candidate id.  Raises ValueError listing the valid names."""
    s = str(raw).strip()
    valid = all_model_ids(label)
    if s in valid:
        return s
    est_lookup = {e.lower(): e for e in rm.ESTIMATORS}
    approach, est = "Direct", s
    for sep in ("|", ":"):
        if sep in s:
            head, est = s.split(sep, 1)
            h = head.strip().lower()
            if h == "direct":
                approach = "Direct"
            elif h.startswith("twostage") or h.startswith("two-stage") or h in ("ts", "two_stage"):
                if "[" in head and head.strip() != f"TwoStage[{label}]":
                    raise ValueError(f"'{raw}': Stage-1 setting differs from the current --stage1-base ({label}); "
                                     f"change --stage1-base instead of the model id.")
                approach = "TwoStage"
            else:
                approach = None
            break
    else:
        if s.lower().startswith("twostage_") or s.lower().startswith("two-stage_"):
            approach, est = "TwoStage", s.split("_", 1)[1]
    key = est.strip().lower()
    if approach is None or key not in est_lookup:
        raise ValueError(f"unknown model identifier '{raw}'.  Valid: bare names {list(rm.ESTIMATORS)} (= Direct), "
                         f"'TwoStage:<name>', or the full ids {valid}")
    e = est_lookup[key]
    return f"Direct|{e}" if approach == "Direct" else f"TwoStage[{label}]|{e}"


def split_model_id(model_id):
    head, est = model_id.split("|", 1)
    return ("TwoStage" if head.startswith("TwoStage") else "Direct"), est


def safe_tag(model_id):
    return model_id.replace("|", "__").replace("[", "_").replace("]", "").replace("=", "-")


# =========================================================================================
# SMALL UTILITIES
# =========================================================================================
def _xkey(X):
    X = np.ascontiguousarray(X)
    return (X.shape, hashlib.blake2b(X.tobytes(), digest_size=16).hexdigest())


class _Memo:
    """Remember the result of fn(X) for the last few distinct X (so lower/upper/point share one computation)."""

    def __init__(self, fn, keep=4):
        self.fn, self.keep, self.store = fn, keep, {}

    def __call__(self, X):
        k = _xkey(X)
        if k not in self.store:
            if len(self.store) >= self.keep:
                self.store.pop(next(iter(self.store)))
            self.store[k] = self.fn(X)
        return self.store[k]


class PrefitPredictor:
    """Minimal fitted 'estimator' for MAPIE prefit mode: MAPIE only needs .fit/.predict and a fitted flag."""

    def __init__(self, fn, name):
        self.fn, self.name = fn, name
        self.is_fitted_ = True

    def fit(self, *a, **k):
        raise RuntimeError("PrefitPredictor is already fitted")

    def predict(self, X, **k):
        return np.asarray(self.fn(np.asarray(X)), dtype=float)

    def __sklearn_is_fitted__(self):
        return True

    def get_params(self, deep=True):
        return {"name": self.name}


# =========================================================================================
# QUANTILE REGRESSION FOREST LAYER ON A FITTED sklearn FOREST (RandomForest / ExtraTrees)
# =========================================================================================
class ForestQuantiles:
    """Meinshausen quantile regression forest on top of an already-fitted RandomForest / ExtraTrees.

    weight of training row i for query x = (1/T) * sum_t  c_ti * 1[i in leaf_t(x)] / sum_j c_tj 1[j in leaf_t(x)]
    with c_ti = in-bag count of row i in tree t (1 when the forest is not bootstrapped).  The weighted mean of y under
    these weights equals forest.predict(x) exactly; the quantiles are weighted empirical quantiles (lower interpolation).
    """

    def __init__(self, forest, X_train, y_train, chunk=400):
        self.forest, self.y, self.chunk = forest, np.asarray(y_train, float), chunk
        n = len(self.y)
        leaves = forest.apply(X_train)                                  # (n, T) terminal-node ids
        self.T = leaves.shape[1]
        if forest.bootstrap:
            from sklearn.ensemble._forest import _generate_sample_indices, _get_n_samples_bootstrap
            nb = _get_n_samples_bootstrap(n, forest.max_samples)
            counts = [np.bincount(_generate_sample_indices(t.random_state, n, nb), minlength=n) for t in forest.estimators_]
            self.bootstrap_weights = "in-bag counts"
        else:
            counts = [None] * self.T
            self.bootstrap_weights = "all training rows (bootstrap=False)"
        self.trees = []
        for t in range(self.T):
            c = np.ones(n) if counts[t] is None else counts[t].astype(float)
            rows = np.where(c > 0)[0]
            leaf = leaves[rows, t]
            o = np.argsort(leaf, kind="stable")
            rows, leaf = rows[o].astype(np.int32), leaf[o]
            n_nodes = int(forest.estimators_[t].tree_.node_count)
            size = np.bincount(leaf, minlength=n_nodes).astype(np.int64)
            start = np.concatenate([[0], np.cumsum(size)[:-1]])
            tot = np.bincount(leaf, weights=c[rows], minlength=n_nodes)
            self.trees.append((rows, c[rows], start, size, tot))

    def _gather(self, Xq):
        L = self.forest.apply(Xq)
        m = len(Xq)
        Q, R, W = [], [], []
        for t, (rows, cw, start, size, tot) in enumerate(self.trees):
            leaf = L[:, t]
            sz = size[leaf]
            total = int(sz.sum())
            qid = np.repeat(np.arange(m), sz)
            off = np.arange(total) - np.repeat(np.cumsum(sz) - sz, sz)
            pos = np.repeat(start[leaf], sz) + off
            Q.append(qid)
            R.append(rows[pos])
            W.append(cw[pos] / np.repeat(tot[leaf], sz) / self.T)
        return np.concatenate(Q), np.concatenate(R), np.concatenate(W)

    def predict(self, X, quantiles, return_mean=False):
        """-> array (len(X), len(quantiles)) [, weighted mean]."""
        X = np.asarray(X)
        out = np.empty((len(X), len(quantiles)))
        mean = np.empty(len(X))
        for s in range(0, len(X), self.chunk):
            Xq = X[s:s + self.chunk]
            Q, R, W = self._gather(Xq)
            Y = self.y[R]
            o = np.lexsort((Y, Q))
            Q, Y, W = Q[o], Y[o], W[o]
            C = np.cumsum(W)                                           # each query's weights sum to 1
            for j, q in enumerate(quantiles):
                idx = np.searchsorted(C, np.arange(len(Xq)) + q - 1e-9, side="left")
                out[s:s + len(Xq), j] = Y[np.minimum(idx, len(Y) - 1)]
            mean[s:s + len(Xq)] = np.bincount(Q, weights=W * Y, minlength=len(Xq))
        return (out, mean) if return_mean else out


# =========================================================================================
# MEMBER BANK: fitted point model + quantile model(s) per base estimator, shared by every selected candidate
# =========================================================================================
class MemberBank:
    """Fits (once, lazily, on TRAIN only) the point and quantile models of the base estimators in one *context*
    ('Direct' = features X; 'TwoStage' = features stack(X, Stage-1 predictions)); every candidate that needs the
    same member reuses it."""

    def __init__(self, a, X_train, y_train, T_train, g_train, confidence):
        self.a, self.X, self.y, self.T, self.g, self.confidence = a, X_train, y_train, T_train, g_train, confidence
        self.q_lo, self.q_hi = (1 - confidence) / 2, (1 + confidence) / 2
        self.models = rm.make_models(a.quick, a.seed, a.n_jobs)
        self.s1 = None
        self.point, self.quant, self.train_matrix, self.timing, self.notes = {}, {}, {}, {}, {}

    # ---- contexts -------------------------------------------------------------------------------------------
    def _stage1(self):
        if self.s1 is None:
            base = {"hgb": "HistGradientBoosting", "xgb": "XGBoost", "rf": "RandomForest"}[self.a.stage1_base]
            f = rm.make_stage1_factory(base, self.models, self.a.quick, self.a.seed)
            t = time.time()
            self.s1 = rm.Stage1Predictor(f, self.a.inner_folds).fit(self.X, self.T, self.g)
            self.timing["stage1_fit_s"] = round(time.time() - t, 1)
            self.timing["stage1_inner_folds_used"] = int(self.s1.n_inner_splits_used_)
            print(f"  Stage 1 ({base}; inner folds {self.s1.n_inner_splits_used_}) fitted in {self.timing['stage1_fit_s']:.0f}s")
        return self.s1

    def train_X(self, ctx):
        if ctx == "Direct":
            return self.X
        if "TwoStage" not in self.train_matrix:
            self.train_matrix["TwoStage"] = rm.stack(self.X, self._stage1().oof_, self.a.add_product)   # OUT-OF-FOLD Stage-1 predictions
        return self.train_matrix["TwoStage"]

    def features(self, ctx, X):
        """raw X -> the matrix the Stage-2 / direct models see.  TwoStage: Stage-1 PREDICTIONS (never measured Voc/Jsc/FF)."""
        if ctx == "Direct":
            return X
        return self._feat_two(X)

    @property
    def _feat_two(self):
        if not hasattr(self, "_feat_two_memo"):
            self._feat_two_memo = _Memo(lambda X: rm.stack(X, self._stage1().predict(X), self.a.add_product), keep=3)
        return self._feat_two_memo

    # ---- members -----------------------------------------------------------------------------------------------
    def point_model(self, ctx, base):
        k = (ctx, base)
        if k not in self.point:
            t = time.time()
            self.point[k] = self.models[base]().fit(self.train_X(ctx), self.y)
            self.timing[f"{ctx}|{base}|point_fit_s"] = round(time.time() - t, 1)
            print(f"  [{ctx}] {base}: point model fitted in {time.time() - t:.0f}s")
        return self.point[k]

    def quantile_fn(self, ctx, base):
        """-> memoised f(X_raw) = array (n, 2) of [lower, upper] quantile predictions of base member in context."""
        k = (ctx, base)
        if k in self.quant:
            return self.quant[k]
        Xc, qs = self.train_X(ctx), [self.q_lo, self.q_hi]
        t = time.time()
        if base in ("RandomForest", "ExtraTrees"):
            forest = self.point_model(ctx, base)
            fq = ForestQuantiles(forest, Xc, self.y)
            self.notes[f"{ctx}|{base}"] = f"quantile regression forest; weights: {fq.bootstrap_weights}"
            raw = lambda X, fq=fq: fq.predict(X, qs)                                          # noqa: E731
        else:
            ms = []
            for q in qs:
                est = self.models[base]()
                est.set_params(**({"loss": "quantile", "quantile": q} if base == "HistGradientBoosting"
                                  else {"objective": "reg:quantileerror", "quantile_alpha": q}))
                ms.append(est.fit(Xc, self.y))
            self.notes[f"{ctx}|{base}"] = QUANTILE_TOOLS[base]
            raw = lambda X, ms=ms: np.column_stack([m.predict(X) for m in ms])                # noqa: E731
        self.timing[f"{ctx}|{base}|quantile_fit_s"] = round(time.time() - t, 1)
        print(f"  [{ctx}] {base}: quantile models ({QUANTILE_TOOLS[base]}) ready in {time.time() - t:.0f}s")
        self.quant[k] = _Memo(lambda Xraw, raw=raw, ctx=ctx: raw(self.features(ctx, Xraw)))
        return self.quant[k]

    # ---- one candidate = (point, lower, upper) functions of the RAW feature matrix --------------------------------
    def build(self, model_id):
        approach, est = split_model_id(model_id)
        ctx = approach
        members = rm.members_of(est)
        w = np.array([rm.VOTING_WEIGHTS[m] for m in members], float)
        w = w / w.sum()
        pts = [self.point_model(ctx, m) for m in members]
        qfs = [self.quantile_fn(ctx, m) for m in members]

        def point(X):
            F = self.features(ctx, X)
            return np.average(np.column_stack([p.predict(F) for p in pts]), axis=1, weights=w)   # == rm.predict_candidates

        def raw_band(X):                                     # equal-weight average of member [lower, upper] predictions
            return sum(wi * qf(X) for wi, qf in zip(w, qfs))

        band_m, point_m = _Memo(lambda X: np.sort(raw_band(X), axis=1)), _Memo(point)       # sort = repair of quantile crossing
        crossing_m = _Memo(lambda X: float((raw_band(X)[:, 0] > raw_band(X)[:, 1]).mean()))
        meta = dict(model_id=model_id, approach=approach, estimator=est, members=list(members),
                    band_construction=("single model: " if len(members) == 1 else "equal-weight average of member quantile predictions: ")
                    + "; ".join(f"{m} -> {QUANTILE_TOOLS[m]}" for m in members),
                    point_prediction=("candidate's own point predictor (mean of member predictions)" if len(members) > 1
                                      else "candidate's own point predictor (as in run_models.py)"),
                    cqr_flavour=("CQR with native quantile models" if all(m in ("HistGradientBoosting", "XGBoost") for m in members)
                                 else "CQR on a quantile-regression-forest band" if len(members) == 1
                                 else "CQR on an averaged quantile band (members: native quantile models and/or quantile-regression forests)"))
        return dict(point=point_m, band=band_m, crossing=crossing_m, meta=meta, ctx=ctx, members=members, pts=pts)


# =========================================================================================
# CQR RUN + METRICS
# =========================================================================================
def mapie_cqr(lower_fn, upper_fn, point_fn, confidence, X_cal, y_cal):
    from mapie.regression import ConformalizedQuantileRegressor
    cqr = ConformalizedQuantileRegressor(estimator=[lower_fn, upper_fn, point_fn], confidence_level=confidence, prefit=True)
    cqr.conformalize(X_cal, y_cal)                          # MAPIE: ONE conformalize call per object
    return cqr


def _textbook_q(scores, alpha_tail):
    """Textbook split-conformal quantile: the ceil((n+1)(1-alpha_tail))-th smallest score (inf if rank > n)."""
    n = len(scores)
    k = math.ceil((n + 1) * (1 - alpha_tail) - 1e-9)
    return float(np.sort(scores)[k - 1]) if k <= n else float("inf")


def _mapie_q(scores, alpha_tail):
    """MAPIE's convention: np.quantile(scores, (n+1)(1-alpha_tail)/n, method='higher').  Equals the textbook rank, or one
    rank higher when (n+1)(1-alpha_tail) is (numerically) an integer - i.e. never less conservative than the textbook rule."""
    n = len(scores)
    return float(np.quantile(scores, min(1.0, (n + 1) * (1 - alpha_tail) / n), method="higher"))


def manual_cqr_check(lo_cal, hi_cal, y_cal, confidence, symmetric=False):
    """Independent re-computation of the CQR correction WITHOUT MAPIE.  Interval = [lower - q_lo, upper + q_hi].
    asymmetric (MAPIE default, Romano et al. 2019): q_lo / q_hi = conformal quantiles of (lower - y) and (y - upper) at tail
    level alpha/2 each; symmetric: one q from max(lower - y, y - upper) at level alpha.
    Returns dict(mapie_convention=(q_lo, q_hi), textbook=(q_lo, q_hi))."""
    alpha = 1 - confidence
    s_lo, s_hi = lo_cal - y_cal, y_cal - hi_cal
    if symmetric:
        s = np.maximum(s_lo, s_hi)
        return dict(mapie_convention=(_mapie_q(s, alpha),) * 2, textbook=(_textbook_q(s, alpha),) * 2)
    return dict(mapie_convention=(_mapie_q(s_lo, alpha / 2), _mapie_q(s_hi, alpha / 2)),
                textbook=(_textbook_q(s_lo, alpha / 2), _textbook_q(s_hi, alpha / 2)))


def interval_metrics(y, lo, hi, point, confidence, groups=None, n_boot=500, seed=0):
    alpha = 1 - confidence
    inside = (y >= lo) & (y <= hi)
    width = hi - lo
    score = width + (2 / alpha) * (lo - y) * (y < lo) + (2 / alpha) * (y - hi) * (y > hi)        # Winkler / interval score
    err = np.abs(y - point)
    m = dict(n=int(len(y)), target_coverage=confidence, coverage=float(inside.mean()),
             miss_below=float((y < lo).mean()), miss_above=float((y > hi).mean()),
             mean_width=float(width.mean()), median_width=float(np.median(width)),
             width_p10=float(np.percentile(width, 10)), width_p90=float(np.percentile(width, 90)),
             interval_score=float(score.mean()),
             spearman_width_vs_abs_error=float(pd.Series(width).corr(pd.Series(err), method="spearman")),
             **{f"point_{k}": float(v) for k, v in rm.score(y, point).items()})
    if groups is not None and len(np.unique(groups)) > 1:                    # coverage uncertainty with PAPER as the resampling unit
        d = pd.DataFrame({"g": groups, "in": inside.astype(float)}).groupby("g")["in"].agg(["sum", "count"])
        s, c = d["sum"].to_numpy(), d["count"].to_numpy()
        rng = np.random.RandomState(seed)
        cov = np.array([(s[i].sum() / c[i].sum()) for i in (rng.randint(0, len(s), len(s)) for _ in range(n_boot))])
        m.update(coverage_paper_bootstrap_95ci=[float(np.percentile(cov, 2.5)), float(np.percentile(cov, 97.5))],
                 n_papers=int(len(s)))
    return m


def coverage_by_bin(y, lo, hi):
    b = pd.cut(pd.Series(y), rm.PCE_BINS, right=False)
    d = pd.DataFrame({"bin": b, "inside": (y >= lo) & (y <= hi), "width": hi - lo})
    g = d.groupby("bin", observed=True)
    return pd.DataFrame({"n": g.size(), "coverage": g["inside"].mean(), "mean_width": g["width"].mean()}).reset_index()


def coverage_by_width_tercile(y, lo, hi):
    w = hi - lo
    t = pd.qcut(pd.Series(w), 3, labels=["narrow", "medium", "wide"], duplicates="drop")
    d = pd.DataFrame({"width_tercile": t, "inside": (y >= lo) & (y <= hi), "width": w})
    g = d.groupby("width_tercile", observed=True)
    return pd.DataFrame({"n": g.size(), "coverage": g["inside"].mean(), "mean_width": g["width"].mean()}).reset_index()


def prefit_trio(cand):
    """[lower, upper, point] PrefitPredictors (MAPIE's required order) for a candidate built by MemberBank.build."""
    return [PrefitPredictor(lambda X: cand["band"](X)[:, 0], "lower"), PrefitPredictor(lambda X: cand["band"](X)[:, 1], "upper"),
            PrefitPredictor(lambda X: cand["point"](X), "point")]


def predict_all(cand, cqr, X):
    """conformalized interval + raw (uncalibrated) quantile band + point prediction for rows X."""
    point, pis = cqr.predict_interval(X)
    raw = cand["band"](X)
    return np.asarray(point), pis[:, 0, 0], pis[:, 1, 0], raw[:, 0], raw[:, 1]
