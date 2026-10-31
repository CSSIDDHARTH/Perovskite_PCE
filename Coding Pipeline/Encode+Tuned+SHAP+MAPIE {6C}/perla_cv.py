"""
perla_cv.py
===========
Leakage-free workflow:  group-aware 70/15/15 split  ->  GroupKFold inside the 70% TRAIN set
->  [band-gap interpolation]  ->  PerlaEncoder  ->  KNN Imputer  ->  model.   Models plug in afterwards.

NOTE: "KNN Imputer" everywhere below means a nearest-neighbour *imputer* for missing cells
(PerlaKNNImputer / sklearn KNNImputer) - NOT a KNN regressor or KNN classifier.

                                 FULL (filtered) DATASET
                                          |
                      group-aware split on Ref_DOI_number (fixed seed)
                                          |
              +---------------------------+---------------------------+
              |                           |                           |
         TRAIN  ~70%               VALIDATION ~15%                TEST ~15%
              |                  (reserved for MAPIE CQR          (untouched until the
              |                   calibration only)                final evaluation)
        GroupKFold  (inside TRAIN only)
              |
    +---------+----------+
    |                    |
 fold TRAIN          fold VAL          <- X_fold_train / X_fold_val  (internal CV only)
    |                    |
 interpolator.fit    interpolator.transform
 encoder.fit_transform   encoder.transform
 KNN Imputer.fit     KNN Imputer.transform

 FINAL (after model selection):  fit interpolator -> encoder -> KNN Imputer on the 70% TRAIN set only,
 then TRANSFORM validation (-> X_mapie_val) and test (-> X_test).  Nothing is ever fitted on them.

Preprocessing order (per fit):   band-gap interpolation  ->  encoding  ->  KNN Imputer
  * interpolation must come first because it writes the raw band-gap column(s) (e.g. band_gap__mean) that
    NumericBlock of PerlaEncoder reads;
  * the KNN Imputer must come last because it operates on the encoded numeric matrix.

BAND-GAP INTERPOLATION: neither this file nor perla_encoding_by_claude.py contains one (see `BandGapInterpolator`
below).  A clearly-marked hook is in place; by default it is a no-op and a warning is printed.  KNN Imputation is
NOT used as a substitute.

Quick start
-----------
    from perla_cv import load_and_split, iter_train_folds, prepare_final_datasets, split_report

    split = load_and_split("perla_final_filtered.pkl", random_state=42)
    split_report(split)
    for fold in iter_train_folds(split, n_splits=8, cache_dir="cv_cache"):     # TRAIN (70%) only
        model.fit(fold.X_fold_train, fold.y_fold_train)
        pred = model.predict(fold.X_fold_val)                                    # compare with fold.y_fold_val

    final = prepare_final_datasets(split)       # only after model selection
    # final.X_final_train / final.X_mapie_val / final.X_test   (+ y_*, T_*)

Command line:
    python perla_cv.py --data perla_final_filtered.pkl --folds 5 --cache-dir cv_cache [--final final_data.joblib]
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import time
import warnings
from dataclasses import dataclass, replace
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import KNNImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import RobustScaler

import perla_encoding_by_claude as enc_module
from perla_encoding_by_claude import PerlaEncoder, TARGET_COLS, get_groups, get_targets

# =========================================================================================
# CONFIGURATION
# =========================================================================================
TARGET = "PCE"                       # the column of get_targets(df) that is predicted
N_SPLITS = 5                         # GroupKFold folds INSIDE the 70% train set
SPLIT_FRACTIONS = (0.70, 0.15, 0.15)  # train / validation (MAPIE CQR calibration) / test
SPLIT_SEED = 42
KNN_KWARGS = dict(n_neighbors=5, scale=True, engine="fast")   # KNN *Imputer* settings
ENCODER_KWARGS: dict = {}            # e.g. dict(min_count=30, te_smoothing=20.0)

# Band-gap interpolation hook.  Set to a zero-argument callable returning a BandGapInterpolator
# (e.g. BAND_GAP_INTERPOLATOR_FACTORY = MyProjectInterpolator) once the project's interpolation exists.
# Candidate Vegard + fitted-bowing interpolation (perla_bandgap_vegard.py).  OFF by default: in the A/B experiment
# it did not improve CV or validation performance (see vegard_diagnostics/).  Set True to enable it; the factory
# is resolved just below the BandGapInterpolator classes.  When False the stage is an explicit pass-through and
# the KNN Imputer handles every missing band gap.
USE_VEGARD_BAND_GAP_INTERPOLATION = False


# =========================================================================================
# BAND-GAP INTERPOLATION  (hook only - the project's method is NOT in the supplied code)
# =========================================================================================
class BandGapInterpolator(BaseEstimator, TransformerMixin):
    """Interface for the project's band-gap interpolation (domain-specific; NOT the KNN Imputer).

    Contract:
      * fit(df)       : receives ONLY the rows allowed for fitting (fold-train rows, or the 70% train set).
                        If the method is deterministic per row/composition, fit does nothing and
                        `data_dependent = False`.  If it learns anything from data, set `data_dependent = True`.
      * transform(df) : returns a DataFrame with the SAME index and all original columns (it may overwrite the raw
                        band-gap column(s), e.g. `band_gap__mean`).  It must never touch target columns.
    The pipeline always calls fit on training rows only and transform on every other set, so a data-dependent
    implementation is leakage-safe by construction."""
    data_dependent = True

    def fit(self, df, y=None):
        raise NotImplementedError

    def transform(self, df):
        raise NotImplementedError


class NoBandGapInterpolation(BandGapInterpolator):
    """Placeholder used while no interpolation implementation exists: returns the data unchanged."""
    data_dependent = False

    def fit(self, df, y=None):
        return self

    def transform(self, df):
        return df


if USE_VEGARD_BAND_GAP_INTERPOLATION:
    from perla_bandgap_vegard import VegardBowingBandGapInterpolator   # bowing fitted on fit() rows only
    BAND_GAP_INTERPOLATOR_FACTORY = VegardBowingBandGapInterpolator
else:
    BAND_GAP_INTERPOLATOR_FACTORY = NoBandGapInterpolation           # explicit pass-through, no warning


_WARNED_NO_INTERP = False


def _make_interpolator(factory):
    global _WARNED_NO_INTERP
    if factory is None:
        if not _WARNED_NO_INTERP:
            warnings.warn("No band-gap interpolation implementation is configured (neither perla_cv.py nor "
                          "perla_encoding_by_claude.py contains one): the interpolation stage is a NO-OP. "
                          "Set BAND_GAP_INTERPOLATOR_FACTORY. KNN Imputation is NOT a substitute.")
            _WARNED_NO_INTERP = True
        return NoBandGapInterpolation()
    interp = factory()
    if not hasattr(interp, "fit") or not hasattr(interp, "transform"):
        raise TypeError("BAND_GAP_INTERPOLATOR_FACTORY must return an object with fit(df) / transform(df).")
    return interp


def _interp_transform(interp, df):
    out = interp.transform(df)
    if not isinstance(out, pd.DataFrame) or not out.index.equals(df.index):
        raise ValueError("Band-gap interpolator must return a DataFrame with the same index as its input.")
    lost = set(df.columns) - set(out.columns)
    if lost:
        raise ValueError(f"Band-gap interpolator dropped columns: {sorted(lost)}")
    return out


# =========================================================================================
# KNN IMPUTER
# =========================================================================================
def _nan_euclidean(Q, A0, A0sq, obsA):
    """sklearn's nan_euclidean distance (squared=False) between rows of Q (may contain NaN) and the training
    rows, via matrix products.  A0 = train with NaN -> 0, A0sq = A0**2, obsA = train observed mask (float)."""
    obsQ = ~np.isnan(Q)
    Q0 = np.where(obsQ, Q, 0.0)
    obsQf = obsQ.astype(np.float64)
    d2 = (Q0 ** 2) @ obsA.T + obsQf @ A0sq.T - 2.0 * (Q0 @ A0.T)     # sum over coordinates present in BOTH rows
    np.maximum(d2, 0.0, out=d2)
    cnt = obsQf @ obsA.T                                              # number of shared coordinates
    with np.errstate(divide="ignore", invalid="ignore"):
        d = np.sqrt(d2 * (Q.shape[1] / cnt))                          # sklearn rescales to all p coordinates
    d[cnt == 0] = np.nan                                              # nothing in common -> unusable
    return d


class PerlaKNNImputer(BaseEstimator, TransformerMixin):
    """KNN Imputer for the encoded feature matrix (a KNN *imputer*, not a KNN regressor/classifier).

    A missing cell is filled with the mean of that column over the `n_neighbors` nearest TRAINING rows
    that have the column observed.  Distance = sklearn's nan_euclidean (only coordinates present in both
    rows count).

    scale=True : distances are computed on a robust-scaled copy of the data (median / IQR learned on the
                 TRAIN fold, NaN-aware) and only the missing cells are written back in the original units.
                 Without it, columns measured in nm or seconds (up to 5e6) decide who the "nearest" rows are
                 and the 0/1 indicator columns hardly count.  The model still receives original-unit values.
    engine     : "sklearn" = sklearn.impute.KNNImputer (reference implementation, slow here)
                 "fast"    = same maths, two changes: (1) the row-to-row distances are computed once per chunk
                             of rows and reused for every column (sklearn recomputes them for each of the ~66
                             columns that contain NaN); (2) the nan-euclidean distance is evaluated with four
                             matrix products instead of sklearn's nan_euclidean_distances (about 20x faster).
                             Gives the same imputed values as the sklearn engine (checked, see the reply).
    """

    def __init__(self, n_neighbors=5, scale=True, engine="fast", chunk_rows=500):
        self.n_neighbors = n_neighbors
        self.scale = scale
        self.engine = engine
        self.chunk_rows = chunk_rows

    # ---------------------------------------------------------------- scaling
    def _scale(self, A):
        return (A - self.center_) / self.scale_

    def _unscale(self, S):
        return S * self.scale_ + self.center_

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("PerlaKNNImputer expects the DataFrame returned by PerlaEncoder.")
        self.columns_ = list(X.columns)
        A = X.to_numpy(dtype=float)
        if not np.isfinite(A[~np.isnan(A)]).all():
            raise ValueError("X contains +-inf.")
        p = A.shape[1]
        if self.scale:
            sc = RobustScaler().fit(A)
            center, scl = np.nan_to_num(sc.center_, nan=0.0), sc.scale_.copy()
            scl[~np.isfinite(scl) | (scl == 0)] = 1.0
        else:
            center, scl = np.zeros(p), np.ones(p)
        self.center_, self.scale_ = center, scl
        S = self._scale(A)
        self.all_nan_cols_ = np.where(np.isnan(S).all(axis=0))[0]       # no donor exists -> filled with 0
        if self.engine == "sklearn":
            self.sk_ = KNNImputer(n_neighbors=self.n_neighbors, keep_empty_features=True).fit(S)
        elif self.engine == "fast":
            self.fit_X_ = S
            obs = ~np.isnan(S)
            self.col_mean_ = np.where(obs.any(0), np.nansum(S, 0) / np.maximum(obs.sum(0), 1), 0.0)
            groups = {}                                                     # columns with identical donors
            for c in range(p):
                if obs[:, c].any():
                    groups.setdefault(hashlib.md5(np.packbits(obs[:, c]).tobytes()).hexdigest(), []).append(c)
            self.groups_ = [(np.where(obs[:, cols[0]])[0], np.array(cols)) for cols in groups.values()]
        else:
            raise ValueError(f"engine must be 'fast' or 'sklearn', got {self.engine!r}")
        return self

    def _impute_fast(self, S):
        out = S.copy()
        missing = np.isnan(S)
        need = np.where(missing.any(axis=1))[0]
        obsA = (~np.isnan(self.fit_X_)).astype(np.float64)
        A0 = np.where(obsA > 0, self.fit_X_, 0.0)
        A0sq = A0 ** 2
        for start in range(0, len(need), self.chunk_rows):
            rows = need[start:start + self.chunk_rows]
            D = _nan_euclidean(S[rows], A0, A0sq, obsA)                      # (rows, n_train)
            miss_chunk = missing[rows]
            for donors, cols in self.groups_:
                sel = np.where(miss_chunk[:, cols].any(axis=1))[0]
                if len(sel) == 0:
                    continue
                Dg = D[np.ix_(sel, donors)]
                Dg = np.where(np.isnan(Dg), np.inf, Dg)                     # no shared coordinates -> unusable
                k = min(self.n_neighbors, len(donors))
                idx = np.argpartition(Dg, k - 1, axis=1)[:, :k] if k < Dg.shape[1] else \
                    np.tile(np.arange(Dg.shape[1]), (len(sel), 1))
                ok = np.take_along_axis(Dg, idx, axis=1) < np.inf           # (sel, k)
                vals = self.fit_X_[donors[idx]][:, :, cols]                 # (sel, k, ncols)
                w = ok[:, :, None].astype(float)
                den = w.sum(axis=1)
                est = np.where(den > 0, (vals * w).sum(axis=1) / np.maximum(den, 1.0), self.col_mean_[cols])
                tgt = rows[sel]
                block = out[np.ix_(tgt, cols)]
                m = miss_chunk[np.ix_(sel, cols)]
                block[m] = est[m]
                out[np.ix_(tgt, cols)] = block
        return out

    def transform(self, X):
        if list(X.columns) != self.columns_:
            raise ValueError("PerlaKNNImputer.transform: columns differ from the ones seen in fit "
                             "(different names or a different order).")
        A = X.to_numpy(dtype=float)
        S = self._scale(A)
        filled = self.sk_.transform(S) if self.engine == "sklearn" else self._impute_fast(S)
        if self.engine == "sklearn" and filled.shape[1] != S.shape[1]:
            raise RuntimeError("sklearn KNNImputer dropped columns.")
        if len(self.all_nan_cols_):
            filled[:, self.all_nan_cols_] = (0.0 - self.center_[self.all_nan_cols_]) / self.scale_[self.all_nan_cols_]
        was_nan = np.isnan(A)
        out = A.copy()
        out[was_nan] = self._unscale(filled)[was_nan]                       # observed cells stay exactly as they were
        return pd.DataFrame(out, index=X.index, columns=self.columns_)

    def get_feature_names_out(self, input_features=None):
        return np.array(self.columns_)


# =========================================================================================
# DATA, GROUPS
# =========================================================================================
def load_dataset(path):
    """-> df (RangeIndex), targets (PCE/Voc/Jsc/FF), groups (paper id).  The parsed lists are NOT touched."""
    df = pd.read_pickle(path).reset_index(drop=True)
    targets = get_targets(df)
    if targets[TARGET].isna().any():
        raise ValueError(f"{int(targets[TARGET].isna().sum())} rows have an unknown {TARGET}; drop them first.")
    groups = get_groups(df)
    return df, targets, groups


# =========================================================================================
# THREE-WAY GROUP-AWARE SPLIT  (70 / 15 / 15)
# =========================================================================================
@dataclass
class ThreeWaySplit:
    """The three EXTERNAL datasets.  Row labels (index) are those of the full dataframe, so every row can be
    traced back; positions (`*_idx`) refer to the full dataframe."""
    train_df: pd.DataFrame           # ~70%  model development (GroupKFold lives inside this set)
    val_df: pd.DataFrame             # ~15%  MAPIE CQR calibration only
    test_df: pd.DataFrame            # ~15%  final evaluation only
    T_train: pd.DataFrame            # PCE, Voc, Jsc, FF
    T_val: pd.DataFrame
    T_test: pd.DataFrame
    g_train: pd.Series               # paper id (DOI) per row
    g_val: pd.Series
    g_test: pd.Series
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    n_total: int
    random_state: int
    fractions: tuple


def make_three_way_split(df, targets, groups, fractions=SPLIT_FRACTIONS, random_state=SPLIT_SEED):
    """Group-aware split.  Whole papers (groups) are shuffled with a fixed seed and assigned one by one to the
    set that is currently furthest below its target row count, so row proportions end up close to
    `fractions` while no paper is ever split."""
    if len(fractions) != 3 or abs(sum(fractions) - 1.0) > 1e-9 or min(fractions) <= 0:
        raise ValueError("fractions must be three positive numbers summing to 1.")
    gs = groups.astype(str)
    sizes = gs.value_counts()
    uniq = np.array(sorted(sizes.index))                    # sorted first -> reproducible regardless of row order
    np.random.RandomState(random_state).shuffle(uniq)
    target_rows = np.asarray(fractions, dtype=float) * len(df)
    cur = np.zeros(3)
    assign = {}
    for g in uniq:
        k = int(np.argmax(target_rows - cur))
        assign[g] = k
        cur[k] += sizes[g]
    which = gs.map(assign).to_numpy()
    idx = [np.where(which == k)[0] for k in range(3)]
    if any(len(i) == 0 for i in idx):
        raise ValueError("A split set is empty (too few papers for a 3-way split).")

    def part(i):
        return df.iloc[i], targets.iloc[i], groups.iloc[i]

    (trd, trt, trg), (vad, vat, vag), (ted, tet, teg) = part(idx[0]), part(idx[1]), part(idx[2])
    split = ThreeWaySplit(trd, vad, ted, trt, vat, tet, trg, vag, teg, idx[0], idx[1], idx[2],
                          len(df), random_state, tuple(fractions))
    check_three_way_split(split)
    return split


def check_three_way_split(split):
    """Raises if papers or rows leak between / disappear from the three sets."""
    tr, va, te = (set(split.g_train.astype(str)), set(split.g_val.astype(str)), set(split.g_test.astype(str)))
    if not tr.isdisjoint(va):
        raise ValueError(f"DOI overlap Train/Validation: {len(tr & va)} papers")
    if not tr.isdisjoint(te):
        raise ValueError(f"DOI overlap Train/Test: {len(tr & te)} papers")
    if not va.isdisjoint(te):
        raise ValueError(f"DOI overlap Validation/Test: {len(va & te)} papers")
    itr, iva, ite = (set(split.train_df.index), set(split.val_df.index), set(split.test_df.index))
    if not (itr.isdisjoint(iva) and itr.isdisjoint(ite) and iva.isdisjoint(ite)):
        raise ValueError("Duplicate row indices across Train / Validation / Test.")
    for name, d in (("train", split.train_df), ("val", split.val_df), ("test", split.test_df)):
        if not d.index.is_unique:
            raise ValueError(f"{name} contains duplicated row indices.")
    n = len(itr) + len(iva) + len(ite)
    if n != split.n_total:
        raise ValueError(f"Rows lost or duplicated: {n} in the three sets vs {split.n_total} in the dataset.")
    pos = np.concatenate([split.train_idx, split.val_idx, split.test_idx])
    if len(np.unique(pos)) != split.n_total:
        raise ValueError("Split positions are not a partition of the dataset.")
    for T, d in ((split.T_train, split.train_df), (split.T_val, split.val_df), (split.T_test, split.test_df)):
        if not T.index.equals(d.index):
            raise ValueError("Targets are not aligned with their dataframe.")


def split_report(split, verbose=True):
    n = split.n_total
    rows = {"Train": len(split.train_df), "Validation": len(split.val_df), "Test": len(split.test_df)}
    papers = {"Train": set(split.g_train.astype(str)), "Validation": set(split.g_val.astype(str)),
              "Test": set(split.g_test.astype(str))}
    rep = dict(total=n, rows=rows, pct={k: 100 * v / n for k, v in rows.items()},
               groups={k: len(v) for k, v in papers.items()},
               overlap={"Train/Validation": len(papers["Train"] & papers["Validation"]),
                        "Train/Test": len(papers["Train"] & papers["Test"]),
                        "Validation/Test": len(papers["Validation"] & papers["Test"])})
    if verbose:
        print(f"Total rows            : {n:,}")
        for k in rows:
            print(f"{k + ' rows':<22}: {rows[k]:,}  ({rep['pct'][k]:.2f}%)   unique DOI groups: {rep['groups'][k]:,}")
        for k, v in rep["overlap"].items():
            print(f"{k} DOI overlap: {v}")
        print("Note: the split is by whole papers, so the row percentages deviate slightly from "
              f"{'/'.join(str(round(100 * f)) for f in split.fractions)} (papers contribute different numbers of rows).")
    return rep


def split_assignment_table(split):
    parts = [pd.DataFrame({"orig_row": d.index, "group": g.values, "set": name})
             for name, d, g in (("train", split.train_df, split.g_train), ("validation", split.val_df, split.g_val),
                                ("test", split.test_df, split.g_test))]
    return pd.concat(parts, ignore_index=True).sort_values("orig_row").reset_index(drop=True)


def load_and_split(path, fractions=SPLIT_FRACTIONS, random_state=SPLIT_SEED):
    """load_dataset + group-aware 70/15/15 split."""
    df, targets, groups = load_dataset(path)
    return make_three_way_split(df, targets, groups, fractions, random_state)


# =========================================================================================
# GROUPKFOLD (INSIDE THE 70% TRAIN SET)
# =========================================================================================
def make_group_folds(groups, n_splits=N_SPLITS):
    """List of (train_idx, val_idx) as POSITIONS.  No paper is ever in both sides."""
    folds = list(GroupKFold(n_splits=n_splits).split(np.zeros(len(groups)), groups=groups))
    for tr, va in folds:
        assert not (set(groups.iloc[tr]) & set(groups.iloc[va])), "a paper appears in train AND validation"
    return folds


def fold_assignment_table(df, groups, folds):
    fold_of = np.full(len(df), -1)
    for k, (_, va) in enumerate(folds):
        fold_of[va] = k
    t = pd.DataFrame({"row": np.arange(len(df)), "orig_row": df.index.to_numpy(), "group": groups.values,
                      "fold": fold_of})
    if "perla_row_id" in df.columns:
        t.insert(1, "perla_row_id", df["perla_row_id"].values)
    return t


# =========================================================================================
# PREPROCESSING PIPELINE:  band-gap interpolation -> encoding -> KNN Imputer
# =========================================================================================
def _forbidden_feature_names(targets):
    banned = set(targets.columns) | set(TARGET_COLS.values())
    return banned


def check_no_target_columns(X, targets):
    """The target (and Voc/Jsc/FF) must not be a column of X.  `quench__te` is allowed: it is a target-ENCODED
    feature computed from the training papers only."""
    banned = _forbidden_feature_names(targets)
    bad = [c for c in X.columns if c in banned or str(c).startswith(("JV_default", "PCE", "Voc__", "Jsc__", "FF__"))]
    if bad:
        raise AssertionError(f"Target columns found in the feature matrix: {bad}")


def _fit_preprocessors(fit_df, y_fit, targets, knn_kwargs, encoder_kwargs, interpolator_factory):
    """Fits interpolation -> encoder -> KNN Imputer on `fit_df` ONLY and returns the transformed fit rows."""
    interp = _make_interpolator(interpolator_factory)
    interp.fit(fit_df)
    interp_df = _interp_transform(interp, fit_df)

    encoder = PerlaEncoder(**encoder_kwargs)
    X_enc = encoder.fit_transform(interp_df, y_fit)          # target encoding is out-of-fold, grouped by paper
    check_no_target_columns(X_enc, targets)

    imputer = PerlaKNNImputer(**knn_kwargs)                  # KNN Imputer (not a KNN regressor)
    X = imputer.fit(X_enc).transform(X_enc)
    assert not X.isna().any().any(), "NaN left after the KNN Imputer"
    assert np.isfinite(X.to_numpy()).all()
    return interp, encoder, imputer, X


def _transform_with(interp, encoder, imputer, df, expected_columns):
    """Applies already-fitted preprocessors (no fitting, no target)."""
    X_enc = encoder.transform(_interp_transform(interp, df))
    assert list(X_enc.columns) == list(expected_columns), "encoded columns differ"
    X = imputer.transform(X_enc)
    assert not X.isna().any().any(), "NaN left after the KNN Imputer"
    assert np.isfinite(X.to_numpy()).all()
    return X


# =========================================================================================
# ONE INTERNAL CV FOLD
# =========================================================================================
@dataclass
class FoldData:
    """One GroupKFold fold INSIDE the 70% train set.  `*_fold_*` names = internal CV only; they are different
    objects from the external validation set (X_mapie_val) and the test set (X_test)."""
    fold: int
    train_idx: np.ndarray            # positions inside the 70% train dataframe
    val_idx: np.ndarray
    X_fold_train: pd.DataFrame       # interpolated + encoded + imputed, fitted on the fold-train rows only
    X_fold_val: pd.DataFrame         # transformed with the fold-train interpolator / encoder / KNN Imputer
    y_fold_train: pd.Series          # PCE
    y_fold_val: pd.Series
    T_fold_train: pd.DataFrame       # PCE, Voc, Jsc, FF  (for the two-stage model; never part of X)
    T_fold_val: pd.DataFrame
    interpolator: object = None
    encoder: object = None
    imputer: object = None

    # backward-compatible aliases of the previous field names
    X_train = property(lambda s: s.X_fold_train)
    X_val = property(lambda s: s.X_fold_val)
    y_train = property(lambda s: s.y_fold_train)
    y_val = property(lambda s: s.y_fold_val)
    T_train = property(lambda s: s.T_fold_train)
    T_val = property(lambda s: s.T_fold_val)


def prepare_fold(df, targets, groups, train_idx, val_idx, fold=0, knn_kwargs=None, encoder_kwargs=None,
                 verbose=True, interpolator_factory=None):
    """`df` must be the 70% TRAIN set (never the full dataset, never val/test)."""
    knn_kwargs = KNN_KWARGS if knn_kwargs is None else knn_kwargs
    encoder_kwargs = ENCODER_KWARGS if encoder_kwargs is None else encoder_kwargs
    interpolator_factory = BAND_GAP_INTERPOLATOR_FACTORY if interpolator_factory is None else interpolator_factory
    t0 = time.time()

    train_df, val_df = df.iloc[train_idx], df.iloc[val_idx]
    T_train, T_val = targets.iloc[train_idx], targets.iloc[val_idx]
    y_train, y_val = T_train[TARGET], T_val[TARGET]

    assert not (set(groups.iloc[train_idx]) & set(groups.iloc[val_idx])), "paper in both train and validation"
    assert y_train.notna().all(), "unknown target in the training fold"

    # interpolation, encoder and KNN Imputer learn from the fold-TRAIN rows only; y_val is never passed anywhere.
    interp, encoder, imputer, X_train = _fit_preprocessors(train_df, y_train, targets, knn_kwargs, encoder_kwargs,
                                                           interpolator_factory)
    t_fit = time.time() - t0
    X_val = _transform_with(interp, encoder, imputer, val_df, X_train.columns)

    if verbose:
        print(f"  fold {fold}: train {X_train.shape}  val {X_val.shape}  "
              f"papers train/val {groups.iloc[train_idx].nunique()}/{groups.iloc[val_idx].nunique()}  "
              f"[fit+transform train {t_fit:.0f}s, transform val {time.time() - t0 - t_fit:.0f}s]")
    return FoldData(fold, np.asarray(train_idx), np.asarray(val_idx), X_train, X_val, y_train, y_val,
                    T_train, T_val, interp, encoder, imputer)


# =========================================================================================
# ALL INTERNAL FOLDS (with an on-disk cache so the slow part runs once, not once per model)
# =========================================================================================
def _fingerprint(df, targets, groups, n_splits, knn_kwargs, encoder_kwargs, interpolator_factory=None, extra=""):
    """`extra` carries what the fold cache otherwise cannot see (split seed + fractions).  This module's own
    source is hashed too, so editing the split / fold logic invalidates old caches."""
    h = hashlib.sha256()
    h.update(f"{len(df)}|{n_splits}|{sorted(knn_kwargs.items())}|{sorted(encoder_kwargs.items())}|{extra}".encode())
    h.update(pd.util.hash_pandas_object(groups.astype(str), index=False).to_numpy().tobytes())
    h.update(pd.util.hash_pandas_object(pd.Series(df.index), index=False).to_numpy().tobytes())   # which rows
    h.update(np.nan_to_num(targets.to_numpy(dtype=float), nan=-1.0).tobytes())
    mods = [enc_module, inspect.getmodule(PerlaKNNImputer), inspect.getmodule(_fingerprint)]
    if interpolator_factory is not None:
        h.update(getattr(interpolator_factory, "__qualname__", repr(interpolator_factory)).encode())
        m = inspect.getmodule(interpolator_factory)
        if m is not None and getattr(m, "__file__", None):
            mods.append(m)
    for mod in mods:                                                              # editing the code invalidates it
        h.update(Path(mod.__file__).read_bytes())
    return h.hexdigest()[:16]


def iter_folds(df, targets, groups, n_splits=N_SPLITS, cache_dir=None, knn_kwargs=None, encoder_kwargs=None,
               verbose=True, interpolator_factory=None, fingerprint_extra=""):
    """Yields FoldData for every GroupKFold split of the dataframe it is given - pass ONLY the 70% train set
    (use `iter_train_folds(split)` which enforces that).  With cache_dir the interpolated + encoded + imputed
    matrices are saved after the first run and reloaded afterwards."""
    knn_kwargs = KNN_KWARGS if knn_kwargs is None else knn_kwargs
    encoder_kwargs = ENCODER_KWARGS if encoder_kwargs is None else encoder_kwargs
    interpolator_factory = BAND_GAP_INTERPOLATOR_FACTORY if interpolator_factory is None else interpolator_factory
    folds = make_group_folds(groups, n_splits)
    fp = _fingerprint(df, targets, groups, n_splits, knn_kwargs, encoder_kwargs, interpolator_factory,
                      fingerprint_extra)
    if cache_dir:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        fold_assignment_table(df, groups, folds).to_csv(cache_dir / "fold_assignment.csv", index=False)
        (cache_dir / "cache_meta.json").write_text(json.dumps(     # read by load_cached_folds
            dict(fingerprint=fp, n_splits=n_splits, n_rows=len(df), extra=fingerprint_extra), indent=2))
    for k, (tr, va) in enumerate(folds):
        path = cache_dir / f"fold{k}_{n_splits}.joblib" if cache_dir else None
        if path is not None and path.exists():
            saved = joblib.load(path)
            if saved["fingerprint"] == fp:
                if verbose:
                    print(f"  fold {k}: loaded from cache")
                yield saved["fold"]
                continue
        fold = prepare_fold(df, targets, groups, tr, va, k, knn_kwargs, encoder_kwargs, verbose,
                            interpolator_factory)
        if path is not None:
            joblib.dump({"fingerprint": fp, "fold": replace(fold, interpolator=None, encoder=None, imputer=None)},
                        path)
        yield fold


def iter_train_folds(split, n_splits=N_SPLITS, cache_dir=None, knn_kwargs=None, encoder_kwargs=None,
                     verbose=True, interpolator_factory=None):
    """GroupKFold over the 70% TRAIN set of a ThreeWaySplit.  The external validation and test sets are not
    even passed in; every fold is additionally checked to lie inside the train set."""
    train_rows = set(split.train_df.index)
    forbidden = set(split.val_df.index) | set(split.test_df.index)
    for fold in iter_folds(split.train_df, split.T_train, split.g_train, n_splits, cache_dir, knn_kwargs,
                           encoder_kwargs, verbose, interpolator_factory,
                           fingerprint_extra=f"split_seed={split.random_state}|fractions={split.fractions}"):
        used = set(fold.X_fold_train.index) | set(fold.X_fold_val.index)
        if not used <= train_rows or used & forbidden:
            raise AssertionError("A GroupKFold fold contains rows from outside the 70% train set "
                                 "(validation/test data leaked into CV).")
        yield fold


def load_cached_folds(cache_dir, n_splits=N_SPLITS):
    """Yields the cached FoldData of the 70% TRAIN set straight from the cache folder (no dataset, no encoder run).
    The data / code cannot be re-checked here (the dataset is not loaded), but the cache is checked for internal
    consistency: cache_meta.json must exist, must say the same n_splits, and every fold file must carry the same
    fingerprint as cache_meta.json (so folds from two different builds cannot be mixed)."""
    cache_dir = Path(cache_dir)
    meta_path = cache_dir / "cache_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"{meta_path} missing: the cache was built by an older perla_cv.py. Rebuild it:  "
                                f"python perla_cv.py --data <pkl> --folds {n_splits}")
    meta = json.loads(meta_path.read_text())
    if meta["n_splits"] != n_splits:
        raise ValueError(f"cache was built with {meta['n_splits']} folds but {n_splits} were requested")
    for k in range(n_splits):
        path = cache_dir / f"fold{k}_{n_splits}.joblib"
        if not path.exists():
            raise FileNotFoundError(f"{path} not found. Build it first:  python perla_cv.py --data <pkl> --folds {n_splits}")
        saved = joblib.load(path)
        if saved["fingerprint"] != meta["fingerprint"]:
            raise ValueError(f"{path.name} belongs to a different cache build than cache_meta.json (stale fold file)")
        yield saved["fold"]


# =========================================================================================
# FINAL PREPROCESSING (after model selection)
# =========================================================================================
@dataclass
class FinalData:
    """Final matrices.  Everything is FITTED on the 70% train set only."""
    X_final_train: pd.DataFrame
    y_final_train: pd.Series
    T_final_train: pd.DataFrame
    X_mapie_val: pd.DataFrame        # 15% validation -> MAPIE CQR calibration only
    y_mapie_val: pd.Series
    T_mapie_val: pd.DataFrame
    X_test: pd.DataFrame             # 15% test -> final evaluation only
    y_test: pd.Series
    T_test: pd.DataFrame
    interpolator: object = None
    encoder: object = None
    imputer: object = None


def fit_final_preprocessors(df, targets, knn_kwargs=None, encoder_kwargs=None, interpolator_factory=None):
    """Fits interpolation + encoder + KNN Imputer on the rows given - these must be the 70% TRAIN set only
    (validation / test rows go through transform).  Returns (interpolator, encoder, imputer, X_train)."""
    knn_kwargs = KNN_KWARGS if knn_kwargs is None else knn_kwargs
    encoder_kwargs = ENCODER_KWARGS if encoder_kwargs is None else encoder_kwargs
    interpolator_factory = BAND_GAP_INTERPOLATOR_FACTORY if interpolator_factory is None else interpolator_factory
    return _fit_preprocessors(df, targets[TARGET], targets, knn_kwargs, encoder_kwargs, interpolator_factory)


def prepare_final_datasets(split, knn_kwargs=None, encoder_kwargs=None, interpolator_factory=None):
    """70% train -> fit interpolation / encoder / KNN Imputer; 15% validation and 15% test -> transform only.
    y_val / y_test are only attached to the output, never passed to any fit."""
    check_three_way_split(split)
    interp, enc, imp, X_tr = fit_final_preprocessors(split.train_df, split.T_train, knn_kwargs, encoder_kwargs,
                                                     interpolator_factory)
    assert set(X_tr.index) == set(split.train_df.index), "final preprocessors were not fitted on the train set only"
    X_val = _transform_with(interp, enc, imp, split.val_df, X_tr.columns)
    X_test = _transform_with(interp, enc, imp, split.test_df, X_tr.columns)
    return FinalData(X_tr, split.T_train[TARGET], split.T_train, X_val, split.T_val[TARGET], split.T_val,
                     X_test, split.T_test[TARGET], split.T_test, interp, enc, imp)


# =========================================================================================
# TINY RUNNER (plumbing only - plug your models in with a factory; internal CV folds only)
# =========================================================================================
def evaluate(model_factory, folds, y_all_len):
    """model_factory() -> an object with fit(X, y) / predict(X).  Returns (out-of-fold predictions, metrics).
    `y_all_len` = number of rows of the 70% train set (positions in FoldData.val_idx refer to it)."""
    oof, rows = np.full(y_all_len, np.nan), []
    for f in folds:
        model = model_factory()
        model.fit(f.X_fold_train, f.y_fold_train)
        pred = model.predict(f.X_fold_val)
        oof[f.val_idx] = pred
        rows.append(dict(fold=f.fold, R2=r2_score(f.y_fold_val, pred), MAE=mean_absolute_error(f.y_fold_val, pred),
                         RMSE=mean_squared_error(f.y_fold_val, pred) ** 0.5))
    return pd.Series(oof, name=f"oof_{TARGET}"), pd.DataFrame(rows).set_index("fold")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="perla_final_filtered.pkl")
    ap.add_argument("--folds", type=int, default=N_SPLITS)
    ap.add_argument("--cache-dir", default="cv_cache")            # internal GroupKFold folds of the 70% train set only
    ap.add_argument("--engine", choices=["fast", "sklearn"], default=KNN_KWARGS["engine"])
    ap.add_argument("--no-scale", action="store_true")
    ap.add_argument("--seed", type=int, default=SPLIT_SEED)
    ap.add_argument("--split-file", default="split_assignment.csv")   # kept OUTSIDE cv_cache on purpose
    ap.add_argument("--final", default=None, help="also fit the final preprocessors on the 70%% train set and "
                                                  "save FinalData (train / MAPIE-validation / test) to this joblib")
    a = ap.parse_args()

    split = load_and_split(a.data, random_state=a.seed)
    split_report(split)
    split_assignment_table(split).to_csv(a.split_file, index=False)
    print(f"{split.g_train.nunique():,} papers in TRAIN, {a.folds} GroupKFold folds")
    knn = dict(KNN_KWARGS, engine=a.engine, scale=not a.no_scale)
    t = time.time()
    for fold in iter_train_folds(split, a.folds, a.cache_dir, knn_kwargs=knn):
        pass
    print(f"CV folds done in {time.time() - t:.0f}s  ->  {a.cache_dir}/")
    if a.final:
        final = prepare_final_datasets(split, knn_kwargs=knn)
        joblib.dump(final, a.final)
        print(f"final: train {final.X_final_train.shape}  mapie-val {final.X_mapie_val.shape}  "
              f"test {final.X_test.shape}  ->  {a.final}")
