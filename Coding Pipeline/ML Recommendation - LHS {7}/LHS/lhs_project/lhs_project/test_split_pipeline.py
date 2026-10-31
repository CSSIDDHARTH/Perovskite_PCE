"""
Checks, on REAL data (a random subsample of whole papers, for speed), that
  * the data is split into TRAIN / VALIDATION / TEST by whole papers (no paper in two sets, every row in exactly one),
  * every GroupKFold fold lives inside TRAIN,
  * the interpolator / PerlaEncoder / KNN Imputer are FITTED only on the rows they are allowed to see
    (fold-train rows inside CV, the 70% TRAIN set for the final fit) and merely TRANSFORM everything else,
  * validation and test receive exactly the same feature columns as train.

Spies record the index of every frame passed to fit / fit_transform / transform.

    PERLA_DATA=perla_final_filtered_1.pkl pytest -q test_split_pipeline.py
"""
import os

import numpy as np
import pandas as pd
import pytest

import perla_cv as c
import perla_encoding_by_claude as e

DATA = next((p for p in (os.environ.get("PERLA_DATA"), "perla_final_filtered_1.pkl", "perla_final_filtered.pkl", "data.pkl")
             if p and os.path.exists(p)), None)
pytestmark = pytest.mark.skipif(DATA is None, reason="real dataset not found (set PERLA_DATA)")

LOG = []                                                   # (component, method, frozenset(index))


def _spy(cls, name):
    class Spy(cls):
        def fit(self, X, *a, **k):
            LOG.append((name, "fit", frozenset(X.index)))
            return super().fit(X, *a, **k)

        def fit_transform(self, X, *a, **k):
            LOG.append((name, "fit", frozenset(X.index)))
            return super().fit_transform(X, *a, **k)

        def transform(self, X, *a, **k):
            LOG.append((name, "transform", frozenset(X.index)))
            return super().transform(X, *a, **k)
    Spy.__name__ = "Spy" + cls.__name__
    return Spy


@pytest.fixture(scope="module")
def split(tmp_path_factory):
    raw = pd.read_pickle(DATA)
    gone = [col for col in e.BOOLEAN_COLS if col not in raw.columns]
    if gone:                                               # same runtime tolerance as run_new_dataset.py
        e.BOOLEAN_COLS = [col for col in e.BOOLEAN_COLS if col in raw.columns]
    g = e.get_groups(raw).astype(str)
    rng = np.random.RandomState(0)
    keep = set(rng.choice(sorted(g.unique()), size=1800, replace=False))
    sub = raw[g.isin(keep)]
    path = tmp_path_factory.mktemp("sub") / "sub.pkl"
    sub.to_pickle(path)
    return c.load_and_split(str(path), random_state=42), len(sub)


def test_split_is_complete_disjoint_and_grouped(split):
    s, n = split
    assert len(s.train_df) + len(s.val_df) + len(s.test_df) == n
    idx = [set(s.train_df.index), set(s.val_df.index), set(s.test_df.index)]
    assert not (idx[0] & idx[1]) and not (idx[0] & idx[2]) and not (idx[1] & idx[2])       # each row in one set
    docs = [set(s.g_train.astype(str)), set(s.g_val.astype(str)), set(s.g_test.astype(str))]
    assert not (docs[0] & docs[1]) and not (docs[0] & docs[2]) and not (docs[1] & docs[2])  # each paper in one set
    assert abs(len(s.train_df) / n - 0.70) < 0.03 and abs(len(s.val_df) / n - 0.15) < 0.03


def test_cv_folds_inside_train_and_preprocessing_fit_on_fold_train_only(split, monkeypatch):
    s, _ = split
    LOG.clear()
    monkeypatch.setattr(c, "PerlaEncoder", _spy(c.PerlaEncoder, "encoder"))
    monkeypatch.setattr(c, "PerlaKNNImputer", _spy(c.PerlaKNNImputer, "imputer"))
    Interp = _spy(c.NoBandGapInterpolation, "interpolator")
    folds = c.make_group_folds(s.g_train, n_splits=5)
    tr, va = folds[0]
    fold = c.prepare_fold(s.train_df, s.T_train, s.g_train, tr, va, fold=0, verbose=False, interpolator_factory=Interp)
    fold_train, fold_val = set(s.train_df.index[tr]), set(s.train_df.index[va])
    external = set(s.val_df.index) | set(s.test_df.index)
    assert fold_train | fold_val <= set(s.train_df.index) and not (fold_train | fold_val) & external
    fits = [(comp, idx) for comp, m, idx in LOG if m == "fit"]
    assert {comp for comp, _ in fits} == {"interpolator", "encoder", "imputer"}             # all three were fitted ...
    for comp, idx in fits:
        assert idx == fold_train, f"{comp} was fitted on rows other than the fold-train rows"  # ... on fold-train only
    transforms = [(comp, idx) for comp, m, idx in LOG if m == "transform"]
    assert {comp for comp, idx in transforms if idx == fold_val} == {"interpolator", "encoder", "imputer"}
    assert list(fold.X_fold_val.columns) == list(fold.X_fold_train.columns)
    assert not fold.X_fold_train.isna().any().any() and not fold.X_fold_val.isna().any().any()


def test_final_fit_on_train_only_and_val_test_only_transformed(split, monkeypatch):
    s, _ = split
    LOG.clear()
    monkeypatch.setattr(c, "PerlaEncoder", _spy(c.PerlaEncoder, "encoder"))
    monkeypatch.setattr(c, "PerlaKNNImputer", _spy(c.PerlaKNNImputer, "imputer"))
    Interp = _spy(c.NoBandGapInterpolation, "interpolator")
    interp, enc, imp, X_train = c.fit_final_preprocessors(s.train_df, s.T_train, interpolator_factory=Interp)
    X_val = c._transform_with(interp, enc, imp, s.val_df, X_train.columns)
    X_test = c._transform_with(interp, enc, imp, s.test_df, X_train.columns)
    train, val, test = set(s.train_df.index), set(s.val_df.index), set(s.test_df.index)
    fits = [(comp, idx) for comp, m, idx in LOG if m == "fit"]
    assert {comp for comp, _ in fits} == {"interpolator", "encoder", "imputer"}
    for comp, idx in fits:
        assert idx == train, f"{comp} was fitted on rows other than the 70% TRAIN set"
        assert not idx & (val | test)
    for who, rows in (("validation", val), ("test", test)):
        assert {comp for comp, m, idx in LOG if m == "transform" and idx == rows} == {"interpolator", "encoder", "imputer"}, who
    assert list(X_val.columns) == list(X_train.columns) == list(X_test.columns)             # identical features everywhere
    assert X_val.index.equals(s.val_df.index) and X_test.index.equals(s.test_df.index)
    assert not X_val.isna().any().any() and not X_test.isna().any().any()
