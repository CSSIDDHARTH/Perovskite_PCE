"""
lhs_predict.py - reconstruct the finalists, encode candidates with the EXISTING preprocessing, predict in batches
==============================================================================================================
WHAT EXISTS ON DISK (inspected): no fitted model, encoder, imputer, SHAP explainer or MAPIE object is serialised anywhere in the
project (run_models.py / perla_explain_uq.py refit everything on every run and only write tables and plots).  This module therefore
RECONSTRUCTS the finalists exactly the way perla_explain_uq.py does:

    load_and_split (70/15/15 by paper)  ->  perla_cv.fit_final_preprocessors on TRAIN only (interpolator -> PerlaEncoder -> PerlaKNNImputer)
    -> perla_mapie.MemberBank (rm.make_models() hyper-parameters, rm.Stage1Predictor, rm.stack)  ->  bank.build(model_id)

Nothing is re-tuned and the model code is not modified.  The only difference from a "load" is that the fit is repeated; the fidelity
checks in lhs_uncertainty.py compare the reconstruction with the artefacts of the reported run when those are available.

Candidates go through `encode_frame`: the SAME fitted interpolator / encoder / imputer objects that produced the training matrix.
No encoder is ever fitted on candidates.  Two-stage inference is bank.features('TwoStage', X) = [X | Stage-1 PREDICTIONS]; the candidate
frame contains no Voc/Jsc/FF column at all, so measured values cannot enter.
"""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

import perla_cv
import perla_encoding_by_claude as E
import perla_mapie as pmq
import run_models as rm
from lhs_config import FORBIDDEN_RAW_COLUMNS, SHORT, TWOSTAGE_ID, LHSConfig
from perla_cv import _interp_transform, check_no_target_columns, fit_final_preprocessors, load_and_split


@dataclass
class Artifacts:
    cfg: LHSConfig
    split: object
    interp: object
    encoder: object
    imputer: object
    X_train_df: pd.DataFrame
    D: dict
    bank: object
    args: argparse.Namespace
    fit_seconds: float
    cache_comparison: dict = field(default_factory=dict)

    @property
    def feature_names(self):
        return list(self.X_train_df.columns)


def make_args(cfg: LHSConfig) -> argparse.Namespace:
    base = {"hgb": "HistGradientBoosting", "xgb": "XGBoost", "rf": "RandomForest"}[cfg.stage1_base]
    return argparse.Namespace(quick=cfg.quick, seed=cfg.model_seed, n_jobs=cfg.n_jobs, stage1_base=cfg.stage1_base, inner_folds=cfg.inner_folds,
                              add_product=cfg.add_product, stage1_label=f"S1={base}", confidence=cfg.confidence, split_seed=cfg.split_seed,
                              cal_fraction=cfg.cal_fraction, shap_method="interventional", shap_background=cfg.shap_background, score_test=cfg.score_test)


def compare_with_cache(X_df: pd.DataFrame, cache_path: str | None) -> dict:
    """Does THIS environment reproduce the training matrix of the reported run?  (Known cause of differences: the out-of-fold target
    encoding uses GroupKFold, whose fold assignment can change between scikit-learn versions; quench__te then differs and so do the
    KNN-imputed cells that use it as a distance coordinate.)"""
    if not cache_path or not Path(cache_path).exists():
        return dict(available=False, note="no cached training matrix given (cfg.reference_prep_cache); reproduction of the reported run not verified")
    import joblib
    D = joblib.load(cache_path)
    Xc, Xn = np.asarray(D["X_train"], np.float32), X_df.to_numpy(np.float32)
    names_equal = list(X_df.columns) == list(D["feature_names"])
    if Xc.shape != Xn.shape or not names_equal:
        return dict(available=True, reproduced=False, shape_equal=Xc.shape == Xn.shape, names_equal=names_equal)
    bad = np.abs(Xn - Xc) > 1e-4 * (1 + np.abs(Xc))
    cols = pd.Series(bad.sum(0), index=X_df.columns)
    cols = cols[cols > 0].sort_values(ascending=False)
    return dict(available=True, reproduced=bool(not bad.any()), cells_differing=int(bad.sum()), cells_total=int(bad.size),
                columns_affected=int(len(cols)), top_columns=cols.head(8).to_dict(), max_abs_diff=float(np.abs(Xn - Xc).max()),
                note="differences => the models fitted here are not bit-identical to those of the reported run")


def load_artifacts(cfg: LHSConfig, log=print) -> Artifacts:
    t0 = time.time()
    split = load_and_split(cfg.data_path, random_state=cfg.split_seed)
    perla_cv.check_three_way_split(split)
    log(f"[artifacts] split: train {len(split.train_df)} / val {len(split.val_df)} / test {len(split.test_df)} rows")
    interp, enc, imp, X_tr = fit_final_preprocessors(split.train_df, split.T_train)         # TRAIN only
    assert set(X_tr.index) == set(split.train_df.index), "preprocessors were not fitted on the train set only"
    log(f"[artifacts] preprocessors fitted on TRAIN in {time.time() - t0:.0f}s: {X_tr.shape[1]} features")
    X_val = perla_cv._transform_with(interp, enc, imp, split.val_df, X_tr.columns)
    D = dict(X_train=X_tr.to_numpy(np.float32), X_val=X_val.to_numpy(np.float32), y_train=split.T_train[perla_cv.TARGET].to_numpy(float),
             y_val=split.T_val[perla_cv.TARGET].to_numpy(float), g_train=split.g_train.to_numpy().astype(str), g_val=split.g_val.to_numpy().astype(str),
             T_train_aux=split.T_train[list(rm.AUX_TARGETS)].reset_index(drop=True), feature_names=list(X_tr.columns))
    assert not (set(D["g_train"]) & set(D["g_val"])), "a paper is in TRAIN and VALIDATION"
    cmp_ = compare_with_cache(X_tr, cfg.reference_prep_cache)
    log(f"[artifacts] cache comparison: {cmp_}")
    args = make_args(cfg)
    bank = pmq.MemberBank(args, D["X_train"], D["y_train"], D["T_train_aux"], D["g_train"], cfg.confidence)
    return Artifacts(cfg, split, interp, enc, imp, X_tr, D, bank, args, time.time() - t0, cmp_)


# =========================================================================================
# ENCODING + SAFETY CHECKS
# =========================================================================================
SINGLE_PREFIXES = ["arch", "etl_stack", "htl_stack", "bc_stack", "substrate", "pvk_proc", "etl_proc", "htl_proc", "bc_proc", "solvent",
                   "atmosphere", "agg_state"]


def assert_no_leakage(frame: pd.DataFrame, X_enc: pd.DataFrame, targets: pd.DataFrame) -> None:
    bad = set(frame.columns) & (set(FORBIDDEN_RAW_COLUMNS) | set(E.TARGET_COLS.values()) | set(targets.columns))
    if bad:
        raise AssertionError(f"target / measured-output columns in the candidate frame: {sorted(bad)}")
    check_no_target_columns(X_enc, targets)


def representability_problems(frame: pd.DataFrame, X_enc: pd.DataFrame) -> list[str]:
    """Candidates must be representable WITHOUT falling into the encoder's `__other` / `__missing` buckets, except where a value is
    missing on purpose (band gap, band-gap basis, unspecified layer thickness, no anti-solvent)."""
    p = []
    for pre in SINGLE_PREFIXES:
        for suf in ("other", "missing"):
            c = f"{pre}__{suf}"
            if c in X_enc and X_enc[c].to_numpy().any():
                p.append(f"{int(X_enc[c].sum())} candidates fall into {c} (token unseen / too rare in TRAIN, or empty)")
    for s in "ABC":
        c = f"{s}_frac__other"
        if X_enc[c].fillna(0).to_numpy().any():
            p.append(f"{int((X_enc[c].fillna(0) > 0).sum())} candidates have {c} > 0 (ion outside the encoder vocabulary)")
    if (X_enc["comp__valid"] != 1).any():
        p.append(f"{int((X_enc['comp__valid'] != 1).sum())} candidates have comp__valid != 1")
    for c in ("anneal__temp_missing", "anneal__time_missing"):
        if X_enc[c].to_numpy().any():
            p.append(f"{int(X_enc[c].sum())} candidates have {c} = 1")
    for col in E.BOOLEAN_COLS:
        if X_enc[f"{col}__unknown"].to_numpy().any():
            p.append(f"boolean {col} encoded as unknown for some candidates (missing is never False; candidates must be explicit)")
    for name in ("Perovskite_thickness__sum", "ETL_thickness__sum", "HTL_thickness__sum", "Backcontact_thickness_list__sum"):
        flag = f"num__{name}__missing"
        if flag in X_enc and not np.array_equal(X_enc[flag].to_numpy() > 0, frame[name].isna().to_numpy()):
            p.append(f"{flag} disagrees with the candidate's own missing pattern")
    for c in ("num__band_gap__mean__missing", "bg_basis__missing"):
        if c in X_enc and not (X_enc[c] == 1).all():
            p.append(f"{c} should be 1 for every candidate (neutralised input)")
    return p


def encode_frame(art: Artifacts, frame: pd.DataFrame, check: bool = True):
    """frame (raw PERLA schema) -> (X float32 as used in training, encoded-before-imputation DataFrame, imputed-cell counts)."""
    X_enc = art.encoder.transform(_interp_transform(art.interp, frame))
    if list(X_enc.columns) != art.feature_names:
        raise AssertionError("encoded columns differ from the training columns (names or order)")
    if check:
        assert_no_leakage(frame, X_enc, art.split.T_train)
        problems = representability_problems(frame, X_enc)
        if problems:
            raise ValueError("candidates cannot be represented safely:\n  - " + "\n  - ".join(problems))
    imputed = X_enc.isna().sum()
    X = art.imputer.transform(X_enc)
    if X.isna().any().any() or not np.isfinite(X.to_numpy()).all():
        raise AssertionError("NaN / inf left after the KNN imputer")
    return X.to_numpy(np.float32), X_enc, imputed[imputed > 0].to_dict()


def encode_in_batches(art: Artifacts, frame: pd.DataFrame, batch: int, log=print):
    Xs, imputed = [], {}
    for s in range(0, len(frame), batch):
        X, _, imp = encode_frame(art, frame.iloc[s:s + batch].reset_index(drop=True), check=True)
        Xs.append(X)
        for k, v in imp.items():
            imputed[k] = imputed.get(k, 0) + v
        log(f"[encode] {min(s + batch, len(frame))}/{len(frame)}")
    return np.vstack(Xs), imputed


# =========================================================================================
# FINALISTS
# =========================================================================================
@dataclass
class Finalist:
    model_id: str
    cand: dict
    approach: str
    cqr: object = None

    @property
    def short(self):
        return SHORT[self.model_id]


def build_finalists(art: Artifacts, log=print) -> list[Finalist]:
    out = []
    for mid in art.cfg.finalists:
        t = time.time()
        cand = art.bank.build(mid)
        log(f"[finalist] {mid}: built / fitted in {time.time() - t:.0f}s (members {cand['members']})")
        out.append(Finalist(mid, cand, cand["ctx"]))
    return out


def stage1_predictions(art: Artifacts, X: np.ndarray) -> pd.DataFrame:
    """Voc_hat, Jsc_hat, FF_hat of the Stage-1 models (PREDICTIONS; never measured values)."""
    F = art.bank.features("TwoStage", X)
    return pd.DataFrame(F[:, -len(rm.AUX_TARGETS):], columns=[f"{k}_hat" for k in rm.AUX_TARGETS])
