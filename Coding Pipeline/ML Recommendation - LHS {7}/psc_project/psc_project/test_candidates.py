"""Fast synthetic checks of the model-comparison logic in run_models.py (no real data needed):
    pytest -q test_candidates.py
"""
import numpy as np
import pandas as pd
import pytest

import run_models as r

pytestmark = pytest.mark.skipif(not r.HAS_XGB, reason="xgboost not installed")


@pytest.fixture(scope="module")
def data():
    rng = np.random.RandomState(0)
    n = 600
    X = rng.normal(size=(n, 6))
    groups = np.repeat(np.arange(120), 5)                      # 120 "papers" x 5 rows
    T = pd.DataFrame({"Voc": X[:, 0] + rng.normal(0, .1, n), "Jsc": X[:, 1] + rng.normal(0, .1, n),
                      "FF": X[:, 2] + rng.normal(0, .1, n)})
    T.loc[::17, "Voc"] = np.nan                                 # missing auxiliary targets
    y = X[:, :3].sum(axis=1) + rng.normal(0, .2, n)
    return X, y, T, groups


def test_twelve_candidates_and_voting_definitions():
    meta = r.candidate_meta("S1=HistGradientBoosting")
    assert len(meta) == 12 and sum(m["approach"] == "Direct" for m in meta.values()) == 6
    assert r.VOTING_SETS["VotingOriginal"] == ("RandomForest", "XGBoost", "HistGradientBoosting")   # untouched original
    assert r.VOTING_SETS["VotingET"] == r.VOTING_SETS["VotingOriginal"] + ("ExtraTrees",)
    assert len(set(r.VOTING_WEIGHTS.values())) == 1                                                  # equal, untuned


def test_voting_is_equal_weight_mean_of_members(data):
    X, y, *_ = data
    models = r.make_models(quick=True, seed=0, n_jobs=1)
    fitted, secs = r.fit_bases(models, X, y, r.BASE_MODELS)
    out = r.predict_candidates(fitted, secs, X)
    assert set(out) == set(r.ESTIMATORS)
    for v, members in r.VOTING_SETS.items():
        np.testing.assert_allclose(out[v][0], np.mean([out[m][0] for m in members], axis=0))


def test_stage1_oof_is_group_disjoint_and_not_in_sample(data):
    X, y, T, groups = data
    models = r.make_models(quick=True, seed=0, n_jobs=1)
    s1 = r.Stage1Predictor(models["RandomForest"], 5).fit(X, T, groups)
    assert s1.oof_.shape == (len(X), 3) and not np.isnan(s1.oof_).any()
    insample = s1.predict(X)
    known = T["Jsc"].notna().to_numpy()
    err_oof = np.abs(s1.oof_[known, 1] - T["Jsc"][known]).mean()
    err_in = np.abs(insample[known, 1] - T["Jsc"][known]).mean()
    assert err_oof > err_in                                    # out-of-fold really is out-of-fold
    assert s1.n_inner_splits_used_ == 5


def test_metrics_never_raise_on_degenerate_input():
    s = r.score(np.array([1.0]), np.array([1.0]))
    assert np.isnan(s["R2"]) and np.isnan(s["Pearson"])
    s = r.score(np.ones(5), np.arange(5.0))
    assert np.isnan(s["R2"]) and np.isnan(s["Pearson"]) and s["RMSE"] > 0


def test_selection_uses_mean_rmse_not_best_fold():
    meta = r.candidate_meta("S1=HistGradientBoosting")
    ids = list(meta)
    rows = []
    for i, m in enumerate(ids):
        for f in range(5):
            rmse = 3 + 0.05 * i + 0.002 * f
            if i == 5 and f == 0:
                rmse = 0.1                      # direct model #5 has by far the best SINGLE fold ...
            elif i == 5:
                rmse = 4.0                      # ... but a worse mean than direct model #0
            rows.append(dict(model_id=m, approach=meta[m]["approach"], fold=f, RMSE=rmse, MAE=2.0, R2=.4, Pearson=.6,
                             seconds=1.0))
    pf = pd.DataFrame(rows)
    summ = pf.groupby("model_id").agg(RMSE_mean=("RMSE", "mean"), RMSE_std=("RMSE", "std"), MAE_mean=("MAE", "mean"),
                                      R2_mean=("R2", "mean"), Pearson_mean=("Pearson", "mean"),
                                      seconds_mean=("seconds", "mean")).loc[ids]
    summ["approach"] = [meta[i]["approach"] for i in summ.index]
    sel = r.select_models(pf, summ, preliminary=False)
    assert sel["best_direct"]["model_id"] == ids[0]                 # lowest MEAN, not the lucky fold
    assert sel["best_two_stage"]["model_id"] == ids[6]
    assert sel["best_overall"]["model_id"] == ids[0]
    assert "selection bias" in sel["caveat"]
