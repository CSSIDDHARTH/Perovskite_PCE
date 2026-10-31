"""Fast synthetic checks for perla_explain_uq.py:  pytest -q test_explain_uq.py"""
import numpy as np
import pytest

pytest.importorskip("mapie")
pytest.importorskip("xgboost")
import perla_explain_uq as u
import run_models as rm


def test_interval_metrics_known_values():
    y = np.array([1.0, 2.0, 3.0, 10.0])
    m = u.interval_metrics(y, y - 1, y + 1, y, 0.9)
    assert m["coverage"] == 1.0 and m["mean_width"] == 2.0 and m["interval_score"] == 2.0
    m = u.interval_metrics(y, y + 0.5, y + 1, y, 0.9)                       # all below the band -> penalised, 0 coverage
    assert m["coverage"] == 0.0 and m["interval_score"] > m["mean_width"]


def test_cqr_calibrated_on_other_rows_reaches_target_coverage():
    rng = np.random.RandomState(0)
    X = rng.normal(size=(6000, 4))
    y = 2 * X[:, 0] + rng.normal(0, 1 + abs(X[:, 1]), 6000)
    base = rm.make_models(quick=True, seed=0, n_jobs=1)["HistGradientBoosting"]
    models = u.fit_quantile_models(base, 0.9, X[:3000], y[:3000])
    assert [m.get_params()["quantile"] for m in models] == pytest.approx([0.05, 0.95, 0.5])   # MAPIE order: lower, upper, median
    cqr = u.conformalize(models, 0.9, X[3000:4500], y[3000:4500])
    med, pis = cqr.predict_interval(X[4500:])
    cov = ((y[4500:] >= pis[:, 0, 0]) & (y[4500:] <= pis[:, 1, 0])).mean()
    assert 0.86 < cov < 0.94                                                 # evaluated on rows NOT used for calibration
