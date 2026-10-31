"""
lhs_rank.py - assemble predictions, rank per finalist, build a diverse consensus list, and test sensitivity to the arbitrary choices
==================================================================================================================================
STRATEGIES (per finalist, among ELIGIBLE candidates = feasible, finite prediction inside the plausible range, support percentile
<= cfg.max_support_percentile)
    A  maximum predicted PCE                          score_A = mu
    B  conservative: maximum calibrated lower bound   score_B = lower        (= mu - half_width up to the asymmetry of the band)
    C  transparent trade-off                          score_C = mu - lambda * half_width,   half_width = (upper - lower)/2
       lambda in [0, 1] interpolates A (0) and B (1).  There is NO empirical basis for a specific lambda (it would need experimental
       outcomes).  The default 0.5 is simply the midpoint; `lhs_score_sensitivity.csv` shows how the top-K changes over lambda and over
       the support threshold, so the dependence is visible instead of hidden.  Uncertainty enters through the band; feasibility and
       support enter as filters, not as weighted terms, so no further weights exist.
CONSENSUS
    eligible for BOTH finalists; consensus score = min(score_C_direct, score_C_twostage) (both models must rate the recipe well; a large
    disagreement lowers the score).  Agreement is NOT evidence of correctness: both finalists share the preprocessing, the data and
    probably systematic errors.  The list is then thinned by a dissimilarity constraint: walking down the score order, a candidate is
    kept only if its Gower distance (lhs_distance.py: categorical mismatch + range-scaled continuous differences, equal weight per
    variable) to every already kept candidate is >= cfg.diversity_min_distance.
Every output row is a MODEL PROPOSAL.  None is an experimentally validated recommendation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from lhs_config import DIRECT_ID, SHORT, TWOSTAGE_ID, LHSConfig
from lhs_design_space import DesignSpace
from lhs_distance import GenomeCoder, pairwise_gower
from lhs_sampler import genome_key_frame

DISPLAY = ["candidate_id", "cell_architecture", "composition_family", "composition_formula", "etl_recipe", "htl_recipe", "bc_recipe", "substrate",
           "process_recipe", "antisolvent_medium", "atmosphere", "anneal_temperature_C", "anneal_time_min", "perovskite_thickness_nm",
           "etl_thickness_nm", "htl_thickness_nm", "bc_thickness_nm", "cell_area_cm2"]


def interpretable_columns(raw: pd.DataFrame) -> pd.DataFrame:
    return raw[[c for c in DISPLAY if c in raw.columns]]


def prediction_table(raw_feasible: pd.DataFrame, rules: pd.DataFrame, support: pd.DataFrame, preds: dict, stage1: pd.DataFrame, cfg: LHSConfig) -> pd.DataFrame:
    """One row per feasible candidate with everything needed for ranking and audit."""
    t = interpretable_columns(raw_feasible).reset_index(drop=True).copy()
    t["tolerance_factor"] = rules["tolerance_factor"].to_numpy()
    t["uncommon_flags"] = rules["uncommon_flags"].to_numpy()
    t["is_uncommon"] = rules["is_uncommon"].to_numpy()
    for c in support.columns:
        t[c] = support[c].to_numpy()
    lo_ok, hi_ok = cfg.plausible_pce_range
    base_ok = ~t["out_of_support"].to_numpy()
    for mid, p in preds.items():
        s = SHORT[mid]
        t[f"{s}_pce"], t[f"{s}_lower"], t[f"{s}_upper"] = p["point"].to_numpy(), p["lower"].to_numpy(), p["upper"].to_numpy()
        t[f"{s}_width"] = t[f"{s}_upper"] - t[f"{s}_lower"]
        t[f"{s}_confidence"] = cfg.confidence
        t[f"{s}_nonfinite"] = ~np.isfinite(p[["point", "lower", "upper"]].to_numpy()).all(axis=1)
        t[f"{s}_pce_out_of_plausible_range"] = (t[f"{s}_pce"] < lo_ok) | (t[f"{s}_pce"] > hi_ok)
        t[f"{s}_point_outside_interval"] = (t[f"{s}_pce"] < t[f"{s}_lower"]) | (t[f"{s}_pce"] > t[f"{s}_upper"])
        t[f"{s}_eligible"] = base_ok & ~t[f"{s}_nonfinite"] & ~t[f"{s}_pce_out_of_plausible_range"]
    for c in stage1.columns:
        t[f"twostage_{c}"] = stage1[c].to_numpy()
    if {"twostage_Voc_hat", "twostage_Jsc_hat", "twostage_FF_hat"} <= set(t.columns):
        t["twostage_VocJscFF_product_pct"] = t["twostage_Voc_hat"] * t["twostage_Jsc_hat"] * t["twostage_FF_hat"]   # V*mA/cm2 = mW/cm2 -> % at 100 mW/cm2
    t["model_abs_difference"] = (t["direct_pce"] - t["twostage_pce"]).abs()
    t["both_eligible"] = t["direct_eligible"] & t["twostage_eligible"]
    t["status"] = "model-proposed (not experimentally validated)"
    return t


def scores(t: pd.DataFrame, s: str, lam: float) -> pd.DataFrame:
    half = (t[f"{s}_upper"] - t[f"{s}_lower"]) / 2
    return pd.DataFrame({"A": t[f"{s}_pce"], "B": t[f"{s}_lower"], "C": t[f"{s}_pce"] - lam * half})


def rank_model(t: pd.DataFrame, s: str, cfg: LHSConfig, lam: float | None = None, mask: np.ndarray | None = None) -> pd.DataFrame:
    lam = cfg.score_lambda if lam is None else lam
    m = t[f"{s}_eligible"].to_numpy() if mask is None else mask
    sc = scores(t, s, lam)
    r = t[m].copy()
    for k in "ABC":
        r[f"score_{k}"] = sc.loc[m, k].to_numpy()
        r[f"rank_{k}"] = r[f"score_{k}"].rank(ascending=False, method="first").astype(int)
    return r.sort_values("rank_C").head(cfg.top_k if mask is None else len(r)).reset_index(drop=True)


def consensus(t: pd.DataFrame, raw_genome: pd.DataFrame, space: DesignSpace, coder: GenomeCoder, cfg: LHSConfig):
    """raw_genome: the sampler's genome table (`g`) restricted to the feasible candidates, indexed like t via candidate_id."""
    m = t["both_eligible"].to_numpy()
    sd, st = scores(t, "direct", cfg.score_lambda)["C"], scores(t, "twostage", cfg.score_lambda)["C"]
    cons = np.minimum(sd, st).to_numpy()
    cons = np.where(m, cons, -np.inf)
    if not m.any():
        return t.iloc[0:0].copy(), dict(selected=0, note="no candidate eligible for both finalists")
    gi = raw_genome.set_index("candidate_id").loc[t["candidate_id"].to_numpy()].reset_index()
    order = np.argsort(-cons)[:min(cfg.consensus_pool, int(m.sum()))]
    kf = genome_key_frame(gi.iloc[order], space)
    Dm = pairwise_gower(coder, kf)
    chosen = []
    for i in range(len(order)):
        if all(Dm[i, j] >= cfg.diversity_min_distance for j in chosen):
            chosen.append(i)
            if len(chosen) == cfg.consensus_k:
                break
    pos = order[chosen]
    out = t.iloc[pos].copy()
    out["consensus_score"] = cons[pos]
    out["consensus_rank"] = np.arange(1, len(pos) + 1)
    sub = Dm[np.ix_(chosen, chosen)]
    off = sub[np.triu_indices(len(chosen), 1)] if len(chosen) > 1 else np.array([np.nan])
    diag = dict(pool=int(len(order)), selected=int(len(chosen)), requested=cfg.consensus_k, min_distance_required=cfg.diversity_min_distance,
                min_pairwise_distance=float(np.nanmin(off)), mean_pairwise_distance=float(np.nanmean(off)), filled=bool(len(chosen) == cfg.consensus_k),
                rule="walk down min(score_C_direct, score_C_twostage); keep a candidate only if its Gower distance to all kept ones >= min_distance_required")
    return out.reset_index(drop=True), diag


def sensitivity(t: pd.DataFrame, cfg: LHSConfig) -> pd.DataFrame:
    """How much does the top-K depend on lambda and on the support threshold?  Compared with the default (lambda, threshold)."""
    K = cfg.top_k
    rows = []
    base_ok = {s: (~t[f"{s}_nonfinite"] & ~t[f"{s}_pce_out_of_plausible_range"]).to_numpy() for s in SHORT.values()}

    def top(s, lam, thr):
        m = base_ok[s] & (t["support_percentile"].to_numpy() <= thr)
        sc = scores(t, s, lam)["C"].to_numpy()
        idx = np.where(m)[0]
        return idx[np.argsort(-sc[idx])[:K]], sc, int(m.sum())

    for s in SHORT.values():
        d_top, d_sc, _ = top(s, cfg.score_lambda, cfg.max_support_percentile)
        for thr in cfg.support_grid:
            for lam in cfg.lambda_grid:
                a_top, a_sc, n_el = top(s, lam, thr)
                inter = len(set(d_top) & set(a_top))
                both = np.where(base_ok[s] & (t["support_percentile"].to_numpy() <= min(thr, cfg.max_support_percentile)))[0]
                rho = spearmanr(d_sc[both], a_sc[both]).statistic if len(both) > 2 else np.nan
                rows.append(dict(model=s, lam=lam, support_threshold=thr, n_eligible=n_el, top_k=K, jaccard_vs_default=inter / (2 * K - inter) if K else np.nan,
                                 spearman_scores_vs_default=rho, mean_pce_top_k=float(t[f"{s}_pce"].to_numpy()[a_top].mean()) if len(a_top) else np.nan,
                                 mean_width_top_k=float(t[f"{s}_width"].to_numpy()[a_top].mean()) if len(a_top) else np.nan,
                                 is_default=bool(lam == cfg.score_lambda and thr == cfg.max_support_percentile)))
    return pd.DataFrame(rows)


def prediction_diagnostics(t: pd.DataFrame) -> dict:
    d = {}
    for s in SHORT.values():
        e = t[t[f"{s}_eligible"]]
        d[s] = dict(n_nonfinite=int(t[f"{s}_nonfinite"].sum()), n_out_of_plausible_range=int(t[f"{s}_pce_out_of_plausible_range"].sum()),
                    n_point_outside_interval=int(t[f"{s}_point_outside_interval"].sum()), n_eligible=int(len(e)),
                    pce_quantiles_all_feasible={k: float(v) for k, v in t[f"{s}_pce"].quantile([0, .01, .5, .99, 1]).items()},
                    width_mean=float(t[f"{s}_width"].mean()),
                    spearman_width_vs_pce=float(spearmanr(t[f"{s}_width"], t[f"{s}_pce"]).statistic))
    d["direct_vs_twostage"] = dict(spearman=float(spearmanr(t["direct_pce"], t["twostage_pce"]).statistic), pearson=float(np.corrcoef(t["direct_pce"], t["twostage_pce"])[0, 1]),
                                   mean_abs_difference=float(t["model_abs_difference"].mean()), p95_abs_difference=float(t["model_abs_difference"].quantile(.95)),
                                   note="agreement between finalists is not evidence of correctness (shared data, preprocessing and likely errors)")
    d["uncommon_share_of_feasible"] = float(t["is_uncommon"].mean())
    d["out_of_support_share_of_feasible"] = float(t["out_of_support"].mean())
    return d
