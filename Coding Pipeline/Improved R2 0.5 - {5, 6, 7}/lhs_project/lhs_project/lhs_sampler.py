"""
lhs_sampler.py - mixed-variable Latin Hypercube Sampling and genome -> PERLA-frame expansion
===========================================================================================
SAMPLING STRATEGY (documented honestly - this is NOT mathematically a single LHS over a continuous space)
    One scrambled Latin hypercube U in [0,1)^(N x 18) is drawn with scipy.stats.qmc.LatinHypercube(seed=cfg.seed).  Each column
    is exactly stratified: row i falls in stratum floor(N*u) and every stratum is hit exactly once.  Columns are then used as:

    continuous dimension  -> scaled to its bounds (linear, or log for anneal time), then rounded to a practical step.
                             Unrounded unit samples are kept (`u_*` columns) so stratification can be verified after the fact;
                             rounding is a documented, separate step (rounding_effect() quantifies what it does).
    categorical dimension -> inverse-CDF of the category probabilities p_k ~ (papers_k)**gamma.  Because the column is
                             stratified, the realised category counts equal N*p_k up to +-1 (exact quotas, no sampling noise
                             in the MARGINAL frequencies).  The assignment of categories to the other dimensions is only
                             pseudo-random: this is stratified quota sampling, not space-filling in the categorical sense.
    conditional dimensions-> ETL / HTL / back-contact recipe and substrate use architecture-specific probability vectors with
                             the same column.  Quotas then hold per architecture only approximately.
    inactive dimensions   -> composition-fraction columns u_Cs / u_FA / u_Br are used only by families that contain those ions;
                             stratification holds for the whole column, not for the active subset.
    Filtering later removes rows and therefore breaks the strata; post-filter coverage is reported separately.

Nothing here reads PCE, Voc, Jsc or FF.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.stats import qmc

import perla_encoding_by_claude as E
from lhs_config import FORBIDDEN_RAW_COLUMNS, LHSConfig
from lhs_design_space import PVK, DesignSpace

DIMS = ["arch", "family", "etl", "htl", "bc", "substrate", "process", "media", "atmosphere",
        "anneal_T", "anneal_logt", "pvk_thick", "etl_thick", "htl_thick", "bc_thick", "u_Cs", "u_FA", "u_Br"]
D = {n: i for i, n in enumerate(DIMS)}


# =========================================================================================
# UNIT HYPERCUBE
# =========================================================================================
def unit_lhs(n: int, seed: int, scramble: bool = True, optimization: str | None = None) -> np.ndarray:
    eng = qmc.LatinHypercube(d=len(DIMS), scramble=scramble, optimization=optimization, seed=seed)
    return eng.random(n)


def stratification_report(U: np.ndarray) -> pd.DataFrame:
    """Per dimension: does every one of the N equal-width strata contain exactly one point?"""
    n = len(U)
    rows = []
    for j, name in enumerate(DIMS):
        s = np.minimum((U[:, j] * n).astype(np.int64), n - 1)
        counts = np.bincount(s, minlength=n)
        rows.append(dict(dimension=name, n=n, strata_covered=int((counts > 0).sum()), max_points_per_stratum=int(counts.max()),
                         exact_lhs=bool((counts == 1).all())))
    return pd.DataFrame(rows)


def _pick(u: np.ndarray, p: np.ndarray) -> np.ndarray:
    c = np.cumsum(p)
    c = c / c[-1]
    return np.minimum(np.searchsorted(c, u, side="right"), len(p) - 1)


def _pick_conditional(u, parent, probs_by_parent) -> np.ndarray:
    out = np.empty(len(u), dtype=np.int64)
    for k, p in probs_by_parent.items():
        m = parent == k
        if m.any():
            if p.sum() <= 0:
                raise ValueError(f"no admissible category for parent '{k}'")
            out[m] = _pick(u[m], p)
    return out


def _scale(u, lo, hi, scale="linear"):
    if scale == "log":
        return np.exp(np.log(lo) + u * (np.log(hi) - np.log(lo)))
    return lo + u * (hi - lo)


def _round(x, step, lo, hi):
    if step is None:
        return x
    r = np.round(x / step) * step
    return np.clip(r, lo, hi)


# =========================================================================================
# GENOME
# =========================================================================================
def sample_genome(space: DesignSpace, U: np.ndarray, seed: int) -> pd.DataFrame:
    cfg, n = space.cfg, len(U)
    spec = {c.name: c for c in space.continuous}
    g = pd.DataFrame(index=np.arange(n))
    g["candidate_id"] = [f"LHS{seed}-{i:07d}" for i in range(n)]
    ai = _pick(U[:, D["arch"]], space.arch["p"].to_numpy())
    arch = space.arch["value"].to_numpy()[ai]
    g["arch"] = arch
    fam = space.families
    fi = _pick(U[:, D["family"]], fam["p"].to_numpy())
    g["family_id"] = fi
    for layer in ("etl", "htl", "bc"):
        g[f"{layer}_id"] = _pick_conditional(U[:, D[layer]], arch, space.layer_probs[layer])
    g["substrate_id"] = _pick_conditional(U[:, D["substrate"]], arch, space.substrate_probs)
    g["process_id"] = _pick(U[:, D["process"]], space.processes["p"].to_numpy())
    g["media_id"] = _pick(U[:, D["media"]], space.media["p"].to_numpy())
    g["atm_id"] = _pick(U[:, D["atmosphere"]], space.atmospheres["p"].to_numpy())
    # ---- continuous (unrounded unit values are returned separately by the caller) --------------------------------
    s = spec["anneal_temperature_C"]
    g["anneal_T"] = _round(_scale(U[:, D["anneal_T"]], s.lo, s.hi), s.round_to, s.lo, s.hi)
    s = spec["anneal_time_min"]
    g["anneal_time_min"] = _round(_scale(U[:, D["anneal_logt"]], s.lo, s.hi, "log"), s.round_to, s.lo, s.hi)
    s = spec["perovskite_thickness_nm"]
    g["pvk_thick"] = _round(_scale(U[:, D["pvk_thick"]], s.lo, s.hi), s.round_to, s.lo, s.hi)
    for layer in ("etl", "htl", "bc"):
        cat = space.layers[layer]
        lo, hi = cat["thick_lo"].to_numpy()[g[f"{layer}_id"]], cat["thick_hi"].to_numpy()[g[f"{layer}_id"]]
        g[f"{layer}_thick"] = _round(_scale(U[:, D[f"{layer}_thick"]], lo, hi), cfg.round_to[f"{layer}_thickness_nm"], lo, hi)   # NaN stays NaN
    # ---- composition ---------------------------------------------------------------------------------------------------
    fr = cfg.round_to["fraction"]

    def famcol(c):
        return fam[c].to_numpy()[fi]
    ions = fam["a_ions"].map(set)
    has = {i: np.array([i in s_ for s_ in ions])[fi] for i in ("Cs", "FA", "MA")}
    free = {i: np.array([i in s_ for s_ in fam["a_free"]])[fi] for i in ("Cs", "FA")}
    rest = fam["a_rest"].to_numpy()[fi]
    cs_s = _round(_scale(U[:, D["u_Cs"]], famcol("Cs_lo").astype(float), famcol("Cs_hi").astype(float)), fr, 0.0, 1.0)
    fa_s = _round(_scale(U[:, D["u_FA"]], famcol("FA_lo").astype(float), famcol("FA_hi").astype(float)), fr, 0.0, 1.0)
    cs = np.where(free["Cs"], cs_s, np.where(has["Cs"], 1.0, 0.0))
    fa = np.where(free["FA"], fa_s, np.where(has["FA"] & (rest != "FA"), 1.0, 0.0))
    ma = np.where(has["MA"] & (rest == "MA"), 1.0 - cs - fa, np.where(has["MA"], 1.0, 0.0))
    fa = np.where(has["FA"] & (rest == "FA"), 1.0 - cs, fa)
    g["frac_A_Cs"], g["frac_A_FA"], g["frac_A_MA"] = np.round(cs, 4), np.round(fa, 4), np.round(ma, 4)
    c_free = fam["c_free"].map(lambda v: v == "Br").to_numpy()[fi]
    has_br = np.array(["Br" in set(c) for c in fam["c_ions"]])[fi]
    br_s = _round(_scale(U[:, D["u_Br"]], famcol("Br_lo").astype(float), famcol("Br_hi").astype(float)), fr, 0.0, 1.0)
    br = np.where(c_free, br_s, np.where(has_br, 1.0, 0.0))
    g["frac_C_Br"], g["frac_C_I"] = np.round(br, 4), np.round(1.0 - br, 4)
    # ---- process-derived and categorical labels ---------------------------------------------------------------------
    pr = space.processes
    g["antisolvent"] = pr["qb"].to_numpy()[g["process_id"]]
    g["solvent_annealing"] = pr["sa"].to_numpy()[g["process_id"]]
    g["antisolvent_medium"] = np.where(g["antisolvent"], space.media["value"].to_numpy()[g["media_id"]], "Unknown")
    g["atmosphere"] = space.atmospheres["token"].to_numpy()[g["atm_id"]]
    g["substrate"] = space.substrates["value"].to_numpy()[g["substrate_id"]]
    g["composition_family"] = famcol("key")
    g["composition_inorganic"] = famcol("inorganic").astype(bool)
    g["composition_leadfree"] = famcol("leadfree").astype(bool)
    g["etl_recipe"] = space.layers["etl"]["label"].to_numpy()[g["etl_id"]]
    g["htl_recipe"] = space.layers["htl"]["label"].to_numpy()[g["htl_id"]]
    g["bc_recipe"] = space.layers["bc"]["label"].to_numpy()[g["bc_id"]]
    g["process_recipe"] = pr["label"].to_numpy()[g["process_id"]]
    g["composition_formula"] = [_formula(a, b, c, d, e) for a, b, c, d, e in zip(g["frac_A_Cs"], g["frac_A_FA"], g["frac_A_MA"], g["frac_C_Br"], g["frac_C_I"])]
    return g


def _formula(cs, fa, ma, br, i) -> str:
    a = "".join(f"{n}{v:g}" for n, v in (("Cs", cs), ("FA", fa), ("MA", ma)) if v > 0)
    c = "".join(f"{n}{3 * v:.3g}" for n, v in (("I", i), ("Br", br)) if v > 0)
    return f"{a}Pb({c})"


def genome_key_frame(g: pd.DataFrame, space: DesignSpace) -> pd.DataFrame:
    """Candidate genome -> the same representation `lhs_design_space.genome_keys` returns for a PERLA frame."""
    out = pd.DataFrame(index=g.index)
    out["arch"] = g["arch"].to_numpy()
    out["family"] = g["composition_family"].to_numpy()
    for layer in ("etl", "htl", "bc"):
        out[layer] = space.layers[layer]["key"].to_numpy()[g[f"{layer}_id"]]
    out["substrate"] = g["substrate"].to_numpy()
    out["process"] = space.processes["key"].to_numpy()[g["process_id"]]
    out["media"] = np.where(g["antisolvent"], space.media["key"].to_numpy()[g["media_id"]], "unknown")
    out["atmosphere"] = g["atmosphere"].to_numpy()
    out["anneal_T"] = g["anneal_T"].to_numpy(float)
    out["anneal_logt"] = np.log10(g["anneal_time_min"].to_numpy(float))
    for c in ("pvk_thick", "etl_thick", "htl_thick", "bc_thick"):
        out[c] = g[c].to_numpy(float)
    out["frac_Cs"], out["frac_FA"], out["frac_Br"] = g["frac_A_Cs"].to_numpy(float), g["frac_A_FA"].to_numpy(float), g["frac_C_Br"].to_numpy(float)
    out["paper"] = "candidate"
    return out


def raw_candidate_table(g: pd.DataFrame, U: np.ndarray, cfg: LHSConfig) -> pd.DataFrame:
    """Interpretable columns + fixed context + the unrounded LHS unit samples (lhs_raw_candidates.csv)."""
    cols = ["candidate_id", "arch", "composition_family", "composition_formula", "frac_A_Cs", "frac_A_FA", "frac_A_MA", "frac_C_Br", "frac_C_I",
            "etl_recipe", "htl_recipe", "bc_recipe", "substrate", "process_recipe", "antisolvent", "antisolvent_medium", "solvent_annealing",
            "atmosphere", "anneal_T", "anneal_time_min", "pvk_thick", "etl_thick", "htl_thick", "bc_thick", "composition_inorganic",
            "composition_leadfree", "family_id", "etl_id", "htl_id", "bc_id", "substrate_id", "process_id", "media_id", "atm_id"]
    t = g[cols].rename(columns={"arch": "cell_architecture", "anneal_T": "anneal_temperature_C", "pvk_thick": "perovskite_thickness_nm",
                                "etl_thick": "etl_thickness_nm", "htl_thick": "htl_thickness_nm", "bc_thick": "bc_thickness_nm"}).copy()
    t["cell_area_cm2"] = cfg.cell_area_cm2
    t["cell_flexible"], t["cell_semitransparent"] = cfg.cell_flexible, cfg.cell_semitransparent
    t["perovskite_single_crystal"], t["perovskite_band_gap_graded"] = cfg.perovskite_single_crystal, cfg.perovskite_band_gap_graded
    for j, name in enumerate(DIMS):
        t[name if name.startswith("u_") else f"u_{name}"] = U[:, j].astype(np.float32)
    return t


def rounding_effect(U: np.ndarray, g: pd.DataFrame, space: DesignSpace) -> pd.DataFrame:
    """How much does rounding change the continuous LHS marginals?  Compares the rounded values with the unrounded scaling of
    the same unit samples (max abs change relative to the range, number of distinct levels, KS statistic vs uniform)."""
    from scipy.stats import kstest
    rows, spec = [], {c.name: c for c in space.continuous}
    for name, col, dim in (("anneal_temperature_C", "anneal_T", "anneal_T"), ("anneal_time_min", "anneal_time_min", "anneal_logt"),
                           ("perovskite_thickness_nm", "pvk_thick", "pvk_thick")):
        s = spec[name]
        raw = _scale(U[:, D[dim]], s.lo, s.hi, s.scale)
        rounded = g[col].to_numpy(float)
        ur = (np.log(rounded) - np.log(s.lo)) / (np.log(s.hi) - np.log(s.lo)) if s.scale == "log" else (rounded - s.lo) / (s.hi - s.lo)
        rows.append(dict(variable=name, round_to=s.round_to, distinct_levels=int(len(np.unique(rounded))),
                         max_abs_change_over_range=float(np.max(np.abs(rounded - raw)) / (s.hi - s.lo)),
                         ks_stat_rounded_vs_uniform=float(kstest(ur, "uniform").statistic),
                         ks_stat_unrounded_vs_uniform=float(kstest(U[:, D[dim]], "uniform").statistic),
                         note="levels = rounding steps; the strata of the unrounded LHS are not preserved at this resolution"))
    return pd.DataFrame(rows)


# =========================================================================================
# GENOME -> RAW PERLA FRAME (the 38 columns PerlaEncoder reads)
# =========================================================================================
def genome_to_perla_frame(g: pd.DataFrame, space: DesignSpace) -> pd.DataFrame:
    cfg, fam, pr = space.cfg, space.families, space.processes
    n = len(g)
    nan = np.nan
    cols = {c: [None] * n for c in (
        "Perovskite_composition_a_ions__parsed", "Perovskite_composition_a_ions_coefficients__parsed", "Perovskite_composition_b_ions__parsed",
        "Perovskite_composition_b_ions_coefficients__parsed", "Perovskite_composition_c_ions__parsed", "Perovskite_composition_c_ions_coefficients__parsed",
        "ETL_stack_sequence__parsed", "ETL_deposition_procedure__parsed", "HTL_stack_sequence__parsed", "HTL_deposition_procedure__parsed",
        "Backcontact_stack_sequence__parsed", "Backcontact_deposition_procedure__parsed", PVK["proc"], PVK["solv"], PVK["atm"], PVK["agg"],
        E.ANNEAL_TEMP_COL, E.ANNEAL_TIME_COL)}
    fidx, eid, hid, bid = g["family_id"].to_numpy(), g["etl_id"].to_numpy(), g["htl_id"].to_numpy(), g["bc_id"].to_numpy()
    pid, aid = g["process_id"].to_numpy(), g["atm_id"].to_numpy()
    cs, fa, ma = g["frac_A_Cs"].to_numpy(), g["frac_A_FA"].to_numpy(), g["frac_A_MA"].to_numpy()
    br, iod = g["frac_C_Br"].to_numpy(), g["frac_C_I"].to_numpy()
    T, t = g["anneal_T"].to_numpy(float), g["anneal_time_min"].to_numpy(float)
    f_a, f_c = fam["a_ions"].tolist(), fam["c_ions"].tolist()
    f_b = fam["b_ions"].tolist()
    etl, htl, bc = space.layers["etl"], space.layers["htl"], space.layers["bc"]
    atm_tok = space.atmospheres["token"].tolist()
    for r in range(n):
        f = fidx[r]
        a_val = {"Cs": cs[r], "FA": fa[r], "MA": ma[r]}
        cols["Perovskite_composition_a_ions__parsed"][r] = [list(f_a[f])]
        cols["Perovskite_composition_a_ions_coefficients__parsed"][r] = [[float(a_val[i]) for i in f_a[f]]]
        cols["Perovskite_composition_b_ions__parsed"][r] = [list(f_b[f])]
        cols["Perovskite_composition_b_ions_coefficients__parsed"][r] = [[1.0] * len(f_b[f])]
        c_val = {"Br": 3.0 * br[r], "I": 3.0 * iod[r]}
        cols["Perovskite_composition_c_ions__parsed"][r] = [list(f_c[f])]
        cols["Perovskite_composition_c_ions_coefficients__parsed"][r] = [[round(float(c_val[i]), 4) for i in f_c[f]]]
        cols["ETL_stack_sequence__parsed"][r], cols["ETL_deposition_procedure__parsed"][r] = etl["stack"].iloc[eid[r]], etl["proc"].iloc[eid[r]]
        cols["HTL_stack_sequence__parsed"][r], cols["HTL_deposition_procedure__parsed"][r] = htl["stack"].iloc[hid[r]], htl["proc"].iloc[hid[r]]
        cols["Backcontact_stack_sequence__parsed"][r], cols["Backcontact_deposition_procedure__parsed"][r] = bc["stack"].iloc[bid[r]], bc["proc"].iloc[bid[r]]
        cols[PVK["proc"]][r], cols[PVK["solv"]][r], cols[PVK["agg"]][r] = pr["proc"].iloc[pid[r]], pr["solv"].iloc[pid[r]], pr["agg"].iloc[pid[r]]
        cols[PVK["atm"]][r] = [[[atm_tok[aid[r]]]]]
        cols[E.ANNEAL_TEMP_COL][r] = [[[float(T[r])]]]
        cols[E.ANNEAL_TIME_COL][r] = [[[float(t[r])]]]
    out = pd.DataFrame(cols, index=np.arange(n))
    out["Cell_architecture"] = g["arch"].to_numpy()
    out["Substrate_stack_sequence"] = g["substrate"].to_numpy()
    out["Perovskite_thickness__sum"] = g["pvk_thick"].to_numpy(float)
    out["ETL_thickness__sum"] = g["etl_thick"].to_numpy(float)
    out["HTL_thickness__sum"] = g["htl_thick"].to_numpy(float)
    out["Backcontact_thickness_list__sum"] = g["bc_thick"].to_numpy(float)
    out[PVK["steps"]] = pr["steps"].to_numpy()[pid].astype(int)
    out[PVK["qb"]] = pd.Series([bool(x) for x in pr["qb"].to_numpy()[pid]], dtype=object)
    out[PVK["sa"]] = pd.Series([bool(x) for x in pr["sa"].to_numpy()[pid]], dtype=object)
    out[PVK["media"]] = g["antisolvent_medium"].to_numpy(dtype=object)
    out["Perovskite_composition_inorganic"] = pd.Series([bool(x) for x in g["composition_inorganic"]], dtype=object)
    out["Perovskite_composition_leadfree"] = pd.Series([bool(x) for x in g["composition_leadfree"]], dtype=object)
    for col, val in (("Cell_flexible", cfg.cell_flexible), ("Cell_semitransparent", cfg.cell_semitransparent),
                     ("Perovskite_single_crystal", cfg.perovskite_single_crystal), ("Perovskite_band_gap_graded", cfg.perovskite_band_gap_graded)):
        out[col] = pd.Series([bool(val)] * n, dtype=object)
    out["Cell_area_measured__parsed"] = float(cfg.cell_area_cm2)
    out["band_gap__mean"] = nan                                  # neutralised: KNN-imputed by the existing pipeline
    out["Perovskite_band_gap_estimation_basis"] = nan            # neutralised
    out[E.GROUP_COL] = None                                      # identifier placeholder (never used in transform)
    return out


def check_frame_schema(frame: pd.DataFrame) -> None:
    """Fail loudly unless the frame has exactly the raw model-input columns and no target / measured-output column."""
    enc = E.PerlaEncoder()
    need = ({c for c, *_ in E.MULTILABEL_COLS} | {c for c, _ in E.ONEHOT_COLS} | set(E.BOOLEAN_COLS) | set(E.NUMERIC_COLS)
            | {c for c, _ in E.TARGET_ENCODED_COLS} | {E.ANNEAL_TEMP_COL, E.ANNEAL_TIME_COL, enc.group_col}
            | {c for pair in E.COMP_COLS.values() for c in pair})
    missing, extra = need - set(frame.columns), set(frame.columns) - need
    if missing or extra:
        raise AssertionError(f"candidate frame schema mismatch; missing={sorted(missing)} extra={sorted(extra)}")
    bad = set(frame.columns) & (set(FORBIDDEN_RAW_COLUMNS) | set(E.TARGET_COLS.values()))
    if bad:
        raise AssertionError(f"forbidden (target / measured-output / metadata) columns in the candidate frame: {sorted(bad)}")


def sample_candidates(space: DesignSpace, cfg: LHSConfig):
    """-> (genome, unit matrix, raw candidate table)."""
    U = unit_lhs(cfg.n_candidates, cfg.seed, cfg.lhs_scramble, cfg.lhs_optimization)
    g = sample_genome(space, U, cfg.seed)
    return g, U, raw_candidate_table(g, U, cfg)
