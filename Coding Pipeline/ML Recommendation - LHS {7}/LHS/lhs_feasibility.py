"""
lhs_feasibility.py - independent, individually testable feasibility rules with recorded reasons
==============================================================================================
Four distinct notions are kept apart (they are NOT merged into one "feasible" flag):

  invalid        numerically / physically impossible or inconsistent with the encoder's schema      -> REJECTED
  incompatible   fabrication-incompatible, justified by explicit domain knowledge or by project data -> REJECTED
  uncommon       a categorical combination seen in fewer than `min_pair_papers` distinct TRAIN papers -> FLAGGED
                 (rejected only if cfg.reject_uncommon; "rare" is not "impossible")
  out_of_support farther (genome Gower distance) from real devices than `max_support_percentile` of held-out-paper
                 reference rows                                                                      -> FLAGGED, excluded from ranking

Every rule is a function (genome, key_frame, space, cfg) -> boolean array (True = the candidate VIOLATES the rule), registered
in RULES with its severity and the reason it exists.  A rejected candidate keeps ALL violated rule names.
Rules deliberately not implemented (no reliable basis in the project data or in settled domain knowledge): solvent /
anti-solvent compatibility tables, ETL/HTL energy-level alignment, material-specific annealing windows.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from lhs_config import LHSConfig
from lhs_design_space import IONIC_RADII_PM, ORGANIC_A, DesignSpace
from lhs_sampler import genome_key_frame

TOL = 1e-6


@dataclass
class Rule:
    name: str
    severity: str            # invalid | incompatible | uncommon
    why: str
    fn: Callable


def tolerance_factor(g: pd.DataFrame) -> np.ndarray:
    r = IONIC_RADII_PM
    rA = g["frac_A_Cs"] * r["Cs"] + g["frac_A_FA"] * r["FA"] + g["frac_A_MA"] * r["MA"]
    rX = g["frac_C_I"] * r["I"] + g["frac_C_Br"] * r["Br"]
    return ((rA + rX) / (np.sqrt(2.0) * (r["Pb"] + rX))).to_numpy(float)


# ----------------------------------------------------------------------------------------- invalid
def r_nonfinite_continuous(g, kf, sp, cfg):
    bad = np.zeros(len(g), bool)
    for c in ("anneal_T", "anneal_time_min", "pvk_thick"):
        v = g[c].to_numpy(float)
        bad |= ~np.isfinite(v) | (v <= 0)
    for layer in ("etl", "htl", "bc"):
        v = g[f"{layer}_thick"].to_numpy(float)
        lo = sp.layers[layer]["thick_lo"].to_numpy(float)[g[f"{layer}_id"].to_numpy()]
        specified = np.isfinite(lo)
        bad |= specified & (~np.isfinite(v) | (v <= 0))          # specified recipe but no usable thickness
        bad |= (~specified) & np.isfinite(v)                    # unspecified recipe must stay missing
    return bad


def r_continuous_out_of_bounds(g, kf, sp, cfg):
    bad = np.zeros(len(g), bool)
    spec = {c.name: c for c in sp.continuous}
    for col, name in (("anneal_T", "anneal_temperature_C"), ("anneal_time_min", "anneal_time_min"), ("pvk_thick", "perovskite_thickness_nm")):
        v, s = g[col].to_numpy(float), spec[name]
        bad |= (v < s.lo - TOL) | (v > s.hi + TOL)
    for layer in ("etl", "htl", "bc"):
        v = g[f"{layer}_thick"].to_numpy(float)
        cat, i = sp.layers[layer], g[f"{layer}_id"].to_numpy()
        lo, hi = cat["thick_lo"].to_numpy(float)[i], cat["thick_hi"].to_numpy(float)[i]
        with np.errstate(invalid="ignore"):
            bad |= np.isfinite(lo) & ((v < lo - TOL) | (v > hi + TOL))
    return bad


def r_composition_fraction_invalid(g, kf, sp, cfg):
    A = g[["frac_A_Cs", "frac_A_FA", "frac_A_MA"]].to_numpy(float)
    C = g[["frac_C_Br", "frac_C_I"]].to_numpy(float)
    bad = (A < -TOL).any(1) | (A > 1 + TOL).any(1) | (C < -TOL).any(1) | (C > 1 + TOL).any(1)
    bad |= np.abs(A.sum(1) - 1) > TOL
    bad |= np.abs(C.sum(1) - 1) > TOL
    fam = sp.families
    ions = fam["a_ions"].map(set)
    for j, ion in enumerate(("Cs", "FA", "MA")):
        in_fam = np.array([ion in s for s in ions])[g["family_id"].to_numpy()]
        bad |= (in_fam & (A[:, j] < 0.005)) | (~in_fam & (A[:, j] > TOL))       # the formula must match its family
    br_in = np.array(["Br" in set(c) for c in fam["c_ions"]])[g["family_id"].to_numpy()]
    i_in = np.array(["I" in set(c) for c in fam["c_ions"]])[g["family_id"].to_numpy()]
    bad |= (br_in & (C[:, 0] < 0.005)) | (~br_in & (C[:, 0] > TOL)) | (i_in & (C[:, 1] < 0.005)) | (~i_in & (C[:, 1] > TOL))
    return bad


def r_quench_inconsistent(g, kf, sp, cfg):
    med, anti = g["antisolvent_medium"].astype(str), g["antisolvent"].astype(bool).to_numpy()
    valid = set(sp.media["value"])
    return (anti & ~med.isin(valid).to_numpy()) | (~anti & (med != "Unknown").to_numpy())


def r_unknown_category(g, kf, sp, cfg):
    bad = ~g["arch"].isin(sp.arch["value"]).to_numpy()
    for col, n in (("family_id", len(sp.families)), ("etl_id", len(sp.layers["etl"])), ("htl_id", len(sp.layers["htl"])),
                   ("bc_id", len(sp.layers["bc"])), ("substrate_id", len(sp.substrates)), ("process_id", len(sp.processes)),
                   ("media_id", len(sp.media)), ("atm_id", len(sp.atmospheres))):
        v = g[col].to_numpy()
        bad |= (v < 0) | (v >= n)
    return bad


def r_derived_flag_mismatch(g, kf, sp, cfg):
    organic = (g["frac_A_FA"] + g["frac_A_MA"]).to_numpy(float) > TOL
    return (g["composition_inorganic"].astype(bool).to_numpy() == organic) | g["composition_leadfree"].astype(bool).to_numpy()   # B site is Pb only


# ----------------------------------------------------------------------------------------- incompatible
def r_ma_thermal_budget(g, kf, sp, cfg):
    return (g["frac_A_MA"].to_numpy(float) >= cfg.ma_min_fraction_for_thermal_rule) & (g["anneal_T"].to_numpy(float) > cfg.ma_max_anneal_temperature_C)


def r_tolerance_factor(g, kf, sp, cfg):
    t = tolerance_factor(g)
    return (t < cfg.tolerance_factor_range[0]) | (t > cfg.tolerance_factor_range[1])


def r_layer_unsupported_for_architecture(g, kf, sp, cfg):
    """Layer recipe seen in fewer than min_papers_category papers WITH this architecture (e.g. Spiro-MeOTAD as bottom HTL of a pin cell)."""
    bad = np.zeros(len(g), bool)
    arch = g["arch"].to_numpy()
    for layer in ("etl", "htl", "bc"):
        cat, i = sp.layers[layer], g[f"{layer}_id"].to_numpy()
        n = np.where(arch == "nip", cat["n_papers_nip"].to_numpy()[i], cat["n_papers_pin"].to_numpy()[i])
        bad |= n < cfg.min_papers_category
    n = np.where(arch == "nip", sp.substrates["n_papers_nip"].to_numpy()[g["substrate_id"].to_numpy()], sp.substrates["n_papers_pin"].to_numpy()[g["substrate_id"].to_numpy()])
    return bad | (n < cfg.min_papers_category)


# ----------------------------------------------------------------------------------------- uncommon
def _pair_rule(a, b):
    def fn(g, kf, sp, cfg):
        table = sp.pair_support[(a, b)]
        n = np.fromiter((table.get(k, 0) for k in zip(kf[a].to_numpy(), kf[b].to_numpy())), dtype=np.int64, count=len(kf))
        return n < cfg.min_pair_papers
    return fn


RULES: list[Rule] = [
    Rule("nonfinite_or_nonpositive_continuous", "invalid", "a required continuous value is NaN/inf/<=0, or an unspecified-thickness recipe received a thickness", r_nonfinite_continuous),
    Rule("continuous_out_of_bounds", "invalid", "value outside the documented sampling bounds (after rounding)", r_continuous_out_of_bounds),
    Rule("composition_fraction_invalid", "invalid", "site fractions negative / not summing to 1 / inconsistent with the composition family", r_composition_fraction_invalid),
    Rule("quench_inconsistent", "invalid", "anti-solvent flag and medium disagree (training data: flag False <=> medium 'Unknown')", r_quench_inconsistent),
    Rule("unknown_category", "invalid", "category id outside its catalogue / architecture not in the encoder vocabulary", r_unknown_category),
    Rule("derived_flag_mismatch", "invalid", "inorganic / lead-free flags inconsistent with the composition (the encoder reads these booleans)", r_derived_flag_mismatch),
    Rule("ma_thermal_budget", "incompatible", "MA-majority perovskite annealed above the configured temperature: thermal decomposition (domain knowledge; threshold in config)", r_ma_thermal_budget),
    Rule("tolerance_factor_out_of_range", "incompatible", "Goldschmidt tolerance factor outside the configured range (domain knowledge; nearly vacuous for the default families)", r_tolerance_factor),
    Rule("layer_unsupported_for_architecture", "incompatible", "layer / substrate used with this architecture in too few papers (project data)", r_layer_unsupported_for_architecture),
] + [Rule(f"rare_pair_{a}_{b}", "uncommon", f"({a}, {b}) combination seen in fewer than min_pair_papers distinct TRAIN papers", _pair_rule(a, b))
     for a, b in (("etl", "htl"), ("htl", "bc"), ("family", "process"), ("family", "htl"), ("arch", "process"), ("etl", "family"))]


def evaluate_rules(g: pd.DataFrame, space: DesignSpace, cfg: LHSConfig, rules: list[Rule] | None = None) -> pd.DataFrame:
    """-> per-candidate DataFrame: one boolean column per rule, `status`, `reasons`, `uncommon_flags`, `rejected`."""
    rules = RULES if rules is None else rules
    unknown = np.asarray(r_unknown_category(g, None, space, cfg), bool)
    g_eval = g
    if unknown.any():           # ids outside the catalogues must be REJECTED with a reason, not crash the other rules
        g_eval = g.copy()
        for col, n in (("family_id", len(space.families)), ("etl_id", len(space.layers["etl"])), ("htl_id", len(space.layers["htl"])),
                       ("bc_id", len(space.layers["bc"])), ("substrate_id", len(space.substrates)), ("process_id", len(space.processes)),
                       ("media_id", len(space.media)), ("atm_id", len(space.atmospheres))):
            g_eval[col] = np.clip(g_eval[col].to_numpy(), 0, n - 1)
        g_eval.loc[~g_eval["arch"].isin(space.arch["value"]), "arch"] = space.arch["value"].iloc[0]
    kf = genome_key_frame(g_eval, space)
    res = pd.DataFrame({r.name: np.asarray(r.fn(g_eval, kf, space, cfg), bool) for r in rules}, index=g.index)
    if "unknown_category" in res:
        res["unknown_category"] = unknown
    sev = {r.name: r.severity for r in rules}
    inv = [n for n, s in sev.items() if s == "invalid"]
    inc = [n for n, s in sev.items() if s == "incompatible"]
    unc = [n for n, s in sev.items() if s == "uncommon"]
    status = np.where(res[inv].any(axis=1), "invalid", np.where(res[inc].any(axis=1), "incompatible", "ok"))
    out = pd.DataFrame(index=g.index)
    out["candidate_id"] = g["candidate_id"].to_numpy()
    out["status"] = status
    out["reasons"] = [";".join(np.array(inv + inc)[row]) for row in res[inv + inc].to_numpy()]
    out["uncommon_flags"] = [";".join(np.array(unc)[row]) for row in res[unc].to_numpy()]
    out["is_uncommon"] = out["uncommon_flags"] != ""
    out["rejected"] = (out["status"] != "ok") | (cfg.reject_uncommon & out["is_uncommon"])
    out.loc[cfg.reject_uncommon & out["is_uncommon"] & (out["status"] == "ok"), "status"] = "uncommon"
    out["tolerance_factor"] = tolerance_factor(g)
    return pd.concat([out, res], axis=1)


def filter_candidates(g: pd.DataFrame, space: DesignSpace, cfg: LHSConfig):
    """-> (feasible genome, rejected genome with reasons, per-candidate rule table, funnel list)."""
    ev = evaluate_rules(g, space, cfg)
    n0 = len(g)
    invalid = ev["status"] == "invalid"
    incompat = ev["status"] == "incompatible"
    funnel = [dict(stage="raw LHS candidates", remaining=n0, removed=0),
              dict(stage="after invalid-candidate rules", remaining=int((~invalid).sum()), removed=int(invalid.sum())),
              dict(stage="after fabrication-incompatibility rules", remaining=int((~invalid & ~incompat).sum()), removed=int((~invalid & incompat).sum()))]
    if cfg.reject_uncommon:
        unc = ev["status"] == "uncommon"
        funnel.append(dict(stage="after rejecting uncommon combinations", remaining=int((~invalid & ~incompat & ~unc).sum()), removed=int(unc.sum())))
    keep = ~ev["rejected"].to_numpy()
    n_unc = int((ev["is_uncommon"] & keep).sum())
    funnel.append(dict(stage="feasible (flagged uncommon among them: %d)" % n_unc, remaining=int(keep.sum()), removed=0))
    rej = g.loc[~keep].copy()
    rej["status"] = ev.loc[~keep, "status"].to_numpy()
    rej["reasons"] = np.where(ev.loc[~keep, "reasons"] != "", ev.loc[~keep, "reasons"], ev.loc[~keep, "uncommon_flags"])
    return g.loc[keep].copy(), rej, ev, funnel
