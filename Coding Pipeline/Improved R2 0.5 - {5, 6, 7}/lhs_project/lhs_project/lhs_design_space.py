"""
lhs_design_space.py - the documented, configurable candidate design space
=========================================================================
Built from the 70 % TRAIN split ONLY (validation / test rows are never read here).

WHAT A CANDIDATE IS
    The trained models do not read "fabrication parameters" directly: PerlaEncoder reads 38 raw PERLA columns (nested
    lists of tokens, numeric sums, three-state booleans) and turns them into 306 features.  A candidate is therefore
    defined by a small interpretable *genome*, and `lhs_sampler.genome_to_perla_frame` expands the genome into exactly
    those 38 raw columns.  Genome variables:

      categorical  architecture | composition family | ETL recipe | HTL recipe | back-contact recipe | substrate |
                   perovskite process recipe | anti-solvent medium | synthesis atmosphere
      continuous   anneal temperature | anneal time (log) | perovskite / ETL / HTL / back-contact thickness |
                   composition fractions (Cs, FA, Br) inside the chosen composition family

    A "recipe" (e.g. ETL = stack [TiO2-c, TiO2-mp] + procedures [spray-pyrolysis, spin-coating]) is an OBSERVED
    combination taken as one categorical level.  This keeps stack and deposition procedure internally consistent and is
    the controlled strategy for the very high cardinality of the layer / composition fields (1011 distinct ETL stacks,
    1277 HTL stacks, 134 composition families in TRAIN).  Levels need >= `min_papers_category` distinct papers.

    Conditional structure that is built in (supported by the data, see README): ETL / HTL / back-contact recipes and the
    substrate are sampled conditionally on the architecture (nip vs pin use different charge-transport layers).

    `genome_keys(df)` maps any PERLA-format frame (training rows or candidate frames) to this genome representation;
    the same function is used for catalogue construction, support distances and the round-trip test.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import perla_encoding_by_claude as E
from lhs_config import ContinuousSpec, LHSConfig

BAD_TOKENS = set(E.MISSING_TOKENS) | {"<na>", "unknown", "unreported"}

CAT_VARS = ["arch", "family", "etl", "htl", "bc", "substrate", "process", "media", "atmosphere"]
CONT_VARS = ["anneal_T", "anneal_logt", "pvk_thick", "etl_thick", "htl_thick", "bc_thick", "frac_Cs", "frac_FA", "frac_Br"]
GENOME_VARS = CAT_VARS + CONT_VARS

LAYER_COLS = {                                     # layer -> (stack column, procedure column, thickness column, vocab prefixes)
    "etl": ("ETL_stack_sequence__parsed", "ETL_deposition_procedure__parsed", "ETL_thickness__sum", "etl_stack", "etl_proc"),
    "htl": ("HTL_stack_sequence__parsed", "HTL_deposition_procedure__parsed", "HTL_thickness__sum", "htl_stack", "htl_proc"),
    "bc": ("Backcontact_stack_sequence__parsed", "Backcontact_deposition_procedure__parsed", "Backcontact_thickness_list__sum",
           "bc_stack", "bc_proc"),
}
PVK = {"proc": "Perovskite_deposition_procedure__parsed", "solv": "Perovskite_deposition_solvents__parsed_clean",
       "steps": "Perovskite_deposition_number_of_deposition_steps__parsed", "qb": "Perovskite_deposition_quenching_induced_crystallisation",
       "sa": "Perovskite_deposition_solvent_annealing", "media": "Perovskite_deposition_quenching_media",
       "agg": "Perovskite_deposition_aggregation_state_of_reactants__parsed",
       "atm": "Perovskite_deposition_synthesis_atmosphere_parsed", "thick": "Perovskite_thickness__sum"}
ORGANIC_A = {"MA", "FA", "GU", "NH4", "EA", "DMA", "PA", "TMA", "BA", "PEA", "5-AVA", "EDA", "ThMA", "TBA"}
# Effective ionic radii (pm) used for the Goldschmidt tolerance factor (Kieslich, Sun, Cheetham, Chem. Sci. 2015).
IONIC_RADII_PM = {"MA": 217.0, "FA": 253.0, "Cs": 167.0, "Pb": 119.0, "I": 220.0, "Br": 196.0}


# =========================================================================================
# GENOME KEYS  (training rows AND candidate frames go through this one function)
# =========================================================================================
def _j(x) -> str:
    return json.dumps(x, default=str)


def _site(ions_cell, coef_cell):
    """-> (sorted ion tuple over all layers, {ion: normalised fraction} for a single valid layer else None)."""
    if ions_cell is None or coef_cell is None:
        return (), None
    try:
        ions = tuple(sorted({str(i) for layer in ions_cell for i in layer if i is not None}))
        if len(ions_cell) != 1 or len(coef_cell) != 1 or len(ions_cell[0]) != len(coef_cell[0]):
            return ions, None
        coefs = [float(c) for c in coef_cell[0]]
        tot = sum(coefs)
        if not np.isfinite(tot) or tot <= 0 or any((not np.isfinite(c)) or c < 0 for c in coefs):
            return ions, None
        fr = {}
        for i, c in zip(ions_cell[0], coefs):
            fr[str(i)] = fr.get(str(i), 0.0) + c / tot
        return ions, fr
    except (TypeError, ValueError):
        return (), None


def _flag(v):
    return None if E._is_missing(v) else bool(v)


def media_key(v) -> str:
    if E._is_missing(v):
        return "unknown"
    s = str(v).strip().casefold()
    return "unknown" if s in E.MISSING_TOKENS else s


def atmosphere_key(cell) -> str:
    t = sorted(E.tokens_nested(cell))
    return "+".join(t) if t else "unreported"


def _num(s) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce").astype("float64")
    return s.where(s > 0)


def genome_keys(df: pd.DataFrame) -> pd.DataFrame:
    """PERLA-format frame -> genome representation (strings for categorical variables, floats for continuous ones).
    Unknown / unresolvable values become NaN (continuous) or the string '?' / 'unreported' (categorical)."""
    A = [_site(a, b) for a, b in zip(df[E.COMP_COLS["A"][0]], df[E.COMP_COLS["A"][1]])]
    B = [_site(a, b) for a, b in zip(df[E.COMP_COLS["B"][0]], df[E.COMP_COLS["B"][1]])]
    C = [_site(a, b) for a, b in zip(df[E.COMP_COLS["C"][0]], df[E.COMP_COLS["C"][1]])]
    out = pd.DataFrame(index=df.index)
    out["arch"] = df["Cell_architecture"].astype(str).str.strip()
    out["family"] = ["|".join("+".join(s[0]) if s[0] else "?" for s in trio) for trio in zip(A, B, C)]
    for name, site, ion in (("frac_Cs", A, "Cs"), ("frac_FA", A, "FA"), ("frac_Br", C, "Br")):
        out[name] = [np.nan if s[1] is None else s[1].get(ion, 0.0) for s in site]
    for layer, (sc, pc, tc, _, _) in LAYER_COLS.items():
        out[layer] = [_j([s, p]) for s, p in zip(df[sc], df[pc])]
    out["substrate"] = df["Substrate_stack_sequence"].astype(str).str.strip()
    steps = pd.to_numeric(df[PVK["steps"]], errors="coerce").fillna(-1).astype(int)
    out["process"] = [_j([p, s, int(n), _flag(q), _flag(a)]) for p, s, n, q, a in
                      zip(df[PVK["proc"]], df[PVK["solv"]], steps, df[PVK["qb"]], df[PVK["sa"]])]
    out["media"] = [media_key(v) for v in df[PVK["media"]]]
    out["atmosphere"] = [atmosphere_key(c) for c in df[PVK["atm"]]]
    ann = E.AnnealingEncoder().transform(df)
    out["anneal_T"] = ann["anneal__temp_max"].to_numpy()
    out["anneal_logt"] = np.log10(ann["anneal__time_max"].where(ann["anneal__time_max"] > 0)).to_numpy()
    out["pvk_thick"] = _num(df[PVK["thick"]]).to_numpy()
    for layer, (_, _, tc, _, _) in LAYER_COLS.items():
        out[f"{layer}_thick"] = _num(df[tc]).to_numpy()
    out["paper"] = E.get_groups(df).astype(str).to_numpy()
    return out[GENOME_VARS + ["paper"]]


# =========================================================================================
# DESIGN SPACE
# =========================================================================================
@dataclass
class DesignSpace:
    cfg: LHSConfig
    arch: pd.DataFrame
    families: pd.DataFrame
    layers: dict
    layer_probs: dict                  # layer -> {arch -> probability vector aligned with layers[layer]}
    substrates: pd.DataFrame
    substrate_probs: dict              # arch -> vector
    processes: pd.DataFrame
    media: pd.DataFrame
    atmospheres: pd.DataFrame
    continuous: list
    observed_stats: pd.DataFrame
    train_genome: pd.DataFrame
    vocab: dict
    pair_support: dict = field(default_factory=dict)   # (var_a, var_b) -> {(key_a, key_b): n_papers}
    n_train_rows: int = 0
    n_train_papers: int = 0

    def global_range(self, var: str) -> tuple[float, float]:
        """(lo, hi) used to normalise a continuous genome variable in Gower distances."""
        cs = {c.name: c for c in self.continuous}
        m = {"anneal_T": cs["anneal_temperature_C"], "pvk_thick": cs["perovskite_thickness_nm"]}
        if var in m:
            return m[var].lo, m[var].hi
        if var == "anneal_logt":
            c = cs["anneal_time_min"]
            return float(np.log10(c.lo)), float(np.log10(c.hi))
        if var in ("etl_thick", "htl_thick", "bc_thick"):
            return self.cfg.layer_hard_bounds_nm[var[:-6]]
        return (0.0, 1.0)                                  # composition fractions


def _tempered(n: np.ndarray, gamma: float) -> np.ndarray:
    n = np.asarray(n, float)
    w = np.where(n > 0, np.maximum(n, 1e-12) ** gamma, 0.0)
    return w / w.sum() if w.sum() > 0 else w


def _vocab(tr: pd.DataFrame, min_count: int = 30) -> dict:
    """Token vocabularies exactly as PerlaEncoder learns them (same class, same min_count)."""
    v = {}
    for col, prefix, kind in E.MULTILABEL_COLS:
        enc = E.MultiLabelTokenEncoder(col, prefix, kind, min_count).fit(tr)
        v[prefix] = set(enc.vocab_)
    v["arch"] = set(E.MultiLabelTokenEncoder("Cell_architecture", "arch", "single", min_count).fit(tr).vocab_)
    return v


def _raw_tokens(cell):
    return [t for t in E._flatten(cell)]


def _all_known(cell, vocab) -> bool:
    toks = _raw_tokens(cell)
    return bool(toks) and all(t.lower() not in BAD_TOKENS and t in vocab for t in toks)


def _layer_catalog(tr, keys, layer, cfg, vocab):
    sc, pc, tc, vs, vp = LAYER_COLS[layer]
    d = pd.DataFrame({"key": keys[layer].to_numpy(), "arch": keys["arch"].to_numpy(), "paper": keys["paper"].to_numpy(),
                      "thick": keys[f"{layer}_thick"].to_numpy(), "stack": tr[sc].to_numpy(dtype=object),
                      "proc": tr[pc].to_numpy(dtype=object)})
    lo_hard, hi_hard = cfg.layer_hard_bounds_nm[layer]
    d["thick"] = d["thick"].where((d["thick"] >= lo_hard) & (d["thick"] <= hi_hard))
    rows = []
    for key, g in d.groupby("key", sort=False):
        stack, proc = g["stack"].iloc[0], g["proc"].iloc[0]
        empty = not _raw_tokens(stack)
        if empty and not (layer == "htl" and cfg.allow_empty_htl):
            continue
        if not empty and not (_all_known(stack, vocab[vs]) and _all_known(proc, vocab[vp])):
            continue
        if len(stack) != len(proc):
            continue
        pa = g.groupby("arch")["paper"].nunique()
        rows.append(dict(key=key, stack=stack, proc=proc, n_rows=len(g), n_papers=int(g["paper"].nunique()),
                         n_papers_nip=int(pa.get("nip", 0)), n_papers_pin=int(pa.get("pin", 0)),
                         label=" / ".join("+".join(l) for l in stack) + " @ " + " / ".join("+".join(l) for l in proc),
                         thick_vals=g["thick"].dropna().to_numpy()))
    cat = pd.DataFrame(rows)
    keep = (cat["n_papers_nip"] >= cfg.min_papers_category) | (cat["n_papers_pin"] >= cfg.min_papers_category)
    cat = cat[keep].sort_values("n_rows", ascending=False).reset_index(drop=True)
    q_lo, q_hi = cfg.layer_fallback_quantiles
    lo_l, hi_l, n_l, src_l = [], [], [], []
    for v in cat["thick_vals"]:
        if len(v) >= cfg.min_thickness_obs and np.quantile(v, q_hi) - np.quantile(v, q_lo) >= 1.0:
            lo = float(np.clip(np.floor(np.quantile(v, q_lo)), lo_hard, hi_hard))
            hi = float(np.clip(np.ceil(np.quantile(v, q_hi)), lo_hard, hi_hard))
            lo_l.append(lo), hi_l.append(hi), src_l.append("recipe_quantiles")
        else:   # too few observations to know a sensible range -> thickness stays UNSPECIFIED (NaN, like unreported training rows)
            lo_l.append(np.nan), hi_l.append(np.nan), src_l.append("unspecified (too few observations)")
        n_l.append(len(v))
    cat["thick_lo"], cat["thick_hi"], cat["thick_n_obs"], cat["thick_source"] = lo_l, hi_l, n_l, src_l
    cat = cat.drop(columns=["thick_vals"])
    cat.insert(0, "id", np.arange(len(cat)))
    probs = {}
    for a in ("nip", "pin"):
        n = np.where(cat[f"n_papers_{a}"] >= cfg.min_papers_category, cat[f"n_papers_{a}"], 0)
        probs[a] = _tempered(n, cfg.tempering_gamma)
    return cat, probs


def _family_catalog(keys, cfg):
    rows = []
    allowed_a, allowed_b, allowed_c = set(cfg.allowed_a_ions), set(cfg.allowed_b_ions), set(cfg.allowed_c_ions)
    for key, g in keys.groupby("family", sort=False):
        parts = [p.split("+") if p != "?" else [] for p in key.split("|")]
        if len(parts) != 3 or not all(parts) or not (set(parts[0]) <= allowed_a and set(parts[1]) <= allowed_b and set(parts[2]) <= allowed_c):
            continue
        n_p = int(g["paper"].nunique())
        if n_p < cfg.min_papers_family:
            continue
        a, c = parts[0], parts[2]
        a_free = ([i for i in ("Cs",) if i in a and len(a) > 1] + [i for i in ("FA",) if i in a and "MA" in a])
        a_rest = "MA" if ("MA" in a and len(a) > 1) else ("FA" if ("FA" in a and len(a) > 1) else None)
        c_free = "Br" if ("Br" in c and "I" in c) else None
        d_lo, d_hi = cfg.composition_default_fraction_bounds
        rec = dict(key=key, a_ions=a, b_ions=parts[1], c_ions=c, a_free=a_free, a_rest=a_rest, c_free=c_free,
                   inorganic=not (set(a) & ORGANIC_A), leadfree="Pb" not in parts[1], n_rows=len(g), n_papers=n_p)
        for ion, col in (("Cs", "frac_Cs"), ("FA", "frac_FA"), ("Br", "frac_Br")):
            lo, hi, src = d_lo, d_hi, "config_default"
            if ion in a_free or ion == c_free:
                v = g[col].dropna().to_numpy()
                if len(v) >= 30 and np.quantile(v, 0.95) - np.quantile(v, 0.05) >= 0.05:
                    lo, hi, src = float(np.quantile(v, 0.05)), float(np.quantile(v, 0.95)), "family_quantiles_5_95"
                lo, hi = float(np.clip(np.round(lo, 2), 0.02, 0.98)), float(np.clip(np.round(hi, 2), 0.02, 0.98))
            rec.update({f"{ion}_lo": lo, f"{ion}_hi": hi, f"{ion}_source": src})
        rows.append(rec)
    fam = pd.DataFrame(rows).sort_values("n_rows", ascending=False).reset_index(drop=True)
    fam.insert(0, "id", np.arange(len(fam)))
    return fam


def _process_catalog(tr, keys, cfg, vocab):
    allowed = {_j([[t]]) for t in cfg.allowed_perovskite_procedures}
    rows = []
    agg_all = tr[PVK["agg"]].map(_j).to_numpy()
    for key, g in keys.groupby("process", sort=False):
        proc, solv, steps, qb, sa = json.loads(key)
        if _j(proc) not in allowed or steps not in cfg.allowed_n_steps or qb is None or sa is None:
            continue
        if not _all_known(solv, vocab["solvent"]):
            continue
        n_p = int(g["paper"].nunique())
        if n_p < cfg.min_papers_category:
            continue
        agg_mode = pd.Series(agg_all[tr.index.get_indexer(g.index)]).mode().iloc[0]
        agg = json.loads(agg_mode)
        if not _all_known(agg, vocab["agg_state"]):
            continue
        rows.append(dict(key=key, proc=proc, solv=solv, steps=int(steps), qb=bool(qb), sa=bool(sa), agg=agg, n_rows=len(g), n_papers=n_p,
                         label=f"{'+'.join(_raw_tokens(proc))} | {'+'.join(_raw_tokens(solv))}"
                               f" | steps={steps} | antisolvent={bool(qb)} | solvent_anneal={bool(sa)}"))
    cat = pd.DataFrame(rows).sort_values("n_rows", ascending=False).reset_index(drop=True)
    cat.insert(0, "id", np.arange(len(cat)))
    return cat


def _media_catalog(tr, keys, procs, cfg):
    qb_true = {k for k, qb in zip(procs["key"], procs["qb"]) if qb}
    m = keys["process"].isin(qb_true)
    raw = tr.loc[m, PVK["media"]].astype(object)
    d = pd.DataFrame({"k": keys.loc[m, "media"].to_numpy(), "raw": raw.to_numpy(), "paper": keys.loc[m, "paper"].to_numpy()})
    excl = {x.casefold() for x in cfg.excluded_media} | BAD_TOKENS
    rows = []
    for k, g in d.groupby("k"):
        if k in excl or g["paper"].nunique() < cfg.min_papers_category:
            continue
        label = g["raw"].astype(str).str.strip().value_counts().index[0]
        rows.append(dict(key=k, value=label, n_rows=len(g), n_papers=int(g["paper"].nunique())))
    cat = pd.DataFrame(rows).sort_values("n_rows", ascending=False).reset_index(drop=True)
    cat.insert(0, "id", np.arange(len(cat)))
    return cat


def _atmosphere_catalog(keys, procs, vocab, cfg):
    m = keys["process"].isin(set(procs["key"]))
    rows = []
    for k, g in keys[m].groupby("atmosphere"):
        if k in BAD_TOKENS or "+" in k or k not in vocab["atmosphere"] or g["paper"].nunique() < cfg.min_papers_category:
            continue
        rows.append(dict(key=k, token=k, n_rows=len(g), n_papers=int(g["paper"].nunique())))
    cat = pd.DataFrame(rows).sort_values("n_rows", ascending=False).reset_index(drop=True)
    cat.insert(0, "id", np.arange(len(cat)))
    return cat


def _observed_stats(tr, keys, cfg) -> pd.DataFrame:
    ann = E.AnnealingEncoder().transform(tr)
    cols = {
        "anneal_temperature_C (anneal__temp_max)": ann["anneal__temp_max"],
        "anneal_time_min (anneal__time_max)": ann["anneal__time_max"],
        "anneal_time_min (anneal__time_sum)": ann["anneal__time_sum"],
        "perovskite_thickness_nm (Perovskite_thickness__sum)": pd.to_numeric(tr["Perovskite_thickness__sum"], errors="coerce"),
        "etl_thickness_nm (ETL_thickness__sum)": pd.to_numeric(tr["ETL_thickness__sum"], errors="coerce"),
        "htl_thickness_nm (HTL_thickness__sum)": pd.to_numeric(tr["HTL_thickness__sum"], errors="coerce"),
        "bc_thickness_nm (Backcontact_thickness_list__sum)": pd.to_numeric(tr["Backcontact_thickness_list__sum"], errors="coerce"),
        "cell_area_cm2 (Cell_area_measured__parsed)": pd.to_numeric(tr["Cell_area_measured__parsed"], errors="coerce"),
        "band_gap_eV (band_gap__mean)": pd.to_numeric(tr["band_gap__mean"], errors="coerce"),
        "n_deposition_steps": pd.to_numeric(tr[PVK["steps"]], errors="coerce").mask(lambda s: s == 0),
        "A_frac_Cs": keys["frac_Cs"], "A_frac_FA": keys["frac_FA"], "C_frac_Br": keys["frac_Br"],
    }
    rows = []
    for name, s in cols.items():
        s = pd.Series(s).astype("float64")
        v = s.dropna()
        top = float(v.value_counts(normalize=True).iloc[0]) if len(v) else np.nan
        q = v.quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]) if len(v) else pd.Series([np.nan] * 7, index=[.01, .05, .25, .5, .75, .95, .99])
        rows.append(dict(variable=name, n_observed=int(len(v)), missing_rate=round(float(s.isna().mean()), 4), n_unique=int(v.nunique()),
                         top_value_share=round(top, 4), min=v.min() if len(v) else np.nan, q01=q[.01], q05=q[.05], q25=q[.25], median=q[.5],
                         q75=q[.75], q95=q[.95], q99=q[.99], max=v.max() if len(v) else np.nan,
                         genuinely_continuous=bool(len(v) and v.nunique() >= 20 and top < 0.8)))
    return pd.DataFrame(rows)


def _pair_support(keys: pd.DataFrame) -> dict:
    pairs = [("etl", "htl"), ("htl", "bc"), ("family", "process"), ("family", "htl"), ("arch", "process"), ("etl", "family")]
    out = {}
    for a, b in pairs:
        g = keys.groupby([a, b])["paper"].nunique()
        out[(a, b)] = {k: int(v) for k, v in g.items()}
    return out


def build_design_space(train_df: pd.DataFrame, cfg: LHSConfig) -> DesignSpace:
    """`train_df` must be the 70 % TRAIN split (the caller asserts this)."""
    tr = train_df
    keys = genome_keys(tr)
    vocab = _vocab(tr)
    arch_papers = keys.groupby("arch")["paper"].nunique()
    arch = pd.DataFrame({"value": [a for a in ("nip", "pin") if a in vocab["arch"]]})
    arch["n_rows"] = [int((keys["arch"] == a).sum()) for a in arch["value"]]
    arch["n_papers"] = [int(arch_papers.get(a, 0)) for a in arch["value"]]
    arch["p"] = _tempered(arch["n_papers"].to_numpy(), cfg.tempering_gamma)
    layers, layer_probs = {}, {}
    for layer in LAYER_COLS:
        layers[layer], layer_probs[layer] = _layer_catalog(tr, keys, layer, cfg, vocab)
    fam = _family_catalog(keys, cfg)
    fam["p"] = _tempered(fam["n_papers"].to_numpy(), cfg.tempering_gamma)
    proc = _process_catalog(tr, keys, cfg, vocab)
    proc["p"] = _tempered(proc["n_papers"].to_numpy(), cfg.tempering_gamma)
    media = _media_catalog(tr, keys, proc, cfg)
    media["p"] = _tempered(media["n_papers"].to_numpy(), cfg.tempering_gamma)
    atm = _atmosphere_catalog(keys, proc, vocab, cfg)
    atm["p"] = _tempered(atm["n_papers"].to_numpy(), cfg.tempering_gamma)
    sub_rows = []
    for s in cfg.allowed_substrates:
        m = keys["substrate"] == s
        sub_rows.append(dict(value=s, **{f"n_papers_{a}": int(keys.loc[m & (keys["arch"] == a), "paper"].nunique()) for a in ("nip", "pin")},
                             n_rows=int(m.sum())))
    subs = pd.DataFrame(sub_rows)
    subs.insert(0, "id", np.arange(len(subs)))
    sub_probs = {a: _tempered(np.where(subs[f"n_papers_{a}"] >= cfg.min_papers_category, subs[f"n_papers_{a}"], 0), cfg.tempering_gamma)
                 for a in ("nip", "pin")}
    for name, cat in (("families", fam), ("processes", proc), ("media", media), ("atmospheres", atm)):
        if len(cat) == 0:
            raise ValueError(f"design space: the catalogue '{name}' is empty with the current thresholds / filters")
    for layer, cat in layers.items():
        if len(cat) == 0:
            raise ValueError(f"design space: the catalogue of layer '{layer}' is empty")
    return DesignSpace(cfg=cfg, arch=arch, families=fam, layers=layers, layer_probs=layer_probs, substrates=subs, substrate_probs=sub_probs,
                       processes=proc, media=media, atmospheres=atm, continuous=cfg.continuous_specs(),
                       observed_stats=_observed_stats(tr, keys, cfg), train_genome=keys, vocab=vocab,
                       pair_support=_pair_support(keys), n_train_rows=len(tr), n_train_papers=int(keys["paper"].nunique()))


# =========================================================================================
# TABLES FOR THE OUTPUT FILES
# =========================================================================================
def _cat_summary(df: pd.DataFrame, label_col: str, extra: dict | None = None) -> list[dict]:
    out = []
    for _, r in df.iterrows():
        d = {"id": int(r["id"]), "label": r[label_col], "n_rows": int(r["n_rows"]), "n_papers": int(r.get("n_papers", -1))}
        if "p" in r:
            d["probability"] = round(float(r["p"]), 6)
        out.append(d)
    return out


def categories_dict(space: DesignSpace) -> dict:
    """JSON-serialisable category mappings (ids -> labels, counts, probabilities) for lhs_sampling_config.json."""
    d = {"architecture": space.arch.to_dict("records"),
         "composition_family": _cat_summary(space.families, "key"),
         "substrate": space.substrates.to_dict("records"),
         "perovskite_process": _cat_summary(space.processes, "label"),
         "antisolvent_medium": _cat_summary(space.media, "value"),
         "atmosphere": _cat_summary(space.atmospheres, "token")}
    for layer, cat in space.layers.items():
        recs = _cat_summary(cat, "label")
        for rec, (_, r) in zip(recs, cat.iterrows()):
            rec.update(thickness_lo_nm=r["thick_lo"], thickness_hi_nm=r["thick_hi"], thickness_source=r["thick_source"],
                       p_nip=round(float(space.layer_probs[layer]["nip"][int(r["id"])]), 6),
                       p_pin=round(float(space.layer_probs[layer]["pin"][int(r["id"])]), 6))
        d[f"{layer}_recipe"] = recs
    return d


def bounds_table(space: DesignSpace) -> pd.DataFrame:
    """lhs_feature_bounds.csv: one row per design variable (bounds, units, source, rationale + descriptive observed stats)."""
    cfg = space.cfg
    obs = space.observed_stats.set_index("variable")

    def o(prefix):
        r = obs[obs.index.str.startswith(prefix)]
        return r.iloc[0].to_dict() if len(r) else {}

    rows = []
    for c in space.continuous:
        base = {"anneal_temperature_C": "anneal_temperature_C", "anneal_time_min": "anneal_time_min (anneal__time_max)",
                "perovskite_thickness_nm": "perovskite_thickness_nm"}[c.name]
        rows.append(dict(variable=c.name, role="design (continuous)", unit=c.unit, lower=c.lo, upper=c.hi, scale=c.scale, round_to=c.round_to,
                         n_categories=np.nan, source="config proposal (lhs_config.py); observed quantiles are descriptive only",
                         rationale=c.rationale, **{f"observed_{k}": v for k, v in o(base).items() if k != "variable"}))
    for layer, nm in (("etl", "etl_thickness_nm"), ("htl", "htl_thickness_nm"), ("bc", "bc_thickness_nm")):
        cat = space.layers[layer]
        lo_h, hi_h = cfg.layer_hard_bounds_nm[layer]
        rows.append(dict(variable=nm, role="design (continuous, recipe-conditional)", unit="nm", lower=float(np.nanmin(cat["thick_lo"])),
                         upper=float(np.nanmax(cat["thick_hi"])), scale="linear", round_to=cfg.round_to[nm], n_categories=len(cat),
                         source=f"per-recipe {int(cfg.layer_fallback_quantiles[0]*100)}-{int(cfg.layer_fallback_quantiles[1]*100)} % quantiles of TRAIN thickness "
                                f"(>= {cfg.min_thickness_obs} values) else thickness left UNSPECIFIED (missing); clipped to hard bounds [{lo_h}, {hi_h}] nm",
                         rationale="A compact oxide and a mesoporous scaffold have very different sensible thicknesses, so a single global range "
                                   "would force unsupported extrapolation. lower/upper here = min/max over recipes; see lhs_sampling_config.json for each recipe.",
                         **{f"observed_{k}": v for k, v in o(nm.replace("_thickness_nm", "_thickness")).items() if k != "variable"}))
    for ion, nm in (("Cs", "A_frac_Cs"), ("FA", "A_frac_FA"), ("Br", "C_frac_Br")):
        f = space.families
        m = f[f"{ion}_source"] != "config_default" if f"{ion}_source" in f else None
        act = f[(f["a_free"].map(lambda x, i=ion: i in x)) | (f["c_free"] == ion)]
        rows.append(dict(variable=nm, role="design (continuous, family-conditional)", unit="fraction on its site", lower=float(act[f"{ion}_lo"].min()),
                         upper=float(act[f"{ion}_hi"].max()), scale="linear", round_to=cfg.round_to["fraction"], n_categories=len(act),
                         source="per-family 5-95 % quantiles of the TRAIN fraction (>= 30 rows) else config default; remainder ion = 1 - sum(free ions)",
                         rationale="Mixing fractions are genuinely continuous but only meaningful inside a composition family; the remainder "
                                   "(MA or FA) is computed so that A-site coefficients sum to 1 and C-site to 3.",
                         **{f"observed_{k}": v for k, v in o(nm).items() if k != "variable"}))
    cat_rows = [("cell_architecture", len(space.arch), "TRAIN papers per level ** gamma"),
                ("composition_family", len(space.families), f"ion-set families with >= {cfg.min_papers_family} papers, A in {cfg.allowed_a_ions}, B in {cfg.allowed_b_ions}, C in {cfg.allowed_c_ions}"),
                ("etl_recipe", len(space.layers["etl"]), "observed (stack, procedure) pairs; conditional on architecture"),
                ("htl_recipe", len(space.layers["htl"]), "observed (stack, procedure) pairs; conditional on architecture"),
                ("bc_recipe", len(space.layers["bc"]), "observed (stack, procedure) pairs; conditional on architecture"),
                ("substrate", len(space.substrates), f"config list {cfg.allowed_substrates}; conditional on architecture"),
                ("perovskite_process_recipe", len(space.processes), f"observed (procedure, solvents, steps, antisolvent flag, solvent-anneal flag); procedures {cfg.allowed_perovskite_procedures}, steps {cfg.allowed_n_steps}"),
                ("antisolvent_medium", len(space.media), "only used when the process recipe has antisolvent=True; otherwise 'Unknown' as in the training data"),
                ("synthesis_atmosphere", len(space.atmospheres), "single-token atmospheres of the eligible processes")]
    for nm, n, src in cat_rows:
        rows.append(dict(variable=nm, role="design (categorical)", unit="", lower=np.nan, upper=np.nan, scale="categorical", round_to=np.nan,
                         n_categories=n, source=src, rationale=f"levels need >= {cfg.min_papers_category} distinct papers; p ~ papers**{cfg.tempering_gamma}"))
    fixed = [("cell_area_cm2", cfg.cell_area_cm2, "cm2", "training median; not a recipe choice, fixed so candidates are comparable"),
             ("cell_flexible", cfg.cell_flexible, "", "device-type context, fixed"), ("cell_semitransparent", cfg.cell_semitransparent, "", "device-type context, fixed"),
             ("perovskite_single_crystal", cfg.perovskite_single_crystal, "", "fixed; not a thin-film recipe"),
             ("perovskite_band_gap_graded", cfg.perovskite_band_gap_graded, "", "fixed")]
    for nm, v, u, why in fixed:
        rows.append(dict(variable=nm, role="fixed context", unit=u, lower=v, upper=v, scale="fixed", round_to=np.nan, n_categories=1, source="lhs_config.py", rationale=why))
    for nm, why in (("composition_inorganic", "derived from the A-site ions (no organic cation)"), ("composition_leadfree", "derived from the B-site ions (no Pb)"),
                    ("number_of_deposition_steps", "taken from the process recipe"), ("aggregation_state_of_reactants", "modal training value for the process recipe"),
                    ("quenching_induced_crystallisation / solvent_annealing", "taken from the process recipe (never sampled independently)")):
        rows.append(dict(variable=nm, role="derived", unit="", scale="derived", source="lhs_design_space.py", rationale=why))
    for nm, why in (("band_gap__mean", "a property of the film, not a fabrication choice; left missing -> existing KNN imputer, num__band_gap__mean__missing = 1"),
                    ("Perovskite_band_gap_estimation_basis", "measurement metadata; left missing -> bg_basis__missing = 1")):
        rows.append(dict(variable=nm, role="neutralised (missing)", unit="", scale="missing", source="lhs_design_space.py", rationale=why,
                         **{f"observed_{k}": v for k, v in o(nm.split("__")[0] if nm.startswith("band") else "zzz").items() if k != "variable"}))
    return pd.DataFrame(rows)


def schema_audit(space: DesignSpace) -> pd.DataFrame:
    """One row per RAW model input column (38) and how a candidate fills it. Also lists every forbidden column."""
    import lhs_config as C
    needed = ({c for c, *_ in E.MULTILABEL_COLS} | {c for c, _ in E.ONEHOT_COLS} | set(E.BOOLEAN_COLS) | set(E.NUMERIC_COLS)
              | {c for c, _ in E.TARGET_ENCODED_COLS} | {E.ANNEAL_TEMP_COL, E.ANNEAL_TIME_COL, E.GROUP_COL}
              | {c for pair in E.COMP_COLS.values() for c in pair})
    role = {}
    for c in needed:
        role[c] = ("sampled/derived from genome", "")
    for c in ("band_gap__mean", "Perovskite_band_gap_estimation_basis"):
        role[c] = ("neutralised: left missing -> KNN-imputed like training rows without it", "")
    for c in ("Cell_area_measured__parsed", "Cell_flexible", "Cell_semitransparent", "Perovskite_single_crystal", "Perovskite_band_gap_graded"):
        role[c] = ("fixed context (lhs_config.py)", "")
    for c in ("Perovskite_composition_inorganic", "Perovskite_composition_leadfree"):
        role[c] = ("derived from composition", "")
    role[E.GROUP_COL] = ("identifier placeholder = None (required by PerlaEncoder._check_columns, never used in transform)", "")
    rows = [dict(raw_column=c, treatment=role[c][0], model_input=True) for c in sorted(needed)]
    for c in C.FORBIDDEN_RAW_COLUMNS:
        rows.append(dict(raw_column=c, treatment="EXCLUDED: target / measured output / identifier / metadata", model_input=False))
    return pd.DataFrame(rows)
