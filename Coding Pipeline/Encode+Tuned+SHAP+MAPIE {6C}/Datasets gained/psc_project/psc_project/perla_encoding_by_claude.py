"""
perla_encoding.py
=================
Turns the *cleaned + parsed* PERLA dataframe (the output of your notebook) into a numeric
feature matrix.  Nothing here changes the cleaning; it only reads the columns your parsers
produced.

Encoding strategy (by column type)
----------------------------------
1. Composition (A/B/C ions + coefficients)  -> site fractions + Magpie descriptors
2. Booleans (11 columns)                     -> three-state: __true / __unknown  (False = both 0)
3. Low-cardinality categoricals              -> one-hot (+ __other, __missing)
4. High-cardinality multi-valued fields      -> multi-label binarization (+ __other, __missing)
   (ETL / HTL / back-contact / substrate stacks, deposition procedures, solvents, atmosphere,
    aggregation state, HTL additives, band-gap basis)
5. Medium-cardinality single-valued field    -> out-of-fold, group-aware target encoding
   (quenching media)
6. Numeric                                   -> pass-through (NaN kept for your Tier-3 imputer),
                                                + annealing temperature/time summaries

Leakage discipline
------------------
Every encoder is an sklearn transformer: vocabularies / category statistics are learned in
`fit` (training fold only).  The target encoder uses GroupKFold on the paper DOI *inside*
`fit_transform`, so a training row never sees its own paper's target.

Expected input columns
----------------------
    clean_df = copy_df  +  parse_perovskite_columns(copy_df)  +  parse_more_columns(copy_df)
(see `assemble_clean_frame` below).
"""
from __future__ import annotations

import re
import warnings
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import GroupKFold

# =========================================================================================
# CONFIGURATION  (edit here, not in the code below)
# =========================================================================================
GROUP_COL = "Ref_DOI_number"                      # paper id -> grouped CV / grouped target encoding

TARGET_COLS = {                                   # stage-1 targets (Voc, Jsc, FF) and stage-2 target (PCE)
    "PCE": "JV_default_PCE__parsed",
    "Voc": "Voc__parsed",
    "Jsc": "Jsc__parsed",
    "FF": "FF__parsed",
}

BOOLEAN_COLS = [
    "Perovskite_composition_inorganic",
    "Perovskite_composition_leadfree",
    "Perovskite_single_crystal",
    "Perovskite_band_gap_graded",
    "Perovskite_deposition_solvent_annealing",
    "Perovskite_deposition_quenching_induced_crystallisation",
    "Cell_flexible",
    "Cell_semitransparent",
]

# ---- single-valued low-cardinality categoricals: (column, prefix)
ONEHOT_COLS = [("Cell_architecture", "arch")]

# ---- multi-valued fields -> multi-label binarization: (column, prefix, kind)
#      kind = "nested"  : column already holds a parsed nested list (your __parsed columns)
#      kind = "string"  : column is a raw string with '|', ';' or '>>' separators
MULTILABEL_COLS = [
    ("ETL_stack_sequence__parsed",                              "etl_stack",   "nested"),
    ("HTL_stack_sequence__parsed",                              "htl_stack",   "nested"),
    ("Backcontact_stack_sequence__parsed",                      "bc_stack",    "nested"),
    ("Substrate_stack_sequence",                                "substrate",   "string"),
    ("Perovskite_deposition_procedure__parsed",                 "pvk_proc",    "nested"),
    ("ETL_deposition_procedure__parsed",                        "etl_proc",    "nested"),
    ("HTL_deposition_procedure__parsed",                        "htl_proc",    "nested"),
    ("Backcontact_deposition_procedure__parsed",                "bc_proc",     "nested"),
    ("Perovskite_deposition_solvents__parsed_clean",            "solvent",     "nested"),
    ("Perovskite_deposition_synthesis_atmosphere_parsed",       "atmosphere",  "nested"),
    ("Perovskite_deposition_aggregation_state_of_reactants__parsed", "agg_state", "nested"),
    ("Perovskite_band_gap_estimation_basis",                    "bg_basis",    "string"),
]

# ---- medium-cardinality single-valued field -> out-of-fold target encoding: (column, prefix)
TARGET_ENCODED_COLS = [("Perovskite_deposition_quenching_media", "quench")]

# ---- numeric pass-through
NUMERIC_COLS = [
    "band_gap__mean",
    "Perovskite_thickness__sum",
    "ETL_thickness__sum",
    "HTL_thickness__sum",
    "Backcontact_thickness_list__sum",
    "Perovskite_deposition_number_of_deposition_steps__parsed",
    "Cell_area_measured__parsed",
]
# database default "0" means "not stated" (your parser flags it as zero_default_placeholder)
ZERO_MEANS_MISSING = {"Perovskite_deposition_number_of_deposition_steps__parsed"}

ANNEAL_TEMP_COL = "Perovskite_deposition_thermal_annealing_temperature__parsed"
ANNEAL_TIME_COL = "Perovskite_deposition_thermal_annealing_time__parsed"

# Optional name harmonisation for stack / procedure tokens.  Left EMPTY on purpose: merging
# names (e.g. Spiro-MeOTAD vs Spiro-OMeTAD, NiOx vs NiO) is part of your open data-quality
# backlog and should be decided explicitly, not inside the encoder.
TOKEN_ALIASES: dict[str, str] = {}

# Deliberately NOT encoded (see notes in the reply):
#   Cell_stack_sequence (+__parsed)      redundant superset of substrate+ETL+HTL+back-contact
#   Cell_number_of_cells_per_substrate   ~99% placeholder zeros
#   *_mixing_ratios__parsed              needs an explicit decision (alignment with solvents)
#   Ref_DOI_number / Ref_publication_year / Source   meta-data, not features
#   targets, *__flags, *_clean / *__original duplicates

# =========================================================================================
# COMPOSITION KNOWLEDGE
# =========================================================================================
# Molecular ions have no Magpie entry, so they are expanded to atoms.  Only unambiguous,
# standard abbreviations are listed.  ANY ion not resolvable (here or as an element symbol)
# makes that site's descriptors NaN instead of being guessed.  Extend / verify as needed.
SPECIES_FORMULAS = {
    # A-site organic cations
    "MA": {"C": 1, "H": 6, "N": 1},            # methylammonium
    "FA": {"C": 1, "H": 5, "N": 2},            # formamidinium
    "GU": {"C": 1, "H": 6, "N": 3},            # guanidinium
    "NH4": {"N": 1, "H": 4},
    "EA": {"C": 2, "H": 8, "N": 1},            # ethylammonium
    "DMA": {"C": 2, "H": 8, "N": 1},           # dimethylammonium
    "PA": {"C": 3, "H": 10, "N": 1},           # propylammonium
    "TMA": {"C": 3, "H": 10, "N": 1},          # trimethylammonium
    "BA": {"C": 4, "H": 12, "N": 1},           # butylammonium
    "PEA": {"C": 8, "H": 12, "N": 1},          # phenethylammonium
    "5-AVA": {"C": 5, "H": 12, "N": 1, "O": 2},  # 5-ammoniumvaleric acid
    "EDA": {"C": 2, "H": 10, "N": 2},          # ethylenediammonium
    "ThMA": {"C": 5, "H": 8, "N": 1, "S": 1},  # thiophene-methylammonium
    "TBA": {"C": 16, "H": 36, "N": 1},         # tetrabutylammonium
    # C-site molecular anions
    "SCN": {"S": 1, "C": 1, "N": 1},
    "BF4": {"B": 1, "F": 4},
    "PF6": {"P": 1, "F": 6},
}
VALID_C_SITE = {"I", "Br", "Cl", "F", "O", "S", "Se", "Te", "N", "SCN", "BF4", "PF6"}
SITE_MULTIPLICITY = {"A": 1.0, "B": 1.0, "C": 3.0}          # ABC3
MAGPIE_PROPS = ["AtomicWeight", "Electronegativity", "CovalentRadius", "MendeleevNumber",
                "NValence", "Row", "Column", "MeltingT"]
COMP_COLS = {
    "A": ("Perovskite_composition_a_ions__parsed", "Perovskite_composition_a_ions_coefficients__parsed"),
    "B": ("Perovskite_composition_b_ions__parsed", "Perovskite_composition_b_ions_coefficients__parsed"),
    "C": ("Perovskite_composition_c_ions__parsed", "Perovskite_composition_c_ions_coefficients__parsed"),
}

MISSING_TOKENS = {"", "unknown", "<unknown>", "nan", "none", "null", "n/a", "na"}
_SPLIT = re.compile(r"\s*(?:\||;|>>)\s*")


# =========================================================================================
# GENERIC HELPERS
# =========================================================================================
def _is_missing(x) -> bool:
    if x is None:
        return True
    if isinstance(x, (list, tuple, np.ndarray)):
        return False
    try:
        return bool(pd.isna(x))
    except (TypeError, ValueError):
        return False


def _flatten(x):
    if isinstance(x, (list, tuple, np.ndarray)):
        for y in x:
            yield from _flatten(y)
    elif not _is_missing(x):
        yield str(x).strip()


def tokens_nested(cell) -> set:
    """parsed nested list -> set of tokens ('Unknown'/None dropped)."""
    return {t for t in _flatten(cell) if t.lower() not in MISSING_TOKENS}


def tokens_string(cell) -> set:
    """raw 'a; b | c >> d' string -> set of tokens."""
    if _is_missing(cell):
        return set()
    return {t for t in _SPLIT.split(str(cell).strip()) if t and t.lower() not in MISSING_TOKENS}


def tokens_single(cell) -> set:
    if _is_missing(cell):
        return set()
    t = str(cell).strip()
    return set() if t.lower() in MISSING_TOKENS else {t}


_TOKENIZERS = {"nested": tokens_nested, "string": tokens_string, "single": tokens_single}


# =========================================================================================
# 1. MULTI-LABEL / ONE-HOT
# =========================================================================================
class MultiLabelTokenEncoder(BaseEstimator, TransformerMixin):
    """Binary column per token seen in >= min_count TRAINING rows.
    + `<prefix>__other`   : row has at least one rarer / unseen token
    + `<prefix>__missing` : row has no usable token at all
    With kind='single' this is an ordinary one-hot encoder."""

    def __init__(self, column, prefix, kind="nested", min_count=30, aliases=None):
        self.column = column
        self.prefix = prefix
        self.kind = kind
        self.min_count = min_count
        self.aliases = aliases

    def _tok(self, cell):
        toks = _TOKENIZERS[self.kind](cell)
        if self.aliases:
            toks = {self.aliases.get(t, t) for t in toks}
        return toks

    def fit(self, X, y=None):
        cnt = Counter()
        for cell in X[self.column]:
            cnt.update(self._tok(cell))
        self.vocab_ = sorted((t for t, n in cnt.items() if n >= self.min_count),
                             key=lambda t: (-cnt[t], t))
        self.feature_names_ = ([f"{self.prefix}__{t}" for t in self.vocab_]
                               + [f"{self.prefix}__other", f"{self.prefix}__missing"])
        return self

    def transform(self, X):
        idx = {t: i for i, t in enumerate(self.vocab_)}
        out = np.zeros((len(X), len(self.vocab_) + 2), dtype=np.float32)
        for r, cell in enumerate(X[self.column]):
            toks = self._tok(cell)
            if not toks:
                out[r, -1] = 1.0
                continue
            for t in toks:
                j = idx.get(t)
                out[r, -2 if j is None else j] = 1.0
        return pd.DataFrame(out, columns=self.feature_names_, index=X.index)

    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names_)


# =========================================================================================
# 2. THREE-STATE BOOLEANS
# =========================================================================================
class ThreeStateBoolean(BaseEstimator, TransformerMixin):
    """True / False / Unknown  ->  two indicators:
         <col>__true     1 if True
         <col>__unknown  1 if missing (not reported)
       False = both 0.  'Not done' and 'not reported' stay distinguishable."""

    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        out = {}
        for c in self.columns:
            s = X[c].astype("boolean")
            out[f"{c}__true"] = s.eq(True).fillna(False).to_numpy(dtype=np.float32)
            out[f"{c}__unknown"] = s.isna().to_numpy(dtype=np.float32)
        return pd.DataFrame(out, index=X.index)

    def get_feature_names_out(self, input_features=None):
        return np.array([f"{c}__{s}" for c in self.columns for s in ("true", "unknown")])


# =========================================================================================
# 3. NUMERIC BLOCKS
# =========================================================================================
class NumericBlock(BaseEstimator, TransformerMixin):
    """Pass-through numeric columns.  NaN is kept (your Tier-3 SafeImputer fills it).
    A `__missing` indicator is added for columns missing in more than `indicator_threshold`
    of the TRAINING rows."""

    def __init__(self, columns, zero_means_missing=(), indicator_threshold=0.05, add_indicators=True):
        self.columns = columns
        self.zero_means_missing = zero_means_missing
        self.indicator_threshold = indicator_threshold
        self.add_indicators = add_indicators

    def _num(self, X, c):
        s = pd.to_numeric(X[c], errors="coerce").astype("float64")
        if c in self.zero_means_missing:
            s = s.mask(s == 0)
        return s

    def fit(self, X, y=None):
        self.indicator_cols_ = []
        if self.add_indicators:
            for c in self.columns:
                if self._num(X, c).isna().mean() > self.indicator_threshold:
                    self.indicator_cols_.append(c)
        return self

    def transform(self, X):
        out = {f"num__{c}": self._num(X, c) for c in self.columns}
        for c in self.indicator_cols_:
            out[f"num__{c}__missing"] = self._num(X, c).isna().astype("float32")
        return pd.DataFrame(out, index=X.index)

    def get_feature_names_out(self, input_features=None):
        return np.array([f"num__{c}" for c in self.columns]
                        + [f"num__{c}__missing" for c in getattr(self, "indicator_cols_", [])])


def _flat_numeric(x):
    if isinstance(x, (list, tuple, np.ndarray)):
        for y in x:
            yield from _flat_numeric(y)
    else:
        yield None if _is_missing(x) else float(x)


class AnnealingEncoder(BaseEstimator, TransformerMixin):
    """Nested [layer][step][value] annealing lists -> scalar summaries.
       temperature : first / last / max  (over KNOWN values, in layer->step->value order)
       time        : sum / max           (sum is NaN if ANY value in the cell is unknown,
                                          same rule as your thickness sums)
       No value is estimated; cells without any known value stay NaN (+ __missing flag)."""

    def __init__(self, temp_col=ANNEAL_TEMP_COL, time_col=ANNEAL_TIME_COL):
        self.temp_col = temp_col
        self.time_col = time_col

    def fit(self, X, y=None):
        return self

    @staticmethod
    def _summ(cell):
        vals = list(_flat_numeric(cell)) if cell is not None else []
        known = [v for v in vals if v is not None]
        return vals, known

    def transform(self, X):
        n = len(X)
        cols = {k: np.full(n, np.nan) for k in
                ("temp_first", "temp_last", "temp_max", "time_sum", "time_max")}
        for r, (tc, mc) in enumerate(zip(X[self.temp_col], X[self.time_col])):
            _, tk = self._summ(tc)
            if tk:
                cols["temp_first"][r], cols["temp_last"][r], cols["temp_max"][r] = tk[0], tk[-1], max(tk)
            allv, mk = self._summ(mc)
            if mk:
                cols["time_max"][r] = max(mk)
                if len(mk) == len(allv):
                    cols["time_sum"][r] = sum(mk)
        out = pd.DataFrame({f"anneal__{k}": v for k, v in cols.items()}, index=X.index)
        out["anneal__temp_missing"] = out["anneal__temp_first"].isna().astype("float32")
        out["anneal__time_missing"] = out["anneal__time_max"].isna().astype("float32")
        return out

    def get_feature_names_out(self, input_features=None):
        return np.array([f"anneal__{k}" for k in ("temp_first", "temp_last", "temp_max", "time_sum",
                                                  "time_max", "temp_missing", "time_missing")])


# =========================================================================================
# 4. COMPOSITION  (site fractions + Magpie descriptors)
# =========================================================================================
class PerovskiteCompositionEncoder(BaseEstimator, TransformerMixin):
    """From the parsed ion / coefficient lists:

       * `A_frac__MA`, `A_frac__other`, `B_frac__Pb`, `C_frac__I` ...   (ions seen in >= min_count
         training rows; fractions are normalised within each site; multi-layer entries are averaged
         with equal layer weight)
       * Magpie descriptors: composition-weighted mean per site (A_, B_, C_) and mean / range over
         the whole ABC3 formula, for 8 elemental properties.  Molecular ions are expanded to atoms
         with SPECIES_FORMULAS.
       * `comp__valid`: 1 if all three sites could be resolved.

       A site whose ions or coefficients are unknown / symbolic / unresolvable gets NaN for ALL its
       columns - no stoichiometry is guessed."""

    def __init__(self, min_count=100):
        self.min_count = min_count

    # ---------- element data (lazy so the class can be pickled / cloned) ----------
    def _magpie(self):
        if not hasattr(self, "_mp"):
            from matminer.utils.data import MagpieData
            from pymatgen.core import Element
            self._mp, self._El, self._elcache = MagpieData(), Element, {}
        return self._mp

    def _element_vec(self, sym):
        self._magpie()
        if sym not in self._elcache:
            try:
                el = self._El(sym)
                v = np.array([float(self._mp.get_elemental_property(el, p)) for p in MAGPIE_PROPS])
            except Exception:
                v = None
            self._elcache[sym] = v
        return self._elcache[sym]

    def _atoms(self, ion):
        """ion label -> {element: count} or None if unresolvable."""
        if ion in SPECIES_FORMULAS:
            return SPECIES_FORMULAS[ion]
        if re.fullmatch(r"[A-Z][a-z]?", ion) and self._element_vec(ion) is not None:
            return {ion: 1}
        return None

    def _species_vec(self, ion):
        atoms = self._atoms(ion)
        if atoms is None:
            return None
        tot = sum(atoms.values())
        return sum(self._element_vec(e) * n for e, n in atoms.items()) / tot

    # ---------- per-site fractions ----------
    def _site(self, site, ions_layers, coef_layers):
        if not ions_layers or not coef_layers or len(ions_layers) != len(coef_layers):
            return None
        acc, n_layers = Counter(), len(ions_layers)
        for ions, coefs in zip(ions_layers, coef_layers):
            if not ions or len(ions) != len(coefs):
                return None
            if any(i is None or _is_missing(i) or str(i).lower() in MISSING_TOKENS for i in ions):
                return None
            if any(c is None or not np.isfinite(c) or c < 0 for c in coefs):
                return None
            tot = float(sum(coefs))
            if tot <= 0:
                return None
            for i, c in zip(ions, coefs):
                acc[str(i)] += c / tot / n_layers
        if site == "C" and any(i not in VALID_C_SITE for i in acc):
            return None
        if any(self._atoms(i) is None for i in acc):
            return None
        return dict(acc)

    def _row(self, row_key, site_inputs):
        """-> ({site: fractions|None}, descriptor vector|None)   (cached)"""
        if row_key in self._cache:
            return self._cache[row_key]
        sites = {s: self._site(s, *site_inputs[s]) for s in "ABC"}
        desc = {}
        for s, fr in sites.items():
            desc[s] = None if fr is None else sum(f * self._species_vec(i) for i, f in fr.items())
        whole = None
        if all(fr is not None for fr in sites.values()):
            counts = Counter()
            for s, fr in sites.items():
                for ion, f in fr.items():
                    for el, n in self._atoms(ion).items():
                        counts[el] += SITE_MULTIPLICITY[s] * f * n
            els = list(counts)
            vecs = np.array([self._element_vec(e) for e in els])
            w = np.array([counts[e] for e in els]) / sum(counts.values())
            whole = (w @ vecs, vecs.max(axis=0) - vecs.min(axis=0))
        res = (sites, desc, whole)
        self._cache[row_key] = res
        return res

    def _iter_rows(self, X):
        cols = {s: (X[a].to_numpy(dtype=object), X[b].to_numpy(dtype=object)) for s, (a, b) in COMP_COLS.items()}
        for r in range(len(X)):
            inp = {s: (cols[s][0][r], cols[s][1][r]) for s in "ABC"}
            key = repr(inp)
            yield self._row(key, inp)

    def fit(self, X, y=None):
        self._magpie()
        self._cache = {}
        cnt = {s: Counter() for s in "ABC"}
        for sites, _, _ in self._iter_rows(X):
            for s, fr in sites.items():
                if fr:
                    cnt[s].update(fr.keys())
        self.vocab_ = {s: sorted((i for i, n in cnt[s].items() if n >= self.min_count),
                                 key=lambda i: (-cnt[s][i], i)) for s in "ABC"}
        names = []
        for s in "ABC":
            names += [f"{s}_frac__{i}" for i in self.vocab_[s]] + [f"{s}_frac__other"]
        for s in "ABC":
            names += [f"{s}_magpie__{p}" for p in MAGPIE_PROPS]
        names += [f"formula_magpie_mean__{p}" for p in MAGPIE_PROPS]
        names += [f"formula_magpie_range__{p}" for p in MAGPIE_PROPS]
        names += ["comp__valid"]
        self.feature_names_ = names
        return self

    def transform(self, X):
        if not hasattr(self, "_cache"):
            self._cache = {}
        self._magpie()
        nP = len(MAGPIE_PROPS)
        out = np.full((len(X), len(self.feature_names_)), np.nan, dtype=np.float64)
        for r, (sites, desc, whole) in enumerate(self._iter_rows(X)):
            c = 0
            for s in "ABC":
                v = self.vocab_[s]
                fr = sites[s]
                if fr is not None:
                    for k, ion in enumerate(v):
                        out[r, c + k] = fr.get(ion, 0.0)
                    out[r, c + len(v)] = sum(f for i, f in fr.items() if i not in v)
                c += len(v) + 1
            for s in "ABC":
                if desc[s] is not None:
                    out[r, c:c + nP] = desc[s]
                c += nP
            if whole is not None:
                out[r, c:c + nP] = whole[0]
                out[r, c + nP:c + 2 * nP] = whole[1]
            out[r, -1] = 1.0 if whole is not None else 0.0
        return pd.DataFrame(out, columns=self.feature_names_, index=X.index)

    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names_)


# =========================================================================================
# 5. OUT-OF-FOLD (GROUPED) TARGET ENCODING
# =========================================================================================
class OOFTargetEncoder(BaseEstimator, TransformerMixin):
    """Smoothed mean-target encoding of one categorical column.

       fit_transform(X, y)  -> out-of-fold values: rows of one paper (GroupKFold on `group_col`)
                               are encoded with statistics from the OTHER papers only.
       fit(X, y).transform  -> full training statistics (use for validation / test rows).
       Unseen categories fall back to the training prior.  Keys are case-folded and stripped;
       missing / 'Unknown' share one key and also get a `__missing` indicator.
       Encoded value = (sum_y + prior * smoothing) / (n + smoothing)."""

    def __init__(self, column, prefix, group_col=GROUP_COL, n_splits=5, smoothing=20.0):
        self.column = column
        self.prefix = prefix
        self.group_col = group_col
        self.n_splits = n_splits
        self.smoothing = smoothing

    @staticmethod
    def _keys(s: pd.Series) -> pd.Series:
        def k(v):
            if _is_missing(v):
                return "__unknown__"
            t = str(v).strip().casefold()
            return "__unknown__" if t in MISSING_TOKENS else t
        return s.map(k)

    def _table(self, keys, y):
        g = pd.DataFrame({"k": keys.to_numpy(), "y": y}).groupby("k")["y"].agg(["sum", "count"])
        prior = float(np.mean(y))
        return (g["sum"] + prior * self.smoothing) / (g["count"] + self.smoothing), prior

    def _check_y(self, y):
        y = np.asarray(y, dtype=float)
        if np.isnan(y).any():
            raise ValueError("OOFTargetEncoder: y contains NaN - drop rows with unknown target first.")
        return y

    def fit(self, X, y):
        y = self._check_y(y)
        self.table_, self.prior_ = self._table(self._keys(X[self.column]), y)
        return self

    def _apply(self, keys, table, prior):
        return keys.map(table).fillna(prior).to_numpy(dtype=float)

    def transform(self, X):
        keys = self._keys(X[self.column])
        return pd.DataFrame({f"{self.prefix}__te": self._apply(keys, self.table_, self.prior_),
                             f"{self.prefix}__missing": (keys == "__unknown__").astype("float32").to_numpy()},
                            index=X.index)

    def fit_transform(self, X, y):
        y = self._check_y(y)
        self.fit(X, y)
        keys = self._keys(X[self.column]).reset_index(drop=True)
        g = X[self.group_col].reset_index(drop=True).astype(object)
        g = g.where(g.notna(), pd.Series([f"__row{i}" for i in range(len(g))]))   # papers without DOI = own group
        te = np.empty(len(X))
        n_splits = min(self.n_splits, g.nunique())
        for tr, va in GroupKFold(n_splits=n_splits).split(keys, y, g):
            table, prior = self._table(keys.iloc[tr], y[tr])
            te[va] = self._apply(keys.iloc[va], table, prior)
        return pd.DataFrame({f"{self.prefix}__te": te,
                             f"{self.prefix}__missing": (keys == "__unknown__").astype("float32").to_numpy()},
                            index=X.index)

    def get_feature_names_out(self, input_features=None):
        return np.array([f"{self.prefix}__te", f"{self.prefix}__missing"])


# =========================================================================================
# 6. THE FULL ENCODER
# =========================================================================================
class PerlaEncoder(BaseEstimator, TransformerMixin):
    """fit(X, y) / transform(X) / fit_transform(X, y) on the cleaned+parsed dataframe.
    Returns a DataFrame of float features with stable, readable column names.

    y is only needed for the target-encoded column(s).  Use fit_transform on the training fold
    (out-of-fold target encoding) and transform on the validation fold."""

    def __init__(self, min_count=30, composition_min_count=100, te_n_splits=5, te_smoothing=20.0,
                 add_missing_indicators=True, aliases=None, group_col=GROUP_COL):
        self.min_count = min_count
        self.composition_min_count = composition_min_count
        self.te_n_splits = te_n_splits
        self.te_smoothing = te_smoothing
        self.add_missing_indicators = add_missing_indicators
        self.aliases = aliases
        self.group_col = group_col

    def _build(self):
        al = TOKEN_ALIASES if self.aliases is None else self.aliases
        parts = [("composition", PerovskiteCompositionEncoder(min_count=self.composition_min_count)),
                 ("booleans", ThreeStateBoolean(BOOLEAN_COLS))]
        parts += [(f"onehot_{p}", MultiLabelTokenEncoder(c, p, "single", self.min_count))
                  for c, p in ONEHOT_COLS]
        parts += [(f"multi_{p}", MultiLabelTokenEncoder(c, p, k, self.min_count, al))
                  for c, p, k in MULTILABEL_COLS]
        parts += [(f"te_{p}", OOFTargetEncoder(c, p, self.group_col, self.te_n_splits, self.te_smoothing))
                  for c, p in TARGET_ENCODED_COLS]
        parts += [("numeric", NumericBlock(NUMERIC_COLS, ZERO_MEANS_MISSING,
                                           add_indicators=self.add_missing_indicators)),
                  ("annealing", AnnealingEncoder())]
        return parts

    @staticmethod
    def _needs_y(t):
        return isinstance(t, OOFTargetEncoder)

    def _check_columns(self, X):
        need = ({c for c, *_ in MULTILABEL_COLS} | {c for c, _ in ONEHOT_COLS} | set(BOOLEAN_COLS)
                | set(NUMERIC_COLS) | {c for c, _ in TARGET_ENCODED_COLS}
                | {ANNEAL_TEMP_COL, ANNEAL_TIME_COL, self.group_col}
                | {c for pair in COMP_COLS.values() for c in pair})
        missing = sorted(need - set(X.columns))
        if missing:
            raise KeyError("Missing columns (did you run parse_perovskite_columns and parse_more_columns "
                           f"and concatenate them? see assemble_clean_frame): {missing}")

    # ---- safety checks (added for the CV pipeline) -------------------------------------------
    @staticmethod
    def _check_index(X):
        # the blocks are glued together on the index; a duplicated index would silently misalign them
        if not X.index.is_unique:
            raise ValueError("PerlaEncoder: X.index must be unique (use df.reset_index(drop=True)).")

    @staticmethod
    def _check_y(X, y):
        if y is None:
            raise ValueError("PerlaEncoder: y (the PCE of the TRAINING rows) is required for target encoding.")
        if len(y) != len(X):
            raise ValueError(f"PerlaEncoder: len(y)={len(y)} but len(X)={len(X)}.")
        if isinstance(y, pd.Series) and not y.index.equals(X.index):
            raise ValueError("PerlaEncoder: y.index differs from X.index - y would be matched to the wrong rows.")

    def fit(self, X, y=None):
        self._check_columns(X)
        self._check_index(X)
        if any(self._needs_y(t) for _, t in self._build()):
            self._check_y(X, y)
        self.parts_ = self._build()
        for name, t in self.parts_:
            t.fit(X, y) if self._needs_y(t) else t.fit(X)
        self.feature_names_ = list(self._transform_blocks(X.iloc[:1]).columns)   # names only
        return self

    def fit_transform(self, X, y=None):
        """TRAINING data only.  Target-encoded columns are filled out-of-fold (grouped by paper)."""
        self._check_columns(X)
        self._check_index(X)
        self._check_y(X, y)
        self.parts_ = self._build()
        blocks = []
        for name, t in self.parts_:
            blocks.append(t.fit_transform(X, y) if self._needs_y(t) else t.fit(X).transform(X))
        out = pd.concat(blocks, axis=1)
        self.feature_names_ = list(out.columns)          # frozen here; transform() must reproduce it
        return out

    def _transform_blocks(self, X):
        return pd.concat([t.transform(X) for _, t in self.parts_], axis=1)

    def transform(self, X):
        """Validation / test data.  Uses ONLY what was learned in fit / fit_transform."""
        if not hasattr(self, "parts_"):
            raise RuntimeError("PerlaEncoder is not fitted yet.")
        self._check_index(X)
        out = self._transform_blocks(X)
        if list(out.columns) != self.feature_names_:
            raise RuntimeError("PerlaEncoder.transform produced different columns than fit_transform.")
        return out

    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names_)


# =========================================================================================
# CONVENIENCE
# =========================================================================================
def assemble_clean_frame(copy_df, parse_perovskite_columns, parse_more_columns):
    """copy_df + composition parse + numeric/nested parse, in one frame.
    NOTE: call it with the two parser functions imported from SEPARATE modules
    (perovskite_parser.py / perovskite_parser_extra.py).  In the notebook both cells define
    _RE_RANGE, _join, MISSING_WORDS, UNKNOWN_WORDS, NAN and flag_report, so after running the
    second cell the first parser breaks (ValueError: could not convert string to float: 'n')."""
    comp = parse_perovskite_columns(copy_df)
    more = parse_more_columns(copy_df)
    extra = [c for c in more.columns if c not in comp.columns]
    return pd.concat([comp, more[extra]], axis=1)


def get_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Numeric PCE / Voc / Jsc / FF from your *__parsed columns (None -> NaN)."""
    return pd.DataFrame({k: pd.to_numeric(df[v], errors="coerce") for k, v in TARGET_COLS.items()},
                        index=df.index)


def get_groups(df: pd.DataFrame) -> pd.Series:
    """Paper id for GroupKFold (papers without a DOI become their own group)."""
    g = df[GROUP_COL].astype(object)
    return g.where(g.notna(), pd.Series([f"__row{i}" for i in range(len(df))], index=df.index))
