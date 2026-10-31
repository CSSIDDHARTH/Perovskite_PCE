"""
lhs_config.py - configuration of the LHS candidate-generation pipeline
======================================================================
Every number that is a *judgement* (bounds, thresholds, weights) lives here, with the reason next to it, so that it can
be reviewed by a domain expert and changed without touching the code.  `LHSConfig.to_dict()` is written verbatim to
`lhs_sampling_config.json`, so a run can always be reproduced from its outputs.

Nothing in this file was measured on validation / test data.  Data-derived quantities (catalogues, bounds that come from
quantiles) are computed in `lhs_design_space.py` from the 70 % TRAIN split only.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

# ------------------------------------------------------------------------------------------------------------------
# The two finalists (ids exactly as written by run_models.py / model_selection.json)
# ------------------------------------------------------------------------------------------------------------------
DIRECT_ID = "Direct|VotingOriginal"
TWOSTAGE_ID = "TwoStage[S1=HistGradientBoosting]|VotingET"
FINALISTS = (DIRECT_ID, TWOSTAGE_ID)
SHORT = {DIRECT_ID: "direct", TWOSTAGE_ID: "twostage"}          # column prefixes in the output tables
REPORTED_RUN_ID = "5fold_seed0_split42_675a0c14"

# ------------------------------------------------------------------------------------------------------------------
# Raw PERLA columns that are NEVER allowed to appear in a candidate (targets, measured outputs, ids, metadata).
# This is an explicit list, not a substring match.  `lhs_predict.assert_no_leakage` additionally checks the candidate
# frame against the encoder's own TARGET_COLS / GROUP_COL, and the encoded matrix with perla_cv.check_no_target_columns.
# ------------------------------------------------------------------------------------------------------------------
FORBIDDEN_RAW_COLUMNS = (
    "JV_default_PCE", "JV_default_PCE__parsed", "JV_default_PCE__flags",        # target (PCE, %)
    "Voc", "Voc__parsed", "Voc__flags",                                         # measured outputs = Stage-1 targets
    "Jsc", "Jsc__parsed", "Jsc__flags",
    "FF", "FF__parsed", "FF__flags",
    "Ref_publication_year", "Source", "perla_row_id",                           # publication metadata / identifiers
)
ID_PLACEHOLDER_COLUMNS = ("Ref_DOI_number",)    # required to exist by PerlaEncoder._check_columns; value is always None


def _default_out_dir() -> str:
    return "/kaggle/working/perovskite_outputs/lhs" if Path("/kaggle/working").exists() else "perovskite_outputs/lhs"


@dataclass
class ContinuousSpec:
    """Sampling definition of one genuinely continuous design variable (global bounds; recipe-conditional ones are
    overridden per row for the layer thicknesses and the composition fractions)."""
    name: str
    unit: str
    lo: float
    hi: float
    scale: str = "linear"            # "linear" | "log"
    round_to: float | None = None    # rounding step applied AFTER scaling (None = no rounding)
    rationale: str = ""


@dataclass
class LHSConfig:
    # ---- paths -----------------------------------------------------------------------------------------------
    data_path: str = "perla_final_filtered.pkl"
    out_dir: str = field(default_factory=_default_out_dir)
    reported_run_dir: str = f"results/{REPORTED_RUN_ID}"            # only read (model_selection.json), never written
    reference_uq_dir: str | None = None                             # e.g. results_uq/uq_seed0_split42_9777e691 (fidelity check)
    reference_prep_cache: str | None = None                         # e.g. results_uq/_prep_cache/prep_923a07d6d1d4.joblib (fidelity check)

    # ---- reproducibility ---------------------------------------------------------------------------------------
    seed: int = 42                      # LHS seed (also seeds the support-reference subsample and the SHAP background)
    model_seed: int = 0                 # seed of the reported run's models (run_models.py --seed 0)
    split_seed: int = 42                # must equal the split of the reported run
    n_candidates: int = 50_000          # initial exploration size, NOT a claim of optimality
    lhs_scramble: bool = True
    lhs_optimization: str | None = None  # None | "random-cd" | "lloyd"  (optimisation is slow for 50k x 18; off by default)
    batch_size: int = 5_000

    # ---- models (reconstructed with the reported run's settings) -------------------------------------------------
    quick: bool = False                 # smoke-test sizes (PRELIMINARY; never use for science)
    n_jobs: int = -1
    stage1_base: str = "hgb"
    inner_folds: int = 5
    add_product: bool = False
    finalists: tuple = FINALISTS

    # ---- conformal prediction (MAPIE CQR, as perla_explain_uq.py) -----------------------------------------------
    confidence: float = 0.90
    cal_fraction: float = 0.5           # development check: share of validation PAPERS used for calibration
    cal_repeats: int = 3
    score_test: bool = False            # one-shot final coverage on TEST (calibrates on all validation); off by default

    # ---- design space: which part of PERLA is explored -------------------------------------------------------------
    tempering_gamma: float = 0.5        # category probability  p_k ~ (distinct papers with k) ** gamma ; 0 = uniform, 1 = empirical
    min_papers_category: int = 10       # a category/recipe needs >= this many DISTINCT PAPERS (per architecture) to be sampled
    min_papers_family: int = 15         # same for composition families
    min_thickness_obs: int = 20         # observed thickness values needed for recipe-specific thickness bounds
    allowed_perovskite_procedures: tuple = ("Spin-coating",)     # one-step solution processing (see README: why)
    allowed_n_steps: tuple = (1,)
    allowed_substrates: tuple = ("SLG | FTO", "SLG | ITO")       # rigid glass; plastic substrates need other temperature limits
    allowed_a_ions: tuple = ("Cs", "FA", "MA")
    allowed_b_ions: tuple = ("Pb",)
    allowed_c_ions: tuple = ("I", "Br")
    allow_empty_htl: bool = False       # an empty HTL stack is ambiguous in PERLA (HTL-free vs not reported)
    excluded_media: tuple = ("Antisolvent", "Gas", "Unknown", "unknown", "Air", "Hot air")   # unspecific labels
    # Bounds of the global continuous variables.  Observed quantiles are only DESCRIPTIVE (see lhs_feature_bounds.csv);
    # the values below are proposals that a domain expert should review.
    bounds: dict = field(default_factory=lambda: {
        "anneal_temperature_C": (70.0, 160.0),
        "anneal_time_min": (2.0, 60.0),
        "perovskite_thickness_nm": (200.0, 650.0),
    })
    layer_hard_bounds_nm: dict = field(default_factory=lambda: {   # absolute sanity limits; recipe bounds are clipped to them
        "etl": (5.0, 400.0), "htl": (5.0, 400.0), "bc": (20.0, 300.0)})
    layer_fallback_quantiles: tuple = (0.10, 0.90)                  # per-recipe quantiles of the observed thickness that define its sampling range
    composition_default_fraction_bounds: tuple = (0.05, 0.95)
    round_to: dict = field(default_factory=lambda: {
        "anneal_temperature_C": 1.0, "anneal_time_min": 0.5, "perovskite_thickness_nm": 5.0,
        "etl_thickness_nm": 1.0, "htl_thickness_nm": 1.0, "bc_thickness_nm": 1.0, "fraction": 0.01})

    # ---- fixed device context (NOT optimised; identical for all candidates, so predictions are comparable) -------
    cell_area_cm2: float = 0.10         # training median; the area strongly influences reported PCE and is not a recipe choice
    cell_flexible: bool = False
    cell_semitransparent: bool = False
    perovskite_single_crystal: bool = False
    perovskite_band_gap_graded: bool = False
    # band_gap__mean and Perovskite_band_gap_estimation_basis are properties/measurement metadata, not fabrication
    # choices: they are left MISSING and handled by the existing KNN imputer, exactly like training rows without them.

    # ---- feasibility ---------------------------------------------------------------------------------------------
    ma_min_fraction_for_thermal_rule: float = 0.5   # domain rule: MA-majority films decompose when annealed hot
    ma_max_anneal_temperature_C: float = 150.0
    tolerance_factor_range: tuple = (0.80, 1.05)    # Goldschmidt t, effective radii of Kieslich et al. 2015
    min_pair_papers: int = 3                        # categorical pairs seen in fewer papers are flagged `uncommon`
    reject_uncommon: bool = False
    support_k: int = 5
    support_reference_rows: int = 2000
    max_support_percentile: float = 0.95            # eligible for ranking only if not farther from the data than 95 % of held-out-paper reference rows
    plausible_pce_range: tuple = (0.0, 30.0)        # predictions outside are FLAGGED (never clipped)

    # ---- ranking -------------------------------------------------------------------------------------------------
    top_k: int = 100
    consensus_k: int = 50
    consensus_pool: int = 2000
    diversity_min_distance: float = 0.15            # Gower distance (see lhs_distance.py)
    score_lambda: float = 0.5                       # score C = mu - lambda * half_width ; 0 = strategy A, 1 = strategy B
    lambda_grid: tuple = (0.0, 0.25, 0.5, 0.75, 1.0)
    support_grid: tuple = (0.90, 0.95, 0.99)

    # ---- optional SHAP --------------------------------------------------------------------------------------------
    shap: bool = False
    shap_background: int = 50
    shap_group_permutations: int = 20
    shap_group_background: int = 20

    # ---------------------------------------------------------------------------------------------------------------
    def to_dict(self) -> dict:
        return json.loads(json.dumps(asdict(self), default=str))

    def dump(self, path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @staticmethod
    def from_json(path) -> "LHSConfig":
        d = json.loads(Path(path).read_text())
        fields = {f for f in LHSConfig.__dataclass_fields__}
        d = {k: (tuple(v) if isinstance(v, list) and k not in ("bounds",) else v) for k, v in d.items() if k in fields}
        return LHSConfig(**d)

    def continuous_specs(self) -> list[ContinuousSpec]:
        r, b = self.round_to, self.bounds
        return [
            ContinuousSpec("anneal_temperature_C", "degC", *b["anneal_temperature_C"], "linear", r["anneal_temperature_C"],
                           "Single final thermal anneal of the perovskite. Lower end: solvent removal / crystallisation of "
                           "DMF-DMSO films is incomplete well below ~70 degC; upper end 160 degC: MA-containing films "
                           "decompose when annealed hot (guarded by a separate rule above 150 degC). Observed 5-95 % range of the "
                           "training data is 50-150 degC; 100 degC alone is ~50 % of the observed values."),
            ContinuousSpec("anneal_time_min", "min", *b["anneal_time_min"], "log", r["anneal_time_min"],
                           "Anneal duration spans more than one order of magnitude, therefore log-uniform LHS. Observed "
                           "median ~15 min, 5-95 % ~3-90 min with a heavy tail of data-entry outliers (max 5400)."),
            ContinuousSpec("perovskite_thickness_nm", "nm", *b["perovskite_thickness_nm"], "linear", r["perovskite_thickness_nm"],
                           "Observed 5-95 % range 175-650 nm (max 4000 = data-entry outlier). Thinner films absorb too little, "
                           "thicker ones exceed typical carrier diffusion lengths."),
        ]
