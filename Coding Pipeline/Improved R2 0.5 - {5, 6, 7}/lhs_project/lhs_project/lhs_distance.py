"""
lhs_distance.py - mixed-variable (Gower) distance on the candidate genome, and the training-support measure
==========================================================================================================
Used for (a) "how far is a candidate from real devices" (support) and (b) the diversity constraint of the consensus list.

DISTANCE
    d(a, b) = mean over the variables available in BOTH rows of
        categorical variable : 0 if equal, 1 otherwise
        continuous variable  : min(|a - b| / range, 1)       (range = global design bounds; log10 for anneal time)
    Every variable has the same weight (9 categorical + 9 continuous).  A continuous value that is missing on either side
    (e.g. an unspecified layer thickness) is simply left out of that pair's average; nothing is imputed.
    Composition fractions are 0 for ions that are absent from the family (same convention in candidates and training rows).

SUPPORT
    support_score(x) = mean Gower distance to the k nearest TRAIN genomes.
    Reference distribution: the same score for a random subsample of TRAIN rows measured against TRAIN rows of OTHER papers
    (the situation of a recipe from a new paper).  support_percentile(x) = share of reference rows whose score is <= x's.
    A candidate with percentile 0.95 is farther from the data than 95 % of real held-out-paper devices.
    This is a descriptive distance in the interpretable genome space, not a probability and not a guarantee of model accuracy.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from lhs_design_space import CAT_VARS, CONT_VARS, DesignSpace


class GenomeCoder:
    """genome DataFrame -> (int category codes, normalised continuous values).  Codes are shared between calls so
    categories unseen in training get a new code and mismatch every training row."""

    def __init__(self, space: DesignSpace):
        self.maps = {v: {} for v in CAT_VARS}
        self.ranges = {v: space.global_range(v) for v in CONT_VARS}

    def encode(self, g: pd.DataFrame):
        cat = np.empty((len(g), len(CAT_VARS)), dtype=np.int32)
        for j, v in enumerate(CAT_VARS):
            m = self.maps[v]
            cat[:, j] = [m.setdefault(k, len(m)) for k in g[v].astype(str).to_numpy()]
        cont = np.empty((len(g), len(CONT_VARS)), dtype=np.float32)
        for j, v in enumerate(CONT_VARS):
            lo, hi = self.ranges[v]
            cont[:, j] = (pd.to_numeric(g[v], errors="coerce").to_numpy(dtype=float) - lo) / (hi - lo)
        return cat, cont


def gower_block(qc, qx, tc, tx) -> np.ndarray:
    """Gower distances between every query row and every target row -> (n_query, n_target) float32."""
    num = np.zeros((len(qc), len(tc)), dtype=np.float32)
    den = np.zeros_like(num)
    for j in range(qc.shape[1]):
        num += (qc[:, None, j] != tc[None, :, j])
        den += 1.0
    for j in range(qx.shape[1]):
        d = np.abs(qx[:, None, j] - tx[None, :, j])
        ok = ~np.isnan(d)
        num += np.where(ok, np.minimum(d, 1.0), 0.0)
        den += ok
    return num / np.maximum(den, 1.0)


class SupportModel:
    def __init__(self, space: DesignSpace, k: int = 5, reference_rows: int = 2000, seed: int = 42, chunk: int = 256):
        self.k, self.chunk = k, chunk
        self.coder = GenomeCoder(space)
        tg = space.train_genome
        tg = tg.drop_duplicates(subset=CAT_VARS + CONT_VARS + ["paper"]).reset_index(drop=True)
        self.n_train_unique = len(tg)
        self.tc, self.tx = self.coder.encode(tg)
        papers = pd.factorize(tg["paper"])[0]
        rng = np.random.RandomState(seed)
        ref_idx = rng.choice(len(tg), size=min(reference_rows, len(tg)), replace=False)
        ref = np.empty(len(ref_idx))
        for s in range(0, len(ref_idx), chunk):
            ii = ref_idx[s:s + chunk]
            D = gower_block(self.tc[ii], self.tx[ii], self.tc, self.tx)
            D[papers[ii][:, None] == papers[None, :]] = np.inf          # recipes of the SAME paper do not count as support
            ref[s:s + len(ii)] = np.partition(D, k - 1, axis=1)[:, :k].mean(axis=1)
        self.reference = np.sort(ref)

    def score(self, g: pd.DataFrame) -> np.ndarray:
        qc, qx = self.coder.encode(g)
        out = np.empty(len(g))
        for s in range(0, len(g), self.chunk):
            D = gower_block(qc[s:s + self.chunk], qx[s:s + self.chunk], self.tc, self.tx)
            out[s:s + self.chunk] = np.partition(D, self.k - 1, axis=1)[:, :self.k].mean(axis=1)
        return out

    def percentile(self, scores: np.ndarray) -> np.ndarray:
        return np.searchsorted(self.reference, scores, side="right") / len(self.reference)


def pairwise_gower(coder: GenomeCoder, g: pd.DataFrame) -> np.ndarray:
    c, x = coder.encode(g)
    return gower_block(c, x, c, x)
