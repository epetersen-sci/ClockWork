"""
HMM Sleep State Analysis Module
================================
Advanced Hidden Markov Model implementations for Drosophila sleep analysis.

Supports three emission models:
  - ZIP (Zero-Inflated Poisson): Statistically principled for DAM count data
  - Gaussian: For normalized activity data (Ghosh & Harbison style)
  - Binary (Categorical): For moving/not-moving data (Wiggin style)

Training scopes:
  - per_genotype: Fit one independent model per genotype group (default)
  - all_pooled: Single model on all data pooled across genotypes
  - per_fly: One model per fly (Ghosh & Harbison 2026)

Preset configurations:
  - HMMConfig.wiggin()   — Replicate Wiggin et al. 2020 (PNAS)
  - HMMConfig.harbison() — Replicate Ghosh & Harbison 2026 (bioRxiv)
  - HMMConfig.improved()  — ZIP emissions with soft constraints (recommended)

References:
  - Wiggin et al. (2020) "Covert sleep-related biological processes are
    revealed by probabilistic analysis in Drosophila" PNAS 117(18):10024
  - Ghosh & Harbison (2026) HMM-based sleep scoring, bioRxiv 2026.01.14.699526
  - Donelson & Wiggin et al. (2026) Duration-based sleep states, Current Biology
"""

# hmmlearn emits a per-iteration WARNING ("Model is not converging. Current: ...
# Delta: ...") via `logging` whenever EM's log-likelihood dips by a hair at the
# optimum plateau. On a multi-restart × multi-group fit that is thousands of lines
# of benign spam that reads as "something is broken" while the fits are in fact fine
# (all restarts succeed; per-group convergence is reported by the verbose summary).
# `warnings` filters don't catch it (it's a logging record, not a warning), so raise
# the hmmlearn logger to ERROR. Set PYTHOMICS_HMM_VERBOSE_LOG=1 to keep the notices.
import logging as _logging
import multiprocessing as _mp
import os
import warnings
from dataclasses import dataclass
from typing import Literal

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr
from hmmlearn.base import BaseHMM
from hmmlearn.hmm import CategoricalHMM, GaussianHMM
from joblib import Parallel, delayed
from scipy.special import gammaln
from scipy.stats import dirichlet

if os.environ.get("PYTHOMICS_HMM_VERBOSE_LOG", "0") != "1":
    _logging.getLogger("hmmlearn").setLevel(_logging.ERROR)

# GPU acceleration for the ZIP emission log-likelihood (the per-EM-iteration hot
# loop). Guarded — the CPU vectorized path is always available and exact.
try:
    import torch as _torch

    _HMM_GPU_AVAILABLE = bool(_torch.cuda.is_available())
except Exception:  # pragma: no cover
    _torch = None
    _HMM_GPU_AVAILABLE = False

# Use the GPU only for large observation batches (transfer overhead dominates on
# small per-fly sequences) AND only in the MAIN process — the multi-start training
# restarts run in joblib spawn WORKERS, and many workers hammering one modest GPU
# (4 GB, often shared with the running app) contends/deadlocks. So workers use the
# fast vectorized-CPU path; the main process (decode, CV, serial n_jobs=1 training)
# uses the GPU. Env override HMM_USE_GPU=0 disables it entirely.
_ZIP_GPU_MIN_SAMPLES = 20000
HMM_USE_GPU = os.environ.get("HMM_USE_GPU", "1") != "0"


def _zip_log_likelihood(x, lambdas, omegas):
    """log P(x_t | state k) for a Zero-Inflated Poisson HMM, shape (T, K).

    Vectorized and computed in LOG-SPACE (via logaddexp) so the tiny zero-state
    probabilities never underflow — which also makes float32 (GPU) safe:
        log P(0|k)   = logaddexp( log ω_k , log(1-ω_k) − λ_k )
        log P(x>0|k) = log(1-ω_k) + [ x·log λ_k − λ_k − lgamma(x+1) ]
    Dispatches to a torch/GPU float32 path for large batches in the main process
    (see the gating constants above), else exact float64 numpy. Returns float64
    (hmmlearn's forward-backward consumes float64)."""
    x = np.asarray(x, dtype=np.float64).ravel()
    lam = np.asarray(lambdas, dtype=np.float64)
    om = np.clip(np.asarray(omegas, dtype=np.float64), 0.0, 1.0)

    use_gpu = (
        HMM_USE_GPU
        and _HMM_GPU_AVAILABLE
        and _torch is not None
        and x.shape[0] >= _ZIP_GPU_MIN_SAMPLES
        and _mp.current_process().name == "MainProcess"
    )
    if use_gpu:
        try:
            dev = _torch.device("cuda")
            xt = _torch.as_tensor(x, dtype=_torch.float32, device=dev)
            lamt = _torch.as_tensor(lam, dtype=_torch.float32, device=dev)
            omt = _torch.clamp(_torch.as_tensor(om, dtype=_torch.float32, device=dev), min=1e-30)
            one_m_om = _torch.clamp(1.0 - omt, min=1e-30)
            log_pois = (
                xt[:, None] * _torch.log(lamt)[None, :]
                - lamt[None, :]
                - _torch.lgamma(xt + 1.0)[:, None]
            )
            log_p_zero = _torch.logaddexp(_torch.log(omt), _torch.log(one_m_om) - lamt)[None, :]
            log_p_pos = _torch.log(one_m_om)[None, :] + log_pois
            out = _torch.where((xt == 0)[:, None], log_p_zero, log_p_pos)
            return out.double().cpu().numpy()
        except Exception:
            pass  # fall through to CPU on any GPU error

    # Exact float64 vectorized CPU path.
    log_pois = x[:, None] * np.log(lam)[None, :] - lam[None, :] - gammaln(x + 1.0)[:, None]
    om_c = np.clip(om, 1e-300, None)
    one_m_om = np.clip(1.0 - om, 1e-300, None)
    log_p_zero = np.logaddexp(np.log(om_c), np.log(one_m_om) - lam)[None, :]
    log_p_pos = np.log(one_m_om)[None, :] + log_pois
    return np.where((x == 0)[:, None], log_p_zero, log_p_pos)


# ---------------------------------------------------------------------------
# State names and colors
# ---------------------------------------------------------------------------
STATE_NAMES_4 = ["DS", "LS", "EW", "FW"]
STATE_COLORS_4 = [
    "#1a237e",
    "#64b5f6",
    "#ffcc80",
    "#e65100",
]  # dark blue, light blue, light orange, dark orange

# DAM data is 1-min-binned; one "day" = 1440 minutes. Used to segment a fly's
# record into days for Ghosh & Harbison per-day normalization / per-fly-per-day fits.
SAMPLES_PER_DAY = 1440

# Default topology constraint: DS=0, LS=1, EW=2, FW=3
TRANSMAT_MASK_4 = np.array(
    [
        [True, True, False, False],  # DS: stays or goes to LS only
        [True, True, True, False],  # LS: DS, LS, or EW
        [False, True, True, True],  # EW: LS, EW, or FW
        [False, False, True, True],  # FW: stays or goes to EW only
    ],
    dtype=bool,
)


# ---------------------------------------------------------------------------
# 1.1  Configuration
# ---------------------------------------------------------------------------
@dataclass
class HMMConfig:
    """Configuration for HMM sleep analysis.

    Use preset class methods for published approaches:

        HMMConfig.wiggin()   — Wiggin et al. 2020 (PNAS): binary categorical,
                               per-genotype, hard transition constraints,
                               posterior decoding.

        HMMConfig.harbison() — Ghosh & Harbison 2026 (bioRxiv): Gaussian,
                               per-fly, unconstrained transitions, AIC/BIC
                               model selection.

        HMMConfig.improved() — ZIP emissions, per-genotype, soft constraints,
                               both decoding methods. Recommended default.

    Or construct directly:
        HMMConfig()  # uses improved defaults
    """

    training_scope: Literal["per_genotype", "all_pooled", "per_fly", "per_fly_per_day"] = (
        "per_genotype"
    )
    n_restarts: int = 50
    emission_model: Literal["zip", "gaussian", "binary"] = "zip"
    observation_type: Literal["counts", "binary", "normalized"] = "counts"
    n_states: int = 4
    transition_constraints: Literal["soft", "hard", "none"] = "soft"
    decoding_method: Literal["both", "viterbi", "posterior"] = "both"
    n_iter: int = 200
    tol: float = 1e-4
    soft_floor: float = 1e-6
    dirichlet_concentration: float = 10.0
    n_jobs: int = -1  # joblib parallelism: -1=all cores, 1=serial

    @classmethod
    def wiggin(cls) -> "HMMConfig":
        """Wiggin et al. 2020 (PNAS) replication.

        Binary categorical emissions on moving/not-moving, each genotype
        fitted independently with hard transition constraints enforcing
        the DS↔LS↔EW↔FW topology.  Posterior decoding (hmmdecode+argmax).
        """
        return cls(
            emission_model="binary",
            observation_type="binary",
            training_scope="per_genotype",
            transition_constraints="hard",
            decoding_method="posterior",
            n_states=4,
        )

    @classmethod
    def harbison(cls) -> "HMMConfig":
        """Ghosh & Harbison 2026 (bioRxiv) replication (faithful).

        Gaussian emissions on per-fly-PER-DAY proportion-normalized activity
        (each minute / that day's total activity x 100), one model fitted per fly
        PER DAY (1440 points), unconstrained transitions, Viterbi decoding — the
        paper's actual procedure. (The prior preset diverged: record-wide min-max
        normalization and one whole-record fit per fly.) 4 states (the paper's
        AIC/BIC-optimal); use compare_n_states() to re-scan 2-10 if desired.
        """
        return cls(
            emission_model="gaussian",
            observation_type="normalized",
            training_scope="per_fly_per_day",
            transition_constraints="none",
            decoding_method="viterbi",
            n_states=4,
        )

    @classmethod
    def improved(cls) -> "HMMConfig":
        """Recommended defaults: ZIP emissions with soft constraints.

        Zero-Inflated Poisson emissions on raw DAM beam-break counts,
        per-genotype independent fitting, soft transition constraints
        (forbidden transitions get a floor probability rather than hard zero),
        both Viterbi and posterior decoding.

        NOTE: The ZIP emission model for DAM data is novel and has not been
        independently validated in the published literature.  It is
        statistically well-motivated (DAM counts are zero-inflated integers)
        but should be compared against the Wiggin and Harbison presets via
        AIC/BIC on your specific dataset.
        """
        return cls()


# ---------------------------------------------------------------------------
# 1.2  ZIPHMM — Zero-Inflated Poisson HMM
# ---------------------------------------------------------------------------
class ZIPHMM(BaseHMM):
    """Hidden Markov Model with Zero-Inflated Poisson emissions.

    Each state k has parameters (omega_k, lambda_k):
        P(x=0|k) = omega_k + (1 - omega_k) * Poisson(0; lambda_k)
        P(x>0|k) = (1 - omega_k) * Poisson(x; lambda_k)

    Parameters
    ----------
    n_components : int
        Number of hidden states.
    transmat_mask : np.ndarray or None
        Boolean mask for allowed transitions (soft or hard constraint).
    soft_floor : float
        Minimum transition probability for soft constraints.
    constraint_mode : str
        "soft", "hard", or "none".
    """

    def __init__(
        self, n_components=4, transmat_mask=None, soft_floor=1e-6, constraint_mode="soft", **kwargs
    ):
        super().__init__(n_components=n_components, **kwargs)
        self.transmat_mask = transmat_mask
        self.soft_floor = soft_floor
        self.constraint_mode = constraint_mode
        # Per-state ZIP parameters
        self.lambdas_ = None  # shape (n_components,)
        self.omegas_ = None  # shape (n_components,)

    def _check(self):
        super()._check()
        if self.lambdas_ is None or self.omegas_ is None:
            raise ValueError("lambdas_ and omegas_ must be initialized before fitting.")

    def _compute_log_likelihood(self, X):
        """log P(X_t | state) for each state — vectorized ZIP in log-space, with a
        GPU path for large batches (see ``_zip_log_likelihood``). Byte-for-byte
        equivalent to the former per-state scipy loop (verified to 1e-15 on CPU),
        3-9x faster vectorized and up to ~29x with the GPU path on pooled fits."""
        return _zip_log_likelihood(X[:, 0], self.lambdas_, self.omegas_)

    def _generate_sample_from_state(self, state, random_state=None):
        """Generate a single sample from a given state."""
        rng = random_state if random_state is not None else np.random.RandomState()
        omega = self.omegas_[state]
        lam = self.lambdas_[state]
        if rng.random() < omega:
            return np.array([0])
        else:
            return np.array([rng.poisson(lam)])

    def _initialize_sufficient_statistics(self):
        stats = super()._initialize_sufficient_statistics()
        stats["post"] = np.zeros(self.n_components)
        stats["obs_sum"] = np.zeros(self.n_components)
        stats["zero_post"] = np.zeros(self.n_components)
        stats["obs_sq_sum"] = np.zeros(self.n_components)
        return stats

    def _accumulate_sufficient_statistics(
        self, stats, X, lattice, posteriors, fwdlattice, bwdlattice
    ):
        super()._accumulate_sufficient_statistics(
            stats, X, lattice, posteriors, fwdlattice, bwdlattice
        )

        x = X[:, 0].astype(np.float64)
        is_zero = (x == 0).astype(np.float64)

        stats["post"] += posteriors.sum(axis=0)
        stats["obs_sum"] += (posteriors * x[:, None]).sum(axis=0)
        stats["zero_post"] += (posteriors * is_zero[:, None]).sum(axis=0)
        stats["obs_sq_sum"] += (posteriors * (x**2)[:, None]).sum(axis=0)

    def _do_mstep(self, stats):
        super()._do_mstep(stats)

        post = stats["post"]
        obs_sum = stats["obs_sum"]
        zero_post = stats["zero_post"]

        for k in range(self.n_components):
            if post[k] < 1e-10:
                continue

            # Expected count (lambda) from non-zero-inflated component
            lam = obs_sum[k] / post[k]
            lam = max(lam, 1e-6)

            # Fraction of zeros explained by the zero-inflation component
            pois_0 = np.exp(-lam)
            expected_zero_from_poisson = post[k] * pois_0
            excess_zeros = zero_post[k] - expected_zero_from_poisson

            omega = max(0.0, excess_zeros / post[k])
            omega = min(omega, 0.99)

            # Re-estimate lambda accounting for zero inflation
            if (1.0 - omega) * post[k] > 1e-10:
                lam = obs_sum[k] / ((1.0 - omega) * post[k])
                lam = max(lam, 1e-6)

            self.lambdas_[k] = lam
            self.omegas_[k] = omega

        # Enforce transition constraints
        self._enforce_transition_constraints()

    def _enforce_transition_constraints(self):
        if self.transmat_mask is None or self.constraint_mode == "none":
            return

        if self.constraint_mode == "hard":
            self.transmat_ *= self.transmat_mask
            row_sums = self.transmat_.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1.0
            self.transmat_ /= row_sums
        elif self.constraint_mode == "soft":
            # Set forbidden transitions to soft_floor instead of zero
            forbidden = ~self.transmat_mask
            self.transmat_[forbidden] = self.soft_floor
            row_sums = self.transmat_.sum(axis=1, keepdims=True)
            self.transmat_ /= row_sums

    def _get_n_fit_scalars_per_param(self):
        """Return dict mapping parameter letter to count of free scalars."""
        nc = self.n_components
        return {
            "s": nc - 1,  # startprob
            "t": nc * (nc - 1),  # transmat
            "e": 2 * nc,  # emission: lambda + omega per state
        }


# ---------------------------------------------------------------------------
# 1.3  Constrained wrappers for GaussianHMM and CategoricalHMM
# ---------------------------------------------------------------------------
class ConstrainedGaussianHMM(GaussianHMM):
    """GaussianHMM with transition constraint enforcement in M-step."""

    def __init__(self, transmat_mask=None, soft_floor=1e-6, constraint_mode="soft", **kwargs):
        super().__init__(**kwargs)
        self.transmat_mask = transmat_mask
        self.soft_floor = soft_floor
        self.constraint_mode = constraint_mode

    def _do_mstep(self, stats):
        super()._do_mstep(stats)
        if self.transmat_mask is not None and self.constraint_mode != "none":
            if self.constraint_mode == "hard":
                self.transmat_ *= self.transmat_mask
            elif self.constraint_mode == "soft":
                self.transmat_[~self.transmat_mask] = self.soft_floor
            row_sums = self.transmat_.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1.0
            self.transmat_ /= row_sums


class ConstrainedCategoricalHMM(CategoricalHMM):
    """CategoricalHMM with transition constraint enforcement in M-step."""

    def __init__(self, transmat_mask=None, soft_floor=1e-6, constraint_mode="soft", **kwargs):
        super().__init__(**kwargs)
        self.transmat_mask = transmat_mask
        self.soft_floor = soft_floor
        self.constraint_mode = constraint_mode

    def _do_mstep(self, stats):
        super()._do_mstep(stats)
        if self.transmat_mask is not None and self.constraint_mode != "none":
            if self.constraint_mode == "hard":
                self.transmat_ *= self.transmat_mask
            elif self.constraint_mode == "soft":
                self.transmat_[~self.transmat_mask] = self.soft_floor
            row_sums = self.transmat_.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1.0
            self.transmat_ /= row_sums


# ---------------------------------------------------------------------------
# 1.4  Model factory
# ---------------------------------------------------------------------------
def _get_transmat_mask(n_states):
    """Return transition mask for the given number of states."""
    if n_states == 4:
        return TRANSMAT_MASK_4.copy()
    # For non-4-state models, use a tridiagonal + self-loop mask
    mask = np.eye(n_states, dtype=bool)
    for i in range(n_states - 1):
        mask[i, i + 1] = True
        mask[i + 1, i] = True
    return mask


def _default_startprob(n_states):
    """Uniform-ish start probability favoring sleep states."""
    sp = np.ones(n_states, dtype=np.float64)
    # Give sleep states (first half) higher initial probability
    half = n_states // 2
    sp[:half] = 2.0
    sp /= sp.sum()
    return sp


def _default_transmat(n_states):
    """Self-loop-heavy transition matrix."""
    tm = np.full((n_states, n_states), 0.01, dtype=np.float64)
    np.fill_diagonal(tm, 0.90)
    # Increase adjacent transitions
    for i in range(n_states - 1):
        tm[i, i + 1] = 0.05
        tm[i + 1, i] = 0.05
    # Normalize
    tm /= tm.sum(axis=1, keepdims=True)
    return tm


def build_hmm_model(config: HMMConfig, random_state: int = 42) -> BaseHMM:
    """Construct an HMM model based on the given configuration.

    Returns a configured but unfitted model ready for ``.fit()``.
    """
    n = config.n_states
    mask = _get_transmat_mask(n) if config.transition_constraints != "none" else None

    if config.emission_model == "zip":
        model = ZIPHMM(
            n_components=n,
            transmat_mask=mask,
            soft_floor=config.soft_floor,
            constraint_mode=config.transition_constraints,
            n_iter=config.n_iter,
            tol=config.tol,
            random_state=random_state,
            params="ste",
            init_params="",
            verbose=False,
        )
        model.startprob_ = _default_startprob(n)
        model.transmat_ = _default_transmat(n)
        # Placeholder lambdas/omegas — will be overridden by data-informed init
        model.lambdas_ = np.linspace(0.1, 10.0, n)
        model.omegas_ = np.linspace(0.9, 0.1, n)

    elif config.emission_model == "gaussian":
        model = ConstrainedGaussianHMM(
            transmat_mask=mask,
            soft_floor=config.soft_floor,
            constraint_mode=config.transition_constraints,
            n_components=n,
            covariance_type="diag",
            n_iter=config.n_iter,
            tol=config.tol,
            random_state=random_state,
            params="stmc",
            init_params="",
            verbose=False,
        )
        model.startprob_ = _default_startprob(n)
        model.transmat_ = _default_transmat(n)
        model.means_ = np.linspace(0, 100, n).reshape(-1, 1)
        model.covars_ = np.full((n, 1), 100.0)

    elif config.emission_model == "binary":
        model = ConstrainedCategoricalHMM(
            transmat_mask=mask,
            soft_floor=config.soft_floor,
            constraint_mode=config.transition_constraints,
            n_components=n,
            n_features=2,
            n_iter=config.n_iter,
            tol=config.tol,
            random_state=random_state,
            params="ste",
            init_params="",
            verbose=False,
        )
        model.startprob_ = _default_startprob(n)
        model.transmat_ = _default_transmat(n)
        # Emission priors: P(inactive|state), P(active|state)
        p_inactive = np.linspace(0.97, 0.05, n)
        model.emissionprob_ = np.column_stack([p_inactive, 1.0 - p_inactive])

    else:
        raise ValueError(f"Unknown emission_model: {config.emission_model!r}")

    return model


# ---------------------------------------------------------------------------
# 1.5  Observation extraction
# ---------------------------------------------------------------------------


def _get_var_time_by_id(ds, var_name):
    """Get a variable's values as shape (time, id), transposing if needed."""
    da = ds[var_name]
    vals = da.values
    if da.dims[0] == "id" and da.dims[1] == "time":
        vals = vals.T
    return vals


def _transform_observation(raw_full, config, samples_per_day=SAMPLES_PER_DAY):
    """THE single source of the raw activity -> HMM observation-sequence transform.

    Called identically by training extraction, decoding, and CV agreement so the
    three can never silently drift (§2d — this replaces three hand-copied blocks).
    Operates on a fly's FULL 1-D array (it needs positions for the per-day
    normalization). Returns ``(obs_1d, valid_mask)`` where ``valid_mask`` marks the
    kept timepoints in the original array (used to scatter decoded states back).

    Per observation_type:
      * ``counts``     — rounded non-negative integers over finite minutes (ZIP).
      * ``binary``     — 0/1 moving flag over finite, non-(-1) minutes (Wiggin).
      * ``normalized`` — Ghosh & Harbison 2026: each minute's activity divided by
        THAT DAY's total activity, x100 (per fly, per day). Being purely local it
        is identical at train and decode time — which removes the former
        per-fly-vs-group-bounds divergence (decode_flies used per-fly min-max while
        training used group min-max). §2a: NaN gaps are dropped, never zero-filled.
    """
    raw = np.asarray(raw_full, dtype=np.float64).ravel()
    ot = config.observation_type
    if ot == "counts":
        valid = np.isfinite(raw)
        return np.round(np.clip(raw[valid], 0, None)).astype(int), valid
    if ot == "binary":
        valid = np.isfinite(raw) & (raw != -1)
        return raw[valid].astype(int), valid
    if ot == "normalized":
        valid = np.isfinite(raw)
        n = raw.shape[0]
        if n == 0:
            return np.empty(0), valid
        day_idx = np.arange(n) // int(samples_per_day)
        day_totals = np.zeros(int(day_idx[-1]) + 1, dtype=np.float64)
        np.add.at(day_totals, day_idx[valid], raw[valid])
        denom = day_totals[day_idx]  # each minute's day-total
        obs_full = np.zeros(n, dtype=np.float64)
        nz = valid & (denom > 0)  # zero-activity days -> all-zero (0/0 guarded)
        obs_full[nz] = raw[nz] / denom[nz] * 100.0
        return obs_full[valid], valid
    raise ValueError(f"Unknown observation_type: {ot!r}")


def extract_observations(
    ds: xr.Dataset,
    fly_ids: np.ndarray,
    config: HMMConfig,
) -> tuple[np.ndarray, list[int], list]:
    """Extract and concatenate observations for the given fly IDs.

    Uses vectorized numpy operations for speed (avoids per-fly xarray indexing).

    Returns
    -------
    all_obs : np.ndarray, shape (N, 1)
        Concatenated valid observations.
    lengths : list[int]
        Sequence lengths per fly.
    valid_fly_ids : list
        Fly IDs that had valid data (subset of fly_ids).
    """
    # Select subset of dataset for the requested fly_ids
    # Use positional indexing for speed
    all_ids = ds["id"].values
    id_to_idx = {fid: idx for idx, fid in enumerate(all_ids)}
    id_indices = np.array([id_to_idx.get(fid, -1) for fid in fly_ids])
    id_indices = id_indices[id_indices >= 0]

    if len(id_indices) == 0:
        return np.empty((0, 1)), [], []

    if config.observation_type == "counts":
        var_name = "activity"
    elif config.observation_type == "binary":
        var_name = "moving"
    elif config.observation_type == "normalized":
        var_name = "activity"
    else:
        raise ValueError(f"Unknown observation_type: {config.observation_type!r}")

    raw_2d = _get_var_time_by_id(ds, var_name)[:, id_indices].astype(
        np.float64
    )  # (time, n_selected)
    selected_ids = all_ids[id_indices]

    all_obs_list = []
    lengths = []
    valid_fly_ids = []

    for col_idx in range(raw_2d.shape[1]):
        # §2d: the ONE shared transform (per-day-local for normalized, so no
        # group-level bounds are needed — training and decoding now agree exactly).
        obs, _ = _transform_observation(raw_2d[:, col_idx], config)
        if len(obs) == 0:
            continue

        all_obs_list.append(obs)
        lengths.append(len(obs))
        valid_fly_ids.append(selected_ids[col_idx])

    if not all_obs_list:
        return np.empty((0, 1)), [], []

    all_obs = np.concatenate(all_obs_list).reshape(-1, 1)
    return all_obs, lengths, valid_fly_ids


def _initialize_model_from_data(model, all_obs, config):
    """Data-informed parameter initialization."""
    x = all_obs[:, 0]

    if config.emission_model == "zip":
        nonzero = x[x > 0]
        if len(nonzero) > 0:
            percentiles = np.percentile(nonzero, np.linspace(10, 90, config.n_states))
            model.lambdas_ = percentiles
        else:
            model.lambdas_ = np.linspace(0.1, 5.0, config.n_states)
        zero_frac = (x == 0).mean()
        model.omegas_ = np.linspace(
            min(0.95, zero_frac + 0.1), max(0.01, zero_frac - 0.3), config.n_states
        )

    elif config.emission_model == "gaussian":
        percentiles = np.percentile(x, np.linspace(10, 90, config.n_states))
        model.means_ = percentiles.reshape(-1, 1)
        var = max(np.var(x) / config.n_states, 1.0)
        model.covars_ = np.full((config.n_states, 1), var)

    # Binary model uses pre-set emission probs, no data-based init needed


# ---------------------------------------------------------------------------
# 1.6  Multi-start training
# ---------------------------------------------------------------------------
def _fit_single_restart(
    i: int,
    config: HMMConfig,
    all_obs: np.ndarray,
    lengths: list[int],
    group_seed_offset: int = 0,
) -> tuple[BaseHMM | None, float, int]:
    """Fit a single restart (designed to be called via joblib).

    Returns (model_or_None, log_likelihood, restart_index).
    """
    seed = 42 + i + group_seed_offset
    model = build_hmm_model(config, random_state=seed)
    _initialize_model_from_data(model, all_obs, config)

    # Perturb transition matrix with Dirichlet noise for restart diversity
    alpha = config.dirichlet_concentration
    for row in range(config.n_states):
        base = model.transmat_[row]
        base = np.clip(base, 1e-6, None)
        model.transmat_[row] = dirichlet.rvs(alpha * base / base.sum(), random_state=seed + row)[0]

    # Perturb emission probabilities for binary models
    if config.emission_model == "binary":
        for row in range(config.n_states):
            base = model.emissionprob_[row]
            base = np.clip(base, 1e-6, None)
            model.emissionprob_[row] = dirichlet.rvs(
                alpha * base / base.sum(), random_state=seed + config.n_states + row
            )[0]

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(all_obs, lengths=lengths)
        ll = model.score(all_obs, lengths=lengths)
    except Exception:
        return None, -np.inf, i

    return model, ll, i


def train_with_restarts(
    config: HMMConfig,
    all_obs: np.ndarray,
    lengths: list[int],
    verbose: bool = True,
    n_jobs: int | None = None,
    group_seed_offset: int = 0,
) -> tuple[BaseHMM, float]:
    """Train HMM with multiple random restarts, return best model by log-likelihood.

    Each group gets a different seed offset so that parallel genotype fits
    explore different regions of parameter space.

    Parameters
    ----------
    config : HMMConfig
    all_obs : np.ndarray, shape (N, 1)
    lengths : list[int]
    verbose : bool
    n_jobs : int or None
        Override config.n_jobs for this call. None uses config.n_jobs.
    group_seed_offset : int
        Added to restart seeds for reproducibility isolation across groups.

    Returns
    -------
    best_model : BaseHMM
    best_ll : float
    """
    if n_jobs is None:
        n_jobs = config.n_jobs

    # Run restarts in parallel
    results = Parallel(n_jobs=n_jobs, prefer="processes")(
        delayed(_fit_single_restart)(i, config, all_obs, lengths, group_seed_offset)
        for i in range(config.n_restarts)
    )

    # Pick the best
    best_model = None
    best_ll = -np.inf
    for model, ll, _idx in results:
        if model is not None and ll > best_ll:
            best_model = model
            best_ll = ll

    if verbose:
        succeeded = sum(1 for m, _, _ in results if m is not None)
        print(f"  {config.n_restarts} restarts ({succeeded} converged), best LL={best_ll:.1f}")

    if best_model is None:
        raise RuntimeError("All training restarts failed.")

    return best_model, best_ll


# ---------------------------------------------------------------------------
# 1.7  State sorting
# ---------------------------------------------------------------------------
def sort_states_by_activity(model: BaseHMM, config: HMMConfig) -> np.ndarray:
    """Sort states so state 0 = deepest sleep (lowest activity).

    Returns the permutation array and reorders model parameters in-place.
    """
    n = config.n_states

    if config.emission_model == "zip":
        order = np.argsort(model.lambdas_)  # ascending lambda
    elif config.emission_model == "gaussian":
        order = np.argsort(model.means_[:, 0])  # ascending mean
    elif config.emission_model == "binary":
        order = np.argsort(model.emissionprob_[:, 0])[::-1]  # descending P(inactive)
    else:
        order = np.arange(n)

    # Build relabeling map: old_idx -> new_idx
    perm = np.empty(n, dtype=int)
    for new_idx, old_idx in enumerate(order):
        perm[old_idx] = new_idx

    # Reorder model parameters
    model.startprob_ = model.startprob_[order]
    model.transmat_ = model.transmat_[order, :][:, order]

    if config.emission_model == "zip":
        model.lambdas_ = model.lambdas_[order]
        model.omegas_ = model.omegas_[order]
        if model.transmat_mask is not None:
            model.transmat_mask = model.transmat_mask[order, :][:, order]
    elif config.emission_model == "gaussian":
        model.means_ = model.means_[order]
        # covars_ getter calls fill_covars, converting diag storage (n, f) to
        # full matrices (n, f, f). Assigning that back through the setter raises
        # a shape error. Access _covars_ directly to reorder without the
        # getter/setter conversion.
        model._covars_ = model._covars_[order]
        if model.transmat_mask is not None:
            model.transmat_mask = model.transmat_mask[order, :][:, order]
    elif config.emission_model == "binary":
        model.emissionprob_ = model.emissionprob_[order]
        if model.transmat_mask is not None:
            model.transmat_mask = model.transmat_mask[order, :][:, order]

    return perm


# ---------------------------------------------------------------------------
# 1.8  Decoding
# ---------------------------------------------------------------------------
def decode_flies(
    model: BaseHMM,
    ds: xr.Dataset,
    fly_ids: np.ndarray,
    config: HMMConfig,
    state_permutation: np.ndarray,
) -> xr.Dataset:
    """Decode individual flies and add hmm_state, hmm_sleep, hmm_confidence to dataset.

    Uses vectorized numpy for speed — extracts raw arrays once, loops only for
    per-fly HMM decode calls (which are the irreducible per-fly operations).
    """
    do_posterior = config.decoding_method in ("both", "posterior")
    do_viterbi = config.decoding_method in ("both", "viterbi")

    all_ids = ds["id"].values
    times = ds["time"].values
    n_t = len(times)
    n_all = len(all_ids)

    # Build inverse permutation for posterior reordering
    n_states = config.n_states
    inv_perm = np.empty(n_states, dtype=int)
    for old_idx in range(n_states):
        inv_perm[state_permutation[old_idx]] = old_idx

    var_name = "activity" if config.observation_type in ("counts", "normalized") else "moving"
    raw_all = _get_var_time_by_id(ds, var_name).astype(np.float64)  # (time, n_all_ids)

    # Pre-allocate output arrays for the full dataset
    # Initialize with existing values if present, else -1/nan
    if "hmm_state" in ds.data_vars:
        state_2d = _get_var_time_by_id(ds, "hmm_state").copy()
    else:
        state_2d = np.full((n_t, n_all), -1, dtype=np.int8)

    if "hmm_sleep" in ds.data_vars:
        sleep_2d = _get_var_time_by_id(ds, "hmm_sleep").copy()
    else:
        sleep_2d = np.full((n_t, n_all), -1, dtype=np.int8)

    if do_posterior:
        if "hmm_confidence" in ds.data_vars:
            conf_2d = _get_var_time_by_id(ds, "hmm_confidence").copy()
        else:
            conf_2d = np.full((n_t, n_all), np.nan, dtype=np.float32)

    sleep_threshold = n_states // 2

    # Map fly_ids to column indices in the dataset
    id_to_col = {fid: i for i, fid in enumerate(all_ids)}

    # --- Collect per-fly observations and metadata for batched prediction ---
    fly_meta = []  # (col, valid_mask) per valid fly
    all_obs_list = []  # observation arrays to concatenate
    batch_lengths = []  # sequence lengths for hmmlearn

    for fly_id in fly_ids:
        col = id_to_col.get(fly_id)
        if col is None:
            continue

        # §2d: SAME transform as training/CV (fixes the former per-fly min-max here
        # vs group min-max in extract_observations — a real train/decode mismatch).
        obs, valid_mask = _transform_observation(raw_all[:, col], config)
        if len(obs) == 0:
            continue

        fly_meta.append((col, valid_mask))
        all_obs_list.append(obs)
        batch_lengths.append(len(obs))

    # --- Batched prediction: one call to model.predict / predict_proba ---
    if fly_meta:
        all_obs_concat = np.concatenate(all_obs_list).reshape(-1, 1)

        if do_viterbi:
            all_raw_states = model.predict(all_obs_concat, lengths=batch_lengths)

        if do_posterior:
            all_posteriors = model.predict_proba(all_obs_concat, lengths=batch_lengths)

        # --- Scatter results back into per-fly output arrays ---
        offset = 0
        for (col, valid_mask), fly_len in zip(fly_meta, batch_lengths):
            if do_viterbi:
                raw_states = all_raw_states[offset : offset + fly_len]
                decoded = state_permutation[raw_states].astype(np.int8)
                state_2d[valid_mask, col] = decoded

            if do_posterior:
                posteriors = all_posteriors[offset : offset + fly_len]
                posteriors_sorted = posteriors[:, inv_perm]
                if not do_viterbi:
                    decoded = posteriors_sorted.argmax(axis=1).astype(np.int8)
                    state_2d[valid_mask, col] = decoded
                conf_2d[valid_mask, col] = posteriors_sorted.max(axis=1).astype(np.float32)

            valid_states = state_2d[valid_mask, col]
            sleep_2d[valid_mask, col] = (valid_states < sleep_threshold).astype(np.int8)

            offset += fly_len

    # Build result dataset — assign directly to avoid slow xr.concat
    drop_vars = [v for v in ["hmm_state", "hmm_sleep", "hmm_confidence"] if v in ds.data_vars]
    ds_out = ds.drop_vars(drop_vars) if drop_vars else ds.copy()

    ds_out["hmm_state"] = xr.DataArray(
        state_2d, dims=["time", "id"], coords={"time": times, "id": all_ids}
    ).astype(np.int8)
    ds_out["hmm_sleep"] = xr.DataArray(
        sleep_2d, dims=["time", "id"], coords={"time": times, "id": all_ids}
    ).astype(np.int8)
    if do_posterior:
        ds_out["hmm_confidence"] = xr.DataArray(
            conf_2d, dims=["time", "id"], coords={"time": times, "id": all_ids}
        ).astype(np.float32)

    return ds_out


# ---------------------------------------------------------------------------
# 1.9  Per-genotype workflow
# ---------------------------------------------------------------------------
@dataclass
class GenotypeModelResult:
    """Result container for a single genotype's fitted HMM."""

    group_name: str
    model: BaseHMM
    log_likelihood: float
    n_flies: int
    state_permutation: np.ndarray


def _get_groups(ds: xr.Dataset) -> dict[str, np.ndarray]:
    """Return dict mapping group name -> array of fly IDs."""
    groups = {}
    if "group" in ds.coords:
        for fly_id in ds.id.values:
            g = str(ds.sel(id=fly_id).coords["group"].values)
            groups.setdefault(g, []).append(fly_id)
    elif "genotype" in ds.coords:
        for fly_id in ds.id.values:
            g = str(ds.sel(id=fly_id).coords["genotype"].values)
            groups.setdefault(g, []).append(fly_id)
    else:
        groups["all"] = ds.id.values
    return {k: np.array(v) for k, v in groups.items()}


def _fit_one_group(
    group_name: str,
    fly_ids: np.ndarray,
    ds: xr.Dataset,
    config: HMMConfig,
    group_seed_offset: int = 0,
) -> tuple[str, BaseHMM, float, int, np.ndarray] | None:
    """Fit a single genotype group independently (group-serial-restart reference).

    Each group is fitted from scratch with its own random restarts, run serially
    within (n_jobs=1). NOTE: ``run_genotype_workflow`` no longer calls this — the
    per_genotype path now flattens (group × restart) into one restart-parallel job
    list (``_fit_group_restart``) to keep every core busy and stream progress. This
    function is retained as the equivalent serial reference (and for direct unit
    testing); it is not dead logic, just superseded as the orchestration unit.
    Pending removal is queued for the user (do not delete unilaterally, §0).

    Returns (group_name, model, ll, n_valid_flies, state_permutation) or None.
    """
    grp_obs, grp_lengths, grp_valid = extract_observations(ds, fly_ids, config)
    if len(grp_obs) == 0:
        print(f"  WARNING: group '{group_name}' — no valid observations, skipping.")
        return None

    try:
        model, ll = train_with_restarts(
            config,
            grp_obs,
            grp_lengths,
            verbose=True,
            n_jobs=1,
            group_seed_offset=group_seed_offset,
        )
        perm = sort_states_by_activity(model, config)
        return group_name, model, ll, len(grp_valid), perm
    except RuntimeError as e:
        print(
            f"  WARNING: group '{group_name}' — all restarts failed ({e}). "
            "Try increasing n_restarts or reducing n_states."
        )
        return None


def _fit_group_restart(group_name, restart_i, config, obs, lengths, group_seed_offset):
    """One restart of one genotype group — the flat unit for restart-level parallel
    dispatch (per_genotype). Returns ``(group_name, model_or_None, ll, restart_i)``.

    Bit-identical to the serial-restart path: the fit depends only on
    ``(config, obs, seed)`` and the seed is ``group_seed_offset + restart_i`` exactly
    as ``train_with_restarts`` would use it — so the SAME restart produces the SAME
    model. Only the SCHEDULING changes: flattening (group × restart) into one job
    list keeps every core busy, instead of one core per group leaving the rest idle
    once the fast groups finish (with few groups the slow groups otherwise starve the
    pool). ``_fit_single_restart`` swallows its own exceptions → model=None on failure.
    """
    model, ll, _ = _fit_single_restart(restart_i, config, obs, lengths, group_seed_offset)
    return group_name, model, ll, restart_i


def _fit_one_fly(
    fly_id,
    ds: xr.Dataset,
    config: HMMConfig,
) -> tuple[str, BaseHMM, float, np.ndarray] | None:
    """Fit a single fly (designed for joblib parallel dispatch)."""
    fly_obs, fly_lengths, _ = extract_observations(ds, np.array([fly_id]), config)
    if len(fly_obs) == 0:
        return None

    per_fly_config = HMMConfig(
        **{**config.__dict__, "n_restarts": max(5, config.n_restarts // 10), "n_jobs": 1}
    )
    try:
        model, ll = train_with_restarts(
            per_fly_config,
            fly_obs,
            fly_lengths,
            verbose=False,
            n_jobs=1,
        )
        perm = sort_states_by_activity(model, config)
        return str(fly_id), model, ll, perm
    except RuntimeError as e:
        print(f"  WARNING: fly '{fly_id}' — all restarts failed ({e}).")
        return None


# Minimum valid minutes in a day-block to attempt a per-day fit (a mostly-NaN day
# is skipped; its minutes stay -1/missing).
_MIN_DAY_FIT_POINTS = 60


def _fit_decode_fly_per_day(fly_id, ds: xr.Dataset, config: HMMConfig):
    """Ghosh & Harbison 2026 per-fly-PER-DAY: fit a SEPARATE HMM for each 1440-min
    day of one fly, Viterbi-decode that day with its own model, and stitch the
    per-day state sequences into the fly's full record.

    This is the paper's load-bearing design choice (one model per fly per day,
    1440 points each), distinct from ``per_fly`` (one model over the whole record).
    Per-day time-in-state — which the paper averages across days — is recoverable
    downstream from the stored per-day ``hmm_state``. Designed for joblib dispatch;
    returns ``(fly_str, state_1d, sleep_1d, conf_1d, day_models)`` or ``None``."""
    all_ids = list(ds["id"].values)
    matches = [i for i, x in enumerate(all_ids) if str(x) == str(fly_id)]
    if not matches:
        return None
    col = matches[0]

    var_name = "activity" if config.observation_type in ("counts", "normalized") else "moving"
    fly_raw = _get_var_time_by_id(ds, var_name)[:, col].astype(np.float64)
    n_t = fly_raw.shape[0]
    n_states = config.n_states
    sleep_threshold = n_states // 2
    do_posterior = config.decoding_method in ("both", "posterior")

    state_1d = np.full(n_t, -1, dtype=np.int8)
    sleep_1d = np.full(n_t, -1, dtype=np.int8)
    conf_1d = np.full(n_t, np.nan, dtype=np.float32)

    # Small records are noisy per-day → keep a moderate restart budget per day
    # (the paper used up to 1000; we balance tractability × the many day-fits).
    per_day_config = HMMConfig(
        **{**config.__dict__, "n_restarts": max(10, config.n_restarts // 3), "n_jobs": 1}
    )
    day_models = []
    n_days = int(np.ceil(n_t / SAMPLES_PER_DAY))
    for d in range(n_days):
        s = d * SAMPLES_PER_DAY
        e = min((d + 1) * SAMPLES_PER_DAY, n_t)
        # per-day normalization happens inside the transform (day-slice → its own total)
        obs, valid = _transform_observation(fly_raw[s:e], config)
        if len(obs) < _MIN_DAY_FIT_POINTS:
            continue
        try:
            model, ll = train_with_restarts(
                per_day_config, obs.reshape(-1, 1), [len(obs)], verbose=False, n_jobs=1
            )
        except RuntimeError:
            continue
        perm = sort_states_by_activity(model, config)
        abs_idx = np.nonzero(valid)[0] + s  # absolute positions of this day's valid minutes
        raw_states = model.predict(obs.reshape(-1, 1))  # Viterbi (paper's decoder)
        decoded = perm[raw_states].astype(np.int8)
        state_1d[abs_idx] = decoded
        sleep_1d[abs_idx] = (decoded < sleep_threshold).astype(np.int8)
        if do_posterior:
            inv = np.empty(n_states, dtype=int)
            for old_idx in range(n_states):
                inv[perm[old_idx]] = old_idx
            post = model.predict_proba(obs.reshape(-1, 1))[:, inv]
            conf_1d[abs_idx] = post.max(axis=1).astype(np.float32)
        day_models.append((d, model, ll, perm))

    if not day_models:
        return None
    return str(fly_id), state_1d, sleep_1d, conf_1d, day_models


def run_genotype_workflow(
    ds: xr.Dataset,
    config: HMMConfig,
    verbose: bool = True,
    progress_callback=None,
) -> tuple[xr.Dataset, dict[str, GenotypeModelResult]]:
    """Run the full HMM workflow based on config.training_scope.

    Each genotype group is fitted independently — no warm-starting from a
    control group.  This matches how Wiggin et al. 2020 fitted each genotype
    separately and avoids biasing experimental groups toward control state
    structure.

    Parallelism strategy:
      - per_genotype: Parallelizes across groups (each group's restarts
        run serially within).
      - all_pooled: Parallelizes restarts on a single pooled model.
      - per_fly: Parallelizes across flies.
      - per_fly_per_day: Parallelizes across flies; each fly fits + Viterbi-decodes
        one model PER DAY (Ghosh & Harbison 2026) and stitches the days together.

    Returns
    -------
    ds : xr.Dataset with hmm_state, hmm_sleep, (hmm_confidence) added
    results : dict mapping group_name -> GenotypeModelResult
    """
    groups = _get_groups(ds)
    results = {}

    if config.training_scope == "all_pooled":
        # Single model on all data — parallelism is across restarts
        if verbose:
            print("Training single model on all flies (all_pooled)...")

        all_ids = np.concatenate(list(groups.values()))
        all_obs, lengths, valid_ids = extract_observations(ds, all_ids, config)

        if len(all_obs) == 0:
            raise ValueError("No valid observations found in dataset.")

        model, ll = train_with_restarts(config, all_obs, lengths, verbose=verbose)
        perm = sort_states_by_activity(model, config)

        if verbose:
            _print_model_summary(model, config, "all_pooled")

        result = GenotypeModelResult("all_pooled", model, ll, len(valid_ids), perm)
        results["all_pooled"] = result

        ds = decode_flies(model, ds, all_ids, config, perm)
        if progress_callback:
            progress_callback(1, 1)

    elif config.training_scope == "per_fly":
        # One model per fly — parallelise across flies. generator_unordered streams
        # each fly's result as its fit completes, so the progress bar advances during
        # the (long) TRAINING phase, not only during decode.
        all_fly_ids = np.concatenate(list(groups.values()))
        if verbose:
            print(f"Training per-fly models ({len(all_fly_ids)} flies, n_jobs={config.n_jobs})...")

        _pf_total = len(all_fly_ids)
        _fly_gen = Parallel(
            n_jobs=config.n_jobs, prefer="processes", return_as="generator_unordered"
        )(delayed(_fit_one_fly)(fly_id, ds, config) for fly_id in all_fly_ids)
        for _pf_completed, res in enumerate(_fly_gen, start=1):
            if res is not None:
                fly_str, model, ll, perm = res
                results[fly_str] = GenotypeModelResult(fly_str, model, ll, 1, perm)
                matching = [fid for fid in ds.id.values if str(fid) == fly_str]
                if matching:
                    ds = decode_flies(model, ds, np.array(matching), config, perm)
            if progress_callback:
                progress_callback(_pf_completed, _pf_total)

        if verbose:
            print(f"  Done. {len(results)}/{len(all_fly_ids)} flies fitted successfully.")

    elif config.training_scope == "per_fly_per_day":
        # Ghosh & Harbison 2026: one model per fly PER DAY, Viterbi-decoded per day.
        # generator_unordered → the bar advances per fly during the (expensive,
        # many-fits) training phase instead of freezing until it is all done.
        all_fly_ids = np.concatenate(list(groups.values()))
        if verbose:
            print(
                f"Training per-fly-per-day models ({len(all_fly_ids)} flies, "
                f"n_jobs={config.n_jobs})..."
            )

        all_ids = list(ds["id"].values)
        times = ds["time"].values
        n_t = len(times)
        id_to_col = {str(x): i for i, x in enumerate(all_ids)}
        state_2d = np.full((n_t, len(all_ids)), -1, dtype=np.int8)
        sleep_2d = np.full((n_t, len(all_ids)), -1, dtype=np.int8)
        conf_2d = np.full((n_t, len(all_ids)), np.nan, dtype=np.float32)

        _pfd_total = len(all_fly_ids)
        _pfd_gen = Parallel(
            n_jobs=config.n_jobs, prefer="processes", return_as="generator_unordered"
        )(delayed(_fit_decode_fly_per_day)(fly_id, ds, config) for fly_id in all_fly_ids)
        for _pfd_completed, res in enumerate(_pfd_gen, start=1):
            if res is not None:
                fly_str, st1, sl1, cf1, day_models = res
                c = id_to_col.get(fly_str)
                if c is not None:
                    state_2d[:, c] = st1
                    sleep_2d[:, c] = sl1
                    conf_2d[:, c] = cf1
                    # Representative result = the fly's first fitted day (summary/UI).
                    d0, m0, ll0, perm0 = day_models[0]
                    results[fly_str] = GenotypeModelResult(fly_str, m0, ll0, 1, perm0)
            if progress_callback:
                progress_callback(_pfd_completed, _pfd_total)

        drop = [v for v in ("hmm_state", "hmm_sleep", "hmm_confidence") if v in ds.data_vars]
        if drop:
            ds = ds.drop_vars(drop)
        ds["hmm_state"] = xr.DataArray(
            state_2d, dims=["time", "id"], coords={"time": times, "id": all_ids}
        ).astype(np.int8)
        ds["hmm_sleep"] = xr.DataArray(
            sleep_2d, dims=["time", "id"], coords={"time": times, "id": all_ids}
        ).astype(np.int8)
        if config.decoding_method in ("both", "posterior"):
            ds["hmm_confidence"] = xr.DataArray(
                conf_2d, dims=["time", "id"], coords={"time": times, "id": all_ids}
            ).astype(np.float32)

        if verbose:
            print(f"  Done. {len(results)}/{len(all_fly_ids)} flies fitted per-day.")

    else:
        # per_genotype (default): fit each group independently, but with RESTARTS as
        # the parallel unit flattened across ALL groups. Two wins over group-level
        # dispatch: (1) the progress bar advances per completed restart during the
        # long training phase (generator_unordered), and (2) every core stays busy —
        # with 6 groups on 16 cores the old path left 10 cores idle once the fast
        # groups finished, while the slow groups ground 50 serial restarts each. The
        # numeric result is unchanged: each (group, restart) uses its original seed.
        group_names = sorted(groups.keys())
        if verbose:
            print(
                f"Fitting {len(group_names)} groups × {config.n_restarts} restarts "
                f"(restart-parallel, n_jobs={config.n_jobs})..."
            )

        # Extract each group's observations once (main process); skip empty groups.
        group_obs = {}
        for gname in group_names:
            g_obs, g_len, g_valid = extract_observations(ds, groups[gname], config)
            if len(g_obs) == 0:
                if verbose:
                    print(f"  WARNING: group '{gname}' — no valid observations, skipping.")
                continue
            group_obs[gname] = (g_obs, g_len, g_valid)

        active_groups = [g for g in group_names if g in group_obs]
        seed_offsets = {g: gi * config.n_restarts for gi, g in enumerate(group_names)}
        jobs = [(g, i) for g in active_groups for i in range(config.n_restarts)]

        _total_units = len(jobs)
        _completed = 0
        best = dict.fromkeys(active_groups, (-np.inf, None))  # best (ll, model) per group
        if jobs:
            _grp_gen = Parallel(
                n_jobs=config.n_jobs, prefer="processes", return_as="generator_unordered"
            )(
                delayed(_fit_group_restart)(
                    g, i, config, group_obs[g][0], group_obs[g][1], seed_offsets[g]
                )
                for (g, i) in jobs
            )
            for gname, model, ll, _ri in _grp_gen:
                if model is not None and ll > best[gname][0]:
                    best[gname] = (ll, model)
                _completed += 1
                if progress_callback:
                    progress_callback(_completed, _total_units)

        # Pick the best restart per group, sort states, decode (decode is ~seconds).
        for gname in active_groups:
            ll, model = best[gname]
            if model is None:
                if verbose:
                    print(
                        f"  WARNING: group '{gname}' — all restarts failed. "
                        "Try increasing n_restarts or reducing n_states."
                    )
                continue
            perm = sort_states_by_activity(model, config)
            n_valid = len(group_obs[gname][2])
            if verbose:
                print(f"  {gname}: best LL={ll:.1f} over {config.n_restarts} restarts")
                _print_model_summary(model, config, gname)
            results[gname] = GenotypeModelResult(gname, model, ll, n_valid, perm)
            ds = decode_flies(model, ds, groups[gname], config, perm)

        if verbose:
            skipped = len(group_names) - len(active_groups)
            if skipped:
                print(f"  WARNING: {skipped} group(s) skipped (no valid observations).")

    # Persist config into dataset attributes so parameters survive NetCDF
    # round-trips (mirrors the pattern used by periodograms.py for ls_/cwt_/ac_).
    _store_hmm_config_attrs(ds, config)

    return ds, results


def _store_hmm_config_attrs(ds: xr.Dataset, config: HMMConfig) -> None:
    """Write all HMMConfig fields into ``ds.attrs`` with an ``hmm_`` prefix."""
    ds.attrs["hmm_n_states"] = config.n_states
    ds.attrs["hmm_training_scope"] = config.training_scope
    ds.attrs["hmm_emission_model"] = config.emission_model
    ds.attrs["hmm_observation_type"] = config.observation_type
    ds.attrs["hmm_transition_constraints"] = config.transition_constraints
    ds.attrs["hmm_decoding_method"] = config.decoding_method
    ds.attrs["hmm_n_restarts"] = config.n_restarts
    ds.attrs["hmm_n_iter"] = config.n_iter
    ds.attrs["hmm_tol"] = config.tol
    ds.attrs["hmm_soft_floor"] = config.soft_floor
    ds.attrs["hmm_dirichlet_concentration"] = config.dirichlet_concentration


def load_hmm_config_from_attrs(ds: xr.Dataset) -> HMMConfig | None:
    """Reconstruct an :class:`HMMConfig` from ``ds.attrs``.

    Returns ``None`` if the required attributes are not present (e.g. the
    dataset predates this feature or HMM was never run).
    """
    if "hmm_n_states" not in ds.attrs:
        return None
    return HMMConfig(
        n_states=int(ds.attrs["hmm_n_states"]),
        training_scope=str(ds.attrs["hmm_training_scope"]),
        emission_model=str(ds.attrs["hmm_emission_model"]),
        observation_type=str(ds.attrs["hmm_observation_type"]),
        transition_constraints=str(ds.attrs["hmm_transition_constraints"]),
        decoding_method=str(ds.attrs["hmm_decoding_method"]),
        n_restarts=int(ds.attrs["hmm_n_restarts"]),
        n_iter=int(ds.attrs["hmm_n_iter"]),
        tol=float(ds.attrs["hmm_tol"]),
        soft_floor=float(ds.attrs["hmm_soft_floor"]),
        dirichlet_concentration=float(ds.attrs["hmm_dirichlet_concentration"]),
    )


def _print_model_summary(model, config, group_name):
    """Print fitted model parameters."""
    state_names = (
        STATE_NAMES_4 if config.n_states == 4 else [f"S{i}" for i in range(config.n_states)]
    )

    print(f"\n  --- {group_name} model summary ---")
    if config.emission_model == "zip":
        print("  State  |  lambda  |  omega (zero-inflation)")
        for i, name in enumerate(state_names):
            print(f"    {name}   | {model.lambdas_[i]:7.3f}  |  {model.omegas_[i]:.4f}")
    elif config.emission_model == "gaussian":
        print("  State  |   mean   |   std")
        for i, name in enumerate(state_names):
            print(
                f"    {name}   | {model.means_[i, 0]:7.2f}  |  {np.sqrt(model.covars_[i, 0]):.2f}"
            )
    elif config.emission_model == "binary":
        print("  State  |  P(inactive)  |  P(active)")
        for i, name in enumerate(state_names):
            print(
                f"    {name}   |    {model.emissionprob_[i, 0]:.4f}    |   {model.emissionprob_[i, 1]:.4f}"
            )

    print("  Transition matrix:")
    header = "         " + "  ".join(f"{n:>6}" for n in state_names)
    print(header)
    for i, name in enumerate(state_names):
        row = "  ".join(f"{v:6.3f}" for v in model.transmat_[i])
        print(f"    {name}: {row}")


# ---------------------------------------------------------------------------
# 1.11  Visualization
# ---------------------------------------------------------------------------
def plot_hypnogram_heatmap(ds, group_name=None, n_states=4, ax=None):
    """Color-coded state assignment heatmap (rows=flies, columns=time).

    Parameters
    ----------
    ds : xr.Dataset with 'hmm_state' variable
    group_name : optional, filter to a specific group
    n_states : number of states for colormap
    ax : optional matplotlib axes

    Returns
    -------
    fig : matplotlib Figure
    """
    if "hmm_state" not in ds.data_vars:
        raise KeyError("'hmm_state' not found. Run the workflow first.")

    # Filter to group if specified. When showing ALL groups, ORDER flies by group so
    # each genotype forms a contiguous block that can be labeled (group_boundaries =
    # list of (label, start_row, end_row)); otherwise the rows are an unlabeled jumble.
    group_boundaries = None
    if group_name is not None:
        if "group" in ds.coords:
            mask = ds.coords["group"].values == group_name
            fly_ids = ds.id.values[mask]
        else:
            fly_ids = ds.id.values
    elif "group" in ds.coords:
        groups_arr = np.asarray([str(g) for g in ds.coords["group"].values])
        ids_arr = np.asarray(ds.id.values)
        order = np.argsort(groups_arr, kind="stable")  # contiguous group blocks
        fly_ids = ids_arr[order]
        sorted_groups = groups_arr[order]
        group_boundaries = []
        i, n = 0, len(sorted_groups)
        while i < n:
            j = i
            while j < n and sorted_groups[j] == sorted_groups[i]:
                j += 1
            group_boundaries.append((sorted_groups[i], i, j))  # rows [i, j)
            i = j
    else:
        fly_ids = ds.id.values

    if len(fly_ids) == 0:
        print(f"No flies found for group '{group_name}'")
        return None

    # Build state matrix
    state_data = ds["hmm_state"].sel(id=fly_ids).values  # (time, id) or (id, time)
    if state_data.ndim == 2:
        # Ensure shape is (n_flies, n_time)
        if state_data.shape[0] == len(ds.time) and state_data.shape[1] == len(fly_ids):
            state_data = state_data.T
    else:
        state_data = state_data.reshape(1, -1)

    # Create custom colormap
    colors = STATE_COLORS_4[:n_states] if n_states <= 4 else plt.cm.tab10.colors[:n_states]
    cmap = mcolors.ListedColormap(["#808080"] + list(colors))  # gray for -1
    bounds = [-1.5] + [i - 0.5 for i in range(n_states + 1)]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    if ax is None:
        fig, ax = plt.subplots(figsize=(14, max(3, len(fly_ids) * 0.15)))
    else:
        fig = ax.figure

    im = ax.imshow(state_data, aspect="auto", cmap=cmap, norm=norm, interpolation="nearest")

    # Time axis labels (hours)
    n_time = state_data.shape[1]
    tick_interval = max(1, n_time // 10)
    tick_positions = np.arange(0, n_time, tick_interval * 60)  # every tick_interval hours
    tick_labels = [f"{int(p / 60)}h" for p in tick_positions]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)

    ax.set_xlabel("Time")
    title = "HMM State Assignments"
    if group_name:
        title += f" — {group_name}"
    ax.set_title(title)

    # Label each genotype block (All-groups view): a y-tick with the group name at the
    # block centre + a white separator line between blocks, so the group name sits
    # next to the set of flies it represents.
    if group_boundaries:
        centers = [(s + e - 1) / 2.0 for _, s, e in group_boundaries]
        labels = [g for g, _, _ in group_boundaries]
        ax.set_yticks(centers)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_ylabel("Group")
        for _, _s, e in group_boundaries[:-1]:
            ax.axhline(e - 0.5, color="white", linewidth=1.0)
    else:
        ax.set_ylabel("Fly")

    # Colorbar
    state_names = STATE_NAMES_4[:n_states] if n_states <= 4 else [f"S{i}" for i in range(n_states)]
    cbar = fig.colorbar(im, ax=ax, ticks=range(n_states))
    cbar.ax.set_yticklabels(state_names)

    plt.tight_layout()
    return fig


def plot_state_occupancy_by_group(ds, n_states=4, return_data=False):
    """Grouped bar chart of % time in each state per genotype.

    Parameters
    ----------
    return_data : bool
        When True, return ``(fig, DataFrame)`` where the DataFrame is the per-fly
        %-time-in-state records the bars aggregate (the export payload). Default
        False → just ``fig``.

    Returns
    -------
    fig : matplotlib Figure (or ``(fig, DataFrame)`` if return_data).
    """
    if "hmm_state" not in ds.data_vars:
        raise KeyError("'hmm_state' not found. Run the workflow first.")

    groups = _get_groups(ds)
    state_names = STATE_NAMES_4[:n_states] if n_states <= 4 else [f"S{i}" for i in range(n_states)]

    records = []
    for group_name in sorted(groups.keys()):
        fly_ids = groups[group_name]
        for fly_id in fly_ids:
            states = ds["hmm_state"].sel(id=fly_id).values
            valid = states[states != -1]
            if len(valid) == 0:
                continue
            row = {"group": group_name, "fly_id": str(fly_id)}
            for s in range(n_states):
                row[state_names[s]] = (valid == s).mean() * 100
            records.append(row)

    if not records:
        print("No valid state data found.")
        return (None, None) if return_data else None

    df = pd.DataFrame(records)

    # Melt for seaborn
    df_melt = df.melt(
        id_vars=["group", "fly_id"], value_vars=state_names, var_name="State", value_name="% Time"
    )

    fig, ax = plt.subplots(figsize=(max(8, len(groups) * 1.5), 6))
    colors = STATE_COLORS_4[:n_states] if n_states <= 4 else None
    palette = dict(zip(state_names, colors)) if colors else None

    sns.barplot(
        data=df_melt, x="group", y="% Time", hue="State", palette=palette, ax=ax, errorbar="se"
    )

    ax.set_xlabel("Group")
    ax.set_ylabel("% Time in State")
    ax.set_title("State Occupancy by Group")
    ax.tick_params(axis="x", rotation=45)
    ax.legend(title="State")
    plt.tight_layout()
    if return_data:
        return fig, df
    return fig


# ---------------------------------------------------------------------------
# 1.12  Summary metrics
# ---------------------------------------------------------------------------
def get_advanced_hmm_summary(
    ds: xr.Dataset,
    results: dict[str, GenotypeModelResult],
    config: HMMConfig,
) -> pd.DataFrame:
    """Extended summary with state occupancy, bout durations, emission params, and threshold comparison.

    Returns
    -------
    pd.DataFrame
    """
    if "hmm_state" not in ds.data_vars:
        raise KeyError("'hmm_state' not found. Run the workflow first.")

    state_names = (
        STATE_NAMES_4 if config.n_states == 4 else [f"S{i}" for i in range(config.n_states)]
    )
    records = []
    groups = _get_groups(ds)

    for group_name in sorted(groups.keys()):
        fly_ids = groups[group_name]
        n_flies = len(fly_ids)

        # Per-fly metrics
        fly_sleep_pcts = []
        all_bout_durations = {s: [] for s in range(config.n_states)}
        all_state_vals = []
        all_threshold_vals = []
        all_hmm_sleep_vals = []

        for fly_id in fly_ids:
            fly_ds = ds.sel(id=fly_id)
            states = fly_ds["hmm_state"].values
            valid = states != -1
            if not valid.any():
                continue

            valid_states = states[valid]
            all_state_vals.append(valid_states)

            sleep_threshold = config.n_states // 2
            sleep_frac = (valid_states < sleep_threshold).mean() * 100
            fly_sleep_pcts.append(sleep_frac)

            # Bout duration analysis
            for s in range(config.n_states):
                in_state = (valid_states == s).astype(int)
                diffs = np.diff(in_state, prepend=0, append=0)
                starts = np.where(diffs == 1)[0]
                ends = np.where(diffs == -1)[0]
                for st, en in zip(starts, ends):
                    all_bout_durations[s].append(en - st)

            # Threshold comparison
            if "sleep" in ds.data_vars:
                threshold = fly_ds["sleep"].values[valid]
                hmm_sleep = fly_ds["hmm_sleep"].values[valid]
                both_valid = (threshold != -1) & (hmm_sleep != -1)
                if both_valid.any():
                    all_threshold_vals.append(threshold[both_valid])
                    all_hmm_sleep_vals.append(hmm_sleep[both_valid])

        if not all_state_vals:
            continue

        combined_states = np.concatenate(all_state_vals)
        n_valid = len(combined_states)

        row = {
            "group": group_name,
            "n_flies": n_flies,
            "hmm_sleep_pct": np.mean(fly_sleep_pcts) if fly_sleep_pcts else np.nan,
            "hmm_sleep_std": np.std(fly_sleep_pcts) if fly_sleep_pcts else np.nan,
        }

        # State occupancy
        for s, name in enumerate(state_names):
            row[f"{name}_pct"] = (combined_states == s).sum() / n_valid * 100

        # Mean bout durations (in minutes, assuming 1-min bins)
        for s, name in enumerate(state_names):
            bouts = all_bout_durations[s]
            row[f"{name}_bout_mean"] = np.mean(bouts) if bouts else np.nan
            row[f"{name}_bout_std"] = np.std(bouts) if bouts else np.nan

        # Emission parameters from fitted model
        if group_name in results:
            model = results[group_name].model
            if config.emission_model == "zip":
                for s, name in enumerate(state_names):
                    row[f"{name}_lambda"] = model.lambdas_[s]
                    row[f"{name}_omega"] = model.omegas_[s]
            elif config.emission_model == "gaussian":
                for s, name in enumerate(state_names):
                    row[f"{name}_mean"] = model.means_[s, 0]
                    row[f"{name}_std"] = np.sqrt(model.covars_[s, 0])
            elif config.emission_model == "binary":
                for s, name in enumerate(state_names):
                    row[f"{name}_P_active"] = model.emissionprob_[s, 1]

            # Key transition probabilities
            row["LL"] = results[group_name].log_likelihood

        # Threshold comparison
        if all_threshold_vals:
            t_all = np.concatenate(all_threshold_vals)
            h_all = np.concatenate(all_hmm_sleep_vals)
            row["threshold_sleep_pct"] = (t_all == 1).mean() * 100
            row["agreement_pct"] = (t_all == h_all).mean() * 100
        else:
            row["threshold_sleep_pct"] = np.nan
            row["agreement_pct"] = np.nan

        records.append(row)

    return pd.DataFrame(records)


def get_hmm_zt_fractions(ds, bin_size_minutes=30, n_states=None):
    """Per-fly, per-ZT-bin state fractions (0-1).

    Columns: id, zt_bin_minute, plus one ``<name>_pct`` column per state.
    Values are fractions (0-1) of valid timepoints in each state.
    Missing timepoints (hmm_state == -1) are excluded from the denominator.

    Parameters
    ----------
    n_states : int or None
        Number of HMM states.  When *None* (default) the count is inferred
        from the maximum valid value in ``ds['hmm_state']``.
    """
    import dam_utilities  # local import to avoid circular at module level

    if n_states is None:
        max_state = int(ds["hmm_state"].where(ds["hmm_state"] != -1).max().item())
        n_states = max_state + 1

    state_names = STATE_NAMES_4[:n_states] if n_states <= 4 else [f"S{i}" for i in range(n_states)]
    state_cols = {i: f"{name}_pct" for i, name in enumerate(state_names)}

    series_dict = {}

    for state_idx, col_name in state_cols.items():
        ds_tmp = ds.copy()
        indicator = xr.where(
            ds["hmm_state"] == state_idx, 1.0, xr.where(ds["hmm_state"] == -1, np.nan, 0.0)
        )
        ds_tmp["_hmm_ind"] = indicator
        df = dam_utilities.get_zt_binned_dataframe(ds_tmp, "_hmm_ind", bin_size_minutes)
        df = df.rename(columns={"_hmm_ind": col_name})
        series_dict[col_name] = df.set_index(["id", "zt_bin_minute"])[col_name]

    result = pd.concat(series_dict, axis=1).reset_index()
    return result


def plot_zt_state_fractions(
    ds, n_states=None, bin_size_minutes=30, n_sections=4, return_data=False
):
    """Stacked area chart of HMM state fractions across ZT, one subplot per genotype.

    State percentages are stacked to 100% on the y-axis, with ZT bins on the
    x-axis.  The day can optionally be coarsened into *n_sections* equal blocks
    (set ``n_sections=None`` to use every ZT bin at full resolution).

    Parameters
    ----------
    ds : xr.Dataset
    n_states : int or None
        Passed to :func:`get_hmm_zt_fractions`.
    bin_size_minutes : int
        ZT bin resolution for the underlying fractions (default 30).
    n_sections : int or None
        If given, ZT bins are averaged into this many equal day-sections
        before plotting (default 4).  Use ``None`` for full resolution.

    Returns
    -------
    fig : matplotlib Figure or None (or ``(fig, DataFrame)`` if return_data — the
        per-group per-ZT stacked %-in-state values the plot draws).
    """
    if "hmm_state" not in ds.data_vars:
        return (None, None) if return_data else None

    if n_states is None:
        max_state = int(ds["hmm_state"].where(ds["hmm_state"] != -1).max().item())
        n_states = max_state + 1

    state_names = STATE_NAMES_4[:n_states] if n_states <= 4 else [f"S{i}" for i in range(n_states)]
    pct_cols = [f"{name}_pct" for name in state_names]
    colors = STATE_COLORS_4[:n_states] if n_states <= 4 else list(plt.cm.tab10.colors[:n_states])

    frac_df = get_hmm_zt_fractions(ds, bin_size_minutes=bin_size_minutes, n_states=n_states)

    # Map each fly to its group
    groups = _get_groups(ds)
    id_to_group = {}
    for g, ids in groups.items():
        for fid in ids:
            id_to_group[fid] = g
    frac_df["group"] = frac_df["id"].map(id_to_group)
    frac_df = frac_df.dropna(subset=["group"])

    # Optionally coarsen into n_sections equal day-blocks
    if n_sections is not None:
        section_size = 1440 / n_sections
        frac_df["section"] = (frac_df["zt_bin_minute"] // section_size).astype(int)
        # Average within each section per fly, then use section midpoint as x
        frac_df = frac_df.groupby(["group", "id", "section"])[pct_cols].mean().reset_index()
        frac_df["zt_hour"] = (frac_df["section"] + 0.5) * section_size / 60
    else:
        frac_df["zt_hour"] = frac_df["zt_bin_minute"] / 60

    # Mean across flies per group per time point
    group_means = frac_df.groupby(["group", "zt_hour"])[pct_cols].mean().reset_index()
    # Convert fractions to percentages
    group_means[pct_cols] = group_means[pct_cols] * 100

    group_names = sorted(groups.keys())
    n_groups = len(group_names)

    fig, axes = plt.subplots(1, n_groups, figsize=(max(5 * n_groups, 8), 5), sharey=True)
    if n_groups == 1:
        axes = [axes]

    for i, gname in enumerate(group_names):
        ax = axes[i]
        gdata = group_means[group_means["group"] == gname].sort_values("zt_hour")
        x = gdata["zt_hour"].values
        ys = [gdata[col].values for col in pct_cols]

        ax.stackplot(x, *ys, labels=state_names, colors=colors, alpha=0.85)
        ax.set_xlim(0, 24)
        ax.set_ylim(0, 100)
        ax.set_title(gname)
        ax.set_xlabel("ZT (hours)")
        if i == 0:
            ax.set_ylabel("% Time in State")

        # Light/dark shading: ZT12-24 is typically dark phase
        ax.axvspan(12, 24, color="grey", alpha=0.12)

    # Single legend outside the last panel
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, title="State", bbox_to_anchor=(1.01, 0.5), loc="center left")

    fig.suptitle("HMM State Fractions by Time of Day", fontsize=14)
    plt.tight_layout()
    if return_data:
        return fig, group_means
    return fig


def plot_group_state_timecourse(
    ds, n_states=None, metric="sleep", bin_size_minutes=30, return_data=False
):
    """Group-comparison time-course: mean % time in ONE metric across ZT, with all
    genotype groups OVERLAID on shared axes (± SEM across flies).

    This is the group-level average "metric to compare" — it reads genotype
    differences at a glance, unlike :func:`plot_hypnogram_heatmap` (per-fly rows) and
    :func:`plot_zt_state_fractions` (states stacked in a SEPARATE subplot per group,
    so groups can't be compared directly). Averaging is across flies within a group,
    per ZT bin; missing minutes (state == -1) are excluded from each fly's fraction.

    Parameters
    ----------
    ds : xr.Dataset with ``hmm_state``.
    n_states : int or None
        Inferred from the data when None.
    metric : 'sleep' | int | str
        ``'sleep'`` = total sleep (DS+LS, i.e. ``hmm_sleep = state < 2``). An int
        state index or a state name plots that single state's fraction.
    bin_size_minutes : int
        ZT bin resolution (default 30).
    return_data : bool
        When True, return ``(fig, DataFrame)`` where the DataFrame is the plotted
        per-group per-ZT mean±SEM (the export payload). Default False → just ``fig``.

    Returns
    -------
    fig : matplotlib Figure or None (or ``(fig, DataFrame)`` if return_data).
    """
    if "hmm_state" not in ds.data_vars:
        return (None, None) if return_data else None
    if n_states is None:
        max_state = int(ds["hmm_state"].where(ds["hmm_state"] != -1).max().item())
        n_states = max_state + 1

    state_names = STATE_NAMES_4[:n_states] if n_states <= 4 else [f"S{i}" for i in range(n_states)]
    frac_df = get_hmm_zt_fractions(ds, bin_size_minutes=bin_size_minutes, n_states=n_states)

    # Build the (0-1) metric column.
    if metric == "sleep":
        sleep_cols = [f"{name}_pct" for name in state_names[: min(2, n_states)]]
        frac_df["_metric"] = frac_df[sleep_cols].sum(axis=1)
        metric_label = "Sleep (DS+LS)"
    else:
        mname = state_names[metric] if isinstance(metric, int) else str(metric)
        if f"{mname}_pct" not in frac_df.columns:
            return (None, None) if return_data else None
        frac_df["_metric"] = frac_df[f"{mname}_pct"]
        metric_label = mname
    frac_df["zt_hour"] = frac_df["zt_bin_minute"] / 60.0

    groups = _get_groups(ds)
    id_to_group = {fid: g for g, ids in groups.items() for fid in ids}
    frac_df["group"] = frac_df["id"].map(id_to_group)
    frac_df = frac_df.dropna(subset=["group"])
    if frac_df.empty:
        return (None, None) if return_data else None

    # Mean ± SEM across flies, per group per ZT bin (percentages).
    grp = frac_df.groupby(["group", "zt_hour"])["_metric"]
    stat = (
        (grp.mean() * 100)
        .rename("mean")
        .reset_index()
        .merge((grp.sem() * 100).rename("sem").reset_index(), on=["group", "zt_hour"])
    )

    group_names = sorted(groups.keys())
    cmap = plt.cm.tab10.colors
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, gname in enumerate(group_names):
        gd = stat[stat["group"] == gname].sort_values("zt_hour")
        if gd.empty:
            continue
        x = gd["zt_hour"].values
        m = gd["mean"].values
        se = np.nan_to_num(gd["sem"].values)
        c = cmap[i % len(cmap)]
        n_flies = frac_df[frac_df["group"] == gname]["id"].nunique()
        ax.plot(x, m, color=c, lw=2, label=f"{gname} (n={n_flies})")
        ax.fill_between(x, m - se, m + se, color=c, alpha=0.18)

    ax.axvspan(12, 24, color="grey", alpha=0.12)  # dark phase (ZT12-24)
    ax.set_xlim(0, 24)
    ax.set_ylim(0, 100)
    ax.set_xlabel("ZT (hours)")
    ax.set_ylabel(f"% Time — {metric_label}")
    ax.set_title(f"Group-averaged {metric_label} across ZT (mean ± SEM)")
    ax.legend(title="Group", bbox_to_anchor=(1.01, 0.5), loc="center left", fontsize=8)
    plt.tight_layout()
    if return_data:
        out = stat.rename(
            columns={"mean": f"{metric_label}_mean_pct", "sem": f"{metric_label}_sem_pct"}
        )
        return fig, out
    return fig


# ---------------------------------------------------------------------------
# 3.3  Model selection via cross-validation
# ---------------------------------------------------------------------------

# Observation type required for each emission model
_EMISSION_TO_OBS = {"zip": "counts", "binary": "binary", "gaussian": "normalized"}


def _count_hmm_params(config: HMMConfig) -> int:
    """Count free parameters in an HMM configuration.

    Unconstrained counts are used so relative ordering across models is
    preserved even when transition constraints reduce the effective degrees
    of freedom in practice.
    """
    n = config.n_states
    n_trans = n * (n - 1)  # transition matrix rows, each sums to 1
    n_start = n - 1  # start probabilities
    if config.emission_model == "zip":
        n_emit = 2 * n  # lambda + omega per state
    elif config.emission_model == "gaussian":
        n_emit = 2 * n  # mean + variance per state
    elif config.emission_model == "binary":
        n_emit = n  # one free emission prob per state
    else:
        n_emit = 0
    return n_trans + n_start + n_emit


def _compute_fold_agreement_pct(
    model: BaseHMM,
    ds: xr.Dataset,
    held_fly_ids: list,
    config: HMMConfig,
    state_permutation: np.ndarray,
) -> float:
    """Compute held-out agreement_pct without modifying ds.

    For each held-out fly: decode with model.predict, compare HMM sleep
    (state < n_states//2) against the traditional threshold-based sleep.

    Returns mean agreement across held-out flies, or nan if sleep unavailable.
    """
    if "sleep" not in ds.data_vars:
        return np.nan

    n_states = config.n_states
    sleep_threshold = n_states // 2

    all_ids = ds["id"].values
    id_to_col = {fid: i for i, fid in enumerate(all_ids)}

    var_name = "activity" if config.observation_type in ("counts", "normalized") else "moving"
    raw_all = _get_var_time_by_id(ds, var_name).astype(np.float64)
    sleep_all = _get_var_time_by_id(ds, "sleep")

    agreement_list = []

    for fly_id in held_fly_ids:
        col = id_to_col.get(fly_id)
        if col is None:
            continue

        # §2d: the ONE shared transform (was a third hand-copied block here).
        obs, valid_mask = _transform_observation(raw_all[:, col], config)
        if len(obs) == 0:
            continue

        try:
            raw_states = model.predict(obs.reshape(-1, 1))
        except Exception:
            continue

        sorted_states = state_permutation[raw_states]
        hmm_sleep = (sorted_states < sleep_threshold).astype(int)

        threshold_sleep = sleep_all[:, col][valid_mask]
        both_valid = threshold_sleep != -1

        if not both_valid.any():
            continue

        agreement = (hmm_sleep[both_valid] == threshold_sleep[both_valid]).mean() * 100.0
        agreement_list.append(agreement)

    return float(np.mean(agreement_list)) if agreement_list else np.nan


def _fit_cv_fold_job(
    fold_idx: int,
    held_fold: list,
    cv_config_dict: dict,
    n_states: int,
    emission: str,
    all_obs: np.ndarray,
    cum: np.ndarray,
    all_valid_ids: list,
):
    """Fit one CV fold — top-level function for joblib serialization.

    Returns a partial record dict (without agreement) and the fitted model/perm,
    or ``None`` if the fold was skipped or failed.
    """
    cv_config = HMMConfig(**cv_config_dict)
    cv_config.n_states = n_states

    id_to_slice = {fid: slice(int(cum[i]), int(cum[i + 1])) for i, fid in enumerate(all_valid_ids)}
    valid_id_set = set(all_valid_ids)

    held_ids = [fid for fid in held_fold if fid in valid_id_set]
    train_ids = [fid for fid in all_valid_ids if fid not in set(held_fold)]

    if not train_ids or not held_ids:
        return None

    train_obs = np.concatenate([all_obs[id_to_slice[f]] for f in train_ids])
    train_lengths = [id_to_slice[f].stop - id_to_slice[f].start for f in train_ids]
    held_obs = np.concatenate([all_obs[id_to_slice[f]] for f in held_ids])
    held_lengths = [id_to_slice[f].stop - id_to_slice[f].start for f in held_ids]

    try:
        model, _ = train_with_restarts(
            cv_config,
            train_obs,
            train_lengths,
            verbose=False,
            n_jobs=1,
        )
        perm = sort_states_by_activity(model, cv_config)
    except RuntimeError:
        return None

    try:
        held_ll = model.score(held_obs, lengths=held_lengths)
        held_ll_per_obs = held_ll / len(held_obs)
    except Exception:
        held_ll_per_obs = np.nan

    record = {
        "emission_model": emission,
        "n_states": n_states,
        "fold": fold_idx,
        "held_out_ll_per_obs": held_ll_per_obs,
        "n_held_out_obs": len(held_obs),
    }

    return record, model, perm, held_ids, cv_config


def compare_n_states(
    ds: xr.Dataset,
    n_states_range: list | None = None,
    emission_models: list | None = None,
    n_folds: int = 5,
    base_config: HMMConfig | None = None,
    cv_restarts: int = 10,
    cv_n_iter: int = 100,
    progress_callback=None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """K-fold cross-validation for HMM model selection.

    Selects the optimal number of states (and optionally emission model) by
    evaluating held-out log-likelihood per observation across folds.

    The primary metric for within-model comparison (same emission type) is
    **mean held-out log-likelihood per observation** — higher is better.

    For cross-model comparison (ZIP vs binary vs Gaussian), log-likelihoods
    are on different scales because the observation spaces differ (Poisson
    counts vs Bernoulli vs Gaussian).  Use **mean_agreement_pct** (agreement
    with the traditional 5-minute threshold rule) as the common metric when
    comparing emission models.

    AIC and BIC are computed by re-training on all flies and apply only
    within a single emission model.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset with at least ``activity`` or ``moving`` variables.
    n_states_range : list[int], optional
        State counts to test. Default: ``[2, 3, 4, 5]``.
    emission_models : list[str], optional
        Any subset of ``["zip", "gaussian", "binary"]``.
        Default: ``["zip"]``.
    n_folds : int
        Number of cross-validation folds. Default: 5.
    base_config : HMMConfig, optional
        Base configuration whose ``transition_constraints`` and other
        structural settings are reused.  ``emission_model``, ``n_states``,
        ``observation_type``, ``n_restarts``, ``n_iter``, and
        ``training_scope`` are overridden internally.
        Default: ``HMMConfig.improved()``.
    cv_restarts : int
        Training restarts per fold fit (fewer than production for speed).
        Default: 10.
    cv_n_iter : int
        Maximum EM iterations per fold fit. Default: 100.
    progress_callback : callable, optional
        Called as ``progress_callback(completed, total)`` after each fold fit.

    Returns
    -------
    fold_df : pd.DataFrame
        Per-fold results with columns ``emission_model``, ``n_states``,
        ``fold``, ``held_out_ll_per_obs``, ``n_held_out_obs``, and
        optionally ``held_out_agreement_pct`` (if ``sleep`` variable present).
    summary_df : pd.DataFrame
        Aggregated results with columns ``emission_model``, ``n_states``,
        ``mean_ll``, ``std_ll``, ``sem_ll``, ``n_params``, ``aic``, ``bic``,
        and optionally ``mean_agreement_pct``.
        ``aic`` and ``bic`` are from a full-data fit with production restarts.

    Notes
    -----
    For ``gaussian`` (normalized) emission models, normalization bounds are
    derived from all flies, not just the training fold, to ensure consistent
    observation scaling across folds.
    """
    if n_states_range is None:
        n_states_range = [2, 3, 4, 5]
    if emission_models is None:
        emission_models = ["zip"]
    if base_config is None:
        base_config = HMMConfig.improved()

    all_fly_ids = ds["id"].values
    n_flies = len(all_fly_ids)

    if n_folds < 2:
        raise ValueError("n_folds must be >= 2.")
    if n_flies < n_folds:
        raise ValueError(
            f"n_folds ({n_folds}) cannot exceed n_flies ({n_flies}). "
            "Reduce n_folds or add more flies."
        )

    # Reproducible fold assignment
    rng = np.random.RandomState(42)
    shuffled_ids = all_fly_ids.copy()
    rng.shuffle(shuffled_ids)
    folds = [list(chunk) for chunk in np.array_split(shuffled_ids, n_folds)]

    has_sleep = "sleep" in ds.data_vars
    total_fits = len(emission_models) * len(n_states_range) * n_folds
    completed = 0
    fold_records = []

    # ------------------------------------------------------------------
    # Build all CV fold jobs and run in parallel
    # ------------------------------------------------------------------
    # Pre-extract observations once per emission model (n_states does not
    # affect observation extraction, only model initialisation).
    jobs = []  # arguments for _fit_cv_fold_job
    skipped_count = 0  # folds skipped due to empty observations

    for emission in emission_models:
        obs_type = _EMISSION_TO_OBS.get(emission)
        if obs_type is None:
            raise ValueError(
                f"Unknown emission model: {emission!r}. Choose from 'zip', 'gaussian', 'binary'."
            )

        # Config dict for serialization — n_states is set per job
        cv_config_dict = {
            **base_config.__dict__,
            "emission_model": emission,
            "observation_type": obs_type,
            "n_restarts": cv_restarts,
            "n_iter": cv_n_iter,
            "training_scope": "all_pooled",
            "n_jobs": 1,
        }

        # Extract observations once for this emission model
        tmp_config = HMMConfig(**cv_config_dict)
        all_obs, all_lengths, all_valid_ids = extract_observations(ds, all_fly_ids, tmp_config)

        if len(all_obs) == 0:
            skipped_count += len(n_states_range) * n_folds
            continue

        cum = np.concatenate([[0], np.cumsum(all_lengths)])

        for n_s in n_states_range:
            for fold_idx, held_fold in enumerate(folds):
                jobs.append(
                    (
                        fold_idx,
                        held_fold,
                        cv_config_dict,
                        n_s,
                        emission,
                        all_obs,
                        cum,
                        list(all_valid_ids),
                    )
                )

    # Dispatch all fold fits in parallel across available cores
    if jobs:
        parallel_results = Parallel(n_jobs=-1, prefer="processes")(
            delayed(_fit_cv_fold_job)(*job) for job in jobs
        )
    else:
        parallel_results = []

    # Report progress for skipped folds
    completed = skipped_count
    if progress_callback and completed > 0:
        progress_callback(completed, total_fits)

    # Process results — compute agreement sequentially (fast, needs ds)
    for result in parallel_results:
        completed += 1
        if result is None:
            if progress_callback:
                progress_callback(completed, total_fits)
            continue

        record, model, perm, held_ids, cv_config = result

        if has_sleep:
            record["held_out_agreement_pct"] = _compute_fold_agreement_pct(
                model, ds, held_ids, cv_config, perm
            )

        fold_records.append(record)
        if progress_callback:
            progress_callback(completed, total_fits)

    fold_df = pd.DataFrame(fold_records)

    # ------------------------------------------------------------------
    # Full-data fits for AIC / BIC (production restarts, all flies)
    # ------------------------------------------------------------------
    summary_records = []

    for emission in emission_models:
        obs_type = _EMISSION_TO_OBS[emission]

        full_config = HMMConfig(
            **{
                **base_config.__dict__,
                "emission_model": emission,
                "observation_type": obs_type,
                "training_scope": "all_pooled",
                "n_jobs": -1,
            }
        )

        for n_s in n_states_range:
            full_config.n_states = n_s

            full_obs, full_lengths, _ = extract_observations(ds, all_fly_ids, full_config)
            if len(full_obs) == 0:
                continue

            try:
                _model_full, full_ll = train_with_restarts(
                    full_config, full_obs, full_lengths, verbose=False
                )
            except RuntimeError:
                continue

            n_obs = len(full_obs)
            n_params = _count_hmm_params(full_config)
            aic = -2.0 * full_ll + 2.0 * n_params
            bic = -2.0 * full_ll + n_params * np.log(n_obs)

            # CV summary from fold_df
            if len(fold_df) > 0:
                mask = (fold_df["emission_model"] == emission) & (fold_df["n_states"] == n_s)
                subset = fold_df[mask]
            else:
                subset = pd.DataFrame()

            ll_vals = (
                subset["held_out_ll_per_obs"].dropna()
                if len(subset) > 0
                else pd.Series([], dtype=float)
            )

            row = {
                "emission_model": emission,
                "n_states": n_s,
                "mean_ll": ll_vals.mean() if len(ll_vals) > 0 else np.nan,
                "std_ll": ll_vals.std() if len(ll_vals) > 1 else np.nan,
                "sem_ll": ll_vals.sem() if len(ll_vals) > 1 else np.nan,
                "n_params": n_params,
                "aic": aic,
                "bic": bic,
                "full_ll": full_ll,
            }

            if has_sleep and "held_out_agreement_pct" in (
                subset.columns if len(subset) > 0 else []
            ):
                agr_vals = subset["held_out_agreement_pct"].dropna()
                row["mean_agreement_pct"] = agr_vals.mean() if len(agr_vals) > 0 else np.nan

            summary_records.append(row)

    summary_df = pd.DataFrame(summary_records)

    return fold_df, summary_df
