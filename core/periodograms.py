"""
periodograms.py
===============
Spectral and rhythmicity analysis of Drosophila activity data.

Implements three complementary methods:

  1. Lomb-Scargle (LS)
       lomb_scargle_analysis() -- handles unevenly-sampled or gappy time series;
       uses astropy LombScargle with optional smoothing and detrending.

  2. Continuous Wavelet Transform (CWT)
       wavelet_analysis() -- time-frequency analysis with
       auto-preprocessing (detrending, envelope normalization).
       GPU-accelerated via PyTorch/ptwt when available; falls back to CPU PyWavelets.
       compute_single_fly_scalogram() -- recompute one fly's scalogram on demand.

  3. Autocorrelation (AC)
       autocorrelation_analysis() -- scipy autocorrelogram with optional
       smoothing; extracts dominant period and Rhythm Strength (RS).

All three functions accept an xr.Dataset and return an augmented Dataset with
per-fly result variables (ls_period, cwt_period, ac_rhythm_strength, etc.).

Time representation:
  Both relative integer-minute and absolute datetime time are supported.
  Functions check np.issubdtype(dtype, np.integer) to branch accordingly.
"""

import multiprocessing as mp
import os
import warnings

import numpy as np
import pywt
import xarray as xr
from astropy.timeseries import LombScargle
from scipy import signal

# MESA (Maximum Entropy Spectral Analysis) uses the Burg autoregressive method.
# statsmodels is a hard dependency (also used elsewhere); guard the import so the
# other three period methods still load if it is somehow absent.
try:
    from statsmodels.regression.linear_model import burg as _sm_burg

    STATSMODELS_AVAILABLE = True
except Exception:  # pragma: no cover - statsmodels is in requirements
    _sm_burg = None
    STATSMODELS_AVAILABLE = False

from tqdm import tqdm

# Fix OpenMP duplicate library issue on Windows (must be set before importing torch/numpy/scipy)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# GPU Support - PyTorch/ptwt for GPU-accelerated CWT
import sys as _sys

# Only print GPU status messages in the main process, not worker processes
_is_main_process = mp.current_process().name == "MainProcess"

try:
    import ptwt
    import torch

    PTWT_AVAILABLE = torch.cuda.is_available()
    if _is_main_process:
        if PTWT_AVAILABLE:
            print(
                "GPU-accelerated CWT enabled via PyTorch Wavelet Toolbox (ptwt)", file=_sys.stderr
            )
        else:
            print("PyTorch/ptwt available but no CUDA GPU detected - using CPU", file=_sys.stderr)
except ImportError:
    torch = None
    ptwt = None
    PTWT_AVAILABLE = False
    if _is_main_process:
        print(
            "GPU CWT not available - using CPU-only PyWavelets. Install pytorch and ptwt for GPU acceleration.",
            file=_sys.stderr,
        )


# Recommended (NOT enforced) minimum DD days for period analysis — a soft,
# user-editable FLAG (not a filter): flies below it are surfaced with their
# record length, never dropped. DEFINED ONCE in calibrations.py (the documented
# home for all user-owned soft thresholds) and re-exported here so callers that
# read ``periodograms.DEFAULT_MIN_DD_DAYS_FLOOR`` (page 3) keep working off the
# single source. Full provenance lives in calibrations.py; edit the value there.
# Default CWT reduction method — single source of truth (calibrations.py).
# wavelet_analysis resolves cwt_method=None -> this, so there is no hardcoded
# literal default copy here. Full provenance (the ar1 -> global_rednoise flip)
# lives in calibrations.py.
# Gap-size cutoff X (small vs large intra-record gap). Single source of truth in
# calibrations.py. Re-exported here so the splitter and its callers read ONE
# value. STAGED: the block extractor accepts this so the small/large boundary is
# wired and user-owned, but it does NOT yet bridge ≤X gaps — bridging a small gap
# requires a CWT-only transient interpolation/fill that does NOT exist downstream
# (the CWT/AC workers both produce NaN on internal-NaN input; LS doesn't route
# through the splitter at all). That fill is DEFERRED pending the user's
# post-distribution decision (see GAP_SIZE_DISTRIBUTION.md). Full provenance in
# calibrations.py.
# Gap-bridge CEILING — the ONE knob for what the CWT/AC block extractor bridges.
# Single source of truth in calibrations.py. A gap whose missing span is ≤ this
# ceiling is BRIDGED (worker-local linear interpolation of the interior NaN
# cells) when bridge_small_gaps=True (wired in the CWT + AC workers below); a gap
# LARGER than the ceiling stays a HARD break → longest-clean-segment. This ONE
# ceiling collapses the former tiny-gap (5 min) bridge AND the previously-deferred
# 5–60 min gray-zone fill. The interpolated values are
# TRANSIENT/worker-local — never written to the Dataset/.nc (§2a). 0 = bridging
# OFF (size-blind longest run). Full provenance in calibrations.py.
from calibrations import (
    DEFAULT_CWT_METHOD,  # noqa: E402
    DEFAULT_GAP_THRESHOLD_MINUTES,  # noqa: F401,E402 (re-export)
    DEFAULT_MAX_BRIDGE_GAP_MINUTES,  # noqa: E402
    DEFAULT_MIN_DD_DAYS_FLOOR,  # noqa: F401,E402 (re-export)
)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# CWT period-grid density (voices per octave) — SINGLE SOURCE OF TRUTH
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# The CWT period grid is a logarithmic (per-octave) scale grid, not a fixed
# number of bins: `resolution = 1 / voices_per_octave` sets how many scales are
# placed in each doubling of period. The grid spans [min_period, max_period]
# (user-facing search range), so the *density* (voices/octave) is the thing that
# controls period discrimination, independent of how wide the range is.
#
# WHY 32 (USER DECISION 2026-06-30, was 10): 32 voices/octave gives a period
# resolution of 2^(1/32) ~= 2.2% per step (~0.5 h spacing near 24 h, ~0.9 h near
# 43 h) — finer period READOUT than the prior 10 v/oct (~7.2%/step, ~1.7 h) while
# still far below the 512 the original code used (~64x over-resolved for the
# circadian band; Torrence & Compo 1998 BAMS §3f call dj=0.125 = 8 v/oct the
# typical fine value, so 32 is comfortably fine, not over-resolved, at ~3x the
# scale count of 10). VERIFIED resolution-stable 10->32 (the readout finer, the
# CALLS unchanged): on the §2c dev cohort 0 classification flips and per-fly
# period median |delta|=0.16 h; the real long-period line (monitor 2044, ~43 h)
# holds detection at both (10 v/oct 43.24 h / 32 v/oct 42.88 h, 31/31 rhythmic).
# Resolution is an INVALIDATION TRIGGER on the CWT verified rows (re-check calls
# /periods if this changes). 1/voices_per_octave = the resolution arg below.
DEFAULT_CWT_VOICES_PER_OCTAVE = 32
DEFAULT_CWT_RESOLUTION = 1.0 / DEFAULT_CWT_VOICES_PER_OCTAVE

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Default CWT period-SEARCH range — ONE knob, single-sourced in calibrations.py.
# Re-exported here so callers reading ``periodograms.DEFAULT_CWT_MIN_PERIOD`` /
# ``..._MAX_PERIOD`` (page 3, tests) keep working off the single source. The
# CLASSIFY window TRACKS this same range — classify_cwt derives its window from
# the search range stamped on the analysed dataset (cwt_min_period/cwt_max_period),
# so search and classify CANNOT diverge. Default (16, 36) h is the circadian-focused
# conservative choice; a short record degrades gracefully (wavelet_analysis WARNs
# when max_period exceeds the COI-resolvable limit). Genuine long-period lines
# (e.g. L775A ~43 h) read ARRHYTHMIC at this default by design — widen the range to
# detect them; 'global_rednoise' stays range-robust when widened. Full provenance
# (user decision, the search==classify coupling, the long-period tradeoff) lives in
# calibrations.py.
import contextlib

from calibrations import (  # noqa: E402 (re-export)
    DEFAULT_CWT_MAX_PERIOD,
    DEFAULT_CWT_MIN_PERIOD,
)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Time-representation helpers (datetime64 vs integer-minute)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def _time_diff_seconds(time_coords):
    """Return np.diff(time_coords) in float seconds, for both datetime64 and integer-minute time."""
    diffs = np.diff(time_coords)
    if np.issubdtype(time_coords.dtype, np.integer) or np.issubdtype(
        time_coords.dtype, np.floating
    ):
        return diffs.astype(float) * 60.0  # minutes → seconds
    return diffs.astype("timedelta64[s]").astype(float)


def _time_span_seconds(t_start, t_end):
    """Return (t_end - t_start) in float seconds, for both datetime64 and integer-minute time.
    Works with scalars and arrays."""
    diff = t_end - t_start
    diff_arr = np.asarray(diff)
    if np.issubdtype(diff_arr.dtype, np.integer) or np.issubdtype(diff_arr.dtype, np.floating):
        return diff_arr.astype(float) * 60.0  # minutes → seconds
    return diff_arr.astype("timedelta64[s]").astype(float)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Phase selection (LD / DD)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def _select_phase(ds, phase="auto", caller="period analysis"):
    """
    Select LD or DD phase from a dataset for period analysis.

    Thin delegate to :func:`dam_utilities.select_phase` (the one core selector).
    Period analysis defaults to DD (``phase="auto"`` → DD when a per-fly boundary
    is derivable). The selector returns a per-fly **NaN-masked VIEW of the whole
    dataset** — the full time axis with out-of-phase cells set to NaN — rather than
    a physically sliced/re-zeroed copy. The per-fly period workers
    (:func:`_extract_longest_continuous_block`, the LS finite-value path) already
    self-extract their own longest finite block and re-reference time internally,
    so consuming the masked view yields the same periods as the old physical split.

    There is intentionally NO "already split, pass through" branch: an explicit
    ``phase`` request always re-derives from the boundary coordinate, so a period
    call can never silently analyse the wrong phase (the prior foot-gun).

    Period policy (see ``dam_utilities.PHASE_POLICY``): default **DD**
    (free-running circadian period) and **reject** ``"both"`` — a periodogram over
    combined LD-entrained + DD free-running data mixes two rhythms and must fail
    loud rather than silently compute. Resolution + validation are delegated to
    :func:`dam_utilities.resolve_phase`; the masking to :func:`select_phase`.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset, carrying ``split_minute`` and/or ``first_DD_day``.
    phase : str
        ``"auto"`` (→ DD), ``"DD"``, or ``"LD"``. ``"both"`` raises (period
        analysis is single-epoch).
    caller : str
        Analysis name used in error messages.

    Returns
    -------
    analysis_ds : xr.Dataset
        The masked phase view.
    phase_used : str
        The phase actually applied (``"DD"`` or ``"LD"``).
    """
    from dam_utilities import resolve_phase

    return resolve_phase(ds, phase, default="DD", allowed=("LD", "DD"), caller=caller)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# CWT Helper Functions
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def cwt_gpu(data, scales, wavelet="cmor1.5-1.0", sampling_period=1):
    """
    GPU-accelerated CWT using PyTorch Wavelet Toolbox (ptwt).
    Falls back to CPU if GPU is not available.

    This provides TRUE GPU acceleration by running the entire CWT computation
    on the GPU, unlike the previous implementation which only did post-processing.

    Parameters:
    -----------
    data : array-like
        Input signal
    scales : array-like
        Scales for CWT
    wavelet : str
        Wavelet name (default: 'cmor1.5-1.0')
    sampling_period : float
        Sampling period

    Returns:
    --------
    coefficients : ndarray
        CWT coefficients
    frequencies : ndarray
        Corresponding frequencies
    power : ndarray
        Power spectrum (|coefficients|^2 for complex wavelets)
    """
    if PTWT_AVAILABLE and len(data) > 1000:  # Only use GPU for larger datasets
        try:
            # Convert to PyTorch tensors and move to GPU
            data_torch = torch.from_numpy(data.astype(np.float32)).cuda()
            scales_torch = torch.from_numpy(scales.astype(np.float32))

            # Perform CWT on GPU - THIS ACTUALLY RUNS ON GPU!
            coefficients_gpu, frequencies = ptwt.cwt(
                data_torch, scales_torch, wavelet, sampling_period=sampling_period
            )

            # Compute power on GPU
            if "cmor" in wavelet:
                power_gpu = torch.abs(coefficients_gpu) ** 2
            else:
                power_gpu = coefficients_gpu**2

            # Transfer back to CPU as numpy arrays
            coefficients = coefficients_gpu.cpu().numpy()
            power = power_gpu.cpu().numpy()

            # Explicitly delete GPU tensors to free memory immediately
            # This prevents memory accumulation across many flies
            del data_torch, scales_torch, coefficients_gpu, power_gpu
            torch.cuda.empty_cache()

            return coefficients, frequencies, power

        except Exception as e:
            print(f"GPU CWT failed, falling back to CPU: {e}")
            # Clean up any GPU memory that might have been allocated
            with contextlib.suppress(Exception):
                torch.cuda.empty_cache()
            # Fall through to CPU version

    # CPU version using PyWavelets
    coefficients, frequencies = pywt.cwt(data, scales, wavelet, sampling_period=sampling_period)
    if "cmor" in wavelet:
        power = np.abs(coefficients) ** 2
    else:
        power = coefficients**2

    return coefficients, frequencies, power


def get_optimal_workers(n_tasks, use_gpu=False, period_range=None):
    """
    Determine the optimal number of worker processes based on available resources.

    Parameters:
    -----------
    n_tasks : int
        Number of tasks to process
    use_gpu : bool
        Whether GPU acceleration is being used (for CWT via ptwt)
    period_range : tuple (min_period, max_period) or None
        Period range in hours for CWT - larger ranges need fewer workers to avoid GPU memory pressure

    Returns:
    --------
    n_workers : int
        Optimal number of worker processes
    """
    cpu_count = mp.cpu_count()

    if use_gpu and PTWT_AVAILABLE:
        # When using GPU for CWT, reduce CPU workers to avoid memory contention
        # Larger period ranges create larger CWT outputs, requiring more GPU memory per fly

        if period_range is not None:
            min_p, max_p = period_range
            range_size = max_p - min_p

            if range_size > 20:
                # Large period range (e.g., 18-50 = 32 hours)
                # Each fly uses ~100+ MB GPU memory
                # Use only 2 workers to avoid GPU memory thrashing
                n_workers = 2
                if _is_main_process:
                    print(
                        f"  Large period range ({min_p}-{max_p}h) detected - using 2 workers to avoid GPU memory pressure"
                    )
            else:
                # Normal period range (e.g., 18-30 = 12 hours)
                n_workers = min(4, max(2, cpu_count // 2))
        else:
            # No period info, use default
            n_workers = min(4, max(2, cpu_count // 2))
    else:
        # For CPU-only, use more workers but leave some headroom
        n_workers = max(1, cpu_count - 1)

    # Don't create more workers than tasks
    n_workers = min(n_workers, n_tasks)

    return n_workers


from contextlib import contextmanager, suppress


@contextmanager
def _worker_pool(pool, n_processes):
    """Yield a worker pool, transferring lifecycle ownership only when we own it.

    On Windows ``spawn``-mode multiprocessing each worker re-imports the
    analysis stack (numpy/scipy/astropy/torch/ptwt) and initializes a CUDA
    context — about 5 s per worker × 4 workers = ~20 s of cold start before
    any compute happens. When called once per analysis (e.g. page 3) that
    overhead amortizes over all flies. When called once per sweep value
    (e.g. page 11) it multiplies, which is the dominant cost in sweep mode.

    Callers that want to amortize cold start across multiple analysis calls
    can pass a pre-built ``mp.Pool`` via the ``pool=`` kwarg of any of the
    analysis functions; the function will use it for `imap_unordered` but
    will *not* close it. When ``pool=None`` (default), the function spawns
    its own pool and closes it on exit — preserving the prior single-call
    behaviour.
    """
    if pool is None:
        with mp.Pool(processes=n_processes) as owned_pool:
            yield owned_pool
    else:
        yield pool


def _extract_longest_continuous_block(
    activity,
    time_coords,
    gap_threshold_minutes=DEFAULT_GAP_THRESHOLD_MINUTES,
    bridge_small_gaps=False,
    max_bridge_gap_minutes=DEFAULT_MAX_BRIDGE_GAP_MINUTES,
):
    """
     Extract the longest continuous valid block from activity/time arrays.

     A block is considered continuous when:
     1) activity is finite (not NaN), and
     2) adjacent samples are consecutive in index, and
     3) adjacent timestamps follow the expected sampling interval.

     Gap-bridge ceiling — ONE knob
     -----------------------------
     ``max_bridge_gap_minutes`` is the SINGLE bridge CEILING (calibrations.py
     ``DEFAULT_MAX_BRIDGE_GAP_MINUTES``, default 60 min): the max gap the extractor
     will BRIDGE. A gap whose missing span is ≤ this ceiling is joined and its
     interior NaN cell(s) are filled by minimal **linear interpolation**; a gap
     LARGER than the ceiling stays a HARD break → longest-clean-segment. This one
     ceiling collapses the former tiny-gap (5 min) bridge AND the previously-deferred
     5–60 min "gray zone" fill. ``max_bridge_gap_minutes=0``
     turns bridging OFF (nothing is bridgeable → size-blind longest run).
     ``gap_threshold_minutes`` (X) is retained on the signature for the page-1/page-2
     segment splitter's provenance but is NOT the bridge ceiling — the bridge ceiling
     is ``max_bridge_gap_minutes``.

     Two modes:

     - ``bridge_small_gaps=False`` (default): **size-blind**, unchanged and
       bit-identical to the original behavior — split on EVERY gap and return the
       single longest contiguous-finite run. (The ceiling is not acted on.)

     - ``bridge_small_gaps=True``: **gap bridge up to the ceiling.** A gap whose
       missing span is ≤ ``max_bridge_gap_minutes`` is NOT treated as a break — the
       flanking segments are joined and the interior NaN cell(s) are filled by
       minimal **linear interpolation**, so a dropped read does not sever an
       otherwise-clean record. Gaps LARGER than the ceiling remain HARD breaks →
       longest-clean-segment (status quo, L775A-preserving). When the ceiling is 0,
       nothing is bridgeable and this mode is equivalent to the size-blind default.
       The returned block is contiguous-finite on a regular grid.

     §2a transient-fill contract (the reason this is safe)
     -----------------------------------------------------
     The interpolated cells are **transient and worker-local ONLY**: they exist in
     the per-fly worker's local array that is fed to CWT/AC, and are **NEVER written
     back to the xarray Dataset or the saved .nc** — stored truth stays NaN
    . This bridge is wired into the CWT and AC workers (which both
     produce NaN on internal-NaN input, so they need a contiguous-finite block); LS
     is untouched (it never routes through this function — gap-native on the raw
     finite timestamps, and a re-grid/fill there would inject low-frequency bias).
    """
    activity = np.asarray(activity).ravel()
    time_coords = np.asarray(time_coords).ravel()

    if len(activity) != len(time_coords) or len(activity) == 0:
        return None, None, None

    valid_activity = np.isfinite(activity)
    if np.issubdtype(time_coords.dtype, np.datetime64):
        valid_time = ~np.isnat(time_coords)
    else:
        valid_time = np.ones(len(time_coords), dtype=bool)
    valid_mask = valid_activity & valid_time

    valid_indices = np.flatnonzero(valid_mask)
    if len(valid_indices) == 0:
        return None, None, None

    # Estimate expected sampling interval from valid data.
    valid_times = time_coords[valid_indices]
    if len(valid_times) < 2:
        return None, None, None
    deltas_s = _time_diff_seconds(valid_times)
    positive_deltas = deltas_s[deltas_s > 0]
    if len(positive_deltas) == 0:
        return None, None, None
    expected_delta_s = np.median(positive_deltas)
    continuity_tol_s = max(1.0, expected_delta_s * 0.2)

    # Vectorized break detection
    index_gaps = np.diff(valid_indices) != 1
    time_gaps = np.abs(deltas_s - expected_delta_s) > continuity_tol_s
    non_positive = deltas_s <= 0
    breaks = index_gaps | time_gaps | non_positive

    if bridge_small_gaps and float(max_bridge_gap_minutes) > 0:
        # Gap bridge up to the ONE ceiling: a positive forward gap whose MISSING
        # span is ≤ the ceiling is not a break — we will join across it and
        # interpolate its interior NaN cell(s). The missing span = (elapsed between
        # consecutive finite samples) minus one expected sampling interval. Only
        # forward-in-time gaps qualify; a non-positive / out-of-order delta is
        # always a hard break. A ceiling of 0 skips this entirely (bridging OFF →
        # size-blind longest run, the pre-bridge behavior).
        bridge_ceiling_s = float(max_bridge_gap_minutes) * 60.0
        missing_span_s = deltas_s - expected_delta_s
        bridgeable = (deltas_s > 0) & (missing_span_s <= bridge_ceiling_s + continuity_tol_s)
        # Demote bridgeable gaps from "break" to "continue" so the longest run can
        # span them; everything else (over-ceiling, non-positive) stays a break.
        breaks = breaks & ~bridgeable

    # Find break positions and compute run lengths between them
    break_positions = np.flatnonzero(breaks)
    # Boundaries: start of valid_indices, each break, end of valid_indices
    boundaries = np.concatenate([[-1], break_positions, [len(valid_indices) - 1]])
    run_lengths = np.diff(boundaries)
    best_run = np.argmax(run_lengths)
    run_start = boundaries[best_run] + 1
    run_end = boundaries[best_run + 1]

    best_start = valid_indices[run_start]
    best_end = valid_indices[run_end]

    block_activity = activity[best_start : best_end + 1]
    block_time = time_coords[best_start : best_end + 1]

    if bridge_small_gaps and np.any(~np.isfinite(block_activity)):
        # The selected run spans one or more bridged (≤ ceiling) gaps → interior NaN
        # cells remain. Fill them by linear interpolation over the in-block time
        # axis so the block handed to CWT/AC is contiguous-finite on a regular grid.
        # This is the §2a TRANSIENT fill: it lives only in this returned local
        # array and is NEVER persisted to the Dataset/.nc by the callers.
        block_activity = block_activity.astype(float, copy=True)
        finite_mask = np.isfinite(block_activity)
        # x-axis for interpolation: elapsed seconds from block start (monotonic).
        x = _time_span_seconds(block_time[0], block_time)
        block_activity[~finite_mask] = np.interp(
            x[~finite_mask], x[finite_mask], block_activity[finite_mask]
        )

    return block_activity, block_time, expected_delta_s


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Lomb-Scargle Periodogram Analysis
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# astropy ``autopower`` chooses a per-fly frequency grid whose resolution scales
# with that fly's baseline (record length), so flies of different DD durations get
# DIFFERENT native grids. Concatenating those native grids across flies produced a
# sparse union (each fly finite only at its own frequencies, NaN elsewhere), which
# made the Periodograms page's group-average spike. Fix (mirrors AC/MESA/CWT): the
# scalar peak (ls_period/power/fap) is still read from the full-resolution NATIVE
# grid, but the STORED display curve is resampled onto one fixed, log-spaced period
# grid shared byte-identically across flies → clean ``xr.concat``, smooth mean.
LS_N_PERIOD_POINTS = 600  # log-spaced period-grid resolution for the stored curve


def _ls_common_period_grid(min_period, max_period, n=LS_N_PERIOD_POINTS):
    """Fixed log-spaced period axis (hours, float32) shared by every fly's stored
    ``ls_periodogram`` for a given (min,max) — so per-fly Datasets concatenate on one
    byte-identical ``ls_periodogram_periods`` coord (clean ``xr.concat``, no NaN
    union). Display-only; the peak stays on the native autopower grid."""
    return np.geomspace(float(min_period), float(max_period), int(n)).astype(np.float32)


def _run_ls_on_series(
    time_hours: np.ndarray,
    y: np.ndarray,
    *,
    min_period: float,
    max_period: float,
    oversampling: int = 8,
    fap_method: str = "baluev",
):
    """Run a single generalized Lomb-Scargle periodogram on (time, value).

    Pure helper — no xarray, no multiprocessing. Both
    :func:`_ls_single_fly_worker` (production circadian path) and
    :func:`ultradian_rhythmicity_ls` use this so the LS methodology is
    defined exactly once. Z-scoring the input is recommended for
    cross-recording amplitude comparability but is not required for
    correct periodogram computation; preprocessing is the caller's
    responsibility.

    Parameters
    ----------
    time_hours : np.ndarray
        Time axis in hours since start (1-D).
    y : np.ndarray
        Signal values (1-D, finite).
    min_period, max_period : float
        Period range in hours.
    oversampling : int
        Rethomics-style ``ofac`` (astropy's ``samples_per_peak``).
    fap_method : str
        ``'baluev'`` (default), ``'naive'``, or ``'bootstrap'``.

    Returns
    -------
    dict or None
        ``None`` on degenerate input. Otherwise a dict with keys
        ``period``, ``power``, ``fap``, ``frequency``, ``periodogram``.
    """
    if len(y) < 5 or len(time_hours) != len(y):
        return None

    # Constant signal (e.g., dead fly with all-zero activity) → YY=0 →
    # astropy emits "invalid value encountered in divide" and returns NaN
    # powers. Guard upstream so we skip cleanly instead of warning.
    if not np.any(y != y[0]):
        return None

    # Generalized LS (Zechmeister & Kürster 2009): fit_mean=True adds a
    # per-trial-frequency floating constant, robust to gaps and edge-of-
    # window peaks. center_data=True subtracts the global mean before
    # fitting (both are astropy defaults; pinned so a future default
    # change can't silently move results). normalization='standard' is
    # required by the Baluev/naive FAP path. autopower(method='fast')
    # is Press-Rybicki FFT, pinned for the same reason.
    ls = LombScargle(time_hours, y, fit_mean=True, center_data=True, normalization="standard")
    min_freq = 1.0 / max_period
    max_freq = 1.0 / min_period
    frequency, power = ls.autopower(
        minimum_frequency=min_freq,
        maximum_frequency=max_freq,
        samples_per_peak=oversampling,
        nyquist_factor=1,
        method="fast",
    )
    if len(power) == 0:
        return None

    peak_idx = int(np.argmax(power))
    peak_power = float(power[peak_idx])
    try:
        fap = float(ls.false_alarm_probability(peak_power, method=fap_method))
    except Exception:
        fap = float("nan")

    return {
        "period": float(1.0 / frequency[peak_idx]),
        "power": peak_power,
        "fap": fap,
        "frequency": frequency,
        "periodogram": power,
    }


def _ls_single_fly_worker(args):
    """Worker function to run Lomb-Scargle on a single fly's data.

    Consumes an already-preprocessed dataset (preprocessing lives in
    :mod:`preprocessing`). The LS math itself lives in
    :func:`_run_ls_on_series`; this wrapper handles xarray IO,
    longest-block extraction, and result packing.
    """
    (
        fly_data,
        fly_id,
        activity_var,
        time_var,
        min_period,
        max_period,
        oversampling,
        fap_method,
        min_num_days,
    ) = args

    # Extract raw arrays — no DataFrame needed
    fly_data = fly_data[[activity_var, time_var]].squeeze()
    activity = fly_data[activity_var].values.ravel()
    time_coords = fly_data[time_var].values

    # LS handles gappy/unevenly-sampled data natively — use all finite values
    valid = np.isfinite(activity)
    if np.issubdtype(time_coords.dtype, np.datetime64):
        valid &= ~np.isnat(time_coords)
    activity = activity[valid]
    time_coords = time_coords[valid]

    if len(activity) < 5:
        return None

    duration_days = _time_span_seconds(time_coords[0], time_coords[-1]) / 86400.0
    if duration_days < min_num_days:
        return None

    # Convert time to hours since start for Lomb-Scargle
    time_hours = _time_span_seconds(time_coords[0], time_coords) / 3600.0

    ls_result = _run_ls_on_series(
        time_hours,
        activity,
        min_period=min_period,
        max_period=max_period,
        oversampling=oversampling,
        fap_method=fap_method,
    )
    if ls_result is None:
        return None

    ls_period = ls_result["period"]
    ls_power = ls_result["power"]
    ls_fap = ls_result["fap"]
    frequency = ls_result["frequency"]
    power = ls_result["periodogram"]

    # Resample the periodogram onto the shared log-period grid for STORAGE/DISPLAY.
    # The scalar peak (ls_period/power/fap) above is read from the native, full-
    # resolution autopower grid — unchanged production path. The stored curve is
    # interpolated onto one fixed grid so every fly shares a byte-identical
    # 'ls_periodogram_periods' coord: xr.concat then stacks cleanly instead of
    # NaN-padding a per-fly frequency union (which made the group-average spike).
    # astropy's grid is uniform in FREQUENCY; convert to period, sort ascending,
    # then interpolate (np.interp clamps at the window edges — no edge NaN).
    period_grid = _ls_common_period_grid(min_period, max_period)
    native_periods = 1.0 / frequency
    srt = np.argsort(native_periods)
    ls_curve = np.interp(period_grid, native_periods[srt], power[srt])

    # Create the result dataset for this fly
    fly_id_scalar = (
        np.asarray(fly_id).item()
        if np.asarray(fly_id).shape == ()
        else np.asarray(fly_id).ravel()[0]
    )
    result_ds = xr.Dataset(
        {
            # §2b: store per-fly results in float32 (period ~24 h, power R^2 ∈ [0,1]
            # — float32's ~7 sig figs preserve them; calc stays float64 internally).
            "ls_period": (("id",), np.array([ls_period], dtype=np.float32), {"units": "h"}),
            "ls_power": (("id",), np.array([ls_power], dtype=np.float32)),
            # ls_fap is KEPT float64 on purpose: Baluev FAP for strong rhythms can
            # underflow below float32's ~1e-38 floor (it is reported, e.g. as
            # -log10(FAP)); float32 would round such p-values to 0 and lose the
            # "how significant" information. §2b allows a justified float64 store.
            "ls_fap": (
                ("id",),
                np.array([ls_fap], dtype=np.float64),
                {"description": f"Lomb-Scargle false-alarm probability at peak ({fap_method})"},
            ),
            "ls_periodogram": (
                ("id", "ls_periodogram_periods"),
                ls_curve[np.newaxis, :].astype(np.float32),
                {
                    "description": "LS power resampled onto the shared "
                    "log-period grid (display); peak read from native grid"
                },
            ),
        },
        coords={
            "id": np.array([fly_id_scalar]),
            "ls_periodogram_periods": period_grid,
        },
    )
    return result_ds


def lomb_scargle_analysis(
    ds,
    activity_var="activity",
    time_var="time",
    id_var="id",
    min_period=16,
    max_period=32,
    n_processes=None,
    oversampling=8,
    fap_method="baluev",
    min_num_days=0,
    phase="auto",
    progress_callback=None,
    pool=None,
):
    """
    Perform Lomb-Scargle analysis on an already-preprocessed dataset.

    Uses the generalized (Zechmeister & Kürster 2009) periodogram:
    ``fit_mean=True`` fits a floating constant offset per trial frequency,
    which is more robust to gaps, weak rhythms, and edge-of-window peaks
    than the classical Scargle periodogram. Z-scoring the input is
    recommended for cross-recording amplitude comparability but is **not**
    required for correct periodogram computation. Power is reported in
    ``normalization='standard'`` (R²-style ∈ [0, 1]); FAP via Baluev
    (default), naive, or bootstrap. The Rethomics/Press exponential
    ``exp(-P)·N·2/ofac`` is mathematically inconsistent with this
    normalization (it requires Scargle variance-scaled power) and is
    therefore not offered. See ``period_analysis_audit.md`` §4.2.

    Preprocessing (smoothing, detrending, normalization) is external —
    apply :func:`preprocessing.preprocess_activity` before calling this
    function.

    Parameters
    ----------
    oversampling : int
        Period-grid oversampling (Rethomics `ofac`, default 8).
    fap_method : str
        astropy ``LombScargle.false_alarm_probability`` method for
        ``normalization='standard'`` power. ``'baluev'`` (default; Baluev 2008
        MNRAS extreme-value bound, recommended per VanderPlas 2018 ApJS),
        ``'naive'`` (Šidák-corrected Beta tail; standard-norm analogue of
        Scargle), or ``'bootstrap'`` (Monte Carlo; gold standard but slow).
    min_num_days : float
        Minimum required duration (days) of the longest continuous valid block.
    n_processes : int or None
        Number of parallel processes (None = auto-detect optimal).
    phase : str
        ``"auto"`` (default) selects DD if available, otherwise full dataset.
        ``"DD"``, ``"LD"``, or ``"both"`` to force a specific phase.
    """
    # Phase selection — default to DD when available
    analysis_ds, phase_used = _select_phase(ds, phase=phase)
    print("\nPerforming Lomb-Scargle analysis...")
    print(
        f"Data phase: {phase_used}"
        + (
            " (DD available, selected automatically)"
            if phase == "auto" and phase_used == "DD"
            else ""
        )
    )

    # Prepare arguments for parallel processing
    fly_groups = list(analysis_ds.groupby(id_var))
    args_list = [
        (
            group,
            fly_id,
            activity_var,
            time_var,
            min_period,
            max_period,
            oversampling,
            fap_method,
            min_num_days,
        )
        for fly_id, group in fly_groups
    ]

    # Determine optimal number of workers
    if n_processes is None:
        n_processes = get_optimal_workers(len(args_list), use_gpu=PTWT_AVAILABLE)
        print(f"Using {n_processes} workers for Lomb-Scargle analysis")

    # Use multiprocessing to parallelize the computations with per-fly progress.
    # If the caller supplied `pool`, reuse it (and don't close it on exit).
    valid_results = []
    _ls_completed = 0
    _ls_total = len(args_list)
    with _worker_pool(pool, n_processes) as worker_pool:
        for result_ds in tqdm(
            worker_pool.imap_unordered(_ls_single_fly_worker, args_list),
            total=_ls_total,
            desc="Lomb-Scargle Analysis",
            unit="fly",
        ):
            if result_ds is not None:
                valid_results.append(result_ds)
            _ls_completed += 1
            if progress_callback:
                progress_callback(_ls_completed, _ls_total)

    if not valid_results:
        print("Warning: Lomb-Scargle analysis failed for all individuals.")
        return ds

    combined_results = xr.concat(valid_results, dim=id_var)

    # Merge results back into original (unsplit) dataset
    # Drop any existing LS variables so re-runs don't conflict
    existing_ls_vars = [v for v in combined_results.data_vars if v in ds.data_vars]
    existing_ls_coords = [c for c in combined_results.coords if c in ds.coords and c != "id"]
    merged_ds = ds.drop_vars(existing_ls_vars + existing_ls_coords, errors="ignore").merge(
        combined_results
    )
    merged_ds.attrs["ls_min_period"] = min_period
    merged_ds.attrs["ls_max_period"] = max_period
    merged_ds.attrs["ls_oversampling"] = oversampling
    merged_ds.attrs["ls_fap_method"] = fap_method
    merged_ds.attrs["ls_min_num_days"] = min_num_days
    merged_ds.attrs["ls_phase"] = phase_used

    return merged_ds


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Continuous Wavelet Transform (CWT) Analysis (Rethomics-style)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def _ar1_significance_threshold(activity, periods_hours, sampling_rate_min, conf_level=0.95):
    """Torrence & Compo 1998 AR(1) red-noise pointwise significance threshold
    per wavelet scale.

    For complex Morlet (dof=2; T&C Table 2), the pointwise upper threshold at
    scale s is

        T_s = σ² · P_red(f_s) · χ²_{2,conf} / 2                  (T&C eq 18)

    where ``σ² = mean(x²)`` for the mean-centered input, ``P_red(f)`` is the
    AR(1) Fourier spectrum (T&C eq 16),

        P_red(f) = (1 − α²) / (1 + α² − 2α cos 2πf),

    ``α`` is the lag-1 autocorrelation, and ``f`` is the Fourier frequency
    (cycles/sample) corresponding to wavelet scale s. ``power[s, t] > T_s``
    rejects the AR(1) null at the requested confidence level.
    """
    from scipy.stats import chi2

    x = activity - np.mean(activity)
    var_x = float(np.mean(x * x))
    if var_x <= 0 or len(x) < 2:
        return np.full_like(periods_hours, np.inf, dtype=np.float64), 0.0

    sumsq = float(np.sum(x * x))
    alpha = float(np.sum(x[1:] * x[:-1]) / sumsq) if sumsq > 0 else 0.0
    alpha = float(np.clip(alpha, 0.0, 0.999))

    # Wavelet period (h) → samples → Fourier frequency (cycles/sample).
    period_samples = periods_hours * 60.0 / sampling_rate_min
    f_normalized = 1.0 / period_samples

    P_red = (1.0 - alpha**2) / (1.0 + alpha**2 - 2.0 * alpha * np.cos(2.0 * np.pi * f_normalized))
    chi2_crit = float(chi2.ppf(conf_level, df=2))  # ≈ 5.991 for 95%
    threshold_s = var_x * P_red * chi2_crit / 2.0
    return threshold_s, alpha


def _parabolic_peak_period(periods_hours, spectrum, peak_idx):
    """Sub-grid (quadratic) refinement of a spectral peak period — Task D.

    REVERSIBLE ADDITION. The coarse logarithmic period grid (10 voices/octave)
    quantizes the argmax period to the nearest node; near 24 h the global
    wavelet spectrum is a near-flat ~1-octave plateau, so the single-node argmax
    can sit up to ~1 grid step (~1.7 h) off the true peak. Fitting a parabola to
    the peak node and its two immediate neighbours recovers the sub-grid vertex.

    The fit is in **log2(period)** space because the grid is geometric (equal
    log spacing): ``log2(p) = log2(p[peak]) + delta * d``, where ``d`` is the
    log-period grid step and ``delta`` is the sub-grid offset of the vertex,

        delta = 0.5 * (y[-1] - y[+1]) / (y[-1] - 2*y[0] + y[+1]),

    with ``y`` the three spectrum values (T&C-style 3-point quadratic peak
    interpolation; standard in spectral analysis). Falls back to the raw argmax
    period when the peak is at a band edge, a neighbour is non-finite, or the
    parabola is not concave (degenerate) — so it can only refine, never invent.
    """
    n = len(periods_hours)
    if peak_idx <= 0 or peak_idx >= n - 1:
        return float(periods_hours[peak_idx])
    y0 = spectrum[peak_idx]
    ym = spectrum[peak_idx - 1]
    yp = spectrum[peak_idx + 1]
    if not (np.isfinite(y0) and np.isfinite(ym) and np.isfinite(yp)):
        return float(periods_hours[peak_idx])
    denom = ym - 2.0 * y0 + yp
    if denom >= 0:  # not a concave peak (flat or a local min) — keep the node
        return float(periods_hours[peak_idx])
    delta = 0.5 * (ym - yp) / denom
    if not np.isfinite(delta) or abs(delta) > 1.0:  # vertex outside the 3-node span
        return float(periods_hours[peak_idx])
    log2_p = np.log2(periods_hours)
    grid_step = log2_p[peak_idx + 1] - log2_p[peak_idx]
    refined_log2 = log2_p[peak_idx] + delta * grid_step
    return float(2.0**refined_log2)


def _coi_excluded_global_spectrum(power, periods_hours, coi_periods, sampling_rate_min):
    """Build the COI-excluded global wavelet spectrum and locate its peak.

    Shared by every ``global*`` reduction (``global``, ``global_rednoise``,
    ``global_baseline``, ``global_neighbor``): they differ ONLY in how the peak
    is *normalized* to a strength, never in how the spectrum or the peak is
    found — so this single function guarantees identical spectra/peaks across
    them (no drift between the metrics).

    For each scale s (period ``p_s``) the time-average of ``power[s, t]`` runs
    over ONLY the timepoints whose period is resolvable there, i.e.
    ``p_s <= coi_periods[t]`` (a scale is INSIDE the COI / unreliable at t iff
    ``p_s > coi_periods[t]``). Scales with too few COI-free timepoints are set
    NaN so they can neither be picked as the peak nor pollute any background.
    T&C 1998 eq 22 (global wavelet spectrum) restricted to the COI.

    Returns
    -------
    dict with keys ``global_spectrum`` (1D, NaN where unusable), ``peak_idx``
    (argmax over finite scales, or None if the whole band is inside the COI),
    ``peak_period`` (sub-grid parabolic-refined), and ``peak_power``.
    """
    coi_ok = periods_hours[:, np.newaxis] <= coi_periods[np.newaxis, :]  # (scale, time)
    n_coi_free = coi_ok.sum(axis=1)
    # A scale needs at least ~one cycle of COI-free samples (never < 10) to count.
    min_coi_free = np.maximum(10, (periods_hours * 60.0 / sampling_rate_min)).astype(int)
    scale_usable = n_coi_free >= min_coi_free

    power_masked = np.where(coi_ok, power, np.nan)
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        global_spectrum = np.nanmean(power_masked, axis=1)  # (scale,)
    global_spectrum = np.where(scale_usable, global_spectrum, np.nan)

    if not np.any(np.isfinite(global_spectrum)):
        return {
            "global_spectrum": global_spectrum,
            "peak_idx": None,
            "peak_period": float("nan"),
            "peak_power": float("nan"),
        }

    peak_idx = int(np.nanargmax(global_spectrum))
    peak_power = float(global_spectrum[peak_idx])
    # Sub-grid parabolic period refinement (REVERSIBLE; Task D) in log2-period
    # space (the grid is geometric). Falls back to the node period at a band edge.
    peak_period = _parabolic_peak_period(periods_hours, global_spectrum, peak_idx)
    return {
        "global_spectrum": global_spectrum,
        "peak_idx": peak_idx,
        "peak_period": peak_period,
        "peak_power": peak_power,
    }


def _ar1_expected_power_at(activity, period_hours, sampling_rate_min):
    """Torrence & Compo 1998 AR(1) red-noise EXPECTED power at one period.

    ``E[P] = var(x) * P_red(f)`` where ``P_red(f) = (1-a^2)/(1+a^2-2a cos 2pi f)``
    (T&C eq 16), ``a`` is the fitted lag-1 autocorrelation of the mean-centered
    signal, and ``f = 1/period_samples`` is the Fourier frequency. This is the
    *modeled noise floor under the peak* — the denominator of the
    ``global_rednoise`` strength. Returns the float expected power (>= 0), or
    NaN if the signal has no variance. ``activity`` must already be mean-centered
    (as it is inside ``_preprocess_and_compute_cwt``)."""
    x = np.asarray(activity, dtype=float)
    var_x = float(np.mean(x * x))
    if var_x <= 0 or len(x) < 2:
        return float("nan")
    sumsq = float(np.sum(x * x))
    alpha = float(np.sum(x[1:] * x[:-1]) / sumsq) if sumsq > 0 else 0.0
    alpha = float(np.clip(alpha, 0.0, 0.999))
    period_samples = period_hours * 60.0 / sampling_rate_min
    f = 1.0 / period_samples
    P_red = (1.0 - alpha**2) / (1.0 + alpha**2 - 2.0 * alpha * np.cos(2.0 * np.pi * f))
    return float(var_x * P_red)


def _running_median_baseline(spectrum, frac=0.35):
    """Smooth (running-median) baseline of a 1D spectrum, in index space.

    The denominator of the ``global_baseline`` strength: a loess-style smooth
    estimate of the noise floor UNDER each scale, computed as the median of a
    sliding window spanning ``frac`` of the (finite) spectrum. NaN entries
    (COI-masked scales) are skipped within each window. Returns an array the
    same length as ``spectrum`` (NaN only where a window has no finite values).
    """
    spectrum = np.asarray(spectrum, dtype=float)
    n = len(spectrum)
    w = max(3, int(round(n * frac)))
    if w % 2 == 0:
        w += 1
    half = w // 2
    base = np.full(n, np.nan)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        seg = spectrum[lo:hi]
        seg = seg[np.isfinite(seg)]
        if seg.size:
            base[i] = np.median(seg)
    return base


def _preprocess_and_compute_cwt(
    activity,
    time_coords,
    min_period,
    max_period,
    cwt_method="global_rednoise",
    resolution=DEFAULT_CWT_RESOLUTION,
    wavelet="cmor1.5-1.0",
    ar1_conf_level=0.95,
    ar1_gamma=2.32,
):
    """
    Mean-center activity, compute the full 2D CWT, and reduce to a periodogram.

    The reduction method is selected by ``cwt_method`` (see the list below):

    - ``cwt_method='ridge'``: legacy Pythomics ridge tracking. Detects the
      argmax-period at each timepoint, masks the cone of influence and low-
      confidence (peak-to-mean ratio < 2) timepoints, and Savitzky-Golay
      smooths what's left. Reports the median ridge period, ridge std as
      stability, and the valid-fraction as ``cwt_rhythmicity``.

    All methods receive the same input: mean-centered activity. No detrending,
    envelope normalization, or z-scoring is applied to the input signal — these
    were the adaptive preprocessing layer that broke comparability with
    Rethomics/SCAMP and have been removed (see ``period_analysis_audit.md`` §4).

    Parameters
    ----------
    activity : np.ndarray
        Raw activity signal (1D, NaN-free).
    time_coords : np.ndarray
        Time coordinates (datetime64 or integer-minute).
    min_period, max_period : float
        Period range in hours.
    cwt_method : {'ar1', 'global', 'global_rednoise', 'global_baseline', \
'global_neighbor', 'ridge'}
        Reduction method. ``ar1`` = AR(1) significance ratio (length-sensitive);
        ``global*`` = COI-excluded strength indices (length-stable), differing
        only in how the peak is normalized (``global`` peak/mean band power;
        ``global_rednoise`` peak / AR(1) red-noise floor at the peak period —
        range-robust local prominence; ``global_baseline`` peak / running-median
        baseline; ``global_neighbor`` peak / local-window median). See
        :func:`wavelet_analysis` for the full per-method description.
    resolution : float
        1 / voices_per_octave for the logarithmic period grid (density per
        octave, NOT a fixed bin count). Default ``DEFAULT_CWT_RESOLUTION``
        (= 1/10, i.e. 10 voices/octave; see the module constant for provenance).
    wavelet : str
        PyWavelets wavelet name (default ``'cmor1.5-1.0'`` — complex Morlet
        with bandwidth=1.5, centre frequency=1.0). Bandwidth/centre encode
        the time-frequency-resolution tradeoff; changing them is a
        methodological choice — document any change.
    ar1_conf_level : float
        Confidence level for the Torrence & Compo 1998 AR(1) red-noise
        significance test (``cwt_method='ar1'`` only). Default 0.95.
    ar1_gamma : float
        Decorrelation factor γ in T&C 1998 eq 25 (time-averaged dof).
        Default 2.32 corresponds to Morlet ω₀=6 ≈ cmor1.5-1.0; if a
        different wavelet bandwidth is used, update γ accordingly.

    Returns
    -------
    dict or None
        None if the signal is invalid. Otherwise, a dict with keys:
        ``power``, ``avg_power``, ``periods_hours``, ``sampling_rate_min``,
        ``activity_processed``, ``original_activity``, ``peak_period``,
        ``peak_period_from_avg``, ``peak_power``, ``period_stability``,
        ``cwt_rhythmicity``, ``cwt_p_value``, ``smoothed_periods``,
        ``smoothed_periods_unmasked``, ``instantaneous_periods``,
        ``instantaneous_powers``, ``median_ridge_period``, ``valid_fraction``.
    """
    from scipy.signal import savgol_filter

    original_activity = activity.copy()

    if len(time_coords) < 2:
        return None

    time_deltas_s = _time_diff_seconds(time_coords)
    positive_deltas_s = time_deltas_s[time_deltas_s > 0]
    if len(positive_deltas_s) == 0:
        return None
    sampling_rate_min = float(np.median(positive_deltas_s)) / 60.0

    # Mean-center only (Rethomics/WaveletComp convention with loess.span=0).
    activity = activity - np.mean(activity)

    # Period / scale grid
    min_period_samples = (min_period * 60) / sampling_rate_min
    max_period_samples = (max_period * 60) / sampling_rate_min

    lower_power = np.floor(np.log2(min_period_samples))
    upper_power = np.ceil(np.log2(max_period_samples))

    # round() not int(): resolution = 1/voices can be a hair below the integer
    # (e.g. 1/(1/10) == 9.999...), and int() would truncate 10 voices to 9.
    n_voices_per_octave = max(1, int(round(1 / resolution)))
    n_octaves = upper_power - lower_power

    j = np.arange(0, n_octaves * n_voices_per_octave + 1) / n_voices_per_octave
    periods_samples = 2 ** (lower_power + j)

    mask = (periods_samples >= min_period_samples) & (periods_samples <= max_period_samples)
    periods_samples = periods_samples[mask]
    scales = periods_samples
    periods_hours = (periods_samples * sampling_rate_min) / 60.0

    # --- WARN-don't-return-junk guard: requested range too wide for the record ---
    # A long period cannot be resolved in a short record: outside the cone of
    # influence (COI), the longest period any timepoint can support is the COI
    # value at the record's CENTRE. With this code's COI definition
    # (coi_period[t] = edge_distance[t] * dt / (60*sqrt(2)), edge_distance maxes
    # at N/2), the centre COI period is  P_coi_max = (N/2) * dt_min / (60*sqrt(2))
    # hours, where N = len(activity) and dt_min = sampling_rate_min. If the
    # requested max_period exceeds P_coi_max, every estimate at that period sits
    # INSIDE the COI (edge-contaminated) — the wavelet "sees" the record boundary,
    # not a real oscillation, so reporting it would be returning junk. We WARN and
    # degrade gracefully (the analysis still runs over the resolvable sub-band;
    # nothing is clamped silently or crashed). As a record-length rule of thumb
    # this means resolving period P needs >~ 2*sqrt(2) * P of record outside edge
    # effects; for the 32 h default max that is ~3.8 days, consistent with the
    # DEFAULT_MIN_DD_DAYS_FLOOR (4-day) period-analysis floor.
    n_samples = len(activity)
    record_hours = n_samples * sampling_rate_min / 60.0
    coi_max_period_h = (n_samples / 2.0) * sampling_rate_min / (60.0 * np.sqrt(2))
    if max_period > coi_max_period_h:
        print(
            "Warning: CWT requested max_period="
            f"{max_period:.1f} h exceeds the COI-resolvable limit "
            f"{coi_max_period_h:.1f} h for this {record_hours / 24.0:.1f}-day "
            f"record ({n_samples} samples). Periods above "
            f"{coi_max_period_h:.1f} h lie inside the cone of influence "
            "(edge-contaminated) and are NOT reliable - interpret only the "
            f"resolvable sub-band, or supply >= {DEFAULT_MIN_DD_DAYS_FLOOR:.0f} "
            "DD days. Proceeding over the requested grid without clamping."
        )

    _coefficients, _frequencies, power = cwt_gpu(activity, scales, wavelet, sampling_period=1)
    power = power.astype(np.float32)

    avg_power = np.mean(power, axis=1)
    peak_idx_avg = int(np.argmax(avg_power))
    peak_period_from_avg = float(periods_hours[peak_idx_avg])
    observed_peak_avg = float(avg_power[peak_idx_avg])

    n_timepoints = power.shape[1]
    t_indices = np.arange(n_timepoints)
    edge_distance = np.minimum(t_indices, n_timepoints - 1 - t_indices)
    coi_periods = edge_distance * sampling_rate_min / (60.0 * np.sqrt(2))

    # Raw ridge for scalogram overlays — produced regardless of method so the
    # visualization layer always has something to draw.
    ridge_indices = np.argmax(power, axis=0)
    raw_ridge_periods = periods_hours[ridge_indices].astype(np.float64)

    scalogram_window = min(361, max(11, len(raw_ridge_periods) // 20))
    if scalogram_window % 2 == 0:
        scalogram_window += 1
    if len(raw_ridge_periods) > scalogram_window:
        smoothed_periods_unmasked = savgol_filter(
            raw_ridge_periods, scalogram_window, 3, mode="nearest"
        )
    else:
        smoothed_periods_unmasked = raw_ridge_periods.copy()

    if cwt_method == "ar1":
        # Torrence & Compo 1998 AR(1) red-noise significance, time-averaged form.
        # The score is Power.avg at the dominant period divided by the AR(1)
        # red-noise expected power at that period — a continuous, unbounded
        # SNR-style metric. Using Power.avg (not per-timepoint peak across
        # scales) avoids the multi-scale max-statistic inflation that biases
        # arrhythmic flies upward.
        from scipy.stats import chi2 as _chi2

        threshold_s_pointwise, ar1_alpha = _ar1_significance_threshold(
            activity, periods_hours, sampling_rate_min, conf_level=ar1_conf_level
        )
        var_x = float(np.mean(activity * activity))
        period_samples = periods_hours * 60.0 / sampling_rate_min
        f_normalized = 1.0 / period_samples
        P_red = (1.0 - ar1_alpha**2) / (
            1.0 + ar1_alpha**2 - 2.0 * ar1_alpha * np.cos(2.0 * np.pi * f_normalized)
        )

        # Peak by Power.avg (time-averaged) — the dominant period across the band.
        peak_idx_avg = int(np.argmax(avg_power))
        peak_period_from_avg = float(periods_hours[peak_idx_avg])
        observed_peak_avg = float(avg_power[peak_idx_avg])

        # Time-averaged dof (T&C 1998 eq 25; gamma=2.32 for Morlet ω₀=6 ≈
        # cmor1.5-1.0). For long records the test is asymptotically equivalent
        # to checking observed/expected > 1.
        n_a = float(power.shape[1])
        peak_scale_samples = float(period_samples[peak_idx_avg])
        dof_avg = 2.0 * np.sqrt(1.0 + (n_a / (ar1_gamma * peak_scale_samples)) ** 2)
        snr_threshold = float(_chi2.ppf(ar1_conf_level, df=dof_avg)) / dof_avg
        expected_red = var_x * P_red[peak_idx_avg]
        snr_raw = observed_peak_avg / expected_red if expected_red > 0 else float("nan")
        # Normalise by the 95% threshold so the rhythmic cutoff is exactly 1.0
        # regardless of the fly-specific dof_avg (which varies with the peak
        # period). >1 ⇔ rejects the AR(1) red-noise null at α=0.05.
        snr = snr_raw / snr_threshold if snr_threshold > 0 else float("nan")

        # Per-timepoint pointwise significance (eq 18) — diagnostic only,
        # used for the ``cwt_sig_fraction`` summary and the ridge series.
        peak_scale_idx = np.argmax(power, axis=0)
        peak_periods = periods_hours[peak_scale_idx].astype(np.float64)
        peak_power_vals = power[peak_scale_idx, np.arange(power.shape[1])].astype(np.float64)
        peak_threshold_vals = threshold_s_pointwise[peak_scale_idx]
        inside_coi = peak_periods > coi_periods
        valid = ~inside_coi
        if np.any(valid):
            sig_fraction = float(np.mean(peak_power_vals[valid] > peak_threshold_vals[valid]))
            ridge_for_period = peak_periods.copy()
            ridge_for_period[~valid] = np.nan
            median_ridge_period = float(np.nanmedian(ridge_for_period))
            period_stability = float(np.nanstd(ridge_for_period))
        else:
            sig_fraction = float("nan")
            median_ridge_period = float("nan")
            period_stability = float("nan")

        return {
            "power": power,
            "avg_power": avg_power,
            "periods_hours": periods_hours,
            "sampling_rate_min": sampling_rate_min,
            "activity_processed": activity,
            "original_activity": original_activity,
            "peak_period": peak_period_from_avg,
            "peak_period_from_avg": peak_period_from_avg,
            "peak_power": observed_peak_avg,
            "period_stability": period_stability,
            "cwt_rhythmicity": float(snr),  # observed/threshold ratio; >1 ⇔ p<0.05 vs AR(1) null
            "cwt_p_value": float("nan"),
            "cwt_ar1_alpha": float(ar1_alpha),
            "cwt_sig_fraction": sig_fraction,
            "cwt_snr_threshold_95": float(snr_threshold),  # diagnostic; depends on dof_avg
            "smoothed_periods": smoothed_periods_unmasked,
            "smoothed_periods_unmasked": smoothed_periods_unmasked,
            "instantaneous_periods": peak_periods,
            "instantaneous_powers": peak_power_vals,
            "median_ridge_period": median_ridge_period,
            "valid_fraction": sig_fraction,
        }

    elif cwt_method == "ridge":
        instantaneous_periods = raw_ridge_periods.copy()
        instantaneous_powers = np.max(power, axis=0).astype(np.float64)

        mean_power_per_timepoint = np.mean(power, axis=0)
        mean_power_per_timepoint[mean_power_per_timepoint == 0] = 1e-10
        peak_to_mean_ratio = instantaneous_powers / mean_power_per_timepoint
        LOW_CONFIDENCE_THRESHOLD = 2.0
        low_confidence = peak_to_mean_ratio < LOW_CONFIDENCE_THRESHOLD
        instantaneous_periods[low_confidence] = np.nan
        instantaneous_powers[low_confidence] = np.nan

        inside_coi = raw_ridge_periods > coi_periods
        instantaneous_periods[inside_coi] = np.nan
        instantaneous_powers[inside_coi] = np.nan

        valid_mask = ~np.isnan(instantaneous_periods)
        smooth_window = min(361, max(11, int(len(instantaneous_periods) / 4)))
        if smooth_window % 2 == 0:
            smooth_window += 1
        if np.sum(valid_mask) > smooth_window:
            filled = np.interp(
                np.arange(len(instantaneous_periods)),
                np.where(valid_mask)[0],
                instantaneous_periods[valid_mask],
            )
            smoothed_periods = savgol_filter(filled, smooth_window, 3, mode="nearest")
            smoothed_periods[~valid_mask] = np.nan
        else:
            smoothed_periods = instantaneous_periods.copy()

        if np.all(np.isnan(instantaneous_periods)):
            return None

        n_coi_free = int(np.sum(~inside_coi))
        n_confident = int(np.sum(valid_mask))
        valid_fraction = n_confident / max(n_coi_free, 1)

        median_ridge_period = float(np.nanmedian(smoothed_periods))
        period_stability = float(np.nanstd(smoothed_periods))

        return {
            "power": power,
            "avg_power": avg_power,
            "periods_hours": periods_hours,
            "sampling_rate_min": sampling_rate_min,
            "activity_processed": activity,
            "original_activity": original_activity,
            "peak_period": median_ridge_period,
            "peak_period_from_avg": peak_period_from_avg,
            "peak_power": float(avg_power[peak_idx_avg]),
            "period_stability": period_stability,
            "cwt_rhythmicity": float(valid_fraction),
            "cwt_p_value": float("nan"),
            "smoothed_periods": smoothed_periods,
            "smoothed_periods_unmasked": smoothed_periods_unmasked,
            "instantaneous_periods": instantaneous_periods,
            "instantaneous_powers": instantaneous_powers,
            "median_ridge_period": median_ridge_period,
            "valid_fraction": float(valid_fraction),
        }

    elif cwt_method in ("global", "global_rednoise", "global_baseline", "global_neighbor"):
        # -------------------------------------------------------------------
        # 'global*' — length-stable, COI-excluded STRENGTH indices. CWT's analog
        # of AC's RI / LS's R^2 (the session-4 LS fix mirrored for CWT). The
        # default 'ar1' metric is a length-SENSITIVE significance ratio (observed
        # Power.avg / a 95% AR(1) threshold whose bar falls as the record
        # lengthens), so it over-calls long records. Every method below instead
        # gates on a continuous strength read from the SAME COI-excluded global
        # wavelet spectrum (T&C 1998 eq 22 restricted to the COI; built once by
        # the shared helper so the spectrum and peak NEVER drift between metrics).
        # They differ ONLY in how the peak is normalized to a strength:
        #
        #   'global'           peak / MEAN(band power)  — the original normalized
        #                      strength. LENGTH-stable, but NOT range-robust: on a
        #                      WIDE search band an arrhythmic red-noise RAMP rising
        #                      toward the long-period edge inflates the mean and a
        #                      noise pile-up at the edge reads as a "peak", so the
        #                      ratio passes (the band-edge artifact). The
        #                      local-prominence methods below fix this by
        #                      measuring the peak against the LOCAL noise floor under
        #                      it, not the whole-band mean — a real rhythm (even at
        #                      48 h) is a localized bump above the floor; a ramp is
        #                      a monotonic rise with no local bump.
        #   'global_rednoise'  peak / AR(1) red-noise EXPECTED power AT the peak
        #                      period (T&C var*P_red(f_peak)). The length-stable
        #                      strength FORM of 'ar1' WITHOUT 'ar1's length-sensitive
        #                      chi-squared threshold division. A band-edge ramp is
        #                      exactly what the AR(1) model predicts at that long
        #                      period, so its ratio is ~1 and it is REJECTED, while
        #                      a genuine bump sits several-fold above the red-noise
        #                      expectation. (The range-robust winner on the simple
        #                      cohort.)
        #   'global_baseline'  peak / smooth (running-median) baseline AT the peak.
        #   'global_neighbor'  peak / median power in a local +/-0.5-octave window
        #                      around the peak (local-neighborhood prominence).
        #
        # (1) COI-excluded global spectrum + peak (shared helper; identical across
        #     all four methods so they are directly comparable).
        _g = _coi_excluded_global_spectrum(power, periods_hours, coi_periods, sampling_rate_min)
        global_spectrum = _g["global_spectrum"]
        peak_idx = _g["peak_idx"]

        if peak_idx is None:
            # Whole band is inside the COI (record too short for any scale) —
            # no resolvable circadian spectrum; report NaN strength (§2a: NaN =
            # "not measured", never a fabricated low value), fall back to the
            # COI-blind peak period only so a value exists for plots.
            fallback_idx = int(np.argmax(avg_power))
            return {
                "power": power,
                "avg_power": global_spectrum,  # all-NaN COI-excluded spectrum
                "periods_hours": periods_hours,
                "sampling_rate_min": sampling_rate_min,
                "activity_processed": activity,
                "original_activity": original_activity,
                "peak_period": float(periods_hours[fallback_idx]),
                "peak_period_from_avg": peak_period_from_avg,
                "peak_power": float("nan"),
                "period_stability": float("nan"),
                "cwt_rhythmicity": float("nan"),
                "cwt_p_value": float("nan"),
                "smoothed_periods": smoothed_periods_unmasked,
                "smoothed_periods_unmasked": smoothed_periods_unmasked,
                "instantaneous_periods": raw_ridge_periods,
                "instantaneous_powers": np.max(power, axis=0).astype(np.float64),
                "median_ridge_period": float("nan"),
                "valid_fraction": float("nan"),
            }

        peak_period = _g["peak_period"]  # sub-grid parabolic-refined
        peak_power_global = _g["peak_power"]

        # (2) Normalize the peak to a strength per the selected method.
        if cwt_method == "global":
            band_vals = global_spectrum[np.isfinite(global_spectrum)]
            band_mean = float(np.mean(band_vals))
            cwt_rhythmicity = peak_power_global / band_mean if band_mean > 0 else float("nan")
        elif cwt_method == "global_rednoise":
            # peak / modeled AR(1) red-noise floor AT the peak period. `activity`
            # is already mean-centered above. Rejects band-edge ramps (a ramp is
            # what AR(1) predicts there) while keeping genuine localized bumps.
            exp_red = _ar1_expected_power_at(activity, peak_period, sampling_rate_min)
            cwt_rhythmicity = (
                peak_power_global / exp_red
                if np.isfinite(exp_red) and exp_red > 0
                else float("nan")
            )
        elif cwt_method == "global_baseline":
            # peak / smooth running-median baseline UNDER the peak (loess-style).
            base = _running_median_baseline(global_spectrum)
            base_pk = base[peak_idx]
            cwt_rhythmicity = (
                peak_power_global / float(base_pk)
                if np.isfinite(base_pk) and base_pk > 0
                else float("nan")
            )
        else:  # 'global_neighbor'
            # peak / median power in a local +/-0.5-octave window around the peak.
            log2p = np.log2(periods_hours)
            step = float(np.median(np.diff(log2p))) if len(log2p) > 1 else 0.1
            voct = (1.0 / step) if step > 0 else 10.0
            half = max(2, int(round(voct * 0.5)))
            lo_i = max(0, peak_idx - half)
            hi_i = min(len(global_spectrum), peak_idx + half + 1)
            nb = global_spectrum[lo_i:hi_i]
            nb = nb[np.isfinite(nb)]
            nb_med = float(np.median(nb)) if nb.size else float("nan")
            cwt_rhythmicity = (
                peak_power_global / nb_med if np.isfinite(nb_med) and nb_med > 0 else float("nan")
            )

        # Period stability: spread of the COI-excluded pointwise ridge (diagnostic).
        peak_scale_idx = np.argmax(power, axis=0)
        peak_periods_pw = periods_hours[peak_scale_idx].astype(np.float64)
        inside_coi = peak_periods_pw > coi_periods
        valid_pw = ~inside_coi
        if np.any(valid_pw):
            ridge_for_period = peak_periods_pw.copy()
            ridge_for_period[~valid_pw] = np.nan
            period_stability = float(np.nanstd(ridge_for_period))
            median_ridge_period = float(np.nanmedian(ridge_for_period))
        else:
            period_stability = float("nan")
            median_ridge_period = float("nan")

        # (3) cwt_powerseries reflects the GATED quantity: store the COI-excluded
        #     spectrum (NaN where unusable) so the page-3 explorer / period
        #     plots show the same spectrum the strength was read from.
        return {
            "power": power,
            "avg_power": global_spectrum.astype(np.float64),
            "periods_hours": periods_hours,
            "sampling_rate_min": sampling_rate_min,
            "activity_processed": activity,
            "original_activity": original_activity,
            "peak_period": peak_period,
            "peak_period_from_avg": peak_period,
            "peak_power": peak_power_global,
            "period_stability": period_stability,
            "cwt_rhythmicity": float(cwt_rhythmicity),  # method-specific strength (see above)
            "cwt_p_value": float("nan"),
            "cwt_sig_fraction": float("nan"),
            "smoothed_periods": smoothed_periods_unmasked,
            "smoothed_periods_unmasked": smoothed_periods_unmasked,
            "instantaneous_periods": peak_periods_pw,
            "instantaneous_powers": np.max(power, axis=0).astype(np.float64),
            "median_ridge_period": median_ridge_period,
            "valid_fraction": float("nan"),
        }

    else:
        raise ValueError(
            f"Unknown cwt_method={cwt_method!r}; expected 'ar1', "
            "'global', 'global_rednoise', 'global_baseline', 'global_neighbor', "
            "or 'ridge'."
        )


def _cwt_single_fly_worker(args):
    """Worker entry point. Delegates to _preprocess_and_compute_cwt().

    Args tuple layout (spawn-safe, picklable): the fixed elements end with
    ``..., ar1_gamma, max_bridge_gap_minutes`` (the bridge ceiling is threaded
    here exactly like ``min_num_days``), optionally followed by a trailing
    ``return_power_2d`` flag (default False). When ``return_power_2d`` is True the
    worker returns a tuple ``(per_fly_ds, power_2d_float32, time_h_float32)`` so the
    main process can stream-aggregate per-group means without re-running CWT later;
    when False the worker returns just ``per_fly_ds`` (legacy behavior). So the
    tuple length is 14 (with the flag) or 13 (without).
    """
    # max_bridge_gap_minutes is threaded through the args tuple exactly like
    # min_num_days (fixed element), before the optional trailing return_power_2d.
    if len(args) == 14:
        (
            fly_data,
            activity_var,
            time_var,
            min_period,
            max_period,
            fly_id,
            cwt_method,
            min_num_days,
            resolution,
            wavelet_name,
            ar1_conf_level,
            ar1_gamma,
            max_bridge_gap_minutes,
            return_power_2d,
        ) = args
    else:
        (
            fly_data,
            activity_var,
            time_var,
            min_period,
            max_period,
            fly_id,
            cwt_method,
            min_num_days,
            resolution,
            wavelet_name,
            ar1_conf_level,
            ar1_gamma,
            max_bridge_gap_minutes,
        ) = args
        return_power_2d = False

    try:
        return _cwt_single_fly_worker_inner(
            fly_data,
            activity_var,
            time_var,
            min_period,
            max_period,
            fly_id,
            cwt_method,
            min_num_days,
            resolution,
            wavelet_name,
            ar1_conf_level,
            ar1_gamma,
            max_bridge_gap_minutes,
            return_power_2d=return_power_2d,
        )
    except Exception as e:
        print(f"Warning: CWT analysis failed for fly {fly_id}: {e}")
        return (None, None, None) if return_power_2d else None


def _cwt_single_fly_worker_inner(
    fly_data,
    activity_var,
    time_var,
    min_period,
    max_period,
    fly_id,
    cwt_method,
    min_num_days,
    resolution,
    wavelet_name,
    ar1_conf_level,
    ar1_gamma,
    max_bridge_gap_minutes=DEFAULT_MAX_BRIDGE_GAP_MINUTES,
    return_power_2d=False,
):
    activity_raw = fly_data[activity_var].values.ravel()
    time_raw = fly_data[time_var].values
    # bridge_small_gaps=True: interior NaN dropouts whose missing span is ≤ the
    # max_bridge_gap_minutes ceiling are bridged by transient, worker-local linear
    # interpolation so a dropped read does not sever a clean record. The filled
    # values feed CWT only and are NEVER written back to the Dataset/.nc (§2a).
    # Gaps LARGER than the ceiling stay hard breaks → longest-clean-segment.
    activity, time_coords, expected_delta_s = _extract_longest_continuous_block(
        activity_raw,
        time_raw,
        bridge_small_gaps=True,
        max_bridge_gap_minutes=max_bridge_gap_minutes,
    )
    if activity is None or len(activity) < 10:
        return None

    duration_days = (
        _time_span_seconds(time_coords[0], time_coords[-1]) + expected_delta_s
    ) / 86400.0
    if duration_days < min_num_days:
        return None

    if np.all(activity == 0) or np.std(activity) == 0:
        return None

    cwt_result = _preprocess_and_compute_cwt(
        activity,
        time_coords,
        min_period,
        max_period,
        cwt_method=cwt_method,
        resolution=resolution,
        wavelet=wavelet_name,
        ar1_conf_level=ar1_conf_level,
        ar1_gamma=ar1_gamma,
    )
    if cwt_result is None:
        return None

    avg_power = cwt_result["avg_power"]
    periods_hours = cwt_result["periods_hours"]
    sampling_rate_min = cwt_result["sampling_rate_min"]
    smoothed_periods = cwt_result["smoothed_periods"]
    instantaneous_powers = cwt_result["instantaneous_powers"]
    peak_period = cwt_result["peak_period"]
    peak_power = cwt_result["peak_power"]
    period_stability = cwt_result["period_stability"]
    cwt_rhythmicity = cwt_result["cwt_rhythmicity"]
    cwt_p_value = cwt_result["cwt_p_value"]
    # ar1-only diagnostics: fall through cleanly to NaN for the ridge/global*
    # methods, which don't fit an AR(1) null and have no significance fraction.
    if cwt_method == "ar1":
        cwt_ar1_alpha = float(cwt_result.get("cwt_ar1_alpha", float("nan")))
        cwt_sig_fraction = float(cwt_result.get("cwt_sig_fraction", float("nan")))
    else:
        cwt_ar1_alpha = float("nan")
        cwt_sig_fraction = float("nan")

    fly_id_scalar = (
        np.asarray(fly_id).item()
        if np.asarray(fly_id).shape == ()
        else np.asarray(fly_id).ravel()[0]
    )
    result_ds = xr.Dataset(
        {
            # §2b: per-fly CWT results stored float32 (period ~24 h, rhythmicity/
            # power/alpha/fraction all within float32's ~7-sig-fig range; calc stays
            # float64). cwt_p_value is retained as an all-NaN placeholder (no
            # retained method emits a p-value), so it cannot underflow float32.
            "cwt_period": (("id",), np.array([peak_period], dtype=np.float32), {"units": "h"}),
            "cwt_power": (("id",), np.array([peak_power], dtype=np.float32)),
            "cwt_period_stability": (
                ("id",),
                np.array([period_stability], dtype=np.float32),
                {"units": "h"},
            ),
            "cwt_rhythmicity": (
                ("id",),
                np.array([cwt_rhythmicity], dtype=np.float32),
                {
                    "description": "ar1: Power.avg at peak period divided by the "
                    "Torrence & Compo 1998 95% AR(1) red-noise "
                    "threshold (continuous, unbounded; >1 ⇔ "
                    "p<0.05 vs AR(1) null); "
                    "global: COI-excluded peak / MEAN band power; "
                    "global_rednoise: COI-excluded peak / AR(1) "
                    "red-noise expected power at the peak period "
                    "(range-robust local prominence); "
                    "global_baseline: peak / running-median baseline; "
                    "global_neighbor: peak / local-window median; "
                    "ridge: fraction of analyzable timepoints "
                    "with confident circadian peak"
                },
            ),
            "cwt_p_value": (
                ("id",),
                np.array([cwt_p_value], dtype=np.float32),
                {"description": "reserved p-value output; NaN for all retained methods"},
            ),
            "cwt_ar1_alpha": (
                ("id",),
                np.array([cwt_ar1_alpha], dtype=np.float32),
                {"description": "fitted AR(1) lag-1 autocorrelation (ar1 method only)"},
            ),
            "cwt_sig_fraction": (
                ("id",),
                np.array([cwt_sig_fraction], dtype=np.float32),
                {
                    "description": "fraction of non-COI timepoints with peak SNR ≥ 1 (ar1 method only)"
                },
            ),
            "cwt_powerseries": (
                ("id", "cwt_periodogram_periods"),
                avg_power[np.newaxis, :].astype(np.float32),
            ),
            "cwt_ridge_periods": (
                ("id", "cwt_time"),
                smoothed_periods.astype(np.float32)[np.newaxis, :],
            ),
            "cwt_ridge_powers": (
                ("id", "cwt_time"),
                instantaneous_powers.astype(np.float32)[np.newaxis, :],
            ),
        },
        coords={
            "id": np.array([fly_id_scalar]),
            "cwt_periodogram_periods": periods_hours,
            "cwt_time": np.arange(len(smoothed_periods)) * sampling_rate_min / 60,
        },
    )
    if return_power_2d:
        # Stream the full 2D power matrix back to the main process so it
        # can accumulate per-group running means without ever persisting
        # the per-fly 2D matrix in the dataset (memory-bounded).
        power_2d = np.asarray(cwt_result["power"], dtype=np.float32)
        time_h = (np.arange(power_2d.shape[1]) * sampling_rate_min / 60.0).astype(np.float32)
        return result_ds, power_2d, time_h
    return result_ds


def wavelet_analysis(
    ds: xr.Dataset,
    activity_var: str = "activity",
    time_var: str = "time",
    id_var: str = "id",
    min_period: float = DEFAULT_CWT_MIN_PERIOD,
    max_period: float = DEFAULT_CWT_MAX_PERIOD,
    resolution: float = DEFAULT_CWT_RESOLUTION,
    wavelet_name: str = "cmor1.5-1.0",
    n_processes: int = None,
    cwt_method: str = None,
    ar1_conf_level: float = 0.95,
    ar1_gamma: float = 2.32,
    min_num_days: float = 0,
    max_bridge_gap_minutes: float = DEFAULT_MAX_BRIDGE_GAP_MINUTES,
    phase: str = "auto",
    progress_callback=None,
    pool=None,
    compute_group_averages: bool = False,
    group_coord: str = "group",
    filter_nonrhythmic_for_average: bool = True,
    phase_label: str | None = None,
) -> tuple[xr.Dataset, list[dict]]:
    """
    CWT periodogram analysis. The default reduction method is resolved from
    ``calibrations.DEFAULT_CWT_METHOD`` (``'global_rednoise'``) when
    ``cwt_method`` is left as ``None`` — a range-robust, length-stable
    local-prominence strength that rejects arrhythmic red-noise band-edge ramps
    on a WIDE search while detecting genuine long-period rhythms. It replaced
    ``'ar1'`` (a length-SENSITIVE significance ratio that over-calls long
    records); ``'ar1'`` and every other method remain selectable by passing
    ``cwt_method`` explicitly. See ``period_analysis_audit.md`` §4.3 for the
    algorithmic reference and ``calibrations.py`` for the
    default-flip provenance.

    Parameters
    ----------
    cwt_method : {'ar1', 'global', 'global_rednoise', \
'global_baseline', 'global_neighbor', 'ridge'}
        Each method reduces the 2D CWT power surface to a per-fly
        ``cwt_rhythmicity`` strength/score and a ``cwt_period``. The classifier
        threshold differs per method (see ``cwt_threshold_for``); ``cwt_period``
        is the dominant circadian peak read the same way for all methods.

        ``ar1`` (former default) — Torrence & Compo 1998 AR(1) red-noise SIGNIFICANCE.
        ``cwt_rhythmicity`` is the time-averaged ``Power.avg`` at the dominant
        period divided by the 95% AR(1) red-noise threshold there (a continuous,
        unbounded significance ratio; >1 ⇔ rejects the AR(1) null at α=0.05).
        Computed over ALL timepoints (COI-blind) — it is a *significance* test,
        not a length-stable strength, so it over-calls long records (see
        ``'global'``). The COI-excluded pointwise significant-area fraction is
        reported separately as the diagnostic ``cwt_sig_fraction`` and is NOT
        what the classifier gates on.

        ``global`` — length-stable, COI-EXCLUDED normalized STRENGTH (CWT's
        analog of AC's RI / LS's R^2; mirrors the session-4 LS power-gate fix).
        ``cwt_rhythmicity`` is the height of the dominant circadian peak in the
        COI-excluded global wavelet spectrum (T&C 1998 eq 22, time-averaged over
        only COI-free timepoints per scale) divided by the MEAN band power of
        that same spectrum — a variance-free ratio that does not inflate with
        record length. NOT range-robust: on a WIDE band an arrhythmic red-noise
        ramp toward the long-period edge inflates the mean and the edge pile-up
        reads as a peak, so it over-calls (the band-edge artifact). Threshold
        provisional (DEFAULT_CWT_GLOBAL_THRESHOLD). ``cwt_powerseries`` is the
        COI-excluded spectrum; ``cwt_period`` uses sub-grid parabolic peak
        refinement.

        ``global_rednoise`` (default) — RANGE-ROBUST, length-stable LOCAL-PROMINENCE
        strength. ``cwt_rhythmicity`` is the COI-excluded global-spectrum peak
        divided by the AR(1) red-noise EXPECTED power AT the peak period (T&C
        1998 ``var*P_red(f_peak)``) — the peak measured against the modeled noise
        floor *under* it, not the whole-band mean. An arrhythmic ramp rising to
        the long-period edge is exactly what AR(1) predicts there, so it scores
        ~1 and is rejected even on a WIDE 16-50 h search, while a genuine
        localized bump (even at 48 h) scores several-fold higher. This is the
        length-stable strength FORM of ``ar1`` WITHOUT ``ar1``'s length-sensitive
        chi-squared threshold division. Threshold provisional
        (DEFAULT_CWT_REDNOISE_THRESHOLD).

        ``global_baseline`` / ``global_neighbor`` — other local-prominence
        candidates (peak / running-median baseline; peak / local-window median).
        Comparison-only; no dedicated calibrated cutoff.

        ``ridge`` — legacy Pythomics ridge tracker (median ridge period,
        peak/band-mean valid-fraction); retained for comparison.
    n_processes : int or None
        Number of parallel processes (None = auto-detect).
    min_num_days : float
        Minimum required duration (days) of the longest continuous valid block.
    max_bridge_gap_minutes : float, default DEFAULT_MAX_BRIDGE_GAP_MINUTES (60)
        The ONE gap-bridge ceiling passed to the block extractor: an interior gap
        whose missing span is ≤ this many minutes is bridged (worker-local linear
        interpolation of the NaN cells, §2a-transient — never persisted); a gap
        larger than the ceiling stays a hard break → longest-clean-segment. 0 turns
        bridging off. Single-sourced in ``calibrations.DEFAULT_MAX_BRIDGE_GAP_MINUTES``.
    phase : str
        ``"auto"`` (default) selects DD if available, else full dataset.
        ``"DD"``, ``"LD"``, or ``"both"`` to force a specific phase.
    compute_group_averages : bool, default False
        If True, stream-aggregate per-group means of the 2D power matrix
        during the per-fly CWT pass and RETURN them (see Returns). Nothing
        is written to disk — rendering and saving are the caller's job.
        Memory-bounded (running sums and counts only; per-fly 2D matrices are
        discarded as soon as their contribution is added). Folded into the CWT
        pass to avoid a second recomputation later. Default off.
    group_coord : str, default 'group'
        Per-fly coord that labels group membership for averaging.
    filter_nonrhythmic_for_average : bool, default True
        When ``compute_group_averages`` is True, restrict each group's
        average to flies flagged ``ac_rhythmic=True`` on ``ds``. AC must
        already have been run + classified for this to take effect; if
        the flag is absent the average will include every fly (caller
        should pre-warn the user).
    phase_label : str or None, default None
        Active phase label (e.g. ``'DD'``, ``'LD'``, ``'full'``). Carried
        through on each returned group average so a caller naming files can
        keep DD and LD reruns from overwriting each other. When None, falls
        back to ``phase_used`` (the value returned by :func:`_select_phase`).

    Returns
    -------
    (merged_ds, group_averages)
        merged_ds : xr.Dataset
            ``ds`` with the CWT outputs merged in and the ``cwt_*`` attrs set.
        group_averages : list of dict
            One entry per group when ``compute_group_averages`` is True, else
            empty. Each carries ``group``, ``n_flies``, the 2D ``mean_power``
            array, its ``period_axis`` and ``time_h`` axes, the
            ``period_range`` and the ``phase_label`` — everything needed to
            render or save it, without this function deciding which.
    """

    # Resolve the default method from the single calibration source (no
    # hardcoded literal default here). 'global_rednoise' replaced 'ar1' as the
    # default; 'ar1' and every other method remain
    # selectable by passing cwt_method explicitly.
    if cwt_method is None:
        cwt_method = DEFAULT_CWT_METHOD

    analysis_ds, phase_used = _select_phase(ds, phase=phase)
    if phase_label is None:
        phase_label = str(phase_used)

    print("\nPerforming CWT analysis...")
    print(
        f"Data phase: {phase_used}"
        + (
            " (DD available, selected automatically)"
            if phase == "auto" and phase_used == "DD"
            else ""
        )
    )
    print(
        f"GPU acceleration: {'ENABLED (ptwt+CUDA)' if PTWT_AVAILABLE else 'DISABLED (CPU-only PyWavelets)'}"
    )
    print(f"Period range: {min_period:.1f} - {max_period:.1f} hours")
    print(f"Resolution: {resolution}")
    print(f"Wavelet: {wavelet_name}")
    print(f"Method: {cwt_method}")

    fly_groups = list(analysis_ds.groupby(id_var))
    # max_bridge_gap_minutes rides in the args tuple exactly like min_num_days
    # (fixed element), before the trailing return_power_2d flag.
    args_list = [
        (
            group,
            activity_var,
            time_var,
            min_period,
            max_period,
            fly_id,
            cwt_method,
            min_num_days,
            resolution,
            wavelet_name,
            ar1_conf_level,
            ar1_gamma,
            max_bridge_gap_minutes,
            bool(compute_group_averages),
        )
        for fly_id, group in fly_groups
    ]

    results = []
    _cwt_completed = 0
    _cwt_total = len(args_list)

    # Streaming per-group aggregator. Keeps a running sum + count per
    # group so we never persist a per-fly 2D matrix (memory-bounded:
    # one (n_periods × n_time) float64 sum and one count per group).
    # Time axes are aligned by truncating to the shortest fly seen so far
    # in each group — drift between flies is rare (same dataset attrs)
    # but truncation keeps the math simple.
    group_sums: dict = {}
    group_counts: dict = {}
    group_period_axis = None
    group_sampling_rate_min = None

    fly_to_group: dict = {}
    fly_to_ac_rhythmic: dict = {}
    if compute_group_averages:
        if group_coord not in ds.coords:
            print(
                f"Warning: '{group_coord}' coord not on dataset — skipping "
                "group-averaged scalogram saves."
            )
            compute_group_averages = False
        else:
            grp_vals = np.asarray(ds[group_coord].values, dtype=object)
            id_vals = np.asarray(ds["id"].values)
            for _i, _id in enumerate(id_vals):
                fly_to_group[_id] = str(grp_vals[_i]) if _i < len(grp_vals) else "unknown"
            if filter_nonrhythmic_for_average and "ac_rhythmic" in ds.coords:
                ac_flag = np.asarray(ds["ac_rhythmic"].values, dtype=bool)
                for _i, _id in enumerate(id_vals):
                    fly_to_ac_rhythmic[_id] = bool(ac_flag[_i]) if _i < len(ac_flag) else False
            else:
                # No AC flag available (or filter disabled) — accept every fly.
                for _id in id_vals:
                    fly_to_ac_rhythmic[_id] = True

    if n_processes is None:
        n_processes = get_optimal_workers(
            len(args_list), use_gpu=PTWT_AVAILABLE, period_range=(min_period, max_period)
        )
    print(f"Using {n_processes} parallel workers" + (" (GPU)" if PTWT_AVAILABLE else " (CPU)"))

    with _worker_pool(pool, n_processes) as worker_pool:
        for raw_result in tqdm(
            worker_pool.imap_unordered(_cwt_single_fly_worker, args_list),
            total=_cwt_total,
            desc="CWT Analysis",
            unit="fly",
        ):
            if compute_group_averages:
                if raw_result is None:
                    result_ds, power_2d, time_h_arr = None, None, None
                else:
                    result_ds, power_2d, time_h_arr = raw_result
            else:
                result_ds = raw_result
                power_2d, time_h_arr = None, None

            if result_ds is not None:
                results.append(result_ds)

                if compute_group_averages and power_2d is not None and time_h_arr is not None:
                    fly_id_scalar = result_ds["id"].values[0]
                    accept = fly_to_ac_rhythmic.get(fly_id_scalar, True)
                    if accept:
                        grp = fly_to_group.get(fly_id_scalar, "unknown")
                        if group_period_axis is None:
                            group_period_axis = np.asarray(
                                result_ds["cwt_periodogram_periods"].values, dtype=np.float32
                            )
                            if time_h_arr.size >= 2:
                                group_sampling_rate_min = float(
                                    (time_h_arr[1] - time_h_arr[0]) * 60.0
                                )
                        if grp in group_sums:
                            existing = group_sums[grp]
                            n_t = min(existing.shape[1], power_2d.shape[1])
                            group_sums[grp] = existing[:, :n_t] + power_2d[:, :n_t].astype(
                                existing.dtype
                            )
                        else:
                            group_sums[grp] = power_2d.astype(np.float64)
                        group_counts[grp] = group_counts.get(grp, 0) + 1

            _cwt_completed += 1
            if progress_callback:
                progress_callback(_cwt_completed, _cwt_total)
            if PTWT_AVAILABLE and len(results) % 50 == 0:
                with suppress(Exception):
                    torch.cuda.empty_cache()

    if not results:
        print("Warning: CWT analysis failed for all individuals.")
        # Same 2-tuple shape as the success path below. Returning a bare Dataset
        # here was a real trap rather than just an inconsistency: a caller doing
        # `ds, averages = wavelet_analysis(...)` unpacks an xr.Dataset over its
        # DATA_VARS, so with exactly two of them the unpack SUCCEEDS and silently
        # binds two variable-name strings. Any other count raises. Both are worse
        # than the empty list.
        return ds, []
    print("CWT analysis done.")
    combined_results = xr.concat(results, dim=id_var)

    existing_cwt_vars = [v for v in combined_results.data_vars if v in ds.data_vars]
    existing_cwt_coords = [c for c in combined_results.coords if c in ds.coords and c != "id"]
    merged_ds = ds.drop_vars(existing_cwt_vars + existing_cwt_coords, errors="ignore").merge(
        combined_results
    )

    merged_ds.attrs["cwt_min_period"] = min_period
    merged_ds.attrs["cwt_max_period"] = max_period
    merged_ds.attrs["cwt_resolution"] = resolution
    merged_ds.attrs["cwt_wavelet"] = wavelet_name
    merged_ds.attrs["cwt_method"] = cwt_method
    merged_ds.attrs["cwt_min_num_days"] = min_num_days
    merged_ds.attrs["cwt_max_bridge_gap_minutes"] = max_bridge_gap_minutes
    merged_ds.attrs["cwt_phase"] = phase_used

    # ----- Group-averaged scalograms: RETURNED, not written -----
    # This block used to `from plotting import save_group_average_scalogram_png`
    # and write PNG + CSV to disk from inside the analysis. An analysis function
    # with a filesystem side effect cannot be called without also deciding where
    # files go, and the deferred plotting import was there to dodge a circular
    # one — the same wrong-seam signal as the deferred analysis imports in
    # plotting.py. The arrays come back to the caller, which decides whether to
    # render or save them.
    group_averages = []
    if compute_group_averages and group_sums:
        sampling_min = group_sampling_rate_min if group_sampling_rate_min is not None else 1.0
        for grp, _sum in group_sums.items():
            n = int(group_counts.get(grp, 0))
            if n <= 0:
                continue
            mean_pw = (_sum / float(n)).astype(np.float32)
            group_averages.append(
                {
                    "group": str(grp),
                    "n_flies": n,
                    "mean_power": mean_pw,
                    "period_axis": group_period_axis,
                    "time_h": (np.arange(mean_pw.shape[1]) * sampling_min / 60.0).astype(
                        np.float32
                    ),
                    "period_range": (float(min_period), float(max_period)),
                    "phase_label": phase_label,
                }
            )


    # Print summary statistics
    valid_periods = merged_ds["cwt_period"].values
    valid_periods = valid_periods[~np.isnan(valid_periods)]

    if len(valid_periods) > 0:
        period_stabilities = merged_ds["cwt_period_stability"].values
        period_stabilities = period_stabilities[~np.isnan(period_stabilities)]

        print("\nEnhanced CWT Analysis Summary:")
        print(f"  Analyzed {len(valid_periods)} flies successfully")
        print(
            f"  Mean peak period: {np.mean(valid_periods):.2f} ± {np.std(valid_periods):.2f} hours"
        )
        print(f"  Median peak period: {np.median(valid_periods):.2f} hours")
        print(f"  Range: {np.min(valid_periods):.2f} - {np.max(valid_periods):.2f} hours")
        if len(period_stabilities) > 0:
            print(f"  Mean period stability: {np.mean(period_stabilities):.2f} hours")

        if "cwt_rhythmicity" in merged_ds:
            rhythmicity_vals = merged_ds["cwt_rhythmicity"].values
            rhythmicity_vals = rhythmicity_vals[~np.isnan(rhythmicity_vals)]
            if len(rhythmicity_vals) > 0:
                print(f"  Mean rhythmicity: {np.mean(rhythmicity_vals):.2f}")
                print(
                    f"  Flies with rhythmicity >= 0.5: {np.sum(rhythmicity_vals >= 0.5)}/{len(rhythmicity_vals)}"
                )

    return merged_ds, group_averages


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# CWT Individual Fly Scalograms (Tier 2 - on-demand recomputation)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def compute_single_fly_scalogram(
    ds, fly_id, activity_var="activity", time_var="time", cwt_method=None
):
    """
    Recompute the full 2D CWT scalogram for a single fly on demand.

    Applies the same preprocessing config as the production analysis run
    that produced this dataset, reconstructed from ``ds.attrs`` (the
    ``prep_*`` fields stamped by :func:`preprocessing.preprocess_activity`).
    Today's CWT default is no-op preprocessing so behaviour is unchanged;
    this future-proofs the scalogram view against any non-default CWT
    preprocessing the user might enable on page 3.

    The reduction method (``cwt_method``) defaults to whatever was
    recorded in ``ds.attrs['cwt_method']``, falling back to the single
    calibration default (``calibrations.DEFAULT_CWT_METHOD``) for legacy
    datasets that pre-date that attribute.

    Returns
    -------
    dict with keys: ``power``, ``periods_hours``, ``time_hours``, ``coi_hours``,
    ``ridge_periods``, ``avg_power``, ``activity``.
    """
    # Apply the recorded preprocessing config (if any) to a 1-fly slice
    # so the scalogram matches the analysis-run numbers for this fly.
    fly_ds = ds.sel(id=[fly_id])
    if any(k.startswith("prep_") for k in fly_ds.attrs):
        from preprocessing import PreprocessConfig, preprocess_activity

        cfg = PreprocessConfig(
            bin_minutes=int(fly_ds.attrs.get("prep_bin_minutes", 0)),
            smooth_sigma_min=float(fly_ds.attrs.get("prep_smooth_sigma_min", 0.0)),
            lopass_hours=float(fly_ds.attrs.get("prep_lopass_hours", 0.0)),
            detrend=str(fly_ds.attrs.get("prep_detrend", "none")),
            normalize=str(fly_ds.attrs.get("prep_normalize", "none")),
            rolling_window_h=float(fly_ds.attrs.get("prep_rolling_window_h", 24.0)),
        )
        fly_ds = preprocess_activity(fly_ds, cfg, activity_var=activity_var, time_var=time_var)
    fly_data = fly_ds.sel(id=fly_id)
    activity = fly_data[activity_var].values.ravel()
    time_coords = fly_data[time_var].values

    # Match the production CWT worker: bridge interior gaps up to the ceiling so the
    # on-demand scalogram is computed on the same block the analysis run used. The
    # ceiling is read from the dataset attr the analysis run stamped
    # (cwt_max_bridge_gap_minutes), falling back to the single calibration default
    # for a legacy dataset. Transient, worker-local fill (§2a) — not persisted.
    _scalogram_bridge_ceiling = float(
        ds.attrs.get("cwt_max_bridge_gap_minutes", DEFAULT_MAX_BRIDGE_GAP_MINUTES)
    )
    activity, time_coords, _ = _extract_longest_continuous_block(
        activity,
        time_coords,
        bridge_small_gaps=True,
        max_bridge_gap_minutes=_scalogram_bridge_ceiling,
    )
    if activity is None or len(activity) < 10:
        raise ValueError(
            f"Fly {fly_id} has no valid activity data (no continuous block >= 10 points)."
        )

    if np.all(activity == 0) or np.std(activity) == 0:
        raise ValueError(f"Fly {fly_id} has no valid activity data.")

    min_period = ds.attrs.get("cwt_min_period", 16.0)
    max_period = ds.attrs.get("cwt_max_period", 32.0)
    resolution = ds.attrs.get("cwt_resolution", DEFAULT_CWT_RESOLUTION)
    wavelet_name = ds.attrs.get("cwt_wavelet", "cmor1.5-1.0")

    if cwt_method is None:
        # Prefer the method actually recorded on this dataset; only a legacy
        # dataset lacking the attr falls back, and it defers to the single
        # calibration default rather than a stale hardcoded method.
        cwt_method = ds.attrs.get("cwt_method", DEFAULT_CWT_METHOD)

    cwt_result = _preprocess_and_compute_cwt(
        activity,
        time_coords,
        min_period,
        max_period,
        cwt_method=cwt_method,
        resolution=resolution,
        wavelet=wavelet_name,
    )

    if cwt_result is None:
        raise ValueError(f"CWT computation failed for fly {fly_id}.")

    sampling_rate_min = cwt_result["sampling_rate_min"]
    n_timepoints = cwt_result["power"].shape[1]
    time_hours = np.arange(n_timepoints) * sampling_rate_min / 60.0

    # Cone of influence: max reliable period at each timepoint
    # At time t, the COI is min(t, N-1-t) / sqrt(2) * sampling_period_hours
    t_indices = np.arange(n_timepoints)
    edge_distance = np.minimum(t_indices, n_timepoints - 1 - t_indices)
    coi_hours = edge_distance * sampling_rate_min / (60.0 * np.sqrt(2))

    return {
        "power": cwt_result["power"],
        "periods_hours": cwt_result["periods_hours"],
        "time_hours": time_hours,
        "coi_hours": coi_hours,
        "ridge_periods": cwt_result["smoothed_periods_unmasked"],
        "avg_power": cwt_result["avg_power"],
        "activity": cwt_result["activity_processed"],
    }


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Autocorrelation Analysis
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Fixed lag axis (hours) for the stored per-fly ACF-vs-lag curve (correlogram).
# Every fly's curve is resampled onto this ONE shared grid so the per-fly
# Datasets concatenate on a common 'ac_lag' coord (mirrors how LS/MESA store
# their curves over a shared log-period grid). 72 h comfortably covers the
# AC peak-search window (default ac_peak=2, max_period=32 → search to 56 h; long-
# period lines search wider) plus context. 1-min resolution matches the data.
AC_CORRELOGRAM_MAX_LAG_HOURS = 72.0


def _ac_common_lag_grid_hours():
    """Return the fixed 'ac_lag' axis (hours, float32, 1-min steps) shared by
    every fly's stored ``ac_correlogram``. Computed from a constant so the
    computed and NaN-template Datasets carry a byte-identical coord (clean
    ``xr.concat``, no outer-join misalignment)."""
    n = int(round(AC_CORRELOGRAM_MAX_LAG_HOURS * 60.0))
    return (np.arange(n + 1, dtype=np.float64) / 60.0).astype(np.float32)


# ── Per-fly analysis skip reporting (shared by AC + MESA) ──────────────────
# When a per-fly worker cannot analyse a fly it returns _skip_result(...) instead
# of a bare None, tagging WHY. The main process then reports each skip with its
# reason: data-availability reasons are INFO (an EXPECTED exclusion — the
# retention-days floor, a dead/empty channel, a too-short record), and only a
# genuine numerical breakdown is a real "Warning:". Either way the fly is
# NaN-filled downstream (§2a: no valid measurement → NaN, never 0). This replaces
# the old blanket "analysis failed for fly X" line, which made a deliberate floor
# exclusion look like a crash.
# NOTE: keep these strings pure ASCII — they print to the app's console, which on
# Windows is cp1252 and raises UnicodeEncodeError on characters like the arrow.
_SKIP_MESSAGES = {
    "below_floor": "excluded - longest {phase} block {days:.1f} d < {floor:.1f} d retention floor",
    "no_data": "excluded - no valid {phase} data (dead/empty channel)",
    "too_short": "excluded - {phase} record too short to analyse",
    "bad_sampling": "excluded - could not determine a valid sampling rate",
    "numerical": "NUMERICAL FAILURE - {detail} -> filled with NaN",
}
_SKIP_IS_WARNING = {"numerical"}


def _skip_result(fly_id, reason, **detail):
    """Marker a per-fly worker returns (in place of a bare None) to record WHY it
    could not analyse the fly. Picklable across the process pool; the main process
    turns it into a reason-tagged message via :func:`_report_analysis_skips`."""
    fid = np.asarray(fly_id)
    fid = fid.item() if fid.shape == () else fid.ravel()[0]
    return {"_skip": True, "id": fid, "reason": reason, "detail": detail}


def _report_analysis_skips(kind, fail_dict, n_total, n_ok, phase="DD"):
    """Print one reason-tagged line per skipped fly plus a one-line summary.

    ``kind`` is the analysis name ('Autocorrelation' / 'MESA'). Output is
    deterministic (flies sorted by id). Data-availability reasons print as INFO;
    only genuine numerical failures print as ``Warning:``."""
    from collections import Counter

    counts = Counter()
    for fid in sorted(fail_dict, key=lambda x: str(x)):
        info = fail_dict[fid]
        reason = info.get("reason", "numerical")
        detail = info.get("detail", {})
        counts[reason] += 1
        tmpl = _SKIP_MESSAGES.get(reason, _SKIP_MESSAGES["numerical"])
        try:
            body = tmpl.format(
                phase=phase,
                days=detail.get("days", float("nan")),
                floor=detail.get("floor", float("nan")),
                detail=detail.get("detail", reason),
            )
        except Exception:
            body = f"excluded - {reason}"
        tag = "Warning: " if reason in _SKIP_IS_WARNING else ""
        print(f"{tag}{kind}: fly {fid} {body}")
    if counts:
        parts = ", ".join(f"{n} {r}" for r, n in counts.most_common())
        print(
            f"{kind}: {n_total} flies -> {n_ok} analysed; {sum(counts.values())} skipped ({parts})"
        )
    else:
        print(f"{kind}: {n_total} flies -> {n_ok} analysed; 0 skipped")


def _ac_single_fly_worker(args):
    """Worker function for autocorrelation.

    Consumes an already-preprocessed dataset (preprocessing — Butterworth
    low-pass, linear detrend, etc. — lives in :mod:`preprocessing`). Only
    longest-continuous-block extraction, mean-centering, and the AC math
    happen here. Defaults match SCAMP's ``dam_panels.m`` → ``autoco.m``
    when the input has been preprocessed with ``ac_default_config()``
    (lopass_hours=4.0, detrend='linear').

    Peak-picker tuning lives inline near the ``find_peaks`` call below
    (``PEAK_DISTANCE_SAMPLES``, ``PEAK_PROMINENCE_FRAC``) — change those
    constants when exploring different filter behaviour.
    """
    (
        fly_data,
        fly_id,
        activity_var,
        min_period,
        max_period,
        ac_peak,
        min_num_days,
        max_bridge_gap_minutes,
    ) = args

    activity_raw = fly_data[activity_var].values.ravel()
    time_raw = fly_data["time"].values
    # bridge_small_gaps=True: interior NaN dropouts whose missing span is ≤ the
    # max_bridge_gap_minutes ceiling are bridged by transient, worker-local linear
    # interpolation (AC also produces NaN on internal-NaN input). Filled values feed
    # AC only — NEVER written back to the Dataset/.nc (§2a). Gaps LARGER than the
    # ceiling stay hard breaks → longest-clean-segment.
    activity, time_coords, expected_delta_s = _extract_longest_continuous_block(
        activity_raw,
        time_raw,
        bridge_small_gaps=True,
        max_bridge_gap_minutes=max_bridge_gap_minutes,
    )
    if activity is None or len(activity) < 10:
        return _skip_result(fly_id, "no_data")

    if activity is None or len(activity) < 2 or np.all(activity == 0):
        return _skip_result(fly_id, "no_data")

    duration_days = (
        _time_span_seconds(time_coords[0], time_coords[-1]) + expected_delta_s
    ) / 86400.0
    if duration_days < min_num_days:
        return _skip_result(
            fly_id, "below_floor", days=float(duration_days), floor=float(min_num_days)
        )

    n_samples = len(activity)

    # Sampling rate from block extractor (median of all valid intervals)
    sampling_rate_min = expected_delta_s / 60.0
    if sampling_rate_min == 0:
        return _skip_result(fly_id, "bad_sampling")

    # Pearson autocorrelation (`xcorr(...,'coeff')` ≡ mean-centre then divide
    # by zero-lag value). After centring, auto_corr[0] = Σ(x−μ)², so dividing
    # each lag by it yields r(k) ∈ [−1, 1].
    activity = activity - np.mean(activity)
    auto_corr = signal.correlate(activity, activity, mode="full")
    auto_corr = auto_corr[len(auto_corr) // 2 :]  # Keep only the positive lags

    if auto_corr[0] <= 0:
        return _skip_result(fly_id, "no_data")  # constant / zero-variance signal
    auto_corr = auto_corr / auto_corr[0]

    # Store the ACF-vs-lag curve (correlogram) on the fixed shared lag axis so it
    # can be group-averaged on the Periodograms page. This fly's ACF lives at
    # sample lags k·sampling_rate_min/60 hours; resample onto the common grid.
    # Lags beyond this fly's record are MISSING → NaN (§2a), never 0 — so the
    # group nanmean averages only real overlap. Filled/stored as float32 (§2b).
    lag_grid_h = _ac_common_lag_grid_hours()
    sample_lags_h = np.arange(len(auto_corr), dtype=np.float64) * sampling_rate_min / 60.0
    correlogram = np.interp(
        lag_grid_h.astype(np.float64), sample_lags_h, auto_corr, left=auto_corr[0], right=np.nan
    ).astype(np.float32)

    # 95% CI matching SCAMP `autoco.m`: sqn = 1.965/sqrt(N).
    confidence_interval = 1.965 / np.sqrt(n_samples)

    # SCAMP `acplot.m` peak-search window:
    #   lag_lo = (acPeak-1)*24 + peakRange[0]
    #   lag_hi = (acPeak-1)*24 + peakRange[1]
    # With default acPeak=2, peakRange=[16,32] → search lag ∈ [40,56]h.
    # Reported period = peak_lag / acPeak (so a true 24-h rhythm yields 24,
    # regardless of which AC harmonic it was picked at).
    lag_lo_hours = (ac_peak - 1) * 24.0 + min_period
    lag_hi_hours = (ac_peak - 1) * 24.0 + max_period
    start_idx = int(lag_lo_hours * 60.0 / sampling_rate_min)
    end_idx = int(lag_hi_hours * 60.0 / sampling_rate_min)

    if start_idx >= len(auto_corr):
        return _skip_result(fly_id, "too_short")  # record shorter than the 2nd-day lag window
    end_idx = min(end_idx, len(auto_corr))

    # ────────────────────────────────────────────────────────────────────
    # AC PEAK FILTER PARAMETERS
    # Tune here when exploring peak-detection behaviour. Values chosen
    # for circadian-band autocorrelation; document any change in the
    # methods section.
    #   distance:   minimum sample-spacing between candidate peaks.
    #               Suppresses adjacent-sample noise ripples.
    #   prominence: required vertical distance from neighbouring valleys,
    #               expressed as a fraction of the search window's range.
    # See scipy.signal.find_peaks for the full argument set (height,
    # threshold, width, plateau_size are all available if needed).
    # ────────────────────────────────────────────────────────────────────
    PEAK_DISTANCE_SAMPLES = 8
    PEAK_PROMINENCE_FRAC = 0.05
    search = auto_corr[start_idx:end_idx]
    prom = PEAK_PROMINENCE_FRAC * (search.max() - search.min())
    peaks, _ = signal.find_peaks(search, distance=PEAK_DISTANCE_SAMPLES, prominence=prom)
    if len(peaks) == 0:
        # No prominent local maximum in the 2nd-day window — a monotonic/flat ACF,
        # i.e. an ARRHYTHMIC fly. Do NOT drop it to NaN: that silently removes a
        # real, valid record from the denominator and biases %rhythmic (MESA, which
        # always yields an argmax peak, keeps such flies — the AC/MESA N-asymmetry).
        # Fall back to the window's highest point. The resulting RI is low, so
        # classify_autocorrelation labels the fly ARRHYTHMIC (RI ≤ threshold) — it is
        # correctly COUNTED as arrhythmic instead of vanishing. Flies that DO have a
        # prominent peak never reach this branch, so every rhythmic ~24 h line stays
        # bit-identical (§4 invariant).
        best = int(np.argmax(search))
    else:
        best = peaks[np.argmax(search[peaks])]
    peak_lag_idx = best + start_idx
    peak_lag_hours = (peak_lag_idx * sampling_rate_min) / 60.0
    ac_power = auto_corr[peak_lag_idx]  # RI ∈ [-1, 1]

    # ────────────────────────────────────────────────────────────────────
    # Long-period anti-halving guard (convention repair).
    # The SCAMP ac_peak=2 convention reports period = peak_lag / ac_peak,
    # ASSUMING the lag selected in the [(ac_peak-1)*24+min, ...] window is the
    # ac_peak-th harmonic (e.g. a 24-h fly's 48-h 2nd-day peak ÷2 → 24 h). For a
    # genuine long-period (~43 h) fly the FIRST-day peak (~43 h lag) falls inside
    # that same window and the ÷ac_peak halving fabricates a spurious ~21 h
    # period. The two cases are separable by INTRINSIC geometry (no magic
    # number): if the selected lag L is truly the ac_peak-th harmonic of a
    # rhythm with period L/ac_peak, the autocorrelation at the implied
    # fundamental lag (L/ac_peak) must itself be a POSITIVE peak (a rhythm
    # repeats at its fundamental, not just its harmonic). For a long-period
    # fundamental, L/ac_peak lands in the first anticorrelation TROUGH → r < 0.
    # So: halve only when the implied-fundamental autocorrelation is positive
    # AND the selected peak is itself a real positive peak (r_sel > 0, which
    # also protects noise-only channels whose "peak" sits at r ≈ 0). When the
    # implied fundamental is anticorrelated, the selected lag IS the fundamental
    # → report it un-halved (the ac_peak=1 result). Measured separation on the
    # §2c control 24-h cohort vs the L775A ~43-h line is clean and sign-based:
    # control real flies r(L/2) ≥ +0.18; L775A real flies r(L/2) ≤ −0.07. This
    # leaves every normal ~24-h line BIT-IDENTICAL (the §4 invariant).
    fund_lag_idx = int(round(peak_lag_idx / ac_peak))
    apply_halving = True
    if ac_peak > 1 and 0 < fund_lag_idx < len(auto_corr):
        r_fundamental = auto_corr[fund_lag_idx]
        if r_fundamental < 0.0 and ac_power > 0.0:
            apply_halving = False
    effective_peak = ac_peak if apply_halving else 1
    ac_period_hours = peak_lag_hours / effective_peak  # SCAMP convention (long-period-guarded)

    # Rhythm Strength = peak / CI (SCAMP `rindex_raw.m:28`)
    rhythm_strength = ac_power / confidence_interval

    fly_id_scalar = (
        np.asarray(fly_id).item()
        if np.asarray(fly_id).shape == ()
        else np.asarray(fly_id).ravel()[0]
    )
    result_ds = xr.Dataset(
        {
            # §2b: per-fly AC results stored float32 (period ~24 h, RI ∈ [-1,1],
            # RS and CI small magnitudes — all within float32 precision; calc
            # stays float64 internally).
            "ac_period": (("id",), np.array([ac_period_hours], dtype=np.float32), {"units": "h"}),
            "ac_power": (
                ("id",),
                np.array([ac_power], dtype=np.float32),
                {"description": "Rhythmicity Index (RI)", "range": "[-1, 1]"},
            ),
            "ac_rhythm_strength": (
                ("id",),
                np.array([rhythm_strength], dtype=np.float32),
                {"description": "Rhythm Strength (RS = RI/CI)", "units": "standard deviations"},
            ),
            "ac_confidence_interval": (
                ("id",),
                np.array([confidence_interval], dtype=np.float32),
                {"description": "95% confidence interval", "units": "correlation units"},
            ),
            # Full ACF-vs-lag curve (correlation coefficient r ∈ [-1,1]) on the
            # shared lag axis. NaN beyond this fly's record (§2a). Consumed by the
            # Periodograms page for group-averaged correlograms.
            "ac_correlogram": (
                ("id", "ac_lag"),
                correlogram[np.newaxis, :],
                {"description": "Autocorrelation coefficient vs lag", "range": "[-1, 1]"},
            ),
        },
        coords={"id": np.array([fly_id_scalar]), "ac_lag": lag_grid_h},
    )
    return result_ds


def autocorrelation_analysis(
    ds,
    activity_var="activity",
    id_var="id",
    min_period=16,
    max_period=32,
    n_processes=None,
    ac_peak=2,
    min_num_days=0,
    max_bridge_gap_minutes=DEFAULT_MAX_BRIDGE_GAP_MINUTES,
    phase="auto",
    progress_callback=None,
    pool=None,
):
    """
    Perform autocorrelation analysis on an already-preprocessed dataset.

    Preprocessing (Butterworth low-pass, linear detrend, smoothing) is now
    external — apply :func:`preprocessing.preprocess_activity` with
    :func:`preprocessing.ac_default_config` before calling this function for
    SCAMP-style results. The worker only does longest-continuous-block
    extraction, mean-centering, and the AC math: Pearson AC → 2nd-day peak
    in lag window [(ac_peak−1)·24+min_period, (ac_peak−1)·24+max_period] h
    → period = peak_lag / ac_peak. See ``period_analysis_audit.md`` §4.4.

    Parameters
    ----------
    min_period, max_period : float
        Period search range in hours (default 16, 32 — SCAMP `peakRange`).
    ac_peak : int
        Which AC harmonic to pick (default 2 — SCAMP default). Set to 1 for
        the 1st-day peak (Rethomics convention).
    min_num_days : float
        Minimum required duration (days) of the longest continuous valid block.
    max_bridge_gap_minutes : float, default DEFAULT_MAX_BRIDGE_GAP_MINUTES (60)
        The ONE gap-bridge ceiling passed to the block extractor: an interior gap
        whose missing span is ≤ this many minutes is bridged (worker-local linear
        interpolation of the NaN cells, §2a-transient — never persisted); a gap
        larger than the ceiling stays a hard break → longest-clean-segment. 0 turns
        bridging off. Single-sourced in ``calibrations.DEFAULT_MAX_BRIDGE_GAP_MINUTES``.
    n_processes : int or None
        Number of parallel processes (None = auto-detect optimal).
    phase : str
        ``"auto"`` (default) selects DD if available, otherwise full dataset.
        ``"DD"``, ``"LD"``, or ``"both"`` to force a specific phase.
    """
    # Phase selection — default to DD when available
    analysis_ds, phase_used = _select_phase(ds, phase=phase)
    print("\nPerforming autocorrelation analysis...")
    print(
        f"Data phase: {phase_used}"
        + (
            " (DD available, selected automatically)"
            if phase == "auto" and phase_used == "DD"
            else ""
        )
    )

    fly_groups = list(analysis_ds.groupby(id_var))
    # max_bridge_gap_minutes rides in the args tuple exactly like min_num_days.
    args_list = [
        (
            group,
            fly_id,
            activity_var,
            min_period,
            max_period,
            ac_peak,
            min_num_days,
            max_bridge_gap_minutes,
        )
        for fly_id, group in fly_groups
    ]

    # Determine optimal number of workers
    if n_processes is None:
        n_processes = get_optimal_workers(len(args_list), use_gpu=PTWT_AVAILABLE)
        print(f"Using {n_processes} workers for autocorrelation analysis")

    results_dict = {}
    fail_dict = {}
    _ac_completed = 0
    _ac_total = len(args_list)
    with _worker_pool(pool, n_processes) as worker_pool:
        for result_ds in tqdm(
            worker_pool.imap_unordered(_ac_single_fly_worker, args_list),
            total=_ac_total,
            desc="Autocorrelation Analysis",
            unit="fly",
        ):
            if isinstance(result_ds, dict) and result_ds.get("_skip"):
                fail_dict[result_ds["id"]] = result_ds
            elif result_ds is not None:
                results_dict[result_ds["id"].values[0]] = result_ds
            _ac_completed += 1
            if progress_callback:
                progress_callback(_ac_completed, _ac_total)

    def create_nan_result(fly_id):
        fly_id_scalar = (
            np.asarray(fly_id).item()
            if np.asarray(fly_id).shape == ()
            else np.asarray(fly_id).ravel()[0]
        )
        # NaN template for unanalyzable flies (§2a NaN ≠ 0). Same float32 dtype as
        # the computed result above, so merging the two does NOT upcast the var back
        # to float64 (and float32 represents NaN exactly). The correlogram carries the
        # SAME 'ac_lag' coord so it concatenates cleanly with the computed results.
        lag_grid_h = _ac_common_lag_grid_hours()
        return xr.Dataset(
            {
                "ac_period": (("id",), np.array([np.nan], dtype=np.float32), {"units": "h"}),
                "ac_power": (
                    ("id",),
                    np.array([np.nan], dtype=np.float32),
                    {"description": "Rhythmicity Index (RI)", "range": "[-1, 1]"},
                ),
                "ac_rhythm_strength": (
                    ("id",),
                    np.array([np.nan], dtype=np.float32),
                    {"description": "Rhythm Strength (RS = RI/CI)", "units": "standard deviations"},
                ),
                "ac_confidence_interval": (
                    ("id",),
                    np.array([np.nan], dtype=np.float32),
                    {"description": "95% confidence interval", "units": "correlation units"},
                ),
                "ac_correlogram": (
                    ("id", "ac_lag"),
                    np.full((1, len(lag_grid_h)), np.nan, dtype=np.float32),
                    {"description": "Autocorrelation coefficient vs lag", "range": "[-1, 1]"},
                ),
            },
            coords={"id": np.array([fly_id_scalar]), "ac_lag": lag_grid_h},
        )

    # Create a result for each fly in original order (either computed or NaN template)
    all_results = []
    for fly_id, _ in fly_groups:
        if fly_id in results_dict:
            all_results.append(results_dict[fly_id])
        else:
            all_results.append(create_nan_result(fly_id))
    _report_analysis_skips(
        "Autocorrelation", fail_dict, _ac_total, len(results_dict), phase=phase_used
    )

    combined_results = xr.concat(all_results, dim=id_var)

    # Replace prior autocorrelation outputs when rerunning analysis.
    # xarray.merge raises MergeError on conflicting values by default.
    ac_output_vars = [var_name for var_name in ds.data_vars if var_name.startswith("ac_")]
    ds_without_ac = ds.drop_vars(ac_output_vars, errors="ignore")
    merged_ds = ds_without_ac.merge(combined_results)

    # Metadata for reproducibility. Preprocessing flags now live on
    # ``ds.attrs['preprocess_config']`` (stamped by ``preprocess_activity``).
    is_scamp_default = (
        ac_peak == 2
        and ds.attrs.get("prep_lopass_hours", 0.0) == 4.0
        and ds.attrs.get("prep_detrend", "none") == "linear"
    )
    merged_ds.attrs["ac_method"] = "scamp" if is_scamp_default else "custom"
    merged_ds.attrs["ac_peak"] = int(ac_peak)
    merged_ds.attrs["ac_normalized"] = True
    merged_ds.attrs["ac_min_num_days"] = min_num_days
    merged_ds.attrs["ac_max_bridge_gap_minutes"] = max_bridge_gap_minutes
    merged_ds.attrs["ac_min_period"] = min_period
    merged_ds.attrs["ac_max_period"] = max_period
    merged_ds.attrs["ac_phase"] = phase_used

    return merged_ds


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# MESA — Maximum Entropy Spectral Analysis (Burg autoregressive method)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# MESA fits an autoregressive (AR) model to the activity series and reads the
# power spectrum off the AR coefficients. Because it does not assume the signal
# is a sum of sinusoids over the whole record, it resolves sharp spectral peaks
# from short records — the reason it entered chronobiology (Dowse & Ringo 1989;
# Burg 1967/1975). SCAMP provides it via `per_mesa.m`.
#
# CRITICAL implementation point (verified against synthetic ground truth): AR
# spectral estimation needs a MODEST number of samples per cycle. At 1-min
# sampling a 24 h period is 1440 samples — an AR order large enough to span that
# is unstable and yields a monotonic (peak-less) spectrum. So the 1-min activity
# is first BINNED to ~30 min (48 samples/day, the Dowse/Ringo convention); a
# Burg AR order of ~20–30 then captures the circadian band cleanly (24 h→24.0,
# 43 h→43.3, 18 h→17.8 on synthetic tests). MESA also requires EVEN sampling
# (unlike LS), so it runs on the longest continuous block (gaps bridged ≤ ceiling,
# §2a-transient, like AC/CWT) and is sensitive to trend/mean (mean removed here;
# upstream detrend via preprocess_activity recommended). MESA is a period
# ESTIMATOR only — it carries no built-in significance test.
DEFAULT_MESA_BIN_MINUTES = 30.0  # bin 1-min activity to this before Burg AR
MESA_ORDER_CEILING = 150  # absolute stability/compute cap on the AR order
MESA_N_PERIOD_POINTS = 600  # log-spaced period-grid resolution for the PSD

# AR model-order selection. Dowse (2013, J. Circadian Rhythms) — the circadian-MESA
# reference — recommends a filter length of a FRACTION of the (binned) sample count
# N: minimum ~N/4, "good safe maximum" ~N/3, chosen so there are enough coefficients
# to resolve long circadian periods against noise. This is preferred over pure
# information criteria (FPE/AIC), which can UNDER-select for period resolution. So the
# default is N/3 (capped at MESA_ORDER_CEILING and N//2); FPE and a fixed integer order
# are offered as alternatives. (Order is a calibration coupled to bin size — surfaced.)
DEFAULT_MESA_ORDER_RULE = "n_over_3"


def _mesa_common_period_grid(min_period, max_period, n=MESA_N_PERIOD_POINTS):
    """Fixed log-spaced period axis (hours, float32) shared by every fly's stored
    ``mesa_periodogram`` for a given (min,max) — so per-fly Datasets concatenate on
    one byte-identical ``mesa_periodogram_periods`` coord (clean ``xr.concat``)."""
    return np.geomspace(float(min_period), float(max_period), int(n)).astype(np.float32)


def _burg_ar_psd(x, periods_hours, dt_hours, order):
    """AR power spectral density from Burg coefficients, evaluated on a period grid.

    For an AR(p) process x_t = Σ a_k x_{t-k} + e_t (statsmodels ``burg`` convention),
    S(f) = sigma2·dt / |1 − Σ a_k e^{−i2π f dt k}|², with f = 1/period (cycles/hr).
    """
    ar, sigma2 = _sm_burg(np.asarray(x, dtype=float), order=int(order))
    freqs = 1.0 / np.asarray(periods_hours, dtype=float)  # cycles per hour
    k = np.arange(1, len(ar) + 1)
    a_of_f = 1.0 - (np.exp(-1j * 2.0 * np.pi * np.outer(freqs, k) * dt_hours) @ ar)
    psd = sigma2 * dt_hours / (np.abs(a_of_f) ** 2)
    return psd, ar, float(sigma2)


def _mesa_select_order(x, max_order):
    """Akaike Final-Prediction-Error (FPE) order selection — the classic objective
    MESA criterion. Returns the order in [1, max_order] minimizing
    FPE(p) = sigma2_p·(N+p+1)/(N−p−1). Cheap: N is small after binning."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    best_order, best_fpe = 1, np.inf
    for p in range(1, int(max_order) + 1):
        denom = n - p - 1
        if denom <= 0:
            break
        try:
            _, sigma2 = _sm_burg(x, order=p)
        except Exception:
            break
        fpe = sigma2 * (n + p + 1) / denom
        if fpe < best_fpe:
            best_fpe, best_order = fpe, p
    return best_order


def _resolve_mesa_order(xb, order):
    """Resolve the AR order for a binned series ``xb`` from the ``order`` argument.

    ``None`` / ``'n_over_3'`` → Dowse's N/3 circadian default; ``'n_over_4'`` → N/4;
    ``'fpe'`` → Akaike FPE auto-selection; an int → that fixed order. Always clamped
    to [1, min(MESA_ORDER_CEILING, len(xb)//2)] for stability. Returns None if the
    series is too short to fit any AR model."""
    nb = len(xb)
    hard_cap = min(MESA_ORDER_CEILING, nb // 2)
    if hard_cap < 1:
        return None
    if order is None:
        order = DEFAULT_MESA_ORDER_RULE
    if isinstance(order, str):
        rule = order.lower()
        if rule == "fpe":
            return _mesa_select_order(xb, hard_cap)
        if rule == "n_over_4":
            return max(1, min(int(round(nb / 4)), hard_cap))
        # default 'n_over_3'
        return max(1, min(int(round(nb / 3)), hard_cap))
    return max(1, min(int(order), hard_cap))


def _mesa_single_fly_worker(args):
    """Worker: longest-block extraction → bin to ~30 min → Burg AR → AR-PSD peak.

    Consumes already-preprocessed data (detrend/smoothing external, like AC/LS).
    Returns a per-fly Dataset with mesa_period/mesa_power/mesa_order scalars and
    the full mesa_periodogram (PSD vs period) on the shared grid, or None (→ NaN
    template) if the fly cannot be analysed."""
    (
        fly_data,
        fly_id,
        activity_var,
        min_period,
        max_period,
        bin_minutes,
        order,
        min_num_days,
        max_bridge_gap_minutes,
    ) = args
    if _sm_burg is None:
        return _skip_result(fly_id, "numerical", detail="statsmodels burg unavailable")

    activity_raw = fly_data[activity_var].values.ravel()
    time_raw = fly_data["time"].values
    # Even sampling required → longest continuous block (≤ceiling gaps bridged,
    # §2a-transient worker-local; never persisted), exactly like AC/CWT.
    activity, time_coords, expected_delta_s = _extract_longest_continuous_block(
        activity_raw,
        time_raw,
        bridge_small_gaps=True,
        max_bridge_gap_minutes=max_bridge_gap_minutes,
    )
    if activity is None or len(activity) < 10 or np.all(activity == 0):
        return _skip_result(fly_id, "no_data")

    duration_days = (
        _time_span_seconds(time_coords[0], time_coords[-1]) + expected_delta_s
    ) / 86400.0
    if duration_days < min_num_days:
        return _skip_result(
            fly_id, "below_floor", days=float(duration_days), floor=float(min_num_days)
        )
    sampling_rate_min = expected_delta_s / 60.0
    if sampling_rate_min <= 0:
        return _skip_result(fly_id, "bad_sampling")

    # Bin to the MESA resolution (see the section header for WHY this is essential).
    bin_factor = max(1, int(round(bin_minutes / sampling_rate_min)))
    n_bins = len(activity) // bin_factor
    if n_bins < 16:  # too few samples for a meaningful AR spectrum
        return _skip_result(fly_id, "too_short")
    xb = activity[: n_bins * bin_factor].reshape(n_bins, bin_factor).sum(axis=1).astype(float)
    xb = xb - np.mean(xb)
    if not np.any(xb):
        return _skip_result(fly_id, "no_data")  # constant series after binning
    dt_hours = (bin_factor * sampling_rate_min) / 60.0

    use_order = _resolve_mesa_order(xb, order)
    if use_order is None:
        return _skip_result(fly_id, "too_short")  # series too short to fit any AR model

    periods = _mesa_common_period_grid(min_period, max_period)
    try:
        psd, _, _ = _burg_ar_psd(xb, periods, dt_hours, use_order)
    except Exception as e:
        return _skip_result(fly_id, "numerical", detail=f"Burg fit raised {type(e).__name__}")
    if not np.all(np.isfinite(psd)):
        return _skip_result(fly_id, "numerical", detail="non-finite AR PSD")

    peak_idx = int(np.argmax(psd))
    mesa_period = float(periods[peak_idx])
    med = float(np.median(psd))
    # peak / median PSD — an SNR-like prominence proxy (MESA has NO significance test)
    mesa_power = float(psd[peak_idx] / med) if med > 0 else np.nan

    # DISPLAY spectrum for the Periodograms group-average plot: an ADAPTIVE (FPE)
    # low-order Burg PSD. The N/3 estimation order (above) is Dowse's "safe maximum"
    # and gives the sharpest PEAK — but a high order also invents spurious low peaks
    # on arrhythmic flies (Burg line-splitting), which the group-average surfaces as a
    # wavy floor of "small humps". FPE adaptively selects a much lower order (~9 on
    # arrhythmic, ~46 on rhythmic), removing the splitting while preserving the
    # rhythmic peak. This curve is DISPLAY-ONLY; the reported period/power stay on the
    # N/3 fit (mesa_period/mesa_power/mesa_periodogram are untouched). Verified on the
    # control cohort: FPE keeps rhythmic period concordance (102/108 within 2 h of N/3)
    # while flattening the arrhythmic humps. The order default itself is a surfaced
    # calibration (Advanced MESA knobs) and is NOT changed here.
    display_order = _resolve_mesa_order(xb, "fpe")
    if display_order is None or display_order == use_order:
        psd_disp = psd
    else:
        try:
            psd_disp, _, _ = _burg_ar_psd(xb, periods, dt_hours, display_order)
            if not np.all(np.isfinite(psd_disp)):
                psd_disp = psd
        except Exception:
            psd_disp = psd

    fly_id_scalar = (
        np.asarray(fly_id).item()
        if np.asarray(fly_id).shape == ()
        else np.asarray(fly_id).ravel()[0]
    )
    return xr.Dataset(
        {
            # §2b: float32 for the results (period ~24 h, SNR proxy, order small).
            "mesa_period": (("id",), np.array([mesa_period], dtype=np.float32), {"units": "h"}),
            "mesa_power": (
                ("id",),
                np.array([mesa_power], dtype=np.float32),
                {"description": "peak PSD / median PSD (SNR proxy; MESA has no significance test)"},
            ),
            "mesa_order": (
                ("id",),
                np.array([float(use_order)], dtype=np.float32),
                {"description": "Burg AR model order used (FPE-selected unless fixed)"},
            ),
            "mesa_periodogram": (
                ("id", "mesa_periodogram_periods"),
                psd[np.newaxis, :].astype(np.float32),
                {"description": "AR (maximum-entropy) power spectral density vs period"},
            ),
            # Display curve (adaptive FPE order) — cleaner group-average, no high-order
            # line-splitting. Same shared coord. Reported period stays on mesa_periodogram.
            "mesa_display_periodogram": (
                ("id", "mesa_periodogram_periods"),
                psd_disp[np.newaxis, :].astype(np.float32),
                {
                    "description": "AR PSD at adaptive (FPE) order — "
                    "Periodograms-page display spectrum; period is "
                    "from the N/3 mesa_periodogram/mesa_period fit"
                },
            ),
        },
        coords={"id": np.array([fly_id_scalar]), "mesa_periodogram_periods": periods},
    )


def mesa_analysis(
    ds,
    activity_var="activity",
    id_var="id",
    min_period=16,
    max_period=36,
    n_processes=None,
    bin_minutes=DEFAULT_MESA_BIN_MINUTES,
    order=None,
    min_num_days=0,
    max_bridge_gap_minutes=DEFAULT_MAX_BRIDGE_GAP_MINUTES,
    phase="auto",
    progress_callback=None,
    pool=None,
):
    """Maximum Entropy Spectral Analysis (Burg AR) period estimation, per fly.

    Mirrors :func:`autocorrelation_analysis`: phase-select → per-fly workers →
    concat (NaN template for unanalyzable flies) → merge, replacing any prior
    ``mesa_*`` outputs. Stores per-fly ``mesa_period`` (peak of the AR spectrum in
    the search window), ``mesa_power`` (peak/median SNR proxy), ``mesa_order``
    (AR order used), and the full ``mesa_periodogram`` (id, mesa_periodogram_periods)
    for the Periodograms page's group-averaged AR spectra.

    Parameters
    ----------
    min_period, max_period : float
        Period search window (hours). Default (16, 36) matches the shared period
        control; the peak is taken over this window.
    bin_minutes : float
        Bin width the 1-min activity is aggregated to before the Burg fit
        (default 30 — the chronobiology MESA convention; see the section header).
    order : int, str or None
        AR model order. ``None``/``'n_over_3'`` (default) → Dowse's N/3 circadian
        rule; ``'n_over_4'`` → N/4; ``'fpe'`` → Akaike FPE auto-selection; an int →
        fixed order. Always clamped to min(MESA_ORDER_CEILING, n_bins//2).
    min_num_days : float
        Minimum required duration (days) of the longest continuous valid block.
    phase : str
        ``"auto"`` (DD if available, else full), ``"DD"``, ``"LD"``, or ``"both"``.
    """
    if _sm_burg is None:
        raise ImportError("MESA requires statsmodels (statsmodels.regression.linear_model.burg).")

    analysis_ds, phase_used = _select_phase(ds, phase=phase)
    print("\nPerforming MESA (maximum entropy / Burg AR) analysis...")
    print(
        f"Data phase: {phase_used}"
        + (
            " (DD available, selected automatically)"
            if phase == "auto" and phase_used == "DD"
            else ""
        )
    )

    fly_groups = list(analysis_ds.groupby(id_var))
    args_list = [
        (
            group,
            fly_id,
            activity_var,
            min_period,
            max_period,
            bin_minutes,
            order,
            min_num_days,
            max_bridge_gap_minutes,
        )
        for fly_id, group in fly_groups
    ]

    if n_processes is None:
        n_processes = get_optimal_workers(len(args_list), use_gpu=PTWT_AVAILABLE)
        print(f"Using {n_processes} workers for MESA analysis")

    results_dict = {}
    fail_dict = {}
    _completed = 0
    _total = len(args_list)
    with _worker_pool(pool, n_processes) as worker_pool:
        for result_ds in tqdm(
            worker_pool.imap_unordered(_mesa_single_fly_worker, args_list),
            total=_total,
            desc="MESA Analysis",
            unit="fly",
        ):
            if isinstance(result_ds, dict) and result_ds.get("_skip"):
                fail_dict[result_ds["id"]] = result_ds
            elif result_ds is not None:
                results_dict[result_ds["id"].values[0]] = result_ds
            _completed += 1
            if progress_callback:
                progress_callback(_completed, _total)

    period_grid = _mesa_common_period_grid(min_period, max_period)

    def create_nan_result(fly_id):
        fly_id_scalar = (
            np.asarray(fly_id).item()
            if np.asarray(fly_id).shape == ()
            else np.asarray(fly_id).ravel()[0]
        )
        # §2a NaN ≠ 0 template, same float32 dtype + same shared period coord so it
        # concatenates cleanly with the computed per-fly results.
        return xr.Dataset(
            {
                "mesa_period": (("id",), np.array([np.nan], dtype=np.float32), {"units": "h"}),
                "mesa_power": (
                    ("id",),
                    np.array([np.nan], dtype=np.float32),
                    {
                        "description": "peak PSD / median PSD (SNR proxy; MESA has no significance test)"
                    },
                ),
                "mesa_order": (
                    ("id",),
                    np.array([np.nan], dtype=np.float32),
                    {"description": "Burg AR model order used (FPE-selected unless fixed)"},
                ),
                "mesa_periodogram": (
                    ("id", "mesa_periodogram_periods"),
                    np.full((1, len(period_grid)), np.nan, dtype=np.float32),
                    {"description": "AR (maximum-entropy) power spectral density vs period"},
                ),
                "mesa_display_periodogram": (
                    ("id", "mesa_periodogram_periods"),
                    np.full((1, len(period_grid)), np.nan, dtype=np.float32),
                    {
                        "description": "AR PSD at adaptive (FPE) order — "
                        "Periodograms-page display spectrum; period is "
                        "from the N/3 mesa_periodogram/mesa_period fit"
                    },
                ),
            },
            coords={"id": np.array([fly_id_scalar]), "mesa_periodogram_periods": period_grid},
        )

    all_results = []
    for fly_id, _ in fly_groups:
        if fly_id in results_dict:
            all_results.append(results_dict[fly_id])
        else:
            all_results.append(create_nan_result(fly_id))
    _report_analysis_skips("MESA", fail_dict, _total, len(results_dict), phase=phase_used)

    combined_results = xr.concat(all_results, dim=id_var)

    mesa_output_vars = [v for v in ds.data_vars if v.startswith("mesa_")]
    ds_without_mesa = ds.drop_vars(mesa_output_vars, errors="ignore")
    merged_ds = ds_without_mesa.merge(combined_results)

    merged_ds.attrs["mesa_bin_minutes"] = float(bin_minutes)
    merged_ds.attrs["mesa_order"] = (
        DEFAULT_MESA_ORDER_RULE
        if order is None
        else (order if isinstance(order, str) else int(order))
    )
    merged_ds.attrs["mesa_min_period"] = min_period
    merged_ds.attrs["mesa_max_period"] = max_period
    merged_ds.attrs["mesa_min_num_days"] = min_num_days
    merged_ds.attrs["mesa_max_bridge_gap_minutes"] = max_bridge_gap_minutes
    merged_ds.attrs["mesa_phase"] = phase_used

    return merged_ds


# ===========================================================================
# Sleep State CWT Analysis (Abhilash et al. 2026)
# ===========================================================================


def _bin_offset(all_time, clean_time, bin_size):
    """Index of a clean run's first bin on the shared binned grid.

    Each fly is reduced to its longest continuous stretch, which may start well
    after the epoch does. The group surface is accumulated on one grid derived
    from the whole time axis, so every fly needs the offset of its own run —
    without it a fly whose data begins on day 3 gets averaged into day 0.

    Handles both time representations: relative integer minutes subtract
    directly, datetimes go through a timedelta.
    """
    if len(clean_time) == 0:
        return 0
    start, first = np.asarray(all_time)[0], np.asarray(clean_time)[0]
    if np.issubdtype(np.asarray(all_time).dtype, np.integer) or np.issubdtype(
        np.asarray(all_time).dtype, np.floating
    ):
        minutes = float(first) - float(start)
    else:
        minutes = (
            np.asarray(first, dtype="datetime64[s]")
            - np.asarray(start, dtype="datetime64[s]")
        ).astype("timedelta64[s]").astype(float) / 60.0
    return max(0, int(round(minutes / bin_size)))


# Bin width of the sleep-state CWT path, in minutes (STAR Methods).
SLEEP_CWT_BIN_MINUTES = 5


def sleep_cwt_analysis(
    ds,
    states=("standard", "short", "intermediate", "long"),
    circadian_range=(18, 30),
    ultradian_range_short_inter=(1, 4),
    ultradian_range_long=(2, 6),
    full_range=(1, 32),
    wavelet_name="cmor1.5-1.0",
    rectify_scale_bias=True,
    resolution=1 / 100,
    phase="auto",
    fly_ids=None,
    n_processes=None,
    progress_callback=None,
):
    """
    Apply CWT to binary sleep state time series (0/1 per minute).

    For each sleep state, the binary 1-minute time series is binned to 5-minute
    intervals by summing (values 0-5), reducing sparsity.  CWT is computed using
    the Morlet wavelet.

    Normalisation: each fly's 2D power matrix is divided by its mean cell value,
    then the normalised surfaces are averaged across flies. That order matters
    and it is the paper's: "we resolved each fly's timeseries in the
    time-frequency domain and then normalized it to the average amplitude of the
    surface. Next, we averaged the normalized surfaces across flies." It also
    matches the authors' own Shiny implementation
    (``phaseR/server.R``: ``power.norm[[i]] <- power.mat[[i]]/avg.power``,
    then ``apply(power.array, c(1,2), mean)``).

    Reference implementation
    ------------------------
    The paper's scalograms come from ``WaveletComp::analyze.wavelet``, called
    with ``loess.span = 0`` (no detrending), ``dt = 1``, ``dj = 1/100``, a
    single ``lowerPeriod``/``upperPeriod`` spanning the whole displayed range,
    and reading its ``$Power``. Three consequences are baked into the defaults
    here, each of which the previous defaults got wrong:

    - **One pass over the full range.** ``full_range`` now defaults to
      ``(1, 32)`` h, so the scalogram is the paper's Figure 5A surface — one
      log-period axis from ultradian to circadian. Previously it defaulted to
      ``None``, which ran two separate narrow-band CWTs (18-30 h and 1-4/2-6 h)
      and normalised each by its OWN band mean. Those z-values cannot be
      compared to the paper's 0-1.5 colour scale, or to each other, because
      the divisor differed per band. Pass ``full_range=None`` for the old
      split-band behaviour.
    - **Scale-rectified power** (``rectify_scale_bias``). WaveletComp computes
      ``Power = Mod(Wave)^2 / scale`` (Liu et al. 2007); a bare ``|W|^2`` is
      biased toward long periods. Measured on a synthetic 24 h + 3 h signal of
      EQUAL amplitude: unrectified reports the 24-h component 7.1x stronger
      than the 3-h one and places its peak at 24.51 h, while rectified reports
      them comparably (5.4 vs 4.8) and peaks at 24.00 h. Unrectified therefore
      understates ultradian power by roughly an order of magnitude relative to
      circadian — the ultradian:circadian ratio on that signal is 0.03
      unrectified against 0.26 rectified, and the paper's Figure 5B sits near
      the latter. See ``tests/test_sleep_cwt_truth.py``.
    - **``dj = 1/100``** (``resolution``), matching WaveletComp rather than the
      1/512 used before. 1/512 is five times finer at five times the cost and
      buys nothing the printed figure resolves.

    One documented difference remains: WaveletComp uses a Torrence & Compo
    Morlet with omega0 = 6, while ``wavelet_name`` defaults to PyWavelets'
    ``cmor1.5-1.0`` (omega0 = 2*pi ~ 6.28, bandwidth 1.5 against T&C's
    equivalent 2.0). That is the convention every other CWT path in this
    repo uses, and changing it would move existing period results, so it is
    left alone; it slightly favours time resolution over period resolution.
    WaveletComp also z-scores its input where this path only mean-centres,
    which cancels exactly under the divide-by-surface-mean normalisation.

    Parameters
    ----------
    ds : xr.Dataset
        Must contain sleep state variables ('sleep', 'sleep_short',
        'sleep_intermediate', 'sleep_long') from sleep_analysis().
    states : tuple of str
        States to analyse.  'standard' uses the 'sleep' variable.
    circadian_range : tuple of float
        (min, max) period in hours for circadian analysis. Default (18, 30).
    ultradian_range_short_inter : tuple of float
        (min, max) period in hours for short/intermediate sleep ultradian.
        Default (1, 4).
    ultradian_range_long : tuple of float
        (min, max) period in hours for long sleep ultradian. Default (2, 6).
    full_range : tuple of float or None
        Run a single CWT spanning this period range (default ``(1, 32)`` h,
        the paper's Figure 5A axis) instead of separate circadian + ultradian
        passes. The scalogram covers the entire range; ultradian amplitude is
        still extracted from the ultradian sub-range of the full scalogram.
        ``None`` restores the two narrow-band passes — see the note above on
        why their z-scales are not comparable to the paper's.
    rectify_scale_bias : bool
        Divide power by scale before normalising, as ``WaveletComp`` does
        (Liu et al. 2007). Default True. Setting this False reproduces the
        long-period bias described above and should only be used to reproduce
        an older run.
    resolution : float
        1 / voices-per-octave for the log period grid. Default 1/100, matching
        WaveletComp's ``dj = 1/100``.
    phase : str
        Phase selection: 'auto', 'DD', 'LD', or 'both'.
    fly_ids : array-like or None
        If provided, restrict analysis to these fly IDs (e.g. for per-group
        analysis). If None, all flies are used.
    n_processes : int, optional
        Number of parallel workers. None = auto.
    progress_callback : callable, optional
        Called with (completed, total) as each fly finishes.

    Returns
    -------
    xr.Dataset
        Variables per state and range:
          sleep_cwt_<state>_<range>_avg_surface   (period, time) group-avg scalogram
          sleep_cwt_<state>_<range>_period_axis    (period,)
          sleep_cwt_<state>_<range>_fly_power      (id, period) per-fly avg power spectrum
          sleep_cwt_<state>_ultradian_amplitude    (id, time) mean ultradian power per fly
    """
    # 5-minute binning is fixed by the method, not a per-fly choice: "Sleep
    # timeseries for all three states of sleep were binned in 5-min intervals
    # and subjected to Continuous Wavelet Transforms."
    bin_size = SLEEP_CWT_BIN_MINUTES

    var_map = {
        "standard": "sleep",
        "short": "sleep_short",
        "intermediate": "sleep_intermediate",
        "long": "sleep_long",
    }

    # Select analysis phase (reuse shared helper that handles already-split datasets)
    analysis_ds, phase_used = _select_phase(ds, phase=phase)

    # Subset to requested fly IDs
    if fly_ids is not None:
        analysis_ds = analysis_ds.sel(id=fly_ids)

    time_vals = analysis_ds["time"].values
    time_is_int = np.issubdtype(time_vals.dtype, np.integer)

    all_fly_ids = analysis_ds["id"].values
    n_flies = len(all_fly_ids)

    if n_processes is None:
        n_processes = min(4, max(1, mp.cpu_count() // 2))

    result_vars = {}
    completed = 0

    for state in states:
        var = var_map.get(state)
        if var is None or var not in analysis_ds.data_vars:
            print(f"sleep_cwt_analysis: variable '{var}' not found, skipping state '{state}'")
            continue

        # Determine ultradian range for this state
        if state == "long":
            ultradian_range = ultradian_range_long
        else:
            ultradian_range = ultradian_range_short_inter

        # Build range list: either one full-range pass or separate circadian + ultradian
        if full_range is not None:
            ranges_to_run = [("full", full_range)]
        else:
            ranges_to_run = [("circadian", circadian_range), ("ultradian", ultradian_range)]

        for range_name, period_range in ranges_to_run:
            min_p, max_p = period_range
            print(f"sleep_cwt_analysis: state={state}, range={range_name} ({min_p}-{max_p}h)")

            # Process each fly.
            #
            # The group-average surface is accumulated as a running SUM rather
            # than by keeping every fly's surface and stacking at the end. A
            # full-range surface is ~500 periods x ~2600 five-minute bins, so
            # 190 flies of them is about a gigabyte per state held at once —
            # enough to thrash a Streamlit worker. Nothing downstream needs the
            # individual surfaces: the two per-fly outputs (a time-averaged
            # spectrum and an ultradian amplitude trace) are both reductions
            # that can be taken as each fly finishes.
            #
            # Flies are accumulated ONTO A SHARED TIME GRID at each fly's own
            # offset, with a per-cell count, rather than being left-aligned and
            # cropped to the shortest. Both halves of that matter:
            #
            # - Cropping to the shortest run threw away most of the recording.
            #   On a real 31-fly group whose median clean run is 9 days, three
            #   flies with 1.1-, 2.3- and 3.4-day runs cut the GROUP surface to
            #   3.4 days. (Before each fly was reduced to its longest clean run
            #   this never bit, because every fly then had the full time axis.)
            # - Re-zeroing each run to t=0 misaligned them. A fly whose clean
            #   stretch begins on day 3 would have its day-3 column averaged
            #   into everyone else's day 0, smearing exactly the daily
            #   structure these scalograms exist to show — and mislabelling the
            #   "days since start" axis.
            #
            # A cell covered by no fly stays NaN rather than 0, so a partly
            # covered surface reads as missing instead of as an absence of
            # rhythm.
            n_bins_total = max(1, len(time_vals) // bin_size)
            surface_sum = None  # (n_periods, n_bins_total)
            surface_count = None
            fly_period_axes = None
            fly_spectra = []  # per-fly time-averaged spectrum, or None
            fly_ultradian_amps = []  # list of (offset, values) or None

            for fly_id in all_fly_ids:
                fly_da = analysis_ds[var].sel(id=fly_id)
                fly_arr = fly_da.values.astype(np.float32)

                # Missing minutes reach this function in TWO representations and
                # both have to become NaN before the extractor sees them: the
                # masks as written by sleep_analysis carry -1, but a
                # select_phase() view has already upcast them to float and put
                # NaN in the out-of-phase minutes. The old `where(arr < 0, 0)`
                # caught only the first, so a phase view fed NaN straight into
                # the transform and the whole averaged surface came out NaN.
                fly_arr = np.where(fly_arr < 0, np.nan, fly_arr)

                # Restrict to the fly's longest continuous stretch of real data,
                # as every other analysis in this module does. Without it the
                # out-of-phase half of a phase view would be carried into the
                # transform as a block of zeros, and the step at the epoch
                # boundary would put broadband power across every period.
                clean, clean_time, _ = _extract_longest_continuous_block(fly_arr, time_vals)
                if clean is None or len(clean) < 2:
                    fly_spectra.append(None)
                    fly_ultradian_amps.append(None)
                    continue

                # Bin to 5-minute intervals by summing
                if time_is_int:
                    n_bins = len(clean) // bin_size
                    if n_bins == 0:
                        fly_spectra.append(None)
                        fly_ultradian_amps.append(None)
                        continue
                    binned = clean[: n_bins * bin_size].reshape(n_bins, bin_size).sum(axis=1)
                    t_binned = np.arange(n_bins, dtype=np.int64) * bin_size
                else:
                    import pandas as _pd

                    df_tmp = _pd.DataFrame({"val": clean}, index=_pd.to_datetime(clean_time))
                    df_binned = df_tmp.resample(f"{bin_size}min").sum()
                    binned = df_binned["val"].values.astype(np.float32)
                    t_binned = df_binned.index.values
                    n_bins = len(binned)

                if n_bins < 10:
                    fly_spectra.append(None)
                    fly_ultradian_amps.append(None)
                    continue

                # CWT — only the raw power matrix is consumed below, so use
                # the lightweight ridge mode.
                # NOTE: resolution here is INTENTIONALLY independent of the
                # circadian-period default (DEFAULT_CWT_RESOLUTION). This is the
                # sleep-state SURFACE path, and its grid is pinned to
                # WaveletComp's dj = 1/100 so the surface matches the paper's.
                # Not the period-analysis default; do not fold the two together.
                cwt_result = _preprocess_and_compute_cwt(
                    binned,
                    t_binned,
                    min_p,
                    max_p,
                    cwt_method="ridge",
                    resolution=resolution,
                    wavelet=wavelet_name,
                )

                if cwt_result is None:
                    fly_spectra.append(None)
                    fly_ultradian_amps.append(None)
                    continue

                power = cwt_result["power"].astype(np.float64)  # (n_scales, n_time)
                fly_period_axes = cwt_result["periods_hours"]

                if rectify_scale_bias:
                    # WaveletComp: Power = Mod(Wave)^2 / scale (Liu et al. 2007).
                    # `scales` are periods expressed in SAMPLES, and this path
                    # samples every `bin_size` minutes — so convert the period
                    # axis to samples rather than reusing hours, which would
                    # rectify by a constant factor off the correct one.
                    scales_samples = fly_period_axes * 60.0 / bin_size
                    power = power / scales_samples[:, None]

                # Normalise to the mean of this fly's own surface, THEN average
                # across flies further down. Reversing those two steps weights
                # flies by how much they slept.
                mat_mean = np.mean(power)
                if mat_mean > 0:
                    power = power / mat_mean

                power = power.astype(np.float32)

                # Where this fly's clean run starts on the shared grid.
                offset = _bin_offset(time_vals, clean_time, bin_size)
                take = min(power.shape[1], n_bins_total - offset)
                if take <= 0:
                    fly_spectra.append(None)
                    fly_ultradian_amps.append(None)
                    continue

                if surface_sum is None:
                    surface_sum = np.zeros((power.shape[0], n_bins_total))
                    surface_count = np.zeros(n_bins_total, dtype=np.int32)
                surface_sum[:, offset : offset + take] += power[:, :take]
                surface_count[offset : offset + take] += 1

                fly_spectra.append(np.mean(power[:, :take], axis=1))

                # Extract ultradian amplitude from ultradian sub-range
                if range_name == "ultradian" or (
                    range_name == "full" and ultradian_range is not None
                ):
                    u_min, u_max = ultradian_range
                    u_mask = (fly_period_axes >= u_min) & (fly_period_axes <= u_max)
                    if np.any(u_mask):
                        # Carry the offset so the per-fly traces line up on the
                        # shared grid too, not just the averaged surface.
                        fly_ultradian_amps.append(
                            (offset, np.mean(power[u_mask, :take], axis=0))
                        )
                    else:
                        fly_ultradian_amps.append(None)
                else:
                    fly_ultradian_amps.append(None)

                completed += 1
                if progress_callback:
                    total_ops = n_flies * len(states) * len(ranges_to_run)
                    progress_callback(completed, total_ops)

            # Group-average normalised surfaces: per-cell mean over the flies
            # that actually cover each time bin.
            if surface_sum is None or fly_period_axes is None:
                continue
            if not surface_count.any():
                continue

            with np.errstate(invalid="ignore"):
                avg_surface = surface_sum / np.where(surface_count > 0, surface_count, np.nan)
            avg_surface = avg_surface.astype(np.float32)

            # Trim bins no fly covers off BOTH ends, so column 0 is the first
            # minute anyone contributed. The shared grid spans the whole time
            # axis, and a select_phase() view masks the out-of-phase epoch — on
            # a DD view of a 6-day recording that is two empty leading days,
            # which the scalogram would otherwise render as blank and label
            # "days since start of constant darkness". Interior gaps stay NaN.
            covered = np.flatnonzero(surface_count > 0)
            lo, hi = int(covered[0]), int(covered[-1]) + 1
            avg_surface = avg_surface[:, lo:hi]
            trim_lo, min_t = lo, avg_surface.shape[1]

            surf_var = f"sleep_cwt_{state}_{range_name}_avg_surface"
            pax_var = f"sleep_cwt_{state}_{range_name}_period_axis"
            fpow_var = f"sleep_cwt_{state}_{range_name}_fly_power"
            pdim = f"cwt_period_{state}_{range_name}"
            tdim = f"cwt_tbin_{state}_{range_name}"
            result_vars[surf_var] = xr.DataArray(
                avg_surface,
                dims=[pdim, tdim],
                attrs={
                    "state": state,
                    "range": range_name,
                    "min_period_h": float(min_p),
                    "max_period_h": float(max_p),
                },
            )
            result_vars[pax_var] = xr.DataArray(
                fly_period_axes,
                dims=[pdim],
                attrs={"units": "hours"},
            )

            # Per-fly time-averaged power spectra for bootstrap CI (Tab 2)
            fly_power_spectra = [
                s if s is not None else np.full(len(fly_period_axes), np.nan)
                for s in fly_spectra
            ]
            result_vars[fpow_var] = xr.DataArray(
                np.stack(fly_power_spectra, axis=0),
                dims=["id", pdim],
                coords={"id": all_fly_ids},
                attrs={"state": state, "range": range_name, "units": "normalised power"},
            )

            # Ultradian amplitude per fly
            has_ultradian = any(a is not None for a in fly_ultradian_amps)
            if has_ultradian:
                amp_arrays = []
                for amp in fly_ultradian_amps:
                    row = np.full(min_t, np.nan)
                    if amp is not None:
                        # Written in at the fly's own offset, so a fly whose
                        # clean run starts late leaves NaN before it rather
                        # than shifting its trace to day 0.
                        off, values = amp
                        # Same trim as the surface, so the amplitude traces and
                        # the scalogram share one x axis.
                        off -= trim_lo
                        start = max(0, off)
                        src = max(0, -off)
                        n = min(len(values) - src, min_t - start)
                        if n > 0:
                            row[start : start + n] = values[src : src + n]
                    amp_arrays.append(row)

                amp_tdim = f"cwt_tbin_{state}_ultradian"
                if range_name == "full":
                    amp_tdim = tdim  # reuse the full-range time dim
                amp_da = xr.DataArray(
                    np.stack(amp_arrays, axis=0),
                    dims=["id", amp_tdim],
                    coords={"id": all_fly_ids},
                    attrs={"state": state, "range": "ultradian"},
                )
                result_vars[f"sleep_cwt_{state}_ultradian_amplitude"] = amp_da

    if not result_vars:
        print("sleep_cwt_analysis: no results produced.")
        return xr.Dataset()

    return xr.Dataset(result_vars)


def ultradian_rhythmicity_ls(
    ds,
    states=("standard", "short", "intermediate", "long"),
    circadian_range=(18, 30),
    oversampling=8,
    fap_method="baluev",
    n_processes=None,
    progress_callback=None,
):
    """
    Test whether ultradian rhythm amplitude is circadian-gated using Lomb-Scargle.

    Applies Lomb-Scargle to the 'sleep_cwt_<state>_ultradian_amplitude' time
    series (mean ultradian CWT power per 5-min bin over days).  A significant
    result (low false-alarm probability) indicates that the strength of ultradian
    rhythms waxes and wanes with the circadian cycle.

    Uses the same generalized-LS engine as :func:`lomb_scargle_analysis`
    (via :func:`_run_ls_on_series`); the previous bespoke ``np.linspace``
    frequency grid and hardcoded FAP method are gone, so ultradian-LS
    numerical values shift to match the production circadian-LS grid.

    Replaces the chi-squared periodogram used in Abhilash et al. 2026 because
    Lomb-Scargle is more powerful, handles gappy series, and provides calibrated
    false-alarm probabilities (Riggle et al. 2022).

    Parameters
    ----------
    ds : xr.Dataset
        Must contain 'sleep_cwt_<state>_ultradian_amplitude' variables, as
        returned by sleep_cwt_analysis().
    states : tuple of str
    circadian_range : tuple of float
        (min, max) period in hours for the LS search. Default (18, 30).
    oversampling : int
        Rethomics-style ``ofac`` (default 8 — matches production LS).
    fap_method : str
        astropy LombScargle.false_alarm_probability method (default 'baluev').
    n_processes : int, optional
    progress_callback : callable, optional

    Returns
    -------
    xr.Dataset
        New variables per state:
          ultra_ls_period_<state>  - (id,) dominant period of amplitude modulation (hours)
          ultra_ls_power_<state>   - (id,) LS power at that period
          ultra_ls_fap_<state>     - (id,) false-alarm probability
    """
    min_p, max_p = circadian_range
    result_vars = {}

    for state in states:
        amp_var = f"sleep_cwt_{state}_ultradian_amplitude"
        if amp_var not in ds.data_vars:
            print(f"ultradian_rhythmicity_ls: '{amp_var}' not found, skipping.")
            continue

        amp_da = ds[amp_var]  # (id, cwt_time_bin)
        fly_ids = amp_da["id"].values

        periods_list = []
        powers_list = []
        faps_list = []

        for fly_id in fly_ids:
            fly_amp = amp_da.sel(id=fly_id).values.astype(np.float64)
            n_t = len(fly_amp)

            # Time axis: 5-minute bins converted to hours
            t_hours = np.arange(n_t, dtype=np.float64) * (5.0 / 60.0)

            valid = ~np.isnan(fly_amp)
            if valid.sum() < 10:
                periods_list.append(np.nan)
                powers_list.append(np.nan)
                faps_list.append(np.nan)
                continue

            ls_result = _run_ls_on_series(
                t_hours[valid],
                fly_amp[valid],
                min_period=min_p,
                max_period=max_p,
                oversampling=oversampling,
                fap_method=fap_method,
            )
            if ls_result is None:
                periods_list.append(np.nan)
                powers_list.append(np.nan)
                faps_list.append(np.nan)
            else:
                periods_list.append(ls_result["period"])
                powers_list.append(ls_result["power"])
                faps_list.append(ls_result["fap"])

            if progress_callback:
                progress_callback(1, 1)

        # §2b: period/power float32; fap kept float64 (Baluev FAP can underflow
        # float32, same rationale as ls_fap above).
        result_vars[f"ultra_ls_period_{state}"] = xr.DataArray(
            np.array(periods_list, dtype=np.float32),
            dims=["id"],
            coords={"id": fly_ids},
            attrs={"units": "hours", "state": state},
        )
        result_vars[f"ultra_ls_power_{state}"] = xr.DataArray(
            np.array(powers_list, dtype=np.float32),
            dims=["id"],
            coords={"id": fly_ids},
            attrs={"state": state},
        )
        result_vars[f"ultra_ls_fap_{state}"] = xr.DataArray(
            np.array(faps_list, dtype=np.float64),
            dims=["id"],
            coords={"id": fly_ids},
            attrs={"state": state, "significance_threshold": 0.05},
        )

    if not result_vars:
        return xr.Dataset()
    return xr.Dataset(result_vars)


def _chi_sq_periodogram(values, period_hours, sampling_min, alpha=0.05):
    """Sokolove & Bushell chi-squared periodogram, with adjusted power.

    Ported from ``zeitgebr::chi_sq_periodogram`` (rethomics), which is what the
    paper's own ``phase::indPeriodogramSleep(method = "ChiSquare")`` calls, so
    the numbers here are directly comparable to Figure 6B/D/F:

        col_num  = round(period * sampling_rate)          # period in samples
        Qp       = sum((col_means - grand_mean)^2) * N * (N / col_num)
                   / sum((values - grand_mean)^2)

    Two details are easy to get wrong and are the reason this is a port rather
    than a fresh implementation:

    - The number of cycles is **fractional** (``N / col_num``) and every sample
      is used, rather than truncating to whole cycles.
    - Degrees of freedom are ``col_num``, not ``col_num - 1``, and the
      significance threshold carries a Sidak correction across the trial
      periods: ``alpha' = 1 - (1 - alpha) ** (1 / n_periods)``.

    Returns
    -------
    dict with ``period`` (hours), ``power`` (Qp), ``threshold``, and
    ``adjusted`` = power - threshold. The paper plots ``adjusted``, which is
    why its y axis is "Adjusted chi-squared periodogram power" and its
    significance line sits at zero rather than at some period-dependent curve.
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    periods = np.asarray(period_hours, dtype=float)
    n = values.size
    if n < 2 or periods.size == 0:
        nan = np.full(periods.shape, np.nan)
        return {"period": periods, "power": nan, "threshold": nan, "adjusted": nan}

    grand = values.mean()
    denom = float(((values - grand) ** 2).sum())
    samples_per_hour = 60.0 / sampling_min

    power = np.full(periods.shape, np.nan)
    for i, p_h in enumerate(periods):
        col_num = int(round(p_h * samples_per_hour))
        if col_num < 1 or col_num > n or denom <= 0:
            continue
        cols = np.arange(n) % col_num
        col_sums = np.bincount(cols, weights=values, minlength=col_num)
        col_counts = np.bincount(cols, minlength=col_num)
        col_means = col_sums / np.maximum(col_counts, 1)
        power[i] = float(((col_means - grand) ** 2).sum()) * n * (n / col_num) / denom

    from scipy.stats import chi2

    corrected_alpha = 1.0 - (1.0 - alpha) ** (1.0 / len(periods))
    dof = np.array([max(1, int(round(p * samples_per_hour))) for p in periods])
    threshold = chi2.isf(corrected_alpha, dof)
    return {
        "period": periods,
        "power": power,
        "threshold": threshold,
        "adjusted": power - threshold,
    }


def ultradian_rhythmicity_chi_sq(
    ds,
    states=("standard", "short", "intermediate", "long"),
    period_range=(16, 32),
    time_resolution_min=20,
    alpha=0.05,
    bin_size_min=5,
):
    """Chi-squared periodogram of each fly's ultradian-amplitude time course.

    This is the test behind Figure 6B/D/F: "These amplitude time-courses were
    then subjected to chi-squared periodogram analyses to test for rhythmicity
    in ultradian rhythm amplitudes in each of the sleep states."

    It sits alongside :func:`ultradian_rhythmicity_ls` rather than replacing
    it. The Lomb-Scargle version is the better test on its merits — calibrated
    false-alarm probabilities, tolerant of gaps — but it is NOT the paper's
    test, so a Lomb-Scargle result cannot be checked against the paper's
    printed periodograms or against Table S2. Both are available: use this one
    to reproduce, that one to decide.

    Defaults are the paper's, via ``phase``'s own defaults: periods 16-32 h at
    20-minute resolution, alpha = 0.05. Figure 6's x axis is exactly 16-32 h.

    Parameters
    ----------
    ds : xr.Dataset
        Must contain ``sleep_cwt_<state>_ultradian_amplitude`` from
        :func:`sleep_cwt_analysis`.
    period_range, time_resolution_min, alpha
        Passed to the periodogram; see above.
    bin_size_min : int
        Sampling interval of the amplitude series, i.e. the CWT bin width.
        Must match ``sleep_cwt_analysis``'s 5-minute binning.

    Returns
    -------
    xr.Dataset
        Per state:
          ``ultra_chisq_power_<state>``     (id, period) Qp
          ``ultra_chisq_adjusted_<state>``  (id, period) Qp - threshold
          ``ultra_chisq_peak_period_<state>`` (id,) period of maximum adjusted power
          ``ultra_chisq_peak_adjusted_<state>`` (id,) that maximum
          ``ultra_chisq_rhythmic_<state>``  (id,) 1 if any adjusted power > 0
    """
    periods = np.arange(
        period_range[0], period_range[1] + 1e-9, time_resolution_min / 60.0
    )
    result_vars = {}

    for state in states:
        amp_var = f"sleep_cwt_{state}_ultradian_amplitude"
        if amp_var not in ds.data_vars:
            continue

        amp_da = ds[amp_var]
        fly_ids = amp_da["id"].values
        time_dim = [d for d in amp_da.dims if d != "id"][0]
        amps = amp_da.transpose("id", time_dim).values

        powers, adjusted, peak_p, peak_a, rhythmic = [], [], [], [], []
        for row in amps:
            res = _chi_sq_periodogram(
                row, periods, sampling_min=bin_size_min, alpha=alpha
            )
            powers.append(res["power"])
            adjusted.append(res["adjusted"])
            adj = res["adjusted"]
            if np.all(~np.isfinite(adj)):
                peak_p.append(np.nan)
                peak_a.append(np.nan)
                rhythmic.append(0)
            else:
                k = int(np.nanargmax(adj))
                peak_p.append(float(periods[k]))
                peak_a.append(float(adj[k]))
                rhythmic.append(int(adj[k] > 0))

        pdim = f"chisq_period_{state}"
        coords = {"id": fly_ids, pdim: periods}
        result_vars[f"ultra_chisq_power_{state}"] = xr.DataArray(
            np.stack(powers), dims=["id", pdim], coords=coords
        )
        result_vars[f"ultra_chisq_adjusted_{state}"] = xr.DataArray(
            np.stack(adjusted),
            dims=["id", pdim],
            coords=coords,
            attrs={"note": "Qp minus Sidak-corrected chi-squared threshold; >0 is significant"},
        )
        result_vars[f"ultra_chisq_peak_period_{state}"] = xr.DataArray(
            np.array(peak_p), dims=["id"], coords={"id": fly_ids}, attrs={"units": "hours"}
        )
        result_vars[f"ultra_chisq_peak_adjusted_{state}"] = xr.DataArray(
            np.array(peak_a), dims=["id"], coords={"id": fly_ids}
        )
        result_vars[f"ultra_chisq_rhythmic_{state}"] = xr.DataArray(
            np.array(rhythmic, dtype=np.int8), dims=["id"], coords={"id": fly_ids}
        )

    if not result_vars:
        print("ultradian_rhythmicity_chi_sq: no ultradian amplitude variables found.")
        return xr.Dataset()
    return xr.Dataset(result_vars)
