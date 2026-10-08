"""
preprocessing.py
================
Canonical preprocessing pipeline for circadian period analysis.

One PreprocessConfig — one ``preprocess_activity()`` call — applies binning,
smoothing, low-pass filtering, detrending, and normalization in a fixed order
to an xarray ``Dataset`` carrying the ``activity`` variable. The three
spectral algorithms in :mod:`periodograms` (Lomb-Scargle, autocorrelation,
CWT) consume an already-preprocessed dataset and only do their own algorithmic
math (longest-block extraction, mean-centering, the transform itself).

Step order (each step optional; sentinel value disables):
    bin → smooth → lopass → detrend → normalize

Mean-centering is NOT in this pipeline — it is per-method math and stays in
the algorithm workers.

NaN preservation
----------------
All steps except binning operate per fly with NaN preservation: for each fly,
the finite samples are gathered, transformed, and scattered back to their
original positions. NaN gaps are preserved as-is. This avoids
``gaussian_filter1d`` / ``signal.filtfilt`` treating NaN as 0.

Binning uses ``xarray.coarsen(time=...).sum(skipna=True)``, so NaN-only bins
become NaN; bins with at least one finite sample carry the partial sum.

Audit trail
-----------
The output dataset has ``ds.attrs['preprocess_config']`` (full config dict)
plus per-field ``prep_<key>`` attrs.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
import xarray as xr
from scipy import signal as _signal
from scipy.ndimage import gaussian_filter1d

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PreprocessConfig:
    """Single source of truth for activity preprocessing parameters.

    Each field has a sentinel that disables the corresponding step:

    * ``bin_minutes`` 0 or 1            → no rebin
    * ``smooth_sigma_min`` 0.0          → no smoothing
    * ``lopass_hours`` 0.0              → no low-pass filter
    * ``detrend`` ``"none"``            → no detrending
    * ``normalize`` ``"none"``          → no normalization

    Defaults are all "no-op" so building a default config returns the input
    unchanged. Per-method recommended defaults live in
    ``app/preprocessing_widgets.py`` and the audit doc (e.g. AC's SCAMP-style
    defaults are ``lopass_hours=4.0, detrend='linear'``).
    """

    bin_minutes: int = 0
    smooth_sigma_min: float = 0.0
    lopass_hours: float = 0.0
    detrend: str = "none"  # "none" | "linear"
    normalize: str = "none"  # "none" | "zscore" | "robust" | "envelope" | "rolling"
    rolling_window_h: float = 24.0  # only used if normalize == "rolling"


# ---------------------------------------------------------------------------
# Time-axis helpers
# ---------------------------------------------------------------------------


def _infer_dt_minutes(time_vals: np.ndarray) -> float:
    """Sampling interval in minutes (median of diffs). Returns 1.0 for short
    arrays so callers can short-circuit safely."""
    if len(time_vals) < 2:
        return 1.0
    if np.issubdtype(time_vals.dtype, np.datetime64):
        diffs = np.diff(time_vals).astype("timedelta64[s]").astype(float) / 60.0
    else:
        diffs = np.diff(time_vals.astype(float))
    diffs = diffs[diffs > 0]
    if len(diffs) == 0:
        return 1.0
    return float(np.median(diffs))


# ---------------------------------------------------------------------------
# Step 1: Binning (full-array; NaN handled by skipna=True)
# ---------------------------------------------------------------------------


def _rebin_activity(
    ds: xr.Dataset, bin_minutes: int, activity_var: str = "activity", time_var: str = "time"
) -> xr.Dataset:
    """Sum-bin ``activity`` along time. ``bin_minutes <= 1`` is a no-op.

    Coords listed in ``COORDS_TO_PRESERVE`` are reattached to the binned
    dataset since ``coarsen`` drops anything not on the time axis.
    """
    if bin_minutes is None or bin_minutes <= 1:
        return ds
    dt_min = _infer_dt_minutes(ds[time_var].values)
    n_samples = int(round(bin_minutes / dt_min)) if dt_min > 0 else bin_minutes
    if n_samples <= 1:
        return ds
    activity_binned = (
        ds[activity_var].coarsen({time_var: n_samples}, boundary="trim").sum(skipna=True)
    )
    new_ds = xr.Dataset(
        {activity_var: activity_binned},
        coords={"id": ds["id"], time_var: activity_binned[time_var]},
    )
    for c in (
        "group",
        "start_datetime",
        "stop_datetime",
        "genotype",
        "temperature",
        "file",
        "first_DD_day",
    ):
        if c in ds.coords:
            new_ds = new_ds.assign_coords({c: ds[c]})
    new_ds.attrs = dict(ds.attrs)
    # Recompute sampling interval after binning
    new_ds.attrs["sampling_interval"] = float(bin_minutes)
    return new_ds


# ---------------------------------------------------------------------------
# Per-fly NaN-preserving transforms
# ---------------------------------------------------------------------------


def _smooth_finite(col: np.ndarray, sigma_samples: float) -> np.ndarray:
    """Gaussian smoothing of the finite samples in ``col`` (1-D), preserving
    NaN positions. This is the canonical NaN-preserving smoother for the
    period-analysis preprocessing path.

    Finite samples are gathered into a dense 1-D array, smoothed with
    ``gaussian_filter1d(..., mode='reflect')``, and scattered back to their
    original indices.
    """
    if sigma_samples <= 0:
        return col
    valid = np.isfinite(col)
    if not valid.any():
        return col
    out = col.astype(np.float64, copy=True)
    finite = out[valid]
    if len(finite) >= 2:
        finite = gaussian_filter1d(finite, sigma=sigma_samples, mode="reflect")
    out[valid] = finite
    return out


def _lopass_finite(col: np.ndarray, lopass_hours: float, sample_rate_per_h: float) -> np.ndarray:
    """2nd-order zero-phase Butterworth low-pass on the finite span.

    Mirrors SCAMP ``butt_filter.m``: 2nd-order Butterworth, ``filtfilt``,
    ``padtype='odd'``, ``padlen`` capped to ``len(x) - 1``. Filter is applied
    on the dense vector of finite samples; NaN positions are preserved.
    """
    if lopass_hours <= 0:
        return col
    valid = np.isfinite(col)
    if valid.sum() < 4:
        return col
    nyquist = sample_rate_per_h / 2.0
    wn = (1.0 / lopass_hours) / nyquist
    if not (0 < wn < 1):
        return col
    b, a = _signal.butter(2, wn, btype="lowpass")
    out = col.astype(np.float64, copy=True)
    finite = out[valid]
    padlen = min(3 * max(len(a), len(b)), len(finite) - 1)
    if padlen < 0:
        return col
    finite = _signal.filtfilt(b, a, finite, padtype="odd", padlen=padlen)
    out[valid] = finite
    return out


def _detrend_finite(col: np.ndarray, kind: str) -> np.ndarray:
    """Linear detrend (SCAMP ``notrend.m``) on the finite span. NaN-preserving."""
    if kind == "none":
        return col
    if kind != "linear":
        raise ValueError(f"Unknown detrend kind: {kind!r}")
    valid = np.isfinite(col)
    if valid.sum() < 2:
        return col
    out = col.astype(np.float64, copy=True)
    out[valid] = _signal.detrend(out[valid], type="linear")
    return out


def _normalize_finite(col: np.ndarray, kind: str, rolling_samples: int) -> np.ndarray:
    """Per-fly normalization on the finite span. NaN-preserving.

    Supported:
      * ``"none"``     — no-op
      * ``"zscore"``   — (x - mean) / std
      * ``"robust"``   — (x - median) / MAD
      * ``"envelope"`` — x / mean(|x|), Riggle-style envelope normalization
      * ``"rolling"``  — x − rolling-mean(x, ``rolling_samples`` samples)
    """
    if kind == "none":
        return col
    valid = np.isfinite(col)
    if valid.sum() < 2:
        return col
    out = col.astype(np.float64, copy=True)
    x = out[valid]

    if kind == "zscore":
        sd = float(np.std(x))
        if sd == 0 or not np.isfinite(sd):
            x = x - float(np.mean(x))
        else:
            x = (x - float(np.mean(x))) / sd
    elif kind == "robust":
        med = float(np.median(x))
        mad = float(np.median(np.abs(x - med)))
        if mad == 0 or not np.isfinite(mad):
            x = x - med
        else:
            x = (x - med) / mad
    elif kind == "envelope":
        mean_abs = float(np.mean(np.abs(x)))
        if mean_abs == 0 or not np.isfinite(mean_abs):
            x = x.copy()
        else:
            x = x / mean_abs
    elif kind == "rolling":
        win = max(1, int(rolling_samples))
        if len(x) < win:
            x = x - float(np.mean(x))
        else:
            s = pd.Series(x)
            roll = (
                s.rolling(window=win, min_periods=max(1, win // 4), center=True).mean().to_numpy()
            )
            x = x - roll
    else:
        raise ValueError(f"Unknown normalize kind: {kind!r}")

    out[valid] = x
    return out


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def _is_noop(config: PreprocessConfig) -> bool:
    """True iff ``config`` would leave activity numerically unchanged.

    All five steps disabled (bin, smooth, lopass, detrend, normalize) ⇒ the
    materialize-and-rewrap path is pure overhead. Used to short-circuit
    :func:`preprocess_activity` for the LS and CWT default configs (both
    no-ops); the AC default has real preprocessing so it falls through.
    """
    return (
        config.bin_minutes in (0, 1)
        and config.smooth_sigma_min == 0.0
        and config.lopass_hours == 0.0
        and config.detrend == "none"
        and config.normalize == "none"
    )


def preprocess_activity(
    ds: xr.Dataset,
    config: PreprocessConfig,
    *,
    activity_var: str = "activity",
    time_var: str = "time",
    id_var: str = "id",
) -> xr.Dataset:
    """Apply ``config`` to ``ds[activity_var]`` per fly, preserving NaNs and coords.

    Parameters
    ----------
    ds
        Input dataset with at least ``activity_var`` along ``(time_var, id_var)``
        (or ``(id_var, time_var)``).
    config
        :class:`PreprocessConfig` describing the steps.
    activity_var, time_var, id_var
        Variable / coordinate names.

    Returns
    -------
    xr.Dataset
        New dataset with the transformed activity (float32 to match upstream
        storage). All other variables and coords are preserved. ``ds.attrs``
        gain ``preprocess_config`` and per-field ``prep_<key>`` entries.
    """
    if activity_var not in ds:
        raise KeyError(f"{activity_var!r} not in dataset")

    cfg_dict = asdict(config)

    # Short-circuit when nothing changes. Avoids an unnecessary materialize +
    # float64 round-trip + re-wrap for LS/CWT default configs.
    if _is_noop(config):
        out = ds.copy()
        out.attrs["preprocess_config"] = str(cfg_dict)
        for k, v in cfg_dict.items():
            out.attrs[f"prep_{k}"] = v
        return out

    # 1. Binning — full-array op
    out = _rebin_activity(ds, config.bin_minutes, activity_var=activity_var, time_var=time_var)

    # Recompute dt after binning
    dt_min = _infer_dt_minutes(out[time_var].values)
    sample_rate_per_h = 60.0 / dt_min if dt_min > 0 else 0.0

    # 2-5. Per-fly NaN-preserving transforms
    sigma_samples = (config.smooth_sigma_min / dt_min) if dt_min > 0 else 0.0
    rolling_samples = (
        max(1, int(round(config.rolling_window_h * 60.0 / dt_min))) if dt_min > 0 else 1
    )

    activity_da = out[activity_var]

    # Identify time axis (we operate per fly: iterate over id dim)
    dims = activity_da.dims
    if id_var not in dims or time_var not in dims:
        raise ValueError(
            f"{activity_var} must have both {id_var!r} and {time_var!r} dims; got {dims}"
        )

    # Transpose to (time, id) for column-major iteration matching legacy code
    activity_2d = activity_da.transpose(time_var, id_var).values  # (T, N)
    out_arr = activity_2d.astype(np.float64, copy=True)

    n_flies = out_arr.shape[1]
    needs_smooth = config.smooth_sigma_min > 0 and sigma_samples > 0
    needs_lopass = config.lopass_hours > 0 and sample_rate_per_h > 0
    needs_detrend = config.detrend != "none"
    needs_normalize = config.normalize != "none"

    if needs_smooth or needs_lopass or needs_detrend or needs_normalize:
        for i in range(n_flies):
            col = out_arr[:, i]
            if needs_smooth:
                col = _smooth_finite(col, sigma_samples)
            if needs_lopass:
                col = _lopass_finite(col, config.lopass_hours, sample_rate_per_h)
            if needs_detrend:
                col = _detrend_finite(col, config.detrend)
            if needs_normalize:
                col = _normalize_finite(col, config.normalize, rolling_samples)
            out_arr[:, i] = col

    # Restore original dim order
    new_da = xr.DataArray(
        out_arr.astype(np.float32),
        dims=(time_var, id_var),
        coords={time_var: out[time_var], id_var: out[id_var]},
        attrs=dict(activity_da.attrs),
    ).transpose(*dims)

    new_ds = out.copy()
    new_ds[activity_var] = new_da

    # Stamp audit trail (cfg_dict is built at the top of the function so the
    # no-op short-circuit can reuse it).
    new_ds.attrs["preprocess_config"] = str(cfg_dict)
    for k, v in cfg_dict.items():
        new_ds.attrs[f"prep_{k}"] = v

    return new_ds


# ---------------------------------------------------------------------------
# Convenience constructors for per-method documented defaults
# ---------------------------------------------------------------------------


def ls_default_config() -> PreprocessConfig:
    """Lomb-Scargle default — Rethomics ``zeitgebr::ls_periodogram`` does no
    preprocessing besides mean-centering (which stays in the LS worker)."""
    return PreprocessConfig()


def ac_default_config() -> PreprocessConfig:
    """Autocorrelation default — SCAMP ``autoco.m`` pipeline:
    2nd-order Butterworth low-pass at 4 h + linear detrend.
    """
    return PreprocessConfig(lopass_hours=4.0, detrend="linear")


def cwt_default_config() -> PreprocessConfig:
    """CWT default — Rethomics/WaveletComp ``loess.span=0`` convention: no
    preprocessing besides mean-centering (which stays in the CWT worker)."""
    return PreprocessConfig()


__all__ = [
    "PreprocessConfig",
    "preprocess_activity",
    "ls_default_config",
    "ac_default_config",
    "cwt_default_config",
]
