"""
rhythmicity_classification.py
==============================
Per-algorithm rhythmic/arrhythmic classification for Drosophila DAM data.

Supports three independent classifiers, each grounded in the chronobiology
literature:

1. Lomb-Scargle (LS)
   Metric:  false-alarm probability (FAP) at the peak. Computed via astropy's
            generalized (Zechmeister-Kürster 2009) periodogram with
            ``fit_mean=True`` (floating constant per trial frequency),
            ``center_data=True``, and ``normalization='standard'`` (R²-style
            ∈ [0, 1]). FAP via ``false_alarm_probability(method='baluev')``.
            Z-scoring the input is recommended for cross-recording amplitude
            comparability but is not required for correct periodogram
            computation.
   Rule:    FAP < threshold AND peak period in [min, max] h
   Default: FAP < 0.05, period in 16–32 h
   Source:  Zechmeister & Kürster 2009 A&A 496:577 (generalized LS); Baluev
            2008 MNRAS 385:1279 (FAP extreme-value bound); VanderPlas 2018
            ApJS 236:16 (astropy LS reference); Refinetti 2007 Biol Rhythm
            Res; Horne & Baliunas 1986 ApJ. Note: zeitgebr/Rethomics'
            Press-style ``exp(-P)·N·2/ofac`` formula requires Scargle
            variance-scaled power (unbounded) and is not valid for astropy's
            'standard' normalization, so it is not offered here.

2. Autocorrelation (AC)
   Metric:  RI = raw peak autocorrelation in the period search window
            (Levine "Rhythmicity Index"). Exposed as `ac_power` in the dataset.
            With the SCAMP-style pipeline, the peak is taken from the 2nd-day
            window [40, 56] h (set by ``ac_peak=2``) — the threshold below is
            applied to that 2nd-day RI.
   Rule:    RI > threshold (default 0.3, a stricter practical cutoff);
            optional dynamic 2/sqrt(N) 95% CI overlay from Levine 2002.
   Default: RI > 0.3
   Source:  Levine et al. 2002 BMC Neurosci; SCAMP ``rindex_raw.m`` /
            ``autoco.m`` (Vecsey et al. 2024 CSH Protoc); Meireles-Filho
            et al. 2018 Behav Genet.

   The dataset also exposes `ac_rhythm_strength` defined as
   ac_power / (1.965/sqrt(n)). This is SCAMP's ``rs = py / in`` from
   ``rindex_raw.m`` line 28, retained for display/export. Use it for the
   normalized statistical interpretation (RI in 95% CI units). The primary
   gating metric is RI (ac_power).

3. CWT
   Metric:  `cwt_rhythmicity`. With ``cwt_method='global_rednoise'`` (default;
            calibrations.DEFAULT_CWT_METHOD), it is the COI-excluded
            global-spectrum peak divided by the AR(1) red-noise EXPECTED power
            AT the peak period — a range-robust, length-stable local-prominence
            strength (threshold DEFAULT_CWT_REDNOISE_THRESHOLD). With
            ``cwt_method='ar1'`` (former default) it is the **time-averaged
            Power.avg at the dominant period divided by the 95% AR(1) red-noise
            threshold** (Torrence & Compo 1998 BAMS, eqs 16 & 25): continuous
            and unbounded; >1 means the peak power is significant at α=0.05
            against the AR(1) red-noise null. With
            ``cwt_method='ridge'`` it is the fraction of analyzable
            timepoints with a confident circadian ridge peak.
   Rule:    cwt_rhythmicity > threshold AND cwt_period in [min, max] h.
            ``classify_cwt`` auto-selects a sensible default per method:
              ar1          -> 1.0 (peak SNR above the 95% AR(1) threshold)
              ridge        -> 0.30 (lab choice; no published binary cutoff)
   Source:  Torrence & Compo 1998 BAMS 79:61 (AR(1) red-noise wavelet
            significance, eqs 16, 18); Leise & Harrington 2011 JBR / Leise
            2013 J Circadian Rhythms (CWT in circadian behavior).

Classification results are written to the dataset as boolean coordinates on
the `id` dimension (`ls_rhythmic`, `ac_rhythmic`, `cwt_rhythmic`), enabling
xarray-native downstream filtering:

    rhythmic = ds.where(ds['ac_rhythmic'], drop=True)

Thresholds used are recorded in `ds.attrs` for reproducibility.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

# ---------------------------------------------------------------------------
# Threshold defaults — re-exported from the single calibration home
# ---------------------------------------------------------------------------
# These are the canonical rhythmic-call thresholds. They are DEFINED ONCE in
# ``calibrations.py`` (the documented home for all user-owned soft thresholds)
# and re-exported here so the classifier defaults below, and the page-3 UI
# (app/pages/3_*), read the same single source. Edit values in calibrations.py,
# not here.
from calibrations import (  # noqa: E402  (re-export; single source of truth)
    DEFAULT_AC_RI_THRESHOLD,
    DEFAULT_CWT_AR1_THRESHOLD,
    DEFAULT_CWT_GLOBAL_THRESHOLD,
    DEFAULT_CWT_MAX_PERIOD,
    DEFAULT_CWT_METHOD,
    DEFAULT_CWT_MIN_PERIOD,
    DEFAULT_CWT_REDNOISE_THRESHOLD,
    DEFAULT_CWT_RIDGE_THRESHOLD,
    DEFAULT_LS_POWER_THRESHOLD,
)


def cwt_threshold_for(method: str) -> float:
    """Return the canonical CWT rhythmicity threshold for the given method.

    ``ar1`` → 1.0; ``global`` → the length-stable
    normalized-strength cutoff (DEFAULT_CWT_GLOBAL_THRESHOLD, provisional);
    ``global_rednoise`` → the range-robust local-prominence cutoff
    (DEFAULT_CWT_REDNOISE_THRESHOLD, provisional); ``ridge`` → 0.30. The
    comparison-only candidates ``global_baseline`` / ``global_neighbor`` have no
    dedicated calibrated cutoff and fall back to the ar1 threshold (their scales
    differ; do not gate production on them). Unknown methods fall back to ar1.
    """
    return {
        "ar1": DEFAULT_CWT_AR1_THRESHOLD,
        "global": DEFAULT_CWT_GLOBAL_THRESHOLD,
        "global_rednoise": DEFAULT_CWT_REDNOISE_THRESHOLD,
        "ridge": DEFAULT_CWT_RIDGE_THRESHOLD,
    }.get(method, DEFAULT_CWT_AR1_THRESHOLD)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _apply_period_window(
    period_da: xr.DataArray, period_window: tuple[float, float]
) -> xr.DataArray:
    """Boolean mask: ``period_window[0] <= period_da <= period_window[1]``.
    Shared by all three classifiers so the gating logic is identical."""
    lo, hi = period_window
    return (period_da >= lo) & (period_da <= hi)


def apply_rhythmic_filter(ds: xr.Dataset, gate: str = "ac", enabled: bool = True) -> xr.Dataset:
    """
    Drop arrhythmic flies from ``ds`` along the ``id`` dimension.

    Used to gate group-level statistics and aggregate plots so that group
    period estimates aren't biased by flies whose period values are
    essentially random (i.e. arrhythmic). Disable when the biology of
    interest *is* loss of rhythmicity (e.g. clock-disrupting treatments).

    Parameters
    ----------
    ds : xr.Dataset
        Per-fly dataset with rhythmic-flag coords assigned by ``classify_*``.
    gate : {'ls', 'ac', 'cwt'}
        Which classifier's flag drives the filter. Default ``'ac'`` (the AC
        RI gate, see ``classify_autocorrelation``). The corresponding
        ``<gate>_rhythmic`` coord must exist; otherwise ``ds`` is returned
        unchanged.
    enabled : bool
        If False, returns ``ds`` unchanged. Default True.

    Returns
    -------
    xr.Dataset
        ``ds`` restricted to flies flagged rhythmic by ``gate``. If the
        flag coord is absent, ``ds`` is returned unchanged (callers should
        not assume filtering was applied).
    """
    if not enabled:
        return ds
    flag = f"{gate}_rhythmic"
    if flag not in ds.coords:
        return ds
    mask = np.asarray(ds[flag].values, dtype=bool)
    if mask.all():
        return ds
    keep_ids = np.asarray(ds["id"].values)[mask]
    return ds.sel(id=keep_ids)


# ---------------------------------------------------------------------------
# Per-algorithm classifiers
# ---------------------------------------------------------------------------


def classify_lomb_scargle(
    ds: xr.Dataset,
    power_threshold: float = DEFAULT_LS_POWER_THRESHOLD,
    period_window: tuple[float, float] = (16.0, 32.0),
) -> xr.Dataset:
    """
    Classify flies rhythmic vs arrhythmic using Lomb-Scargle POWER (strength).

    Primary metric: ``ls_power`` — the standard-normalized peak power (R^2-style in
    [0, 1]: the fraction of variance the best period explains). This is LS's
    continuous rhythmicity-STRENGTH index, the native equivalent of AC's RI
    (``ac_power``) and CWT's rhythmicity. A fly is rhythmic iff
    ``ls_power > power_threshold`` AND ``period_window[0] <= ls_period <=
    period_window[1]``.

    ``power_threshold`` is a SOFT, user-owned cutoff (like AC's 0.3 RI
    convention) — surfaced and editable, not a hard magic number baked in.

    FAP is NOT a gate. ``ls_fap`` (Baluev significance) remains reported as a
    secondary value, but FAP and power answer different questions: power = how
    strong the rhythm is; FAP = how unlikely this peak is under noise. In a long
    dense record FAP fires on weak-but-consistent periodicity, so gating on
    ``ls_fap < 0.05`` over-called genuinely-arrhythmic flies (the Opa1xLdh doubles
    M20/M22) that sit at very low power (~0.003-0.004 vs rhythmic ~0.013-0.16).
    Power separates the groups by an order of magnitude; FAP does not.

    (FAP-citation fix: the FAP<0.05 convention was mis-attributed to Pfeiffenberger
    et al. 2010, which actually prescribes a chi-squared periodogram; the correct
    LS-FAP references are Horne & Baliunas 1986 / Refinetti 2007.)

    Requires ``ls_power`` and ``ls_period`` (from ``lomb_scargle_analysis``); if
    either is missing the dataset is returned unchanged.

    Writes:
      ds.coords['ls_rhythmic'] : bool, dim ('id',)
      ds.attrs['ls_power_threshold']   : float
      ds.attrs['ls_period_window_min'] : float
      ds.attrs['ls_period_window_max'] : float
    """
    if "ls_power" not in ds or "ls_period" not in ds:
        return ds

    rhythmic = (ds["ls_power"] > power_threshold) & _apply_period_window(
        ds["ls_period"], period_window
    )
    # Fill NaNs as False (cannot classify without a valid power/period)
    rhythmic = rhythmic.fillna(False).astype(bool)

    ds = ds.assign_coords(ls_rhythmic=("id", rhythmic.values))
    ds.attrs["ls_power_threshold"] = float(power_threshold)
    ds.attrs["ls_period_window_min"] = float(period_window[0])
    ds.attrs["ls_period_window_max"] = float(period_window[1])
    return ds


def classify_autocorrelation(
    ds: xr.Dataset,
    ri_threshold: float = DEFAULT_AC_RI_THRESHOLD,
    use_dynamic_ci: bool = False,
    period_window: tuple[float, float] = (16.0, 32.0),
) -> xr.Dataset:
    """
    Classify flies rhythmic vs arrhythmic using autocorrelation RI.

    Primary metric: RI = raw peak autocorrelation value stored in ``ac_power``
    (Levine 2002). A fly is rhythmic iff ``RI > threshold`` AND
    ``period_window[0] <= ac_period <= period_window[1]`` — gating on the
    period window matches LS and CWT classifiers.

    If ``use_dynamic_ci`` is True, the effective threshold per fly is
    ``max(ri_threshold, 2/sqrt(n_samples))`` — Levine's 95% CI. Otherwise the
    fixed default ``ri_threshold`` (0.3) is used for all flies. See the
    ``ac_rhythm_strength`` variable for the RI normalized to 95% CI units.

    Writes:
      ds.coords['ac_rhythmic']           : bool, dim ('id',)
      ds.attrs['ac_ri_threshold']        : float
      ds.attrs['ac_use_dynamic_ci']      : bool
      ds.attrs['ac_period_window_min']   : float
      ds.attrs['ac_period_window_max']   : float
    """
    if "ac_power" not in ds:
        return ds

    ri = ds["ac_power"]

    if use_dynamic_ci and "ac_confidence_interval" in ds:
        # Levine's 95% CI is 2/sqrt(N); the stored `ac_confidence_interval`
        # is 1.965/sqrt(N) (a close variant). Use the stored value directly
        # as the floor, combined with the user threshold via element-wise max.
        ci = ds["ac_confidence_interval"]
        effective = xr.where(ci > ri_threshold, ci, ri_threshold)
        rhythmic_metric = ri > effective
    else:
        rhythmic_metric = ri > ri_threshold

    if "ac_period" in ds:
        rhythmic = rhythmic_metric & _apply_period_window(ds["ac_period"], period_window)
    else:
        rhythmic = rhythmic_metric

    rhythmic = rhythmic.fillna(False).astype(bool)

    ds = ds.assign_coords(ac_rhythmic=("id", rhythmic.values))
    ds.attrs["ac_ri_threshold"] = float(ri_threshold)
    ds.attrs["ac_use_dynamic_ci"] = bool(use_dynamic_ci)
    ds.attrs["ac_period_window_min"] = float(period_window[0])
    ds.attrs["ac_period_window_max"] = float(period_window[1])
    return ds


def classify_cwt(
    ds: xr.Dataset,
    rhythmicity_threshold: float | None = None,
    period_window: tuple[float, float] | None = None,
) -> xr.Dataset:
    """
    Classify flies rhythmic vs arrhythmic using the CWT rhythmicity metric.

    A fly is rhythmic iff ``cwt_rhythmicity > rhythmicity_threshold`` AND
    ``period_window[0] <= cwt_period <= period_window[1]``.

    ONE KNOB — the classify window TRACKS the search range. When
    ``period_window`` is ``None`` (the default), it is taken from the SEARCH
    range that produced this dataset, stamped by ``wavelet_analysis`` on
    ``ds.attrs['cwt_min_period'/'cwt_max_period']`` (falling back to the
    calibration default ``(DEFAULT_CWT_MIN_PERIOD, DEFAULT_CWT_MAX_PERIOD)`` if
    absent). This makes the classify window EQUAL the search range by
    construction, so the two cannot diverge — the bug where a wide search
    (16-50) met a narrow classify (16-32) is structurally impossible here. Pass
    an explicit ``period_window`` only to deliberately classify over a different
    band than the search (e.g. a diagnostic sweep).

    The default threshold depends on which CWT method produced the dataset
    (``ds.attrs['cwt_method']``); when the attr is absent it falls back to
    ``calibrations.DEFAULT_CWT_METHOD``:

    - ``'global_rednoise'`` (default): threshold
      **DEFAULT_CWT_REDNOISE_THRESHOLD** (provisional 2.0) — range-robust,
      length-stable local-prominence strength (peak / AR(1) expected power at
      the peak period).
    - ``'ar1'`` (former default): threshold **1.0** — peak SNR above the 95%
      AR(1) red-noise threshold (Torrence & Compo 1998).
    - ``'ridge'`` (legacy): threshold **0.30** (lab choice; no peer-reviewed
      binary cutoff exists for fraction-of-confident-timepoints).

    Pass an explicit ``rhythmicity_threshold`` to override.

    Writes:
      ds.coords['cwt_rhythmic']                : bool, dim ('id',)
      ds.attrs['cwt_rhythmicity_threshold']    : float
      ds.attrs['cwt_period_window_min']        : float
      ds.attrs['cwt_period_window_max']        : float
    """
    if "cwt_rhythmicity" not in ds or "cwt_period" not in ds:
        return ds

    if rhythmicity_threshold is None:
        # The live path stamps ds.attrs['cwt_method'] on every freshly-computed
        # dataset (periodograms.py); this fallback only fires for a legacy
        # dataset missing the attr, and defers to the single calibration default
        # (DEFAULT_CWT_METHOD) rather than a stale hardcoded method.
        rhythmicity_threshold = cwt_threshold_for(ds.attrs.get("cwt_method", DEFAULT_CWT_METHOD))

    if period_window is None:
        # ONE KNOB: the classify window tracks the SEARCH range that produced
        # this dataset (stamped by wavelet_analysis), so the two cannot diverge.
        # Fall back to the calibration default only for a dataset missing the attrs.
        period_window = (
            float(ds.attrs.get("cwt_min_period", DEFAULT_CWT_MIN_PERIOD)),
            float(ds.attrs.get("cwt_max_period", DEFAULT_CWT_MAX_PERIOD)),
        )

    rhythmic = (ds["cwt_rhythmicity"] > rhythmicity_threshold) & _apply_period_window(
        ds["cwt_period"], period_window
    )
    rhythmic = rhythmic.fillna(False).astype(bool)

    ds = ds.assign_coords(cwt_rhythmic=("id", rhythmic.values))
    ds.attrs["cwt_rhythmicity_threshold"] = float(rhythmicity_threshold)
    ds.attrs["cwt_period_window_min"] = float(period_window[0])
    ds.attrs["cwt_period_window_max"] = float(period_window[1])
    return ds


def classify_all(
    ds: xr.Dataset,
    ls_power_threshold: float = DEFAULT_LS_POWER_THRESHOLD,
    ac_ri_threshold: float = DEFAULT_AC_RI_THRESHOLD,
    cwt_rhythmicity_threshold: float | None = None,
    period_window: tuple[float, float] = (16.0, 32.0),
    cwt_period_window: tuple[float, float] | None = None,
    ac_use_dynamic_ci: bool = False,
    run_ls: bool = True,
    run_ac: bool = True,
    run_cwt: bool = True,
) -> xr.Dataset:
    """
    Run all applicable classifiers and write the bool coords to ``ds``.

    Each classifier is skipped if its required variables are missing or if
    its corresponding ``run_*`` flag is False. The returned dataset is a
    (shallow) copy with the new coords and attrs assigned.

    ``cwt_period_window`` overrides the CWT classify window only. Default
    ``None`` ⇒ ``classify_cwt`` derives the window from the CWT SEARCH range
    stamped on the dataset (``cwt_min_period``/``cwt_max_period``) — i.e. the CWT
    classify window tracks the CWT search range (one knob), NOT the LS/AC
    ``period_window``. CWT's range is independent and may be wider (e.g. widened
    for long-period detection). Pass an explicit tuple only to classify CWT over a
    band different from its search range (e.g. a diagnostic sweep).
    """
    out = ds
    if run_ls and "ls_power" in out:
        out = classify_lomb_scargle(
            out, power_threshold=ls_power_threshold, period_window=period_window
        )
    if run_ac and "ac_power" in out:
        out = classify_autocorrelation(
            out,
            ri_threshold=ac_ri_threshold,
            use_dynamic_ci=ac_use_dynamic_ci,
            period_window=period_window,
        )
    if run_cwt and "cwt_rhythmicity" in out:
        # CWT window tracks its OWN search range (one knob), independent of the
        # LS/AC period_window: pass an explicit cwt_period_window if given, else
        # None so classify_cwt derives the window from the search-range attrs
        # stamped on the dataset (cwt_min_period/cwt_max_period). CWT's range may
        # legitimately differ from LS/AC (e.g. widened for long-period detection).
        out = classify_cwt(
            out, rhythmicity_threshold=cwt_rhythmicity_threshold, period_window=cwt_period_window
        )
    return out


# ---------------------------------------------------------------------------
# Tabular helpers (per-fly DataFrame, summaries)
# ---------------------------------------------------------------------------

_ALGO_META = {
    "ls": {
        "label": "Lomb-Scargle",
        "metric_var": "ls_power",
        "metric_name": "LS power",
        "period_var": "ls_period",
        "flag_coord": "ls_rhythmic",
        "threshold_attr": "ls_power_threshold",
        "lower_is_rhythmic": False,
    },
    "ac": {
        "label": "Autocorrelation",
        "metric_var": "ac_power",
        "metric_name": "RI",
        "period_var": "ac_period",
        "flag_coord": "ac_rhythmic",
        "threshold_attr": "ac_ri_threshold",
        "lower_is_rhythmic": False,
    },
    "cwt": {
        "label": "CWT",
        "metric_var": "cwt_rhythmicity",
        "metric_name": "cwt_rhythmicity",
        "period_var": "cwt_period",
        "flag_coord": "cwt_rhythmic",
        "threshold_attr": "cwt_rhythmicity_threshold",
        "lower_is_rhythmic": False,
    },
}


def available_algorithms(ds: xr.Dataset) -> list[str]:
    """Return the subset of ['ls','ac','cwt'] whose metric variables exist."""
    return [a for a, m in _ALGO_META.items() if m["metric_var"] in ds]


def per_fly_classification_df(ds: xr.Dataset) -> pd.DataFrame:
    """
    Build a wide per-fly DataFrame with one row per fly and columns for each
    algorithm's period, metric, and rhythmic flag. Only algorithms whose
    classification has been run (i.e. the *_rhythmic coord exists) contribute
    columns.
    """
    fly_ids = ds["id"].values
    df = pd.DataFrame({"fly_id": fly_ids})

    if "group" in ds.coords:
        df["group"] = ds["group"].values

    for algo, meta in _ALGO_META.items():
        if meta["metric_var"] in ds:
            df[f"{algo}_period"] = (
                ds[meta["period_var"]].values if meta["period_var"] in ds else np.nan
            )
            df[f"{algo}_metric"] = ds[meta["metric_var"]].values
        if meta["flag_coord"] in ds.coords:
            df[f"{algo}_rhythmic"] = ds[meta["flag_coord"]].values

    # Also surface the AC RS diagnostic if present
    if "ac_rhythm_strength" in ds:
        df["ac_rhythm_strength_legacy"] = ds["ac_rhythm_strength"].values

    return df


def summarize_rhythmicity(ds: xr.Dataset) -> pd.DataFrame:
    """
    Summary table: one row per (algorithm, group), with rhythmic counts and
    percentages. If no 'group' coord exists, a single 'all' group is used.

    Note: this function deliberately does NOT honor ``filter_nonrhythmic``.
    The whole point of the table is to count rhythmic vs arrhythmic flies
    per group; pre-filtering would make every percentage 100% and the
    arrhythmic counts zero (circular).
    """
    if "group" in ds.coords:
        groups = np.asarray(ds["group"].values, dtype=object)
    else:
        groups = np.full(ds.sizes["id"], "all", dtype=object)

    rows = []
    for _algo, meta in _ALGO_META.items():
        flag = meta["flag_coord"]
        if flag not in ds.coords:
            continue
        flags = np.asarray(ds[flag].values, dtype=bool)
        threshold = ds.attrs.get(meta["threshold_attr"], np.nan)
        for grp in sorted(set(groups.tolist())):
            mask = groups == grp
            n = int(mask.sum())
            n_rhythmic = int((flags & mask).sum())
            rows.append(
                {
                    "algorithm": meta["label"],
                    "group": grp,
                    "n_total": n,
                    "n_rhythmic": n_rhythmic,
                    "n_arrhythmic": n - n_rhythmic,
                    "percent_rhythmic": (100.0 * n_rhythmic / n) if n > 0 else 0.0,
                    "threshold": threshold,
                }
            )
        # Also an overall row per algorithm
        total = int(flags.size)
        total_rhyth = int(flags.sum())
        rows.append(
            {
                "algorithm": meta["label"],
                "group": "ALL",
                "n_total": total,
                "n_rhythmic": total_rhyth,
                "n_arrhythmic": total - total_rhyth,
                "percent_rhythmic": (100.0 * total_rhyth / total) if total > 0 else 0.0,
                "threshold": threshold,
            }
        )

    return pd.DataFrame(rows)


def get_rhythmic_flies(ds: xr.Dataset, algorithm: str) -> list[str]:
    """Return fly IDs classified as rhythmic by the named algorithm."""
    algo = algorithm.lower()
    if algo not in _ALGO_META:
        raise ValueError(f"Unknown algorithm: {algorithm!r}")
    flag = _ALGO_META[algo]["flag_coord"]
    if flag not in ds.coords:
        return []
    return ds["id"].values[np.asarray(ds[flag].values, dtype=bool)].tolist()


def compare_thresholds(ds: xr.Dataset, algorithm: str, thresholds: list[float]) -> pd.DataFrame:
    """
    Sensitivity analysis: sweep a single algorithm's threshold and report
    n_rhythmic / percent_rhythmic. Does not mutate ``ds``.
    """
    algo = algorithm.lower()
    if algo not in _ALGO_META:
        raise ValueError(f"Unknown algorithm: {algorithm!r}")

    results = []
    for t in thresholds:
        if algo == "ls":
            # LS now gates on POWER (higher = rhythmic), not FAP — the sweep
            # threshold is the power cutoff.
            tmp = classify_lomb_scargle(ds, power_threshold=t)
        elif algo == "ac":
            tmp = classify_autocorrelation(ds, ri_threshold=t)
        else:
            tmp = classify_cwt(ds, rhythmicity_threshold=t)
        flag = _ALGO_META[algo]["flag_coord"]
        if flag in tmp.coords:
            flags = np.asarray(tmp[flag].values, dtype=bool)
            n = int(flags.size)
            n_r = int(flags.sum())
        else:
            n = n_r = 0
        results.append(
            {
                "algorithm": _ALGO_META[algo]["label"],
                "threshold": t,
                "n_total": n,
                "n_rhythmic": n_r,
                "percent_rhythmic": (100.0 * n_r / n) if n > 0 else 0.0,
            }
        )
    return pd.DataFrame(results)
