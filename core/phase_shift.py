"""
phase_shift.py
==============
Per-fly phase-shift analysis for light-pulse experiments.

A light pulse is delivered at a known circadian time (``pulse_time`` in the metadata,
a ZT hour); the fly's rhythm shifts; this module measures that shift for each fly.

Two methods, one engine
-----------------------
Both published approaches reduce to the same three steps, so they share one
implementation and differ only in how the daily phase marker is found:

1. **Marker per day.** One clock time per calendar day marking the rhythm's phase.
   * ``method="peak"`` (default) — the peak of the Butterworth-low-pass-filtered
     activity trace. This is the lab's ``peakphaseplot.m`` approach (Levine et al.
     2002, BMC Neuroscience 3:1, Fig. 9).
   * ``method="onset"`` — the onset of the daily active phase (the classic
     Aschoff / Daan-Pittendrigh actogram-onset convention).
2. **Regress.** Fit marker-time against day-index separately over the days before
   the pulse and the days after it. Each fit's slope is that epoch's period in
   minutes/day (~1440 when entrained, the endogenous period when free-running).
3. **Compare at the pulse.** Extrapolate the pre-pulse line forward to the pulse
   day and subtract it from the post-pulse line evaluated at the same day. That
   difference is the phase shift: positive = DELAY, negative = ADVANCE.

Because step 2 makes no assumption about the slope, the same code handles a pulse
given during LD entrainment (pre-pulse slope ~1440) and one given during DD
free-run (pre-pulse slope = the endogenous period). No LD/DD fork.

Measured accuracy — why ``peak`` is the default
-----------------------------------------------
Scored by injecting a KNOWN shift into real §2c-cohort flies and asking each method
to recover it (6 flies x 6 shift magnitudes; see ``tests/test_phase_shift.py``):

===========  ============  ============  ==============  ==========
method       median error  90th pct      worst           abstained
===========  ============  ============  ==============  ==========
``peak``     ~1 min        ~8 min        ~60 min         0
``onset``    ~3 min        ~170 min      ~590 min        8 / 36
===========  ============  ============  ==============  ==========

Both are accurate in the typical case; they differ in the TAIL. Onset detection on
1-minute DAM data is intrinsically fragile — the trace is a sparse spike train, so
the "first rise of the active phase" is a much less well-defined feature than the
peak of a smoothed hump, and when the day-to-day chain latches onto the wrong
feature the resulting shift is confidently wrong. Prefer ``peak`` unless there is a
specific reason to want the onset convention, and check an actogram if you use
``onset``. The onset calibrations are PROVISIONAL (see ``calibrations.py``).

Two deliberate departures from ``peakphaseplot.m``
--------------------------------------------------
The MATLAB original is group-averaged and asks a human to click which peaks
correspond across days. Neither survives in an unattended per-fly pipeline, so:

* **Per fly, not per group.** Every fly gets its own markers, fits and shift, so
  the result is a distribution rather than one number per group.
* **Automatic marker matching.** Within an epoch the marker is tracked day to day
  inside a window centred on where the previous day's marker predicts it will be
  (see ``DEFAULT_PHASE_SHIFT_MARKER_SEARCH_HALF_WIDTH_HOURS``). This replaces the
  manual click-to-match step. The post-pulse chain is seeded independently of the
  pre-pulse chain, so the measurable shift is NOT limited by that window.

Main entry point:
    compute_phase_shift_analysis()  — per-fly phase shifts across the cohort

Helpers:
    compute_fly_phase_shift()  — the regression/extrapolation engine
    detect_peak_markers()      — daily markers, peak method
    detect_onset_markers()     — daily markers, onset method
"""

import contextlib
from collections.abc import Mapping

import numpy as np
import pandas as pd
import xarray as xr
from scipy import signal as _signal
from scipy import stats as _stats

from calibrations import (
    DEFAULT_CWT_MAX_PERIOD,
    DEFAULT_CWT_MIN_PERIOD,
    DEFAULT_PHASE_SHIFT_FILTER_HOURS,
    DEFAULT_PHASE_SHIFT_MARKER_SEARCH_HALF_WIDTH_HOURS,
    DEFAULT_PHASE_SHIFT_MIN_POST_DAYS,
    DEFAULT_PHASE_SHIFT_MIN_PRE_DAYS,
    DEFAULT_PHASE_SHIFT_ONSET_SMOOTH_MINUTES,
    DEFAULT_PHASE_SHIFT_ONSET_THRESHOLD_FRAC,
    DEFAULT_PHASE_SHIFT_PEAK_DISTANCE_HOURS,
    DEFAULT_PHASE_SHIFT_PEAK_PROMINENCE_FRAC,
    DEFAULT_PHASE_SHIFT_TRANSIENT_SKIP_DAYS,
)

MINUTES_PER_DAY = 1440

METHODS = ("peak", "onset")


# ---------------------------------------------------------------------------
# Regression engine (shared by both methods)
# ---------------------------------------------------------------------------


def _fit_line(days, markers):
    """Least-squares fit of marker-minute against day-index.

    Returns ``{'slope', 'intercept', 'r2', 'n'}`` where ``slope`` is minutes per
    day — i.e. the epoch's period. Requires at least 2 points (the algebraic floor
    for a line, enforced here regardless of the user's day-count calibrations).
    """
    days = np.asarray(days, dtype=float)
    markers = np.asarray(markers, dtype=float)
    if days.size < 2:
        raise ValueError("_fit_line needs at least 2 points")
    fit = _stats.linregress(days, markers)
    return {
        "slope": float(fit.slope),
        "intercept": float(fit.intercept),
        "r2": float(fit.rvalue**2),
        "n": int(days.size),
    }


def compute_fly_phase_shift(
    day_markers,
    pulse_day_index,
    *,
    reference_day_index=None,
    min_pre_days=DEFAULT_PHASE_SHIFT_MIN_PRE_DAYS,
    min_post_days=DEFAULT_PHASE_SHIFT_MIN_POST_DAYS,
    transient_skip_days=DEFAULT_PHASE_SHIFT_TRANSIENT_SKIP_DAYS,
    min_period_hours=DEFAULT_CWT_MIN_PERIOD,
    max_period_hours=DEFAULT_CWT_MAX_PERIOD,
):
    """Turn one fly's daily markers into a single phase-shift value.

    Parameters
    ----------
    day_markers : dict[int, float]
        ``{day_index: marker_minute}``. Marker minutes are absolute (measured from
        the recording start, NOT wrapped into 0-1440) so that a free-running
        rhythm gives a straight line rather than a sawtooth.
    pulse_day_index : int
        Day containing the light pulse. Excluded from both fits — it is a
        transition day, part entrained and part shifted.
    reference_day_index : int, optional
        Day at which the two lines are compared. Defaults to ``pulse_day_index``
        (the Aschoff/Daan-Pittendrigh convention: read the offset at the pulse).
        If the pre- and post-pulse slopes differ, the reported shift depends on
        this choice, which is why it is exposed.
    min_pre_days, min_post_days : int
        Refuse to report a shift with fewer usable days than this on either side.
    transient_skip_days : int
        Days after the pulse day to exclude from the post-pulse fit, letting
        transient cycles pass before the new steady state is read.

    Returns
    -------
    dict
        ``status == 'ok'`` plus the shift and both fits' diagnostics, or a
        ``status`` of ``'insufficient_pre'`` / ``'insufficient_post'`` with
        ``phase_shift_minutes`` set to NaN. A poorly-constrained fit abstains
        rather than reporting a number that looks equally authoritative.
    """
    pre = sorted(d for d in day_markers if d < pulse_day_index)
    post = sorted(d for d in day_markers if d > pulse_day_index + transient_skip_days)

    out = {
        "status": "ok",
        "phase_shift_minutes": np.nan,
        "phase_shift_hours": np.nan,
        "n_pre_days": len(pre),
        "n_post_days": len(post),
        "pulse_day_index": int(pulse_day_index),
        "reference_day_index": (
            int(pulse_day_index) if reference_day_index is None else int(reference_day_index)
        ),
        "pre_period_hours": np.nan,
        "post_period_hours": np.nan,
        "pre_r2": np.nan,
        "post_r2": np.nan,
    }

    if len(pre) < max(2, int(min_pre_days)):
        out["status"] = "insufficient_pre"
        return out
    if len(post) < max(2, int(min_post_days)):
        out["status"] = "insufficient_post"
        return out

    pre_fit = _fit_line(pre, [day_markers[d] for d in pre])
    post_fit = _fit_line(post, [day_markers[d] for d in post])
    out["pre_period_hours"] = pre_fit["slope"] / 60.0
    out["post_period_hours"] = post_fit["slope"] / 60.0
    out["pre_r2"] = pre_fit["r2"]
    out["post_r2"] = post_fit["r2"]

    # A fitted slope IS the epoch's period, so an implausible slope means the marker
    # chain slipped onto the wrong cycle rather than that the fly has a wild rhythm.
    # Left ungated this surfaces as a confident phase shift that is wrong by about a
    # whole day — the worst kind of error, because nothing about the number looks
    # suspicious. Abstain instead. Bounds reuse the user's existing period window;
    # no second definition of "plausible circadian period".
    lo, hi = float(min_period_hours), float(max_period_hours)
    if not (lo <= out["pre_period_hours"] <= hi and lo <= out["post_period_hours"] <= hi):
        out["status"] = "implausible_period"
        return out

    ref = out["reference_day_index"]
    pre_at_ref = pre_fit["intercept"] + pre_fit["slope"] * ref
    post_at_ref = post_fit["intercept"] + post_fit["slope"] * ref
    raw_shift = post_at_ref - pre_at_ref

    # Wrap into (-period/2, +period/2]. A phase shift is only defined modulo the
    # period: a "+25 h delay" and a "+1 h delay" are the same observation, and the
    # convention is to report the smaller magnitude. This also absorbs the marker
    # chain locking onto the adjacent cycle, which otherwise reports a shift wrong
    # by almost exactly one day while every other diagnostic looks healthy.
    period = post_fit["slope"]
    if np.isfinite(period) and period > 0:
        shift = raw_shift - period * np.round(raw_shift / period)
    else:
        shift = raw_shift
    out["unwrapped_shift_minutes"] = float(raw_shift)

    out.update(
        {
            "phase_shift_minutes": float(shift),
            "phase_shift_hours": float(shift / 60.0),
            "_pre_fit": pre_fit,
            "_post_fit": post_fit,
        }
    )
    return out


# ---------------------------------------------------------------------------
# Daily-marker detection
# ---------------------------------------------------------------------------


def _day_index(minute):
    return int(np.floor(minute / MINUTES_PER_DAY))


def _day_phase(day, split_minute):
    """Label a whole day LD / DD / mixed relative to the LD-DD boundary."""
    if split_minute is None or not np.isfinite(split_minute):
        return "all"
    lo = day * MINUTES_PER_DAY
    hi = lo + MINUTES_PER_DAY - 1
    if hi < split_minute:
        return "LD"
    if lo >= split_minute:
        return "DD"
    return "mixed"


def _epoch_day_indices(minutes, pulse_minute, transient_skip_days, split_minute=None):
    """Split the recording's day indices into (pre-pulse, post-pulse).

    The pulse day itself belongs to neither — it is a transition day, part
    entrained and part shifted. Only days fully inside the record are returned, so
    a partial trailing day cannot contribute a marker found in a truncated window.

    Each side is additionally restricted to a SINGLE LD/DD phase — the phase of the
    days adjacent to the pulse. A fit spanning the LD-DD boundary would be a
    straight line through a kink (entrained days advance ~1440 min/day, free-running
    days advance by the endogenous period), so its slope describes neither epoch and
    the extrapolation to the pulse day would be biased. A day straddling the
    boundary is itself a transition day and is dropped.
    """
    pulse_day = _day_index(pulse_minute)
    last_full_day = int(np.floor((minutes[-1] + 1) / MINUTES_PER_DAY)) - 1
    first_day = _day_index(minutes[0])
    cand_pre = list(range(first_day, pulse_day))
    cand_post = list(range(pulse_day + 1 + int(transient_skip_days), last_full_day + 1))

    def phase_of(day):
        return _day_phase(day, split_minute)

    pre_phase = next((phase_of(d) for d in reversed(cand_pre) if phase_of(d) != "mixed"), None)
    post_phase = next((phase_of(d) for d in cand_post if phase_of(d) != "mixed"), None)
    pre = [d for d in cand_pre if phase_of(d) == pre_phase]
    post = [d for d in cand_post if phase_of(d) == post_phase]
    return pre, post


def _window_slice(minutes, lo, hi):
    """Index range of ``minutes`` falling in [lo, hi]. ``minutes`` is sorted."""
    i0 = int(np.searchsorted(minutes, lo, side="left"))
    i1 = int(np.searchsorted(minutes, hi, side="right"))
    return i0, i1


def _track_days(days, values, minutes, pick_seed, pick_tracked, search_half_width_minutes):
    """Walk a run of days, finding one marker per day.

    The first day is seeded by ``pick_seed`` over the whole day. Each later day is
    resolved by ``pick_tracked`` inside a window centred where the previous
    marker, advanced by whole days, predicts the next one — this is the automatic
    substitute for ``peakphaseplot.m``'s manual peak matching. A day whose window
    holds no usable data contributes no marker (it is absent from the result, never
    a fabricated value); tracking then predicts from the last marker actually found,
    so a single gap day does not break the chain.
    """
    markers = {}
    prev_day = None
    prev_marker = None
    for day in days:
        day_lo = day * MINUTES_PER_DAY
        day_hi = day_lo + MINUTES_PER_DAY - 1
        if prev_marker is None:
            lo, hi = day_lo, day_hi
            picker = pick_seed
        else:
            predicted = prev_marker + MINUTES_PER_DAY * (day - prev_day)
            lo = max(day_lo, predicted - search_half_width_minutes)
            hi = min(day_hi, predicted + search_half_width_minutes)
            picker = pick_tracked
        i0, i1 = _window_slice(minutes, lo, hi)
        if i1 - i0 < 2:
            continue
        chosen = picker(values[i0:i1], minutes[i0:i1])
        if chosen is None:
            continue
        markers[day] = float(chosen)
        prev_day, prev_marker = day, float(chosen)
    return markers


def _peak_pickers(dt_min, prominence_frac, distance_hours):
    """Build the (seed, tracked) peak pickers for :func:`_track_days`."""
    distance = max(1, int(round(distance_hours * 60.0 / dt_min)))

    def candidates(win_vals, win_mins):
        finite = np.isfinite(win_vals)
        if finite.sum() < 4:
            return None, None
        vals = np.where(finite, win_vals, -np.inf)
        span = np.nanmax(win_vals) - np.nanmin(win_vals)
        if not np.isfinite(span) or span <= 0:
            return None, None
        idx, _ = _signal.find_peaks(vals, distance=distance, prominence=prominence_frac * span)
        return idx, vals

    def seed(win_vals, win_mins):
        idx, vals = candidates(win_vals, win_mins)
        if vals is None:
            return None
        if len(idx) == 0:
            # No qualifying peak: fall back to the window maximum rather than
            # dropping the day, mirroring the AC no-peak fallback convention.
            return win_mins[int(np.nanargmax(vals))]
        return win_mins[int(idx[np.argmax(vals[idx])])]

    def tracked(win_vals, win_mins):
        idx, vals = candidates(win_vals, win_mins)
        if vals is None:
            return None
        if len(idx) == 0:
            return win_mins[int(np.nanargmax(vals))]
        # Inside a tracking window every candidate is already near the predicted
        # time, so the strongest one is the right pick.
        return win_mins[int(idx[np.argmax(vals[idx])])]

    return seed, tracked


def _rolling_mean_finite(values, window):
    """NaN-preserving centred-forward rolling mean over ``window`` samples.

    A slot stays NaN unless at least half its window was actually measured, so a
    data gap cannot be smoothed into apparent activity (§2a).
    """
    if window <= 1:
        return values
    finite = np.isfinite(values)
    filled = np.where(finite, values, 0.0)
    kernel = np.ones(window, dtype=float)
    total = np.convolve(filled, kernel, mode="same")
    count = np.convolve(finite.astype(float), kernel, mode="same")
    out = np.full(values.shape, np.nan, dtype=float)
    ok = count >= (window / 2.0)
    out[ok] = total[ok] / count[ok]
    return out


def _onset_pickers(threshold_frac, reference_level):
    """Build the (seed, tracked) onset pickers for :func:`_track_days`.

    Expects an ALREADY-SMOOTHED trace and a reference level computed once for the
    whole fly (see :func:`detect_onset_markers`). Both matter:

    * Smoothing is not optional. At 1-minute resolution real DAM activity is sparse
      and spiky — a quiet-but-alive fly can have ~100 nonzero minutes in a day with
      no run of consecutive nonzero minutes longer than a few — so any "N minutes
      above a level" rule applied to raw counts can never fire. The smoothed trace
      is the activity envelope an experimenter actually reads off an actogram.
    * The level must come from the whole record, not from the search window. Windows
      are 24 h wide on an epoch's seed day but only twice the tracking half-width on
      every later day, so a window-local mean is a different threshold on different
      days — which shows up as a systematic pre-vs-post offset, i.e. a phase shift
      the fly never had.

    The level is a MULTIPLE of that reference rather than an absolute count, because
    count magnitudes are not comparable across flies, genotypes or recorders (a
    FlyBox channel and a DAM tube are not on the same scale), so an absolute cutoff
    tuned on one cohort finds nothing on another.
    """
    level = threshold_frac * reference_level

    def upward_crossings(win_vals):
        above = np.isfinite(win_vals) & (win_vals > level)
        if not above.any():
            return np.array([], dtype=int), above
        prev = np.concatenate(([False], above[:-1]))
        return np.flatnonzero(above & ~prev), above

    def seed(win_vals, win_mins):
        # Anchor to the rest phase: the onset is the first rise after the day's
        # longest quiescent stretch, which is what an experimenter reads off an
        # actogram. Without the anchor the earliest crossing of the day wins, which
        # on an entrained day is the lights-on startle, not the activity onset.
        crossings, above = upward_crossings(win_vals)
        if len(crossings) == 0:
            return None
        quiet = ~above
        best_len, best_end = 0, None
        i, n = 0, len(quiet)
        while i < n:
            if quiet[i]:
                j = i
                while j < n and quiet[j]:
                    j += 1
                if (j - i) > best_len:
                    best_len, best_end = j - i, j
                i = j
            else:
                i += 1
        if best_end is not None:
            after = crossings[crossings >= best_end]
            if len(after):
                return win_mins[int(after[0])]
        return win_mins[int(crossings[0])]

    def tracked(win_vals, win_mins):
        crossings, _ = upward_crossings(win_vals)
        if len(crossings) == 0:
            return None
        return win_mins[int(crossings[0])]

    return seed, tracked


def detect_peak_markers(
    minutes,
    values,
    pulse_minute,
    *,
    dt_min=1.0,
    prominence_frac=DEFAULT_PHASE_SHIFT_PEAK_PROMINENCE_FRAC,
    distance_hours=DEFAULT_PHASE_SHIFT_PEAK_DISTANCE_HOURS,
    search_half_width_hours=DEFAULT_PHASE_SHIFT_MARKER_SEARCH_HALF_WIDTH_HOURS,
    transient_skip_days=DEFAULT_PHASE_SHIFT_TRANSIENT_SKIP_DAYS,
    split_minute=None,
):
    """Daily peak-time markers for one fly. ``values`` must already be low-pass
    filtered (see :func:`compute_phase_shift_analysis`, which filters the whole
    cohort once via ``preprocessing.preprocess_activity``).

    The pre- and post-pulse runs are tracked independently, each seeded by a
    whole-day search, so an arbitrarily large shift across the pulse is still
    measurable.
    """
    pre_days, post_days = _epoch_day_indices(
        minutes, pulse_minute, transient_skip_days, split_minute
    )
    seed, tracked = _peak_pickers(dt_min, prominence_frac, distance_hours)
    half = search_half_width_hours * 60.0
    markers = _track_days(pre_days, values, minutes, seed, tracked, half)
    markers.update(_track_days(post_days, values, minutes, seed, tracked, half))
    return markers


def detect_onset_markers(
    minutes,
    values,
    pulse_minute,
    *,
    dt_min=1.0,
    threshold_frac=DEFAULT_PHASE_SHIFT_ONSET_THRESHOLD_FRAC,
    smooth_minutes=DEFAULT_PHASE_SHIFT_ONSET_SMOOTH_MINUTES,
    search_half_width_hours=DEFAULT_PHASE_SHIFT_MARKER_SEARCH_HALF_WIDTH_HOURS,
    transient_skip_days=DEFAULT_PHASE_SHIFT_TRANSIENT_SKIP_DAYS,
    split_minute=None,
):
    """Daily activity-onset markers for one fly, from the raw activity counts.

    The trace is smoothed and the threshold level fixed ONCE for the whole record
    here, so every day is judged against the same line (see :func:`_onset_pickers`).
    """
    pre_days, post_days = _epoch_day_indices(
        minutes, pulse_minute, transient_skip_days, split_minute
    )
    window = max(1, int(round(smooth_minutes / dt_min)))
    smoothed = _rolling_mean_finite(np.asarray(values, dtype=float), window)
    if not np.isfinite(smoothed).any():
        return {}
    reference_level = float(np.nanmean(smoothed))
    if not np.isfinite(reference_level) or reference_level <= 0:
        return {}
    seed, tracked = _onset_pickers(threshold_frac, reference_level)
    half = search_half_width_hours * 60.0
    markers = _track_days(pre_days, smoothed, minutes, seed, tracked, half)
    markers.update(_track_days(post_days, smoothed, minutes, seed, tracked, half))
    return markers


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def _fly_group(ds, fly_id):
    fly = ds.sel(id=fly_id)
    if "group" in ds.coords:
        return str(fly["group"].item())
    if "genotype" in ds.coords:
        return str(fly["genotype"].item())
    return "All"


def _mask_pulse_window(values, minutes, pulse_minute, duration_minutes):
    """NaN out the pulse itself so the acute startle response cannot be mistaken
    for the rhythm's phase marker. NaN (not 0) because no rhythm-driven behavior
    was observed here — a 0 would read downstream as genuine inactivity."""
    if not np.isfinite(duration_minutes) or duration_minutes <= 0:
        return values
    hit = (minutes >= pulse_minute) & (minutes < pulse_minute + duration_minutes)
    if not hit.any():
        return values
    out = values.copy()
    out[hit] = np.nan
    return out


def compute_phase_shift_analysis(
    ds,
    method="peak",
    *,
    reference_day_index=None,
    min_pre_days=DEFAULT_PHASE_SHIFT_MIN_PRE_DAYS,
    min_post_days=DEFAULT_PHASE_SHIFT_MIN_POST_DAYS,
    transient_skip_days=DEFAULT_PHASE_SHIFT_TRANSIENT_SKIP_DAYS,
    filter_hours=DEFAULT_PHASE_SHIFT_FILTER_HOURS,
    peak_prominence_frac=DEFAULT_PHASE_SHIFT_PEAK_PROMINENCE_FRAC,
    peak_distance_hours=DEFAULT_PHASE_SHIFT_PEAK_DISTANCE_HOURS,
    onset_threshold_frac=DEFAULT_PHASE_SHIFT_ONSET_THRESHOLD_FRAC,
    onset_smooth_minutes=DEFAULT_PHASE_SHIFT_ONSET_SMOOTH_MINUTES,
    search_half_width_hours=DEFAULT_PHASE_SHIFT_MARKER_SEARCH_HALF_WIDTH_HOURS,
    min_period_hours=DEFAULT_CWT_MIN_PERIOD,
    max_period_hours=DEFAULT_CWT_MAX_PERIOD,
    activity_var="activity",
):
    """Measure each fly's phase shift around its light pulse.

    Requires the ``pulse_time`` metadata column — a ZT hour such as ``ZT15`` — plus
    ``first_DD_day``, which anchors that ZT to the last entrained day (see
    ``metadata_template.csv``). Flies whose pulse time is blank — an unpulsed
    control cohort — are reported with status ``'no_pulse'`` rather than dropped,
    so the cohort accounting stays complete.

    Parameters
    ----------
    ds : xr.Dataset
        Whole dataset on a relative-integer-minute time axis, carrying
        ``pulse_time``/``first_DD_day``. Not phase-sliced: the pulse may fall in LD or DD and
        both epochs around it are used, so masking to one phase would discard the
        days the analysis needs.
    method : {'peak', 'onset'}
        Which daily marker to use. ``'peak'`` is the default (the ported lab
        method, and the better-calibrated of the two).

    Returns
    -------
    dict
        ``per_fly`` (DataFrame, one row per fly), ``day_markers``
        (``{fly_id: {day_index: marker_minute}}``, for plotting), ``fits``
        (``{fly_id: {'pre': ..., 'post': ...}}``), ``method``, and ``params``.
    """
    from dam_utilities import add_phase_metadata, add_pulse_metadata
    from preprocessing import PreprocessConfig, preprocess_activity

    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")
    if activity_var not in ds:
        raise KeyError(f"{activity_var!r} not in dataset")
    if "pulse_minute" not in ds.coords:
        ds = add_pulse_metadata(ds)

    time_vals = np.asarray(ds["time"].values)
    if not np.issubdtype(time_vals.dtype, np.integer):
        raise ValueError(
            "compute_phase_shift_analysis requires a relative-integer-minute time axis; "
            "convert with dam_utilities.convert_to_relative_time first."
        )
    minutes = time_vals.astype(float)
    dt_min = float(np.median(np.diff(minutes))) if minutes.size > 1 else 1.0

    # The peak method needs the Butterworth-filtered trace; filter the whole cohort
    # once here rather than per fly. preprocess_activity IS the production filter
    # (the same filtfilt Butterworth as SCAMP butt_filter.m) — no private copy.
    if method == "peak" and filter_hours > 0:
        source = preprocess_activity(
            ds, PreprocessConfig(lopass_hours=float(filter_hours)), activity_var=activity_var
        )[activity_var]
    else:
        source = ds[activity_var]
    source = source.transpose("time", "id")
    values_2d = np.asarray(source.values, dtype=float)

    params = {
        "method": method,
        "min_pre_days": int(min_pre_days),
        "min_post_days": int(min_post_days),
        "transient_skip_days": int(transient_skip_days),
        "search_half_width_hours": float(search_half_width_hours),
        "reference_day_index": reference_day_index,
    }
    if method == "peak":
        params.update(
            filter_hours=float(filter_hours),
            peak_prominence_frac=float(peak_prominence_frac),
            peak_distance_hours=float(peak_distance_hours),
        )
    else:
        params.update(
            onset_threshold_frac=float(onset_threshold_frac),
            onset_smooth_minutes=float(onset_smooth_minutes),
        )

    fly_ids = [str(i) for i in ds["id"].values]
    pulse_minutes = np.asarray(ds["pulse_minute"].values, dtype=float)
    if "pulse_duration_minutes" in ds.coords:
        durations = np.asarray(ds["pulse_duration_minutes"].values, dtype=float)
    else:
        durations = np.full(len(fly_ids), np.nan)
    # The LD/DD boundary keeps each regression inside one phase. split_minute is
    # normally attached during preprocessing, but this page can be reached straight
    # from data loading, so derive it here when only first_DD_day is present — via
    # the shared helper, never a private re-derivation. Without this
    # the boundary logic silently does nothing on a freshly-loaded dataset.
    if "split_minute" not in ds.coords and "first_DD_day" in ds.coords:
        with contextlib.suppress(ValueError):
            ds = add_phase_metadata(ds)
    if "split_minute" in ds.coords:
        split_minutes = np.asarray(ds["split_minute"].values, dtype=float)
    else:
        split_minutes = np.full(len(fly_ids), np.nan)

    rows = []
    all_markers = {}
    all_fits = {}

    for i, fly_id in enumerate(fly_ids):
        pulse_minute = pulse_minutes[i]
        row = {"fly_id": fly_id, "group": _fly_group(ds, ds["id"].values[i]), "method": method}

        if not np.isfinite(pulse_minute):
            rows.append({**row, "status": "no_pulse", "phase_shift_hours": np.nan})
            continue
        if not (minutes[0] <= pulse_minute <= minutes[-1]):
            rows.append({**row, "status": "pulse_outside_record", "phase_shift_hours": np.nan})
            continue

        vals = _mask_pulse_window(values_2d[:, i], minutes, pulse_minute, durations[i])

        if method == "peak":
            markers = detect_peak_markers(
                minutes,
                vals,
                pulse_minute,
                dt_min=dt_min,
                prominence_frac=peak_prominence_frac,
                distance_hours=peak_distance_hours,
                search_half_width_hours=search_half_width_hours,
                transient_skip_days=transient_skip_days,
                split_minute=split_minutes[i],
            )
        else:
            markers = detect_onset_markers(
                minutes,
                vals,
                pulse_minute,
                dt_min=dt_min,
                threshold_frac=onset_threshold_frac,
                smooth_minutes=onset_smooth_minutes,
                search_half_width_hours=search_half_width_hours,
                transient_skip_days=transient_skip_days,
                split_minute=split_minutes[i],
            )

        result = compute_fly_phase_shift(
            markers,
            _day_index(pulse_minute),
            reference_day_index=reference_day_index,
            min_pre_days=min_pre_days,
            min_post_days=min_post_days,
            transient_skip_days=transient_skip_days,
            min_period_hours=min_period_hours,
            max_period_hours=max_period_hours,
        )
        all_markers[fly_id] = markers
        all_fits[fly_id] = {
            "pre": result.pop("_pre_fit", None),
            "post": result.pop("_post_fit", None),
        }
        row.update(result)
        row["pulse_minute"] = float(pulse_minute)
        rows.append(row)

    per_fly = pd.DataFrame(rows)
    return {
        "per_fly": per_fly,
        "day_markers": all_markers,
        "fits": all_fits,
        "method": method,
        "params": params,
    }


# ---------------------------------------------------------------------------
# Control-referenced group comparison (the direct peakphaseplot.m analogue)
# ---------------------------------------------------------------------------


def group_labels(ds, group_by):
    """Per-fly group label built from one or more coords, joined with '_'."""
    cols = [c for c in group_by if c in ds.coords]
    if not cols:
        raise ValueError(
            f"None of {list(group_by)} are coordinates on this dataset; cannot form groups."
        )
    parts = [np.asarray(ds[c].values).astype(str) for c in cols]
    return np.array(["_".join(vals) for vals in zip(*parts)]), cols


def _group_meta(ds, cols, labels):
    """``{group label: {column: value}}`` — the coord values behind each group label.

    Read from the coords rather than by splitting the label, so a genotype or
    condition that itself contains '_' still resolves correctly.
    """
    values = {c: np.asarray(ds[c].values).astype(str) for c in cols}
    meta = {}
    for i, lab in enumerate(labels):
        if lab not in meta:
            meta[lab] = {c: values[c][i] for c in cols}
    return meta


def group_extras(ds, labels, groups, describe_by):
    """``{group: {column: [values present]}}`` for coords not used to define the group.

    Public because the page asks the same question twice: once to preview which
    apparatus each arm of a control pairing sat in, before anything is computed, and
    again on the result (where it arrives as ``group_extras``). One implementation
    means the preview cannot disagree with what the analysis then records — and a
    page reaching for the private name would be backlog item 8 again.

    Kept as a list rather than a single value: a group that spans two flyboxes is
    pooling two sub-experiments, and that has to stay visible rather than collapse to
    whichever value happened to come first.
    """
    out = {grp: {} for grp in groups}
    for col in describe_by:
        if col not in ds.coords:
            continue
        vals = np.asarray(ds[col].values).astype(str)
        for grp in groups:
            out[grp][col] = sorted(set(vals[labels == grp].tolist()))
    return out


def build_matched_control_map(
    ds,
    control_value,
    *,
    group_by=("genotype", "condition"),
    control_on="condition",
    match_on=None,
):
    """Pair every group with the control group that matches it on ``match_on``.

    Passing a single ``control_group`` to :func:`compute_group_phase_difference` refers
    every group to one cohort, so an ``Hr38`` light-pulse arm ends up measured against a
    ``Mito`` unpulsed one and any baseline phase difference between those two genotypes
    is read as a shift. This builds the within-genotype pairing instead: each group's
    reference is the group that shares its genotype (or whatever ``match_on`` names) and
    whose ``control_on`` value is ``control_value``.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset carrying the grouping coords.
    control_value : str
        Value of ``control_on`` marking the unpulsed arm, e.g. ``"noLP"``.
    group_by : sequence of str
        Coordinates whose combination defines a group — the same value passed to
        :func:`compute_group_phase_difference`, so the labels line up.
    control_on : str
        Coordinate separating control from treated groups (default ``"condition"``).
    match_on : sequence of str, optional
        Coordinates the control must share with the group it references. Defaults to
        every column in ``group_by`` except ``control_on`` — genotype-matched for the
        standard ``("genotype", "condition")`` grouping.

    Returns
    -------
    dict
        ``{group label: control label or None}``. A group with no matching control maps
        to ``None`` and :func:`compute_group_phase_difference` reports NaN for it,
        rather than silently borrowing another genotype's control. A control group maps
        to itself (its own difference is 0 by construction).
    """
    labels, cols = group_labels(ds, group_by)
    if control_on not in cols:
        raise ValueError(f"control_on {control_on!r} is not one of the grouping columns {cols}")
    if match_on is None:
        match_on = [c for c in cols if c != control_on]
    match_on = [str(c) for c in match_on]
    missing = [c for c in match_on if c not in cols]
    if missing:
        raise ValueError(
            f"match_on columns {missing} are not among the grouping columns {cols}"
        )

    meta = _group_meta(ds, cols, labels)
    control_value = str(control_value)
    by_key = {}
    for grp, m in meta.items():
        if m[control_on] == control_value:
            by_key.setdefault(tuple(m[c] for c in match_on), []).append(grp)
    ambiguous = {k: v for k, v in by_key.items() if len(v) > 1}
    if ambiguous:
        raise ValueError(
            f"More than one {control_value!r} group shares the same {match_on} key: "
            f"{ambiguous}. The reference would be ambiguous — add the distinguishing "
            "column to match_on (or to group_by)."
        )
    return {
        grp: by_key.get(tuple(m[c] for c in match_on), [None])[0] for grp, m in meta.items()
    }


DAY_ORIGINS = ("recording_start", "dd_onset")
DIFFERENCE_SIGNS = ("group_minus_control", "control_minus_group")


def _resolve_origin_days(ds, labels, groups, day_origin):
    """``{group: absolute day index that becomes that group's day 0}``.

    ``"recording_start"`` puts every group's day 0 at its own recording start, so the
    origin is 0 and nothing moves. ``"dd_onset"`` puts it on the group's first full DD
    day, read from the per-fly ``split_minute`` boundary.
    """
    if day_origin not in DAY_ORIGINS:
        raise ValueError(f"day_origin must be one of {DAY_ORIGINS}, got {day_origin!r}")
    if day_origin == "recording_start":
        return dict.fromkeys(groups, 0)

    import dam_utilities

    if "split_minute" in ds.coords:
        split = np.asarray(ds["split_minute"].values, dtype=float)
    else:
        try:
            split = np.asarray(
                dam_utilities.add_phase_metadata(ds)["split_minute"].values, dtype=float
            )
        except ValueError as exc:
            raise ValueError(
                "day_origin='dd_onset' needs the LD->DD boundary, which comes from the "
                f"metadata's first_DD_day column: {exc}"
            ) from exc

    origins = {}
    for grp in groups:
        member = labels == grp
        days = np.rint(split[member] / MINUTES_PER_DAY)
        distinct = sorted(set(days.tolist()))
        if len(distinct) > 1:
            # A group averaged across two different DD-release days has no single
            # phase to report; say so rather than picking one of them.
            raise ValueError(
                f"Group {grp!r} mixes flies released into DD on different days of their "
                f"recording (day {distinct}). Split them into separate groups, or use "
                "day_origin='recording_start'."
            )
        origins[grp] = int(distinct[0])
    return origins


#: Which days after the pulse the phase response is read from. The pulse is given
#: late on the last entrained day, so ``pulse_day + 1`` is the first full DD day —
#: the transient — and is deliberately skipped. Averaging three days rather than
#: taking one is what keeps a single badly-detected peak from setting the answer.
RESPONSE_DAYS_AFTER_PULSE = (2, 3, 4)

#: Day offsets averaged as each fly's PRE-pulse baseline, subtracted from its
#: response. ``-1``, the day BEFORE the pulse day, is the last clean reading of
#: where this fly sat relative to its control before anything happened.
#:
#: Not ``0``: the marker tracker leaves the pulse day out of both its pre- and
#: post-pulse runs, because that day straddles the pulse. Asking for offset 0 finds
#: no marker at all and silently applies no correction — which is exactly what
#: happened on the first run of this, where the corrected and uncorrected numbers
#: came out byte-identical.
#:
#: Set to ``()`` to leave the offset in deliberately.
BASELINE_DAYS_AFTER_PULSE = (-1,)

#: Coord names that describe the light pulse itself rather than the animal. Groups
#: differ in these BECAUSE of the treatment, so they are the columns a control is
#: matched ACROSS rather than on — see :func:`control_map_from_pulse`.
PULSE_COORDS = ("pulse_zt_hour", "pulse_duration_minutes", "pulse_intensity")


def control_map_from_pulse(ds, group_by, *, duration_coord="pulse_duration_minutes"):
    """Pair every group with its unpulsed control, using the pulse itself.

    A control is a group whose pulse duration is 0: no pulse was given, whatever
    else the metadata says. That is a property of the experiment rather than of a
    naming convention, which is the point — the previous version looked for the
    literal string ``"noLP"`` in a ``condition`` column, so it worked on one lab's
    spreadsheet and silently found nothing on anybody else's.

    Groups are matched on every grouping column EXCEPT the pulse ones
    (:data:`PULSE_COORDS`). Those are what the treatment varies, so matching on them
    would only ever pair a group with itself. Everything else — genotype, sex,
    whatever else defines the group — must agree, which is what makes the comparison
    within-genotype. It also means one control serves every pulse a genotype
    received: the same flies are the right reference for 20 min at ZT21 and for
    60 min at ZT21.

    Returns
    -------
    (dict, list)
        ``{group label: control label or None}``, and the list of control group
        labels. A group with no match maps to ``None`` rather than borrowing
        another genotype's control.
    """
    labels, cols = group_labels(ds, group_by)
    if duration_coord not in ds.coords:
        raise ValueError(
            f"No {duration_coord!r} coordinate, so the unpulsed control cannot be "
            "identified. Add a pulse-duration column to the metadata (0 for the "
            "unpulsed arm) and reload."
        )

    meta = _group_meta(ds, cols, labels)
    durations = np.asarray(ds[duration_coord].values, dtype=float)
    # A group is a control when every fly in it had no pulse. Mixed groups are a
    # metadata error rather than a case to guess at, so they simply are not controls.
    is_control = {}
    for grp in meta:
        vals = durations[labels == grp]
        vals = vals[np.isfinite(vals)]
        is_control[grp] = bool(vals.size) and bool(np.all(vals == 0))

    match_on = [c for c in cols if c not in PULSE_COORDS]
    controls = [g for g, c in is_control.items() if c]
    by_key = {}
    for grp in controls:
        by_key.setdefault(tuple(meta[grp][c] for c in match_on), []).append(grp)

    ambiguous = {k: v for k, v in by_key.items() if len(v) > 1}
    if ambiguous:
        raise ValueError(
            f"More than one unpulsed group shares the same {match_on or ['(no columns)']} "
            f"key: {ambiguous}. Add the column that tells them apart to the grouping."
        )

    out = {}
    for grp in meta:
        if is_control[grp]:
            out[grp] = grp
            continue
        key = tuple(meta[grp][c] for c in match_on)
        out[grp] = by_key.get(key, [None])[0]
    return out, sorted(controls)


def grouping_suggestion(ds):
    """What to group by, for the message shown when no control could be matched.

    Genotype nearly always defines a group; the pulse columns are what make a
    treated arm treated; and a factor like sex belongs in the grouping when it is
    present, because pooling sexes and then matching one control to both is a
    different experiment from the one that was run.
    """
    import dam_utilities

    available = set(dam_utilities.group_defining_coords(ds))
    wanted = ["genotype", *PULSE_COORDS, "sex", "block"]
    return [c for c in wanted if c in available]


def _per_fly_day_phase_hours(
    ds,
    *,
    fallback_pulse_minute=None,
    progress_callback=None,
    activity_var="activity",
    filter_hours=DEFAULT_PHASE_SHIFT_FILTER_HOURS,
    peak_prominence_frac=DEFAULT_PHASE_SHIFT_PEAK_PROMINENCE_FRAC,
    peak_distance_hours=DEFAULT_PHASE_SHIFT_PEAK_DISTANCE_HOURS,
    search_half_width_hours=DEFAULT_PHASE_SHIFT_MARKER_SEARCH_HALF_WIDTH_HOURS,
):
    """``{fly id: {absolute day index: peak time in hours past that day's start}}``.

    The same detection the per-fly page uses — one Butterworth pass over the whole
    cohort, then :func:`detect_peak_markers` per fly — so a number here and a number
    there come from one algorithm rather than two that drifted.

    Hours **within the day**, not from the recording start, because that is what a
    phase is: two flies whose peaks are 24 h apart in absolute time are at the same
    phase.

    ``fallback_pulse_minute`` is used for flies that had no pulse. The marker tracker
    splits a record into pre- and post-pulse runs and seeds each independently, so it
    needs a split point even for a control fly — and the right one is the cohort's,
    since a control exists precisely to say what the pulsed flies would have done on
    those same days. Without it the controls raise, which is how this was found.
    """
    from dam_utilities import add_phase_metadata, add_pulse_metadata
    from preprocessing import PreprocessConfig, preprocess_activity

    if "pulse_minute" not in ds.coords:
        ds = add_pulse_metadata(ds)
    if "split_minute" not in ds.coords and "first_DD_day" in ds.coords:
        with contextlib.suppress(ValueError):
            ds = add_phase_metadata(ds)

    source = (
        preprocess_activity(
            ds, PreprocessConfig(lopass_hours=float(filter_hours)), activity_var=activity_var
        )[activity_var]
        if filter_hours > 0
        else ds[activity_var]
    )
    values_2d = np.asarray(source.transpose("time", "id").values, dtype=float)
    minutes = np.asarray(ds["time"].values, dtype=float)

    fly_ids = [str(i) for i in ds["id"].values]
    pulse_minutes = np.asarray(ds["pulse_minute"].values, dtype=float)
    split_minutes = (
        np.asarray(ds["split_minute"].values, dtype=float)
        if "split_minute" in ds.coords
        else np.full(len(fly_ids), np.nan)
    )

    out = {}
    n = len(fly_ids)
    for i, fly_id in enumerate(fly_ids):
        # Per-fly detection is the slow half of a phase response, so the caller
        # gets to say how far along it is. Reported per fly rather than per
        # chunk: a cohort of 200 is the common case and a bar that moves once
        # is not a bar.
        if progress_callback is not None:
            progress_callback(i / n if n else 1.0)
        own_pulse = pulse_minutes[i]
        if not np.isfinite(own_pulse):
            own_pulse = fallback_pulse_minute
        if own_pulse is None or not np.isfinite(own_pulse):
            out[fly_id] = {}
            continue
        markers = detect_peak_markers(
            minutes,
            values_2d[:, i],
            own_pulse,
            prominence_frac=peak_prominence_frac,
            distance_hours=peak_distance_hours,
            search_half_width_hours=search_half_width_hours,
            split_minute=split_minutes[i] if np.isfinite(split_minutes[i]) else None,
        )
        out[fly_id] = {
            int(day): (float(m) - int(day) * MINUTES_PER_DAY) / 60.0
            for day, m in markers.items()
            if np.isfinite(m)
        }
    if progress_callback is not None:
        progress_callback(1.0)
    return out


def compute_phase_response(
    ds,
    *,
    group_by=("genotype", "pulse_zt_hour", "pulse_duration_minutes"),
    control_map=None,
    response_days=RESPONSE_DAYS_AFTER_PULSE,
    baseline_days=BASELINE_DAYS_AFTER_PULSE,
    duration_coord="pulse_duration_minutes",
    zt_coord="pulse_zt_hour",
    progress_callback=None,
    activity_var="activity",
    filter_hours=DEFAULT_PHASE_SHIFT_FILTER_HOURS,
    peak_prominence_frac=DEFAULT_PHASE_SHIFT_PEAK_PROMINENCE_FRAC,
    peak_distance_hours=DEFAULT_PHASE_SHIFT_PEAK_DISTANCE_HOURS,
    search_half_width_hours=DEFAULT_PHASE_SHIFT_MARKER_SEARCH_HALF_WIDTH_HOURS,
):
    """One phase response per FLY, for a phase response curve.

    The quantity, stated once so the graphs cannot disagree about it::

        response(fly) = mean over the response days of
                        (that day's control-group mean peak time - this fly's peak time)

    **Sign.** Control minus fly, so an ADVANCE — the fly peaking earlier than its
    control — is POSITIVE, and a delay is negative. That is the convention a PRC is
    normally drawn in.

    **Per fly, against a group mean.** Every fly in a genotype has the same control
    mean subtracted, which has two consequences worth knowing. The mean of a group's
    per-fly responses is exactly (group mean - control mean), so a violin's centre is
    the group-level phase response and its spread is the real between-fly variation —
    one computation gives both. And because the subtracted constant is shared, it
    cancels when two groups are compared, so comparing genotypes to each other, or to
    the control's own distribution, is sound. What is NOT sound is testing one group
    against zero and reading it as a test against the control: that ignores the
    uncertainty in the control mean itself.

    **Days.** ``response_days`` are offsets from the fly's pulse day. The default
    skips ``+1``, the first full DD day, because it carries the transient.

    **Baseline.** ``baseline_days`` are subtracted from the response, per fly, so
    what is reported is the change the pulse produced rather than the change plus a
    constant. Two cohorts are rarely at exactly the same phase beforehand — the
    pulsed arm and its control usually sit in different boxes — and that offset is
    present on the response days as much as before them. The default is the pulse
    day itself, the last clean reading. Pass ``()`` to leave it in; the controls
    still centre on zero either way, since each is measured against its own mean.

    A fly with no usable peak on any response day gets NaN and is counted in
    ``n_dropped`` rather than silently thinning its group — per-fly peak detection
    is markedly noisier than the group-mean detection ``peakphaseplot.m`` was built
    around, and an arrhythmic fly has no peak to find at all.

    **Pairing.** ``control_map`` is derived from the pulse itself
    (:func:`control_map_from_pulse`) unless one is passed in, which is how a caller
    that resolved the pairing some other way — a dataset with no pulse-duration
    column, where the unpulsed arm is named rather than computed — gets the same
    analysis.

    Returns
    -------
    dict
        ``per_fly`` (DataFrame: ``id``, ``group``, ``control_group``, ``zt``,
        ``response_hours``, ``n_days_used``, plus one column per grouping coord),
        ``control_map``, ``controls``, ``dropped`` (DataFrame: ``group``,
        ``n_dropped``, ``n_total``), and ``params``.
    """
    from dam_utilities import add_pulse_metadata

    if zt_coord not in ds.coords:
        raise ValueError(f"No {zt_coord!r} coordinate; a PRC needs the pulse time.")
    if "pulse_minute" not in ds.coords:
        ds = add_pulse_metadata(ds)

    labels, cols = group_labels(ds, group_by)
    meta = _group_meta(ds, cols, labels)
    if control_map is None:
        control_map, controls = control_map_from_pulse(
            ds, group_by, duration_coord=duration_coord
        )
    else:
        # A caller that already knows the pairing passes it in, and this does not
        # go looking for a duration column it may not have. That is the case for a
        # dataset recorded before anyone wrote pulse durations down: its unpulsed
        # arm is a string in a `condition` column, which no rule about zero can
        # find, but the page has already resolved it and there is no reason the
        # phase response should be the one result such a dataset cannot have.
        control_map = dict(control_map)
        controls = sorted({c for c in control_map.values() if c is not None})

    fly_ids = [str(i) for i in ds["id"].values]
    pulse_minutes = np.asarray(ds["pulse_minute"].values, dtype=float)
    zt_vals = np.asarray(ds[zt_coord].values, dtype=float)

    # The pulse day, resolved before detection because the control flies need it too.
    pulsed = pulse_minutes[np.isfinite(pulse_minutes)]
    if not pulsed.size:
        raise ValueError(
            "No fly in this dataset has a light pulse, so there is no phase response "
            "to measure. Check that pulse_time is set for the treated arm."
        )
    median_pulse = float(np.median(pulsed))
    pulse_day = int(np.floor(median_pulse / MINUTES_PER_DAY))
    days = [pulse_day + int(d) for d in response_days]
    base_days = [pulse_day + int(d) for d in (baseline_days or ())]

    phases = _per_fly_day_phase_hours(
        ds,
        fallback_pulse_minute=median_pulse,
        progress_callback=progress_callback,
        activity_var=activity_var,
        filter_hours=filter_hours,
        peak_prominence_frac=peak_prominence_frac,
        peak_distance_hours=peak_distance_hours,
        search_half_width_hours=search_half_width_hours,
    )

    # The control mean per day, over the control flies that HAVE a peak that day.
    control_day_mean = {}
    for ctrl in controls:
        members = [f for f, lab in zip(fly_ids, labels) if lab == ctrl]
        for day in [*days, *base_days]:
            vals = [phases[f][day] for f in members if day in phases.get(f, {})]
            if vals:
                control_day_mean[(ctrl, day)] = float(np.mean(vals))

    rows, dropped = [], {}
    for fly_id, lab, zt in zip(fly_ids, labels, zt_vals):
        ctrl = control_map.get(lab)
        dropped.setdefault(lab, [0, 0])
        dropped[lab][1] += 1
        if ctrl is None:
            dropped[lab][0] += 1
            continue
        # ctrl/fly_id bound as defaults: a closure over the loop variables would
        # read whichever fly the loop had reached by the time it was called.
        def _diff(day_list, *, ctrl=ctrl, fly_id=fly_id):
            out = []
            for day in day_list:
                ref = control_day_mean.get((ctrl, day))
                own = phases.get(fly_id, {}).get(day)
                if ref is not None and own is not None:
                    # Control minus fly: an advance reads positive.
                    out.append(ref - own)
            return out

        per_day = _diff(days)
        if not per_day:
            dropped[lab][0] += 1
            continue
        base = _diff(base_days) if base_days else []
        # A fly with no usable baseline keeps its raw response rather than being
        # dropped: the offset is a correction, and losing the measurement to fix a
        # correction is the worse trade. base_used records which it was.
        offset = float(np.mean(base)) if base else 0.0
        row = {
            "id": fly_id,
            "group": lab,
            "control_group": ctrl,
            "zt": float(zt) if np.isfinite(zt) else np.nan,
            "response_hours": float(np.mean(per_day)) - offset,
            "response_hours_raw": float(np.mean(per_day)),
            "n_days_used": len(per_day),
            "baseline_used": bool(base),
        }
        row.update(meta.get(lab, {}))
        rows.append(row)

    per_fly = pd.DataFrame(rows)
    drop_df = pd.DataFrame(
        [
            {"group": g, "n_dropped": d, "n_total": t}
            for g, (d, t) in sorted(dropped.items())
        ]
    )
    return {
        "per_fly": per_fly,
        "control_map": control_map,
        "controls": controls,
        "dropped": drop_df,
        "params": {
            "group_by": list(cols),
            "response_days": [int(d) for d in response_days],
            "baseline_days": [int(d) for d in (baseline_days or ())],
            "pulse_day": pulse_day,
            "filter_hours": float(filter_hours),
            "peak_prominence_frac": float(peak_prominence_frac),
            "peak_distance_hours": float(peak_distance_hours),
            "search_half_width_hours": float(search_half_width_hours),
            "sign": "control_minus_fly (advance positive, delay negative)",
        },
    }


def summarize_phase_response(per_fly, by=("zt", "group")):
    """Group mean, SD, SEM and n of the per-fly responses — the PRC line's points.

    The mean here is the group-level phase response by construction (see
    :func:`compute_phase_response`), so the line through these points and the
    violins behind it are the same quantity rather than two estimators that happen
    to sit near each other.
    """
    if per_fly is None or per_fly.empty:
        return pd.DataFrame(columns=[*by, "mean", "sd", "sem", "n"])
    g = per_fly.groupby(list(by), dropna=False)["response_hours"]
    out = g.agg(mean="mean", sd=lambda x: x.std(ddof=1), n="count").reset_index()
    out["sem"] = out["sd"] / np.sqrt(out["n"].clip(lower=1))
    out.loc[out["n"] <= 1, ["sd", "sem"]] = np.nan
    return out


def compute_group_phase_difference(
    ds,
    control_group,
    *,
    group_by=("genotype", "condition"),
    describe_by=(),
    day_origin="recording_start",
    baseline_day=None,
    difference_sign="group_minus_control",
    filter_hours=DEFAULT_PHASE_SHIFT_FILTER_HOURS,
    peak_prominence_frac=DEFAULT_PHASE_SHIFT_PEAK_PROMINENCE_FRAC,
    peak_distance_hours=DEFAULT_PHASE_SHIFT_PEAK_DISTANCE_HOURS,
    search_half_width_hours=DEFAULT_PHASE_SHIFT_MARKER_SEARCH_HALF_WIDTH_HOURS,
    activity_var="activity",
):
    """Per-day phase difference between each group and its unpulsed control group.

    This is the direct analogue of the lab's ``peakphaseplot.m``: average each
    group's activity, smooth it, take one peak per day, and report the pulsed
    group's peak time minus the control group's on the same day.

    The reference is either one cohort for the whole experiment (pass a group label)
    or a per-group pairing (pass a mapping, usually from
    :func:`build_matched_control_map`). Prefer the pairing when the experiment holds
    more than one genotype: genotypes differ in baseline phase, so measuring an
    ``Hr38`` pulsed arm against a ``Mito`` unpulsed one reports that genotype
    difference as part of the shift.

    Aligning days across experiments
    --------------------------------
    By default a day is counted from each fly's own ``start_datetime``, which is the
    right anchor when every group comes from one run. It is the WRONG anchor when the
    control comes from a different experiment that was released into DD on a different
    day of its recording: at the same day index the two cohorts would then have been
    free-running for different numbers of days, and the difference picks up roughly
    ``24 - tau`` of drift for every day of offset. Pass ``day_origin="dd_onset"`` to
    count days from each group's own LD->DD boundary instead, so day 0 is the first
    full DD day in both experiments. Because the pulse is by definition given on the
    last entrained day (see ``dam_utilities._derive_pulse_minute``), under this origin
    the pulse always falls on day -1 — a further reason it lines two runs up correctly.

    Taking out the starting offset
    ------------------------------
    Two cohorts are rarely at exactly the same phase before the pulse — different
    flyboxes sit at slightly different phases, and that offset is present on day 1 as
    much as on the last day. ``baseline_day`` subtracts each group's difference on that
    day from all of its days, so every group starts at zero and what the plot shows is
    the CHANGE produced by the pulse rather than the change plus a constant. It is
    reported in ``phase_difference_from_baseline_hours``; the raw difference is always
    kept alongside it.

    ``difference_sign`` picks which way round the subtraction reads. The default
    ``"group_minus_control"`` gives the pulsed group's peak minus its control's, so
    positive = later = delayed. ``"control_minus_group"`` is the lab's plotting
    convention (``noLP - LP``), where a delay reads negative.

    Use this rather than :func:`compute_phase_shift_analysis` when the protocol does
    not leave enough pre-pulse days to establish each fly's own baseline — which is
    the usual case when the pulse is given on the last entrained day. The unpulsed
    control cohort supplies the "what the phase would have been" reference that the
    within-fly version gets by extrapolating a pre-pulse fit.

    Note this is a GROUP-level measure by construction: the peak is taken from the
    group-mean trace (as ``peakphaseplot.m`` does — average first, then filter), so
    there is one number per group per day and no per-fly distribution. It is also a
    difference between two cohorts, not a within-animal change, so a nonzero value on
    day 0 (before any pulse could act) is a baseline offset between the cohorts, not
    a shift.

    Concordance with SCAMP, and the first-day caveat
    ------------------------------------------------
    Validated against the lab's own ``peakphaseplot.m`` exports for this protocol
    (3 genotypes x 2 pulse intensities x 6 days; see ``tests/test_phase_shift.py``):
    from day 1 onward the two agree to a **median of 2 min, worst case 12 min**.

    **Day 0 is the exception and should not be trusted** — there the two disagree by
    1.1-1.7 h. The cause is the zero-phase Butterworth filter: ``filtfilt`` pads the
    start of the record, so the first day's smoothed peak sits on synthetic padding
    rather than on measured behavior, and its position is not well determined. This
    is inherent to the smoothing, not a difference in bookkeeping. Day 0 is normally
    a pre-pulse baseline day anyway, so it carries no shift information; the
    ``filter_edge`` column flags it.

    Note also that SCAMP reports peak times from MIDNIGHT of the start date whereas
    ``peak_hours`` here is measured from ``start_datetime`` (ZT0). Absolute peak
    times therefore differ by that offset (9 h for a 09:00 lights-on); the phase
    DIFFERENCES this function returns are unaffected.

    Parameters
    ----------
    ds : xr.Dataset
        Whole dataset on a relative-integer-minute time axis.
    control_group : str or mapping
        Either the label of one unpulsed reference group (as built from ``group_by``),
        used for every group; or a ``{group: control group}`` mapping so each group is
        referenced to its own control — see :func:`build_matched_control_map` for the
        within-genotype pairing. A group mapped to ``None`` gets NaN differences.
    group_by : sequence of str
        Coordinates whose combination defines a group.
    describe_by : sequence of str
        Coordinates to record per group without grouping on them — ``("flybox",)``
        labels each comparison with the apparatus it came from. A group holding more
        than one value gets all of them, which is how a condition label covering two
        sub-experiments shows itself.
    day_origin : {"recording_start", "dd_onset"}
        Where day 0 sits. ``"recording_start"`` (default) counts from each fly's
        ``start_datetime``, the historical behaviour. ``"dd_onset"`` counts from the
        group's LD->DD boundary, which is what makes a control from another
        experiment comparable; it needs ``first_DD_day`` in the metadata.
    baseline_day : int, optional
        Day index whose difference is subtracted from every day of the same group, so
        each group starts at zero there and the starting offset between the two cohorts
        drops out. ``None`` (default) leaves the differences as measured.
    difference_sign : {"group_minus_control", "control_minus_group"}
        Direction of the subtraction. Default is group minus control (positive =
        delayed); ``"control_minus_group"`` is the ``noLP - LP`` convention.

    Returns
    -------
    dict
        ``per_day`` (DataFrame: group, control_group, day_index, peak_hours,
        control_peak_hours, phase_difference_hours,
        phase_difference_from_baseline_hours, plus ``absolute_day_index`` and
        ``peak_hours_from_start`` holding the unaligned values), ``peak_times``
        ({group: {absolute day: hours from start}}), ``control_group`` (the argument
        as given), ``control_map`` ({group: control group}, always resolved),
        ``origin_day`` ({group: absolute day index of that group's day 0}),
        ``group_values`` ({group: {column: value}}, for faceting without splitting
        labels), and ``params``.
    """
    from preprocessing import PreprocessConfig, preprocess_activity

    if activity_var not in ds:
        raise KeyError(f"{activity_var!r} not in dataset")
    time_vals = np.asarray(ds["time"].values)
    if not np.issubdtype(time_vals.dtype, np.integer):
        raise ValueError(
            "compute_group_phase_difference requires a relative-integer-minute time axis; "
            "convert with dam_utilities.convert_to_relative_time first."
        )

    labels, cols = group_labels(ds, group_by)
    groups = sorted(set(labels))
    if isinstance(control_group, Mapping):
        control_map = {grp: control_group.get(grp) for grp in groups}
        unknown = sorted({c for c in control_map.values() if c is not None and c not in groups})
        if unknown:
            raise ValueError(
                f"control group(s) {unknown} are not among the groups present: {groups}"
            )
        if all(c is None for c in control_map.values()):
            raise ValueError(
                "No group has a control group to be referenced to. Check that the control "
                "arm is present and that its label matches the grouping columns."
            )
    else:
        if control_group not in groups:
            raise ValueError(
                f"control_group {control_group!r} is not one of the groups present: {groups}"
            )
        control_map = dict.fromkeys(groups, control_group)

    origin_day = _resolve_origin_days(ds, labels, groups, day_origin)

    minutes = time_vals.astype(float)
    dt_min = float(np.median(np.diff(minutes))) if minutes.size > 1 else 1.0
    activity = np.asarray(ds[activity_var].transpose("time", "id").values, dtype=float)

    # peakphaseplot.m averages the group FIRST and filters the mean, so match that
    # order — filtering per fly and then averaging is a different operation.
    peak_times = {}
    for grp in groups:
        member = labels == grp
        block = activity[:, member]
        finite = np.isfinite(block)
        counts = finite.sum(axis=1)
        # Mean over the flies actually measured at each minute. A minute measured in
        # no fly of the group stays NaN, never 0 (§2a).
        mean_trace = np.full(block.shape[0], np.nan, dtype=float)
        measured = counts > 0
        mean_trace[measured] = (
            np.where(finite[measured], block[measured], 0.0).sum(axis=1) / counts[measured]
        )
        one = xr.Dataset(
            {activity_var: (["time", "id"], mean_trace[:, None].astype("float32"))},
            coords={"time": ("time", time_vals), "id": ("id", [grp])},
            attrs=dict(ds.attrs),
        )
        if filter_hours > 0:
            smoothed = np.asarray(
                preprocess_activity(one, PreprocessConfig(lopass_hours=float(filter_hours)))[
                    activity_var
                ]
                .transpose("time", "id")
                .values[:, 0],
                dtype=float,
            )
        else:
            smoothed = mean_trace

        n_days = max(1, int(np.floor((minutes[-1] + 1) / MINUTES_PER_DAY)))
        seed, tracked = _peak_pickers(dt_min, peak_prominence_frac, peak_distance_hours)
        markers = _track_days(
            list(range(n_days)),
            smoothed,
            minutes,
            seed,
            tracked,
            search_half_width_hours * 60.0,
        )
        peak_times[grp] = {d: m / 60.0 for d, m in markers.items()}

    # Shift each group onto its own origin. Both the day index and the peak time move,
    # because the peak is measured from the recording start: leaving the hours on the
    # old origin while moving the day would put a whole 24 h into every difference
    # between two groups whose origins differ. With day_origin="recording_start" every
    # origin is 0 and these are identities.
    aligned = {
        grp: {
            d - origin_day[grp]: h - origin_day[grp] * 24.0
            for d, h in times.items()
        }
        for grp, times in peak_times.items()
    }

    rows = []
    for grp in groups:
        ctrl_grp = control_map.get(grp)
        control = aligned[ctrl_grp] if ctrl_grp is not None else {}
        last_day = max(peak_times[grp]) if peak_times[grp] else None
        for day, hours in sorted(peak_times[grp].items()):
            rel_day = day - origin_day[grp]
            rel_hours = hours - origin_day[grp] * 24.0
            ctrl = control.get(rel_day)
            rows.append(
                {
                    "group": grp,
                    "control_group": ctrl_grp,
                    "day_index": rel_day,
                    "peak_hours": rel_hours,
                    "control_peak_hours": ctrl,
                    "phase_difference_hours": (rel_hours - ctrl if ctrl is not None else np.nan),
                    "n_flies": int((labels == grp).sum()),
                    # Kept alongside the aligned values so a row can always be traced
                    # back to a wall-clock position in its own recording.
                    "absolute_day_index": day,
                    "peak_hours_from_start": hours,
                    # filtfilt pads the record ends, so the first/last day's peak
                    # sits partly on synthetic padding — flag, don't silently report.
                    "filter_edge": day == 0 or day == last_day,
                }
            )

    per_day = pd.DataFrame(rows)

    if difference_sign not in DIFFERENCE_SIGNS:
        raise ValueError(
            f"difference_sign must be one of {DIFFERENCE_SIGNS}, got {difference_sign!r}"
        )
    if difference_sign == "control_minus_group":
        per_day["phase_difference_hours"] = -per_day["phase_difference_hours"]

    # Rebasing is a per-group shift, so it must happen after the sign is settled.
    per_day["phase_difference_from_baseline_hours"] = np.nan
    if baseline_day is not None and not per_day.empty:
        at_baseline = per_day[per_day["day_index"] == int(baseline_day)]
        base = at_baseline.set_index("group")["phase_difference_hours"]
        per_day["phase_difference_from_baseline_hours"] = per_day[
            "phase_difference_hours"
        ] - per_day["group"].map(base)

    return {
        "per_day": per_day,
        "peak_times": peak_times,
        "control_group": control_group,
        "control_map": control_map,
        "origin_day": origin_day,
        "group_values": _group_meta(ds, cols, labels),
        "group_extras": group_extras(ds, labels, groups, describe_by),
        "params": {
            "group_by": list(cols),
            "describe_by": [c for c in describe_by if c in ds.coords],
            "day_origin": str(day_origin),
            "baseline_day": (None if baseline_day is None else int(baseline_day)),
            "difference_sign": str(difference_sign),
            "filter_hours": float(filter_hours),
            "peak_prominence_frac": float(peak_prominence_frac),
            "peak_distance_hours": float(peak_distance_hours),
            "search_half_width_hours": float(search_half_width_hours),
        },
    }


def bootstrap_group_phase_difference(
    ds,
    control_group,
    *,
    n_boot=200,
    seed=0,
    ci=95.0,
    progress_callback=None,
    **kwargs,
):
    """Uncertainty for :func:`compute_group_phase_difference`, by resampling flies.

    There is no per-fly spread to average here: following ``peakphaseplot.m``, the
    plotted point is the peak of the group's MEAN trace, not the mean of per-fly
    peaks. Those two are different estimators, so a SEM built from per-fly peaks
    would not describe the line that is drawn.

    What this does instead is resample the flies — with replacement, independently
    within every group, so the pulsed arm and its control both vary — and re-run the
    whole pipeline (average, smooth, find one peak per day, subtract the control,
    rebase) on each resampled cohort. The spread of the resulting differences is the
    sampling uncertainty of exactly the quantity plotted, and it correctly carries
    the control's uncertainty as well as the pulsed group's.

    Every keyword of :func:`compute_group_phase_difference` is accepted and passed
    through unchanged, so the bootstrap describes the same analysis as the point
    estimate.

    Parameters
    ----------
    n_boot : int
        Resampled cohorts to draw. 200 is enough for error bars; a percentile
        interval steadies at a few hundred.
    seed : int
        Makes the interval reproducible — the same data and seed give the same bars.
    ci : float
        Central interval width in percent (95 gives the 2.5th-97.5th percentiles).

    Returns
    -------
    DataFrame
        One row per group and day: ``group``, ``day_index``, and for both the raw and
        the rebased difference a ``_boot_sd`` (the bootstrap standard error), ``_lo``
        and ``_hi`` (percentile interval bounds) column, plus ``n_boot_ok`` — how many
        resamples yielded a usable peak for that group-day. A group-day whose peak
        detection fails in most resamples shows a small ``n_boot_ok``, and its bar
        should not be trusted.
    """
    if int(n_boot) < 2:
        raise ValueError("n_boot must be at least 2")

    labels, _cols = group_labels(ds, kwargs.get("group_by", ("genotype", "condition")))
    groups = sorted(set(labels))
    member_idx = {g: np.flatnonzero(labels == g) for g in groups}
    if not any(len(v) for v in member_idx.values()):
        raise ValueError("no flies to resample")

    rng = np.random.default_rng(seed)
    value_cols = ["phase_difference_hours", "phase_difference_from_baseline_hours"]
    draws = {c: {} for c in value_cols}     # {col: {(group, day): [values]}}

    for b in range(int(n_boot)):
        # Resample within each group, so group sizes are preserved and no group can
        # vanish from a resample.
        picks = np.concatenate([
            idx[rng.integers(0, len(idx), len(idx))] if len(idx) else idx
            for idx in member_idx.values()
        ])
        try:
            res = compute_group_phase_difference(
                ds.isel(id=picks), control_group, **kwargs
            )
        except Exception:
            # A degenerate resample (e.g. every fly identical) can fail peak finding;
            # skip it rather than losing the whole run.
            continue
        pd_b = res["per_day"]
        for col in value_cols:
            if col not in pd_b.columns:
                continue
            for grp, day, val in zip(pd_b["group"], pd_b["day_index"], pd_b[col]):
                if np.isfinite(val):
                    draws[col].setdefault((grp, int(day)), []).append(float(val))
        if progress_callback is not None:
            progress_callback((b + 1) / float(n_boot))

    lo_q = (100.0 - float(ci)) / 2.0
    hi_q = 100.0 - lo_q
    keys = sorted({k for col in value_cols for k in draws[col]})
    rows = []
    for grp, day in keys:
        row = {"group": grp, "day_index": day}
        for col in value_cols:
            vals = np.asarray(draws[col].get((grp, day), ()), dtype=float)
            stem = col.replace("_hours", "")
            if vals.size >= 2:
                row[f"{stem}_boot_sd"] = float(vals.std(ddof=1))
                row[f"{stem}_lo"] = float(np.percentile(vals, lo_q))
                row[f"{stem}_hi"] = float(np.percentile(vals, hi_q))
            else:
                row[f"{stem}_boot_sd"] = np.nan
                row[f"{stem}_lo"] = np.nan
                row[f"{stem}_hi"] = np.nan
        row["n_boot_ok"] = int(len(draws[value_cols[0]].get((grp, day), ())))
        rows.append(row)

    out = pd.DataFrame(rows)
    out.attrs["n_boot"] = int(n_boot)
    out.attrs["ci"] = float(ci)
    out.attrs["seed"] = int(seed)
    return out
