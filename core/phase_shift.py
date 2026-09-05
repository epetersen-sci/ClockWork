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


def _group_labels(ds, group_by):
    """Per-fly group label built from one or more coords, joined with '_'."""
    cols = [c for c in group_by if c in ds.coords]
    if not cols:
        raise ValueError(
            f"None of {list(group_by)} are coordinates on this dataset; cannot form groups."
        )
    parts = [np.asarray(ds[c].values).astype(str) for c in cols]
    return np.array(["_".join(vals) for vals in zip(*parts)]), cols


def compute_group_phase_difference(
    ds,
    control_group,
    *,
    group_by=("genotype", "condition"),
    filter_hours=DEFAULT_PHASE_SHIFT_FILTER_HOURS,
    peak_prominence_frac=DEFAULT_PHASE_SHIFT_PEAK_PROMINENCE_FRAC,
    peak_distance_hours=DEFAULT_PHASE_SHIFT_PEAK_DISTANCE_HOURS,
    search_half_width_hours=DEFAULT_PHASE_SHIFT_MARKER_SEARCH_HALF_WIDTH_HOURS,
    activity_var="activity",
):
    """Per-day phase difference between each group and an unpulsed control group.

    This is the direct analogue of the lab's ``peakphaseplot.m``: average each
    group's activity, smooth it, take one peak per day, and report the pulsed
    group's peak time minus the control group's on the same day.

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
    control_group : str
        Label of the unpulsed reference group, as built from ``group_by``.
    group_by : sequence of str
        Coordinates whose combination defines a group.

    Returns
    -------
    dict
        ``per_day`` (DataFrame: group, day_index, peak_hours, control_peak_hours,
        phase_difference_hours), ``peak_times`` ({group: {day: hours}}),
        ``control_group``, and ``params``.
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

    labels, cols = _group_labels(ds, group_by)
    groups = sorted(set(labels))
    if control_group not in groups:
        raise ValueError(
            f"control_group {control_group!r} is not one of the groups present: {groups}"
        )

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

    control = peak_times[control_group]
    rows = []
    for grp in groups:
        for day, hours in sorted(peak_times[grp].items()):
            ctrl = control.get(day)
            rows.append(
                {
                    "group": grp,
                    "day_index": day,
                    "peak_hours": hours,
                    "control_peak_hours": ctrl,
                    "phase_difference_hours": (hours - ctrl if ctrl is not None else np.nan),
                    "n_flies": int((labels == grp).sum()),
                    # filtfilt pads the record ends, so the first/last day's peak
                    # sits partly on synthetic padding — flag, don't silently report.
                    "filter_edge": day == 0 or day == max(peak_times[grp]),
                }
            )

    return {
        "per_day": pd.DataFrame(rows),
        "peak_times": peak_times,
        "control_group": control_group,
        "params": {
            "group_by": list(cols),
            "filter_hours": float(filter_hours),
            "peak_prominence_frac": float(peak_prominence_frac),
            "peak_distance_hours": float(peak_distance_hours),
            "search_half_width_hours": float(search_half_width_hours),
        },
    }
