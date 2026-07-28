"""
calibrations.py — the single, documented home for user-owned SOFT thresholds.
================================================================================

Every threshold the *user* owns and may adjust lives here, defined **once**, with
its provenance. The UI controls and the code defaults both import from this
module — there is **no second copy anywhere**. Change a value here and it changes
everywhere (production classifier defaults, UI control defaults, the sweep
harness reference lines).

Why this module exists
----------------------
These are *calibrations*, not invariants: soft, movable lines the user is
entitled to set, never hard-gated into correctness.
They were previously scattered across ``rhythmicity_classification.py`` and
``periodograms.py``, plus a hardcoded copy in the page-3 UI. The classic rot is
the UI editing one copy while a stale default survives elsewhere; consolidating
removes that failure mode. The user reading this block can see *every line they
own at once*, and why each value sits where it does.

How to read each entry
----------------------
name — what it gates / classifies
default — the value (the constant below IS the single source of truth)
units — what the number measures
range — sensible allowed bounds, where one applies
provenance — the evidence it was calibrated against, and whether it's PROVISIONAL

Open question, recorded here so it is not mistaken for a settled value: whether
0.3 or SCAMP's historical 0.195 is the right per-fly AC rhythmicity-index cutoff
is still under investigation. Neither should be assumed correct; see the
AC section below for the full argument.
"""

from __future__ import annotations

# =============================================================================
# Lomb-Scargle (LS)
# =============================================================================

# DEFAULT_LS_POWER_THRESHOLD
#   name       : LS rhythmicity STRENGTH gate (the live classification cutoff).
#                A fly is rhythmic iff ``ls_power > this`` AND period in window.
#   default    : 0.006
#   units      : LS standard-normalized peak power — R^2 in [0, 1], the fraction
#                of variance the best period explains (LS's native strength
#                index, the equivalent of AC's RI and CWT's rhythmicity).
#   range      : (0.0, 1.0)
#   provenance : PROVISIONAL. Set in session 4 in the OPEN gap between the
#                arrhythmic Opa1xLdh doubles (M20/M22, power p90 ~0.0033) and the
#                rhythmic monitors (p10 ~0.0122) on the strong-phenotype §2c
#                cohort. On clean data
#                that gap is empty; on complex/marginal data it may fill in and
#                this line may need revisiting (Part 2 concordance test). NOT a
#                hard magic number — a soft, user-owned cutoff like AC's 0.3 RI.
DEFAULT_LS_POWER_THRESHOLD = 0.006

# DEFAULT_LS_FAP_THRESHOLD
#   name       : LS false-alarm probability (Baluev). REPORTED ONLY — not a gate.
#                rhythmicity_classification.classify_lomb_scargle classifies on
#                POWER (above); FAP is carried as a secondary significance value.
#   default    : 0.05
#   units      : probability (false-alarm probability of the peak).
#   provenance : Conventional α. Retained for the parameter-sweep harness and
#                FAP context. (FAP-citation fix: the FAP<0.05 convention was
#                mis-attributed to Pfeiffenberger 2010, which prescribes a
#                chi-squared periodogram; canonical LS-FAP refs are
#                Horne & Baliunas 1986 / Refinetti 2007.)
DEFAULT_LS_FAP_THRESHOLD = 0.05


# =============================================================================
# Autocorrelation (AC) — TWO DISTINCT numbers; do NOT collapse them
# =============================================================================
# The soft group-average sanity check and the internal per-fly classification
# cutoff are conceptually different things; keep them as separate entries so
# they can diverge. (See the SCAMP reference and the
# OPEN 0.195-vs-0.3 investigation below — note the live per-fly default is 0.3,
# NOT 0.195; this is exactly what that investigation is about.)

# DEFAULT_AC_RI_THRESHOLD
#   name       : AC per-fly classification cutoff — the LIVE value the AC
#                classifier uses (classify_autocorrelation: rhythmic iff
#                ``ac RI > this``). This drives every downstream filter.
#   default    : 0.3
#   units      : RI = raw peak autocorrelation (Levine "Rhythmicity Index"),
#                exposed as ``ac_power``.
#   range      : (0.0, 1.0)
#   provenance : Stricter practical cutoff (vs SCAMP's historical 0.195 below).
#                Soft lab convention; see ac_rhythm_strength for the normalized
#                statistical (Levine 2002 95% CI) interpretation.
DEFAULT_AC_RI_THRESHOLD = 0.3

# AC_RI_GROUP_CHECK
#   name       : AC group-average sanity check. An arrhythmic GROUP passes if
#                its MEAN AC RI falls below this. A soft eval criterion for
#                tests, NOT a per-fly gate.
#   default    : 0.3
#   units      : RI (group mean).
#   provenance : Manual lab convention. Numerically equal to the
#                per-fly classification cutoff today, but a DISTINCT concept —
#                kept separate so the two can move independently. (Currently used
#                as a test/eval criterion, not read by production classifier code.)
AC_RI_GROUP_CHECK = 0.3

# AC_RI_SCAMP_REFERENCE
#   name       : SCAMP historical per-fly RI cutoff (RI > 0.195), 2nd-day-peak
#                convention. REFERENCE ONLY — *not* currently wired as a live
#                default (the live per-fly cutoff is DEFAULT_AC_RI_THRESHOLD=0.3).
#   default    : 0.195
#   units      : RI.
#   provenance : SCAMP default (Levine 2002 in spirit; SCAMP rindex_raw.m /
#                autoco.m). Kept here as the concrete reference point for the
#                OPEN "0.195 vs 0.3" investigation: is 0.3 right, or is the
#                SCAMP 0.195 the intended per-fly cutoff? Unresolved — do NOT
#                wire this as a default without an explicit decision.
AC_RI_SCAMP_REFERENCE = 0.195


# =============================================================================
# CWT — reduction-method default + per-method rhythmicity cutoffs
# =============================================================================
# CWT's strength metric and its cutoff depend on the reduction method. Resolve
# the cutoff via rhythmicity_classification.cwt_threshold_for(method).

# DEFAULT_CWT_METHOD
#   name       : the DEFAULT CWT reduction method — which strength metric the
#                CWT classifier gates on when the user does not pick one. This
#                selects WHICH cutoff below applies (via cwt_threshold_for).
#   default    : 'global_rednoise'
#   units      : method name (one of 'ar1', 'global',
#                'global_rednoise', 'global_baseline', 'global_neighbor',
#                'ridge'); all remain selectable.
#   provenance : Flipped 'ar1' -> 'global_rednoise'.
#                'ar1' is a LENGTH-SENSITIVE significance ratio (its chi-squared
#                threshold divisor shrinks with record length), so it OVER-CALLS
#                long records — e.g. it calls the Per0 arrhythmic null ~88%
#                rhythmic on the 21-day complex data (B5.1), the over-call this
#                flip fixes. 'global_rednoise' is the range-robust, length-stable
#                local-prominence FORM of the same red-noise comparison WITHOUT
#                that length-sensitive division. It has BOTH halves of evidence,
#                independently verified in prior sessions:
#                  - REJECTION (simple §2c cohort, production path): arrhythmic
#                    red-noise ramps rejected; arrhythmic false-positive rate
#                    ~0/0/1.9% across 16-32/16-50/16-60 h search windows
#                    (width-stable, vs old 'global's 13.5/59.6/71.2% explosion);
#                    AC concordance ~96% (yardstick 98.8%); 24 h center 100%.
#                  - DETECTION (a real long-period line, ~43 h, on a long
#                    recording): LS 42.6 h vs CWT global_rednoise 43.2 h,
#                    LS<->CWT 92% within 2 h, per-fly strengths ~7-73 >> the 3.0
#                    cutoff; CWT correctly REFUSED the unresolvable short/gappy
#                    monitor 2044 (NaN, COI-correct) -> no spurious over-call.
#                Flipping the default therefore strictly improves the default
#                path. The 'global_rednoise' STRENGTH cutoff
#                (DEFAULT_CWT_REDNOISE_THRESHOLD) is still PROVISIONAL — this
#                flip does not finalize it (see that entry). The CWT
#                period-SEARCH window stays a SEPARATE, still-unfinalized user
#                calibration — NOT changed by this flip.
DEFAULT_CWT_METHOD = "global_rednoise"

# DEFAULT_CWT_AR1_THRESHOLD
#   name       : CWT rhythmicity cutoff for cwt_method='ar1' (a SELECTABLE method;
#                no longer the default — see DEFAULT_CWT_METHOD).
#   default    : 1.0
#   units      : Power.avg at the dominant period / 95% AR(1) red-noise threshold
#                (Torrence & Compo 1998). >1 ⇔ peak power significant at α=0.05.
#   provenance : Torrence & Compo 1998 BAMS 95% AR(1) red-noise significance.
DEFAULT_CWT_AR1_THRESHOLD = 1.0

# DEFAULT_CWT_GLOBAL_THRESHOLD
#   name       : CWT rhythmicity cutoff for cwt_method='global' — the length-
#                stable, COI-excluded NORMALIZED STRENGTH index (peak global
#                wavelet-spectrum power / MEAN band power). CWT's analog of
#                AC's RI / LS's R^2; mirrors the session-4 LS power-gate fix
#                (gate on strength, not significance). A fly is rhythmic iff
#                ``cwt_rhythmicity > this`` AND period in window.
#   default    : 1.7
#   units      : ratio (peak band power / mean band power, COI-excluded);
#                dimensionless, length-stable, >= 1 by construction (the peak is
#                >= the mean). 1.0 = perfectly flat spectrum (no peak); larger
#                = a sharper dominant circadian peak above the band background.
#   range      : [1.0, inf)
#   provenance : PROVISIONAL — placed in the gap between the known-rhythmic
#                monitors (M17/M19/M21: median strength ~2.0, p10 ~1.87, min
#                ~1.63) and the known-arrhythmic Opa1xLdh doubles (M20/M22:
#                median ~1.39, p75 ~1.61) on the strong-phenotype §2c simple
#                cohort (the normalization comparison; CSV in the CWT 'global'
#                writeup). The clusters are NOT cleanly disjoint on strength
#                ALONE: ~half the arrhythmic flies carry a spurious high-strength
#                peak pinned to the 30-32 h BAND EDGE — this is NOT a COI bug
#                (COI exclusion is working); it is the intrinsic red-noise scale
#                bias of the COI-excluded Torrence & Compo global wavelet spectrum
#                (unrectified power rises monotonically with scale; Liu et al.
#                2007), so a peak-finder on a non-oscillatory signal lands at the
#                top scale. Strength + the default (16,32) h window therefore
#                over-calls them (independently-measured AC concordance 92.5%).
#                The genuinely-rhythmic flies cluster tightly at ~24.7 +/- 0.45 h
#                and 0/107 exceed 30 h, so a tighter period window (e.g. (16,30) h)
#                lifts AC concordance to 96.9% (vs the 98.8% AC<->LS yardstick) —
#                a SEPARATE user calibration (the CWT period window), NOT baked
#                here. NOT a hard magic number — a
#                soft, user-owned cutoff set in the page-3 threshold explorer.
#                The 'ar1' default cutoff (1.0) is a significance ratio on a
#                DIFFERENT scale and is unrelated to this value.
DEFAULT_CWT_GLOBAL_THRESHOLD = 1.7

# DEFAULT_CWT_REDNOISE_THRESHOLD
#   name       : CWT rhythmicity cutoff for cwt_method='global_rednoise' — the
#                RANGE-ROBUST, length-stable local-prominence strength. A fly is
#                rhythmic iff ``cwt_rhythmicity > this`` AND period in window.
#   default    : 3.0
#   units      : ratio = peak power of the COI-excluded global wavelet spectrum
#                divided by the AR(1) red-noise EXPECTED power AT the peak period
#                (Torrence & Compo 1998 var*P_red(f_peak)). Dimensionless, >= 0;
#                ~1 = the peak is no stronger than the modeled red-noise floor
#                under it (an arrhythmic ramp); >> 1 = a genuine localized bump
#                several-fold above that floor.
#   range      : [0.0, inf)
#   provenance : PROVISIONAL. The fix for the 'global' band-edge artifact:
#                on a WIDE 16-50 h search an arrhythmic
#                red-noise ramp rising toward the long-period edge inflated the
#                'global' peak/mean ratio and passed; 'global_rednoise' measures
#                the peak against the LOCAL modeled red-noise floor under it, so
#                a ramp (which is exactly what AR(1) predicts at that period)
#                scores ~1 and is rejected, while a real bump (even at 48 h)
#                scores high. This is the length-stable strength FORM of the 'ar1'
#                significance ratio WITHOUT 'ar1's length-sensitive chi-squared
#                threshold division. Placed in the clean gap measured on the
#                strong-phenotype simple §2c cohort (DD) at the wide 16-50 h
#                window via the production path (evaluator+reviewer re-measured;
#                an earlier write-up had the arrhythmic max wrong): known-rhythmic
#                flies score min ~2.34 (an M18), most >> 5; known-arrhythmic
#                Opa1xLdh doubles score p95 ~2.2, max ~2.40 (an M22 at 33.8 h).
#                The clusters CROSS — no clean gap. At 3.0 the arrhythmic
#                false-positive rate is ~0% (max 2.40 < 3.0) but 3.0 CLIPS 2
#                genuine M18 rhythmic flies (2.34, 2.89); ~2.4-2.5 maximizes AC
#                concordance but is razor-thin. The rejection is STABLE as the
#                search widens 16-32 -> 16-50 -> 16-60 (the property 'global' lacks).
#                NOT a hard magic number — a soft, user-owned cutoff set in the
#                page-3 threshold explorer. The default-flip ar1->global_rednoise
#                is DONE (DEFAULT_CWT_METHOD = 'global_rednoise'): the REAL-DATA
#                detection gate ran on L775A (B8) and confirmed detection of the
#                known long period while still rejecting arrhythmic ramps.
#   USER DECISION (2026-06-30): finalized at **2.0** — a deliberately PERMISSIVE
#                default (conservatism is opt-in by raising it). 2.0 is NOT tied to
#                a metric-intrinsic significance level (the metric is peak / AR(1)
#                expected power at the peak; ~1 = no excess over the local red-noise
#                floor, so 2.0 means "peak is >=2x the modeled floor" — a strength
#                ratio, not an alpha). CONSEQUENCE (intended, not a failure): 2.0
#                sits BELOW the measured dev arrhythmic-strength max (~2.40), so a
#                few known-arrhythmic Opa1xLdh doubles (M20/M22) flip to
#                rhythmic-CALLED, and it RECOVERS the 2 genuine M18 rhythmic flies
#                that 3.0 clipped (~2.34/2.89). Was 3.0 (earlier provisional value).
DEFAULT_CWT_REDNOISE_THRESHOLD = 2.0

# DEFAULT_CWT_RIDGE_THRESHOLD
#   name       : CWT rhythmicity cutoff for cwt_method='ridge'; also the page-3
#                manual CWT classification-control default.
#   default    : 0.30
#   units      : fraction of analyzable timepoints with a confident circadian ridge.
#   range      : (0.0, 1.0)
#   provenance : Lab choice — no published binary cutoff exists for CWT-based
#                metrics in fly DAM data (Leise 2011/2013/2015). Use with caution
#                and report the exact value used.
DEFAULT_CWT_RIDGE_THRESHOLD = 0.30

# DEFAULT_CWT_MIN_PERIOD / DEFAULT_CWT_MAX_PERIOD — the CWT period range (ONE knob).
#   name       : the CWT period-SEARCH range. The CWT CLASSIFY window TRACKS this
#                same range (one knob — see below); they are not allowed to differ.
#   default    : (16.0, 36.0) h
#   units      : hours (free-running circadian period).
#   range      : min >= 1; max > min. 16 h floor = the ultradian/circadian boundary.
#   provenance : USER DECISION (2026-06-30). Moved here from periodograms.py to be
#                single-sourced (§5); periodograms re-exports these names.
#                **ONE KNOB:** search range == classify window. The bug just killed
#                was these differing (search 16-50, classify 16-32); to prevent
#                recurrence, classify_cwt derives its window from the search range
#                stamped on the analysed dataset (ds.attrs['cwt_min_period'/
#                'cwt_max_period']) rather than from a separate hardcoded default —
#                so they CANNOT diverge. Page 3's single period-range control feeds
#                both the search and (via those attrs) the classify gate.
#                **TRADEOFF (surfaced, intended):** a 36 h ceiling is the
#                circadian-focused conservative default. It does NOT detect genuine
#                LONG-period lines: e.g. L775A (~43 h, monitor 1013) reads
#                ARRHYTHMIC at this default — that is the intended knob, not a
#                regression. To detect a known long-period line, WIDEN this range
#                (e.g. to 50 h); the 'global_rednoise' metric stays range-robust so
#                widening does not re-introduce arrhythmic over-calls (Part B:
#                rejection is the metric's job, not the window's). The prior 50 h
#                default's "must reach 50 to see long periods" rationale is
#                superseded by this user choice.
DEFAULT_CWT_MIN_PERIOD = 16.0
DEFAULT_CWT_MAX_PERIOD = 36.0


# =============================================================================
# MESA (Maximum Entropy / Burg AR) — no user threshold
# =============================================================================
# MESA has NO significance test of its own, so it does not own a rhythmicity
# cutoff. In the page-3 threshold explorer its rhythmic call BORROWS the
# Autocorrelation RI (DEFAULT_AC_RI_THRESHOLD above); MESA supplies only the
# period. There is therefore intentionally no DEFAULT_MESA_* threshold here.
# (mesa_power, the peak/median PSD SNR, is still reported in the summary table
# as information, but it does not gate anything.)


# =============================================================================
# Period analysis — record length
# =============================================================================

# DEFAULT_MIN_DD_DAYS_FLOOR
#   name       : Recommended (NOT enforced) minimum DD days for period analysis.
#                A FLAG, not a filter: flies below it are surfaced
#                with their record length, never dropped (page 3 passes
#                min_num_days=0 to the kernels).
#   default    : 4.0
#   units      : days of constant darkness (DD).
#   range      : >= 0
#   provenance : The robustness characterization
#                shows the
#                rhythmicity call recovers by ~4 DD days and the period has
#                largely converged toward the longest-record result there. This
#                is convergence-from-longest-record guidance — NOT an accuracy
#                threshold and NOT a hard pass/fail gate. Soft, user-editable.
DEFAULT_MIN_DD_DAYS_FLOOR = 4.0


# =============================================================================
# Gap handling — the size cutoff between a "small" (bridgeable) and a "large"
# (unfillable) intra-record gap
# =============================================================================

# DEFAULT_GAP_THRESHOLD_MINUTES
#   name       : gap-size cutoff X separating a SMALL gap (≤ X — in principle
#                bridgeable for the one algorithm that needs a regular grid, CWT)
#                from a LARGE gap (> X — unfillable, falls through to the
#                large-gap policy = longest-clean-segment). Formalizes the
#                reserved `gap_threshold_minutes=60` slot. Cross-references the
#                page-1 "Gap threshold" control (default 1.0 h) and
#                `dam_utilities.split_xarray_dataset`/`_select_longest_segments`'s
#                `gap_threshold_minutes` (default 60) — they should read X from
#                here so there is one definition, no second copy.
#   NOTE (2026-07-01): for the CWT/AC block extractor the small/large SPLIT here
#                is SUPERSEDED by the single bridge ceiling
#                DEFAULT_MAX_BRIDGE_GAP_MINUTES below — that one knob now decides
#                what is bridged (≤ ceiling) vs a hard break (> ceiling), collapsing
#                the former tiny/gray-zone split. This X value is retained for the
#                page-1/page-2 splitter and split_xarray_dataset (segment splitting),
#                which are a SEPARATE concern from the worker-local bridge.
#   default    : 60.0   (X = 1 h, PROVISIONAL)
#   units      : minutes (elapsed time between two consecutive finite samples,
#                minus one sampling interval = the missing span).
#   range      : > 0
#   provenance : PROVISIONAL. 1 h matches the value already live elsewhere in the
#                app (page-1/page-2 "Gap threshold" 1.0 h; split_xarray_dataset's
#                gap_threshold_minutes 60). RATIONALE: you cannot honestly fill a
#                gap much beyond ~1 h — past that the flanking data no longer
#                constrains the interior and any interpolant assumes behavior that
#                was never measured (§2a forbids assuming it was zero/sleep), so
#                a sub-1 h bridge is below CWT band sensitivity (the 16-50 h
#                analysis band) while a longer bridge injects a low-frequency ramp
#                the wavelet can see (see GAP_SIZE_DISTRIBUTION.md).
#                CALIBRATION, not an invariant (rubric §3): surfaced and
#                user-editable, NEVER baked as a gate. This value DEFINES the
#                small/large boundary but does NOT itself enable any fill — the
#                CWT small-gap transient fill is DEFERRED pending the user's
#                post-distribution decision. The BUILD C gap-size distribution
#                found that on real long-recording data EVERY fly carrying a
#                5-60 min "gray zone" gap ALSO carries a multi-day (unfillable)
#                gap, so bridging
#                ≤ X gaps would recover NO additional fly — the distribution
#                evidence the user weighs in deciding whether to build the fill
#                at all.
DEFAULT_GAP_THRESHOLD_MINUTES = 60.0

# DEFAULT_MAX_BRIDGE_GAP_MINUTES
#   name       : the SINGLE gap-bridge CEILING — the max gap (minutes) that the
#                CWT/AC block extractor will BRIDGE (worker-local linear
#                interpolation of the interior NaN cells). A gap whose missing span
#                is ≤ this ceiling is bridged (the flanking segments are joined and
#                the interior NaN cell(s) are linearly interpolated), so a dropped
#                read does NOT sever an otherwise-clean record. A gap LARGER than
#                the ceiling stays a HARD BREAK → longest-clean-segment (status quo,
#                L775A-preserving). One knob; nothing between "tiny" and the ceiling
#                is treated specially any more.
#   default    : 60.0
#   units      : minutes (the missing span — elapsed time between two consecutive
#                finite samples, minus one sampling interval).
#   range      : [0, 120].  **0 = bridging OFF** (nothing is bridgeable → the
#                extractor falls back to a size-blind longest contiguous-finite run,
#                exactly the pre-bridge behavior).
#   readers    : periodograms._extract_longest_continuous_block (the ceiling param),
#                the wavelet_analysis / autocorrelation_analysis
#                `max_bridge_gap_minutes` parameter (threaded through the workers),
#                compute_single_fly_scalogram, and the page-3 "Max gap to bridge"
#                control (app/pages/3_Period_Analysis.py). One definition, no copy.
#   provenance : USER-OWNED CALIBRATION, not an invariant: surfaced and
#                user-editable, NEVER baked as a gate. This ONE ceiling COLLAPSES the
#                former tiny-gap bridge (the old T = 5 min DEFAULT_TINY_GAP_BRIDGE_
#                MINUTES) AND the 5–60 min "gray zone" fill into a single controlled
#                ceiling: the gray zone is bridged UP TO this
#                ceiling. Default 60 = the full small/large boundary
#                DEFAULT_GAP_THRESHOLD_MINUTES (X, 1 h): you cannot honestly fill a
#                gap much beyond ~1 h — past that the flanking data no longer
#                constrains the interior and any interpolant assumes behavior that was
#                never measured (§2a forbids assuming it was zero/sleep); a sub-1 h
#                bridge stays below CWT band sensitivity (the 16–50 h analysis band)
#                while a longer bridge would inject a low-frequency ramp the wavelet
#                can see (see GAP_SIZE_DISTRIBUTION.md). VALIDATED ≤ 60 min on the
#                fully-alive control cohort:
#                bridging ≤ 60 min does not distort the CWT period beyond the SCAMP
#                0.5 h concordance tolerance. The [60, 120] band is user-adjustable but
#                BEYOND the validated range. Over-ceiling behavior is UNCHANGED — a gap
#                > ceiling is a hard break → longest-clean-segment (so a real
#                multi-day outage still severs the record and L775A-style long lines
#                are preserved).
#                §2a CONTRACT: the interpolated cells are TRANSIENT and worker-local —
#                fed to CWT/AC only, NEVER written back to the xarray Dataset or the
#                .nc. Stored truth stays NaN.
DEFAULT_MAX_BRIDGE_GAP_MINUTES = 60.0


# =============================================================================
# Phase shift (light-pulse) — page 11
# =============================================================================
# Both phase-shift methods reduce to the same shape: derive ONE phase-marker time
# per calendar day, regress marker-vs-day separately before and after the pulse,
# and read the gap between the extrapolated pre-pulse line and the post-pulse line
# at the pulse day. The knobs below split into (a) how the daily marker is found —
# separate sets for the two methods — and (b) how many days a fit needs to be
# trusted, shared by both.

# DEFAULT_PHASE_SHIFT_FILTER_HOURS
#   name       : Butterworth low-pass cutoff used by the PEAK method before peak
#                picking. Smooths the bimodal/noisy daily activity trace down to
#                one dominant daily hump so a single peak time is well defined.
#   default    : 12.0
#   units      : hours (filter cutoff period, as in preprocessing.PreprocessConfig
#                .lopass_hours — the same filtfilt Butterworth, not a second copy).
#   range      : (0, 24). 0 disables filtering (raw trace peak — very noisy).
#   provenance : DIRECT LAB PRECEDENT — this is the default in the lab's own MATLAB
#                script, SCAMP Scripts/phaseshiftanalyses/peakphaseplot.m
#                (``if nargin<4, filter_hours=12``), which is the method being
#                ported. The MATLAB docstring notes larger values as the first
#                remedy for false peaks, so raising it is the intended adjustment.
DEFAULT_PHASE_SHIFT_FILTER_HOURS = 12.0

# DEFAULT_PHASE_SHIFT_PEAK_PROMINENCE_FRAC
#   name       : minimum peak prominence for the PEAK method, as a fraction of the
#                day's filtered activity range. Rejects ripple on the smoothed
#                trace from being picked as the daily peak.
#   default    : 0.05
#   units      : fraction of (max - min) of the day's filtered trace.
#   range      : (0, 1).
#   provenance : PROVISIONAL. Mirrors the existing PEAK_PROMINENCE_FRAC = 0.05
#                convention already used for autocorrelation peak picking in
#                periodograms.py, for consistency of idiom. NOT independently
#                validated on the per-day activity axis — the daily-peak problem is
#                not the AC-lag problem, so this number is inherited, not proven.
DEFAULT_PHASE_SHIFT_PEAK_PROMINENCE_FRAC = 0.05

# DEFAULT_PHASE_SHIFT_PEAK_DISTANCE_HOURS
#   name       : minimum separation between candidate peaks in the PEAK method.
#                Stops two shoulders of one broad activity bout from registering as
#                two competing daily peaks.
#   default    : 12.0
#   units      : hours.
#   range      : (0, 24).
#   provenance : PROVISIONAL. Exactly one dominant peak per ~24 h day is expected,
#                so a half-day floor is the natural geometric choice; no empirical
#                calibration behind the specific value.
DEFAULT_PHASE_SHIFT_PEAK_DISTANCE_HOURS = 12.0

# DEFAULT_PHASE_SHIFT_ONSET_THRESHOLD_FRAC
#   name       : activity level a fly must exceed for the ONSET method to call the
#                start of the daily active phase, expressed as a MULTIPLE OF THE
#                DAY'S OWN MEAN activity (measured on the sustain-window rolling
#                mean, not the raw per-minute counts).
#   default    : 0.75
#   units      : dimensionless multiple of the record's mean smoothed activity.
#   range      : (0, ~6). Above the trace's peak-to-mean ratio nothing is ever
#                crossed and the method finds no onsets at all.
#   provenance : PROVISIONAL. RELATIVE rather than absolute because absolute count
#                magnitudes are not comparable across flies, genotypes or recorders
#                (a FlyBox channel and a DAM tube are not on the same scale), so an
#                absolute cutoff tuned on one cohort finds nothing on another. An
#                earlier absolute-count version of this knob failed exactly that
#                way: at 1-minute resolution a quiet-but-alive fly has ~100 nonzero
#                minutes per day with no run of consecutive above-threshold minutes
#                longer than ~5, so no "N minutes above a level" rule could fire.
#                0.75 was then SELECTED AGAINST GROUND TRUTH — a sweep over
#                (threshold_frac x smooth_minutes) scored by how well a known
#                injected shift is recovered across 6 flies x 6 shift magnitudes on
#                the §2c cohort (see tests/test_phase_shift.py). Caveat that matters:
#                that is ONE cohort with ONE waveform shape, and while the median
#                error at this setting is ~2 min, the TAIL is poor (~10% of cases off
#                by >2.5 h, worst ~10 h) — see the method comparison in the module
#                docstring. Onset is the non-default method for this reason. Verify
#                against an actogram before trusting onset numbers.
DEFAULT_PHASE_SHIFT_ONSET_THRESHOLD_FRAC = 0.75

# DEFAULT_PHASE_SHIFT_ONSET_SMOOTH_MINUTES
#   name       : width of the rolling mean the ONSET method reads the activity
#                envelope from. Onset is the first upward crossing of the threshold
#                on THIS smoothed trace, so this sets how much per-minute spikiness
#                is ignored before an onset is called.
#   default    : 180.0
#   units      : minutes.
#   range      : (0, 360). Too small and single-minute spikes during the rest phase
#                trigger a false onset; too large and the onset edge itself is
#                smeared past the resolution the analysis needs.
#   provenance : PROVISIONAL, selected against ground truth in the same sweep as
#                DEFAULT_PHASE_SHIFT_ONSET_THRESHOLD_FRAC (see that entry). 3 h is
#                also roughly the visual smoothing implied by reading onsets off a
#                30-min-binned actogram, which is how the value is arrived at
#                manually, so the number is at least not at odds with practice.
#                NOTE this knob was previously (mis)named "sustain" and described as
#                a dwell time; it has never been a dwell time in the shipped code —
#                it is a smoothing width.
DEFAULT_PHASE_SHIFT_ONSET_SMOOTH_MINUTES = 180.0

# DEFAULT_PHASE_SHIFT_MARKER_SEARCH_HALF_WIDTH_HOURS
#   name       : half-width of the window, centred on the PREVIOUS day's marker, in
#                which the next day's marker is sought. This is what replaces the
#                MATLAB script's manual click-to-match step: instead of a human
#                deciding which peak on day N corresponds to which peak on day N+1,
#                the marker is tracked day to day within this window. Applies to
#                BOTH methods (peak and onset).
#   default    : 4.0
#   units      : hours.
#   range      : (0, 12). Too small and a free-running marker drifts out of reach;
#                too large and a spurious secondary peak can capture the track.
#   provenance : PROVISIONAL. Must comfortably exceed the largest plausible one-day
#                drift (a 22-26 h period drifts <= ~2 h/day) while staying well
#                under the ~12 h at which the wrong daily hump becomes reachable;
#                4 h sits between those bounds. Not empirically tuned. NOTE this
#                window bounds day-to-day tracking WITHIN an epoch only: the
#                post-pulse run is seeded by its own whole-day search rather than
#                from the last pre-pulse marker, so an arbitrarily large shift
#                across the pulse is still measurable.
DEFAULT_PHASE_SHIFT_MARKER_SEARCH_HALF_WIDTH_HOURS = 4.0

# DEFAULT_PHASE_SHIFT_MIN_PRE_DAYS / DEFAULT_PHASE_SHIFT_MIN_POST_DAYS
#   name       : minimum days with a usable marker on each side of the pulse before
#                a per-fly phase shift is reported at all. Below either floor the fly
#                is returned with an explicit "insufficient" status and NO number —
#                a badly-constrained 2-point regression is worse than an abstention.
#   default    : 3 and 3
#   units      : calendar days (days carrying a marker, not elapsed days — a day
#                whose marker was lost to a data gap does not count).
#   range      : [2, inf). 2 is the algebraic floor for a line and is enforced in
#                code regardless; these calibrations sit ABOVE that floor.
#   provenance : PROVISIONAL / SCIENTIFIC DEFAULT. 3 days is the conventional
#                minimum for reading a free-running slope off an actogram, and is
#                kept here as a user-editable calibration rather than hard-coded
#                precisely because it is a judgment call (rubric §3), not a derived
#                bound. Users with short records will want to lower it and should
#                know the fits get noisier when they do.
DEFAULT_PHASE_SHIFT_MIN_PRE_DAYS = 3
DEFAULT_PHASE_SHIFT_MIN_POST_DAYS = 3

# DEFAULT_PHASE_SHIFT_TRANSIENT_SKIP_DAYS
#   name       : days immediately after the pulse EXCLUDED from the post-pulse fit,
#                to let transient cycles pass before the new steady state is read.
#   default    : 0 (no days skipped — the post-pulse fit starts the first full day
#                after the pulse day, which is itself always excluded)
#   units      : days.
#   range      : [0, inf) in principle; practically 0-3.
#   provenance : USER-OWNED, DEFAULT OFF. Transient cycles before re-established
#                steady state are a real documented phenomenon, but how many days
#                to discard is protocol- and genotype-specific, so this ships as a
#                no-op the user opts into rather than a silent default that would
#                quietly change every reported number.
DEFAULT_PHASE_SHIFT_TRANSIENT_SKIP_DAYS = 0


# =============================================================================
# Reserved — calibrations that will land HERE rather than scatter again
# =============================================================================
# When these are formalized, define them in this module and have the UI + code
# read them from here (one definition, no second copy):
#   - Sleep-state duration bins (short / intermediate / long bout cutoffs;
#     Phase 4, Abhilash 2026). Currently not centralized.
#   - (Gap-length limit is now formalized above as DEFAULT_GAP_THRESHOLD_MINUTES;
#     the live page-1 "Gap threshold" control and split_xarray_dataset's
#     gap_threshold_minutes should be migrated to read X from there.)


__all__ = [
    "DEFAULT_LS_POWER_THRESHOLD",
    "DEFAULT_LS_FAP_THRESHOLD",
    "DEFAULT_AC_RI_THRESHOLD",
    "AC_RI_GROUP_CHECK",
    "AC_RI_SCAMP_REFERENCE",
    "DEFAULT_CWT_METHOD",
    "DEFAULT_CWT_AR1_THRESHOLD",
    "DEFAULT_CWT_GLOBAL_THRESHOLD",
    "DEFAULT_CWT_REDNOISE_THRESHOLD",
    "DEFAULT_CWT_RIDGE_THRESHOLD",
    "DEFAULT_CWT_MIN_PERIOD",
    "DEFAULT_CWT_MAX_PERIOD",
    "DEFAULT_MIN_DD_DAYS_FLOOR",
    "DEFAULT_GAP_THRESHOLD_MINUTES",
    "DEFAULT_MAX_BRIDGE_GAP_MINUTES",
    "DEFAULT_PHASE_SHIFT_FILTER_HOURS",
    "DEFAULT_PHASE_SHIFT_PEAK_PROMINENCE_FRAC",
    "DEFAULT_PHASE_SHIFT_PEAK_DISTANCE_HOURS",
    "DEFAULT_PHASE_SHIFT_ONSET_THRESHOLD_FRAC",
    "DEFAULT_PHASE_SHIFT_ONSET_SMOOTH_MINUTES",
    "DEFAULT_PHASE_SHIFT_MARKER_SEARCH_HALF_WIDTH_HOURS",
    "DEFAULT_PHASE_SHIFT_MIN_PRE_DAYS",
    "DEFAULT_PHASE_SHIFT_MIN_POST_DAYS",
    "DEFAULT_PHASE_SHIFT_TRANSIENT_SKIP_DAYS",
]
