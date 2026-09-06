# ClockWork backlog

Thirteen issues found while mapping the codebase for the sidebar/page reorganization
(branch `page-reorganization`), plus one (item 14) found later while closing them. None is
a regression introduced by the reorganization — they all predate it. They were left alone
there because fixing them changes analysis behaviour or touches `core/`, both of which were
out of scope for a re-cut of `app/`.

They were triaged into four sections:

- **[A. Ready to fix](#a-ready-to-fix)** — the decision is made and written down. Pick one
  up and implement it as described; no further judgement needed.
- **[B. Held](#b-held)** — a real decision that has not been made yet.
- **[C. Audit](#c-audit)** — no code change; something to go and check.
- **[D. Done](#d-done)** — closed, with the commit that closed it.

**All thirteen triaged issues, plus two found while fixing them, are closed**
(9, 8, 2, 1, 11, 14, 3, 4, 5, 15, 12, 10, 7). One new issue is open — item 16,
found by an xarray review of the last branch — plus one held decision (item 6)
and one data audit (item 13).

**Item numbers are original and stable.** They are referenced from the
reorganization PR and from the commits that closed them, so they are not renumbered
when items move between sections or get closed. A new issue takes the next free
number (14 and 15 were added this way), and never reuses a closed one.

(`app/app_pages/data_groups.py` used to carry a pointer to item 12 in its module
docstring. Item 12 is closed and that pointer is gone, so no code references this
file any more.)

**Line references** in the two remaining items have been re-checked against
`d998b53`. One had moved and is fixed: item 6's phase-registry
reference, `core/dam_utilities.py`, is now at line 1413 (was 1380). Item 13 cites
no line numbers. Re-check any reference against `main` before trusting it — several
drifted over the course of closing section A, and a wrong line number costs more
than no line number.

---

## What this file is for now

Section A holds one item (16). Two habits from clearing the rest are worth
keeping:

**Capture a baseline before any item whose Verify step says "same as before".**
Item 5 needed one and it paid for itself — 721 of 724 SCAMP files matched
byte-for-byte, so the three that differed could be examined individually instead
of argued about. The recipe: run the export on `main`, `sha256sum` the tree, keep
a copy, then re-run and `sha256sum -c` after the change.

**Grep for call sites; do not trust a list in this file.** Item 5's consumer
table was missing two of five, and item 3's Verify step asked for a difference
the code could not produce. Both are recorded in their Done entries.

There is now a test suite (`tests/`, run by CI on every PR), so a Verify step can
usually become a test rather than a one-off check done by hand. `tests/README.md`
covers what the fixtures do and do not imitate.

---

# A. Ready to fix

## 16. The gap trim masks `(id, time)` variables on the wrong axis

`core/dam_utilities.py:1076-1079`, inside `_select_longest_segments`. The loop
that blanks each fly's out-of-segment minutes indexes `[time, fly]`:

```python
for fly_idx, info in enumerate(segment_info):
    arr[: info["segment_start_idx"], fly_idx] = np.nan
    if info["segment_end_idx"] + 1 < arr.shape[0]:
        arr[info["segment_end_idx"] + 1 :, fly_idx] = np.nan
```

**Dimension order is not uniform in this codebase.** `activity` and `moving` are
`(time, id)`; the four sleep masks are `(id, time)`. `activity` has its own
correctly-shaped block above, so the sleep masks are the casualties, and they
are hit twice over:

- The **trimmed fly keeps** its out-of-segment sleep — the data the trim exists
  to discard survives into the export.
- **Every other fly** gets a spurious `-1` at low time indices.
- The tail branch tests `end_idx + 1 < arr.shape[0]`, but `shape[0]` is `n_id`
  for these vars, so it is essentially never true and the post-gap tail is never
  masked at all.

Reproduced on the test fixture (6 flies × 8640 min, one 1200-min gap in fly 0,
sliced to DD):

```
moving (time,id) masked per fly: [2321, 0, 0, 0, 0, 0]   <- correct
sleep  (id,time)     -1 per fly: [   1, 1, 1, 1, 1, 1]   <- wrong on both counts
```

This reaches real output: `export_helpers.phase_slice` →
`split_xarray_dataset` → the per-phase `.nc` written by `export_data.py` and the
SCAMP files written by `export_scamp.py`.

**Fix.** Mask in xarray so it broadcasts by dim NAME rather than by position:
build one `(id, time)` boolean `keep` DataArray from `segment_info` and apply
`xr.where(keep, da, sentinel)` per variable. The repo already has the pattern —
`select_phase` restores dim order with `masked[var] = m.transpose(*da.dims)`
(`core/dam_utilities.py:1384`), and every other numpy drop-out in `core/` calls
`.transpose("time", "id")` first. Doing it in xarray also removes the explicit
`astype(float)` above, so the int8 sleep masks would never need restoring
afterwards — which would let `export_helpers.phase_slice` drop its int8 repair
block entirely.

**Verify.** A fly with a ≥60-minute gap must lose the same minutes from `sleep`
as from `moving`, and a fly without a gap must lose none. `tests/conftest.py`
already builds the mixed dim order deliberately, so the fixture is one punched
gap away from covering it. Then re-run the SCAMP baseline diff from item 5 —
this changes exported values, so the 721-of-724 byte match will move, and the
files that change should be exactly the trimmed flies.

**Found by** an xarray-skill review of the item 12/10/7 branch, and verified
before filing. Not fixed there because it is pre-existing, changes exported
numbers, and wants its own baseline diff.


# B. Held

## 6. ~1900 lines of a complete, unwired feature

13 of the 26 public functions in `core/plotting.py` have no caller anywhere. Eleven sit
inside one section — `# Sleep State Analysis Plots (Abhilash et al. 2026)`:

`normalized_waveform_overlay`, `initiation_probability_plot`, `rebound_bar_plot`,
`period_amplitude_plot`, `sleep_state_scalogram`, `single_fly_scalogram_plotly`,
`group_ridge_density_plotly`, `rose_plot`, `rose_plot_with_activity`, `polar_gating_plot`,
`ultradian_amplitude_plot`

They pair exactly with two orphaned analysis functions in `core/periodograms.py` —
`sleep_cwt_analysis` and `ultradian_rhythmicity_ls` — plus `compute_single_fly_scalogram`.
There is even a phase-requirement registry entry for `sleep_cwt_analysis` at
`core/dam_utilities.py:1413`, so some wiring was anticipated.

**Re-verified at `d0a477c`: all of these still have zero real callers.** The only textual
hits are docstrings, module-header lists and print strings.

**This is a decision, not a cleanup.** It is a coherent, fully-built Abhilash sleep-state
analysis that never got a UI. Either build the page it was written for, or delete it.
Leaving ~1900 lines of dead scientific code in the tree is the worst of the three options.

**Status: next in line.** Being tackled as its own piece of work, not as part of the
category-A sweep.

Also unused, but ordinary rot rather than a lost feature: `rhythmicity_violin_grid` and
`rhythmicity_long_to_wide_csv`; `core/load_and_save_datasets.py`'s four flat exporters
(`save_data_to_csv`, `save_data_to_excel`, `export_activity_to_csv`,
`export_averaged_data_to_csv`); and `core/dam_utilities.py`'s `split_activity_dataframe`
and `trim_first_dd_day`. These can go with whichever decision item 6 reaches.

---

# C. Audit

## 13. Datasets built before the fly/metadata mislabeling fix

**The code bug is fixed** in `d0a477c` — `create_xarray_dataset` now reindexes the metadata
onto `dam_data.columns` before building any coord. What remains is a **data** question.

The bug: `create_xarray_dataset` paired the activity matrix with the metadata
*positionally* — values from `dam_data.columns`, per-fly coords (`id`, `genotype`, `group`,
start/stop, …) from `metadata` row order — and nothing checked the two agreed. They
diverged whenever the metadata rows were not already in **string**-sorted
`(Monitor, start_datetime)` order, because the activity frame is built from
`unique_combos.sort_values(["Monitor", "start_datetime"])` with `Monitor` cast to `str` —
so `"10"` sorts before `"2"`. A metadata file listing monitors 2 and 10 in natural numeric
order produced:

```
metadata row order: ['20250301_2_1', '20250301_2_2', '20250301_10_1', '20250301_10_2']
data column order : ['20250301_10_1', '20250301_10_2', '20250301_2_1', '20250301_2_2']
xarray genotype   : ['AAA',           'AAA',           'ZZZ',          'ZZZ']
```

Monitor 10's traces were analysed under monitor 2's ids and genotypes, and vice versa. It
failed silently — the shapes matched, so nothing raised. Any experiment mixing single- and
double-digit monitor numbers was exposed.

**The audit.** Loading a `.nc` via Path B does not go through `create_xarray_dataset`, so a
file saved before `d0a477c` keeps its mislabeled coords. Re-importing from the raw DAM
files is the only repair.

To identify affected datasets, check each experiment's metadata file for the trigger: are
the `Monitor` values in an order that differs from their **string** sort? A file whose
monitors are all the same digit count is safe. Mixed digit counts listed in numeric order
are affected.

Work out which saved datasets and published figures came from an affected metadata file.

**Note on `example_data`.** It cannot exercise this bug: its monitors are 17–22, all two
digits, so numeric and string order agree. The audit needs the lab's real metadata files.

---

# D. Done

Closed items, newest first. The full description of each lives in the commit that closed
it — `git show <sha>` — rather than being kept here, so this section stays an index.

## 7. Inverted dependencies in `core/plotting.py`

`d998b53`. `sleep_bout_duration_lines` takes the computed frames;
`group_ridge_density_plotly` takes an already-filtered dataset;
`wavelet_analysis` returns the group averages instead of writing PNGs, which
`export_helpers.save_group_average_scalograms` now does. No deferred analysis
imports remain in `plotting.py`, and it imports first in a clean interpreter.

Note for whoever settles item 6: the rhythmic-filter inversion was inside
`group_ridge_density_plotly`, which has no callers and is on item 6's unwired
list. If that section is deleted, this part goes with it.

## 10. Surface the rest of the data-integrity report

`5386621`. Per-monitor detail now renders in an expander at import instead of
going only to the console, and the aggregate counters are stamped onto `attrs`
as plain ints so a reloaded `.nc` can still report its own quality. The renderer
is silent on files that predate the counters — absence means "not recorded",
which is not "clean".

## 12. Wire up `regroup_dataset` so groups can be redefined after import

`e1e6841`. "Redefine groups" on Groups & subsets, writing back to `dataset` and
`dataset_full` and clearing every derived cache. `README.md:189` was already
claiming this worked; it does now.

Needed a new `dam_utilities.group_defining_coords`, because
`group_defining_columns` asks its question of a metadata DataFrame and a
reloaded `.nc` never had one.

## 15. Re-running sleep analysis on a reloaded `.nc` crashes

`546e69b`. The drop of the previous run's variables now happens before
`analysis_ds` is derived, so the per-fly loop no longer takes the cartesian
product of `time` and `sleep_bout_number`.

Worth remembering how it presented: the visible symptom was a duplicate-index
ValueError, but the cause was a 134x row blow-up that ran for minutes first. A
slow step that then fails is worth reading as one bug, not two.

## 5. Delete the `dataset_LD` / `dataset_DD` session caches

`1868ba6`. Consumers derive phase views instead: analysis pages via
`select_phase`, the two export pages via the new `export_helpers.phase_slice`,
which re-slices from the split parameters recorded on the master.

**The consumer table in this entry was incomplete.** It listed three sites and
missed two analysis-side readers — `app/ui/period_context.py` (a whole
phase-picker branch plus a merge-back block) and `rhythmicity.py`. It also said
"item 3 removes the last analysis-side reader", which was wrong for the same
reason. Removing the writes would have broken both. If you inventory call sites
for a future item, grep rather than trusting a table in this file.

Also note the int8 restoration **moved** rather than being deleted as this entry
implied: slicing pads with NaN and upcasts the sleep masks, so dropping it
outright would silently change every per-phase `.nc` from int8 to float64. It
now lives in `phase_slice`, at the point of slicing.

## 4. Retire the legacy `split_phase` attr

`de207a4`. Both writers removed; the three `data_curate_split.py` reads migrated
to `dataset_meta`. The read-side migration is kept and now says so at each
surviving site — it is what loads `.nc` files saved before the change.

## 3. HMM model selection and analysis run on different data

`06bb8e7`. Explicit LD/DD phase picker sourced from `select_phase`, matching
`hmm_analysis.py`, plus a note to match the phase on both pages. Widget keys
deliberately left unlinked.

**This entry's Verify step needed a correction.** It says to "confirm the fly
counts differ" between LD and DD. `select_phase` returns a NaN-masked view over
the full time axis, so `len(ds['id'])` is *identical* for both — the page had to
start reporting the count of flies with usable minutes in the selected phase
(189 LD vs 178 DD on `example_data`) before the check meant anything. That count
now also drives the fold slider.

## 14. Dataset fingerprints excluded from every `@st.cache_data` key

`8e1f6ce`, `486330d`. Found while verifying item 11. Every cached helper on
`sleep_activity.py`, `periodograms.py` and `rhythmicity.py` took its fingerprint as
`_fp`, and Streamlit's underscore rule is **syntactic** — it drops *any*
leading-underscore parameter from the cache key, not just the unhashable dataset. So the
one argument that exists to encode which flies are in `ds` never reached the key.

`rhythmicity._build_period_summary_df(_fp, _ds)` was the worst: those were its only two
parameters, so the key was **empty** and the exported per-fly period summary was computed
once per session and then returned unchanged for every later dataset. The reference
example in `core/dataset_meta.dataset_fingerprint` that all three sites were copied from
was itself half wrong (`fp` right, `ds` missing its underscore) and has been corrected.

Renaming `_fp` → `fp` is the whole fix; call sites already pass positionally. **If you
add a cached helper, the fingerprint parameter takes no leading underscore and the
dataset parameter does.**

## 11. Unify the two display group filters

`72e15b7`. Both call sites now pass `ui.filters.DISPLAY_GROUPS_KEY`.

**This entry was wrong**, and the correction is worth keeping: it called the change "a
one-line change here". Sharing the key is necessary but not sufficient. Streamlit
garbage-collects widget state for widgets not rendered on the previous run, and a page
switch is exactly that, so the selection was gone before the other page's multiselect was
instantiated. The value is now mirrored under a plain non-widget key and re-seeds
`session_state[key]` before the widget is created; `default=` had to go at the same time,
because passing both makes Streamlit log a warning. Treat effort estimates in this file
as guesses, not findings.

## 1. `sleep_bouts.csv` written by two pages with different contents

`e6c0802`. `export_data.py` keeps `sleep_bouts.csv` (all flies) and now sources it from
`sleep_analysis.raw_bout_dataframe`; `sleep_activity.py` writes `sleep_bouts_filtered.csv`
(current group selection). Both button labels say which is which.

## 2. ZT export column-order and casing drift

`ecbfb52`. Both pages now build the table with `export_helpers.zt_group_summary_table`.

**Release notes:** this changes the header row of the Sleep & activity
`activity_binned.csv` / `sleep_binned.csv` exports, from `mean,n,sd` to `Mean,SD,N`.

## 8. Private function called across a module boundary

`96b0e8a`. `phase_shift._group_labels` → `group_labels`, both call sites, no alias.

## 9. CV fold help text is wrong for any non-default fold count

`4839d45`. Rendered as a caption from the widget's own return value.
