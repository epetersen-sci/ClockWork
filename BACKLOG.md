# ClockWork backlog

Thirteen issues found while mapping the codebase for the sidebar/page reorganization
(branch `page-reorganization`), plus one (item 14) found later while closing them. None is
a regression introduced by the reorganization — they all predate it. They were left alone
there because fixing them changes analysis behaviour or touches `core/`, both of which were
out of scope for a re-cut of `app/`.

They were triaged into these sections:

- **[A. Ready to fix](#a-ready-to-fix)** — the decision is made and written down. Pick one
  up and implement it as described; no further judgement needed.
- **[B. Held](#b-held)** — a real decision that has not been made yet.
- **[D. Done](#d-done)** — closed, with the commit that closed it.

(There was a section C, "Audit" — no code change, something to go and check. Its
one item is closed, so the section is gone. The letter is not reused, for the
same reason item numbers are not.)

**Every issue from the original triage is closed.** Thirteen of them, plus
four found while fixing them (14, 15, 16, 17). Item 6 — the held decision on
~1900 lines of unwired Abhilash sleep-state code — was settled in favour of
building the page; item 17 is the part of it that deliberately stayed unwired.

**Item numbers are original and stable.** They are referenced from the
reorganization PR and from the commits that closed them, so they are not renumbered
when items move between sections or get closed. A new issue takes the next free
number (14 and 15 were added this way), and never reuses a closed one.

(`app/app_pages/data_groups.py` used to carry a pointer to item 12 in its module
docstring. Item 12 is closed and that pointer is gone, so no code references this
file any more.)

**Line references** drift. Re-check any reference against `main` before
trusting it — several moved over the course of closing section A, and a wrong
line number costs more than no line number.

---

## What this file is for now

Section A is empty. Three habits from clearing it are worth keeping:

**Capture a baseline before any item whose Verify step says "same as before".**
Item 5 needed one and it paid for itself — 721 of 724 SCAMP files matched
byte-for-byte, so the three that differed could be examined individually instead
of argued about. The recipe: run the export on `main`, `sha256sum` the tree, keep
a copy, then re-run and `sha256sum -c` after the change.

**Grep for call sites; do not trust a list in this file.** Item 5's consumer
table was missing two of five, and item 3's Verify step asked for a difference
the code could not produce. Both are recorded in their Done entries.

**Check whether `example_data` can even reach the code you changed.** It is real
lab data, but it is one experiment: its monitors are all two digits (so item 13's
trigger cannot occur) and no fly has a gap (so item 16's trim never fires). Twice
a "clean" verification run proved only that the trigger was absent. Build the
triggering case from copies — items 13 and 16 both do.

There is now a test suite (`tests/`, run by CI on every PR), so a Verify step can
usually become a test rather than a one-off check done by hand. `tests/README.md`
covers what the fixtures do and do not imitate.

---

# A. Ready to fix

One item: **17**, the remainder of item 6. Everything else triaged here has
been closed — see [D. Done](#d-done).

New issues go here as they are found, with the same shape the closed ones had:
a Decision, a numbered Fix, and a Verify step that says what to check rather
than "it works".


## 17. Four Abhilash renderers the new page deliberately does not use

Item 6 built `app/app_pages/sleep_states.py`, which wires eight of the twelve
renderers in `core/plotting.py`'s `# Sleep State Analysis Plots` section plus
`sleep_cwt_analysis`, `ultradian_rhythmicity_ls` and the new
`ultradian_rhythmicity_chi_sq`. Four are still unreferenced outside tests:

| function | why it was left | 
|---|---|
| `rebound_bar_plot` | Paper Figure 4. Needs a deprivation experiment with matched undisturbed controls — a different experimental design, and `sleep_deprivation.py` already owns that workflow. |
| `single_fly_scalogram_plotly` + `compute_single_fly_scalogram` | Per-fly scalograms. Useful for QC, but the paper's figures are all group averages and the page had no place for 190 of them. |
| `group_ridge_density_plotly` | Not a figure in this paper. Survived item 7's refactor with a sane signature but no home. |
| `rose_plot` | Superseded in practice by `rose_plot_with_activity`, which draws the paper's five-panel row. Kept because it is the single-panel building block and is what the rose-plot tests assert against. |

**Decision, not a cleanup**, and a smaller one than item 6: either give
`rebound_bar_plot` a home on the sleep-deprivation page and the single-fly
scalogram a QC expander, or delete all four. Do not leave them for another
year — that is how item 6 happened.

Also still unused, and ordinary rot rather than a lost feature:
`rhythmicity_violin_grid` and `rhythmicity_long_to_wide_csv`;
`core/load_and_save_datasets.py`'s four flat exporters (`save_data_to_csv`,
`save_data_to_excel`, `export_activity_to_csv`, `export_averaged_data_to_csv`);
and `core/dam_utilities.py`'s `split_activity_dataframe` and
`trim_first_dd_day`.



# D. Done

Closed items, newest first. The full description of each lives in the commit that closed
it — `git show <sha>` — rather than being kept here, so this section stays an index.

## 6. ~1900 lines of a complete, unwired feature

**Settled in favour of building.** The page is `app/app_pages/sleep_states.py`,
five tabs against the paper's Figures 1, 2, 3, 5 and 6. Figure 4 (rebound) is
out of scope by design — see item 17 for the four renderers left unwired.

The diagnosis was not "it needed a UI". Three separate things were wrong, one
per complaint:

**1. There was no data layer at all.** Four docstrings in `plotting.py` named
`sleep_state_metrics.compute_normalized_waveforms` and friends as their input.
`core/sleep_state_metrics.py` did not exist, and nothing anywhere built
`bout_times_df`, `activity_zt_counts`, `waveform_df`, `init_df` or `rebound_df`
— those names appeared *only* inside `plotting.py`. The eleven renderers were
unreachable by construction, not merely uncalled. That module is now written.

**2. The rose plots showed the wrong quantity.** `rose_plot` binned sleep-bout
INITIATION times. The paper's rose plots are profiles — "We calculated sleep
time series, binned at 30-minute intervals... averaged over days and across
flies" — and the authors' own `phase::rosePlotsSleep` is handed the binned
series and never sees a bout table. Bout initiation is a different figure
(Figure 2). Also fixed: the palette was Plotly's default cycle, which gave
short sleep the paper's long-sleep blue and long sleep its activity red; no
day/night shading; and `polar_gating_plot` derived gates from bout
onset/offset rather than angular deviation about the centre of mass, drew
every fly's arc at r = 1 instead of as concentric rings, and averaged phases
arithmetically so any gate straddling midnight — long sleep under DD, the
state the figure is about — landed on the far side of the clock.

**3. The CWT could not be checked because nothing ever checked it.** Fixed by
adding ground truth (`tests/test_sleep_cwt_truth.py`) and by correcting two
defaults against `WaveletComp::analyze.wavelet`, which is what the paper used:

- **Scale rectification was missing.** WaveletComp computes
  `Power = Mod(Wave)^2 / scale` (Liu et al. 2007); ours returned a bare
  `|W|^2`. On a synthetic 24 h + 3 h signal of EQUAL amplitude, unrectified
  reports the 24-h component **7.1x** stronger and puts its peak at
  **24.51 h**; rectified reports them comparably and peaks at **24.00 h**. So
  ultradian power was understated by about an order of magnitude relative to
  circadian — the entire subject of Figures 5 and 6.
- **`full_range` defaulted to `None`**, which ran two separate narrow-band
  CWTs (18-30 h and 1-4/2-6 h) and normalised each by its own band mean. Those
  z-values are comparable neither to each other nor to the paper's fixed
  0-1.5 colour scale. It now defaults to the paper's `(1, 32)` h single pass.

Two further bugs surfaced while wiring it, neither in this file:

- A `select_phase` view upcasts the int8 masks to float with NaN out of phase.
  `sleep_cwt_analysis` guarded only `< 0`, so **every cell of the averaged
  surface came back NaN**; and it was the one analysis in `periodograms.py`
  that skipped `_extract_longest_continuous_block`, so the out-of-phase epoch
  would have been carried in as a block of zeros.
- `period_amplitude_plot` let `add_vline` attach its own `12-h`/`24-h` labels.
  Shapes on a log axis take data units but annotations take log10 and are not
  converted, so an annotation sat at 10^24 and autoranged the axis out 24
  decades, flattening every curve against the left edge.

Also replaced: `ultradian_rhythmicity_ls` is Lomb-Scargle, but the paper uses a
chi-squared periodogram, so its numbers could not be checked against Figure 6
or Table S2. `ultradian_rhythmicity_chi_sq` is a port of
`zeitgebr::chi_sq_periodogram` (fractional cycle count, dof = period in
samples, Sidak-corrected threshold, and the **adjusted** power the paper plots
— which is why its critical line is a flat zero). Both are offered on the page.

**Verified on real lab data** (189 flies, DD), the page reproduces the paper's
findings rather than merely running: activity peaks at CT12 and P(long) spikes
in the hour immediately after while P(short) peaks *with* the activity peak and
P(inter.) is high pre-dawn (Figure 2's three claims); short sleep's circadian
peak is ~40% of standard's and long's, with the ultradian shoulder ordered
short > standard > long (Figure 5B); and the chi-squared test gives 11/12 flies
rhythmic at 23.8-24.0 h against Table S2's 93-100% at 23.4-23.8 h.

## 16. The gap trim masked `(id, time)` variables on the wrong axis

`_select_longest_segments` now builds one boolean `keep` over `("id", "time")`
and lets xarray broadcast it, so dimension order is irrelevant. Each dtype keeps
its own missing marker (`-1` for the int8 sleep masks), which removed the
`astype(float)` upcast and let `export_helpers.phase_slice` drop its int8 repair
block entirely.

Measured on a dataset with one 180-minute gap (31 flies trimmed):

| | trimmed fly | untrimmed fly |
|---|---|---|
| before | sleep 211 missing vs moving 1080 | sleep 31 missing |
| after | sleep 1080 vs moving 1080 | sleep 0 |

So ~869 minutes per trimmed fly that the trim exists to discard were kept, and
every fly picked up one spurious missing marker per trimmed fly.

**Correcting the scope this entry originally claimed:** it reached the per-phase
`.nc`, NOT the SCAMP export. SCAMP writes `activity`, which had its own
correctly-shaped masking block and was never affected. Verified: the SCAMP
output is byte-identical before and after on both a clean dataset (724/724
files) and the gap dataset (722/722).

## 13. Datasets built before the fly/metadata mislabeling fix

**Closed as not applicable — there is nothing to audit.** The code bug is fixed
in `d0a477c`; what this item asked was which SAVED datasets and published
figures came from the buggy version. The project is still in beta: no `.nc`
files and no figures were produced by the old code, so the population the audit
was over is empty.

What the item was right about is that nothing exercised the fix. `example_data`
is real lab data, but its monitors are 17–22 — all two digits — so string and
numeric order agree and the trigger (mixed digit counts listed in numeric order)
cannot occur. `tests/test_monitor_label_pairing.py` now renames COPIES of those
files to monitors 2 and 10 and asserts each fly keeps the genotype and the
activity trace of the monitor its data actually came from. 192/192 flies pass on
the full dataset; the committed test uses two monitors to stay fast.

The committed `example_data` is deliberately unchanged: its monitor numbers and
the genotype attached to each are real provenance, not test scaffolding.

**If this ever stops being true** — if a `.nc` predating `d0a477c` turns up —
the trigger test is: does the metadata file list its `Monitor` values in an
order that differs from their **string** sort? Same digit count everywhere is
safe. Re-importing from the raw DAM files is the only repair.

## 7. Inverted dependencies in `core/plotting.py`

`d998b53`. `sleep_bout_duration_lines` takes the computed frames;
`group_ridge_density_plotly` takes an already-filtered dataset;
`wavelet_analysis` returns the group averages instead of writing PNGs, which
`export_helpers.save_group_average_scalograms` now does. No deferred analysis
imports remain in `plotting.py`, and it imports first in a clean interpreter.

Note for whoever settles item 17: the rhythmic-filter inversion was inside
`group_ridge_density_plotly`, which item 6's page did not find a home for and
which is still uncalled. If it is deleted, this part goes with it.

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
