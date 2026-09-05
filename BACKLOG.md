# ClockWork backlog

Thirteen issues found while mapping the codebase for the sidebar/page reorganization
(branch `page-reorganization`). None is a regression introduced by that work — they all
predate it. They were left alone there because fixing them changes analysis behaviour or
touches `core/`, both of which were out of scope for a re-cut of `app/`.

They have since been triaged into three sections:

- **[A. Ready to fix](#a-ready-to-fix)** — the decision is made and written down. Pick one
  up and implement it as described; no further judgement needed.
- **[B. Held](#b-held)** — a real decision that has not been made yet.
- **[C. Audit](#c-audit)** — no code change; something to go and check.

**Item numbers are original and stable.** They are referenced from code
(`app/app_pages/data_groups.py:15-17`) and from the reorganization PR, so they are not
renumbered when items move between sections or get closed.

**Line references are pinned to `d0a477c`** (post-reorganization). The pre-reorganization
references in the original version of this file pointed at `app/pages/3_Period_Analysis.py`
and friends, which no longer exist.

---

## Suggested order

Dependencies are light but real: **3 before 5** (item 3 removes the last analysis-side
reader of the LD/DD caches, which item 5 then deletes), and **4 before 5** (both edit
`data_curate_split.py`). Items 1 and 2 both touch `export_data.py` and `sleep_activity.py`,
so they are easiest done together.

| Order | Items | Why here |
|---|---|---|
| 1 | **9, 8** | Isolated one-liners. Good first commits. |
| 2 | **3** | Unblocks 5. Also gates the eventual HMM page merge. |
| 3 | **4**, then **5** | Both edit `data_curate_split.py`; 5 is the largest item in A. |
| 4 | **2, 1** | Both edit the same two export paths. |
| 5 | **11, 12, 10** | Independent, small to medium. |
| 6 | **7** | Largest refactor; touches the compute/render seam. Do it last. |

---

# A. Ready to fix

## 9. CV fold help text is wrong for any non-default fold count

`app/app_pages/hmm_model_selection.py:44-49`. The slider's `max_value` is
`min(10, n_flies)` but the help string hard-codes the hold-out size as
`n_flies // min(5, n_flies)` — the *default* fold count, not the selected one. Move the
slider to 7 folds and the hint still describes 5.

**Fix.** Render the caption from the widget's own value after the widget, rather than
computing it inside `help=`:

```python
n_folds = st.slider("Number of folds", min_value=2, max_value=min(10, n_flies),
                    value=min(5, n_flies), key="cv_n_folds")
st.caption(f"Dataset has {n_flies} flies. Each fold holds out ~{n_flies // n_folds} flies.")
```

**Verify.** Change the fold count and confirm the hold-out figure tracks it.

## 8. Private function called across a module boundary

`app/app_pages/phase_shift.py:169` calls `ps_module._group_labels(...)`, defined at
`core/phase_shift.py:739` and used internally at `:826`.

**Fix.** Promote it to `group_labels` and update both call sites, leaving no alias — one
internal caller and one page caller is a small enough blast radius that a deprecation
shim is not worth it.

**Verify.** Load a dataset with a `pulse_time` column and open Phase shift; the group
breakdown still renders.

## 3. HMM model selection and analysis run on different data

`app/app_pages/hmm_model_selection.py:28-31` still uses the legacy pattern — `dataset_LD`
if present, else the full LD+DD master — with no phase control at all.
`app/app_pages/hmm_analysis.py:70-86` documents moving *off* that exact pattern, because
with no split applied it silently mixes paradigms, and offers an explicit radio
(LD / DD / Both together / Both separate, default LD) sourced from
`select_phase(master, …)` at `:227-241`.

So cross-validation can select a model on one data source while the production fit runs on
another. This also **blocks merging the two HMM pages into one tabbed page** — a single
visible phase selector would not honestly govern both tabs.

The two pages also configure the same three knobs under unlinked widget keys
(`cv_states_min`/`cv_states_max`, `cv_emissions`, `cv_trans` vs `hmm_n_states`,
`hmm_emission`, `hmm_trans`), so the winning configuration has to be retyped by hand.
(`hmm_analysis.py:124-139` looks like a hand-off but is not — it syncs the *preset* radio
to the advanced widgets.)

**Decision.** Both pages default to **LD**, with **DD** available on both. Model selection
gets an explicit picker rather than an implicit cache read, plus a message telling the user
to select the same phase on both pages.

**Fix.**

1. Replace `hmm_model_selection.py:28-31` with the same `_has_transition` guard and phase
   radio that `hmm_analysis.py:70-86` uses, restricted to `["LD", "DD"]`, `index=0`.
2. Source the data with `dam_utilities.select_phase(master, phase)` instead of reading
   `st.session_state.dataset_LD`.
3. When there is no LD/DD transition in the dataset, fall back to the full recording and
   say so — mirror the `else` branch at `hmm_analysis.py:84-86`.
4. Add a persistent note on model selection: the phase chosen here should match the phase
   chosen on HMM Analysis, or the selected model will not describe the data being fitted.
   Analysis also offers *Both (together)* and *Both (separate)*, which CV does not — the
   note should say to match the LD or DD case.

**Do not** unify the widget keys as part of this item; that is a separate hand-off feature.

**Verify.** On a dataset with an LD/DD transition, run CV on LD and on DD and confirm the
fly counts differ as expected; confirm a dataset with no transition still runs.

## 4. Retire the legacy `split_phase` attr

Two phase-metadata schemes coexist. Canonical: `ds.attrs['phase']` +
`ds.attrs['split_applied']`, owned by `core/dataset_meta.py`. Legacy:
`ds.attrs['split_phase']` in `{LD, DD, both}`.

Writes (to remove): `app/app_pages/data_curate_split.py:212`, `core/dam_utilities.py:1598`.

Reads (to migrate to the canonical API): `data_curate_split.py:120, 305-306, 341`.

**Fix.** Delete the two writes; convert the three `data_curate_split.py` reads to
`dataset_meta.dataset_phase()` / `is_split_applied()`.

**Keep** the read-side migration at `app/app_pages/data_import.py:379-389` and the legacy
handling in `core/dataset_meta.py:64, 88, 108`. Those exist to load `.nc` files written
*before* this change and must stay — this item stops **writing** the alias, it does not
stop reading it. Say so in a comment at each surviving site so a later sweep does not
remove them too.

**Verify.** Save a `.nc` after a split, reload it, confirm the phase is reported
correctly; then reload an older `.nc` that still carries `split_phase` and confirm the
migration path still resolves it.

## 5. Delete the `dataset_LD` / `dataset_DD` session caches

Two ways to get a phase view coexist: the pre-sliced `st.session_state.dataset_LD` /
`dataset_DD`, and on-the-fly `dam_utilities.select_phase`.

The caches are **physical slices** from `split_xarray_dataset`
(`app/app_pages/data_curate_split.py:190-204`). `select_phase` cannot replace them for
every consumer: it returns a NaN-masked view over the *full* time axis
(`core/dam_utilities.py:1210-1216`), while `export_scamp` needs real equal-length files per
board.

Consumers today:

| Site | Purpose |
|---|---|
| `app/app_pages/export_data.py:42, 68-79` | per-phase `.nc` save |
| `app/app_pages/export_scamp.py:24-25` | the legacy luc files |
| `app/app_pages/data_curate_split.py:126-130` | its own post-split UI state |

Item 3 removes the last analysis-side reader. What remains is the maintenance cost:
`app/app_pages/sleep_detection.py:174-195`, labelled `TRANSITIONAL`, must **regenerate both
caches every time sleep is computed**, because they are a copy that otherwise goes stale.

**Decision.** Delete the caches. They are a staleness-prone copy of something derivable;
the two export pages can slice on demand when the user clicks export.

**Fix.**

1. `export_scamp.py` and `export_data.py`: call
   `dam_utilities.split_xarray_dataset(master, phase=…, gap_threshold_minutes=…,
   discard_first_dd_day=…)` at export time, reading the split parameters from
   `master.attrs` exactly as `sleep_detection.py:180, 187` does today.
2. Delete the regeneration block at `sleep_detection.py:174-195` — including the int8
   restoration that only exists because slicing upcasts the sleep masks.
3. Delete the writes at `data_curate_split.py:203-204` and the clears at
   `data_groups.py:46-47, 58-59`. Replace the reads at `data_curate_split.py:126-130` with
   the canonical `dataset_meta` phase check.
4. Update the docstring at `export_scamp.py:5-9`, which currently tells the user the
   caches are a prerequisite.

**Cost.** A few seconds of slicing per export, in exchange for removing a copy that four
files have to keep in sync.

**Verify.** Full path: import → curate → split → sleep analysis → SCAMP export produces the
same file count and contents as before (189 LD files across 6 boards on `example_data`),
and the per-phase `.nc` save still writes both files.

## 2. ZT export column-order and casing drift

Two exports of the same quantity disagree on column order *and* casing, which downstream
GraphPad templates will care about.

- `app/app_pages/export_data.py:187, 193` deliberately builds
  `values=["Mean", "SD", "N"]` and reindexes to
  `[(g, s) for g in sorted(groups) for s in ("Mean", "SD", "N")]`, with a comment warning
  that a plain sort gives the wrong order.
- `app/app_pages/sleep_activity.py:212, 217` and `:291, 296` do exactly what that comment
  warns against — lowercase `values=["mean", "sd", "n"]` followed by
  `sort_index(axis=1, level=0)`, which orders the stats alphabetically as
  **mean, n, sd**.

**Decision.** `export_data.py` is correct; make `sleep_activity.py` match it.

**Fix.** In both `sleep_activity.py` blocks, rename the aggregation outputs to
`Mean`/`SD`/`N` and replace the `sort_index` call with the same explicit reindex
`export_data.py:193` uses. Factor the ordering into one helper both pages import — the
whole point is that they cannot drift again. `app/export_helpers.py` is the natural home.

**Note.** This changes the header row of an existing export. Worth a line in the release
notes.

**Verify.** Export the same ZT table from both pages and diff the headers.

## 1. `sleep_bouts.csv` written by two pages with different contents

Two pages write a file with the same name and different contents; whichever the user
opens last silently wins.

| | `app/app_pages/sleep_activity.py:373-381` | `app/app_pages/export_data.py:325-343` |
|---|---|---|
| Source | `sleep_analysis.raw_bout_dataframe(ds, …)` | `ds[bout_vars].to_dataframe()` |
| Columns | id, sleep_bout_number, **group**, duration, **sleep_state**, start/end_time | id, bout index, duration, start/end_time |
| Flies | **filtered** by the page's group selection | all flies |
| Destination | working folder | browser download |

The export version is a strict subset — the same rows, minus `group` and `sleep_state`.

**Decision.** One source of truth, and disambiguate the two files by name.

**Fix.**

1. `export_data.py`: replace the ad-hoc `ds[bout_vars].to_dataframe()` with
   `sleep_analysis.raw_bout_dataframe(ds)` (no group filter). The export gains `group` and
   `sleep_state`, and `raw_bout_dataframe` becomes the single source of truth its own
   docstring already claims it is (`core/sleep_analysis.py:~470`).
2. Rename so the two are distinguishable: `export_data.py` keeps `sleep_bouts.csv` (all
   flies); `sleep_activity.py` writes `sleep_bouts_filtered.csv`.
3. Say which is which in the button labels — "all flies" vs "current group selection".

**Verify.** With a group filter active, confirm the two files differ only in row count and
that the unfiltered one matches the fly total.

## 11. Unify the two display group filters

`app/ui/filters.py` shares the implementation but deliberately keeps separate session keys
per page: `pgram_groups` (`app/app_pages/periodograms.py:27`) and `viz_groups`
(`app/app_pages/sleep_activity.py:144`). Selecting groups on one page therefore has no
effect on the other, which surprises people.

Both are *display* filters over a page-local view. The filter on Groups & subsets is a
different thing entirely — it replaces the master dataset and invalidates every downstream
cache.

**Decision.** Share one app-wide key.

**Fix.** Pass the same key from both call sites and update the module docstring at
`app/ui/filters.py:9-13`, which currently explains why they are separate. Note the two
calls differ in one respect — `sleep_activity.py:144` passes `subset=True`, so it actually
`.sel()`s its local view while Periodograms filters by hand. That asymmetry is fine and
should stay; only the key is shared.

**Verify.** Select a subset on Periodograms, switch to Sleep & activity, and confirm the
same subset is selected and the plots honour it.

## 12. Wire up `regroup_dataset` so groups can be redefined after import

`core/dam_utilities.py:452-468` is complete and correct — it re-derives the `group` coord
from the per-id metadata coords stored at import and updates `attrs['group_columns']`.
Nothing calls it. The `group` coord is only ever set at import, inside Create Dataset
(`data_import.py` → `derive_group_labels` → `create_xarray_dataset`).

`README.md:189` claims the opposite: that group columns "can be re-grouped later without
re-importing".

**Decision.** Wire it up and make the README true. The capability matters most for a
reloaded `.nc`, where the alternative is re-reading raw DAM files.

**Fix.**

1. On Groups & subsets, add a multiselect of the available group-defining coords —
   the candidates are the per-id coords that `create_xarray_dataset` stored, filtered by
   `dam_utilities.GROUP_EXCLUDE_COLUMNS`.
2. On apply, call `regroup_dataset(ds, chosen)` and write the result back to
   `st.session_state.dataset` **and** `dataset_full`, so the subset filter's restore point
   is regrouped too.
3. Call `ui.state.invalidate_derived_caches()` (`app/ui/state.py:96`) — `group` feeds every
   group-level comparison, plot and export, so every cached result is invalid after a
   regroup.
4. Delete the "cannot yet be redefined here" note at `app/app_pages/data_groups.py:15-17`.

**Verify.** Load a `.nc` grouped by genotype+temperature, regroup by genotype alone,
confirm the group count drops and that a previously computed period analysis is cleared
rather than shown against the new labels.

## 10. Surface the rest of the data-integrity report

`core/dam_integrity.py` is 492 lines of load-time data-quality analysis — status
resolution, duplicate handling, gap scanning, and a cosmetic-vs-data-loss classification.

Partly resolved: the **aggregate** summary is now rendered at
`app/app_pages/data_import.py:173` via `MetadataProcessor.integrity_summary_lines`.

Still missing:

1. The **per-monitor** detail from `dam_integrity.format_monitor_report`, which
   `core/dam_processor.py:652-656` prints to the console and nowhere else.
2. Anything at all after a `.nc` reload — the report only exists during a raw import.

**Decision.** Surface the per-monitor detail at import, and persist scalar counters into
`attrs` so a reloaded dataset can still report its own quality.

**Fix.**

1. Add a per-monitor expander to the import page, alongside the import report added in
   `f942522`. `MetadataProcessor.integrity_report` already holds the per-monitor dicts
   (`status`, `scan`, `classification`) keyed by monitor id.
2. Aggregate scalars into `attrs` when the dataset is built: `integrity_n_status_bad`,
   `integrity_n_cosmetic_slots`, `integrity_n_dataloss_slots`. Follow the existing
   precedent at `core/dam_utilities.py:591-597` (`time_regularized`, `gaps_filled`,
   `gap_fill_value`) — plain scalars, which NetCDF serializes cleanly.
3. Show those counters wherever a reloaded dataset is summarized.

**Do not** serialize the full per-monitor structure as a JSON string in `attrs`. Nested
blobs in NetCDF attributes are fragile and the spans are only actionable at import time,
when the raw files are still in hand.

**Verify.** Import a monitor with a known gap, confirm the per-monitor detail appears;
save, reload, and confirm the counters survive.

## 7. Inverted dependencies in `core/plotting.py`

`plotting.py` defer-imports analysis code specifically to dodge circular imports:

- `core/plotting.py:542` — `sleep_analysis.bout_duration_summary`,
  `per_fly_bout_duration_curves`, `bout_duration_group_stats`, inside
  `sleep_bout_duration_lines`
- `core/plotting.py:1939` — `rhythmicity_classification.apply_rhythmic_filter`

Those three `sleep_analysis` functions are called from *nowhere else*, so the
compute/render seam is in the wrong place: the plotting module is running the analysis.

Related: `core/periodograms.py:2058` reaches into `plotting` to write PNGs to disk — an
analysis function with a filesystem side effect.

**Decision.** Move the computation to the caller and let the plot function take a
dataframe; invert `periodograms.py:2058` so the caller writes the file.

**Fix.**

1. Change `sleep_bout_duration_lines` to accept the already-computed frames, and compute
   them in `app/app_pages/sleep_activity.py` before the call.
2. Same shape for the rhythmic filter at `:1939` — the caller applies the filter and
   passes the filtered dataset in.
3. Have `periodograms.wavelet_analysis` return the figure (or the array behind it) and let
   its caller write the PNG, rather than importing `plotting` at `:2058`.
4. Delete the deferred imports. If a circular import remains after the move, the seam is
   still in the wrong place — that is the test.

**Sequenced last:** the largest refactor in this section, and it touches the most call
sites.

**Verify.** Bout-duration curves and the rhythmic-filter path render identically on
`example_data`, and `core/plotting.py` imports cleanly at module scope with no deferred
analysis imports left.

---

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
`core/dam_utilities.py:1380`, so some wiring was anticipated.

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
