# ClockWork backlog

Known issues found while mapping the codebase for the sidebar/page reorganization
(branch `page-reorganization`). Each was left alone in that pass because fixing it would
change analysis behaviour or touch `core/`, both of which were explicitly out of scope.
Line references are as of commit `13b2cab`.

Nothing here is a regression introduced by the reorganization — these all predate it.

---

## 1. `sleep_bouts.csv` name collision

Two pages write a file with the same name and different contents:

- `app/app_pages/sleep_activity.py` (was `5_Sleep_and_Activity.py:641`) writes it from
  `sleep_analysis.raw_bout_dataframe(...)` into `<working dir>/Graph Exports/`.
- `app/app_pages/export_data.py` (was `11_Export.py:333`) writes a *different* table under the
  same name, built from `ds[bout_vars].to_dataframe()`, as a browser download.

Whichever the user opens last silently wins. Pick one source of truth, or rename one.

## 2. ZT export column-order drift

`11_Export.py:199-203` deliberately reindexes the group aggregate to `("Mean", "SD", "N")`, with
a comment warning that a plain sort gives the wrong (alphabetical) order.
`5_Sleep_and_Activity.py:481` then does exactly that — `_pivot.sort_index(axis=1, level=0)` with
lowercase `mean`/`sd`/`n` — so two exports of the same quantity disagree on column order and
casing. Downstream GraphPad templates will care.

## 3. HMM phase-source divergence

`7_HMM_Model_Selection.py:42-44` uses the legacy "`dataset_LD` if present, else the full master"
pattern. `6_HMM_Analysis.py:80-86` documents moving *off* that exact pattern, because with no
split applied it silently mixes LD and DD:

> Previously the page silently ran on dataset_LD if present, else the full LD+DD master — so
> with no split it mixed paradigms. Now the phase is explicit [...]

So cross-validation can select a model on one data source while the production fit runs on
another. **Fix this before merging the two HMM pages into one tabbed page** — the merge is
otherwise dishonest, because a single visible phase selector would not govern both tabs.

The two pages also configure the same three knobs (`n_states`, `emission_model`,
`transition_constraints`) under unlinked widget keys (`cv_states_min`/`cv_states_max`,
`cv_emissions`, `cv_trans` vs `hmm_n_states`, `hmm_emission`, `hmm_trans`), so the winning
configuration has to be retyped by hand. A "use these settings" hand-off would fix that.

## 4. Two parallel phase-metadata schemes

Canonical: `ds.attrs['phase']` plus `ds.attrs['split_applied']`, owned by `core/dataset_meta.py`.
Legacy: `ds.attrs['split_phase']` in `{LD, DD, both}`.

`2_Preprocessing.py:227` still writes the legacy alias and reads it back at `:135` and `:320`,
while pages 3 and 5 read the canonical API. Retire the legacy alias.

## 5. `dataset_LD` / `dataset_DD` session caches vs `select_phase`

Two ways to get a phase view coexist: the pre-sliced `st.session_state.dataset_LD` /
`dataset_DD`, and on-the-fly `dam_utilities.select_phase`. Pages 2, 3 and 5 each handle both,
with long comments explaining why.

Worst instance: `5_Sleep_and_Activity.py:323-346` re-derives the split by scraping the split
parameters back out of `ds.attrs` and re-calling `split_xarray_dataset` — duplicating page 2's
logic. The code labels itself "TRANSITIONAL (retires with the dataset_LD/DD sweep)". Finish
that sweep.

## 6. ~1900 lines of a complete, unwired feature

13 of the 26 public functions in `core/plotting.py` have no caller anywhere. Eleven of them sit
inside one section — `# Sleep State Analysis Plots (Abhilash et al. 2026)`,
`core/plotting.py:886-2450`:

`normalized_waveform_overlay` (:908), `initiation_probability_plot` (:1005),
`rebound_bar_plot` (:1111), `period_amplitude_plot` (:1224), `sleep_state_scalogram` (:1574),
`single_fly_scalogram_plotly` (:1638), `group_ridge_density_plotly` (:1883), `rose_plot` (:1989),
`rose_plot_with_activity` (:2087), `polar_gating_plot` (:2248), `ultradian_amplitude_plot` (:2351)

They pair exactly with two orphaned analysis functions in `core/periodograms.py`:
`sleep_cwt_analysis` (:3125) and `ultradian_rhythmicity_ls` (:3395), plus
`compute_single_fly_scalogram` (:2142).

**This is a decision, not a cleanup.** It is a coherent, fully-built Abhilash sleep-state
analysis that never got a UI. Either build the page it was written for, or delete it. Leaving
~1900 lines of dead scientific code in the tree is the worst of the three options.

Also unused, but ordinary rot rather than a lost feature: `rhythmicity_violin_grid` (:2689) and
`rhythmicity_long_to_wide_csv` (:3392); `core/load_and_save_datasets.py`'s four flat exporters
(`save_data_to_csv`, `save_data_to_excel`, `export_activity_to_csv`,
`export_averaged_data_to_csv`); `core/dam_utilities.py`'s `split_activity_dataframe` (:157) and
`trim_first_dd_day` (:233).

## 7. Inverted dependencies in `core/plotting.py`

`plotting.py` defer-imports analysis code specifically to dodge circular imports:

- `:542` — `sleep_analysis.bout_duration_summary`, `per_fly_bout_duration_curves`,
  `bout_duration_group_stats`, inside `sleep_bout_duration_lines`
- `:1939` — `rhythmicity_classification.apply_rhythmic_filter`

Those three `sleep_analysis` functions are called from *nowhere else*, so the compute/render seam
is in the wrong place: the plotting module is running the analysis. Move the computation to the
caller and let the plot function take a dataframe.

Related: `periodograms.wavelet_analysis:2058` reaches into plotting to write PNGs to disk — an
analysis function with a filesystem side effect.

## 8. Private function called across a module boundary

`9_Phase_Shift.py:182` calls `phase_shift._group_labels(...)`. Promote it, or add a public
wrapper.

## 9. Wrong help text on the CV fold slider

`7_HMM_Model_Selection.py:56-63` — `max_value` is `min(10, n_flies)` but the help text computes
the hold-out size with a hard-coded `min(5, n_flies)`, so the hint is wrong for any fold count
other than the default.

## 10. `core/dam_integrity.py` has no UI surface

492 lines of load-time data-quality analysis — status resolution, duplicate handling, gap
scanning, and a cosmetic-vs-data-loss classification of irregular grid slots. It runs on every
raw import (via `dam_processor.py:385,396,405,420`) and `format_monitor_report` (:412) renders a
report that **is never displayed in the app**. Today it only reaches the terminal.

A "Data quality" panel on the Import page would surface it. Note the constraint: the report is
only produced on a raw load, so it is unavailable after a `.nc` reload unless it is persisted
into the dataset attrs.

## 11. Two display group filters that do not track each other

`app/ui/filters.py` shares the *implementation*, but deliberately keeps separate session keys per
page (`pgram_groups` on Periodograms, `viz_groups` on Sleep & activity) to preserve existing
behaviour. Selecting groups on one page therefore has no effect on the other, which surprises
people.

Unifying them to one app-wide key is a one-line change in `ui/filters.py` once you decide you
want it. Note this is a *third* distinct thing called a group filter — the one on
Groups & subsets destructively subsets the master dataset, while these two only subset a
page-local view for plotting.

## 12. Groups cannot be redefined after import

`dam_utilities.regroup_dataset` (:450) exists and is called by nothing. The `group` coordinate is
only ever set at import, inside Create Dataset (`data_import.py` → `derive_group_labels` →
`create_xarray_dataset:610`).

`README.md` claims the opposite — that group columns "can be re-grouped later without
re-importing". Either wire `regroup_dataset` into the Groups & subsets page, or correct the
README.

If wiring it up: regrouping must clear every downstream cache, because `group` feeds every
group-level comparison, plot and export. `ui/state.invalidate_derived_caches()` already does
exactly that.
