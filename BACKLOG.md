# ClockWork backlog

Open issues, by number. Nothing here blocks anything; each is a real problem
somebody noticed and wrote down rather than fixed on the spot.

Items 1–16 are closed and their entries have been deleted. Their numbers stay
retired: a new issue takes the next free one and never reuses a closed one.

Several comments in `app/` and `core/` still cite those numbers — "backlog item 8
all over again", and so on. The commits are the record now, so
`git log --grep "item 8"` is how you find what one of them meant.

**Line references drift.** Re-check any reference against the current branch
before trusting it. A wrong line number costs more than no line number.

---

## Habits worth keeping

These came out of clearing items 1–16 and are the reason that took as long as it
did.

**Capture a baseline before any fix whose verification is "same as before".**
The SCAMP-cache removal needed one and it paid for itself: 721 of 724 exported
files matched byte-for-byte, so the three that differed could be examined
individually instead of argued about. The recipe is to run the export on `main`,
`sha256sum` the tree, keep a copy, then re-run and `sha256sum -c` after the
change.

**Grep for call sites; do not trust a list in this file.** One closed item's
consumer table was missing two of five. Another's verify step asked for a
difference the code could not produce.

**Check whether `example_data` can even reach the code you changed.** It is real
lab data, but it is one experiment: its monitors are all two digits, no fly has a
gap, and its metadata has no `pulse_time` column. Three separate items had a
"clean" verification run that proved only that the trigger was absent. Build the
triggering case from copies — a metadata copy in a scratch folder over the same
raw monitor files exercises the whole path without touching what ships.

There is a test suite (`tests/`, run by CI on every PR), so a verify step can
usually become a test rather than a one-off check done by hand.
`tests/README.md` covers what the fixtures do and do not imitate.

---

## 17. Four Abhilash renderers the new page deliberately does not use

`app/app_pages/sleep_states.py` wires eight of the twelve renderers in
`core/plotting.py`'s `# Sleep State Analysis Plots` section plus
`sleep_cwt_analysis`, `ultradian_rhythmicity_ls` and
`ultradian_rhythmicity_chi_sq`. Four are still unreferenced outside tests:

| function | why it was left |
|---|---|
| `rebound_bar_plot` | Paper Figure 4. Needs a deprivation experiment with matched undisturbed controls — a different experimental design, and `sleep_deprivation.py` already owns that workflow. |
| `single_fly_scalogram_plotly` + `compute_single_fly_scalogram` | Per-fly scalograms. Useful for QC, but the paper's figures are all group averages and the page had no place for 190 of them. |
| `group_ridge_density_plotly` | Not a figure in this paper. Survived the plotting refactor with a sane signature but no home. |
| `rose_plot` | Superseded in practice by `rose_plot_with_activity`, which draws the paper's five-panel row. Kept because it is the single-panel building block and is what the rose-plot tests assert against. |

**Decision, not a cleanup:** either give `rebound_bar_plot` a home on the
sleep-deprivation page and the single-fly scalogram a QC expander, or delete all
four. Do not leave them for another year — that is how ~1900 lines of unwired
sleep-state code happened in the first place.

Also unused, and ordinary rot rather than a lost feature:
`rhythmicity_violin_grid` and `rhythmicity_long_to_wide_csv`;
`core/load_and_save_datasets.py`'s four flat exporters (`save_data_to_csv`,
`save_data_to_excel`, `export_activity_to_csv`, `export_averaged_data_to_csv`);
and `core/dam_utilities.py`'s `split_activity_dataframe` and `trim_first_dd_day`.

`summary_bars` joined this list when the day/night totals became violins. It is
kept on purpose, so that bars stay available if the distribution view turns out
to be the wrong default.

---

## 18. The figure-save button shares one key across every page

`app/ClockWork.py` calls `charts.save_figures_button()` after every page body,
which takes the default `key="save_page_figures"`. The confirmation is
remembered against that key, so "Saved 6 PNG file(s) to …" follows you to the
next page and sits under figures it does not refer to.

Seen in the app, not inferred. The fix is a per-page key — the router knows which
page it just ran — or clearing the remembered message on a page switch.

---

## 19. The shared "which apparatus did this group sit in" helper has no users

`core/phase_shift.group_extras` exists to answer this question once, so that a
preview and the result it previews cannot disagree. The phase-shift page no
longer calls it: the whole apparatus annotation was removed from that page,
because hardcoding `flybox` into every legend, title and exported group id
assumed one box per arm and put a factor nobody had chosen into every
comparison. If the box matters, it belongs in the grouping, where it gets
compared.

So `group_extras` is now reachable only through `compute_group_phase_difference`'s
`describe_by` parameter, which every caller leaves empty.

This started as "four copies of the same lookup, unify them". One copy is gone
and the shared version is unused, which is not the state to leave it in — either
the remaining copies adopt it or it goes. **Find the remaining copies by grep
rather than trusting this entry**; that is this file's own advice and it applies
here.

---

## 20. The app quotes a validation number no test in the repo produces

`core/phase_shift.py` cites `tests/test_phase_shift.py` for its SCAMP
concordance (2 min median, 12 min worst) in two places, and
`core/calibrations.py` cites it in a third. **That file does not exist on any
branch.**

This is worse than a stale docstring reference, because
`app/app_pages/phase_shift.py` repeats those numbers to the user on screen. A
validation claim the app makes cannot be re-checked from the repo, and it is the
kind of number that ends up in a methods section.

Either restore the test that produced it or remove the claim. Do not simply
re-point the citation at a different file without re-deriving the numbers.

---

## 21. `group_columns` means two different things depending on who wrote it

ARCHITECTURE rule 1 says `attrs['group_columns']` records the metadata COLUMN
names that were ticked. `create_xarray_dataset` does that. `regroup_dataset`
(`core/dam_utilities.py`) writes the COORD names instead, so after "Redefine
groups" by `pulse_time` the attr says `pulse_zt_hour`, a name no metadata file
contains. Only two columns are renamed (`METADATA_COORD_RENAMES`), which is why
it has not bitten.

It is not a one-line fix, because readers assume both meanings:
`app_pages/data_groups.py` defaults its regroup picker by testing
`group_columns` against coord names (so a freshly imported dataset grouped by
`pulse_time` loses that column from the default), while `ui/facet_panels.py`
maps names back and copes with either. `group_coord_names` already records the
coord side, so the fix is for `regroup_dataset` to write column names, with the
readers moved to `get_group_coord_names` where they need coords.

`pipeline.GroupsConfig.from_attrs` reads either spelling and reports column
names, so exported settings are right either way; this item is about the attr.

---

## 22. Changing groups after curation silently undoes curation and the split

`dataset_full`, the Groups page's restore point, is set only at import. Apply
selection, Redefine groups and Reset all rebuild the working dataset from it.
Done after curation, they bring dead flies back (or keep them trimmed by id but
with uncurated data) and drop the curation and split attrs, with no warning.
The page's docstring acknowledges the ordering ("subsetting before curation
... avoids silently invalidating curation afterwards") but the page does not
enforce it or say so when it happens.

The recorded settings stay honest — a reverted curation has no curation attrs,
so Export settings will not claim it ran — but a user who curated, then
narrowed the groups, has lost work they cannot see. Either warn (and name what
will be undone) when the dataset has curation or split attrs, or apply the
group change to the current dataset instead of to `dataset_full`.

**Partly addressed (2026-10-02):** the page now warns, beside both group-change
buttons, naming exactly what would be undone (curation, the split, and which
analyses' results), and says nothing when there is nothing to lose
(`tests/test_data_pages_pipeline.py`). The rebuild itself is unchanged; whether
a group change should instead apply to the current dataset is still open.

---

## 23. Running a period method reorders the master's flies and adds `split_minute`

`pipeline.merge_period_outputs` (formerly `ui.period_context.merge_analysis_outputs`,
moved unchanged) merges an estimator's per-fly outputs onto the master with an
outer join. The estimator returns its flies in a different order, and an outer
join of two differently ordered `id` indexes SORTS the union, so after any
period run the master's flies are in lexical order (`17_1, 17_10, 17_11, ...`)
rather than the order they were imported in. It also copies every per-fly
coord of the result onto the master, which includes the `split_minute` coord
`select_phase` adds, so an unsplit dataset gains a split boundary coord from
running Lomb-Scargle.

Nothing numerical changes — every value is still under its own fly id, which
`tests/test_pipeline_period.py` checks fly by fly — but anything that relies on
fly ORDER (a table exported "in import order", a figure whose rows follow the
dataset) silently changes order after a period run. Either reindex the merge to
the master's `id` order, or decide the sorted order is the canonical one and
apply it at import.
