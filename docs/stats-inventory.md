# Figure statistics inventory

The automatic statistics added to ClockWork figures: which figures carry them,
which tests run, and how results are shown and exported. Written against `main`
at `97a594c` (2026-10-26). Nothing here is implemented yet unless marked
**existing**.

## Tests

The test for each kind of measure is fixed in advance; it does not depend on the
data.

| Family | 3+ groups | 2 groups | Used for |
|---|---|---|---|
| **W** — Welch | Welch ANOVA, then Games-Howell (all pairs) | Welch's t-test | Activity, period, rhythm strength, phase shift, HMM state occupancy, sleep deprivation |
| **R** — Rank-based | Kruskal-Wallis, then Dunn's test with Holm correction (all pairs) | Mann-Whitney U | Sleep totals, sleep bout duration, sleep-state totals |

All post-hoc tests compare every pair of groups. The phase response curve is the
one exception: it compares each pulsed group with its unpulsed control, which is
defined by `pulse_duration_min = 0` (`phase_shift.control_map_from_pulse`).

### Libraries

| Step | Function | Package |
|---|---|---|
| Welch ANOVA | `statsmodels.stats.oneway.anova_oneway(..., use_var="unequal")` | statsmodels |
| Games-Howell | `scikit_posthocs.posthoc_games_howell` | scikit-posthocs |
| Kruskal-Wallis | `scipy.stats.kruskal` | scipy |
| Dunn | `scikit_posthocs.posthoc_dunn(..., p_adjust="holm")` | scikit-posthocs |
| Welch's t / Mann-Whitney U | `scipy.stats.ttest_ind(equal_var=False)` / `scipy.stats.mannwhitneyu` | scipy |
| Paired t / Wilcoxon signed-rank | `scipy.stats.ttest_rel` / `scipy.stats.wilcoxon` | scipy |
| Fisher's exact / χ² | `scipy.stats.fisher_exact` / `scipy.stats.chi2_contingency` | scipy |
| Compact letter display | `scikit_posthocs.compact_letter_display` | scikit-posthocs |

scikit-posthocs is a new dependency, `scikit-posthocs>=0.12`. A regression test
checks the wrapper's output against reference values computed in R.

### Setting

A **Group statistics** control in the sidebar, with a matching key in the CLI
YAML config:

- **Recommended per measure** (default): the families in the table above.
- **Welch ANOVA + Games-Howell for everything.**
- **Kruskal-Wallis + Dunn for everything.**

### On the figure

1. **Notice line.** One line directly under every tested figure (and under the
   rhythmic-fraction table), naming the test and asking the user to check it,
   for example:

   > *Kruskal-Wallis + Dunn (Holm), one value per fly — check this test suits your data before relying on the result.*

2. **Significance marks.** Group count is per panel when facet panels are on.

   | Groups | Marks |
   |---|---|
   | 2 or 3 | A bracket on every pair. Significant pairs: `*` p < 0.05, `**` p < 0.01, `***` p < 0.001. Non-significant pairs: the p-value, e.g. `p = 0.21`. |
   | 4 or more | A compact letter display above each group, sized to the width of its group. |

   When the omnibus test is not significant (p ≥ 0.05), the figure shows the
   omnibus p-value, e.g. `Kruskal-Wallis p = 0.34`, with no brackets or letters.
   Letters are left out only if the letter display cannot be computed.

### In the export

Every figure and analysis export on the analysis pages is an Excel workbook
(`.xlsx`, written by `save_excel_button`). The CSV exports on those pages
(`save_df_button`, `save_multi_df_button`, `save_csv_button`, and the CSV
`st.download_button`s) become workbooks:

- the data that was in the CSV becomes a sheet named for its contents;
- the content is unchanged, including the per-fly and per-group data with the
  calculated N and SD or SEM;
- a multi-file CSV export becomes one workbook with one sheet per former file;
- browser downloads are built with `export_helpers.excel_bytes`;
- the Phase Shift per-day CSV download is removed. Its table is the `per_day`
  sheet of the phase-difference workbook.

The Export Data page, the SCAMP export, the CLI `tables` output, and the
group-average scalogram files (PNG + CSV, shared with the CLI) keep their
current formats.

A figure with statistics adds one sheet, `Stats`, to its workbook. It holds the
statistics that ran and are shown, and nothing else:

- the test names;
- the omnibus statistic and p-value;
- the p-value for each pair (adjusted where the post-hoc test adjusts);
- the letters, when letters are shown.

### General rules

- Every test runs on one value per fly.
- With facet panels on, the `compare_by` levels are tested within each panel.
  With panels off, all groups in the figure are tested together.
- Groups with fewer than 3 flies are left out of the test.
- The computation lives in a new `core/group_stats.py`. Plotting functions take
  an optional `stats=` result and draw from it, and pages export the same result.

## Inventory

| Mark | Meaning |
|---|---|
| ✅ | Add stats |
| ⚠️ | Add stats after core work (per-fly data is not yet returned) |
| ◐ | Optional, later |
| — | No stats |

### Activity & Sleep page (`app_pages/sleep_activity.py`)

| Figure | Function | Stats | Family | Test |
|---|---|---|---|---|
| Daily activity / sleep pattern | `daily_pattern_line`, `faceted_profiles` | ◐ | | Mixed model on per-fly binned values, `value ~ group * ZT_bin + (1\|fly)`, reporting the group × time interaction. |
| Activity totals (All day / Day / Night / Custom ZT window) | `group_violins`, `faceted_violins` via `per_fly_summary_table` | ✅ | W | One test per panel. |
| Sleep totals (All day / Day / Night / Custom ZT window) | same | ✅ | R | One test per panel. |
| Sleep bout duration curves | `sleep_bout_duration_lines` | ✅ **existing** | R | On per-fly mean bout duration. Replaces `sleep_analysis.bout_duration_group_stats`, which currently chooses ANOVA or Kruskal-Wallis from Shapiro-Wilk and Levene results. |

### Sleep States page (`app_pages/sleep_states.py`)

| Figure | Function | Stats | Family | Test |
|---|---|---|---|---|
| Short / Intermediate / Long sleep totals | `group_violins` on `per_fly_sleep_state_totals` | ✅ | R | One test per state panel. |
| Normalised sleep-state waveforms | `normalized_waveform_overlay` | — | | |
| Daily state profiles | `state_profile_plot` | — | | |
| P(initiation) with activity | `initiation_probability_plot` | — | | |
| Rose plots with activity | `rose_plot_with_activity` | — | | |
| Circadian gating arcs | `polar_gating_plot` | ◐ | | Rayleigh test within a group; Watson-Williams between groups, on each fly's mean phase. |
| Group-averaged scalograms | `sleep_state_scalogram` | — | | |
| Period vs amplitude | `period_amplitude_plot` | — **existing** | | Bootstrap 95 % CI over flies. |
| Ultradian amplitude over time | `ultradian_amplitude_plot` | — **existing** | | Bootstrap 95 % CI over flies. |
| Adjusted χ² periodograms | `chi_sq_periodogram_plot` | — **existing** | | Significance threshold at zero. |
| Sleep rebound per state | `rebound_bar_plot` (no page calls it) | ◐ **existing** | R | Currently a one-sample t-test against 0, computed inside the plotting function. |

### Sleep Deprivation page (`app_pages/sleep_deprivation.py`)

| Figure | Function | Stats | Family | Test |
|---|---|---|---|---|
| Day-selection preview | inline `fig_sel` | — | | |
| Baseline vs recovery profiles | `_baseline_recovery_fig` | — | | |
| Recovery − baseline difference per day | `_difference_fig` | — | | |
| Cumulative sleep difference | `_cumulative_fig` | ◐ | W | On each fly's end-of-day value. |
| Sleep totals by phase (Light / Dark × day) | `_totals_fig` | ⚠️ | W | Between groups per (day, phase). Within a group, baseline vs each recovery day: paired t-test (Wilcoxon signed-rank under the rank-based setting), Holm across recovery days. |
| Sleep rebound % | `_rebound_fig` | ⚠️ | W | Between groups per (recovery day, phase). |

`core/sleep_deprivation.compute_sd_analysis` currently returns only group-level
`phase_totals` and `rebound_pct`. It will return per-fly tables before these
tests are added.

### Phase Shift page (`app_pages/phase_shift.py`)

| Figure | Function | Stats | Family | Test |
|---|---|---|---|---|
| Phase response curve | `phase_response_curve` | ✅ | W | At each pulse ZT, Welch's t-test of pulsed vs matched unpulsed control (`control_map_from_pulse`), Holm across ZTs. |
| Phase response per fly | `phase_response_violins` | ✅ | W | Same tests as the curve, drawn as brackets. |
| Phase difference by day | inline `go.Scatter` | — **existing** | | Bootstrap SE / CI. |
| Phase shift by group, regression method | inline `go.Box` | ✅ | W | Across all groups. |
| Single-fly actograms with fits | `phase_shift_actogram` | — | | |

### Period & Rhythmicity page (`period_views/`)

| Figure | Function | Stats | Family | Test |
|---|---|---|---|---|
| Period by condition | `by_condition.render` → `group_violins` / `faceted_violins` | ✅ | W | |
| Rhythm strength by condition | same | ✅ | W | |
| Rhythmic flies by group (table) | `summarize_rhythmicity` | ✅ | | Fisher's exact test (2 groups) or χ² test (3+ groups; Fisher-Freeman-Halton when an expected count is under 5), then pairwise Fisher's exact with Holm correction. |
| Group-averaged periodograms | `group_spectrum_plot`, `faceted_spectra` | — | | |
| Threshold explorer | `threshold_coupled_figure` | — | | |
| CWT scalogram PNGs (CLI) | `save_group_average_scalogram_png` | — | | |

### HMM (`hmm_analysis_view.py`, `hmm_selection_view.py`)

| Figure | Function | Stats | Family | Test |
|---|---|---|---|---|
| State occupancy by group | `hmm_models.plot_state_occupancy_by_group` | ✅ | W | One test per state, on per-fly % occupancy. |
| Group state time course | `plot_group_state_timecourse` | — | | |
| State fractions by ZT | `plot_zt_state_fractions` | — | | |
| Hypnogram heatmap | `plot_hypnogram_heatmap` | — | | |
| Held-out LL, AIC/BIC, agreement vs K | `hmm_selection_view` | — | | |

### Data pages and batch report

| Figure | Function | Stats | Family | Test |
|---|---|---|---|---|
| Activity / moving heatmaps | `dataset_to_heatmap` | — | | |
| Actograms | inline `go.Bar` in `actograms.py` | — | | |
| Report: sleep summary bars | `summary_bars` | ✅ | R | Notice line and stats table printed under the figure. |
| Report: sleep-state totals bars | `sleep_state_totals_bars` | ✅ | R | As above. |
| Report: HMM occupancy bars | inline in `pipeline/report.py` | ✅ | W | As above. |
| Report: threshold explorer | `threshold_coupled_figure` | — | | |

## Order of work

1. Convert the analysis-page CSV exports to Excel workbooks.
2. Add `scikit-posthocs>=0.12` to `pyproject.toml`. Build `core/group_stats.py`
   with both families, the notice line, brackets or letters, the export table,
   facet-panel handling, and the R regression test.
3. Wire it into `group_violins` / `faceted_violins` and add the sidebar setting.
4. Move the bout-duration test onto `group_stats`.
5. HMM occupancy and the batch report.
6. Phase response curve and the phase-shift box.
7. Rhythmic-fraction test.
8. Sleep deprivation: per-fly tables, then tests.
9. Optional items.
