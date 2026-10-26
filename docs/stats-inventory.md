# Figure statistics

The automatic statistics on ClockWork figures: which figures carry them, which
tests run, and how results are shown and exported. The tests live in
`core/group_stats.py`; the marks on Plotly figures are drawn by
`plotting.add_group_stats`, and the app's setting, notice line and `Stats` sheet
are in `app/ui/stats.py`.

## Tests

The test for each kind of measure is fixed in advance; it does not depend on the
data.

| Family | 3+ groups | 2 groups | Used for |
|---|---|---|---|
| **W** — Welch | Welch ANOVA, then Games-Howell (all pairs) | Welch's t-test | Activity, period, rhythm strength, phase shift, HMM state occupancy, sleep deprivation |
| **R** — Rank-based | Kruskal-Wallis, then Dunn's test with Holm correction (all pairs) | Mann-Whitney U | Sleep totals, sleep bout duration, sleep-state totals |

Post-hoc tests run only when the omnibus test is significant (p < 0.05), and
compare every pair of groups.

Other tests:

| Test | Used for |
|---|---|
| Welch's t-test (Mann-Whitney U under the rank-based setting), each pulsed group vs its unpulsed control, Holm across the pulse times of each series | Phase response curve and per-fly phase response violins. The control is the group with `pulse_duration_min = 0` matched on the other grouping columns (`phase_shift.control_map_from_pulse`). |
| Paired t-test (Wilcoxon signed-rank under the rank-based setting), Holm across recovery days | Sleep deprivation: each group's baseline vs each recovery day, same flies |
| Fisher's exact test (2 groups); χ² test (3+ groups), or Fisher-Freeman-Halton by Monte Carlo (99,999 tables, fixed seed) when an expected count is under 5; then pairwise Fisher's exact tests with Holm correction | Share of rhythmic flies per group |
| Rayleigh test | Circadian gating: concentration of each fly's mean phase, per group and state |
| One-sample t-test vs 0 (Wilcoxon signed-rank vs 0 under the rank-based setting) | `plotting.rebound_bar_plot`, when given results; no page calls it |

### Libraries

| Step | Function | Package |
|---|---|---|
| Welch ANOVA | `statsmodels.stats.oneway.anova_oneway(..., use_var="unequal")` | statsmodels |
| Games-Howell | `scikit_posthocs.posthoc_games_howell` | scikit-posthocs |
| Kruskal-Wallis | `scipy.stats.kruskal` | scipy |
| Dunn | `scikit_posthocs.posthoc_dunn(..., p_adjust="holm")` | scikit-posthocs |
| Welch's t / Mann-Whitney U | `scipy.stats.ttest_ind(equal_var=False)` / `scipy.stats.mannwhitneyu` | scipy |
| Paired t / Wilcoxon signed-rank / one-sample t | `scipy.stats.ttest_rel` / `scipy.stats.wilcoxon` / `scipy.stats.ttest_1samp` | scipy |
| Fisher's exact / χ² / Fisher-Freeman-Halton | `scipy.stats.fisher_exact` / `scipy.stats.chi2_contingency` / `scipy.stats.fisher_exact(method=MonteCarloMethod(...))` (scipy ≥ 1.15; on older scipy the χ² result is reported) | scipy |
| Holm correction | `statsmodels.stats.multitest.multipletests(method="holm")` | statsmodels |
| Rayleigh test | `astropy.stats.rayleightest` | astropy |
| Compact letter display | `scikit_posthocs.compact_letter_display` | scikit-posthocs |

scikit-posthocs is a dependency, `scikit-posthocs>=0.17.1`.
`tests/test_group_stats.py` checks Welch ANOVA, Kruskal-Wallis, Games-Howell,
Dunn, Welch's t and Mann-Whitney U against reference values computed in base R
(`tests/reference/group_stats_reference.R`).

### Setting

`outputs.group_stats` in the CLI config for the QC report. The app always uses
the recommended families.

- **Recommended per measure** (`recommended`, default): the families above.
- **Welch ANOVA + Games-Howell for everything** (`welch`).
- **Kruskal-Wallis + Dunn for everything** (`rank`).

### On the figure

1. **Notice line.** One line under every tested figure or table, naming the test
   and asking the user to check it, for example:

   > *Kruskal-Wallis + Dunn (Holm), one value per fly — check this test suits your data before relying on the result.*

2. **Significance marks.** Group count is per panel when facet panels are on.

   | Groups | Marks |
   |---|---|
   | 2 or 3 | A bracket on every pair. Significant pairs: `*` p < 0.05, `**` p < 0.01, `***` p < 0.001. Non-significant pairs: the p-value, e.g. `p = 0.21`. |
   | 4 or more | A compact letter display above each group, sized to the width of one group. |

   When the omnibus test is not significant, the figure shows the omnibus
   p-value, e.g. `Kruskal-Wallis p = 0.340`, with no brackets or letters.
   Letters are left out only if the letter display cannot be computed.

   Figures not laid out by group carry their result as text: the phase response
   curve and violins label each pulsed point or column with stars or the
   p-value; the gating figures list the Rayleigh result per state under the
   title; the bout-duration curves name the omnibus test and p-value. Paired
   sleep-deprivation tests, the cumulative sleep difference and the
   rhythmic-fraction tests are tables under their figure.

### In the export

Every analysis-page export is an Excel workbook (`.xlsx`), written by
`save_excel_button` or, for browser downloads, `export_helpers.excel_bytes`. The
per-fly and per-group data are unchanged, including the calculated N and SD or
SEM. The threshold explorer's per-algorithm export is one workbook with a sheet
per algorithm. The Phase Shift per-day table is the `per_day` sheet of the
phase-difference workbook.

The Export Data page, the SCAMP export, the CLI `tables` output, and the
group-average scalogram files (PNG + CSV) keep their formats.

A figure with statistics adds one sheet, `Stats`, to its workbook. It holds the
statistics that ran and are shown, and nothing else: `Panel` and `Measure` (when
set), `Test`, `Comparison`, `Statistic`, `df`, `p value` (adjusted where the
test adjusts) and `Letters` (when letters are shown).

### General rules

- Every test runs on one value per fly.
- With facet panels on, the `compare_by` levels are tested within each panel.
  With panels off, all groups in the figure are tested together.
- Groups with fewer than 3 flies are left out of the test.

## Inventory

| Mark | Meaning |
|---|---|
| ✅ | Stats shown |
| — | No stats |

### Activity & Sleep page (`app_pages/sleep_activity.py`)

| Figure | Function | Stats | Test |
|---|---|---|---|
| Daily activity / sleep pattern | `daily_pattern_line`, `faceted_profiles` | — | |
| Activity totals (All day / Day / Night / Custom ZT window) | `group_violins`, `faceted_violins` | ✅ | W, one test per panel |
| Sleep totals (All day / Day / Night / Custom ZT window) | same | ✅ | R, one test per panel |
| Sleep bout duration curves | `sleep_bout_duration_lines` | ✅ | R, on per-fly geometric mean bout duration; pairs in an expander |

### Sleep States page (`app_pages/sleep_states.py`)

| Figure | Function | Stats | Test |
|---|---|---|---|
| Short / Intermediate / Long sleep totals | `group_violins` | ✅ | R, one test per state |
| Normalised sleep-state waveforms | `normalized_waveform_overlay` | — | |
| Daily state profiles | `state_profile_plot` | — | |
| P(initiation) with activity | `initiation_probability_plot` | — | |
| Rose plots with activity | `rose_plot_with_activity` | — | |
| Circadian gating arcs | `polar_gating_plot` | ✅ | Rayleigh test per group and state |
| Group-averaged scalograms | `sleep_state_scalogram` | — | |
| Period vs amplitude | `period_amplitude_plot` | — | Bootstrap 95 % CI over flies |
| Ultradian amplitude over time | `ultradian_amplitude_plot` | — | Bootstrap 95 % CI over flies |
| Adjusted χ² periodograms | `chi_sq_periodogram_plot` | — | Significance threshold at zero |

### Sleep Deprivation page (`app_pages/sleep_deprivation.py`)

| Figure | Function | Stats | Test |
|---|---|---|---|
| Day-selection preview | inline `fig_sel` | — | |
| Baseline vs recovery profiles | `_baseline_recovery_fig` | — | |
| Recovery − baseline difference per day | `_difference_fig` | — | |
| Cumulative sleep difference | `_cumulative_fig` | ✅ | W on each fly's value at ZT24, per recovery day (table) |
| Sleep totals by phase (bars grouped by day) | `_totals_fig` | ✅ | W between groups per (day, phase); paired baseline vs recovery per group and phase (table) |
| Sleep rebound % (bars grouped by day) | `_rebound_fig` | ✅ | W between groups per (recovery day, phase) |

`core/sleep_deprivation.compute_sd_analysis` returns the per-fly tables
`phase_totals_per_fly`, `rebound_per_fly` and `cumulative_end_per_fly`; the
group tables are their means.

### Phase Shift page (`app_pages/phase_shift.py`)

| Figure | Function | Stats | Test |
|---|---|---|---|
| Phase response curve | `phase_response_curve` | ✅ | Pulsed vs unpulsed control per pulse ZT, Holm across ZTs |
| Phase response per fly | `phase_response_violins` | ✅ | Same tests as the curve |
| Phase difference by day | inline `go.Scatter` | — | Bootstrap SE / CI |
| Phase shift by group, regression method | inline `go.Box` | ✅ | W across all groups |
| Single-fly actograms with fits | `phase_shift_actogram` | — | |

### Period & Rhythmicity page (`period_views/`)

| Figure | Function | Stats | Test |
|---|---|---|---|
| Period by condition | `by_condition.render` | ✅ | W |
| Rhythm strength by condition | `by_condition.render` | ✅ | W |
| Rhythmic flies by group (table) | `summarize_rhythmicity` | ✅ | Proportions, one test per algorithm |
| Group-averaged periodograms | `group_spectrum_plot`, `faceted_spectra` | — | |
| Threshold explorer | `threshold_coupled_figure` | — | |
| CWT scalogram PNGs (CLI) | `save_group_average_scalogram_png` | — | |

### HMM (`hmm_analysis_view.py`, `hmm_selection_view.py`)

| Figure | Function | Stats | Test |
|---|---|---|---|
| State occupancy by group (bars grouped by state) | `hmm_models.plot_state_occupancy_by_group` | ✅ | W, one test per state |
| Group state time course | `plot_group_state_timecourse` | — | |
| State fractions by ZT | `plot_zt_state_fractions` | — | |
| Hypnogram heatmap | `plot_hypnogram_heatmap` | — | |
| Held-out LL, AIC/BIC, agreement vs K | `hmm_selection_view` | — | |

### Data pages and QC report

| Figure | Function | Stats | Test |
|---|---|---|---|
| Activity / moving heatmaps | `dataset_to_heatmap` | — | |
| Actograms | inline `go.Bar` in `actograms.py` | — | |
| Report: sleep totals (Total / Day / Night violins) | `report._violins_with_stats` | ✅ | R; notice and stats table under the figures |
| Report: sleep-state totals (violins) | same | ✅ | R |
| Report: HMM occupancy (violins per state) | same | ✅ | W |
| Report: threshold explorer | `threshold_coupled_figure` | — | |
