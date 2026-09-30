# Tests

```bash
python -m pip install -e ".[dev]"
python -m pytest
```

About 600 tests, a few minutes on a laptop. Most of that time goes to the
real-data period estimators and the page Run buttons. No browser, no server, no
port.

## How the app tests work

Streamlit ships a first-party headless test framework, `st.testing.v1.AppTest`.
It runs a Streamlit script **in-process**, lets you set widget values and rerun,
and exposes the resulting elements as lists you can assert on. That is what the
`app` fixture in `conftest.py` wraps:

```python
def test_something(app, master_ds):
    at = app(ds=master_ds, page="hmm")
    assert at.radio(key="cv_phase_choice").value == "LD"
```

Two things matter about that fixture:

- It always initialises from the **entrypoint** (`src/clockwork/app/ClockWork.py`) and then
  calls `switch_page`, because `st.navigation` resolves page paths relative to
  the entrypoint. Passing a child page to `AppTest.from_file` makes it the main
  script and changes how those paths resolve.
- It seeds `st.session_state["dataset"]` directly, so tests skip the whole
  import → curate → split pipeline. A page only needs a dataset in session
  state; how it got there is the import page's problem, not every page's.

## The fixtures are synthetic, and must stay faithful

`conftest.py` builds a small xarray dataset by hand rather than committing a
`.nc`. A real one is tens of megabytes and would rot against the loader. The
exception is the real-data fixture described below, which is imported from the
raw monitor files on every run, never loaded from a saved `.nc`.

The catch is that a synthetic fixture only tests what it faithfully imitates.
Three contracts were discovered the hard way while writing these, and are
commented in `conftest.py` because getting them wrong produces confusing
failures a long way from the cause:

- `activity` and `moving` are `(time, id)`; the sleep masks are `(id, time)`.
  The dimension order is genuinely not uniform, and code that extracts per-fly
  series indexes the wrong axis if a fixture normalises it.
- `stop_datetime` is required as well as `start_datetime`;
  `get_zt_binned_dataframe` raises without it, which empties every ZT plot.
- Group signals must actually differ. Streamlit derives a `plotly_chart`'s
  element id from its serialized spec, so two figures that come out identical
  raise `StreamlitDuplicateElementId` — an error real data never triggers.

If a page needs something else a real import produces, add it to the fixture.

## Real data, and pinned snapshots

A synthetic fixture checks that a contract is kept. It cannot check what an
estimator says about a real fly, because a sine wave with noise is far easier
than a real record with its dead channels, weak rhythms and phase drift. Those
tests use `example_ds` (in `conftest.py`) instead: Monitors 17 and 18 of the
committed `example_data`, 64 flies, imported through the real loader.

Real flies have no ground truth, so those tests compare against a **snapshot**:
the per-fly output that a person reviewed and committed under `tests/snapshots/`
(see `snapshot_util.py`). When a number changes, the failure lists the flies that
moved and by how much. If the change is intended, regenerate:

```bash
python -m pytest --update-snapshots
git diff tests/snapshots/
```

Each regenerated file begins with a `_summary` block, for example the median
period and the number of rhythmic flies per genotype. Read it, and the diff,
before you commit. A missing snapshot fails the test; it is never written
automatically.

Tests marked `gpu` need CUDA with torch and ptwt, and skip everywhere else,
including CI. On a GPU machine, run them with `python -m pytest -m gpu`.

## What is covered

| Area | Files |
|---|---|
| Pages render, and their Run buttons work | `test_pages_smoke`, `test_page_runs`, `test_page_behaviour`, `test_file_dialogs` |
| Import: raw files, status codes, gaps, metadata pairing | `test_dam_parsing`, `test_monitor_file_resolution`, `test_monitor_label_pairing`, `test_integrity_surfacing` |
| Curation, split, gap trim | `test_curation`, `test_segment_trim_masking`, `test_phase_metadata` |
| Grouping | `test_grouping_provenance`, `test_regroup` |
| Period estimators (LS, AC, MESA, CWT) and rhythmicity calls on real flies | `test_period_estimators_snapshot` |
| Sleep: bouts, states, re-runs, deprivation, sleep-state CWT | `test_sleep_bout_boundaries`, `test_sleep_analysis_alignment`, `test_sleep_analysis_rerun`, `test_sleep_reorganisation`, `test_sleep_state_metrics`, `test_sleep_states_edge_cases`, `test_sleep_cwt_truth`, `test_sleep_deprivation` |
| Phase shifts and the phase response curve | `test_phase_response`, `test_phase_response_figures`, `test_phase_shift_controls` |
| Performance rewrites match the code they replaced (gap fill, rolling mean, batched GPU CWT, worker counts) | `test_vectorized_equivalence` |
| Figures | `test_actograms`, `test_dd_heatmap_alignment`, `test_plotting_seam`, `test_figure_export` |
| Exports and the saved `.nc` | `test_exports`, `test_workbook_exports`, `test_scamp_export`, `test_netcdf_roundtrip`, `test_experiment_export_dirs`, `test_experiment_naming_applies` |
| Caching | `test_cache_keys` |

Each file's docstring says which bug or contract it protects.
`test_sleep_bout_boundaries` also documents an open question: bout duration is
off by one minute compared with the standard 5-minute definition. It holds that
with `xfail(strict=True)` tests, which will need updating if the rule is fixed.

## When not to use AppTest

Reach for the real app when the thing under test is not the script's output:
byte-level export files, actual rendering, CSS, or custom-component JavaScript.
The SCAMP export was first verified by diffing 724 real files against a
baseline, which `AppTest` could not have done. `test_scamp_export` now reads the
files back and pins a checksum of each one.

Pure logic that never touches `st.*` should be tested directly with pytest
rather than through `AppTest`; see `test_exports.py`, most of which does.
