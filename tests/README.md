# Tests

```bash
python -m pip install -r requirements-dev.txt
python -m pytest
```

79 tests, about 20 seconds. No browser, no server, no port.

## How the app tests work

Streamlit ships a first-party headless test framework, `st.testing.v1.AppTest`.
It runs a Streamlit script **in-process**, lets you set widget values and rerun,
and exposes the resulting elements as lists you can assert on. That is what the
`app` fixture in `conftest.py` wraps:

```python
def test_something(app, master_ds):
    at = app(ds=master_ds, page="hmm_model_selection")
    assert at.radio(key="cv_phase_choice").value == "LD"
```

Two things matter about that fixture:

- It always initialises from the **entrypoint** (`app/ClockWork.py`) and then
  calls `switch_page`, because `st.navigation` resolves page paths relative to
  the entrypoint. Passing a child page to `AppTest.from_file` makes it the main
  script and changes how those paths resolve.
- It seeds `st.session_state["dataset"]` directly, so tests skip the whole
  import → curate → split pipeline. A page only needs a dataset in session
  state; how it got there is the import page's problem, not every page's.

## The fixtures are synthetic, and must stay faithful

`conftest.py` builds a small xarray dataset by hand rather than committing a
`.nc`. A real one is tens of megabytes and would rot against the loader.

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

## What is covered

| File | Covers |
|---|---|
| `test_pages_smoke.py` | Every page renders, with a dataset, without one, and on an unsplit dataset. Pages are top-level scripts, so a `NameError` in an unclicked branch still takes the page down. |
| `test_phase_metadata.py` | Canonical `phase` / `split_applied` attrs, and the legacy `split_phase` alias that is still **read** for old `.nc` files (backlog item 4). |
| `test_cache_keys.py` | `@st.cache_data` helpers take the fingerprint as `fp`, not `_fp` (backlog item 14), plus a test pinning the upstream Streamlit behaviour that rule depends on. |
| `test_exports.py` | ZT summary column order and casing (item 2), `phase_slice` dtypes and parameters (item 5), the single bout-dataframe source (item 1). |
| `test_page_behaviour.py` | The HMM phase picker (item 3) and the shared display group filter surviving a page switch (item 11) — both previously verified by hand in a browser. |

## When not to use AppTest

Reach for the real app when the thing under test is not the script's output:
byte-level export files, actual rendering, CSS, or custom-component JavaScript.
The SCAMP export was verified by diffing 724 real files against a baseline —
`AppTest` could not have told you those bytes matched.

Pure logic that never touches `st.*` should be tested directly with pytest
rather than through `AppTest`; see `test_exports.py`, most of which does.
