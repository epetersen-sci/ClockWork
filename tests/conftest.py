"""Shared fixtures.

The datasets here are SYNTHETIC and deliberately tiny — a few flies over a few
days — so the suite runs in seconds and needs nothing on disk. They are built to
satisfy the same contracts the real loader produces, because that is what the
pages read:

- a relative integer-minute ``time`` axis plus ``attrs['time_is_relative_minutes']``,
  which ``dam_utilities.select_phase`` requires;
- per-fly ``start_datetime`` / ``first_DD_day`` coords, from which ``split_minute``
  (the LD→DD boundary) is derived;
- a ``group`` coord, which every group filter and group-level export keys off.

If a test needs something a real import produces that is missing here, add it to
the fixture rather than loading a ``.nc`` — a committed fixture file would be tens
of megabytes and would rot against the loader.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

REPO_ROOT = Path(__file__).resolve().parent.parent
ENTRYPOINT = REPO_ROOT / "app" / "ClockWork.py"

# core/ and app/ are import roots at runtime; the router sets sys.path once, but
# tests import from them directly (and before any page runs), so do it here too.
for _p in (REPO_ROOT / "core", REPO_ROOT / "app", REPO_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


N_DAYS = 6
DD_DAY = 2  # LD occupies days 0-1, DD days 2-5
MINUTES_PER_DAY = 1440


def _build(n_per_group=3, groups=("ctrl", "mut"), with_sleep=True, seed=0):
    """A small master dataset spanning an LD epoch and a DD epoch."""
    rng = np.random.default_rng(seed)
    ids, group_labels, genotypes = [], [], []
    for g in groups:
        for i in range(n_per_group):
            ids.append(f"20250115_1_{g}{i}")
            group_labels.append(f"{g}-25C")
            genotypes.append(g)

    n_id = len(ids)
    n_time = N_DAYS * MINUTES_PER_DAY
    time = np.arange(n_time, dtype=np.int64)

    # A crude circadian signal so period/sleep code has something non-degenerate.
    # Amplitude and phase vary BY GROUP and the noise varies by fly: identical
    # traces across groups would make separate figures serialize identically,
    # and Streamlit derives a plotly_chart's element id from its spec — so a
    # degenerate fixture raises StreamlitDuplicateElementId where real data
    # never would.
    tod = (time % MINUTES_PER_DAY) / MINUTES_PER_DAY * 2 * np.pi
    rows = []
    for idx, g in enumerate(group_labels):
        gi = groups.index(g.split("-")[0])
        amp = 3.0 + 1.5 * gi
        shift = 0.4 * gi
        rows.append(
            (np.sin(tod + shift) + 1.0) * amp + rng.poisson(0.5 + 0.2 * idx, n_time)
        )
    activity = np.stack(rows).astype(np.float32)
    moving = (activity > 2.0).astype(np.float32)

    start = np.datetime64("2025-01-15T09:00:00")
    # DIMENSION ORDER IS NOT UNIFORM, and the fixture has to match what the real
    # loader produces or the segment/extraction code indexes the wrong axis:
    # `activity` and `moving` are (time, id); the sleep masks are (id, time).
    data_vars = {
        "activity": (("time", "id"), activity.T),
        "moving": (("time", "id"), moving.T),
    }
    if with_sleep:
        # int8 with -1 for "no data", matching what sleep_analysis writes.
        sleep = (moving == 0).astype(np.int8)
        data_vars["sleep"] = (("id", "time"), sleep)
        for var in ("sleep_short", "sleep_intermediate", "sleep_long"):
            data_vars[var] = (("id", "time"), np.zeros((n_id, n_time), dtype=np.int8))

    ds = xr.Dataset(
        data_vars,
        coords={
            "id": ids,
            "time": time,
            "group": ("id", np.array(group_labels)),
            "genotype": ("id", np.array(genotypes)),
            "temperature": ("id", np.array(["25C"] * n_id)),
            "start_datetime": ("id", np.array([start] * n_id)),
            # get_zt_binned_dataframe requires BOTH datetimes and raises without
            # stop_datetime; every ZT-binned plot and export goes through it.
            "stop_datetime": (
                "id",
                np.array([start + np.timedelta64(n_time, "m")] * n_id),
            ),
            "first_DD_day": (
                "id",
                np.array([start + np.timedelta64(DD_DAY * MINUTES_PER_DAY, "m")] * n_id),
            ),
        },
        attrs={
            "time_is_relative_minutes": 1,
            "phase": "full",
            "split_applied": 1,
            "gap_threshold_minutes": 60,
            "split_discard_first_dd_day": 0,
            "group_columns": "genotype,temperature",
            "sleep_threshold_seconds": 300,
        },
    )
    return ds


@pytest.fixture(scope="session")
def master_ds():
    """Split-applied master spanning LD and DD, with sleep already computed."""
    return _build()


@pytest.fixture(scope="session")
def unsplit_ds():
    """Same shape but with no LD/DD boundary — exercises the fallback branches."""
    ds = _build().drop_vars("first_DD_day")
    ds.attrs["phase"] = "full"
    ds.attrs["split_applied"] = 0
    return ds


@pytest.fixture
def binned_df():
    """The long per-fly ZT-binned frame the ZT summary exports consume."""
    rows = []
    for fly, group in (("f1", "b-grp"), ("f2", "b-grp"), ("f3", "a-grp")):
        for zt_bin in (0, 30, 60):
            rows.append(
                {"id": fly, "group": group, "zt_bin_minute": zt_bin, "activity": zt_bin / 10.0}
            )
    return pd.DataFrame(rows)


@pytest.fixture
def app():
    """Factory: an AppTest on the real entrypoint, seeded with a dataset.

    Always initialised from the ENTRYPOINT (never a child page) so ``st.navigation``
    resolves page paths the way it does in the running app.
    """
    from streamlit.testing.v1 import AppTest

    def _make(ds=None, page=None, **session_state):
        at = AppTest.from_file(str(ENTRYPOINT), default_timeout=120)
        if ds is not None:
            at.session_state["dataset"] = ds
            at.session_state["dataset_full"] = ds
        for key, value in session_state.items():
            at.session_state[key] = value
        if page is not None:
            at.switch_page(f"app_pages/{page}.py")
        return at.run()

    return _make
