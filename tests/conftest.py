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
    # TWO temperatures, so genotype+temperature and genotype alone give a
    # DIFFERENT number of groups. With one temperature, narrowing the grouping
    # changes nothing and a regroup test proves nothing.
    temps = ("25C", "29C")
    ids, group_labels, genotypes, temperatures = [], [], [], []
    for g in groups:
        for i in range(n_per_group):
            t = temps[i % len(temps)]
            ids.append(f"20250115_1_{g}{i}")
            group_labels.append(f"{g}-{t}")
            genotypes.append(g)
            temperatures.append(t)

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
            "temperature": ("id", np.array(temperatures)),
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
            # A LIST, because create_xarray_dataset writes list(chosen) — a
            # comma-joined string made get_group_columns return one bogus name
            # ("genotype,temperature"), and any page defaulting its grouping to
            # the dataset's own silently got nothing.
            "group_columns": ["genotype", "temperature"],
            "sleep_threshold_seconds": 300,
            # create_xarray_dataset writes ONE ATTR PER METADATA COLUMN (its
            # unique values). That is how a built dataset records which of its
            # per-id coords came from the metadata, and it is what
            # group_defining_coords uses to tell them from coords a later
            # analysis added. Without these the regroup UI offers nothing.
            "genotype": sorted(set(genotypes)),
            "temperature": sorted(set(temperatures)),
            "start_datetime": [str(start)],
            "stop_datetime": [str(start)],
            "first_DD_day": [str(start)],
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


@pytest.fixture(autouse=True)
def _isolate_bare_session_state():
    """Empty Streamlit's bare-mode session state around every test.

    Two session states are in play and only one of them is per-test. ``AppTest``
    carries its own (``at.session_state``), but a test that calls app code
    DIRECTLY — no AppTest — touches ``st.session_state``, which in bare mode is a
    single process-global dict that outlives the test that wrote it.

    That makes ordering bugs: ``ui.charts`` reads the loaded dataset from session
    state to prefix download filenames, so one test leaving a named dataset behind
    renamed another test's figure, and only when the two ran in that order.
    """
    import streamlit as st

    st.session_state.clear()
    yield
    st.session_state.clear()


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


def _build_with_sleep_structure(n_per_group=3, groups=("ctrl", "mut"), seed=1):
    """Like ``_build`` but with immobility long enough to make real bouts.

    ``_build``'s ``moving`` flickers with Poisson noise, so it yields almost no
    bout over 60 minutes and every long-sleep figure comes out empty. The
    sleep-state work needs all three states populated, so this lays down an
    explicit structure: mostly immobile through the subjective night, mostly
    mobile through the day, and a siesta — which is also roughly what a real
    fly does, and what makes the rose plots and the CWT non-degenerate.
    """
    rng = np.random.default_rng(seed)
    ds = _build(n_per_group=n_per_group, groups=groups, with_sleep=False, seed=seed)
    n_time = ds.sizes["time"]
    n_id = ds.sizes["id"]
    minute_of_day = (ds["time"].values % MINUTES_PER_DAY).astype(int)

    moving = np.zeros((n_time, n_id), dtype=np.float32)
    for fly in range(n_id):
        # Base probability of moving by time of day: awake ZT0-12 with a
        # midday siesta, asleep ZT12-24.
        p = np.where(minute_of_day < 12 * 60, 0.55, 0.04)
        siesta = (minute_of_day >= 5 * 60) & (minute_of_day < 8 * 60)
        p = np.where(siesta, 0.10, p)
        # A per-fly offset so no two flies are identical (identical traces make
        # two figures serialize the same and trip StreamlitDuplicateElementId).
        p = np.clip(p + 0.03 * (fly - n_id / 2) / n_id, 0.01, 0.95)
        moving[:, fly] = (rng.random(n_time) < p).astype(np.float32)

    ds["moving"] = (("time", "id"), moving)
    ds["activity"] = (("time", "id"), (moving * rng.poisson(4, (n_time, n_id))).astype(np.float32))
    return ds


@pytest.fixture(scope="session")
def states_ds():
    """Master dataset with sleep states and the per-bout table really computed.

    Runs the production ``sleep_analysis`` rather than hand-writing masks, so
    the bout table, the state labels and the masks are guaranteed consistent
    with each other the way the pages require.
    """
    import sleep_analysis

    return sleep_analysis.sleep_analysis(
        _build_with_sleep_structure(), phase="both", sleep_threshold_sec=300
    )


# The phase-shift cohort is built to a KNOWN design, so the analysis can be checked
# against a right answer rather than against itself. Two genotypes that differ in
# baseline phase by PULSE_BASELINE_GAP_H, each with a pulsed and an unpulsed arm, and
# the pulsed arms delayed by PULSE_SHIFT_H from the first DD day onward.
#
# The baseline gap is the point: referring every group to ONE control reports it as
# part of the shift, which is what the matched pairing and the rebasing exist to stop.
PULSE_DD_DAY = 3
PULSE_ZT_HOUR = 15.0
PULSE_SHIFT_H = 1.5
PULSE_BASELINE_GAP_H = 2.0
PULSE_BASE_PHASE = {"A": 12.0, "B": 12.0 + PULSE_BASELINE_GAP_H}


def _build_pulse_cohort(n_per_arm=6, n_days=8, seed=0):
    """Two genotypes x (pulsed, unpulsed), with a designed phase delay after the pulse.

    Each fly's activity is one Gaussian bump per day centred on its group's peak
    hour — sharp enough to survive the 12 h low-pass filter the peak method applies,
    which a Poisson trace like ``_build``'s is not. The pulsed arms sit in one flybox
    and the controls in another, so ``describe_by`` has something real to report.
    """
    rng = np.random.default_rng(seed)
    ids, genotypes, conditions, boxes, zt = [], [], [], [], []
    for gene in ("A", "B"):
        for cond in ("LP", "noLP"):
            for k in range(n_per_arm):
                ids.append(f"{gene}_{cond}_{k}")
                genotypes.append(gene)
                conditions.append(cond)
                boxes.append("bun" if cond == "LP" else "pie")
                # Blank for the unpulsed arm, which is what an unpulsed cohort's
                # metadata really looks like, and must survive as NaN.
                zt.append(PULSE_ZT_HOUR if cond == "LP" else np.nan)

    n_id = len(ids)
    time = np.arange(n_days * MINUTES_PER_DAY)
    activity = np.zeros((time.size, n_id))
    for j in range(n_id):
        for day in range(n_days):
            # The pulse is given late on the last entrained day and moves the peak
            # from the first DD day onward — see dam_utilities._derive_pulse_minute.
            shifted = conditions[j] == "LP" and day >= PULSE_DD_DAY
            hour = PULSE_BASE_PHASE[genotypes[j]] + (PULSE_SHIFT_H if shifted else 0.0)
            centre = day * MINUTES_PER_DAY + hour * 60.0
            activity[:, j] += 20 * np.exp(-0.5 * ((time - centre) / 120.0) ** 2)
    activity += rng.normal(0, 0.05, activity.shape).clip(0)

    start = np.datetime64("2025-01-15T09:00:00")
    return xr.Dataset(
        {"activity": (("time", "id"), activity)},
        coords={
            "id": ids,
            "time": time.astype(np.int64),
            "genotype": ("id", np.array(genotypes)),
            "condition": ("id", np.array(conditions)),
            "flybox": ("id", np.array(boxes)),
            "pulse_zt_hour": ("id", np.array(zt, dtype=float)),
            "start_datetime": ("id", np.array([start] * n_id)),
            "first_DD_day": (
                "id",
                np.array(
                    [start + np.timedelta64(PULSE_DD_DAY * MINUTES_PER_DAY, "m")] * n_id
                ),
            ),
        },
        attrs={
            "time_is_relative_minutes": 1,
            "phase": "full",
            "split_applied": 1,
            "group_columns": ["genotype", "condition"],
            # create_xarray_dataset writes one attr per metadata column; that is what
            # group_defining_coords reads to tell metadata coords from derived ones,
            # and the page's grouping picker is built from it.
            "genotype": ["A", "B"],
            "condition": ["LP", "noLP"],
            "flybox": ["bun", "pie"],
        },
    )


@pytest.fixture(scope="session")
def pulse_ds():
    """The known-design phase-shift cohort (see :func:`_build_pulse_cohort`)."""
    return _build_pulse_cohort()
