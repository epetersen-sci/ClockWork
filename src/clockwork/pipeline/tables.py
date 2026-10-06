"""The result tables a run writes: one function per table, each a DataFrame of
the dataset as it stands.

The Export page offers the same tables, from the same functions, so a CSV from
the GUI and one from ``clockwork run`` cannot differ in shape.

Tables of time-of-day measures (sleep totals, sleep states) are per EPOCH: one
block of rows for LD and one for DD when the recording has both, because
averaging entrained and free-running days onto one 24 h axis describes neither
(the Sleep pages do the same).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from clockwork.core import dam_utilities, plotting, sleep_analysis
from clockwork.core.dataset_meta import PHASE_DD, PHASE_LD, dataset_phase

#: (column, variable, present when this variable is) — the period summary's
#: per-fly quantities, in the order the Export page has always written them.
_PERIOD_FLOAT_COLUMNS = (
    ("CWT_Period_h", "cwt_period", "cwt_period"),
    ("CWT_Power", "cwt_power", "cwt_period"),
    ("CWT_Period_Stability_h", "cwt_period_stability", "cwt_period_stability"),
    ("CWT_Rhythmicity", "cwt_rhythmicity", "cwt_rhythmicity"),
    ("LS_Period_h", "ls_period", "ls_period"),
    ("LS_Power", "ls_power", "ls_period"),
    ("LS_FAP", "ls_fap", "ls_fap"),
    ("AC_Period_h", "ac_period", "ac_period"),
    ("AC_Power_RI", "ac_power", "ac_period"),
    ("AC_Rhythm_Strength", "ac_rhythm_strength", "ac_rhythm_strength"),
    ("MESA_Period_h", "mesa_period", "mesa_period"),
    ("MESA_Power", "mesa_power", "mesa_period"),
)
#: Per-method rhythmic flags (written by rhythmicity_classification.classify_*).
_PERIOD_FLAG_COLUMNS = (
    ("LS_Rhythmic", "ls_rhythmic"),
    ("AC_Rhythmic", "ac_rhythmic"),
    ("CWT_Rhythmic", "cwt_rhythmic"),
)


def period_summary(ds: xr.Dataset) -> pd.DataFrame:
    """One row per fly: each method's period, its strength, and its rhythmic call."""
    cols = {"ID": [str(i) for i in ds["id"].values]}
    if "group" in ds.coords:
        cols["Group"] = [str(g) for g in ds["group"].values]
    for col, var, present_if in _PERIOD_FLOAT_COLUMNS:
        if present_if in ds.data_vars and var in ds.data_vars:
            cols[col] = [float(v) for v in ds[var].values]
    for col, coord in _PERIOD_FLAG_COLUMNS:
        if coord in ds.coords:
            cols[col] = [bool(v) for v in ds[coord].values]
    df = pd.DataFrame(cols)
    # Group then ID: a predictable, GraphPad-friendly layout.
    keys = [c for c in ("Group", "ID") if c in df.columns]
    return df.sort_values(keys).reset_index(drop=True) if keys else df


def epochs(ds: xr.Dataset) -> list[tuple[str, xr.Dataset]]:
    """``(label, view)`` for each lighting epoch the dataset holds, as the Sleep
    pages offer them: LD and DD when there is a boundary, the one epoch of a
    saved partition, else the whole recording."""
    stamped = dataset_phase(ds)
    if stamped in (PHASE_LD, PHASE_DD):
        return [(stamped, ds)]
    if "split_minute" in ds.coords or "first_DD_day" in ds.coords:
        return [(p, dam_utilities.select_phase(ds, p)[0]) for p in (PHASE_LD, PHASE_DD)]
    return [("full", ds)]


def _per_epoch(ds, make) -> pd.DataFrame:
    frames = []
    for label, view in epochs(ds):
        df = make(view)
        if not df.empty:
            df.insert(0, "Phase", label)
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def sleep_summary(ds: xr.Dataset) -> pd.DataFrame:
    """Per fly and epoch: minutes of sleep in an average day, all day / day / night."""
    return _per_epoch(ds, lambda v: plotting.per_fly_summary_table(v, "sleep"))


def sleep_states(ds: xr.Dataset) -> pd.DataFrame:
    """Per fly and epoch: minutes in short, intermediate and long sleep."""
    return _per_epoch(ds, plotting.per_fly_sleep_state_totals)


def sleep_bouts(ds: xr.Dataset) -> pd.DataFrame:
    """Every sleep bout of every fly."""
    return sleep_analysis.raw_bout_dataframe(ds)


def hmm_occupancy(ds: xr.Dataset) -> pd.DataFrame:
    """Per fly: % of the fitted minutes spent in each HMM state."""
    from clockwork.core.hmm_models import STATE_NAMES_4, load_hmm_config_from_attrs

    cfg = load_hmm_config_from_attrs(ds)
    states = ds["hmm_state"].transpose("id", ...).values
    n_states = cfg.n_states if cfg is not None else int(np.nanmax(states)) + 1
    names = list(STATE_NAMES_4[:n_states]) if n_states <= 4 else [f"S{i}" for i in range(n_states)]
    group_of = dam_utilities.fly_group_map(ds, default="All Flies")
    rows = []
    for fly, row in zip(ds["id"].values, states):
        valid = row[row != -1]
        if valid.size == 0:
            continue
        rows.append(
            {"ID": str(fly), "Group": group_of.get(fly, "All Flies")}
            | {name: float((valid == s).mean() * 100) for s, name in enumerate(names)}
        )
    df = pd.DataFrame(rows, columns=["ID", "Group", *names])
    return df.sort_values(["Group", "ID"]).reset_index(drop=True)


def hmm_states(ds: xr.Dataset) -> pd.DataFrame:
    """The decoded state of every fly at every minute (-1 = not fitted)."""
    df = ds["hmm_state"].transpose("time", "id").to_pandas().reset_index()
    return df.rename(columns={df.columns[0]: "time"})


def hmm_zt_fractions(ds: xr.Dataset, bin_size_minutes: int = 30) -> pd.DataFrame:
    """Per fly and ZT bin: the fraction of time in each state."""
    from clockwork.core.hmm_models import get_hmm_zt_fractions

    df = get_hmm_zt_fractions(ds, bin_size_minutes=bin_size_minutes)
    if not df.empty:
        group_of = dam_utilities.fly_group_map(ds, default="All Flies")
        df["group"] = df["id"].map(group_of)
    return df


TABLES = {
    "period_summary": period_summary,
    "sleep_summary": sleep_summary,
    "sleep_states": sleep_states,
    "sleep_bouts": sleep_bouts,
    "hmm_occupancy": hmm_occupancy,
    "hmm_states": hmm_states,
    "hmm_zt_fractions": hmm_zt_fractions,
}


def available(ds: xr.Dataset) -> list[str]:
    """The tables this dataset holds the results for."""
    have = set(ds.data_vars)
    out = []
    if have & {"ls_period", "ac_period", "cwt_period", "mesa_period"}:
        out.append("period_summary")
    if "sleep" in have:
        out.append("sleep_summary")
        if "sleep_long" in have:
            out.append("sleep_states")
        if "duration" in have:
            out.append("sleep_bouts")
    if "hmm_state" in have:
        out += ["hmm_occupancy", "hmm_states", "hmm_zt_fractions"]
    return out
