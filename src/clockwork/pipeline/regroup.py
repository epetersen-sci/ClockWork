"""Carrying work across a group change (BACKLOG 22).

The Groups page builds every subset and regroup from the import-time copy of the
dataset, because a subset is chosen from ALL groups and the working copy may
already be narrowed. That copy predates curation, the split and every analysis,
so a group change used to discard them.

:func:`carry_over` brings that work onto the rebuilt dataset instead:

- curation and the split are re-applied with the settings recorded on the
  dataset. Curation decides each fly from its own record alone, so curating
  after the change gives every fly exactly what it had (checked on
  example_data: identical activity, movement and alive mask, minute for minute);
- sleep is re-detected with its recorded settings. It is per fly and takes
  seconds, and re-running it is exact for flies a widened subset brings in;
- period results (estimates, spectra, rhythmic calls) are per fly, so they are
  carried over by fly id, as long as every fly in the new set was analysed.
  A change that brings in flies never analysed drops them instead: a NaN for a
  new fly would read as "could not be analysed", which is not what happened;
- the HMM and the CWT group-averaged scalograms describe groups, not flies (the
  model is fitted per group or pooled), so they do not survive a group change.
"""

from __future__ import annotations

import xarray as xr

from clockwork.core.dataset_meta import is_split_applied
from clockwork.pipeline.data import CurationConfig, SplitConfig, curate, split
from clockwork.pipeline.period import merge_period_outputs
from clockwork.pipeline.sleep import SleepConfig, detect_sleep

_PERIOD_PREFIXES = ("ls_", "ac_", "cwt_", "mesa_")
#: cwt_ attrs that describe groups (the averaged scalograms), not flies.
_GROUP_LEVEL_ATTRS = ("cwt_group_average_paths", "cwt_group_average_dir")


def carry_over(rebuilt: xr.Dataset, current: xr.Dataset) -> tuple[xr.Dataset, list[str]]:
    """``rebuilt`` (a group change of the import copy) with ``current``'s work on it.

    Returns ``(dataset, notes)``: ``notes`` says what was carried and what could
    not be, in words for the person who made the change.
    """
    out, notes = rebuilt, []

    curation = CurationConfig.from_attrs(current.attrs)
    if curation is not None and CurationConfig.from_attrs(out.attrs) is None:
        result = curate(out, curation)
        out = result.live
        notes.append(
            f"Curation re-applied with its settings ({result.removed} flies removed as dead)."
        )

    split_cfg = SplitConfig.from_attrs(current.attrs)
    if split_cfg is not None and not is_split_applied(out):
        out = split(out, split_cfg)
        notes.append("The LD/DD split re-applied.")

    sleep_cfg = SleepConfig.from_attrs(current.attrs)
    if sleep_cfg is not None:
        if "moving" in out.data_vars:
            out = detect_sleep(out, sleep_cfg)
            notes.append("Sleep re-detected with its settings.")
        else:
            notes.append("Sleep was not carried over: it needs curation, which this dataset lacks.")

    out, period_note = _carry_period(out, current)
    if period_note:
        notes.append(period_note)

    lost = []
    if any(str(v).startswith("hmm_") for v in current.data_vars):
        lost.append("the HMM (its model is fitted per group or pooled across flies)")
    if "cwt_group_average_paths" in current.attrs:
        lost.append("the CWT group-averaged scalograms")
    if any(str(k).startswith("sd_") for k in current.attrs):
        lost.append("sleep deprivation")
    if any(str(k).startswith("phase_shift_") for k in current.attrs):
        lost.append("phase shift")
    if lost:
        notes.append("Re-run " + ", ".join(lost) + ": they describe groups, so they do not carry over.")
    return out, notes


def _carry_period(out: xr.Dataset, current: xr.Dataset) -> tuple[xr.Dataset, str | None]:
    names = [
        v
        for v in current.data_vars
        if str(v).startswith(_PERIOD_PREFIXES) and "id" in current[v].dims and "time" not in current[v].dims
    ]
    if not names:
        return out, None
    new_ids = [str(i) for i in out["id"].values]
    analysed = {str(i) for i in current["id"].values}
    unanalysed = [i for i in new_ids if i not in analysed]
    if unanalysed:
        return out, (
            f"Period results were not carried over: {len(unanalysed)} of the flies now "
            "included were never analysed. Re-run period & rhythmicity."
        )
    result = current[names].sel(id=out["id"].values)
    result.attrs = {
        k: v for k, v in current.attrs.items() if str(k).startswith(_PERIOD_PREFIXES) and k not in _GROUP_LEVEL_ATTRS
    }
    return merge_period_outputs(out, result), "Period & rhythmicity results kept for every fly."
