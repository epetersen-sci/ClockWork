"""Curating a dataset that already carries analysis results.

Reported: period analysis run before curation, then curation failed asking for a
109 GiB array. Curation masks the dataset with the per-fly alive mask, and a
period spectrum carries a period axis, so the mask broadcast every fly x period x
minute. Results computed on uncurated flies cannot be kept anyway — they include
flies curation trims or removes — so curation drops them first.
"""

import numpy as np
import xarray as xr

from clockwork import pipeline
from clockwork.core import dam_utilities


def _with_period_results(ds):
    out = ds.copy()
    n_id = out.sizes["id"]
    out["ls_period"] = ("id", np.full(n_id, 24.0))
    out["ls_power_spectrum"] = (("id", "ls_period_axis"), np.zeros((n_id, 50)))
    out = out.assign_coords(
        ls_period_axis=np.linspace(16, 36, 50), ls_rhythmic=("id", np.ones(n_id, bool))
    )
    out.attrs.update({"ls_min_period": 16.0, "ls_max_period": 36.0, "prep_detrend": "none"})
    return out


def test_drop_analysis_outputs_keeps_activity_and_metadata(unsplit_ds):
    clean, dropped = dam_utilities.drop_analysis_outputs(_with_period_results(unsplit_ds))
    assert list(clean.data_vars) == ["activity"]
    assert "ls_period_axis" not in clean.dims
    assert {"ls_period", "ls_power_spectrum", "ls_rhythmic", "ls_min_period"} <= set(dropped)
    assert "genotype" in clean.coords and "start_datetime" in clean.coords


def test_curation_after_period_analysis_drops_it_instead_of_broadcasting(unsplit_ds):
    ds = _with_period_results(unsplit_ds)
    result = pipeline.curate(ds, pipeline.CurationConfig())
    live = result.live
    assert "ls_period" not in live.data_vars and "ls_rhythmic" not in live.coords
    # No variable picked up the (id, time) mask's dims on top of its own.
    assert all(set(live[v].dims) <= {"id", "time"} for v in live.data_vars)
    assert not any(str(k).startswith("ls_") for k in live.attrs)
    assert "ls_period" in result.dropped and "moving" not in result.dropped
    assert isinstance(live["is_alive"], xr.DataArray)


def test_curate_page_warns_before_dropping_results(app, unsplit_ds):
    ds = _with_period_results(unsplit_ds)
    at = app(ds=ds, page="data_curate_split")
    assert not at.exception
    assert any("Lomb-Scargle" in w.value and "removes them" in w.value for w in at.warning)


def test_inspect_defaults_to_moving_before_curation(app, unsplit_ds):
    ds = unsplit_ds.drop_vars([v for v in ("moving", "is_alive") if v in unsplit_ds])
    at = app(ds=ds, page="data_curate_split")
    assert not at.exception
    assert at.radio(key="heatmap_var_choice").value == "moving"
