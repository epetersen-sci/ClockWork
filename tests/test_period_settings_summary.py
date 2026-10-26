"""What the Analysis tab says a stored period method was run with.

Returning to a dataset, the settings behind each stored result have to be
readable without opening every option. The tab lists only the ones moved off
their defaults, or says plainly that none were.
"""

import xarray as xr

from clockwork.app.period_views import analysis_tab

_LS_DEFAULTS = {
    "ls_min_period": 16.0,
    "ls_max_period": 36.0,
    "ls_phase": "DD",
    "ls_min_num_days": 4.0,
    "ls_oversampling": 8,
    "ls_fap_method": "baluev",
    "ls_prep_bin_minutes": 0,
    "ls_prep_smooth_sigma_min": 0.0,
    "ls_prep_lopass_hours": 0.0,
    "ls_prep_detrend": "none",
    "ls_prep_normalize": "none",
    "ls_prep_rolling_window_h": 24.0,
}


def _ds(**attrs):
    return xr.Dataset(attrs={**_LS_DEFAULTS, **attrs})


def test_all_defaults_is_said_in_words():
    assert analysis_tab._changed_settings(_ds(), "ls") == []
    assert "all settings at defaults" in analysis_tab._stored_summary(_ds(), "ls")


def test_only_the_changed_settings_are_listed():
    ds = _ds(ls_oversampling=16, ls_max_period=48.0, ls_prep_normalize="zscore")
    changed = dict(analysis_tab._changed_settings(ds, "ls"))
    assert changed == {
        "period range (h)": "16–48",
        "oversampling": "16",
        "preprocessing normalize": "zscore",
    }


def test_a_method_never_run_has_no_summary():
    assert analysis_tab._changed_settings(_ds(), "cwt") is None


def test_flag_marks_only_a_moved_value():
    from clockwork.app.ui.changed import is_changed

    assert not is_changed(8, 8.0)
    assert is_changed("naive", "baluev")
    assert is_changed(False, True)
