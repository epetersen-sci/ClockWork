"""The custom ZT window on Sleep & activity's totals.

Day and Night are fixed halves of the cycle; the custom window is any span the
user picks. It must total by the same bin-content rule as Day and Night, so a
window of ZT 0–12 reproduces Day exactly, and it must wrap past ZT 24.
"""

import json

import numpy as np

from clockwork.core import plotting


def test_window_mask_selects_bins_starting_inside_it():
    zt = np.arange(0, 1440, 30)
    mask = plotting.zt_window_mask(zt, 3, 6)
    assert zt[mask].tolist() == [180, 210, 240, 270, 300, 330]


def test_window_mask_wraps_past_zt24():
    zt = np.arange(0, 1440, 60)
    assert zt[plotting.zt_window_mask(zt, 22, 2)].tolist() == [0, 60, 1320, 1380]


def test_equal_ends_mean_the_whole_day():
    assert plotting.zt_window_mask(np.arange(0, 1440, 60), 5, 5).all()


def test_custom_zt0_to_12_is_exactly_day(master_ds):
    per_fly = plotting.per_fly_summary_table(master_ds, "activity", custom_window=(0, 12))
    np.testing.assert_allclose(per_fly["Custom"], per_fly["Day Only"])


def test_custom_column_only_when_asked(master_ds):
    assert "Custom" not in plotting.per_fly_summary_table(master_ds, "activity").columns
    tbl = plotting.summary_table(master_ds, "activity", custom_window=(20, 4))
    assert {"Custom", "Custom_sem"} <= set(tbl.columns)


def _custom_titles(at):
    titles = []
    for el in at.get("plotly_chart"):
        title = json.loads(el.proto.spec).get("layout", {}).get("title", {})
        text = title.get("text", "") if isinstance(title, dict) else str(title)
        if "Custom (ZT" in text:
            titles.append(text)
    return titles


def test_page_draws_a_custom_panel_and_updates_only_on_the_button(app, master_ds):
    at = app(ds=master_ds, page="sleep_activity")
    assert not at.exception
    assert any("Custom (ZT 0–6)" in t for t in _custom_titles(at))

    at.number_input(key="act_custom_zt_start").set_value(2.0)
    at.number_input(key="act_custom_zt_end").set_value(5.0)
    at = at.run()
    # Inside a form: nothing changes until Update graph is pressed.
    assert any("Custom (ZT 0–6)" in t for t in _custom_titles(at))

    # AppTest drops a form's pending values at the end of a run, so set them again
    # alongside the click, as the browser would still hold them.
    at.number_input(key="act_custom_zt_start").set_value(2.0)
    at.number_input(key="act_custom_zt_end").set_value(5.0)
    next(b for b in at.button if b.label == "Update graph").click()
    at = at.run()
    assert not at.exception
    assert any("Custom (ZT 2–5)" in t for t in _custom_titles(at))
