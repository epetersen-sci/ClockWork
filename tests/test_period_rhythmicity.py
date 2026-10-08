"""Period & rhythmicity: one page, five tabs, one set of controls.

It was three pages (Period analysis, Rhythmicity, Periodograms). What the merge
has to keep: every tab renders on a dataset with and without results, the four
controls sit in one row, Analysis runs exactly the ticked methods, and Classify
writes the flag for the chosen algorithm at the slider's value.
"""

import json

import numpy as np
import pytest

PAGE = "period_rhythmicity"
TAB_KEY = "period_rhythm_tab"
TABS = ["Analysis", "Rhythmicity cutoff", "Period length", "Rhythmicity", "Periodograms"]


def _with_results(ds):
    """``ds`` with stored AC and LS results: periods, strengths and spectra."""
    n = ds.sizes["id"]
    temps = np.asarray([str(t) for t in ds["temperature"].values])
    period = np.where(temps == "29C", 26.0, 24.0) + np.linspace(0, 0.5, n)
    lags = np.arange(0.0, 72.0, 1.0)
    grid = np.linspace(16.0, 36.0, 41)
    out = ds.assign(
        ac_period=("id", period),
        ac_power=("id", np.linspace(0.1, 0.6, n)),
        ls_period=("id", period + 0.1),
        ls_power=("id", np.linspace(0.001, 0.02, n)),
        ls_fap=("id", np.full(n, 0.01)),
        ac_correlogram=(("id", "ac_lag"), np.cos(2 * np.pi * lags[None, :] / period[:, None])),
        ls_periodogram=(
            ("id", "ls_periodogram_periods"),
            np.exp(-((grid[None, :] - period[:, None]) ** 2)),
        ),
    )
    return out.assign_coords(ac_lag=lags, ls_periodogram_periods=grid)


@pytest.mark.parametrize("tab", TABS)
def test_every_tab_renders_without_results(app, master_ds, tab):
    at = app(ds=master_ds, page=PAGE, **{TAB_KEY: tab})
    assert not at.exception, at.exception


@pytest.mark.parametrize("tab", TABS)
def test_every_tab_renders_with_results(app, master_ds, tab):
    at = app(ds=_with_results(master_ds), page=PAGE, **{TAB_KEY: tab})
    assert not at.exception, at.exception


def test_the_four_controls_share_one_row(app, master_ds):
    at = app(ds=master_ds, page=PAGE)
    labels = [n.label for n in at.number_input]
    for want in ("Min period (h)", "Max period (h)", "Min DD days", "Max gap to bridge (min)"):
        assert want in labels
    # Rendered through one st.columns(4): four adjacent columns holding one each.
    want = ["Min period (h)", "Max period (h)", "Min DD days", "Max gap to bridge (min)"]
    per_column = [[n.label for n in col.number_input] for col in at.columns]
    assert any(
        per_column[i : i + 4] == [[w] for w in want] for i in range(len(per_column) - 3)
    ), per_column


def test_analysis_tab_offers_one_checkbox_per_method_and_one_button(app, master_ds):
    at = app(ds=master_ds, page=PAGE)
    for key in ("run_method_cwt", "run_method_ls", "run_method_ac", "run_method_mesa"):
        assert at.checkbox(key=key) is not None
    assert at.button(key="run_period_analysis") is not None


def test_run_analysis_runs_only_the_ticked_methods(app, master_ds, monkeypatch):
    from clockwork.core import periodograms

    called = []

    def _fake(name):
        def _run(ds, **kw):
            called.append(name)
            return _with_results(ds)

        return _run

    monkeypatch.setattr(periodograms, "lomb_scargle_analysis", _fake("ls"))
    monkeypatch.setattr(periodograms, "autocorrelation_analysis", _fake("ac"))
    monkeypatch.setattr(periodograms, "mesa_analysis", _fake("mesa"))
    monkeypatch.setattr(
        periodograms, "wavelet_analysis", lambda ds, **kw: (called.append("cwt"), (ds, None))[1]
    )
    at = app(ds=master_ds, page=PAGE)
    at.checkbox(key="run_method_cwt").uncheck()
    at.checkbox(key="run_method_mesa").uncheck()
    at.run()
    at.button(key="run_period_analysis").click().run()
    assert not at.exception, at.exception
    assert sorted(called) == ["ac", "ls"]
    assert "ac_period" in at.session_state["dataset"].data_vars


def test_classify_applies_the_slider_value(app, master_ds):
    at = app(ds=_with_results(master_ds), page=PAGE, **{TAB_KEY: "Rhythmicity cutoff"})
    assert not at.exception, at.exception
    at.slider(key="expl_thr_ac").set_value(0.4).run()
    at.button(key="classify_rhythmicity").click().run()
    assert not at.exception, at.exception
    ds = at.session_state["dataset"]
    assert "ac_rhythmic" in ds.coords
    assert float(ds.attrs["ac_ri_threshold"]) == pytest.approx(0.4)
    expected = np.asarray(ds["ac_power"].values) > 0.4
    assert (np.asarray(ds["ac_rhythmic"].values, bool) <= expected).all()
    assert "ls_rhythmic" not in ds.coords, "only the chosen algorithm is classified"


def test_classifying_mesa_sets_the_autocorrelation_flag(app, master_ds):
    ds = _with_results(master_ds)
    ds = ds.assign(mesa_period=("id", np.asarray(ds["ac_period"].values)))
    at = app(ds=ds, page=PAGE, **{TAB_KEY: "Rhythmicity cutoff"}, cutoff_algo="mesa")
    assert not at.exception, at.exception
    at.button(key="classify_rhythmicity").click().run()
    assert "ac_rhythmic" in at.session_state["dataset"].coords


def test_rhythmicity_tab_shows_strength_by_condition(app, master_ds):
    at = app(ds=_with_results(master_ds), page=PAGE, **{TAB_KEY: "Rhythmicity"})
    assert not at.exception, at.exception
    titles = [
        json.loads(e.proto.spec)["layout"].get("title", {}).get("text")
        for e in at.get("plotly_chart")
    ]
    assert "Rhythm strength (Autocorrelation)" in titles


def test_periodograms_tab_draws_the_stored_spectra(app, master_ds):
    at = app(ds=_with_results(master_ds), page=PAGE, **{TAB_KEY: "Periodograms"})
    assert not at.exception, at.exception
    titles = [
        json.loads(e.proto.spec)["layout"].get("title", {}).get("text")
        for e in at.get("plotly_chart")
    ]
    assert "Lomb-Scargle — group-averaged" in titles
    assert "Autocorrelation — group-averaged" in titles
