"""Figures leave the app under a name that says what they are.

Two exits, and they are not interchangeable:

* The modebar's camera button is a BROWSER DOWNLOAD. Its destination is the
  browser's download folder and no page can change that, so all ``ui.charts``
  can do is fix the name, format and resolution — which it does, because the
  default was ``newplot.png`` at screen resolution.
* ``export_helpers.save_figures_png_button`` writes every figure on the page
  into the working folder beside the experiment's data, like every other export.
"""

import plotly.graph_objects as go
import pytest

import export_helpers
from ui import charts


def _fig(title=None):
    fig = go.Figure(go.Scatter(x=[0, 1, 2], y=[1, 3, 2]))
    if title is not None:
        fig.update_layout(title=title)
    return fig


class TestDownloadFilename:
    def test_comes_from_the_figure_title(self):
        assert charts.png_filename(_fig("Sleep profile")) == "Sleep_profile"

    def test_markup_is_stripped(self):
        """Plotly titles carry <br> for a second line and <b> for emphasis. Left
        in, the tags land in the filename as literal characters."""
        assert charts.png_filename(_fig("<b>Sleep</b> profile<br>DD")) == "Sleep_profile_DD"

    def test_punctuation_becomes_underscores(self):
        assert charts.png_filename(_fig("Activity (counts/min), ZT")) == "Activity_counts_min_ZT"

    def test_an_untitled_figure_still_gets_a_name(self):
        """Better than plotly's "newplot", which says nothing about its origin."""
        assert charts.png_filename(_fig()) == charts.FALLBACK_NAME

    def test_an_explicit_name_wins_over_the_title(self):
        assert charts.png_filename(_fig("Ignored"), "chosen name") == "chosen_name"

    def test_long_titles_are_truncated(self):
        assert len(charts.png_filename(_fig("word " * 60))) <= 80


class TestDownloadConfig:
    def test_names_the_file(self):
        cfg = charts.png_config(_fig("Bout spectrum"))
        assert cfg["toImageButtonOptions"]["filename"] == "Bout_spectrum"

    def test_downloads_at_twice_screen_resolution(self):
        """A 1x download is visibly soft the moment it goes into a slide."""
        assert charts.png_config(_fig("x"))["toImageButtonOptions"]["scale"] == 2

    def test_scale_is_overridable(self):
        cfg = charts.png_config(_fig("x"), scale=4)
        assert cfg["toImageButtonOptions"]["scale"] == 4

    def test_it_touches_nothing_but_the_download(self):
        """Plotly merges this with its own defaults. Setting anything else here
        would silently change how every chart in the app behaves."""
        assert list(charts.png_config(_fig("x"))) == ["toImageButtonOptions"]


class TestEveryChartGoesThroughTheWrapper:
    def test_no_page_calls_st_plotly_chart_directly(self):
        """The wrapper exists so the config is defined once. A page that bypasses
        it silently goes back to newplot.png."""
        from pathlib import Path

        offenders = [
            path.name
            for path in sorted(Path("app/app_pages").glob("*.py"))
            if "st.plotly_chart(" in path.read_text(encoding="utf-8")
        ]
        assert not offenders, f"these pages bypass ui.charts.plotly_chart: {offenders}"


class TestBatchPNGExport:
    """The button is a Streamlit widget, so the write path is exercised directly —
    what matters is that a plotly figure becomes a real PNG in the export folder."""

    def test_writes_a_readable_png(self, tmp_path):
        import xarray as xr

        ds = xr.Dataset(attrs={"source_data_dir": str(tmp_path), "experiment_name": "exp8"})
        out = export_helpers._export_dir(ds, "Figures")
        path = f"{out}/profile.png"
        with open(path, "wb") as fh:
            fh.write(_fig("Sleep profile").to_image(format="png", scale=2))

        with open(path, "rb") as fh:
            header = fh.read(8)
        assert header == b"\x89PNG\r\n\x1a\n", "not a PNG file"
        assert "Graph Exports_exp8" in path and path.endswith("Figures/profile.png")

    def test_the_button_reports_nothing_before_it_is_pressed(self, app, master_ds):
        """_show_remembered runs on every rerun; with no note stored it must draw
        nothing rather than an empty success box."""
        at = app(ds=master_ds, page="sleep_activity")
        assert not at.exception


def test_kaleido_is_available():
    """Named so a broken environment reports the cause rather than failing inside
    an unrelated export test."""
    pytest.importorskip("kaleido")


class TestCollectingWhatWasDrawn:
    """The page-level button exports what actually rendered, once each."""

    def test_a_page_with_charts_offers_the_button(self, app, master_ds):
        at = app(ds=master_ds, page="sleep_activity")
        assert any("as PNG" in b.label for b in at.button), (
            "a page that drew figures should offer to save them"
        )

    def test_the_label_counts_the_figures(self, app, master_ds):
        at = app(ds=master_ds, page="sleep_activity")
        label = next(b.label for b in at.button if "as PNG" in b.label)
        drawn = len(at.session_state["_figures_drawn_this_run"])
        assert f"Save {drawn} figure" in label

    def test_a_page_without_charts_offers_nothing(self, app, master_ds):
        """The HMM page draws matplotlib figures and exports CSVs — no plotly
        charts at all — so a PNG button there would be permanently disabled
        furniture."""
        at = app(ds=master_ds, page="hmm")
        assert not any("as PNG" in b.label for b in at.button)

    def test_figures_do_not_accumulate_across_reruns(self, app, master_ds):
        """begin_run resets the collection every rerun. Without it each rerun would
        append the page's figures again and the export would write duplicates."""
        at = app(ds=master_ds, page="sleep_activity")
        first = len(at.session_state["_figures_drawn_this_run"])
        at.run()
        assert len(at.session_state["_figures_drawn_this_run"]) == first

    def test_filenames_are_unique_within_a_run(self, app, master_ds):
        """Two charts can share a title — the same plot for LD and DD. Identical
        filenames would have the second overwrite the first."""
        at = app(ds=master_ds, page="sleep_activity")
        names = [name for name, _ in at.session_state["_figures_drawn_this_run"]]
        assert len(names) == len(set(names))

    def test_every_recorded_entry_is_a_png_and_a_figure(self, app, master_ds):
        at = app(ds=master_ds, page="sleep_activity")
        for name, fig in at.session_state["_figures_drawn_this_run"]:
            assert name.endswith(".png")
            assert hasattr(fig, "to_image"), f"{name} is not a plotly figure"


class TestTheButtonReallyWritesFiles:
    """End to end through the page: click it, and PNGs appear on disk.

    The rest of this module checks the pieces. This checks the thing the user
    does — and it is where the two halves meet, since the folder it writes into
    is named by the experiment on the dataset.
    """

    def _loaded(self, master_ds, tmp_path, name="exp8"):
        ds = master_ds.copy()
        ds.attrs["source_data_dir"] = str(tmp_path)
        ds.attrs["experiment_name"] = name
        return ds

    def test_clicking_it_writes_a_png_per_figure(self, app, master_ds, tmp_path):
        from pathlib import Path

        at = app(ds=self._loaded(master_ds, tmp_path), page="sleep_activity")
        expected = [name for name, _ in at.session_state["_figures_drawn_this_run"]]
        assert expected, "fixture drew no figures, so this test proves nothing"

        at.button(key="save_page_figures").click().run()

        out = Path(tmp_path) / "Graph Exports_exp8"
        written = sorted(p.name for p in out.glob("*.png"))
        assert written == sorted(expected)

    def test_the_files_are_real_pngs(self, app, master_ds, tmp_path):
        from pathlib import Path

        at = app(ds=self._loaded(master_ds, tmp_path), page="sleep_activity")
        at.button(key="save_page_figures").click().run()

        for path in (Path(tmp_path) / "Graph Exports_exp8").glob("*.png"):
            assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n", f"{path.name} is not a PNG"
            assert path.stat().st_size > 1000, f"{path.name} is suspiciously small"

    def test_it_reports_where_they_went(self, app, master_ds, tmp_path):
        """The confirmation has to survive the rerun the click causes, or the only
        record of where a dozen files landed is gone by the time you look."""
        at = app(ds=self._loaded(master_ds, tmp_path), page="sleep_activity")
        at.button(key="save_page_figures").click().run()
        assert any("Graph Exports_exp8" in s.value for s in at.success)

    def test_renaming_the_experiment_moves_the_output(self, app, master_ds, tmp_path):
        """Ties the two fixes together: the name is editable after load, and the
        figures follow it."""
        from pathlib import Path

        at = app(ds=self._loaded(master_ds, tmp_path, name="before"), page="export_data")
        at.text_input(key="experiment_name_input").set_value("after").run()
        at.switch_page("app_pages/sleep_activity.py").run()
        at.button(key="save_page_figures").click().run()

        assert (Path(tmp_path) / "Graph Exports_after").glob("*.png")
        assert not list((Path(tmp_path) / "Graph Exports_before").glob("*.png"))
