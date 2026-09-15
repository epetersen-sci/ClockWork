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
