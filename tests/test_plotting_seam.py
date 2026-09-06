"""core/plotting.py renders; it does not run analyses (backlog item 7).

Three functions used to defer-import analysis code specifically to dodge
circular imports, which is the tell that the compute/render seam was in the
wrong place — plotting was running the analysis. A fourth inversion went the
other way: ``periodograms.wavelet_analysis`` imported ``plotting`` to write PNGs,
so an analysis function could not be called without also deciding where files go.

The import-level tests are the load-bearing ones. A deferred import is invisible
until the moment it runs, so only checking the source keeps it from creeping back.
"""

import ast
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
from conftest import REPO_ROOT

import plotting

ANALYSIS_MODULES = {"sleep_analysis", "rhythmicity_classification", "periodograms"}


def _function_level_imports(path):
    """Every import that sits inside a function body, as (function, module)."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for inner in ast.walk(node):
            if isinstance(inner, ast.ImportFrom) and inner.module:
                found.append((node.name, inner.module.split(".")[0]))
            elif isinstance(inner, ast.Import):
                found.extend((node.name, a.name.split(".")[0]) for a in inner.names)
    return found


class TestNoDeferredAnalysisImports:
    def test_plotting_has_none(self):
        offenders = [
            (fn, mod)
            for fn, mod in _function_level_imports(REPO_ROOT / "core" / "plotting.py")
            if mod in ANALYSIS_MODULES
        ]
        assert not offenders, (
            f"plotting.py defer-imports analysis code in {offenders}. A deferred "
            "import to dodge a circular one means the analysis is being run by "
            "the plotting module."
        )

    def test_periodograms_does_not_import_plotting(self):
        src = (REPO_ROOT / "core" / "periodograms.py").read_text(encoding="utf-8")
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("plotting"):
                pytest.fail("periodograms.py imports plotting; the caller should render")
            if isinstance(node, ast.Import):
                assert not any(a.name.startswith("plotting") for a in node.names)

    def test_plotting_imports_first_in_a_clean_interpreter(self):
        """The real test of the seam. If a circular dependency remained, importing
        plotting before anything else would fail at module scope."""
        code = (
            "import sys; sys.path[:0]=['core','app']; "
            "import plotting, sleep_analysis, rhythmicity_classification, periodograms; "
            "print('ok')"
        )
        proc = subprocess.run(
            [sys.executable, "-c", code],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=180,
        )
        assert proc.returncode == 0, proc.stderr[-1500:]
        assert "ok" in proc.stdout


class TestBoutDurationLinesTakesFrames:
    """It renders what it is given, and computes nothing."""

    @pytest.fixture
    def curves(self):
        rows = []
        for group in ("a", "b"):
            for fly in range(3):
                for x in np.linspace(0, 2, 8):
                    rows.append(
                        {
                            "id": f"{group}{fly}",
                            "group": group,
                            "x": float(x),
                            "y": float(x) + (0.5 if group == "b" else 0.0),
                        }
                    )
        return pd.DataFrame(rows)

    def test_renders_a_figure_from_a_frame(self, curves):
        fig = plotting.sleep_bout_duration_lines(curves)
        assert fig.data, "expected traces"

    def test_empty_frame_gives_the_placeholder_not_a_crash(self):
        fig = plotting.sleep_bout_duration_lines(pd.DataFrame(columns=["id", "group", "x", "y"]))
        assert not fig.data
        assert len(fig.layout.annotations) == 1

    def test_none_is_tolerated(self):
        assert not plotting.sleep_bout_duration_lines(None).data

    def test_stats_drive_the_annotation(self, curves):
        without = plotting.sleep_bout_duration_lines(curves)
        with_stats = plotting.sleep_bout_duration_lines(
            curves, {"test": "anova", "pvalue": 0.0004}
        )
        assert not [a.text for a in without.layout.annotations if a.text]
        assert "p<0.001" in [a.text for a in with_stats.layout.annotations if a.text][0]

    def test_nan_pvalue_skips_the_annotation(self, curves):
        fig = plotting.sleep_bout_duration_lines(
            curves, {"test": "anova", "pvalue": float("nan")}
        )
        assert not [a.text for a in fig.layout.annotations if a.text]

    def test_method_selects_the_axes(self, curves):
        kde = plotting.sleep_bout_duration_lines(curves, method="kde")
        surv = plotting.sleep_bout_duration_lines(curves, method="survival")
        assert kde.layout.yaxis.title.text != surv.layout.yaxis.title.text
        assert "P(bout duration > t)" in surv.layout.yaxis.title.text


class TestWaveletAnalysisReturnsAverages:
    def test_signature_no_longer_takes_an_output_dir(self):
        import inspect

        import periodograms

        params = inspect.signature(periodograms.wavelet_analysis).parameters
        assert "average_output_dir" not in params, (
            "an analysis function should not decide where files go"
        )
        assert "compute_group_averages" in params

    def test_documents_the_two_part_return(self):
        import inspect

        import periodograms

        doc = inspect.getdoc(periodograms.wavelet_analysis)
        assert "group_averages" in doc


class TestRidgeDensityTakesAFilteredDataset:
    def test_no_longer_filters_internally(self):
        import inspect

        params = inspect.signature(plotting.group_ridge_density_plotly).parameters
        for gone in ("filter_nonrhythmic", "filter_gate"):
            assert gone not in params, (
                f"{gone} made the plot function run rhythmicity_classification; "
                "the caller should pass an already-filtered dataset"
            )
