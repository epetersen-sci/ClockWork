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

    def test_annotation_matches_the_actual_return(self):
        """The annotation still said `-> xr.Dataset` after the return became a
        2-tuple, which is how the early-return bug below went unnoticed."""
        import inspect

        import periodograms

        ann = inspect.signature(periodograms.wavelet_analysis).return_annotation
        assert "tuple" in str(ann), f"annotation is {ann!r}, but the function returns a 2-tuple"

    def test_all_flies_failed_still_returns_a_2_tuple(self, master_ds):
        """The `if not results` path returned a BARE Dataset while the success
        path returned a tuple. That is not a clean crash: unpacking an
        xr.Dataset iterates its data_vars, so a dataset with exactly two of them
        silently binds two variable-NAME strings, and any other count raises.

        Triggered by demanding more days than the record holds, so every fly is
        filtered out and no result survives.
        """
        import periodograms

        ds = master_ds.drop_vars(
            [v for v in ("sleep", "sleep_short", "sleep_intermediate", "sleep_long")
             if v in master_ds.data_vars]
        )
        out = periodograms.wavelet_analysis(ds, min_num_days=999, compute_group_averages=True)
        assert isinstance(out, tuple) and len(out) == 2, (
            f"expected a 2-tuple from the all-failed path, got {type(out).__name__}"
        )
        returned_ds, averages = out
        assert averages == []
        assert hasattr(returned_ds, "data_vars"), "first element should be the Dataset"


class TestMonitorReportOrdering:
    """Monitors are scanned by eye, so they must be listed in numeric order.

    `key=str` put monitor 10 before monitor 2 — the same string-vs-numeric trap
    behind the metadata mispairing in tests/test_monitor_label_pairing.py.
    """

    @staticmethod
    def _processor(report):
        from dam_processor import MetadataProcessor

        obj = MetadataProcessor.__new__(MetadataProcessor)
        obj.integrity_report = report
        return obj

    @staticmethod
    def _monitor():
        return {
            "status": {"n_status_bad": 2, "n_rows_dropped": 2, "bad_status_counts": {51: 2}},
            "scan": {"spans": []},
            "classification": {"n_cosmetic_slots": 2, "n_dataloss_slots": 0},
        }

    def test_integer_ids_sort_numerically(self):
        p = self._processor({m: self._monitor() for m in (2, 10, 3, 11)})
        assert [m for m, _, _ in p.integrity_monitor_reports()] == [2, 3, 10, 11]

    def test_numeric_string_ids_sort_numerically(self):
        p = self._processor({m: self._monitor() for m in ("2", "10", "3")})
        assert [m for m, _, _ in p.integrity_monitor_reports()] == ["2", "3", "10"]

    def test_non_numeric_ids_do_not_raise(self):
        """Comparing an int against a str would raise; the sort key must keep the
        two kinds apart rather than assuming every monitor id is a number."""
        p = self._processor({2: self._monitor(), "A12": self._monitor(), 10: self._monitor()})
        assert [m for m, _, _ in p.integrity_monitor_reports()] == [2, 10, "A12"]


class TestRidgeDensityTakesAFilteredDataset:
    def test_no_longer_filters_internally(self):
        import inspect

        params = inspect.signature(plotting.group_ridge_density_plotly).parameters
        for gone in ("filter_nonrhythmic", "filter_gate"):
            assert gone not in params, (
                f"{gone} made the plot function run rhythmicity_classification; "
                "the caller should pass an already-filtered dataset"
            )


class TestCategoryTickAngle:
    """Genotype names on an x axis have to stay readable.

    Two figures already rotated their labels, but only above six groups. Count
    alone is the wrong test: six labels like "dsOpa1(32358)+Ldhmut" overprint
    each other into a smear at 0 degrees — the exact case rotation exists for —
    while six labels like "ctrl" need none. The rule now also looks at how long
    the longest label is.
    """

    REAL_GENOTYPES = [
        "dsOpa1(32358)",
        "dsOpa1(32358)+Ldhmut",
        "dsOpa1(67159)",
        "dsOpa1(67159)+Ldhmut",
        "dsmcherry",
        "dsmcherry+Ldhmut",
    ]

    def test_six_long_names_rotate(self):
        """The reported case: six groups, so the old count-only rule said no."""
        assert plotting.category_tickangle(self.REAL_GENOTYPES) < 0

    def test_few_short_names_stay_flat(self):
        assert plotting.category_tickangle(["ctrl-25C", "mut-25C"]) == 0
        assert plotting.category_tickangle(["All Flies"]) == 0

    def test_many_short_names_still_rotate(self):
        assert plotting.category_tickangle([f"g{i}" for i in range(8)]) < 0

    def test_stacked_labels_measure_the_widest_line(self):
        """Some x labels stack two facts with <br>; the width is the widest
        LINE, not the total string, or every stacked label would rotate."""
        assert plotting.category_tickangle(["Short<br>ctrl", "Long<br>mut"]) == 0
        assert (
            plotting.category_tickangle(
                ["Short sleep<br>dsOpa1(32358)+Ldhmut", "Long sleep<br>dsmcherry"]
            )
            < 0
        )

    def test_rotated_axes_claim_their_margin(self):
        """Rotating without automargin trades an overlapped axis for a clipped
        one — the label is drawn into whatever bottom margin already existed."""
        import plotly.graph_objects as go

        fig = plotting.apply_category_ticks(go.Figure(), self.REAL_GENOTYPES)
        assert fig.layout.xaxis.tickangle < 0
        assert fig.layout.xaxis.automargin is True

    def test_every_group_axis_figure_uses_it(self):
        """The rule is only worth having if it is applied everywhere, so this
        pins the list of figures that put group names on an x axis."""
        import inspect

        source = inspect.getsource(plotting)
        for fn in (
            "sleep_state_totals_bars",
            "summary_bars",
            "rebound_bar_plot",
            "rhythmicity_violin_grid",
            "threshold_coupled_figure",
        ):
            body = source.split(f"def {fn}(", 1)[1].split("\ndef ", 1)[0]
            assert "apply_category_ticks" in body, (
                f"{fn} puts group names on its x axis but does not rotate them"
            )

