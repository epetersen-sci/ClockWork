"""Surfacing the data-integrity report (backlog item 10).

``core/dam_integrity.py`` is ~490 lines of load-time quality analysis whose
output mostly went to the console and nowhere else. Two gaps: the per-monitor
detail — the half that says WHERE a gap is — was invisible to anyone not running
the app from a terminal, and a reloaded ``.nc`` could say nothing at all about
the quality of the recording behind it.
"""

import numpy as np
import pytest
import xarray as xr

from dam_processor import MetadataProcessor


class _FakeProcessor(MetadataProcessor):
    """A processor with a hand-built integrity report and no files behind it.

    Subclassed rather than constructed because ``__init__`` reads a metadata
    file; the accessors under test only touch ``self.integrity_report``.
    """

    def __init__(self, report):
        self.integrity_report = report


def _monitor(n_status_bad=0, n_cosmetic=0, n_dataloss=0):
    return {
        "status": {
            "n_status_bad": n_status_bad,
            "n_rows_dropped": n_cosmetic,
            "bad_status_counts": {51: n_status_bad} if n_status_bad else {},
        },
        "scan": {"spans": []},
        "classification": {
            "n_cosmetic_slots": n_cosmetic,
            "n_dataloss_slots": n_dataloss,
        },
    }


class TestIntegrityScalars:
    def test_empty_report_gives_an_empty_dict(self):
        """So a caller can attrs.update(...) unconditionally."""
        assert _FakeProcessor({}).integrity_scalars() == {}

    def test_sums_across_monitors(self):
        p = _FakeProcessor(
            {
                17: _monitor(n_status_bad=3, n_cosmetic=2, n_dataloss=1),
                18: _monitor(n_status_bad=5, n_cosmetic=0, n_dataloss=4),
            }
        )
        s = p.integrity_scalars()
        assert s["integrity_n_status_bad"] == 8
        assert s["integrity_n_cosmetic_slots"] == 2
        assert s["integrity_n_dataloss_slots"] == 5
        assert s["integrity_n_monitors"] == 2

    def test_values_are_plain_ints(self):
        """NetCDF serializes plain scalars cleanly; numpy types and nested blobs
        are the thing this deliberately avoids."""
        p = _FakeProcessor({17: _monitor(n_status_bad=np.int64(3))})
        for key, value in p.integrity_scalars().items():
            assert type(value) is int, f"{key} is {type(value).__name__}, not int"

    def test_survives_a_netcdf_roundtrip(self, master_ds, tmp_path):
        """The point of using scalars: a reloaded file still reports its quality."""
        ds = master_ds.copy()
        ds.attrs.update(
            _FakeProcessor({17: _monitor(n_status_bad=3, n_cosmetic=2, n_dataloss=1)}).integrity_scalars()
        )
        path = tmp_path / "roundtrip.nc"
        ds.to_netcdf(path)
        reloaded = xr.load_dataset(path)
        assert int(reloaded.attrs["integrity_n_status_bad"]) == 3
        assert int(reloaded.attrs["integrity_n_dataloss_slots"]) == 1


class TestPerMonitorReports:
    def test_clean_monitors_are_omitted(self):
        """The expander should show what needs attention, not a roll-call."""
        p = _FakeProcessor({17: _monitor(), 18: _monitor()})
        assert p.integrity_monitor_reports() == []

    def test_data_loss_is_a_warning_cosmetic_is_info(self):
        p = _FakeProcessor(
            {
                17: _monitor(n_status_bad=2, n_cosmetic=2),
                18: _monitor(n_status_bad=2, n_cosmetic=1, n_dataloss=3),
            }
        )
        severities = {mon: sev for mon, sev, _ in p.integrity_monitor_reports()}
        assert severities[17] == "info"
        assert severities[18] == "warning"

    def test_names_the_monitor(self):
        """The whole point is saying WHERE."""
        p = _FakeProcessor({17: _monitor(n_status_bad=2, n_cosmetic=2)})
        reports = p.integrity_monitor_reports()
        assert reports and "17" in reports[0][2]

    def test_sorted_by_monitor(self):
        p = _FakeProcessor(
            {m: _monitor(n_status_bad=1, n_cosmetic=1) for m in (22, 17, 19)}
        )
        assert [m for m, _, _ in p.integrity_monitor_reports()] == [17, 19, 22]


class TestCounterRendering:
    """``ui.status.render_integrity_counters`` — the reload-side half."""

    def test_silent_when_the_dataset_predates_the_counters(self, app, master_ds):
        """Absence means "not recorded", which is NOT "clean" and must not be
        reported as such."""
        from ui.status import render_integrity_counters

        ds = master_ds.copy()
        for key in list(ds.attrs):
            if key.startswith("integrity_"):
                del ds.attrs[key]
        assert render_integrity_counters(ds) is False

    @pytest.mark.parametrize(
        ("attrs", "expected"),
        [
            ({"integrity_n_status_bad": 0, "integrity_n_dataloss_slots": 0}, True),
            ({"integrity_n_status_bad": 5, "integrity_n_dataloss_slots": 2}, True),
        ],
    )
    def test_renders_when_counters_are_present(self, master_ds, attrs, expected):
        from ui.status import render_integrity_counters

        ds = master_ds.copy()
        ds.attrs.update(attrs)
        assert render_integrity_counters(ds) is expected


class TestPagesRenderWithCounters:
    def test_groups_page_handles_a_dataset_carrying_counters(self, app, master_ds):
        ds = master_ds.copy()
        ds.attrs.update(
            {
                "integrity_n_status_bad": 7,
                "integrity_n_cosmetic_slots": 3,
                "integrity_n_dataloss_slots": 4,
                "integrity_n_monitors": 2,
            }
        )
        at = app(ds=ds, page="data_groups")
        assert not at.exception

    def test_groups_page_handles_a_dataset_without_them(self, app, master_ds):
        at = app(ds=master_ds, page="data_groups")
        assert not at.exception
