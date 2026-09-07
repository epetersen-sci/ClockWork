"""The gap trim must mask every per-fly series on the same minutes (item 16).

``_select_longest_segments`` masked with ``arr[:start, fly_idx]`` — positional
``[time, fly]``. Dimension order is not uniform here: ``activity`` and ``moving``
are ``(time, id)`` but the sleep masks are ``(id, time)``, so for those it
blanked the wrong axis. Measured on a real dataset with one 180-minute gap
punched into monitor 17 (31 flies trimmed):

    before:  trimmed fly  sleep  211 missing  vs  moving 1080 missing
             untrimmed fly sleep  31 missing  (should be 0)
    after:   trimmed fly  sleep 1080 missing  vs  moving 1080 missing
             untrimmed fly sleep   0 missing

So ~869 minutes per trimmed fly that the trim exists to discard were kept, and
every fly — trimmed or not — picked up one spurious missing marker per trimmed
fly. It reached the per-phase ``.nc``; the SCAMP export was unaffected, because
that writes ``activity``, which had its own correctly-shaped block.

These tests assert the invariant rather than the numbers: whatever the trim
discards, it discards from ALL of a fly's series alike.
"""

import numpy as np
import pytest

import export_helpers

SLEEP_VARS = ("sleep", "sleep_short", "sleep_intermediate", "sleep_long")


def _missing_count(da, fly):
    """Missing cells for one fly, whatever this var's sentinel and dim order."""
    values = da.transpose("id", "time").sel(id=fly).values
    if np.issubdtype(values.dtype, np.integer):
        return int((values == -1).sum())
    return int(np.isnan(values).sum())


@pytest.fixture
def gapped(master_ds):
    """One fly given a 1200-minute hole inside the DD epoch, in both dim orders."""
    ds = master_ds.copy(deep=True)
    hole = dict(time=slice(4000, 5200))
    ds["activity"][:, 0].loc[hole] = np.nan
    ds["moving"][:, 0].loc[hole] = np.nan
    return ds, str(ds["id"].values[0])


class TestTrimmedFly:
    def test_sleep_loses_the_same_minutes_as_moving(self, gapped):
        """The headline invariant. `moving` was always masked correctly, so it is
        the reference for what the trim decided to discard."""
        ds, trimmed = gapped
        out = export_helpers.phase_slice(ds, "DD")
        moving_missing = _missing_count(out["moving"], trimmed)
        assert moving_missing > 0, "fixture precondition: the fly must be trimmed"
        for var in SLEEP_VARS:
            if var in out.data_vars:
                assert _missing_count(out[var], trimmed) == moving_missing, (
                    f"{var} kept data the trim discarded from moving — the "
                    f"(id, time) vars are being masked on the wrong axis"
                )


class TestUntrimmedFlies:
    def test_gain_no_spurious_missing_markers(self, gapped):
        """The other half of the bug: flies with no gap were being marked missing
        at low time indices, once per trimmed fly."""
        ds, trimmed = gapped
        out = export_helpers.phase_slice(ds, "DD")
        for fly in [str(f) for f in out["id"].values if str(f) != trimmed]:
            for var in SLEEP_VARS:
                if var in out.data_vars:
                    assert _missing_count(out[var], fly) == 0, (
                        f"{fly} was not trimmed but {var} has missing cells"
                    )


class TestDtypesAndDims:
    """The masking writes each dtype's own sentinel, so nothing round-trips
    through float — which is why phase_slice no longer repairs int8."""

    def test_sleep_masks_stay_int8(self, gapped):
        ds, _ = gapped
        out = export_helpers.phase_slice(ds, "DD")
        for var in SLEEP_VARS:
            if var in out.data_vars:
                assert out[var].dtype == np.int8, f"{var} upcast during the trim"

    def test_float_vars_stay_float32(self, gapped):
        ds, _ = gapped
        out = export_helpers.phase_slice(ds, "DD")
        for var in ("activity", "moving"):
            assert out[var].dtype == np.float32

    def test_dimension_order_is_preserved(self, gapped):
        """xr.where broadcasts to a canonical order; the transpose puts each var
        back the way it came, since downstream numpy code assumes it."""
        ds, _ = gapped
        out = export_helpers.phase_slice(ds, "DD")
        for var in ("activity", "moving", *SLEEP_VARS):
            if var in out.data_vars:
                assert out[var].dims == ds[var].dims, f"{var} dim order changed"


def test_clean_dataset_is_untouched_by_the_trim(master_ds):
    """No gap, nothing trimmed, nothing masked — the path that the 724-file
    SCAMP baseline confirmed is byte-identical."""
    out = export_helpers.phase_slice(master_ds, "DD")
    for fly in [str(f) for f in out["id"].values]:
        for var in SLEEP_VARS:
            if var in out.data_vars:
                assert _missing_count(out[var], fly) == 0
