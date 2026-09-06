"""Phase/split metadata — the canonical attrs and the legacy alias (backlog item 4).

The legacy ``split_phase`` alias is no longer written, but it is still READ so
that ``.nc`` files saved before that change resolve to the right phase. That
read path has no other coverage: it only fires for files that no current code
path can produce, so nothing else would notice if it were deleted as dead code.
"""

import numpy as np
import pytest
import xarray as xr

from dataset_meta import (
    PHASE_DD,
    PHASE_FULL,
    PHASE_LD,
    dataset_phase,
    is_split_applied,
    stamp_phase,
)


def _bare():
    return xr.Dataset({"activity": (("id", "time"), np.zeros((1, 4), dtype=np.float32))},
                      coords={"id": ["f1"], "time": np.arange(4)})


class TestCanonicalAttrs:
    def test_stamp_and_read_roundtrip(self):
        ds = _bare()
        stamp_phase(ds, PHASE_LD)
        assert dataset_phase(ds) == PHASE_LD
        assert is_split_applied(ds) is True

    def test_no_metadata_defaults_to_full_and_unsplit(self):
        ds = _bare()
        assert dataset_phase(ds) == PHASE_FULL
        assert is_split_applied(ds) is False

    def test_split_applied_survives_netcdf_int_roundtrip(self):
        """NetCDF stores bools as ints; is_split_applied must accept that."""
        ds = _bare()
        ds.attrs["phase"] = PHASE_FULL
        ds.attrs["split_applied"] = np.int64(1)  # what comes back off disk
        assert is_split_applied(ds) is True


class TestLegacyAliasStillRead:
    """A pre-canonical file carries ONLY split_phase. Do not delete these paths."""

    @pytest.mark.parametrize(
        ("legacy", "expected_phase"),
        [("LD", PHASE_LD), ("DD", PHASE_DD), ("both", PHASE_FULL)],
    )
    def test_legacy_alias_resolves(self, legacy, expected_phase):
        ds = _bare()
        ds.attrs["split_phase"] = legacy
        assert dataset_phase(ds) == expected_phase
        assert is_split_applied(ds) is True

    def test_canonical_wins_over_legacy(self):
        ds = _bare()
        ds.attrs["phase"] = PHASE_DD
        ds.attrs["split_phase"] = "LD"  # a stale alias must not override
        assert dataset_phase(ds) == PHASE_DD


class TestAliasNoLongerWritten:
    def test_split_xarray_dataset_writes_canonical_only(self, master_ds):
        import dam_utilities

        out = dam_utilities.split_xarray_dataset(
            master_ds, phase="LD", gap_threshold_minutes=60
        )
        assert out.attrs["phase"] == PHASE_LD
        assert out.attrs["split_applied"] is True
        assert "split_phase" not in out.attrs, (
            "split_phase is the retired legacy alias; writing it again would grow "
            "the population of files carrying two sources of truth"
        )
