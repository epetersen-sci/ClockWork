"""Metadata must follow its own monitor, whatever the digit counts (item 13's bug).

``create_xarray_dataset`` used to pair the activity matrix with the metadata
POSITIONALLY: values came from ``dam_data.columns``, per-fly coords from
``metadata`` row order, and nothing checked the two agreed. The activity frame is
built from ``unique_combos.sort_values(["Monitor", "start_datetime"])`` with
``Monitor`` cast to **str**, so ``"10"`` sorts before ``"2"``. A metadata file
listing monitors in natural numeric order therefore produced::

    metadata row order : ['..._2_1',  '..._2_2',  '..._10_1', '..._10_2']
    data column order  : ['..._10_1', '..._10_2', '..._2_1',  '..._2_2']
    xarray genotype    : ['AAA',      'AAA',      'ZZZ',      'ZZZ']

Monitor 10's traces were analysed under monitor 2's ids and genotypes. It failed
SILENTLY — the shapes matched, so nothing raised.

Fixed in d0a477c, which reindexes the metadata onto ``dam_data.columns`` before
building any coord. Nothing exercised that fix: ``example_data``'s monitors are
17-22, all two digits, so string and numeric order agree and the trigger cannot
occur. This test renames COPIES of that real data to monitors 2 and 10 so the
mismatch is present, and asserts the labels still land on the right traces.

The committed ``example_data`` is deliberately left alone — its monitor numbers
are a fact about the recordings, and the genotype attached to each is real
experimental provenance.
"""

import shutil

import numpy as np
import pandas as pd
import pytest
from conftest import REPO_ROOT

import dam_processor
import dam_utilities

EXAMPLE = REPO_ROOT / "example_data"

# original monitor -> renamed monitor. Digit counts differ, and the metadata
# below lists them in NUMERIC order, which is what string-sorting reverses.
RENAME = {17: 2, 18: 10}

pytestmark = pytest.mark.skipif(
    not (EXAMPLE / "Monitor17.txt").is_file(),
    reason="example_data monitor files not present",
)


def _import(metadata_path, data_dir):
    processor = dam_processor.MetadataProcessor(
        str(metadata_path), str(data_dir), gap_threshold_hours=1.0
    )
    metadata, data = processor.run()
    return dam_utilities.create_xarray_dataset(
        dam_utilities.convert_to_relative_time(data, metadata), metadata
    )


@pytest.fixture(scope="module")
def mixed_digit_import(tmp_path_factory):
    """Two real monitors renamed so their numbers have different digit counts."""
    out = tmp_path_factory.mktemp("mixdigit")
    for old, new in RENAME.items():
        shutil.copy(EXAMPLE / f"Monitor{old}.txt", out / f"Monitor{new}.txt")

    meta = pd.read_excel(EXAMPLE / "metadata.xlsx")
    meta = meta[meta["Monitor"].isin(RENAME)].copy()
    meta["Monitor"] = [RENAME[m] for m in meta["Monitor"]]
    meta = meta.sort_values("Monitor")  # numeric order — the trigger
    meta.to_csv(out / "metadata.csv", index=False)

    listed = [str(m) for m in meta["Monitor"]]
    assert listed != sorted(listed), (
        "fixture precondition: the metadata must list monitors in an order that "
        "differs from their STRING sort, or the bug cannot occur"
    )
    return _import(out / "metadata.csv", out), meta


@pytest.fixture(scope="module")
def original_import(tmp_path_factory):
    """The same two monitors under their real numbers — the ground truth."""
    out = tmp_path_factory.mktemp("original")
    for old in RENAME:
        shutil.copy(EXAMPLE / f"Monitor{old}.txt", out / f"Monitor{old}.txt")
    meta = pd.read_excel(EXAMPLE / "metadata.xlsx")
    meta = meta[meta["Monitor"].isin(RENAME)].copy()
    meta.to_csv(out / "metadata.csv", index=False)
    return _import(out / "metadata.csv", out)


def _twin(fly_id):
    """Map a renamed fly back to the same physical channel under its real number."""
    back = {str(new): str(old) for old, new in RENAME.items()}
    date, monitor, channel = str(fly_id).split("_")
    return f"{date}_{back[monitor]}_{channel}"


def test_genotype_follows_its_own_monitor(mixed_digit_import, original_import):
    """The headline assertion: each fly keeps the genotype of the monitor its
    data actually came from, not of whichever row sat in that position."""
    mixed, _ = mixed_digit_import
    mismatches = [
        (fid, str(mixed["genotype"].sel(id=fid).values),
         str(original_import["genotype"].sel(id=_twin(fid)).values))
        for fid in mixed["id"].values
        if str(mixed["genotype"].sel(id=fid).values)
        != str(original_import["genotype"].sel(id=_twin(fid)).values)
    ]
    assert not mismatches, f"genotype attached to the wrong monitor: {mismatches[:5]}"


def test_activity_traces_are_unchanged_by_the_renaming(
    mixed_digit_import, original_import
):
    """Guards the other half. A pairing bug could be masked if the DATA moved
    too; this pins each channel's trace to the physical file it came from."""
    mixed, _ = mixed_digit_import
    for fid in mixed["id"].values:
        np.testing.assert_array_equal(
            np.nan_to_num(mixed["activity"].sel(id=fid).values),
            np.nan_to_num(original_import["activity"].sel(id=_twin(fid)).values),
            err_msg=f"{fid} carries a different trace than its physical monitor",
        )


def test_every_group_label_matches_its_genotype(mixed_digit_import):
    """`group` is derived from the same per-id coords, so it inherits any
    mispairing — and it is what every downstream comparison keys on."""
    mixed, meta = mixed_digit_import
    expected = dict(zip(meta["Monitor"].astype(str), meta["genotype"]))
    for fid in mixed["id"].values:
        monitor = str(fid).split("_")[1]
        assert str(mixed["genotype"].sel(id=fid).values) == expected[monitor]
        assert str(mixed["group"].sel(id=fid).values).startswith(expected[monitor])
