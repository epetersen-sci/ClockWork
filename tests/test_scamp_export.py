"""The SCAMP export: files the lab's MATLAB SCAMP loader can read, holding the data.

This was verified once, by hand, by diffing 724 real files against a baseline
(tests/README.md), and never by a test — ``scamp_export`` was the one package no
test imported. SCAMP is unforgiving in ways that fail quietly: a filename with a
stray ``C`` is mis-parsed and the file dropped, boards whose files differ in
length abort the load, and a missing reading written as 0 is read as a fly that
did not move.

The format primitives are tested directly. The export itself runs the page's
path — curate, ``phase_slice``, ``export_dataset_to_scamp`` — on example_data's
Monitors 17 and 18, reads every file back with ``read_dam_file`` (the repo's
Python port of SCAMP's ``dam_file.m``), and checks it against the dataset. A
checksum per file is then pinned, standing in for the one-off byte diff.
"""

import json

import numpy as np
import pandas as pd
import pytest

from clockwork.app import export_helpers
from clockwork.core import dam_utilities
from clockwork.scamp_export import scamp_writer
from clockwork.scamp_export.scamp_exporter import export_dataset_to_scamp
from snapshot_util import check_snapshot

# ---------------------------------------------------------------------------
# Format primitives
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "stamp, expected",
    [("2025-01-15 09:00:00", 900), ("2025-01-15 23:30:59", 2330), ("2025-01-15 00:05", 5)],
)
def test_military_time(stamp, expected):
    assert scamp_writer.military_time(pd.Timestamp(stamp)) == expected


def test_encode_activity_rounds_and_marks_missing_negative():
    out = scamp_writer.encode_activity([0.0, 2.4, 2.6, np.nan, 7.0])
    np.testing.assert_array_equal(out, [0, 2, 3, -1, 7])
    assert out.dtype == np.int64


def test_bin_to_30min_sums_and_poisons_blocks_with_a_missing_minute():
    one_min = np.ones(90, dtype=np.int64)
    one_min[40] = -1  # the second block has a missing reading
    np.testing.assert_array_equal(scamp_writer.bin_to_30min(one_min), [30, -1, 30])


def test_bin_to_30min_refuses_a_partial_block():
    with pytest.raises(ValueError, match="divisible by 30"):
        scamp_writer.bin_to_30min(np.ones(45, dtype=np.int64))


@pytest.mark.parametrize(
    "name", ["PYM17C1.txt", "PYC1M17C1", "PYm17C1", "M17C", "PY_17_1"], ids=repr
)
def test_filenames_scamp_would_misparse_are_refused(tmp_path, name):
    with pytest.raises(ValueError):
        scamp_writer.write_dam_file(str(tmp_path / name), "h", [1, 2], 1, 900)


def test_a_written_file_reads_back_identically(tmp_path):
    values = np.array([0, 5, -1, 12, 3])
    path = tmp_path / "PYM17C1"
    scamp_writer.write_dam_file(str(path), "header\nwith newline", values, 1, 900)
    back = scamp_writer.read_dam_file(str(path))
    assert back["header"] == "header with newline"  # one line, always
    assert (back["len"], back["int"], back["start"]) == (5, 1, 900)
    np.testing.assert_array_equal(back["values"], values)


# ---------------------------------------------------------------------------
# The export, on real data
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def dd_export(example_ds, tmp_path_factory):
    live = dam_utilities.curate_dead_animals(example_ds)[0]
    live.attrs["split_applied"] = 1  # what Curate & split's "Apply split" records
    dd = export_helpers.phase_slice(live, "DD")
    out = tmp_path_factory.mktemp("scamp") / "DD"
    summary = export_dataset_to_scamp(dd, str(out), phase_label="DD")
    return dd, out, summary


def _files(folder):
    return sorted(p for p in folder.iterdir() if p.is_file())


def test_one_file_per_exported_fly_in_each_interval(dd_export):
    dd, out, summary = dd_export
    assert summary["n_out"] == len(_files(out / "1min")) == len(_files(out / "30min"))
    assert summary["n_out"] + len(summary["dropped"]) == dd.sizes["id"]


def test_every_board_has_one_length_in_whole_days(dd_export):
    """SCAMP's dam_read_names.m aborts a board whose files differ in length."""
    _, out, summary = dd_export
    lengths = {}
    for path in _files(out / "1min"):
        back = scamp_writer.read_dam_file(str(path))
        assert back["len"] == len(back["values"])
        assert back["int"] == 1 and back["start"] == 900
        board = path.name.rsplit("C", 1)[0]
        lengths.setdefault(board, set()).add(back["len"])
    assert len(lengths) == len(summary["boards"]) == 2
    for board, lens in lengths.items():
        assert len(lens) == 1, f"board {board} has lengths {lens}"
        assert next(iter(lens)) % 1440 == 0


def test_the_files_hold_the_datasets_activity(dd_export):
    dd, out, _ = dd_export
    key = pd.read_csv(out / "scamp_group_key.csv")
    for _, row in key.iterrows():
        back = scamp_writer.read_dam_file(str(out / "1min" / row["filename"]))
        trace = dd["activity"].sel(id=row["fly_id"]).values
        first = int(np.flatnonzero(np.isfinite(trace))[0])
        expected = np.rint(trace[first : first + back["len"]]).astype(np.int64)
        np.testing.assert_array_equal(back["values"], expected, err_msg=row["fly_id"])
        assert f"id={row['fly_id']}" in back["header"]


def test_the_30min_files_are_sums_of_the_1min_files(dd_export):
    _, out, _ = dd_export
    for path in _files(out / "1min"):
        one = scamp_writer.read_dam_file(str(path))["values"]
        thirty = scamp_writer.read_dam_file(str(out / "30min" / path.name))
        assert thirty["int"] == 30
        np.testing.assert_array_equal(thirty["values"], one.reshape(-1, 30).sum(axis=1))


def test_the_group_key_carries_each_flys_genotype(dd_export):
    dd, out, _ = dd_export
    key = pd.read_csv(out / "scamp_group_key.csv")
    for _, row in key.iterrows():
        assert row["genotype"] == str(dd["genotype"].sel(id=row["fly_id"]).values)
    manifest = json.loads((out / "export_manifest.json").read_text())
    assert manifest["n_exported_flies"] == len(key)


def test_the_export_is_pinned(dd_export, update_snapshots):
    _, out, _ = dd_export
    records = {}
    for interval in ("1min", "30min"):
        for path in _files(out / interval):
            back = scamp_writer.read_dam_file(str(path))
            records[f"{interval}/{path.name}"] = {
                "len": back["len"],
                "sum": int(back["values"].sum()),
                "n_missing": int((back["values"] < 0).sum()),
            }
    check_snapshot("scamp_dd_example", records, update_snapshots, rtol=0)
