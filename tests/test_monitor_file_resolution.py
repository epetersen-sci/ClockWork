"""A monitor's raw file is found whatever the export spelled it.

Trikinetics writes ``Monitor1.txt``. FlyBox's video tracker writes the
zero-padded ``Monitor01.txt``. Both mean monitor 1, and the loader used to build
exactly one filename from the metadata's monitor number and give up if that name
was not on disk — so a whole folder of padded exports failed to import with a
"file not found" naming a file the user could plainly see was there under a
slightly different name.

Case matters too, and only on some machines: Windows and a default macOS volume
resolve ``monitor01.txt`` against ``Monitor01.txt`` themselves, so a lowercase
export loads there and fails on Linux. Matching the case here means one folder
loads the same way everywhere.

The region parser gets the same treatment for ``1..16``, which hand-written
sheets use as often as ``1-16``.
"""

import os
import shutil

import numpy as np
import pandas as pd
import pytest
from conftest import REPO_ROOT

import dam_processor
import dam_utilities
from dam_processor import _find_monitor_file, _monitor_file_candidates, _parse_region_ids

EXAMPLE = REPO_ROOT / "example_data"


def _touch(folder, name):
    (folder / name).write_text("placeholder", encoding="utf-8")
    return folder / name


class TestCandidateNames:
    def test_plain_form_is_tried_first(self):
        assert _monitor_file_candidates(1)[0] == "Monitor1.txt"

    def test_padded_forms_follow(self):
        assert _monitor_file_candidates(1) == [
            "Monitor1.txt",
            "Monitor01.txt",
            "Monitor001.txt",
        ]

    def test_a_two_digit_monitor_does_not_repeat_itself(self):
        """`f"{17:02d}"` is already "17", so the 2-wide form must not be listed
        twice — a duplicate would make the "tried X or Y" message read wrong."""
        assert _monitor_file_candidates(17) == ["Monitor17.txt", "Monitor017.txt"]

    def test_a_string_monitor_number_behaves_like_the_int(self):
        assert _monitor_file_candidates("7") == _monitor_file_candidates(7)

    def test_a_non_numeric_monitor_gets_only_its_literal_name(self):
        """Padding is meaningless for a name like "A" — offer it unchanged rather
        than inventing spellings that cannot exist."""
        assert _monitor_file_candidates("A") == ["MonitorA.txt"]


class TestFindingTheFile:
    def test_finds_the_plain_spelling(self, tmp_path):
        _touch(tmp_path, "Monitor1.txt")
        assert os.path.basename(_find_monitor_file(tmp_path, 1)) == "Monitor1.txt"

    def test_finds_the_zero_padded_spelling(self, tmp_path):
        """The headline case: metadata says 1, the folder holds Monitor01.txt."""
        _touch(tmp_path, "Monitor01.txt")
        found = _find_monitor_file(tmp_path, 1)
        assert found is not None and os.path.basename(found) == "Monitor01.txt"

    def test_prefers_the_plain_spelling_when_both_exist(self, tmp_path):
        _touch(tmp_path, "Monitor3.txt")
        _touch(tmp_path, "Monitor03.txt")
        assert os.path.basename(_find_monitor_file(tmp_path, 3)) == "Monitor3.txt"

    def test_matches_case_insensitively(self, tmp_path):
        """Asserted on the lower-cased basename so it holds on a case-insensitive
        filesystem (which resolves the name itself) and a case-sensitive one
        (where the folder scan is what finds it)."""
        _touch(tmp_path, "monitor07.txt")
        found = _find_monitor_file(tmp_path, 7)
        assert found is not None and os.path.basename(found).lower() == "monitor07.txt"

    def test_returns_none_when_no_spelling_is_present(self, tmp_path):
        _touch(tmp_path, "Monitor2.txt")
        assert _find_monitor_file(tmp_path, 1) is None

    def test_returns_none_for_a_folder_that_cannot_be_read(self, tmp_path):
        """A wrong data directory is the commonest cause of a missing file, so it
        must return None rather than raise out of the import loop."""
        assert _find_monitor_file(tmp_path / "does_not_exist", 1) is None


class TestRegionRanges:
    def test_dotted_range_matches_the_hyphen_range(self):
        assert _parse_region_ids("1..16") == _parse_region_ids("1-16")

    def test_dotted_range_expands_inclusively(self):
        assert _parse_region_ids("1..4") == [1, 2, 3, 4]

    def test_the_two_spellings_mix_in_one_cell(self):
        assert _parse_region_ids("1-4,17..20") == [1, 2, 3, 4, 17, 18, 19, 20]

    def test_an_unparseable_cell_names_both_spellings(self):
        with pytest.raises(ValueError, match=r"1\.\.16"):
            _parse_region_ids("1..banana")


@pytest.mark.skipif(
    not (EXAMPLE / "Monitor17.txt").is_file(),
    reason="example_data monitor files not present",
)
class TestPaddedFilesImportEndToEnd:
    """The unit tests above prove the name resolves. This proves the import runs.

    Real monitor 17 is copied twice — once as ``Monitor1.txt`` and once as
    ``Monitor01.txt`` — and imported from two folders whose metadata is identical.
    Identical activity out of both is the claim: the padding changes the filename
    and nothing else.
    """

    @staticmethod
    def _import_as(out, filename):
        shutil.copy(EXAMPLE / "Monitor17.txt", out / filename)
        meta = pd.read_excel(EXAMPLE / "metadata.xlsx")
        meta = meta[meta["Monitor"] == 17].copy()
        meta["Monitor"] = 1
        meta.to_csv(out / "metadata.csv", index=False)
        processor = dam_processor.MetadataProcessor(
            str(out / "metadata.csv"), str(out), gap_threshold_hours=1.0
        )
        metadata, data = processor.run()
        return dam_utilities.create_xarray_dataset(
            dam_utilities.convert_to_relative_time(data, metadata), metadata
        )

    @pytest.fixture(scope="class")
    def plain(self, tmp_path_factory):
        return self._import_as(tmp_path_factory.mktemp("plain"), "Monitor1.txt")

    @pytest.fixture(scope="class")
    def padded(self, tmp_path_factory):
        return self._import_as(tmp_path_factory.mktemp("padded"), "Monitor01.txt")

    def test_the_padded_folder_yields_flies(self, padded):
        assert padded.sizes["id"] > 0

    def test_both_spellings_import_the_same_flies(self, plain, padded):
        assert list(padded["id"].values) == list(plain["id"].values)

    def test_both_spellings_import_the_same_activity(self, plain, padded):
        np.testing.assert_array_equal(
            np.nan_to_num(padded["activity"].values),
            np.nan_to_num(plain["activity"].values),
        )
