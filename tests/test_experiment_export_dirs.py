"""Exports land in a folder named for the experiment that produced them.

A paired run keeps both metadata files in ONE working folder. Every export went
to ``Graph Exports/`` under that folder, so the second experiment's figures
overwrote the first's — same filenames, same directory, no warning. The folder now
carries the experiment's name, taken from the metadata filename.

The name lives in ``ds.attrs['experiment_name']`` and nowhere else. That is the
load-bearing part: reading it from a session key instead would let a name left
over from the previously loaded experiment redirect a reloaded ``.nc``'s exports,
which is a silent wrong answer rather than a visible failure.
"""

import xarray as xr

import dam_utilities
import export_helpers
from dam_utilities import (
    experiment_name_from_path,
    experiment_suffix,
    sanitize_experiment_name,
)


def _ds(working_dir, name=None):
    """A stand-in dataset carrying only the two attrs the export path reads."""
    attrs = {"source_data_dir": str(working_dir)}
    if name is not None:
        attrs["experiment_name"] = name
    return xr.Dataset(attrs=attrs)


class TestNameFromMetadataFilename:
    def test_drops_the_word_metadata(self):
        assert experiment_name_from_path("metadata_exp8_8_18_26.xlsx") == "exp8_8_18_26"

    def test_drops_it_from_either_end(self):
        assert experiment_name_from_path("exp6_metadata.csv") == "exp6"

    def test_a_bare_metadata_file_has_no_name(self):
        """Nothing is left to name the run with, so exports go where they always
        did rather than to a folder called ``Graph Exports_``."""
        assert experiment_name_from_path("metadata.csv") == ""

    def test_case_does_not_matter(self):
        assert experiment_name_from_path("METADATA_exp3.XLSX") == "exp3"

    def test_the_directory_is_ignored(self):
        assert experiment_name_from_path(r"D:\runs\july\metadata_exp2.csv") == "exp2"

    def test_punctuation_becomes_underscores(self):
        assert experiment_name_from_path("metadata-exp 4.2.csv") == "exp_4_2"

    def test_no_path_at_all_is_not_an_error(self):
        """The import page calls this with whatever is in the text input, which is
        empty until the user fills it in."""
        assert experiment_name_from_path("") == ""
        assert experiment_name_from_path(None) == ""


class TestSuffix:
    def test_reads_the_dataset(self, tmp_path):
        assert experiment_suffix(_ds(tmp_path, "exp8")) == "_exp8"

    def test_a_dataset_without_a_name_gets_no_suffix(self, tmp_path):
        """Datasets saved before this existed carry no name and must keep
        exporting exactly where they always did."""
        assert experiment_suffix(_ds(tmp_path)) == ""

    def test_an_empty_name_gets_no_suffix(self, tmp_path):
        assert experiment_suffix(_ds(tmp_path, "")) == ""

    def test_a_whitespace_name_gets_no_suffix(self, tmp_path):
        assert experiment_suffix(_ds(tmp_path, "   ")) == ""

    def test_no_dataset_at_all_gets_no_suffix(self):
        assert experiment_suffix(None) == ""


class TestExportDirectory:
    def test_folder_is_named_for_the_experiment(self, tmp_path):
        out = export_helpers._export_dir(_ds(tmp_path, "exp8_8_18_26"))
        assert out.endswith("Graph Exports_exp8_8_18_26")

    def test_unnamed_experiments_keep_the_plain_folder(self, tmp_path):
        out = export_helpers._export_dir(_ds(tmp_path))
        assert out.endswith("Graph Exports")

    def test_the_folder_is_created(self, tmp_path):
        import os

        out = export_helpers._export_dir(_ds(tmp_path, "exp1"))
        assert os.path.isdir(out)

    def test_a_subfolder_nests_inside_it(self, tmp_path):
        import os

        out = export_helpers._export_dir(_ds(tmp_path, "exp1"), "Sleep_bouts")
        assert os.path.basename(out) == "Sleep_bouts"
        assert os.path.basename(os.path.dirname(out)) == "Graph Exports_exp1"
        assert os.path.isdir(out)

    def test_two_experiments_in_one_working_folder_do_not_collide(self, tmp_path):
        """The bug this exists to prevent: same working folder, same filenames,
        one export silently overwriting the other."""
        first = export_helpers._export_dir(_ds(tmp_path, "exp7"))
        second = export_helpers._export_dir(_ds(tmp_path, "exp8"))
        assert first != second


class TestSurvivesTheNetCDFRoundTrip:
    """A reloaded .nc must export beside its own experiment, not the last one
    that happened to be open."""

    def test_name_survives(self, master_ds, tmp_path):
        ds = master_ds.copy()
        ds.attrs["experiment_name"] = "exp8_8_18_26"
        path = tmp_path / "roundtrip.nc"
        ds.to_netcdf(path)
        reloaded = xr.load_dataset(path)
        assert reloaded.attrs["experiment_name"] == "exp8_8_18_26"

    def test_reloaded_dataset_exports_to_the_same_folder(self, master_ds, tmp_path):
        ds = master_ds.copy()
        ds.attrs["experiment_name"] = "exp8_8_18_26"
        ds.attrs["source_data_dir"] = str(tmp_path)
        path = tmp_path / "roundtrip.nc"
        ds.to_netcdf(path)
        reloaded = xr.load_dataset(path)
        assert export_helpers._export_dir(reloaded) == export_helpers._export_dir(ds)


def test_the_suffix_is_read_from_the_dataset_only():
    """Guards the decision, not just the behaviour. experiment_suffix takes no
    fallback argument and reads no session state, so there is no way for a stale
    name to reach it — see the module docstring."""
    import inspect

    params = list(inspect.signature(experiment_suffix).parameters)
    assert params == ["ds"], (
        f"experiment_suffix grew parameters {params}. A second source for the name "
        "reintroduces the stale-name bug this signature exists to prevent."
    )
    assert "session_state" not in inspect.getsource(dam_utilities.experiment_suffix)


class TestSanitizing:
    """The name becomes a folder, so it has to survive being one."""

    def test_spaces_and_punctuation_collapse(self):
        assert sanitize_experiment_name("exp 8 (repeat)") == "exp_8_repeat"

    def test_path_separators_cannot_survive(self):
        """The whole point: a typed name must not be able to escape the working
        folder or name a drive."""
        for hostile in ("../../etc", r"C:\Windows", "a/b", r"a\b"):
            out = sanitize_experiment_name(hostile)
            assert "/" not in out and "\\" not in out and ".." not in out

    def test_it_is_capped(self):
        assert len(sanitize_experiment_name("x" * 200)) == 60

    def test_empty_stays_empty(self):
        assert sanitize_experiment_name("") == ""
        assert sanitize_experiment_name(None) == ""

    def test_a_stored_name_is_sanitised_on_the_way_out_too(self, tmp_path):
        """A dataset stamped before the rule existed, or edited by hand, still
        cannot put a separator into a path."""
        assert experiment_suffix(_ds(tmp_path, "a/b")) == "_a_b"


class TestTheImportFieldPrefill:
    """The filename is a starting point offered in an editable field, not the
    source of truth it used to be."""

    def test_it_is_prefilled_from_the_metadata_filename(self, app):
        at = app(page="data_import")
        at.text_input(key="metadata_path_input").set_value(
            "/runs/metadata_exp7_7_31_26.xlsx"
        ).run()
        assert at.session_state["experiment_name_input"] == "exp7_7_31_26"

    def test_an_uninformative_filename_prefills_nothing(self, app):
        """The case the filename source could never handle: the repo's own example
        data is named plainly metadata.xlsx."""
        at = app(page="data_import")
        at.text_input(key="metadata_path_input").set_value("/runs/metadata.xlsx").run()
        assert at.session_state["experiment_name_input"] == ""

    def test_a_typed_name_survives_a_rerun(self, app):
        """The prefill is guarded on the path it came from, so it must not stamp
        over an edit on the next rerun."""
        at = app(page="data_import")
        at.text_input(key="metadata_path_input").set_value("/runs/metadata.xlsx").run()
        at.text_input(key="experiment_name_input").set_value("my own name").run()
        at.run()
        assert at.session_state["experiment_name_input"] == "my own name"

    def test_pointing_at_a_different_file_re_guesses(self, app):
        """Keeping the previous run's name here would be worse than losing an edit —
        it is how two experiments end up sharing one export folder again."""
        at = app(page="data_import")
        at.text_input(key="metadata_path_input").set_value("/runs/metadata_exp7.xlsx").run()
        at.text_input(key="experiment_name_input").set_value("edited").run()
        at.text_input(key="metadata_path_input").set_value("/runs/metadata_exp8.xlsx").run()
        assert at.session_state["experiment_name_input"] == "exp8"
