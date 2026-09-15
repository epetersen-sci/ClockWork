"""Renaming the experiment takes effect on the dataset already loaded.

The first version of this stamped the name inside the "Create Dataset" button and
nowhere else. Typing it afterwards, or loading a .nc at all, left a filled-in
field that every export ignored — exports went to the unsuffixed folder and the
PNG download filenames lost their prefix, with nothing on screen saying why.

So the property under test is not "the name can be set". It is that the field and
the loaded dataset cannot disagree.
"""

import xarray as xr

import export_helpers
from ui import charts, experiment


class TestApplyingToTheLoadedDataset:
    def test_editing_the_field_renames_the_dataset(self, app, master_ds):
        at = app(ds=master_ds.copy(), page="export_data")
        at.text_input(key=experiment.KEY).set_value("exp8").run()
        assert at.session_state["dataset"].attrs["experiment_name"] == "exp8"

    def test_it_renames_the_unfiltered_copy_too(self, app, master_ds):
        """dataset_full is what Groups & subsets restores from. A name written to
        only one of them vanishes the moment a group filter is reset."""
        at = app(ds=master_ds.copy(), page="export_data")
        at.text_input(key=experiment.KEY).set_value("exp8").run()
        assert at.session_state["dataset_full"].attrs["experiment_name"] == "exp8"

    def test_the_name_is_sanitised_on_the_way_in(self, app, master_ds):
        at = app(ds=master_ds.copy(), page="export_data")
        at.text_input(key=experiment.KEY).set_value("exp 8 (repeat)").run()
        assert at.session_state["dataset"].attrs["experiment_name"] == "exp_8_repeat"

    def test_clearing_it_returns_to_the_plain_folder(self, app, master_ds):
        ds = master_ds.copy()
        ds.attrs["experiment_name"] = "exp8"
        at = app(ds=ds, page="export_data")
        at.text_input(key=experiment.KEY).set_value("").run()
        assert at.session_state["dataset"].attrs["experiment_name"] == ""

    def test_the_export_folder_follows_immediately(self, app, master_ds, tmp_path):
        """The whole point: rename, and the next save lands somewhere else."""
        ds = master_ds.copy()
        ds.attrs["source_data_dir"] = str(tmp_path)
        at = app(ds=ds, page="export_data")
        at.text_input(key=experiment.KEY).set_value("exp8").run()
        out = export_helpers._export_dir(at.session_state["dataset"])
        assert out.endswith("Graph Exports_exp8")

    def test_the_png_filename_prefix_follows_too(self, app, master_ds):
        """Same attr feeds the camera button's download name, which is why both
        symptoms appeared together."""
        import plotly.graph_objects as go

        at = app(ds=master_ds.copy(), page="export_data")
        at.text_input(key=experiment.KEY).set_value("exp8").run()
        fig = go.Figure(go.Scatter(x=[0, 1], y=[1, 2]))
        fig.update_layout(title="Sleep profile")
        # charts reads the dataset out of session state, so mirror what the page saw.
        import streamlit as st

        st.session_state["dataset"] = at.session_state["dataset"]
        assert charts.png_filename(fig) == "exp8_Sleep_profile"


class TestSyncingFromTheDataset:
    def test_it_takes_the_name_from_the_dataset(self):
        import streamlit as st

        st.session_state[experiment.KEY] = "left over from the last import"
        experiment.sync_from_dataset(xr.Dataset(attrs={"experiment_name": "exp8"}))
        assert st.session_state[experiment.KEY] == "exp8"

    def test_a_dataset_without_a_name_clears_the_field(self):
        """A stale name must not follow a newly loaded dataset around and send its
        exports into the previous experiment's folder."""
        import streamlit as st

        st.session_state[experiment.KEY] = "exp7"
        experiment.sync_from_dataset(xr.Dataset())
        assert st.session_state[experiment.KEY] == ""


class TestApplyToLoadedIsASafeNoOp:
    def test_it_does_nothing_without_a_dataset(self):
        import streamlit as st

        st.session_state.pop("dataset", None)
        st.session_state.pop("dataset_full", None)
        assert experiment.apply_to_loaded("exp8") is False

    def test_it_reports_whether_anything_changed(self):
        """Used to decide nothing today, but a caller that starts invalidating on a
        rename needs the answer to be true only when it really changed."""
        import streamlit as st

        st.session_state["dataset"] = xr.Dataset(attrs={"experiment_name": "exp8"})
        st.session_state.pop("dataset_full", None)
        assert experiment.apply_to_loaded("exp8") is False
        assert experiment.apply_to_loaded("exp9") is True
