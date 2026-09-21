"""The .xlsx exports, and the field that names the folder they land in.

Three bugs shipped together and none of them was caught by a test, which is the
point of this file. Each was invisible in exactly the way that makes a bug
survive: the app reported success.

1. ``save_excel_button`` joined the caller's stem onto the export directory and
   wrote there verbatim. Callers name the workbook, not the format — they pass
   ``"sleep_totals"`` — so the file landed with NO EXTENSION, and the button
   cheerfully reported "Saved 2 sheet(s) to …/sleep_totals". Windows will not
   open it.

2. The ZT tables are indexed by bin and carry ``(group, stat)`` MultiIndex
   columns, which pandas REFUSES to write with ``index=False``. The button caught
   the raise and blamed a missing openpyxl.

3. ``experiment.sync_from_dataset`` assigned the name field's session key. The
   Import page creates that field near the top of its first tab, every tab body
   runs on every rerun, and loading a ``.nc`` happens further down in ANOTHER
   tab — so the assignment came after the widget and Streamlit refused it. The
   raise landed inside the loader's own ``except``, so the dataset loaded, an
   error appeared, and the name silently stayed on the previous experiment,
   sending its exports to that folder.
"""

import numpy as np
import pandas as pd
import pytest

import export_helpers as ex


@pytest.fixture
def workbook_ds(tmp_path):
    """The smallest dataset the export helpers will accept a working folder from."""
    import xarray as xr

    ds = xr.Dataset(
        {"activity": (("time", "id"), np.zeros((4, 2), dtype=float))},
        coords={"time": np.arange(4), "id": ["a", "b"]},
        attrs={"experiment_name": "exp7"},
    )
    return ds


def _write(ds, sheets, filename, tmp_path, monkeypatch):
    """Drive save_excel_button's writer without a Streamlit run.

    The button itself needs a session; what is under test is the path it builds
    and the frames it hands to pandas, so the export directory is pinned and the
    button pressed by hand.
    """
    monkeypatch.setattr(ex, "_export_dir", lambda _ds, _sub=None: str(tmp_path))

    import streamlit as st

    monkeypatch.setattr(st, "button", lambda *a, **k: True)
    monkeypatch.setattr(st, "info", lambda *a, **k: None)
    monkeypatch.setattr(st, "error", lambda *a, **k: None)
    monkeypatch.setattr(ex, "_remember", lambda *a, **k: None)
    monkeypatch.setattr(ex, "_show_remembered", lambda *a, **k: None)
    return ex.save_excel_button("go", sheets, ds, filename, key="k")


class TestTheWorkbookIsActuallyAWorkbook:
    def test_a_stem_gets_the_extension(self, workbook_ds, tmp_path, monkeypatch):
        """Callers pass "sleep_totals". A file by that exact name is not openable
        by anything, and the button reported success anyway."""
        path = _write(
            workbook_ds,
            [("per_fly", pd.DataFrame({"ID": ["a"], "v": [1.0]}))],
            "sleep_totals",
            tmp_path,
            monkeypatch,
        )
        assert path is not None
        assert path.endswith(".xlsx"), path
        assert (tmp_path / "sleep_totals.xlsx").exists()

    def test_a_caller_that_names_the_format_is_left_alone(
        self, workbook_ds, tmp_path, monkeypatch
    ):
        path = _write(
            workbook_ds,
            [("s", pd.DataFrame({"v": [1]}))],
            "already.xlsx",
            tmp_path,
            monkeypatch,
        )
        assert path.endswith("already.xlsx")
        assert not (tmp_path / "already.xlsx.xlsx").exists()

    def test_it_opens_and_holds_every_sheet(self, workbook_ds, tmp_path, monkeypatch):
        summary = pd.DataFrame({"group": ["ctrl", "mut"], "mean": [1.0, 2.0]})
        per_fly = pd.DataFrame({"ID": ["a", "b"], "v": [1.0, 2.0]})
        path = _write(
            workbook_ds,
            [("summary", summary), ("per_fly", per_fly)],
            "totals",
            tmp_path,
            monkeypatch,
        )
        book = pd.read_excel(path, sheet_name=None)
        assert set(book) == {"summary", "per_fly"}
        assert list(book["summary"]["group"]) == ["ctrl", "mut"]


class TestTheZTTableSurvivesTheTrip:
    """Its shape is load-bearing: a GraphPad template reads the columns
    positionally, so the two header rows have to arrive intact."""

    @pytest.fixture
    def zt_table(self):
        binned = pd.DataFrame(
            {
                "zt_bin_minute": [0, 0, 30, 30, 0, 0, 30, 30],
                "group": ["ctrl"] * 4 + ["mut"] * 4,
                "sleep": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                "id": list("abababab"),
            }
        )
        return ex.zt_group_summary_table(binned, "sleep", 30)

    def test_the_fixture_really_has_multiindex_columns(self, zt_table):
        """If this ever stops being true the bug below cannot recur, and this file
        should say so rather than quietly testing nothing."""
        assert isinstance(zt_table.columns, pd.MultiIndex)

    def test_writing_it_does_not_raise(self, workbook_ds, tmp_path, monkeypatch, zt_table):
        """pandas refuses index=False for MultiIndex columns outright. The button
        caught that and reported a missing openpyxl, which sent at least one
        person looking in the wrong place — see TestTheErrorNamesItsOwnCause."""
        path = _write(
            workbook_ds,
            [("group_summary", zt_table)],
            "sleep_binned",
            tmp_path,
            monkeypatch,
        )
        assert path is not None and (tmp_path / "sleep_binned.xlsx").exists()

    def test_the_bin_index_is_kept(self, workbook_ds, tmp_path, monkeypatch, zt_table):
        """Dropping it would leave a table whose rows say nothing about which bin
        they are."""
        path = _write(
            workbook_ds, [("g", zt_table)], "binned", tmp_path, monkeypatch
        )
        back = pd.read_excel(path, sheet_name="g", header=[0, 1], index_col=0)
        assert len(back) == len(zt_table)
        assert back.index.name == zt_table.index.name

    def test_a_plain_table_still_loses_its_row_numbers(
        self, workbook_ds, tmp_path, monkeypatch
    ):
        """The other half of the rule: a RangeIndex is noise in a spreadsheet."""
        path = _write(
            workbook_ds,
            [("per_fly", pd.DataFrame({"ID": ["a", "b"], "v": [1.0, 2.0]}))],
            "plain",
            tmp_path,
            monkeypatch,
        )
        back = pd.read_excel(path, sheet_name="per_fly")
        assert list(back.columns) == ["ID", "v"], back.columns


class TestLoadingANetCDFCanRenameTheField:
    """Bug 3: the field is created in the first tab, the loader runs in another."""

    def test_the_name_waits_for_the_widget_instead_of_raising(self, app, master_ds, tmp_path):
        from load_and_save_datasets import save_dataset_to_netcdf

        nc = tmp_path / "exp9_run.nc"
        named = master_ds.copy()
        named.attrs["experiment_name"] = "exp9_run"
        save_dataset_to_netcdf(named, str(nc))

        at = app(page="data_import")
        # A metadata path, so the name field is populated and definitely rendered
        # before the loader runs — the user's own situation.
        at = at.text_input(key="metadata_path_input").set_value(
            "/runs/metadata_exp7.xlsx"
        ).run()
        assert at.session_state["experiment_name_input"] == "exp7"

        at = at.text_input(key="nc_path_input").set_value(str(nc)).run()
        at = at.button(key="load_nc").click().run()

        assert not at.exception, at.exception
        assert not [e.value for e in at.error], [e.value for e in at.error]
        assert at.session_state["dataset"] is not None
        assert at.session_state["experiment_name_input"] == "exp9_run", (
            "the loaded file's own name must reach the field, or its exports go to "
            "the previous experiment's folder"
        )

    def test_the_parked_name_is_consumed_once(self, app, master_ds, tmp_path):
        """Popped rather than read, so an edit made after the load is not undone on
        the next rerun."""
        from load_and_save_datasets import save_dataset_to_netcdf

        nc = tmp_path / "exp9_run.nc"
        named = master_ds.copy()
        named.attrs["experiment_name"] = "exp9_run"
        save_dataset_to_netcdf(named, str(nc))

        at = app(page="data_import")
        at = at.text_input(key="nc_path_input").set_value(str(nc)).run()
        at = at.button(key="load_nc").click().run()
        at = at.text_input(key="experiment_name_input").set_value("my own name").run()
        at = at.run()
        assert at.session_state["experiment_name_input"] == "my own name"
        from ui.experiment import PENDING

        assert PENDING not in at.session_state


class TestTheErrorNamesItsOwnCause:
    """The export used to answer every failure with "install openpyxl"."""

    def test_a_missing_package_still_says_so(self, workbook_ds, tmp_path, monkeypatch):
        import streamlit as st

        said = []
        monkeypatch.setattr(ex, "_export_dir", lambda _ds, _sub=None: str(tmp_path))
        monkeypatch.setattr(st, "button", lambda *a, **k: True)
        monkeypatch.setattr(st, "error", lambda m, *a, **k: said.append(m))
        monkeypatch.setattr(ex, "_remember", lambda *a, **k: None)
        monkeypatch.setattr(ex, "_show_remembered", lambda *a, **k: None)

        def _no_engine(*a, **k):
            raise ImportError("Missing optional dependency 'openpyxl'")

        # The module, not ``ex.pd``: export_helpers imports pandas INSIDE the
        # function, so there is no module attribute to patch on ex itself.
        monkeypatch.setattr(pd, "ExcelWriter", _no_engine)
        ex.save_excel_button(
            "go", [("s", pd.DataFrame({"v": [1]}))], workbook_ds, "x", key="k"
        )
        assert said and "pip install openpyxl" in said[0]

    def test_any_other_failure_reports_itself(self, workbook_ds, tmp_path, monkeypatch):
        """The regression: with openpyxl installed and a frame pandas would not
        write, the message named the wrong cause and hid the right one."""
        import streamlit as st

        said = []
        monkeypatch.setattr(ex, "_export_dir", lambda _ds, _sub=None: str(tmp_path))
        monkeypatch.setattr(st, "button", lambda *a, **k: True)
        monkeypatch.setattr(st, "error", lambda m, *a, **k: said.append(m))
        monkeypatch.setattr(ex, "_remember", lambda *a, **k: None)
        monkeypatch.setattr(ex, "_show_remembered", lambda *a, **k: None)

        def _boom(*a, **k):
            raise NotImplementedError("MultiIndex columns and no index")

        monkeypatch.setattr(pd, "ExcelWriter", _boom)
        ex.save_excel_button(
            "go", [("s", pd.DataFrame({"v": [1]}))], workbook_ds, "x", key="k"
        )
        assert said, "a failure must be reported"
        assert "MultiIndex columns" in said[0]
        assert "openpyxl" not in said[0], said[0]
