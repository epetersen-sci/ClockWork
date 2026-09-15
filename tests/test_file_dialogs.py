"""The Browse buttons, without ever opening a dialog.

Every test here either patches the subprocess away or patches `_ask` itself. A test
that actually opened a file dialog would block CI forever waiting for a click that
is never coming, so the one thing this module must never do is run the child.
"""

import subprocess

import pytest

from ui import file_dialogs
from ui.file_dialogs import DialogUnavailable


class _Done:
    """Stand-in for subprocess.CompletedProcess."""

    def __init__(self, stdout="", stderr="", returncode=0):
        self.stdout, self.stderr, self.returncode = stdout, stderr, returncode


class TestAsk:
    def test_returns_the_chosen_path(self, monkeypatch):
        monkeypatch.setattr(
            subprocess, "run", lambda *a, **k: _Done(stdout="C:/data/metadata.xlsx\n")
        )
        assert file_dialogs._ask() == "C:/data/metadata.xlsx"

    def test_a_cancelled_dialog_returns_empty(self, monkeypatch):
        """Cancel prints nothing. It is a decision, not a failure, so it must not
        raise — the caller leaves the text inputs alone."""
        monkeypatch.setattr(subprocess, "run", lambda *a, **k: _Done(stdout=""))
        assert file_dialogs._ask() == ""

    def test_a_timeout_is_reported_as_unavailable(self, monkeypatch):
        """The likeliest cause is the app running on a different machine than the
        browser, where the dialog opened on a desktop nobody is watching."""

        def _timeout(*a, **k):
            raise subprocess.TimeoutExpired(cmd="dialog", timeout=file_dialogs.TIMEOUT_SECONDS)

        monkeypatch.setattr(subprocess, "run", _timeout)
        with pytest.raises(DialogUnavailable, match="different machine"):
            file_dialogs._ask()

    def test_a_missing_interpreter_is_reported_as_unavailable(self, monkeypatch):
        def _oserror(*a, **k):
            raise OSError("no such file")

        monkeypatch.setattr(subprocess, "run", _oserror)
        with pytest.raises(DialogUnavailable, match="Could not start"):
            file_dialogs._ask()

    def test_a_crashed_dialog_surfaces_its_last_line(self, monkeypatch):
        """No display and no tkinter both land here. The last stderr line is the
        one that names the cause."""
        monkeypatch.setattr(
            subprocess,
            "run",
            lambda *a, **k: _Done(
                stderr="Traceback...\nImportError: no module named tkinter", returncode=1
            ),
        )
        with pytest.raises(DialogUnavailable, match="tkinter"):
            file_dialogs._ask()

    def test_a_silent_crash_still_says_something(self, monkeypatch):
        monkeypatch.setattr(subprocess, "run", lambda *a, **k: _Done(returncode=1))
        with pytest.raises(DialogUnavailable, match="failed to open"):
            file_dialogs._ask()


class TestTheChildScript:
    def test_it_exists_where_the_module_points(self):
        """The path is built from __file__, so a move breaks it silently — the
        dialog would just report 'could not start' forever."""
        import os

        assert os.path.isfile(file_dialogs._CHILD)

    def test_it_compiles(self):
        import py_compile

        py_compile.compile(file_dialogs._CHILD, doraise=True)

    def test_it_never_asks_for_a_directory(self):
        """askdirectory is the old tree dialog on Windows. Picking a file and taking
        its parent is what keeps the modern dialog on every platform, so a return to
        askdirectory is a regression, not a refactor."""
        from pathlib import Path

        source = Path(file_dialogs._CHILD).read_text(encoding="utf-8")
        assert "askdirectory" not in source.split('"""', 2)[-1]


class TestFillingTheInputs:
    """`_ask` is patched, so the dialog is never opened — what is under test is what
    the page does with a path once it has one."""

    def test_metadata_fills_both_inputs(self, app, monkeypatch):
        chosen = "/runs/july/metadata_exp7.xlsx"
        monkeypatch.setattr(file_dialogs, "_ask", lambda **k: chosen)
        at = app(page="data_import")
        at.button(key="browse_metadata").click().run()
        assert at.session_state["metadata_path_input"] == chosen
        assert at.session_state["data_dir_input"] == "/runs/july"

    def test_data_folder_uses_the_parent_of_the_picked_file(self, app, monkeypatch):
        """The dialog asks for a Monitor file because that gets the modern Windows
        dialog; the folder is what the page actually wants."""
        monkeypatch.setattr(file_dialogs, "_ask", lambda **k: "/runs/july/Monitor01.txt")
        at = app(page="data_import")
        at.button(key="browse_data_dir").click().run()
        assert at.session_state["data_dir_input"] == "/runs/july"

    def test_netcdf_fills_its_own_input(self, app, monkeypatch):
        monkeypatch.setattr(file_dialogs, "_ask", lambda **k: "/runs/july/saved.nc")
        at = app(page="data_import")
        at.button(key="browse_nc").click().run()
        assert at.session_state["nc_path_input"] == "/runs/july/saved.nc"

    def test_cancelling_leaves_the_inputs_alone(self, app, monkeypatch):
        monkeypatch.setattr(file_dialogs, "_ask", lambda **k: "")
        at = app(page="data_import")
        at.session_state["metadata_path_input"] = "typed/by/hand.csv"
        at.button(key="browse_metadata").click().run()
        assert at.session_state["metadata_path_input"] == "typed/by/hand.csv"

    def test_an_unavailable_dialog_warns_and_keeps_the_typed_path(self, app, monkeypatch):
        """The whole fallback story: no dialog, no crash, and the text input the
        user can still paste into is untouched."""

        def _unavailable(**k):
            raise DialogUnavailable("No display.")

        monkeypatch.setattr(file_dialogs, "_ask", _unavailable)
        at = app(page="data_import")
        at.session_state["metadata_path_input"] = "typed/by/hand.csv"
        at.button(key="browse_metadata").click().run()
        assert not at.exception
        assert at.session_state["metadata_path_input"] == "typed/by/hand.csv"
        assert any("paste the path" in w.value for w in at.warning)
