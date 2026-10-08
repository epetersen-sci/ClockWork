"""Native OS file pickers, offered from the import page.

Streamlit has no path picker. ``st.file_uploader`` hands back bytes with no path
and cannot select a folder at all, while the loader needs a real directory on disk
because it reads the monitor files itself. So the paths had to be typed or pasted,
and a typo is indistinguishable from a wrong folder until the import fails.

These open the REAL dialog — the Explorer window on Windows, the Finder panel on
macOS — by running :mod:`ui._dialog_child` in its own process. In-process is what
makes tkinter unreliable under Streamlit: it wants a main thread and its own event
loop, and the script runner is neither. A subprocess has both, and cannot take the
server down with it.

Everything is a FILE dialog, including choosing a folder, which is picked by
selecting any file inside it. See :mod:`ui._dialog_child` for why: on Windows the
folder dialog is the old tree widget while the file dialog is the modern Explorer
window, so going through a file gets the good one on every platform.

**The dialog opens on the machine RUNNING the app, not the one viewing it.** Those
are the same machine for a local ``streamlit run``, which is how ClockWork is used.
Served from a shared machine they are not, and the dialog would open on a desktop
nobody is looking at — so a dialog that cannot open is not an error here. The text
inputs take a pasted path and always have; that is the fallback, and it is the same
one on every platform.
"""

import json
import os
import subprocess
import sys

import streamlit as st

#: Long enough that nobody hunting through folders gets cut off, short enough that a
#: dialog opened where no one can see it (a remote server) eventually releases the
#: rerun instead of holding it forever.
TIMEOUT_SECONDS = 300

#: Patterns stay simple — a bare ``*.ext`` — because Windows matches the wildcard
#: itself while macOS reads only the extension. A pattern like ``Monitor*.txt`` would
#: filter correctly on one platform and unpredictably on the other.
METADATA_FILETYPES = [
    ["Metadata files", "*.xlsx *.xls *.csv"],
    ["Excel", "*.xlsx *.xls"],
    ["CSV", "*.csv"],
    ["All files", "*"],
]
MONITOR_FILETYPES = [["Monitor files", "*.txt"], ["All files", "*"]]
NETCDF_FILETYPES = [["NetCDF datasets", "*.nc"], ["All files", "*"]]

_CHILD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_dialog_child.py")


class DialogUnavailable(RuntimeError):
    """No dialog could be shown — no display, no tkinter, or it never returned."""


def _ask(*, initial=None, filetypes=None, title="Select a file"):
    """Run the child process and return the chosen path, or ``''`` if cancelled.

    Raises :class:`DialogUnavailable` when the dialog could not be shown at all,
    which the caller reports as "type the path instead" rather than as a failure.
    """
    argv = [sys.executable, _CHILD, str(initial or ""), json.dumps(filetypes or []), title]
    try:
        done = subprocess.run(argv, capture_output=True, text=True, timeout=TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired as e:
        raise DialogUnavailable(
            f"The dialog was still open after {TIMEOUT_SECONDS // 60} minutes. If you "
            "cannot see it, the app is running on a different machine than this browser."
        ) from e
    except OSError as e:
        raise DialogUnavailable(f"Could not start the dialog: {e}") from e

    if done.returncode != 0:
        detail = (done.stderr or "").strip().splitlines()
        raise DialogUnavailable(detail[-1] if detail else "The dialog failed to open.")
    return done.stdout.strip()


def _browse(key, targets, *, initial=None, filetypes=None, title, label, help=None):
    """Render one Browse button. On a choice, fill ``targets`` and report.

    ``targets`` maps a text input's session-state key to a function turning the
    chosen path into that input's value — which is how picking one file can fill
    both the metadata path and the data directory.

    Writing another widget's session-state key is only legal BEFORE that widget is
    created on the same run, so every caller renders these buttons above the inputs
    they fill. The confirmation is stored rather than drawn directly, because the
    click is over by the time the input below shows the value.
    """
    note_key = f"_dialog_note_{key}"
    if st.button(label, key=key, help=help):
        try:
            with st.spinner("Waiting for the dialog…"):
                chosen = _ask(initial=initial, filetypes=filetypes, title=title)
        except DialogUnavailable as e:
            st.session_state[note_key] = ("warning", f"{e} You can paste the path below.")
        else:
            if chosen:
                for target, derive in targets.items():
                    st.session_state[target] = derive(chosen)
                st.session_state[note_key] = ("success", f"Selected `{chosen}`")
            # A cancelled dialog is a decision, not an event worth reporting.

    note = st.session_state.get(note_key)
    if note:
        level, message = note
        (st.success if level == "success" else st.warning)(message)


def browse_metadata_file(*, initial=None, metadata_target, data_dir_target, key="browse_metadata"):
    """Pick the metadata file, and set the data directory to the folder holding it.

    Both from one dialog because the metadata file normally sits with the monitor
    files, and the codebase already assumes it does — ``resolve_export_dir`` treats
    that directory as the working folder. When the monitors live elsewhere,
    :func:`browse_data_folder` overrides the directory afterwards.
    """
    _browse(
        key,
        {metadata_target: lambda p: p, data_dir_target: os.path.dirname},
        initial=initial,
        filetypes=METADATA_FILETYPES,
        title="Select the metadata file",
        label="Browse for metadata file…",
        help="Also sets the data directory to the folder the metadata file is in.",
    )


def browse_data_folder(*, initial=None, data_dir_target, key="browse_data_dir"):
    """Pick the folder holding the MonitorXXX.txt files, by picking one of them.

    Selecting a file rather than the folder is deliberate: it gets the modern
    dialog on Windows (see the module docstring), and the ``*.txt`` filter doubles
    as confirmation — a folder showing no monitor files in the dialog is the wrong
    folder, which is visible there instead of after a failed import.
    """
    _browse(
        key,
        {data_dir_target: os.path.dirname},
        initial=initial,
        filetypes=MONITOR_FILETYPES,
        title="Select any Monitor file in the data folder",
        label="Browse for data folder…",
        help="Pick any MonitorXXX.txt file in the folder; the folder itself is used. "
        "Only needed when the monitor files are somewhere other than beside the "
        "metadata file.",
    )


def browse_netcdf_file(*, initial=None, nc_target, key="browse_nc"):
    """Pick a saved ``.nc`` dataset."""
    _browse(
        key,
        {nc_target: lambda p: p},
        initial=initial,
        filetypes=NETCDF_FILETYPES,
        title="Select a saved .nc dataset",
        label="Browse for .nc file…",
    )
