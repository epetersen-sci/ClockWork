"""An in-app folder browser, so paths can be clicked rather than typed.

Streamlit has no directory picker. ``st.file_uploader`` hands back bytes with no
path and cannot select a folder at all, but the loader needs a real directory of
monitor files on disk — it reads them itself. A native OS dialog (tkinter) would
run on the server, which is the right machine when the app is local, but it blocks
the Streamlit thread and can open behind the browser window. So this walks the
filesystem inside the page instead: always available, never blocks, and behaves the
same whether the app is local or remote.

The picker WRITES INTO the text inputs' session-state keys rather than replacing
the inputs, so typing or pasting a path keeps working exactly as before, and the
value it sets is visible and editable afterwards. Streamlit only allows that write
before the widget for a key is created on that run, which is why
:func:`browse` has to be rendered above the inputs it fills.
"""

import os
import string

import streamlit as st

#: Extensions offered as metadata files. Lower-cased at compare time, so a
#: METADATA.XLSX is listed — dam_processor lower-cases its check for the same reason.
METADATA_EXTENSIONS = (".xlsx", ".xls", ".csv")


def _drives():
    """Root candidates: Windows drive letters, or '/' elsewhere."""
    if os.name != "nt":
        return ["/"]
    return [f"{letter}:\\" for letter in string.ascii_uppercase if os.path.exists(f"{letter}:\\")]


def _subdirs(path):
    """Visible subfolder names, case-insensitively sorted. A folder that cannot be
    read lists as empty rather than raising — an unreadable folder is a normal
    thing to browse past, not an error worth stopping on."""
    try:
        with os.scandir(path) as it:
            return sorted(
                (e.name for e in it if e.is_dir() and not e.name.startswith(".")),
                key=str.lower,
            )
    except OSError:
        return []


def _files(path, extensions):
    """Files in ``path`` whose extension is in ``extensions`` (already lower-cased)."""
    try:
        with os.scandir(path) as it:
            names = [
                e.name
                for e in it
                if e.is_file()
                # Excel writes a ~$ lock file beside any open workbook. It is not a
                # metadata file and offering it only produces a confusing read error.
                and not e.name.startswith("~$")
                and os.path.splitext(e.name)[1].lower() in extensions
            ]
    except OSError:
        return []
    return sorted(names, key=str.lower)


def _count_monitor_files(path):
    """How many Monitor*.txt files a folder holds — the one signal that says
    "this is the data directory" before you commit to it."""
    try:
        with os.scandir(path) as it:
            return sum(
                1
                for e in it
                if e.is_file()
                and e.name.lower().startswith("monitor")
                and e.name.lower().endswith(".txt")
            )
    except OSError:
        return 0


def _initial_dir(start):
    """Where the browser opens: the caller's suggestion, else the desktop, else
    home, else wherever the app was launched from."""
    for candidate in (start, os.path.expanduser("~/Desktop"), os.path.expanduser("~")):
        if candidate and os.path.isdir(candidate):
            return os.path.abspath(candidate)
    return os.path.abspath(os.getcwd())


def browse(
    key,
    *,
    start=None,
    file_extensions=METADATA_EXTENSIONS,
    dir_target=None,
    file_target=None,
    file_label="Metadata file",
):
    """Render the browser and return the folder it is currently showing.

    ``dir_target`` and ``file_target`` name the session-state keys of the text
    inputs the two "Use this…" buttons fill in. Either may be omitted to hide that
    button — the NetCDF tab wants a file and no data directory.

    ``key`` namespaces this picker's own widgets, so two browsers can sit on one
    page without sharing a current folder.
    """
    dir_key = f"_browse_dir_{key}"
    sel_key = f"_browse_sel_{key}"

    if dir_key not in st.session_state:
        st.session_state[dir_key] = _initial_dir(start)

    current = st.session_state[dir_key]
    if not os.path.isdir(current):
        # The remembered folder has been moved or unmounted since it was picked.
        current = _initial_dir(None)
        st.session_state[dir_key] = current

    st.caption("Current folder")
    st.code(current, language=None)

    def _descend():
        choice = st.session_state.get(sel_key)
        if choice and choice != "—":
            target = (
                os.path.dirname(current)
                if choice == ".. (up one level)"
                else os.path.join(current, choice)
            )
            if os.path.isdir(target):
                st.session_state[dir_key] = os.path.abspath(target)
        # Reset to the no-op entry so picking the same subfolder twice in a row
        # still fires on_change the second time.
        st.session_state[sel_key] = "—"

    nav1, nav2 = st.columns([3, 2])
    with nav1:
        options = ["—"]
        if os.path.dirname(current) != current:
            options.append(".. (up one level)")
        options += _subdirs(current)
        st.selectbox(
            "Open a subfolder",
            options,
            key=sel_key,
            on_change=_descend,
            help="Pick a folder to move into it. '—' does nothing.",
        )
    with nav2:
        roots = _drives()
        if roots:
            root_key = f"{sel_key}_root"

            def _to_root():
                root = st.session_state.get(root_key)
                if root and os.path.isdir(root):
                    st.session_state[dir_key] = root

            st.selectbox("Jump to drive", roots, key=root_key, on_change=_to_root)

    n_monitors = _count_monitor_files(current)
    if n_monitors:
        st.caption(f"This folder holds **{n_monitors}** `Monitor*.txt` file(s).")

    b1, b2 = st.columns(2)
    with b1:
        if dir_target and st.button(
            "Use this folder as the data directory", key=f"{key}_use_dir"
        ):
            st.session_state[dir_target] = current
            st.session_state[f"_picked_dir_{key}"] = current
    with b2:
        found = _files(current, {e.lower() for e in file_extensions})
        file_key = f"{sel_key}_file"
        if found:
            st.selectbox(f"{file_label} in this folder", found, key=file_key)
            if file_target and st.button(
                f"Use this {file_label.lower()}", key=f"{key}_use_file"
            ):
                chosen = os.path.join(current, st.session_state[file_key])
                st.session_state[file_target] = chosen
                st.session_state[f"_picked_file_{key}"] = chosen
        else:
            st.caption("No " + " / ".join(file_extensions) + " file in this folder.")

    # The confirmations persist across reruns: the button press that set the path
    # is over by the time the text input below shows it, and without this the
    # picker looks like it did nothing.
    for note_key, label in (
        (f"_picked_dir_{key}", "Data directory"),
        (f"_picked_file_{key}", file_label),
    ):
        picked = st.session_state.get(note_key)
        if picked:
            st.success(f"{label} set to `{picked}`")

    return current
