"""Opens one native file dialog and prints the chosen path. Run as a script.

This lives in its own process on purpose. tkinter wants a main thread and an event
loop of its own; Streamlit's script runner is neither, and a dialog opened inside it
can hang the server or never draw. As a subprocess it gets both, and the worst case
is a process the parent can kill.

Only ever a FILE dialog, never ``askdirectory``. On Windows those are two different
dialogs: ``askopenfilename`` is the modern Explorer window with an address bar,
Quick access and search, while ``askdirectory`` is the old narrow "Browse For
Folder" tree. A folder is therefore chosen by picking any file inside it and taking
its parent — one good dialog on every platform instead of a good one on macOS and a
dated one on Windows.

Arguments: ``<initialdir> <json filetypes> <title>``. Prints the selected path to
stdout, or nothing if the dialog was cancelled.
"""

import json
import sys
import tkinter as tk
from tkinter import filedialog


def main():
    initial = sys.argv[1] if len(sys.argv) > 1 else ""
    types = json.loads(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2] else []
    title = sys.argv[3] if len(sys.argv) > 3 else "Select a file"

    root = tk.Tk()
    root.withdraw()
    # Three separate nudges, because no one of them is enough everywhere. -topmost
    # keeps the dialog above the browser on Windows; lift() and focus_force() are
    # what macOS needs, where a process not launched from the Dock does not take
    # focus on its own and the dialog can open behind the window that asked for it,
    # with only a bouncing Dock icon to say so.
    root.wm_attributes("-topmost", True)
    root.lift()
    root.focus_force()
    root.update()

    path = filedialog.askopenfilename(
        parent=root,
        initialdir=initial or None,
        title=title,
        filetypes=[tuple(t) for t in types] or None,
    )

    root.destroy()
    # Cancel returns '' (an empty tuple on some platforms), which prints as nothing
    # and the caller reads as "no choice made".
    sys.stdout.write(str(path or ""))


if __name__ == "__main__":
    main()
