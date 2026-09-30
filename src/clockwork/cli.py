"""The ``clockwork`` command.

    clockwork gui [streamlit options...]   launch the Streamlit app
    clockwork scamp [options...]           curate + split + export to SCAMP format

``run`` / ``validate`` / ``init`` (the YAML-driven batch pipeline) will land here
once the pipeline layer exists; see the refactor plan.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from clockwork import __version__

APP_ENTRYPOINT = Path(__file__).resolve().parent / "app" / "ClockWork.py"


def _gui(extra: list[str]) -> int:
    # A subprocess rather than streamlit.web.cli in-process, so the app runs
    # exactly as `streamlit run` runs it, which is what the AppTest suite and
    # every existing user has exercised.
    #
    # theme.base=light is what .streamlit/config.toml sets for a checkout; an
    # installed copy has no such file next to the user's cwd, so pass it here.
    # Anything the user passes after `gui` comes later and wins.
    cmd = [
        sys.executable, "-m", "streamlit", "run", str(APP_ENTRYPOINT),
        "--theme.base=light",
        *extra,
    ]
    try:
        return subprocess.call(cmd)
    except KeyboardInterrupt:
        return 130


def _scamp(extra: list[str]) -> int:
    from clockwork.scamp_export.curate_and_export import main as scamp_main

    return scamp_main(extra)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="clockwork",
        description="Analysis of Drosophila Activity Monitor (DAM) data.",
    )
    parser.add_argument("--version", action="version", version=f"clockwork {__version__}")
    sub = parser.add_subparsers(dest="command", metavar="<command>")
    sub.add_parser(
        "gui",
        help="launch the Streamlit app",
        description="Launch the Streamlit app. Extra options go to `streamlit run`.",
        add_help=False,
    )
    sub.add_parser(
        "scamp",
        help="curate, split and export DAM data to SCAMP format",
        add_help=False,
    )

    args, extra = parser.parse_known_args(argv)
    if args.command == "gui":
        return _gui(extra)
    if args.command == "scamp":
        return _scamp(extra)
    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
