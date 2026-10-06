"""The ``clockwork`` command.

    clockwork run config.yaml [...]        run each config file
    clockwork validate config.yaml [...]   check config files without running them
    clockwork init [config.yaml]           write a commented starting config
    clockwork schema                       print the config's JSON Schema
    clockwork gui [streamlit options...]   launch the Streamlit app
    clockwork scamp [options...]           curate + split + export to SCAMP format

See docs/cli-config.md for the config file.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import traceback
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


# ---------------------------------------------------------------------------
# validate / run
# ---------------------------------------------------------------------------


def _err(msg: str = "") -> None:
    print(msg, file=sys.stderr)


def _plan(loaded) -> str:
    """One line per step the file runs, so a valid file can be read back."""
    cfg = loaded.config
    lines = []
    if cfg.inputs.dataset is not None:
        lines.append(f"load      {cfg.inputs.dataset}")
    else:
        lines.append(f"import    {cfg.inputs.metadata} + {cfg.inputs.monitors}")
    if cfg.groups is not None:
        lines.append(
            "groups    by "
            + (", ".join(cfg.groups.by) if cfg.groups.by else "the default columns")
            + (", subset" if cfg.groups.keep else "")
        )
    if cfg.curation is not None:
        lines.append("curation")
    if cfg.split is not None:
        lines.append("split     LD/DD")
    an = cfg.analyses
    if an.period is not None:
        from clockwork.pipeline.period import METHOD_KEYS

        methods = [k for k in METHOD_KEYS if getattr(an.period.methods, k) is not None]
        lines.append(f"period    {', '.join(methods)} ({an.period.phase})")
    if an.sleep is not None:
        lines.append(f"sleep     threshold {an.sleep.threshold_seconds} s")
    if an.hmm is not None:
        lines.append(f"hmm       {an.hmm.preset} preset, fitted on {an.hmm.phase}")
    lines.append(f"outputs   {loaded.output_dir}")
    return "\n".join(f"  {line}" for line in lines)


def _load(path):
    """``(loaded, report)``, or None after printing why the file cannot be used."""
    from clockwork.pipeline.experiment import ConfigError, check_config, load_config

    try:
        loaded = load_config(path)
    except ConfigError as e:
        _err(f"{e.path}: not valid")
        for p in e.problems:
            _err(f"  error: {p}")
        return None
    return loaded, check_config(loaded)


def _validate(args) -> int:
    failed = 0
    for path in args.config:
        got = _load(path)
        if got is None:
            failed += 1
            continue
        loaded, report = got
        for w in report.warnings:
            _err(f"  warning: {w}")
        for e in report.errors:
            _err(f"  error: {e}")
        if report.ok:
            print(f"{loaded.path}: OK" + (f" ({len(report.warnings)} warning(s))" if report.warnings else ""))
            print(_plan(loaded))
        else:
            _err(f"{loaded.path}: {len(report.errors)} error(s)")
            failed += 1
    return 1 if failed else 0


class _ProgressLine:
    """A single updating ``step  done/total`` line, on a terminal only."""

    def __init__(self, stream):
        self.stream = stream
        self.live = stream.isatty()

    def __call__(self, step):
        if not self.live:
            return None

        def update(done, total):
            self.stream.write(f"\r  {step} {done}/{total}  ")
            if done >= total:
                self.stream.write("\r" + " " * 40 + "\r")
            self.stream.flush()

        return update


def _run(args) -> int:
    from clockwork.core.dam_processor import MetadataError
    from clockwork.pipeline.data import ImportFailed
    from clockwork.pipeline.experiment import ConfigError
    from clockwork.pipeline.run import OutputsExist, run_experiment

    failed = []
    for path in args.config:
        got = _load(path)
        if got is None:
            failed.append(path)
            continue
        loaded, report = got
        if not report.ok:
            for e in report.errors:
                _err(f"  error: {e}")
            _err(f"{loaded.path}: not run ({len(report.errors)} error(s)); see clockwork validate")
            failed.append(path)
            continue
        print(f"{loaded.path}")
        try:
            out = run_experiment(
                loaded,
                force=args.force,
                # The terminal as it is now: during a step, sys.stdout is run.log.
                log=lambda msg, out=sys.stdout: print(f"  {msg}", file=out, flush=True),
                progress=_ProgressLine(sys.stdout),
                verbose=args.verbose,
            )
        except (ConfigError, OutputsExist, ImportFailed, MetadataError) as e:
            _err(f"  error: {e}")
            failed.append(path)
            continue
        except KeyboardInterrupt:
            _err("  interrupted")
            return 130
        except Exception:
            _err(traceback.format_exc())
            _err(f"  error: {path} stopped on an unexpected error (above)")
            failed.append(path)
            continue
        if out.report is not None:
            print(f"  QC report: {out.report}")
    if len(args.config) > 1:
        print(f"{len(args.config) - len(failed)} of {len(args.config)} config(s) ran")
    for path in failed:
        _err(f"failed: {path}")
    return 1 if failed else 0


# ---------------------------------------------------------------------------
# init / schema
# ---------------------------------------------------------------------------


def _init(args) -> int:
    from clockwork.pipeline.init_template import init_text

    out = Path(args.path)
    if out.exists() and not args.force:
        _err(f"{out} already exists; pass --force to overwrite it")
        return 1
    try:
        text = init_text(out, metadata=args.metadata, monitors=args.monitors)
    except (OSError, ValueError) as e:
        _err(f"error: {e}")
        return 1
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(f"wrote {out}")
    print(f"next:  clockwork validate {out}")
    return 0


def _schema(args) -> int:
    from clockwork.pipeline.experiment import json_schema

    text = json.dumps(json_schema(), indent=2)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
        print(f"wrote {args.output}")
    else:
        print(text)
    return 0


# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="clockwork",
        description="Analysis of Drosophila Activity Monitor (DAM) data.",
    )
    parser.add_argument("--version", action="version", version=f"clockwork {__version__}")
    sub = parser.add_subparsers(dest="command", metavar="<command>")

    p = sub.add_parser("run", help="run config file(s)", description="Run each config file in turn.")
    p.add_argument("config", nargs="+", help="config file(s) (.yaml)")
    p.add_argument("--force", action="store_true", help="overwrite existing outputs")
    p.add_argument(
        "-v", "--verbose", action="store_true", help="show the analyses' own output (else it goes to run.log)"
    )
    p.set_defaults(func=_run)

    p = sub.add_parser(
        "validate",
        help="check config file(s) without running them",
        description="Check each config file: settings, files, metadata columns, and that "
        "every analysis has the steps it needs.",
    )
    p.add_argument("config", nargs="+", help="config file(s) (.yaml)")
    p.set_defaults(func=_validate)

    p = sub.add_parser(
        "init",
        help="write a commented starting config",
        description="Write a commented config to start from. Given a metadata file, it is "
        "filled in from that file's columns.",
    )
    p.add_argument("path", nargs="?", default="clockwork.yaml", help="file to write (default: clockwork.yaml)")
    p.add_argument("--metadata", help="the metadata file (.xlsx or .csv)")
    p.add_argument("--monitors", help="the folder of monitor files (default: the metadata's folder)")
    p.add_argument("--force", action="store_true", help="overwrite an existing file")
    p.set_defaults(func=_init)

    p = sub.add_parser(
        "schema",
        help="print the config file's JSON Schema",
        description="Print the JSON Schema for config files, for editor autocompletion.",
    )
    p.add_argument("-o", "--output", help="write to this file instead of printing")
    p.set_defaults(func=_schema)

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
    if args.command is None:
        parser.print_help()
        return 1
    if extra:
        parser.error(f"unrecognized arguments: {' '.join(extra)}")
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
