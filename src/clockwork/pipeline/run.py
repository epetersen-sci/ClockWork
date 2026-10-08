"""``clockwork run``: one config file, start to finish.

The steps are the ones the GUI calls, in the order the GUI takes them: import,
groups, curation, the LD/DD split, then period, sleep and the HMM. What this
module adds is the bookkeeping of an unattended run — where the outputs go,
never overwriting them by accident, and recording exactly what ran.

Outputs, in ``<outputs.dir>/<experiment>/``:

- ``<experiment>.nc`` — the analysed dataset, with the resolved config in its attrs
- ``config.resolved.yaml`` — every value used, defaults included, plus versions
- ``tables/<name>.csv`` — the result tables
- ``scalograms/`` — the CWT's group-averaged scalograms, when asked for
- ``qc_report.html`` — the figures to check the run by eye
- ``run.log`` — everything the analyses printed (per-fly exclusions and the like)
"""

from __future__ import annotations

import contextlib
import datetime as _dt
import io
import json
import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from clockwork.core import dam_utilities
from clockwork.core.load_and_save_datasets import save_dataset_to_netcdf
from clockwork.pipeline import tables as _tables
from clockwork.pipeline.data import (
    apply_groups,
    build_dataset,
    curate,
    load_netcdf,
    read_monitors,
    split,
    split_report,
)
from clockwork.pipeline.experiment import (
    ConfigError,
    LoadedConfig,
    check_config,
    dump_yaml,
    provenance,
    resolved_config,
)
from clockwork.pipeline.hmm import run_hmm
from clockwork.pipeline.period import run_period
from clockwork.pipeline.report import RunRecord, build_report
from clockwork.pipeline.scalograms import save_group_average_scalograms
from clockwork.pipeline.sleep import detect_sleep

#: The attr the resolved config is stored under on the saved dataset.
RUN_CONFIG_ATTR = "clockwork_run_config"

Progress = Callable[[int, int], None]


class OutputsExist(FileExistsError):
    """The output folder already has files in it, and --force was not given."""

    def __init__(self, path: Path):
        self.path = path
        super().__init__(
            f"{path} already has files in it; pass --force to overwrite them, or "
            "change experiment: or outputs.dir"
        )


@dataclass
class RunOutputs:
    out_dir: Path
    resolved: Path
    dataset: Path | None = None
    tables: dict[str, Path] = field(default_factory=dict)
    report: Path | None = None
    #: The CWT group-averaged scalograms written (pipeline.scalograms' manifest).
    scalograms: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def run_experiment(
    loaded: LoadedConfig,
    *,
    force: bool = False,
    log: Callable[[str], None] | None = None,
    progress: Callable[[str], Progress | None] | None = None,
    verbose: bool = False,
) -> RunOutputs:
    """Run ``loaded`` and write its outputs.

    ``log`` gets a line per step; ``progress(step)`` may return a ``(done,
    total)`` callback for that step's per-fly progress. What the analyses print
    themselves goes to ``run.log`` rather than the terminal, unless ``verbose``.

    Raises :class:`ConfigError` when the file does not pass
    :func:`check_config`, and :class:`OutputsExist` before doing any work if it
    would overwrite.
    """
    cfg = loaded.config
    log = log or (lambda _msg: None)
    progress = progress or (lambda _step: None)
    out_dir = loaded.output_dir
    if out_dir.is_dir() and any(out_dir.iterdir()) and not force:
        raise OutputsExist(out_dir)

    checked = check_config(loaded)
    if not checked.ok:
        raise ConfigError(loaded.path, checked.errors)
    record = RunRecord(
        experiment=loaded.experiment,
        config_yaml=dump_yaml(cfg.overrides()),
        started=_dt.datetime.now(),
        # The output-folder warning is answered: we got here, so --force was given.
        warnings=[w for w in checked.warnings if "--force" not in w],
    )
    for w in record.warnings:
        log(f"warning: {w}")

    core_log = io.StringIO()

    @contextlib.contextmanager
    def step(name):
        log(f"{name}...")
        core_log.write(f"\n=== {name} ===\n")
        t0 = time.perf_counter()
        # stderr too: tqdm's progress bars and library warnings go there.
        with contextlib.ExitStack() as quiet:
            if not verbose:
                quiet.enter_context(contextlib.redirect_stdout(core_log))
                quiet.enter_context(contextlib.redirect_stderr(core_log))
            yield
        record.timings[name] = time.perf_counter() - t0

    try:
        return _run(loaded, record, step, log, progress, core_log)
    except BaseException:
        # What the analyses said before it stopped is often the explanation.
        if not verbose and core_log.getvalue().strip():
            tail = core_log.getvalue().strip().splitlines()[-25:]
            log("last output before the error:\n    " + "\n    ".join(tail))
        raise


def _run(loaded, record, step, log, progress, core_log) -> RunOutputs:
    cfg = loaded.config
    out_dir = loaded.output_dir

    # -- the dataset -----------------------------------------------------------
    with step("import"):
        if cfg.inputs.dataset is not None:
            ds = load_netcdf(cfg.inputs.dataset)
            if cfg.groups is not None:
                ds = apply_groups(ds, cfg.groups)
            if cfg.experiment is not None:
                ds.attrs["experiment_name"] = loaded.experiment
        else:
            raw = read_monitors(cfg.inputs, progress=progress("import"))
            record.n_imported = raw.n_flies
            ds = build_dataset(raw, cfg.inputs, cfg.groups, experiment_name=loaded.experiment)
        log(f"  {ds.sizes['id']} flies")

    if cfg.curation is not None:
        with step("curation"):
            result = curate(ds, cfg.curation, progress=progress("curation"))
            record.curation = result
            ds = result.live
            log(f"  {result.removed} removed, {result.trimmed} trimmed, {result.total_after} kept")

    if cfg.split is not None:
        with step("split"):
            record.split_report = split_report(ds, cfg.split)
            ds = split(ds, cfg.split)

    scalogram_arrays: list[dict] = []
    # -- the analyses ------------------------------------------------------------
    an = cfg.analyses
    if an.period is not None:
        with step("period"):
            ds = run_period(
                ds, an.period, progress=progress("period"), on_scalograms=scalogram_arrays.extend
            )
    if an.sleep is not None:
        with step("sleep"):
            ds = detect_sleep(ds, an.sleep, progress=progress("sleep"))
    if an.hmm is not None:
        with step("hmm"):
            ds = run_hmm(ds, an.hmm, progress=progress("hmm")).master

    # -- outputs ---------------------------------------------------------------------
    record.finished = _dt.datetime.now()
    out_dir.mkdir(parents=True, exist_ok=True)
    # Only now, with the analyses done: a run that fails leaves the last outputs.
    _clear_previous_outputs(out_dir, loaded.experiment)
    outputs = RunOutputs(out_dir=out_dir, resolved=out_dir / "config.resolved.yaml")
    outputs.warnings = list(record.warnings)
    if scalogram_arrays:
        with step("scalograms"):
            # Before the dataset is saved, so it records where they went.
            outputs.scalograms = save_group_average_scalograms(
                scalogram_arrays, str(out_dir / "scalograms"), ds=ds, timestamp=False
            )
            record.scalograms = outputs.scalograms

    resolved = resolved_config(cfg, group_columns=dam_utilities.get_group_columns(ds) or None)
    prov = provenance() | {
        "config_file": str(loaded.path),
        "extends": [str(p) for p in loaded.chain[1:]],
        "started": record.started.isoformat(timespec="seconds"),
        "finished": record.finished.isoformat(timespec="seconds"),
    }
    outputs.resolved.write_text(
        dump_yaml(
            {**resolved, "provenance": prov},
            header=(
                "Every value this run used, defaults included. Runnable as it stands:\n"
                "  clockwork run config.resolved.yaml\n"
                "(provenance is a record, and is ignored when the file is read.)"
            ),
        ),
        encoding="utf-8",
    )

    ds.attrs[RUN_CONFIG_ATTR] = json.dumps({**resolved, "provenance": prov}, default=str)
    if cfg.outputs.dataset:
        with step("save dataset"):
            outputs.dataset = out_dir / f"{loaded.experiment}.nc"
            save_dataset_to_netcdf(ds, str(outputs.dataset))

    names = cfg.outputs.tables if cfg.outputs.tables is not None else _tables.available(ds)
    if names:
        with step("tables"):
            (out_dir / "tables").mkdir(exist_ok=True)
            for name in names:
                path = out_dir / "tables" / f"{name}.csv"
                _tables.TABLES[name](ds).to_csv(path, index=False)
                outputs.tables[name] = path

    if cfg.outputs.qc_report:
        with step("qc report"):
            outputs.report = out_dir / "qc_report.html"
            outputs.report.write_text(build_report(ds, record, cfg), encoding="utf-8")
    (out_dir / "run.log").write_text(core_log.getvalue(), encoding="utf-8")
    log(f"done: {out_dir}")
    return outputs


#: What a run writes into its folder, by name. --force replaces exactly these.
_OUTPUT_FILES = ("config.resolved.yaml", "qc_report.html", "run.log")
_OUTPUT_DIRS = ("tables", "scalograms")


def _clear_previous_outputs(out_dir: Path, experiment: str) -> None:
    """Remove an earlier run's outputs before this run writes its own.

    Without this, --force overwrote the files this run writes and left the rest:
    drop the HMM from a config and rerun, and the old hmm_*.csv sat beside the
    new tables looking like part of the new run. Only ClockWork's own outputs go;
    anything else someone put in the folder stays.
    """
    for name in (*_OUTPUT_FILES, f"{experiment}.nc"):
        (out_dir / name).unlink(missing_ok=True)
    for name in _OUTPUT_DIRS:
        if (out_dir / name).is_dir():
            shutil.rmtree(out_dir / name)
