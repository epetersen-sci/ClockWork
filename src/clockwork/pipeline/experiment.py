"""The config file: one experiment, from the raw files to the outputs.

:class:`ExperimentConfig` is the whole file. Each section is the config of the
step that runs it (``pipeline.data``, ``period``, ``sleep``, ``hmm``), so the file
and the GUI describe an analysis with the same objects. This module adds what
only a FILE needs:

- reading YAML, with ``extends:`` and paths relative to the file that names
  them (:func:`load_config`);
- the checks no single section can make on its own — a step whose upstream is
  missing, a column the metadata does not have (:func:`check_config`);
- the resolved config a run writes beside its outputs (:func:`resolved_config`).

See docs/cli-config.md for the rules this follows.
"""

from __future__ import annotations

import copy
import importlib.metadata
import os
import platform
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, ValidationError, field_validator, model_validator

from clockwork.pipeline.data import (
    _COLUMN_FOR,
    CurationConfig,
    GroupsConfig,
    InputsConfig,
    SplitConfig,
)
from clockwork.pipeline.hmm import TUNABLE, HmmConfig
from clockwork.pipeline.period import METHOD_KEYS, PeriodConfig, Preprocessing, _Classified
from clockwork.pipeline.sleep import SleepConfig

#: The schema version this ClockWork reads and writes. A file names the version
#: it was written for; an older one is migrated on load (docs rule 11), a newer
#: one is refused rather than half-understood.
CONFIG_VERSION = 1

#: Analyses that exist in the GUI but not (yet) in a config file. Naming one is
#: an error that says so, rather than "unknown key".
GUI_ONLY = {"phase_shift": "Phase shift", "sleep_deprivation": "Sleep deprivation"}

TableName = Literal[
    "period_summary",
    "sleep_summary",
    "sleep_states",
    "sleep_bouts",
    "hmm_occupancy",
    "hmm_states",
    "hmm_zt_fractions",
]
#: The analysis each table is made from.
TABLE_NEEDS: dict[str, str] = {
    "period_summary": "period",
    "sleep_summary": "sleep",
    "sleep_states": "sleep",
    "sleep_bouts": "sleep",
    "hmm_occupancy": "hmm",
    "hmm_states": "hmm",
    "hmm_zt_fractions": "hmm",
}

#: Keys whose value is a path, relative to the file that names it.
_PATH_KEYS = (("inputs", "metadata"), ("inputs", "monitors"), ("inputs", "dataset"), ("outputs", "dir"))
_SOURCE_KEYS = ("metadata", "monitors", "dataset")
#: Sections where an empty entry (``curation:`` with nothing under it) means
#: "run this with its defaults". YAML reads an empty entry as null, and null
#: otherwise means "absent", so without this a step the file NAMES would
#: silently not run.
_RUN_WITH_DEFAULTS = (
    ("groups",),
    ("curation",),
    ("split",),
    ("outputs",),
    ("analyses", "sleep"),
    ("analyses", "hmm"),
    ("analyses", "period"),
    ("analyses", "period", "methods", "lomb_scargle"),
    ("analyses", "period", "methods", "autocorrelation"),
    ("analyses", "period", "methods", "cwt"),
    ("analyses", "period", "methods", "mesa"),
)


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Analyses(_Section):
    """The analyses to run. One that is absent does not run."""

    period: PeriodConfig | None = None
    sleep: SleepConfig | None = None
    hmm: HmmConfig | None = None

    @model_validator(mode="before")
    @classmethod
    def _gui_only(cls, data):
        if isinstance(data, dict):
            named = [GUI_ONLY[k] for k in data if k in GUI_ONLY]
            if named:
                raise ValueError(
                    f"{' and '.join(named)} can only be run from the GUI (clockwork gui) for now"
                )
        return data


class OutputsConfig(_Section):
    #: The run writes into ``<dir>/<experiment>/``. Relative to the config file.
    dir: str = "results"
    #: Save the analysed dataset (.nc).
    dataset: bool = True
    #: CSV tables to write. None = every table the analyses that ran can make.
    tables: list[TableName] | None = None
    qc_report: bool = True


class ExperimentConfig(_Section):
    clockwork_config: int
    #: Names the output folder. None = the config file's name.
    experiment: str | None = None
    inputs: InputsConfig
    groups: GroupsConfig | None = None
    curation: CurationConfig | None = None
    split: SplitConfig | None = None
    analyses: Analyses = Analyses()
    outputs: OutputsConfig = OutputsConfig()

    @field_validator("clockwork_config")
    @classmethod
    def _known_version(cls, v):
        if v > CONFIG_VERSION:
            raise ValueError(
                f"this file is for config version {v}; this ClockWork reads up to "
                f"{CONFIG_VERSION}. Upgrade ClockWork (pip install -U clockwork-sci)"
            )
        if v < 1:
            raise ValueError("clockwork_config must be 1 or more")
        return v

    def overrides(self) -> dict[str, Any]:
        """The file as it should be written: only what differs from the defaults."""
        out: dict[str, Any] = {"clockwork_config": self.clockwork_config}
        if self.experiment is not None:
            out["experiment"] = self.experiment
        out["inputs"] = self.inputs.overrides()
        for name in ("groups", "curation", "split"):
            section = getattr(self, name)
            if section is not None:
                out[name] = section.overrides()
        analyses = {
            name: getattr(self.analyses, name).overrides()
            for name in ("period", "sleep", "hmm")
            if getattr(self.analyses, name) is not None
        }
        if analyses:
            out["analyses"] = analyses
        outputs = self.outputs.model_dump(mode="json", exclude_defaults=True)
        if outputs:
            out["outputs"] = outputs
        return out


# ---------------------------------------------------------------------------
# Reading a file
# ---------------------------------------------------------------------------


class ConfigError(Exception):
    """A config file that cannot be used. ``problems`` lists every reason found."""

    def __init__(self, path, problems: list[str]):
        self.path = Path(path)
        self.problems = list(problems)
        super().__init__(f"{self.path}: " + "; ".join(self.problems))


@dataclass
class LoadedConfig:
    config: ExperimentConfig
    #: The file named on the command line.
    path: Path
    #: It and every file it extends, nearest first.
    chain: list[Path] = field(default_factory=list)

    @property
    def experiment(self) -> str:
        """The experiment's name: the file's ``experiment:``, else the file's name."""
        from clockwork.core.dam_utilities import sanitize_experiment_name

        return sanitize_experiment_name(self.config.experiment or self.path.stem) or "experiment"

    @property
    def output_dir(self) -> Path:
        # load_config made outputs.dir absolute (relative to the file).
        return (self.path.parent / self.config.outputs.dir).resolve() / self.experiment


def load_config(path) -> LoadedConfig:
    """Read, merge and validate a config file. Raises :class:`ConfigError`."""
    path = Path(path).resolve()
    chain: list[Path] = []
    data = _read_tree(path, chain)
    data.pop("provenance", None)  # written by `clockwork run`; never an input
    try:
        config = ExperimentConfig.model_validate(_migrate(data, path))
    except ValidationError as e:
        raise ConfigError(path, format_validation_error(e)) from None
    # The default output folder is relative to the file too; pin it now, so the
    # config means the same wherever it is written out again.
    out_dir = _relative_to(config.outputs.dir, path.parent)
    config = config.model_copy(update={"outputs": config.outputs.model_copy(update={"dir": str(out_dir)})})
    return LoadedConfig(config, path, chain)


def _read_tree(path: Path, chain: list[Path]) -> dict:
    if path in chain:
        cycle = " -> ".join(p.name for p in [*chain, path])
        raise ConfigError(chain[0], [f"extends: goes round in a circle ({cycle})"])
    if not path.is_file():
        where = f" (extended by {chain[-1].name})" if chain else ""
        raise ConfigError(chain[0] if chain else path, [f"no such file: {path}{where}"])
    chain.append(path)
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as e:
        raise ConfigError(path, [f"not valid YAML: {e}"]) from None
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ConfigError(path, ["the file must be a mapping of section: settings"])
    _named_sections_run(data)
    _resolve_paths(data, path.parent)
    base_name = data.pop("extends", None)
    if base_name is None:
        return data
    if not isinstance(base_name, str):
        raise ConfigError(path, ["extends: takes one file name"])
    base = _read_tree(_relative_to(base_name, path.parent), chain)
    return _merge(base, data)


def _named_sections_run(data: dict) -> None:
    for keys in _RUN_WITH_DEFAULTS:
        parent = data
        for k in keys[:-1]:
            parent = parent.get(k) if isinstance(parent, dict) else None
        if isinstance(parent, dict) and keys[-1] in parent and parent[keys[-1]] is None:
            parent[keys[-1]] = {}


def _relative_to(value: str, base: Path) -> Path:
    p = Path(os.path.expanduser(str(value)))
    return p if p.is_absolute() else (base / p).resolve()


def _resolve_paths(data: dict, base: Path) -> None:
    for section, key in _PATH_KEYS:
        block = data.get(section)
        if isinstance(block, dict) and isinstance(block.get(key), str):
            block[key] = str(_relative_to(block[key], base))


def _merge(base: dict, over: dict) -> dict:
    """``over`` on top of ``base``: mappings merge key by key, anything else is
    replaced. Two exceptions, where merging would change the meaning:
    ``groups.keep`` is replaced whole (merging two subsets keeps neither), and a
    file that names an input source replaces the base's source entirely (raw
    files in one and a .nc in the other would otherwise be both)."""
    out = copy.deepcopy(base)
    for key, value in over.items():
        if (
            key == "inputs"
            and isinstance(value, dict)
            and isinstance(out.get(key), dict)
            and any(k in value for k in _SOURCE_KEYS)
        ):
            out[key] = {k: v for k, v in out[key].items() if k not in _SOURCE_KEYS}
        if key == "keep":
            out[key] = copy.deepcopy(value)
        elif isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _migrate(data: dict, path: Path) -> dict:
    """Bring an older file up to :data:`CONFIG_VERSION`. Version 1 is the first,
    so there is nothing to migrate yet; this is where a renamed key would be
    translated (docs rule 11)."""
    if "clockwork_config" not in data:
        raise ConfigError(
            path, [f"clockwork_config: missing; put `clockwork_config: {CONFIG_VERSION}` at the top"]
        )
    return data


def format_validation_error(e: ValidationError) -> list[str]:
    """pydantic's errors as ``section.key: what is wrong`` lines."""
    out = []
    for err in e.errors():
        loc = [str(p) for p in err["loc"] if not _is_union_tag(p)]
        msg = err["msg"]
        for prefix in ("Value error, ", "Assertion failed, "):
            if msg.startswith(prefix):
                msg = msg[len(prefix):]
        if err["type"] == "extra_forbidden":
            msg = "not a setting here (misspelt?)"
        where = ".".join(loc) or "file"
        line = f"{where}: {msg}"
        if line not in out:
            out.append(line)
    return out


def _is_union_tag(part) -> bool:
    # pydantic names the union member it tried ('list[...]', 'function-after[...]').
    return isinstance(part, str) and ("[" in part or part in {"str", "int", "float", "bool", "none"})


# ---------------------------------------------------------------------------
# Checks across sections
# ---------------------------------------------------------------------------


@dataclass
class CheckReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


@dataclass
class _Source:
    """What the input offers, read without importing it."""

    columns: list[str]  # metadata column names (coord names translated back)
    has_boundary: bool  # an LD/DD boundary can be derived (first_DD_day)
    curated: bool = False
    split: bool = False
    slept: bool = False
    stamped_phase: str | None = None  # a saved LD- or DD-only partition
    ac_classified: bool = False  # carries autocorrelation's rhythmic call


def check_config(loaded: LoadedConfig) -> CheckReport:
    """Everything that can be checked before running (docs: ``clockwork validate``)."""
    cfg = loaded.config
    report = CheckReport()
    source = _inspect_source(cfg.inputs, report)

    # -- groups ------------------------------------------------------------
    if source is not None and cfg.groups is not None:
        from clockwork.core.dam_utilities import GROUP_EXCLUDE_COLUMNS

        for col in cfg.groups.by or []:
            if col not in source.columns:
                report.errors.append(
                    f"groups.by: {col!r} is not a metadata column; there are {source.columns}"
                )
            elif col in GROUP_EXCLUDE_COLUMNS:
                report.errors.append(f"groups.by: {col!r} identifies flies or times, not a group")
        keep = cfg.groups.keep
        combos = [keep] if isinstance(keep, dict) else list(keep or [])
        for col in sorted({c for combo in combos for c in combo}):
            if col not in source.columns:
                report.errors.append(
                    f"groups.keep: {col!r} is not a metadata column; there are {source.columns}"
                )

    # -- the Data steps on a saved dataset ----------------------------------
    if source is not None and cfg.curation is not None and source.curated:
        report.errors.append(
            "curation: the input dataset is already curated; leave curation out, "
            "or start from the raw files (metadata + monitors) to curate differently"
        )
    if source is not None and cfg.split is not None:
        if source.split:
            report.errors.append(
                "split: the input dataset is already split; leave split out, or start "
                "from the raw files to split differently"
            )
        elif not source.has_boundary:
            report.errors.append(
                "split: the metadata has no first_DD_day column, so there is no LD/DD "
                "boundary to split at; leave split out"
            )

    # -- upstream steps (docs rule 5) ----------------------------------------
    an = cfg.analyses
    curated = cfg.curation is not None or (source is not None and source.curated)
    slept = an.sleep is not None or (source is not None and source.slept)
    if an.sleep is not None and not curated:
        report.errors.append(
            "analyses.sleep needs curation, which marks each minute moving or still; "
            "add `curation: {}` to curate with the defaults"
        )
    if an.hmm is not None and not slept:
        report.errors.append(
            "analyses.hmm needs analyses.sleep: the sleep threshold changes the "
            "model's input, so it has to be chosen; add `sleep: {}` under analyses "
            "for the standard 5-minute definition"
        )
    if an.period is not None and not curated:
        report.warnings.append(
            "analyses.period runs on uncurated data: dead flies are analysed until "
            "the end of the recording. Add `curation: {}` to remove them"
        )

    # -- phases ----------------------------------------------------------------
    if source is not None:
        if an.period is not None:
            for key in METHOD_KEYS:
                if getattr(an.period.methods, key) is not None:
                    _check_phase(
                        report, f"analyses.period ({key})", an.period.effective(key)["phase"], source
                    )
        if an.hmm is not None:
            _check_phase(report, "analyses.hmm", an.hmm.phase, source)

    if an.period is not None:
        m = an.period.methods
        cwt = m.cwt
        if cwt is not None and cwt.group_scalograms and cwt.scalogram_flies == "rhythmic":
            ac_called = (m.autocorrelation is not None and m.autocorrelation.classify) or (
                source is not None and source.ac_classified
            )
            if not ac_called:
                report.errors.append(
                    "analyses.period.methods.cwt: scalogram_flies: rhythmic averages the "
                    "flies autocorrelation calls rhythmic, and nothing makes that call; add "
                    "autocorrelation under methods, or set scalogram_flies: all"
                )
        if m.mesa is not None and m.autocorrelation is None:
            report.warnings.append(
                "analyses.period: MESA has no significance test of its own and borrows "
                "autocorrelation's rhythmic call; without autocorrelation its periods "
                "are left unclassified"
            )

    # -- outputs -------------------------------------------------------------------
    ran = {k for k in ("period", "sleep", "hmm") if getattr(an, k) is not None}
    if source is not None:
        ran |= {"sleep"} if source.slept else set()
    for table in cfg.outputs.tables or []:
        if TABLE_NEEDS[table] not in ran:
            report.errors.append(
                f"outputs.tables: {table} needs analyses.{TABLE_NEEDS[table]}, which this file does not run"
            )
    out_dir = loaded.output_dir
    if out_dir.is_dir() and any(out_dir.iterdir()):
        report.warnings.append(f"{out_dir} already has files in it; `clockwork run` will need --force")
    return report


def _check_phase(report: CheckReport, where: str, phase: str, source: _Source) -> None:
    if source.stamped_phase is not None:
        if phase != source.stamped_phase:
            report.errors.append(
                f"{where}: phase {phase}, but the input dataset holds only {source.stamped_phase}"
            )
        return
    if phase in ("LD", "DD") and not source.has_boundary:
        report.errors.append(
            f"{where}: phase {phase} needs an LD/DD boundary, and the metadata has no "
            "first_DD_day column; use phase: full"
        )
    if phase == "full" and source.has_boundary and where.startswith("analyses.period"):
        report.errors.append(
            f"{where}: this recording has an LD/DD boundary; a period estimated across "
            "both describes neither, so choose phase: DD or LD"
        )


def _inspect_source(inputs: InputsConfig, report: CheckReport) -> _Source | None:
    if inputs.dataset is not None:
        path = Path(inputs.dataset)
        if not path.is_file():
            report.errors.append(f"inputs.dataset: no such file: {path}")
            return None
        return _inspect_netcdf(path, report)
    ok = True
    if not Path(inputs.metadata).is_file():
        report.errors.append(f"inputs.metadata: no such file: {inputs.metadata}")
        ok = False
    if not Path(inputs.monitors).is_dir():
        report.errors.append(f"inputs.monitors: no such folder: {inputs.monitors}")
        ok = False
    if not ok:
        return None
    try:
        columns = metadata_columns(inputs.metadata)
    except Exception as e:  # unreadable: the importer would say the same, louder
        report.errors.append(f"inputs.metadata: cannot be read: {e}")
        return None
    from clockwork.core.dam_processor import REQUIRED_METADATA_COLUMNS

    missing = [c for c in REQUIRED_METADATA_COLUMNS if c not in columns]
    if missing:
        report.errors.append(f"inputs.metadata: missing required column(s) {missing}")
    return _Source(columns=columns, has_boundary="first_DD_day" in columns)


def metadata_columns(path) -> list[str]:
    import pandas as pd

    path = Path(path)
    reader = pd.read_csv if path.suffix.lower() == ".csv" else pd.read_excel
    return [str(c) for c in reader(path, nrows=0).columns]


def _inspect_netcdf(path: Path, report: CheckReport) -> _Source | None:
    import xarray as xr

    from clockwork.core.dataset_meta import PHASE_DD, PHASE_LD

    try:
        with xr.open_dataset(path) as ds:
            attrs = dict(ds.attrs)
            per_fly = [str(c) for c in ds.coords if ds[c].dims == ("id",)]
            has_boundary = "first_DD_day" in ds.coords or "split_minute" in ds.coords
            slept = "sleep" in ds.data_vars
            ac_classified = "ac_rhythmic" in ds.coords
    except Exception as e:
        report.errors.append(f"inputs.dataset: cannot be opened: {e}")
        return None
    phase = attrs.get("phase")
    return _Source(
        columns=[_COLUMN_FOR.get(c, c) for c in per_fly],
        has_boundary=has_boundary,
        curated="curation_min_alive_days" in attrs,
        split=bool(int(attrs.get("split_applied", 0) or 0)) or "split_discard_first_dd_day" in attrs,
        slept=slept,
        stamped_phase=phase if phase in (PHASE_LD, PHASE_DD) else None,
        ac_classified=ac_classified,
    )


# ---------------------------------------------------------------------------
# The resolved config
# ---------------------------------------------------------------------------


def resolved_config(config: ExperimentConfig, group_columns=None) -> dict[str, Any]:
    """Every value the run uses, defaults included (docs rule 2).

    Loadable as a config itself, and running it does what ``config`` does.
    ``group_columns`` is what the import actually grouped by, when the file left
    ``groups.by`` to the default.
    """
    out = config.model_dump(mode="json")
    # A step that does not run is LEFT OUT, never written as null: an empty
    # entry means "run with the defaults" when the file is read back.
    for name in ("groups", "curation", "split"):
        if out[name] is None:
            del out[name]
    out["analyses"] = {k: v for k, v in out["analyses"].items() if v is not None}
    if group_columns is not None:
        out["groups"] = {**(out.get("groups") or {"keep": None}), "by": list(group_columns)}
    period = config.analyses.period
    if period is not None:
        methods = {}
        for key in METHOD_KEYS:
            m = getattr(period.methods, key)
            if m is None:
                continue
            d = m.model_dump(mode="json")
            eff = period.effective(key)
            d.update({k: list(v) if isinstance(v, tuple) else v for k, v in eff.items()})
            if not m.BRIDGES_GAPS:
                d.pop("max_bridge_gap_minutes", None)
            d["preprocessing"] = Preprocessing.from_core(m.preprocess()).model_dump(mode="json")
            if isinstance(m, _Classified):
                d["rhythmic_threshold"] = m.threshold()
                d["rhythmic_window_hours"] = list(m.rhythmic_window_hours or eff["period_range_hours"])
            methods[key] = d
        out["analyses"]["period"]["methods"] = methods
    hmm = config.analyses.hmm
    if hmm is not None:
        core = hmm.core()
        out["analyses"]["hmm"].update({k: getattr(core, k) for k in TUNABLE})
    return out


#: Distributions whose version can change a number ClockWork reports.
PROVENANCE_PACKAGES = (
    "clockwork-sci",
    "numpy",
    "scipy",
    "pandas",
    "xarray",
    "astropy",
    "PyWavelets",
    "hmmlearn",
    "statsmodels",
    "netCDF4",
    "torch",
)


def provenance() -> dict[str, Any]:
    """The software a run used."""
    versions = {}
    for dist in PROVENANCE_PACKAGES:
        try:
            versions[dist] = importlib.metadata.version(dist)
        except importlib.metadata.PackageNotFoundError:
            continue
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": versions,
    }


def dump_yaml(data: dict, header: str = "") -> str:
    """YAML in the file's order (not sorted), with an optional comment header."""
    body = yaml.safe_dump(data, sort_keys=False, allow_unicode=True, default_flow_style=None, width=88)
    if header:
        header = "".join(f"# {line}".rstrip() + "\n" for line in header.splitlines())
    return header + body


def json_schema() -> dict[str, Any]:
    """The JSON Schema for a config file, for editor autocompletion and checking."""
    schema = ExperimentConfig.model_json_schema()
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    schema["title"] = "ClockWork config"
    # A file may also extend another, and carry the provenance a run wrote.
    schema.setdefault("properties", {})["extends"] = {
        "type": "string",
        "description": "Another config file whose settings this one starts from; "
        "a path relative to this file.",
    }
    schema["properties"]["provenance"] = {
        "type": "object",
        "description": "Written by clockwork run; ignored when the file is read.",
    }
    schema["required"] = [r for r in schema.get("required", []) if r != "inputs"]
    return schema


__all__ = [
    "CONFIG_VERSION",
    "Analyses",
    "CheckReport",
    "ConfigError",
    "ExperimentConfig",
    "LoadedConfig",
    "OutputsConfig",
    "TABLE_NEEDS",
    "check_config",
    "dump_yaml",
    "format_validation_error",
    "json_schema",
    "load_config",
    "metadata_columns",
    "provenance",
    "resolved_config",
]
