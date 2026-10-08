"""Export settings: the config file for the analyses a dataset records.

Reads the DATASET, not a session (docs: "Where a config file comes from"). Every
step records the config that produced its result, so the file written here
names exactly the analyses that ran, with the settings they ran with — and
works as well on a .nc saved months ago as on the one open in the GUI.
"""

from __future__ import annotations

import datetime as _dt
import os
from pathlib import Path
from typing import Any

import xarray as xr

from clockwork.pipeline.data import CurationConfig, GroupsConfig, InputsConfig, SplitConfig
from clockwork.pipeline.experiment import CONFIG_VERSION, Analyses, ExperimentConfig, dump_yaml
from clockwork.pipeline.hmm import HmmConfig
from clockwork.pipeline.period import PeriodConfig
from clockwork.pipeline.sleep import SleepConfig

#: Written where the dataset does not say where it was imported from.
PLACEHOLDER = "FILL_IN"


def config_from_dataset(ds: xr.Dataset) -> tuple[ExperimentConfig, list[str]]:
    """``(config, notes)``: the config that reproduces ``ds``, and anything the
    person should know before running it."""
    attrs = ds.attrs
    notes = []
    inputs = InputsConfig.from_attrs(attrs)
    if inputs is None:
        inputs = InputsConfig(metadata=Path(PLACEHOLDER), monitors=Path(PLACEHOLDER))
        notes.append(
            "This dataset does not record which files it was imported from (it was "
            f"saved by an older ClockWork). Replace {PLACEHOLDER} under inputs with "
            "the metadata file and the monitor folder."
        )
    period = PeriodConfig.from_attrs(attrs, coords=list(ds.coords))
    hmm = HmmConfig.from_attrs(attrs)
    for name, label in (("phase_shift_", "Phase shift"), ("sd_", "Sleep deprivation")):
        if any(str(k).startswith(name) for k in attrs):
            notes.append(f"{label} was run on this dataset; it is GUI-only, so it is not in this file.")
    config = ExperimentConfig(
        clockwork_config=CONFIG_VERSION,
        experiment=str(attrs["experiment_name"]) if attrs.get("experiment_name") else None,
        inputs=inputs,
        groups=GroupsConfig.from_attrs(attrs),
        curation=CurationConfig.from_attrs(attrs),
        split=SplitConfig.from_attrs(attrs),
        analyses=Analyses(period=period, sleep=SleepConfig.from_attrs(attrs), hmm=hmm),
    )
    return config, notes


def settings_yaml(ds: xr.Dataset, relative_to=None) -> str:
    """The config file for ``ds``, overrides only, with a comment header.

    ``relative_to`` is the folder the file will be saved in: input paths are
    written relative to it (paths in a config are relative to the file), or
    absolute when the file's destination is unknown or on another drive.
    """
    config, notes = config_from_dataset(ds)
    data = config.overrides()
    if relative_to is not None:
        data["inputs"] = {
            k: _relative(v, relative_to) if k in ("metadata", "monitors", "dataset") else v
            for k, v in data["inputs"].items()
        }
    header = [
        f"ClockWork settings, exported {_dt.date.today().isoformat()} from the analyses "
        "this dataset records.",
        "Run with:  clockwork run <this file>     Check with:  clockwork validate <this file>",
        "Only settings that differ from the defaults are listed.",
    ]
    for note in notes:
        header += ["", f"NOTE: {note}"]
    return dump_yaml(data, header="\n".join(header))


def _relative(value: Any, base) -> Any:
    if not isinstance(value, str) or value == PLACEHOLDER:
        return value
    try:
        return Path(os.path.relpath(value, base)).as_posix()
    except ValueError:  # another drive on Windows: no relative path exists
        return Path(value).as_posix()
