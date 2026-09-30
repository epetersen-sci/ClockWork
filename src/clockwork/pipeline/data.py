"""Building the dataset: import, groups & subsets, curation, and the LD/DD split.

The first four steps of every run, in the order the GUI's Data section takes
them. Each is a plain function of a dataset and a config, records that config on
its result, and knows nothing about Streamlit — the Data pages and the CLI are
both callers.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
from pydantic import Field, model_validator

from clockwork.core import dam_processor, dam_utilities
from clockwork.core.dataset_meta import (
    PHASE_DD,
    PHASE_FULL,
    PHASE_LD,
    VALID_PHASES,
    has_split_evidence,
    is_split_applied,
    stamp_phase,
)
from clockwork.core.load_and_save_datasets import load_dataset_from_netcdf
from clockwork.pipeline._config import StepConfig

# Two metadata columns are stored under different coord names. The config speaks
# in COLUMN names — the ones in the user's metadata file — and translates here.
_COORD_FOR = dict(dam_utilities.METADATA_COORD_RENAMES)
_COLUMN_FOR = {coord: column for column, coord in _COORD_FOR.items()}

Scalar = str | int | float | bool | None


# ---------------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------------


class InputsConfig(StepConfig):
    """Where the data comes from: raw monitor files plus metadata, or a saved .nc."""

    metadata: Path | None = None
    monitors: Path | None = None
    dataset: Path | None = None
    #: A run of missing reads longer than this is a gap rather than a dropped read.
    gap_threshold_hours: float = Field(1.0, gt=0)

    ATTRS = {
        "metadata": "import_metadata_file",
        "monitors": "import_monitor_dir",
        "gap_threshold_hours": "import_gap_threshold_hours",
    }

    @model_validator(mode="after")
    def _one_source(self):
        raw = self.metadata is not None or self.monitors is not None
        if raw and self.dataset is not None:
            raise ValueError("give either metadata + monitors, or dataset, not both")
        if not raw and self.dataset is None:
            raise ValueError("give metadata + monitors (raw files) or dataset (a saved .nc)")
        if raw and (self.metadata is None or self.monitors is None):
            raise ValueError("raw import needs both metadata and monitors")
        return self


class GroupsConfig(StepConfig):
    """Which metadata columns define a group, and which flies to keep.

    ``keep`` selects by COLUMN VALUE, never by the joined group label (which
    changes whenever the grouping does). Two forms:

    - ``{"genotype": ["w1118", "per0"]}`` — every fly whose value is listed, for
      each column named (all columns must match);
    - ``[{"genotype": "w1118", "temperature": "25C"}, ...]`` — exact
      combinations, any of which may match. The GUI records this form, because a
      hand-picked set of groups is not always a product of column values.
    """

    by: list[str] | None = None
    keep: dict[str, list[Scalar]] | list[dict[str, Scalar]] | None = None

    ATTRS = {"by": "group_columns", "keep": "subset_keep"}
    JSON_FIELDS = frozenset({"keep"})
    #: group_columns is written by the core (create_xarray_dataset, regroup_dataset)
    #: together with group_coord_names; writing it again here could only disagree.
    RECORDED_BY_CORE = frozenset({"by"})

    @model_validator(mode="after")
    def _not_empty(self):
        if self.by is not None and not self.by:
            raise ValueError("groups.by needs at least one column")
        if self.keep is not None and not self.keep:
            raise ValueError("groups.keep keeps nothing; leave it out to keep every fly")
        return self

    @classmethod
    def _normalise_from_attrs(cls, values):
        # regroup_dataset records coord names where import records column names
        # (BACKLOG); read either, report column names.
        if values.get("by") is not None:
            values["by"] = [_COLUMN_FOR.get(c, c) for c in values["by"]]
        return values


class CurationConfig(StepConfig):
    """Dead-fly detection: a fly is dead from the point its activity stops."""

    min_alive_days: float = Field(2.0, gt=0)
    rolling_window_hours: int = Field(24, ge=1)
    immobility_proportion: float = Field(0.01, gt=0, lt=1)

    ATTRS = {
        "min_alive_days": "curation_min_alive_days",
        "rolling_window_hours": "curation_time_window_hours",
        "immobility_proportion": "curation_prop_immobile_threshold",
    }
    RECORDED_BY_CORE = frozenset(ATTRS)


class SplitConfig(StepConfig):
    """The LD/DD split, which marks each fly's phase boundary on the master."""

    discard_first_dd_day: bool = False
    #: A NaN run longer than this is a real gap; each fly keeps its longest
    #: continuous segment within a phase. 0 disables.
    gap_threshold_minutes: int = Field(60, ge=0)

    ATTRS = {
        "discard_first_dd_day": "split_discard_first_dd_day",
        "gap_threshold_minutes": "gap_threshold_minutes",
    }


# ---------------------------------------------------------------------------
# Import
# ---------------------------------------------------------------------------


class ImportFailed(RuntimeError):
    """Nothing usable came in. ``processor`` holds the import report, if there is one."""

    def __init__(self, message, processor=None):
        super().__init__(message)
        self.processor = processor


class AmbiguousPhase(RuntimeError):
    """A saved .nc looks LD/DD-partitioned but does not say which phase it holds.

    The GUI asks; the CLI cannot, so it is an error there. ``dataset`` is the
    loaded file, so a caller that can ask does not have to read it twice.
    """

    def __init__(self, dataset):
        super().__init__(
            "this .nc looks like an LD- or DD-only partition but has no 'phase' attr; "
            "re-save it from the GUI, which asks which phase it holds"
        )
        self.dataset = dataset


@dataclass
class RawImport:
    """What the monitor files and the metadata gave, before the dataset exists.

    A separate stage because the GUI shows the import report and lets the user
    pick the grouping columns in between; the CLI goes straight through.

    ``integrity`` is the processor's handful of counters, which is all that
    build_dataset needs. The processor itself (its reports) is optional, because
    it holds the whole raw scan and the GUI drops it rather than keep that in the
    session between the two buttons.
    """

    metadata: pd.DataFrame
    data: pd.DataFrame
    integrity: dict[str, Any]
    processor: dam_processor.MetadataProcessor | None = None

    @property
    def n_flies(self) -> int:
        return len(self.data.columns)


def read_monitors(inputs: InputsConfig, progress=None) -> RawImport:
    """Read and validate the metadata and monitor files.

    Raises :class:`dam_processor.MetadataError` when the metadata file itself is
    unusable. An import that yields no flies is returned, not raised, so the
    caller can show the processor's report of why.
    """
    if inputs.dataset is not None:
        raise ValueError("read_monitors is for raw imports; use load_netcdf for a .nc")
    processor = dam_processor.MetadataProcessor(
        str(inputs.metadata), str(inputs.monitors), gap_threshold_hours=inputs.gap_threshold_hours
    )
    metadata, data = processor.run(progress_callback=progress)
    return RawImport(
        metadata=metadata, data=data, integrity=processor.integrity_scalars(), processor=processor
    )


def build_dataset(
    raw: RawImport,
    inputs: InputsConfig,
    groups: GroupsConfig | None = None,
    experiment_name: str | None = None,
) -> xr.Dataset:
    """The core dataset, from a raw import.

    ``experiment_name`` None means "guess it from the metadata file's name", as
    the Import page does; "" means "no name".
    """
    if raw.n_flies == 0:
        raise ImportFailed("no flies were imported", raw.processor)
    groups = groups or GroupsConfig()

    full = dam_utilities.convert_to_relative_time(raw.data, raw.metadata)
    ds = dam_utilities.create_xarray_dataset(full, raw.metadata, group_columns=groups.by)
    if ds is None:
        raise ImportFailed("the dataset has an empty time dimension", raw.processor)

    # Raw loads have no partitioning: this is the full recording.
    stamp_phase(ds, PHASE_FULL, split_applied=False)
    # The working folder — where saves and exports default — is the folder that
    # holds the metadata file, absolute so it cannot later resolve against
    # wherever the app or the CLI happened to be started.
    ds.attrs["source_data_dir"] = os.path.abspath(os.path.dirname(str(inputs.metadata)))
    if experiment_name is None:
        experiment_name = dam_utilities.experiment_name_from_path(str(inputs.metadata))
    ds.attrs["experiment_name"] = dam_utilities.sanitize_experiment_name(experiment_name)
    # Import-time integrity counters, so a reloaded .nc can report its own quality.
    ds.attrs.update(raw.integrity)
    ds.attrs.update(
        inputs.model_copy(
            update={
                "metadata": Path(os.path.abspath(inputs.metadata)),
                "monitors": Path(os.path.abspath(inputs.monitors)),
            }
        ).to_attrs()
    )
    if groups.keep is not None:
        ds = subset(ds, groups.keep)
    return ds


def load_netcdf(path) -> xr.Dataset:
    """A saved dataset, with its canonical phase metadata resolved.

    Raises :class:`AmbiguousPhase` for a partitioned file that does not say which
    phase it holds.
    """
    ds = load_dataset_from_netcdf(str(path))
    existing = ds.attrs.get("phase")
    if isinstance(existing, str) and existing in VALID_PHASES:
        stamp_phase(
            ds, existing, split_applied=ds.attrs.get("split_applied", existing != PHASE_FULL)
        )
    elif isinstance(ds.attrs.get("split_phase"), str):
        # Read-side migration for files saved before `phase` existed. Nothing
        # writes `split_phase` any more, but every older .nc on disk still carries
        # it, and this is the only thing that reads those files correctly.
        legacy = ds.attrs["split_phase"]
        resolved = PHASE_LD if legacy == PHASE_LD else PHASE_DD if legacy == PHASE_DD else PHASE_FULL
        stamp_phase(ds, resolved, split_applied=(resolved != PHASE_FULL or legacy == "both"))
    elif has_split_evidence(ds):
        raise AmbiguousPhase(ds)
    else:
        stamp_phase(ds, PHASE_FULL, split_applied=False)
    return ds


# ---------------------------------------------------------------------------
# Groups & subsets
# ---------------------------------------------------------------------------


def apply_groups(ds: xr.Dataset, groups: GroupsConfig) -> xr.Dataset:
    """Regroup (when ``by`` is given) and then subset (when ``keep`` is).

    Regrouping first is safe because a subset is by column value: it keeps the
    same flies whatever the grouping is.
    """
    out = ds
    if groups.by is not None:
        coords = [_COORD_FOR.get(c, c) for c in groups.by]
        missing = [col for col, coord in zip(groups.by, coords) if coord not in ds.coords]
        if missing:
            raise ValueError(
                f"groups.by names {missing}, which this dataset does not carry per fly; "
                f"it has {sorted(_COLUMN_FOR.get(c, c) for c in dam_utilities.group_defining_coords(ds))}"
            )
        out = dam_utilities.regroup_dataset(out, coords)
    if groups.keep is not None:
        out = subset(out, groups.keep)
    return out


def subset(ds: xr.Dataset, keep) -> xr.Dataset:
    """The flies whose metadata values match ``keep`` (see :class:`GroupsConfig`).

    Records ``keep`` on the result, so the subset is on the dataset rather than
    only in the session that made it (ARCHITECTURE rule 5).
    """
    config = GroupsConfig(keep=keep)
    combos = (
        [config.keep] if isinstance(config.keep, dict) else list(config.keep)
    )
    n = ds.sizes["id"]
    mask = np.zeros(n, dtype=bool)
    for combo in combos:
        match = np.ones(n, dtype=bool)
        for column, wanted in combo.items():
            values = _column_values(ds, column)
            wanted = wanted if isinstance(config.keep, dict) else [wanted]
            match &= np.array([any(_same(v, w) for w in wanted) for v in values])
        mask |= match
    if not mask.any():
        raise ValueError(f"groups.keep {config.keep!r} matches no flies")
    out = ds.isel(id=np.flatnonzero(mask))
    out.attrs = dict(ds.attrs)
    out.attrs.update(config.to_attrs())
    return out


def keep_for_groups(ds: xr.Dataset, labels) -> list[dict[str, Any]]:
    """The exact column-value combinations behind the group ``labels`` in ``ds``.

    How the Groups page turns "these groups, ticked by name" into a ``keep`` that
    still means the same flies after a regroup, a reload, or in a config file.
    """
    wanted = {str(label) for label in labels}
    coords = dam_utilities.get_group_coord_names(ds)
    rows = set()
    for i, label in enumerate(ds["group"].values):
        if str(label) in wanted:
            rows.add(tuple(_scalar(ds[c].values[i]) for c in coords))
    return [
        {_COLUMN_FOR.get(c, c): v for c, v in zip(coords, row)}
        for row in sorted(rows, key=lambda r: tuple(str(v) for v in r))
    ]


def _column_values(ds, column):
    coord = _COORD_FOR.get(column, column)
    if coord not in ds.coords or ds[coord].dims != ("id",):
        raise ValueError(f"groups.keep names {column!r}, which is not a per-fly metadata column")
    return [_scalar(v) for v in ds[coord].values]


def _scalar(value):
    value = value.item() if hasattr(value, "item") else value
    if isinstance(value, bytes):
        value = value.decode()
    if isinstance(value, float) and np.isnan(value):
        return None
    return value


def _same(value, wanted) -> bool:
    """Metadata values compare by meaning, not by type.

    A metadata cell read as 15 may be stored as 15.0, and a column of "25"s may
    be text; a config written by hand cannot be expected to know which. Missing
    (NaN, "nan", blank) matches None.
    """
    missing = (None, "", "nan", "NaN")
    if value in missing or wanted in missing:
        return value in missing and wanted in missing
    try:
        return float(value) == float(wanted)
    except (TypeError, ValueError):
        return str(value) == str(wanted)


# ---------------------------------------------------------------------------
# Curation and the split
# ---------------------------------------------------------------------------


@dataclass
class CurationResult:
    live: xr.Dataset
    dead: xr.Dataset
    total_before: int
    total_after: int
    removed: int
    unchanged: int
    trimmed: int


def curate(ds: xr.Dataset, config: CurationConfig, progress=None) -> CurationResult:
    """Separate dead flies from live ones, and trim each live one at its death."""
    (
        live,
        dead,
        _error_ids,
        _success_ids,
        total_before,
        total_after,
        removed,
        unchanged,
        trimmed,
    ) = dam_utilities.curate_dead_animals(
        ds,
        time_window=config.rolling_window_hours,
        prop_immobile=config.immobility_proportion,
        min_alive_days=config.min_alive_days,
        progress_callback=progress,
    )
    return CurationResult(live, dead, total_before, total_after, removed, unchanged, trimmed)


def split(ds: xr.Dataset, config: SplitConfig) -> xr.Dataset:
    """Mark the master dataset as LD/DD-split.

    The master keeps every timepoint: phase views are derived on demand
    (``select_phase``) from the parameters recorded here, never stored.
    """
    if "first_DD_day" not in ds.coords:
        raise ValueError(
            "the LD/DD split needs a first_DD_day metadata column; this dataset has "
            "only one lighting phase, so leave the split out"
        )
    if is_split_applied(ds):
        raise ValueError(
            "this dataset has already been split; re-splitting needs the unsplit "
            "dataset, so re-import it"
        )
    out = ds.copy(deep=False)
    out.attrs = dict(ds.attrs)
    stamp_phase(out, PHASE_FULL, split_applied=True)
    out.attrs.update(config.to_attrs())
    return out


def split_report(ds: xr.Dataset, config: SplitConfig) -> dict[str, dict[str, Any]]:
    """Per phase: flies, record lengths in days, and how many were trimmed at a gap."""
    import ast

    report = {}
    for phase in (PHASE_DD, PHASE_LD):
        view = dam_utilities.split_xarray_dataset(
            ds,
            phase=phase,
            discard_first_dd_day=config.discard_first_dd_day if phase == PHASE_DD else False,
            gap_threshold_minutes=config.gap_threshold_minutes,
        )
        entry = {"n_flies": len(view["id"]), "n_timepoints": len(view["time"])}
        segments = view.attrs.get("segment_info")
        if segments:
            segments = ast.literal_eval(segments)
            days = [s["segment_duration_days"] for s in segments]
            original = [s["original_duration_days"] for s in segments]
            entry.update(
                days_min=min(days),
                days_mean=float(np.mean(days)),
                days_max=max(days),
                n_trimmed=sum(1 for d, o in zip(days, original) if d < o),
            )
        report[phase] = entry
    return report
