"""``clockwork init``: a commented config to start from.

Every setting a person is likely to change is in the file, commented out at its
default, so the file documents itself and an uncommented line is a deliberate
change. Given a metadata file, the groups and the LD/DD choices are filled in
from its columns.
"""

from __future__ import annotations

import os
from pathlib import Path

from clockwork.pipeline.data import CurationConfig, InputsConfig, SplitConfig
from clockwork.pipeline.experiment import CONFIG_VERSION, OutputsConfig, metadata_columns
from clockwork.pipeline.hmm import PRESETS
from clockwork.pipeline.period import PeriodConfig
from clockwork.pipeline.sleep import SleepConfig


def init_text(out: Path, metadata=None, monitors=None) -> str:
    out = Path(out)
    base = out.resolve().parent
    columns = None
    if metadata is not None:
        metadata = Path(metadata).resolve()
        if not metadata.is_file():
            raise ValueError(f"no such metadata file: {metadata}")
        columns = metadata_columns(metadata)
        monitors = Path(monitors).resolve() if monitors else metadata.parent
    from clockwork.core.dam_utilities import (
        GROUP_EXCLUDE_COLUMNS,
        experiment_name_from_path,
        sanitize_experiment_name,
    )

    if columns is not None:
        groupable = [c for c in columns if c not in GROUP_EXCLUDE_COLUMNS]
        by = [c for c in ("genotype", "temperature") if c in columns] or groupable[:1]
        boundary = "first_DD_day" in columns
    else:
        groupable, by, boundary = None, ["genotype", "temperature"], True

    def rel(p, fallback):
        if p is None:
            return fallback
        # Relative when the data sits beside the config (portable with it);
        # absolute when it is somewhere else entirely, which a row of ../ hides.
        try:
            r = Path(os.path.relpath(p, base)).as_posix()
        except ValueError:  # another drive
            return Path(p).as_posix()
        return Path(p).as_posix() if r.startswith("../..") else r

    inputs, cur, spl = InputsConfig.model_fields, CurationConfig.model_fields, SplitConfig.model_fields
    period, sleep = PeriodConfig.model_fields, SleepConfig.model_fields
    lo, hi = period["period_range_hours"].default
    name = (
        experiment_name_from_path(str(metadata)) or sanitize_experiment_name(metadata.parent.name)
        if metadata
        else sanitize_experiment_name(out.stem)
    ) or "my_experiment"
    cols_note = f"   # columns: {', '.join(groupable)}" if groupable else ""

    lines = [
        "# ClockWork config.",
        f"#   check it:  clockwork validate {out.name}",
        f"#   run it:    clockwork run {out.name}",
        "# Paths are relative to this file. A commented line shows a setting at its",
        "# default; uncomment it to change it. `clockwork schema` lists every setting.",
        "",
        f"clockwork_config: {CONFIG_VERSION}",
        f"experiment: {name}            # names the output folder",
        "",
        "inputs:",
        f"  metadata: {rel(metadata, 'metadata.xlsx')}",
        f"  monitors: {rel(monitors, 'monitors/')}      # the folder holding Monitor<N>.txt",
        f"  # gap_threshold_hours: {inputs['gap_threshold_hours'].default:g}   # missing reads longer than this are a gap",
        "",
        "groups:",
        f"  by: [{', '.join(by)}]{cols_note}",
        "  # keep: {genotype: [w1118, per0]}   # analyse only some flies, by column value",
        "",
        "curation:                     # remove dead flies",
        f"  # min_alive_days: {cur['min_alive_days'].default:g}",
        f"  # rolling_window_hours: {cur['rolling_window_hours'].default}",
        f"  # immobility_proportion: {cur['immobility_proportion'].default:g}",
        "",
    ]
    if boundary:
        lines += [
            "split:                        # mark each fly's LD/DD boundary (first_DD_day)",
            f"  # discard_first_dd_day: {str(spl['discard_first_dd_day'].default).lower()}",
            f"  # gap_threshold_minutes: {spl['gap_threshold_minutes'].default}",
            "",
        ]
    else:
        lines += [
            "# No first_DD_day column, so no LD/DD split: analyses use the whole recording.",
            "",
        ]
    phase = "DD" if boundary else "full"
    hmm_phase = "LD" if boundary else "full"
    lines += [
        "analyses:                     # an analysis that is absent does not run",
        "  period:",
        f"    phase: {phase}",
        f"    # period_range_hours: [{lo:g}, {hi:g}]",
        f"    # min_dd_days: {period['min_dd_days'].default:g}",
        "    methods:                   # each one listed runs",
        "      lomb_scargle:",
        "      autocorrelation:",
        "      cwt:",
        "      # mesa:",
        "  sleep:",
        f"    # threshold_seconds: {sleep['threshold_seconds'].default}",
        f"    # short_max_minutes: {sleep['short_max_minutes'].default:g}",
        f"    # intermediate_max_minutes: {sleep['intermediate_max_minutes'].default:g}",
        "  # hmm:                       # slow: minutes to hours, faster with a GPU",
        f"  #   preset: improved        # {' | '.join(PRESETS)}",
        f"  #   phase: {hmm_phase}",
        "",
        "outputs:",
        f"  dir: {OutputsConfig.model_fields['dir'].default}",
        "  # tables: [period_summary, sleep_summary]   # default: every table there is",
        "  # dataset: true",
        "  # qc_report: true",
        "",
    ]
    return "\n".join(lines)
