"""
import_diagnostics.py
=====================
Reasons an import dropped flies, in a form the UI can show the user.

Motivation: a raw DAM import could previously end with "0 channels" and no
visible explanation. Every reason a (Monitor, start_datetime) combo was skipped
was printed to the terminal and then thrown away — the Streamlit user saw a
green success box reporting zero flies. Working out *why* meant opening the DAM
file by hand and comparing timestamps against the metadata.

The real-world case that motivated this: a collaborator truncated a DAM file so
its first reading was 09:01, while the metadata said ``start_datetime`` 09:00.
The requested window was one minute wider than the file, the combo failed the
range check, and the whole monitor vanished silently.

This module owns the vocabulary (``REASON_*``), the generic fix guidance
(``HINTS``), and the rendering (``describe``, ``summary_lines``).
``dam_processor.MetadataProcessor`` records the facts; nothing here inspects
data. Formatting is deliberately separated from detection so the same issue list
can go to the console, the Streamlit page, or a test assertion unchanged.

Not in scope: anticipating or repairing bad input. The goal is only to name the
basic reason so the user can go fix the underlying file.
"""

from dataclasses import dataclass, field

import pandas as pd

# --- Reason codes -----------------------------------------------------------
# Collected per (Monitor, start_datetime) combo during validation.
REASON_FILE_MISSING = "monitor_file_missing"
REASON_FILE_UNREADABLE = "monitor_file_unreadable"
REASON_WINDOW_STARTS_EARLY = "window_starts_before_data"
REASON_WINDOW_ENDS_LATE = "window_ends_after_data"
REASON_WINDOW_OUTSIDE = "window_outside_data"
REASON_NO_ROWS_IN_WINDOW = "no_rows_in_window"
REASON_REGION_NOT_IN_FILE = "region_not_in_file"
REASON_NO_USABLE_DATA = "no_usable_data"

# Raised (not collected) — these abort the load before any combo is reached.
REASON_METADATA_UNREADABLE = "metadata_unreadable"
REASON_METADATA_MISSING_COLUMNS = "metadata_missing_columns"
REASON_METADATA_BAD_DATETIME = "metadata_bad_datetime"
REASON_METADATA_EMPTY = "metadata_empty"
REASON_STOP_BEFORE_START = "stop_before_start"

TITLES = {
    REASON_FILE_MISSING: "monitor file not found",
    REASON_FILE_UNREADABLE: "monitor file could not be read",
    REASON_WINDOW_STARTS_EARLY: "metadata starts before the data in the file",
    REASON_WINDOW_ENDS_LATE: "metadata ends after the data in the file",
    REASON_WINDOW_OUTSIDE: "metadata window falls outside the data in the file",
    REASON_NO_ROWS_IN_WINDOW: "no readings inside the requested window",
    REASON_REGION_NOT_IN_FILE: "requested tubes are not in the file",
    REASON_NO_USABLE_DATA: "readings present but all of them are unusable",
    REASON_METADATA_UNREADABLE: "metadata file could not be read",
    REASON_METADATA_MISSING_COLUMNS: "metadata is missing required columns",
    REASON_METADATA_BAD_DATETIME: "metadata has unparseable date/time values",
    REASON_METADATA_EMPTY: "metadata file has no rows",
    REASON_STOP_BEFORE_START: "stop_datetime is not after start_datetime",
}

# Generic guidance per reason. An issue may carry its own `hint` when the facts
# allow something more specific (e.g. the exact timestamp to move a bound to).
HINTS = {
    REASON_FILE_MISSING: (
        "Check the Monitor number in the metadata against the filenames in the "
        "data directory. The loader looks for exactly 'Monitor<N>.txt'."
    ),
    REASON_FILE_UNREADABLE: (
        "The file must be the raw tab-separated Trikinetics DAM output with the "
        "10 standard leading columns and a 'DD Mon YY' date. An Excel re-save, a "
        "header row, or a partially written file will all fail here."
    ),
    REASON_WINDOW_STARTS_EARLY: (
        "Either re-export the DAM file so it covers the requested start, or move "
        "start_datetime later. start_datetime must stay at ZT0 (lights-on), so "
        "the usual correct fix is to move it forward to the NEXT ZT0 that the "
        "file does cover, not to the file's first reading."
    ),
    REASON_WINDOW_ENDS_LATE: (
        "Shorten stop_datetime to the file's last reading (or earlier), or "
        "re-export the DAM file so it runs to the requested stop."
    ),
    REASON_WINDOW_OUTSIDE: (
        "The metadata window and the file barely overlap or do not overlap at "
        "all. Check that this Monitor number points at the intended file and "
        "that the dates are the right year/month."
    ),
    REASON_NO_ROWS_IN_WINDOW: (
        "The file covers the window but contains no rows in it. Check for a "
        "recording gap over this period."
    ),
    REASON_REGION_NOT_IN_FILE: (
        "region_id must be a channel that exists in the file. A standard DAM2 "
        "monitor has 32 channels; a 96-well recorder has more. Blank region_id "
        "means 'the whole monitor'."
    ),
    REASON_NO_USABLE_DATA: (
        "Every reading in the window failed (monitor_status != 1 means no data "
        "was received). Check that the monitor was powered and connected for "
        "this period."
    ),
    REASON_METADATA_UNREADABLE: "Save the metadata as .csv or .xlsx and try again.",
    REASON_METADATA_MISSING_COLUMNS: (
        "See 'Metadata file format' on the import page, or start from "
        "metadata_template.csv in the repository root."
    ),
    REASON_METADATA_BAD_DATETIME: (
        "Use a format pandas can parse, e.g. '2024-01-15 09:00:00'. A blank cell "
        "in a required datetime column also lands here."
    ),
    REASON_METADATA_EMPTY: "The metadata file has a header but no data rows.",
    REASON_STOP_BEFORE_START: (
        "Each row's stop_datetime must be later than its start_datetime. Swapped "
        "columns and a mis-typed year are the usual causes."
    ),
}


class MetadataError(ValueError):
    """A metadata problem that aborts the load before any monitor is read.

    Carries the machine-readable ``reason`` alongside the rendered message so a
    caller can branch on the cause without parsing text. Subclasses ``ValueError``
    so existing ``except ValueError`` / ``except Exception`` callers are unaffected.
    """

    def __init__(self, reason, detail, hint=None):
        self.reason = reason
        self.detail = detail
        self.hint = hint or HINTS.get(reason, "")
        message = f"{TITLES.get(reason, reason)}: {detail}"
        if self.hint:
            message = f"{message}\nFix: {self.hint}"
        super().__init__(message)


@dataclass
class ImportIssue:
    """One reason some flies did not make it into the dataset.

    Parameters
    ----------
    reason : str
        One of the ``REASON_*`` codes.
    detail : str
        The specific facts — actual ranges, deltas, names. This is the part that
        tells the user *which* file and *by how much*.
    monitor, start_datetime : optional
        The combo this issue belongs to; ``None`` for whole-file issues.
    n_flies : int
        How many metadata rows (flies) this issue cost.
    excluded : bool
        True when the issue removed flies from the import (an error the user must
        act on); False when the import continued and this is a caveat.
    hint : str, optional
        Overrides the generic ``HINTS`` entry when the facts allow something more
        specific.
    """

    reason: str
    detail: str
    monitor: object = None
    start_datetime: object = None
    n_flies: int = 0
    excluded: bool = True
    hint: str = None
    ids: list = field(default_factory=list)

    @property
    def title(self):
        return TITLES.get(self.reason, self.reason)

    @property
    def guidance(self):
        return self.hint or HINTS.get(self.reason, "")

    @property
    def combo_label(self):
        if self.monitor is None:
            return ""
        if self.start_datetime is None:
            return f"Monitor {self.monitor}"
        return f"Monitor {self.monitor} / {self.start_datetime}"


def format_timedelta(delta):
    """Render a ``Timedelta`` the way a person would say it out loud.

    Deliberately coarse: the point is 'one minute' vs 'three days', because that
    difference is what tells a truncated file apart from a wrong year.
    """
    delta = pd.Timedelta(delta)
    if delta < pd.Timedelta(0):
        return "-" + format_timedelta(-delta)
    total_minutes = delta.total_seconds() / 60.0
    if total_minutes < 1:
        return f"{delta.total_seconds():.0f} s"
    if total_minutes < 60:
        n = round(total_minutes)
        return f"{n} minute{'s' if n != 1 else ''}"
    hours = total_minutes / 60.0
    if hours < 48:
        return f"{hours:.1f} h"
    return f"{hours / 24.0:.1f} days"


def describe(issue, indent="  "):
    """Render one issue as console/UI text: headline, facts, then the fix."""
    head = issue.title
    if issue.combo_label:
        head = f"{issue.combo_label}: {head}"
    if issue.n_flies:
        verb = "dropped" if issue.excluded else "affected"
        head = f"{head} ({issue.n_flies} fl{'y' if issue.n_flies == 1 else 'ies'} {verb})"
    lines = [head, f"{indent}{issue.detail}"]
    if issue.guidance:
        lines.append(f"{indent}Fix: {issue.guidance}")
    return "\n".join(lines)


def summary_lines(issues, n_requested=None, n_imported=None):
    """Build the import report as ``(severity, text)`` tuples.

    Mirrors ``MetadataProcessor.integrity_summary_lines`` so both reports can be
    rendered by the same few lines of Streamlit (and printed the same way on the
    console). severity is ``"error"`` (flies were dropped), ``"warning"`` (the
    import continued but something is off) or ``"info"``.

    The headline comes first and states the count plainly, because "0 of 128
    flies imported" is the fact the user is actually looking at when they come
    to read this.
    """
    lines = []
    excluded = [i for i in issues if i.excluded]
    caveats = [i for i in issues if not i.excluded]

    if n_requested is not None and n_imported is not None:
        if n_imported == 0:
            lines.append(
                (
                    "error",
                    f"No flies were imported ({n_requested} requested by the metadata). "
                    f"{'Reasons below.' if issues else 'No reason was recorded — see the console log.'}",
                )
            )
        elif n_imported < n_requested:
            lines.append(
                (
                    "warning",
                    f"Imported {n_imported} of {n_requested} flies requested by the "
                    f"metadata — {n_requested - n_imported} were dropped.",
                )
            )
        else:
            # Everything asked for arrived. The headline is emitted even when
            # there are caveats below it, so callers can always treat line 0 as
            # the headline and the remainder as one entry per reason.
            lines.append(("info", f"Imported all {n_imported} flies requested by the metadata."))

    for issue in excluded:
        lines.append(("error", describe(issue)))
    for issue in caveats:
        lines.append(("warning", describe(issue)))
    return lines


def format_report(issues, n_requested=None, n_imported=None):
    """The same report as one console string; empty when there is nothing to say."""
    lines = summary_lines(issues, n_requested=n_requested, n_imported=n_imported)
    if not lines:
        return ""
    out = ["--- Import report ---"]
    for severity, text in lines:
        prefix = "  [!] " if severity in ("warning", "error") else "  "
        first, *rest = text.split("\n")
        out.append(f"{prefix}{first}")
        out.extend(f"      {r.strip()}" for r in rest)
    return "\n".join(out)
