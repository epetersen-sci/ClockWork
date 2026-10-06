"""The QC report: one HTML file a person can open after an unattended run to
check it by eye.

For each step that ran it shows what someone at the GUI would have looked at
before accepting the result: who curation removed, how long each fly's epochs
are, the rhythmicity cutoff distributions, the sleep totals, HMM state
occupancy. The figures are the GUI's own (``core.plotting``), so the report and
the pages cannot disagree about what a result looks like.

Self-contained: plotly.js is embedded once, so the file opens offline and can be
mailed or archived on its own.
"""

from __future__ import annotations

import datetime as _dt
import html
import math
import traceback
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
import xarray as xr

from clockwork.core import dam_utilities, plotting
from clockwork.pipeline import tables
from clockwork.pipeline.period import METHOD_KEYS, _Classified

#: The plotting module's short algorithm keys.
_ALGO = {"lomb_scargle": "ls", "autocorrelation": "ac", "cwt": "cwt"}
_METHOD_LABEL = {
    "lomb_scargle": "Lomb-Scargle",
    "autocorrelation": "Autocorrelation",
    "cwt": "CWT",
    "mesa": "MESA",
}


@dataclass
class RunRecord:
    """What the runner learned along the way that the dataset does not hold."""

    experiment: str
    config_yaml: str
    started: _dt.datetime
    finished: _dt.datetime | None = None
    warnings: list[str] = field(default_factory=list)
    #: step name -> seconds
    timings: dict[str, float] = field(default_factory=dict)
    n_imported: int | None = None
    curation: Any = None  # pipeline.CurationResult
    split_report: dict | None = None


def build_report(ds: xr.Dataset, record: RunRecord, config) -> str:
    """The report as an HTML document."""
    from plotly.offline import get_plotlyjs

    body = [_header(ds, record)]
    for title, section in (
        ("Data", lambda: _data_section(ds, record)),
        ("Curation", lambda: _curation_section(record)),
        ("LD/DD split", lambda: _split_section(record)),
        ("Period & rhythmicity", lambda: _period_section(ds, config)),
        ("Sleep", lambda: _sleep_section(ds)),
        ("HMM sleep states", lambda: _hmm_section(ds)),
    ):
        try:
            content = section()
        except Exception:  # a broken figure must not cost the run its report
            content = (
                "<p class='warn'>This section could not be drawn:</p>"
                f"<pre>{html.escape(traceback.format_exc())}</pre>"
            )
        if content:
            body.append(f"<section><h2>{html.escape(title)}</h2>{content}</section>")
    body.append(
        "<section><h2>Config</h2><p>The file as given (overrides only); "
        "<code>config.resolved.yaml</code> beside this report has every value used.</p>"
        f"<pre>{html.escape(record.config_yaml)}</pre></section>"
    )
    return _PAGE.format(
        title=html.escape(f"ClockWork QC — {record.experiment}"),
        plotly=get_plotlyjs(),
        body="\n".join(body),
    )


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------


def _header(ds, record) -> str:
    from clockwork import __version__

    took = ""
    if record.finished is not None:
        took = f" in {(record.finished - record.started).total_seconds():.0f} s"
    rows = [
        ("Experiment", record.experiment),
        ("Run", f"{record.started:%Y-%m-%d %H:%M}{took}, ClockWork {__version__}"),
        ("Flies analysed", ds.sizes.get("id", 0)),
    ]
    if record.timings:
        rows.append(
            ("Steps", ", ".join(f"{k} {v:.0f} s" for k, v in record.timings.items()))
        )
    out = f"<h1>{html.escape(record.experiment)}</h1>" + _kv(rows)
    if record.warnings:
        items = "".join(f"<li>{html.escape(w)}</li>" for w in record.warnings)
        out += f"<div class='warn'><b>Warnings</b><ul>{items}</ul></div>"
    return out


def _data_section(ds, record) -> str:
    rows = []
    if record.n_imported is not None:
        rows.append(("Flies imported", record.n_imported))
    for key, label in (
        ("integrity_n_monitors", "Monitor files read"),
        ("integrity_n_dataloss_slots", "Readings lost"),
        ("integrity_n_status_bad", "Readings with a bad status"),
    ):
        if key in ds.attrs:
            rows.append((label, ds.attrs[key]))
    rows.append(("Grouped by", ", ".join(dam_utilities.get_group_columns(ds)) or "—"))
    if "subset_keep" in ds.attrs:
        rows.append(("Subset kept", ds.attrs["subset_keep"]))
    out = _kv(rows)
    if "group" in ds.coords:
        counts = pd.Series([str(g) for g in ds["group"].values]).value_counts().sort_index()
        out += _table(counts.rename_axis("Group").reset_index(name="Flies"))
    return out


def _curation_section(record) -> str:
    cur = record.curation
    if cur is None:
        return ""
    out = _kv(
        [
            ("Flies before", cur.total_before),
            ("Removed (died too early)", cur.removed),
            ("Trimmed at death", cur.trimmed),
            ("Unchanged", cur.unchanged),
            ("Flies after", cur.total_after),
        ]
    )
    out += "<p>Movement of the flies kept (each row a fly; grey = no reading):</p>"
    out += _figure(plotting.dataset_to_heatmap(cur.live, "moving", "Kept"))
    if cur.dead is not None and cur.dead.sizes.get("id", 0):
        out += "<p>Flies removed:</p>"
        out += _figure(plotting.dataset_to_heatmap(cur.dead, "moving", "Removed"))
    return out


def _split_section(record) -> str:
    if not record.split_report:
        return ""
    df = pd.DataFrame(record.split_report).T.rename_axis("Phase").reset_index()
    return "<p>Each fly keeps its longest continuous block within each epoch.</p>" + _table(df)


def _period_section(ds, config) -> str:
    period = config.analyses.period if config is not None else None
    if period is None:
        return ""
    parts = []
    counts = []
    for key in METHOD_KEYS:
        method = getattr(period.methods, key)
        if method is None:
            continue
        eff = period.effective(key)
        label = _METHOD_LABEL[key]
        period_var = f"{method.PREFIX}_period"
        analysed = int(ds[period_var].notnull().sum()) if period_var in ds else 0
        row = {"Method": label, "Phase": eff["phase"], "Analysed": analysed}
        if isinstance(method, _Classified) and method.classify and key in _ALGO:
            window = method.rhythmic_window_hours or eff["period_range_hours"]
            fig, summary = plotting.threshold_coupled_figure(
                ds, _ALGO[key], method.threshold(), period_window=tuple(window)
            )
            total = summary.get("_total", {})
            row["Rhythmic"] = total.get("n_rhythmic")
            parts.append(
                f"<h3>{label}</h3><p>Cutoff {method.threshold():g} within "
                f"{window[0]:g}–{window[1]:g} h. Left: periods of the flies called "
                "rhythmic. Right: every fly's strength against the cutoff.</p>"
                + _figure(fig)
            )
        counts.append(row)
    return _table(pd.DataFrame(counts)) + "".join(parts)


def _sleep_section(ds) -> str:
    if "sleep" not in ds.data_vars:
        return ""
    out = []
    for label, view in tables.epochs(ds):
        phase_label = None if label == "full" else label
        out.append(f"<h3>{'Whole recording' if label == 'full' else label}</h3>")
        out.append(_figure(plotting.summary_bars(view, "sleep", phase_label=phase_label)))
        if "sleep_long" in view.data_vars:
            fig, _ = plotting.sleep_state_totals_bars(view)
            out.append(_figure(fig))
    return "".join(out)


def _hmm_section(ds) -> str:
    if "hmm_state" not in ds.data_vars:
        return ""
    import plotly.graph_objects as go

    occ = tables.hmm_occupancy(ds)
    if occ.empty:
        return "<p class='warn'>No fly has a decoded state.</p>"
    states = [c for c in occ.columns if c not in ("ID", "Group")]
    grouped = occ.groupby("Group")[states]
    mean, sem = grouped.mean(), grouped.sem()
    fig = go.Figure(
        [
            go.Bar(
                name=s,
                x=list(mean.index),
                y=mean[s],
                error_y={"type": "data", "array": sem[s].fillna(0)},
            )
            for s in states
        ]
    )
    fig.update_layout(
        barmode="group",
        title="Time in each state (mean ± SEM across flies)",
        yaxis_title="% of fitted time",
        template="plotly_white",
    )
    attrs = ds.attrs
    rows = [
        (k, attrs[a])
        for k, a in (
            ("States", "hmm_n_states"),
            ("Emission", "hmm_emission_model"),
            ("Training", "hmm_training_scope"),
            ("Fitted on", "hmm_phase"),
        )
        if a in attrs
    ]
    return _kv(rows) + _figure(fig)


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------


def _figure(fig) -> str:
    return fig.to_html(full_html=False, include_plotlyjs=False, config={"displaylogo": False})


def _kv(rows) -> str:
    cells = "".join(
        f"<tr><th>{html.escape(str(k))}</th><td>{html.escape(str(v))}</td></tr>" for k, v in rows
    )
    return f"<table class='kv'>{cells}</table>"


def _table(df: pd.DataFrame) -> str:
    return df.to_html(index=False, classes="data", border=0, na_rep="—", float_format=_number)


def _number(v: float) -> str:
    # Counts arrive as floats after a transpose; show them as the integers they are.
    if math.isfinite(v) and v == int(v):
        return str(int(v))
    return f"{v:.3g}"


_PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<script>{plotly}</script>
<style>
body {{ font: 15px/1.5 system-ui, sans-serif; color: #1d2228; background: #fff;
       max-width: 1100px; margin: 0 auto; padding: 16px; }}
h1 {{ margin-bottom: .2em; }}
h2 {{ border-bottom: 1px solid #d8dde3; padding-bottom: .2em; margin-top: 2em; }}
table {{ border-collapse: collapse; margin: .6em 0; }}
th, td {{ padding: 3px 10px; text-align: left; border-bottom: 1px solid #eceff2; }}
table.kv th {{ font-weight: 600; color: #4a5560; }}
table.data th {{ background: #f3f5f7; }}
pre {{ background: #f6f8fa; padding: 10px; overflow-x: auto; font-size: 13px; }}
.warn {{ background: #fff6e0; border-left: 4px solid #e0a000; padding: 6px 12px; }}
</style></head>
<body>
{body}
</body></html>
"""
