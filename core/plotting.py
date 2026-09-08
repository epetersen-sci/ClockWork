"""
plotting.py
===========
Plotly-based visualization functions for the Pythomics Streamlit app.

All functions return plotly.graph_objects.Figure objects that are rendered
via st.plotly_chart() in the Streamlit UI.

Functions
---------
dataset_to_heatmap()              — Activity/sleep heatmap (all flies × time)
daily_pattern_line()              — ZT-binned mean ± SEM line plot per group
sleep_bout_duration_lines()       — Per-fly sleep bout duration curves (KDE/survival) by group
group_spectrum_plot()             — Generic mean±SEM curve overlay per group on a shared x-axis
summary_bars()                    — Grouped bar chart: total activity/sleep by day/night/all-day
single_fly_scalogram_plotly()        — Interactive Plotly scalogram for one fly
group_ridge_density_plotly()         — Plotly ridge density of CWT periods per group
phase_shift_actogram()            — Double-plotted actogram for one fly with the
                                    light pulse, daily phase markers and the
                                    pre/post-pulse regression fits overlaid
save_group_average_scalogram_png()   — Disk-saved per-group 2D averaged CWT
                                       periodogram (matplotlib PNG; rendered via
                                       st.image to bypass Streamlit's plotly
                                       JSON message-size ceiling)
"""

# dam_utilities is co-located in the core/ directory; the app pages add core/ to sys.path
import numpy as np
import pandas as pd
import plotly.colors as pc
import plotly.graph_objects as go
import xarray as xr

import dam_utilities


def _circadian_period_ticks(min_period: float, max_period: float) -> list:
    """Return integer-hour tick positions for a circadian scalogram y-axis.

    Conventions used by both the matplotlib PNG saver and the plotly
    single-fly scalogram so the two views stay visually consistent:

    - For "wide" windows (>= 8h, e.g. the typical 16–32h circadian
      band): every 2 hours.
    - For "narrow" windows (< 8h, e.g. ultradian sub-band views):
      every 1 hour.

    Always anchored to integer hours and clipped to [min_period,
    max_period] so the ticks don't fall outside the plotted range.
    """
    lo = float(min_period)
    hi = float(max_period)
    if hi <= lo:
        return [int(round(lo))]
    width = hi - lo
    step = 1 if width < 8.0 else 2
    start = int(np.ceil(lo))
    end = int(np.floor(hi))
    return list(range(start, end + 1, step))


def dataset_to_heatmap(ds: xr.Dataset, var: str, title: str) -> go.Figure:
    """
    Create a Plotly heatmap of a dataset variable (flies × time).

    Parameters
    ----------
    ds : xr.Dataset
    var : str
        Variable to plot (e.g., ``'moving'``, ``'activity'``, ``'sleep'``).
    title : str
        Plot title.

    Returns
    -------
    go.Figure  (empty figure if data or variable is missing)

    Notes (C1, 2026-07-01)
    ----------------------
    * ``moving`` is rendered with a **discrete 3-band legend** (−1 missing /
      0 still / 1 moving), NOT a continuous scale. A continuous scale makes the
      −1 "missing" cells read as "low movement", burying the §2a distinction
      between *no measurement* (−1) and *measured-but-still* (0). ``activity``
      (and any other var) keeps the continuous Viridis scale.
    * The **x-axis is truncated to the finite-data extent** (first→last day that
      any fly has in-phase data), not the full time span. On an LD view this
      shows only the LD window; on a DD view it starts at the earliest DD run
      rather than day 0 over the empty LD gap.
    * Flies are ordered so each **group** is a contiguous block, and the y-axis
      shows the **group name centred on its block** instead of unreadable
      per-fly labels; the exact fly id stays on **hover**.
    """
    if ds is None or var not in ds.data_vars:
        return go.Figure()
    da = ds[var]
    if "time" not in da.dims or "id" not in da.dims:
        return go.Figure()

    # --- group each fly (contiguous blocks) for the y-axis labels ---------- #
    ids = [str(i) for i in da["id"].values]
    grp_of = None
    if "group" in ds.coords:
        grp_of = {str(i): str(g) for i, g in zip(da["id"].values, ds["group"].values)}
    elif "genotype" in ds.coords and "temperature" in ds.coords:
        grp_of = {
            str(i): f"{gt}-{tp}"
            for i, gt, tp in zip(da["id"].values, ds["genotype"].values, ds["temperature"].values)
        }
    if grp_of is not None:
        ids_ord = sorted(ids, key=lambda f: (grp_of.get(f, ""), f))
    else:
        ids_ord = ids

    df = da.to_dataframe().reset_index()
    df["id"] = df["id"].astype(str)

    # Convert time to days elapsed from the first time point
    start_time = df["time"].min()
    if np.issubdtype(df["time"].dtype, np.integer):
        df["days_elapsed"] = (df["time"] - start_time) / 1440.0
    else:
        df["days_elapsed"] = (df["time"] - start_time).dt.total_seconds() / 86400.0

    pivot_df = df.pivot(index="id", columns="days_elapsed", values=var).reindex(ids_ord)
    z = pivot_df.values
    x = pivot_df.columns.values.astype(float)
    y = np.array(ids_ord, dtype=object)

    if var == "moving":
        # Discrete 3-band scale on [-1, 1]: -1 missing, 0 still, 1 moving.
        # Missing (gray) is visually distinct from still (light) so it never
        # reads as "low movement" (§2a). Out-of-phase cells are NaN -> blank.
        colorscale = [
            [0.0, "#cccccc"],
            [1 / 3, "#cccccc"],
            [1 / 3, "#9ecae1"],
            [2 / 3, "#9ecae1"],
            [2 / 3, "#08519c"],
            [1.0, "#08519c"],
        ]
        heat = go.Heatmap(
            z=z,
            x=x,
            y=y,
            colorscale=colorscale,
            zmin=-1,
            zmax=1,
            colorbar=dict(
                title="state", tickvals=[-1, 0, 1], ticktext=["missing", "still", "moving"]
            ),
            hovertemplate="Fly: %{y}<br>Day: %{x:.2f}<br>state: %{z}<extra></extra>",
        )
    else:
        heat = go.Heatmap(
            z=z,
            x=x,
            y=y,
            colorscale="Viridis",
            colorbar_title=var,
            hovertemplate="Fly: %{y}<br>Day: %{x:.2f}<br>" + var + ": %{z}<extra></extra>",
        )
    fig = go.Figure(data=heat)
    fig.update_layout(title=title, xaxis_title="Days elapsed", yaxis_title="Fly ID")

    # --- x-axis: truncate to the finite-data extent (not the full span) ---- #
    if z.size and np.isfinite(z).any():
        col_has = np.isfinite(z).any(axis=0)
        x_fin = x[col_has]
        if x_fin.size:
            fig.update_xaxes(range=[float(x_fin.min()), float(x_fin.max())])

    # --- y-axis: group name centred on each contiguous block --------------- #
    if grp_of is not None and len(ids_ord):
        grp_ord = [grp_of.get(f, "") for f in ids_ord]
        tickvals, ticktext = [], []
        i = 0
        while i < len(grp_ord):
            j = i
            while j < len(grp_ord) and grp_ord[j] == grp_ord[i]:
                j += 1
            mid = (i + j - 1) // 2
            tickvals.append(ids_ord[mid])
            ticktext.append(grp_ord[i])
            i = j
        fig.update_yaxes(tickmode="array", tickvals=tickvals, ticktext=ticktext)
        fig.update_layout(yaxis_title="Group")

    return fig


def daily_pattern_line(
    ds: xr.Dataset,
    value_col: str,
    title: str,
    selected_genotypes=None,
    selected_temperatures=None,
    phase_label=None,
    bin_size_minutes=30,
) -> go.Figure:
    """
    Create a ZT-binned daily pattern line plot with SEM shading, grouped by condition.

    Parameters
    ----------
    ds : xr.Dataset
    value_col : str
        Variable to plot (e.g., 'activity', 'sleep').
    title : str
    selected_genotypes : list, optional
        If provided, only these genotypes are plotted.
    selected_temperatures : list, optional
        If provided, only these temperatures are plotted.
    phase_label : str or None
        One of ``'full'``, ``'LD'``, ``'DD'``. When ``'DD'``, the x-axis
        label switches to ``'CT (hours)'`` (circadian time, anchored to
        last lights-on) so DD waveforms aren't mis-read as ZT.
    bin_size_minutes : int, default 30
        ZT bin width; wider bins give a smoother waveform. Wired to the
        Visualization page's "Bin size" control.

    Returns
    -------
    go.Figure
    """
    if ds is None:
        return go.Figure()

    # Apply group filters before computing ZT bins
    if selected_genotypes is not None or selected_temperatures is not None:
        filter_mask = True
        if "genotype" in ds.coords and selected_genotypes:
            filter_mask = filter_mask & ds["genotype"].isin(selected_genotypes)
        if "temperature" in ds.coords and selected_temperatures:
            filter_mask = filter_mask & ds["temperature"].isin(selected_temperatures)

        if hasattr(filter_mask, "any"):
            filtered_ids = ds["id"].where(filter_mask, drop=True)
            if len(filtered_ids) == 0:
                return go.Figure().add_annotation(
                    text="No data available for selected groups.",
                    xref="paper",
                    yref="paper",
                    x=0.5,
                    y=0.5,
                    showarrow=False,
                )
            ds = ds.sel(id=filtered_ids)

    try:
        df = dam_utilities.get_zt_binned_dataframe(ds, value_col, bin_size_minutes)
    except Exception:
        return go.Figure()

    if df.empty:
        return go.Figure()

    # Build group labels for each fly
    id_to_group = {}
    for fly_id in df["id"].unique():
        try:
            fly_ds = ds.sel(id=fly_id)
            if "group" in ds.coords:
                id_to_group[fly_id] = str(fly_ds["group"].item())
            elif "genotype" in ds.coords and "temperature" in ds.coords:
                id_to_group[fly_id] = f"{fly_ds['genotype'].item()}-{fly_ds['temperature'].item()}"
            else:
                id_to_group[fly_id] = "All"
        except (KeyError, IndexError):
            id_to_group[fly_id] = "All"

    df["group"] = df["id"].map(id_to_group)

    # Aggregate: mean and SEM per group per ZT bin
    agg = (
        df.groupby(["group", "zt_bin_minute"])
        .agg(
            mean_val=(value_col, "mean"),
            sem_val=(
                value_col,
                lambda x: x.std(ddof=1) / (len(x.dropna()) ** 0.5) if len(x.dropna()) > 1 else 0,
            ),
        )
        .reset_index()
    )

    # 0/1 fraction masks (sleep, moving) read most naturally as a PERCENT of measured
    # time. The bin value is already the fraction over MEASURED minutes (the -1 missing
    # sentinel is excluded in get_zt_binned_dataframe, §2a), so scale to % for display
    # and label the axis accordingly. Activity (counts) is left as-is.
    _pct_labels = {"sleep": "% time asleep", "moving": "% time moving"}
    if value_col in _pct_labels:
        agg["mean_val"] = agg["mean_val"] * 100.0
        agg["sem_val"] = agg["sem_val"] * 100.0
        y_label = _pct_labels[value_col]
    else:
        y_label = value_col.capitalize()
    agg["zt_hours"] = dam_utilities.zt_bin_to_hours(agg["zt_bin_minute"], bin_size_minutes)

    fig = go.Figure()
    # Colours assigned EXPLICITLY, one per group, and the SEM band takes its
    # group's own colour. Two things were wrong with leaving it to Plotly:
    #
    # - The band was a hardcoded blue, so every group's error region was blue
    #   whatever colour its mean line happened to get.
    # - Each group adds TWO traces, and plotly.js walks its colourway by trace
    #   index, so the mean lines landed on every OTHER colour. With a 10-colour
    #   cycle that means six groups wrap around and the sixth is drawn in the
    #   first one's colour — two genotypes rendered identically.
    groups_in_order = [str(g) for g in agg["group"].unique()]
    palette = pc.qualitative.Plotly
    for idx, grp_key in enumerate(groups_in_order):
        sub = agg[agg["group"].astype(str) == grp_key]
        color = palette[idx % len(palette)]
        fig.add_trace(
            go.Scatter(
                x=sub["zt_hours"],
                y=sub["mean_val"],
                mode="lines",
                name=grp_key,
                legendgroup=grp_key,
                line=dict(color=color),
            )
        )
        # SEM shading — SAME legendgroup as the mean line so a legend click toggles
        # BOTH the mean AND its error band together (not just the line).
        fig.add_trace(
            go.Scatter(
                x=pd.concat([sub["zt_hours"], sub["zt_hours"][::-1]]),
                y=pd.concat(
                    [sub["mean_val"] + sub["sem_val"], (sub["mean_val"] - sub["sem_val"])[::-1]]
                ),
                fill="toself",
                line=dict(color="rgba(0,0,0,0)"),
                fillcolor=_rgba(color, 0.15),
                legendgroup=grp_key,
                showlegend=False,
                hoverinfo="skip",
            )
        )

    is_dd = (str(phase_label).upper() == "DD") if phase_label else False
    x_label = "CT (hours, subjective time)" if is_dd else "ZT (hours)"
    # Transparent background + explicit BLACK text so the "Download plot as PNG" export
    # is a clear-background figure with legible labels (not the theme's white text, which
    # is invisible on a clear/light export). Page 4 renders these with theme=None so this
    # styling — not Streamlit's dark theme — drives both the on-screen chart and the PNG.
    _BLACK = "black"
    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=_BLACK),
        legend=dict(font=dict(color=_BLACK)),
    )
    fig.update_xaxes(
        range=[0, 24],
        dtick=4,
        title_font=dict(color=_BLACK),
        tickfont=dict(color=_BLACK),
        linecolor=_BLACK,
    )
    fig.update_yaxes(title_font=dict(color=_BLACK), tickfont=dict(color=_BLACK), linecolor=_BLACK)
    return fig


_SLEEP_STATE_VARS = [
    ("sleep_short", "Short"),
    ("sleep_intermediate", "Intermediate"),
    ("sleep_long", "Long"),
]


def per_fly_sleep_state_totals(
    ds, selected_genotypes=None, selected_temperatures=None, as_percent=False
):
    """Per-FLY time in each Abhilash sleep state (Short / Intermediate / Long) —
    ONE row per fly: ``ID``, ``Group``, and one column per state present. This is the
    raw per-fly data behind :func:`sleep_state_totals_bars`: that function's group
    mean±SEM is just this table melted and aggregated by group, so the per-fly export
    and the plotted bars cannot drift.

    Minutes in a state = count of that state's int8 mask == 1 (1 = in state, 0 = not,
    −1 = missing — §2a: missing is excluded, never counted as 0/awake).
    ``as_percent=True`` gives each state as a % of that fly's total classified sleep.
    Rows are sorted alphabetically by Group then ID. Empty DataFrame if no
    sleep-state masks are present or no fly passes the group filter.
    """
    present = [(v, lbl) for v, lbl in _SLEEP_STATE_VARS if v in ds.data_vars]
    cols = ["ID", "Group"] + [lbl for _, lbl in present]
    if not present:
        return pd.DataFrame(columns=["ID", "Group"])

    # Group filter (mirrors daily_pattern_line so the sidebar selection applies).
    if selected_genotypes is not None or selected_temperatures is not None:
        filter_mask = True
        if "genotype" in ds.coords and selected_genotypes:
            filter_mask = filter_mask & ds["genotype"].isin(selected_genotypes)
        if "temperature" in ds.coords and selected_temperatures:
            filter_mask = filter_mask & ds["temperature"].isin(selected_temperatures)
        if hasattr(filter_mask, "any"):
            fids = ds["id"].where(filter_mask, drop=True)
            if len(fids) == 0:
                return pd.DataFrame(columns=cols)
            ds = ds.sel(id=fids)

    def _grp(fid):
        try:
            f = ds.sel(id=fid)
            if "group" in ds.coords:
                return str(f["group"].item())
            if "genotype" in ds.coords and "temperature" in ds.coords:
                return f"{f['genotype'].item()}-{f['temperature'].item()}"
        except Exception:
            pass
        return "All"

    rows = []
    for fid in ds["id"].values:
        mins = {lbl: int(np.sum(np.asarray(ds[v].sel(id=fid).values) == 1)) for v, lbl in present}
        total = sum(mins.values())
        row = {"ID": str(fid), "Group": _grp(fid)}
        for lbl, m in mins.items():
            row[lbl] = (m / total * 100.0) if (as_percent and total > 0) else float(m)
        rows.append(row)

    out = pd.DataFrame(rows, columns=cols)
    return out.sort_values(["Group", "ID"]).reset_index(drop=True) if not out.empty else out


def sleep_state_totals_bars(
    ds, selected_genotypes=None, selected_temperatures=None, as_percent=False
):
    """Grouped bars of time in each Abhilash sleep state (short / intermediate /
    long) per genotype group — surfaces the 'Sleep State Thresholds' data computed
    on the Preprocessing page (which was otherwise unused).

    Bars show the GROUP MEAN across flies (± SEM), computed as the group aggregation
    of :func:`per_fly_sleep_state_totals` (one per-fly computation, no drift — §2d).
    ``as_percent=True`` shows each state as a % of that fly's total classified sleep
    instead of absolute minutes. Returns ``(go.Figure, tidy DataFrame)``; the
    DataFrame (group, state, mean, sem, n) is the export payload. If the masks are
    absent, returns a figure with a "run Sleep State Thresholds first" note and
    ``None``.
    """
    present = [(v, lbl) for v, lbl in _SLEEP_STATE_VARS if v in ds.data_vars]
    if not present:
        fig = go.Figure().add_annotation(
            text="No sleep-state data — run 'Sleep State Thresholds' on the "
            "Preprocessing page first.",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
        )
        return fig, None

    per_fly = per_fly_sleep_state_totals(ds, selected_genotypes, selected_temperatures, as_percent)
    if per_fly.empty:
        return go.Figure().add_annotation(
            text="No data for selected groups.",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
        ), None

    state_order = [lbl for _, lbl in present if lbl in per_fly.columns]
    df = per_fly.melt(
        id_vars=["ID", "Group"], value_vars=state_order, var_name="state", value_name="value"
    )
    stat = (
        df.groupby(["Group", "state"])["value"]
        .agg(
            mean="mean",
            sem=lambda x: x.std(ddof=1) / (x.notna().sum() ** 0.5) if x.notna().sum() > 1 else 0.0,
            n=lambda x: int(x.notna().sum()),
        )
        .reset_index()
        .rename(columns={"Group": "group"})
    )

    groups = sorted(stat["group"].unique())
    palette = {"Short": "#56B4E9", "Intermediate": "#E69F00", "Long": "#009E73"}
    ylab = "% of classified sleep" if as_percent else "Minutes per fly (mean ± SEM)"
    fig = go.Figure()
    for lbl in state_order:
        sub = stat[stat["state"] == lbl].set_index("group").reindex(groups)
        fig.add_trace(
            go.Bar(
                name=lbl,
                x=groups,
                y=sub["mean"].values,
                error_y=dict(type="data", array=np.nan_to_num(sub["sem"].values), visible=True),
                marker_color=palette.get(lbl),
            )
        )
    fig.update_layout(
        barmode="group",
        title="Sleep-state totals by genotype",
        xaxis_title="Group",
        yaxis_title=ylab,
        legend_title="Sleep state",
    )
    apply_category_ticks(fig, groups)
    return fig, stat


def sleep_bout_duration_lines(
    curves_df: pd.DataFrame,
    stats_result: dict | None = None,
    method: str = "kde",
    show_individual: bool = True,
    stat_col: str = "log_mean_duration_min",
) -> go.Figure:
    """
    Per-fly sleep-bout-duration curve, overlaid as one line per genotype
    (mean ± SEM across flies, with faint per-fly lines underneath).

    Replaces the old pooled-bout histogram: pooling every bout across flies
    let flies with more bouts dominate the shape and made 3+ genotypes
    unreadable as overlapping bars. Here every fly contributes exactly one
    curve first, so genotypes overlay cleanly as lines regardless of how
    fragmented any one fly's sleep is.

    **Takes the computed frames, not a Dataset.** This used to accept ``ds`` and
    defer-import three ``sleep_analysis`` functions to compute the curves, the
    per-fly summary and the group test itself — a deferred import that existed
    only to dodge a circular one, which is the signal that the compute/render
    seam was in the wrong place. It was also the sole caller of all three, so
    the analysis effectively lived inside the plotting module. The caller now
    computes and passes the result; ``plotting`` only renders.

    Parameters
    ----------
    curves_df : pd.DataFrame
        Tidy ``(id, group, x, y)`` from
        ``sleep_analysis.per_fly_bout_duration_curves``. Empty or None renders
        the "no data" placeholder.
    stats_result : dict, optional
        From ``sleep_analysis.bout_duration_group_stats``. Only used for the
        significance annotation; omit it and the annotation is skipped.
    method : {'kde', 'survival'}
        Which curve ``curves_df`` holds — chooses the axis labels and scales.
        'kde': density of log10(bout duration) per fly, log-x.
        'survival': empirical P(bout duration > t) per fly, log-y.
    show_individual : bool
        Draw faint per-fly lines under each genotype's bold mean.
    stat_col : str
        Named in the significance annotation, so it matches what the caller
        actually tested.

    Returns
    -------
    go.Figure
    """
    if curves_df is None or getattr(curves_df, "empty", True):
        return go.Figure().add_annotation(
            text="No sleep bout duration data available for the selected groups.",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
        )

    pivot = curves_df.pivot_table(index=["group", "id"], columns="x", values="y")
    x_axis = pivot.columns.to_numpy(dtype=float)
    per_group_curves = {
        str(g): sub.to_numpy(dtype=float) for g, sub in pivot.groupby(level="group")
    }

    if method == "kde":
        xlabel, ylabel = "Bout duration (min, log scale)", "Density (log10-duration space)"
        title = "Sleep Bout Duration — per-fly log-duration KDE"
        x_log, y_log = True, False
    else:
        xlabel, ylabel = "t (min)", "P(bout duration > t)"
        title = "Sleep Bout Duration — per-fly survival curve (CCDF)"
        x_log, y_log = False, True

    fig = group_spectrum_plot(
        per_group_curves,
        x_axis,
        xlabel=xlabel,
        ylabel=ylabel,
        title=title,
        x_log=x_log,
        y_log=y_log,
        ref_period=None,
        zero_line=False,
        show_individual=show_individual,
    )

    p_val = (stats_result or {}).get("pvalue", float("nan"))
    if np.isfinite(p_val):
        p_str = "p<0.001" if p_val < 0.001 else f"p={p_val:.3f}"
        label = "ANOVA" if stats_result["test"] == "anova" else "Kruskal-Wallis"
        fig.add_annotation(
            xref="paper",
            yref="paper",
            x=0.99,
            y=1.06,
            xanchor="right",
            yanchor="bottom",
            text=f"{label} on {stat_col}: {p_str}",
            showarrow=False,
            font=dict(size=11, color="gray"),
        )

    return fig


def per_fly_summary_table(
    ds: xr.Dataset,
    variable="activity",
    selected_genotypes=None,
    selected_temperatures=None,
    bin_size_minutes=30,
) -> pd.DataFrame:
    """Per-FLY total minutes of ``variable`` for All Day / Day Only (ZT 0–12) /
    Night Only (ZT 12–24) — ONE row per fly (``ID``, ``Group``, and the three
    totals). This is the raw per-fly data downstream of which :func:`summary_table`
    is just a group mean±SEM, so a per-fly export "for stats" and the plotted
    group summary cannot drift.

    Rows are sorted alphabetically by Group then ID. Empty DataFrame when there is
    no usable data.
    """
    cols = ["ID", "Group", "All Day", "Day Only", "Night Only"]
    if ds is None or variable not in ds.data_vars:
        return pd.DataFrame(columns=cols)

    # Apply group filters (same as the group summary consumed them)
    if selected_genotypes is not None or selected_temperatures is not None:
        filter_mask = True
        if "genotype" in ds.coords and selected_genotypes:
            filter_mask = filter_mask & ds["genotype"].isin(selected_genotypes)
        if "temperature" in ds.coords and selected_temperatures:
            filter_mask = filter_mask & ds["temperature"].isin(selected_temperatures)
        if hasattr(filter_mask, "any"):
            filtered_ids = ds["id"].where(filter_mask, drop=True)
            if len(filtered_ids) == 0:
                return pd.DataFrame(columns=cols)
            ds = ds.sel(id=filtered_ids)

    try:
        df = dam_utilities.get_zt_binned_dataframe(ds, variable, bin_size_minutes)
    except Exception:
        return pd.DataFrame(columns=cols)
    if df.empty:
        return pd.DataFrame(columns=cols)

    # Build group labels
    id_to_group = {}
    for fly_id in df["id"].unique():
        try:
            fly_ds = ds.sel(id=fly_id)
            if "group" in ds.coords:
                id_to_group[fly_id] = str(fly_ds["group"].item())
            elif "genotype" in ds.coords and "temperature" in ds.coords:
                id_to_group[fly_id] = f"{fly_ds['genotype'].item()}-{fly_ds['temperature'].item()}"
            else:
                id_to_group[fly_id] = "All Flies"
        except (KeyError, IndexError):
            id_to_group[fly_id] = "All Flies"

    df["group"] = df["id"].map(id_to_group)

    rows = []
    for fly_id in df["id"].unique():
        fly_df = df[df["id"] == fly_id]
        # Sum within each period and multiply by bin width to get total minutes.
        # Day/Night is partitioned by the bin's CONTENT — which side of ZT12 (720 min)
        # the bin's data lies — NOT by the profile plot's display label. Bins are
        # [start, start+bin), so a bin is Day iff start < 720. This is independent of
        # the closing-edge label convention (dam_utilities.zt_bin_to_hours), so the
        # export totals never move when that display convention changes.
        all_day_sum = fly_df[variable].sum() * bin_size_minutes
        day_df = fly_df[fly_df["zt_bin_minute"] < 720]
        night_df = fly_df[(fly_df["zt_bin_minute"] >= 720) & (fly_df["zt_bin_minute"] < 1440)]
        day_sum = day_df[variable].sum() * bin_size_minutes if len(day_df) > 0 else np.nan
        night_sum = night_df[variable].sum() * bin_size_minutes if len(night_df) > 0 else np.nan
        rows.append(
            {
                "ID": fly_id,
                "Group": fly_df["group"].iloc[0],
                "All Day": all_day_sum,
                "Day Only": day_sum,
                "Night Only": night_sum,
            }
        )

    out = pd.DataFrame(rows, columns=cols)
    return out.sort_values(["Group", "ID"]).reset_index(drop=True)


def summary_table(
    ds: xr.Dataset,
    variable="activity",
    selected_genotypes=None,
    selected_temperatures=None,
    bin_size_minutes=30,
    phase_label=None,
) -> pd.DataFrame:
    """Per-group mean + SEM total minutes of ``variable`` for All Day / Day Only
    (ZT 0–12) / Night Only (ZT 12–24), across the flies in each group.

    The group numbers are just :func:`per_fly_summary_table` aggregated by group
    (one computation, no drift) — the same DataFrame :func:`summary_bars` plots
    and the page exports. Returns columns ``group``, ``n`` (flies), and per period
    the mean and its ``*_sem``. Empty DataFrame when there is no usable data.
    ``phase_label`` is accepted for signature parity with :func:`summary_bars`
    (the per-fly ZT sums are phase-independent; only the display labels differ).
    """
    cols = [
        "group",
        "n",
        "All Day",
        "All Day_sem",
        "Day Only",
        "Day Only_sem",
        "Night Only",
        "Night Only_sem",
    ]
    per_fly = per_fly_summary_table(
        ds, variable, selected_genotypes, selected_temperatures, bin_size_minutes
    )
    if per_fly.empty:
        return pd.DataFrame(columns=cols)

    def _sem(vals):
        """SEM across flies (SD / sqrt(n)); NaN-safe, 0.0 when n < 2."""
        vals = vals.dropna()
        n = len(vals)
        return float(vals.std(ddof=1) / (n**0.5)) if n > 1 else 0.0

    summary_data = []
    for group in sorted(per_fly["Group"].unique()):
        g = per_fly[per_fly["Group"] == group]
        summary_data.append(
            {
                "group": group,
                "n": int(len(g)),
                "All Day": g["All Day"].mean(),
                "All Day_sem": _sem(g["All Day"]),
                "Day Only": g["Day Only"].mean(),
                "Day Only_sem": _sem(g["Day Only"]),
                "Night Only": g["Night Only"].mean(),
                "Night Only_sem": _sem(g["Night Only"]),
            }
        )

    return pd.DataFrame(summary_data, columns=cols)


def summary_bars(
    ds: xr.Dataset,
    variable="activity",
    selected_genotypes=None,
    selected_temperatures=None,
    bin_size_minutes=30,
    phase_label=None,
) -> go.Figure:
    """
    Create a grouped bar chart: mean total minutes of activity or sleep for
    All Day, Day Only (ZT 0–12), and Night Only (ZT 12–24) per group, with
    SEM error bars (SD / sqrt(n) across the flies in each group).

    The per-group numbers come from :func:`summary_table` (the same DataFrame
    the Visualization page exports as CSV — one computation, no drift).

    Parameters
    ----------
    ds : xr.Dataset
    variable : str
        Variable to summarize ('activity' or 'sleep').
    selected_genotypes : list, optional
    selected_temperatures : list, optional
    bin_size_minutes : int, default 30
        ZT bin width passed through to ``get_zt_binned_dataframe``.
    phase_label : str or None
        One of ``'full'``, ``'LD'``, ``'DD'``. When ``'DD'``, the bar
        labels switch to **subjective time** ("Subjective day (CT 0–12)"
        / "Subjective night (CT 12–24)") and the title gets a "(DD —
        circadian time)" suffix to make the free-running interpretation
        explicit. There is no light/dark in DD, so ZT-style "Day/Night"
        labels would be misleading even though the math (sums over
        per-fly ZT bins) is unchanged. See core/dataset_meta.py for the
        canonical phase metadata convention.

    Returns
    -------
    go.Figure
    """
    summary_df = summary_table(
        ds, variable, selected_genotypes, selected_temperatures, bin_size_minutes, phase_label
    )
    if summary_df.empty:
        return go.Figure().add_annotation(
            text="No summary data available for the current selection.",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
        )

    ylabel = f"Total {variable.replace('_', ' ').capitalize()} (minutes)"

    # Phase-aware labels. In DD there's no light/dark, so the bins are
    # really subjective time relative to last lights-on (CT). Switch the
    # legend names + title accordingly so DD plots aren't misread as
    # "day" vs "night" data.
    is_dd = (str(phase_label).upper() == "DD") if phase_label else False
    if is_dd:
        full_label = "All 24h"
        first_half_label = "Subjective day (CT 0–12)"
        second_half_label = "Subjective night (CT 12–24)"
        title_suffix = " (DD — circadian time)"
    else:
        full_label = "All Day"
        first_half_label = "Day Only (ZT 0–12)"
        second_half_label = "Night Only (ZT 12–24)"
        title_suffix = ""

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name=full_label,
            x=summary_df["group"],
            y=summary_df["All Day"],
            error_y=dict(type="data", array=summary_df["All Day_sem"], visible=True),
            marker_color="steelblue",
        )
    )
    fig.add_trace(
        go.Bar(
            name=first_half_label,
            x=summary_df["group"],
            y=summary_df["Day Only"],
            error_y=dict(type="data", array=summary_df["Day Only_sem"], visible=True),
            marker_color="gold",
        )
    )
    fig.add_trace(
        go.Bar(
            name=second_half_label,
            x=summary_df["group"],
            y=summary_df["Night Only"],
            error_y=dict(type="data", array=summary_df["Night Only_sem"], visible=True),
            marker_color="navy",
        )
    )

    fig.update_layout(
        title=f"Summary: {ylabel} by Period and Group (mean ± SEM){title_suffix}",
        xaxis_title="Group",
        yaxis_title=ylabel,
        barmode="group",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    apply_category_ticks(fig, list(summary_df["group"]))
    return fig


# ===========================================================================
# Sleep State Analysis Plots (Abhilash et al. 2026)
# ===========================================================================





def _state_palette():
    """Paper colours, imported lazily so plotting.py keeps no import-time dep."""
    from sleep_state_metrics import STATE_COLORS, STATE_LABELS

    return STATE_COLORS, STATE_LABELS


CATEGORY_TICKANGLE = -40


def category_tickangle(labels, angle=CATEGORY_TICKANGLE):
    """Tick angle for genotype/group category labels: diagonal unless they are
    both few AND short.

    Two figures already rotated, but only above six groups. Group COUNT is the
    wrong test on its own: six labels like "dsOpa1(32358)+Ldhmut" overprinted
    each other into an unreadable smear at 0 degrees — exactly the case the
    rotation exists for — while six labels like "ctrl" need no rotation at all.
    What decides it is the room each label needs, so the longest one is part of
    the test.

    Lines are measured separately because some x labels stack two facts with a
    ``<br>`` (state over group, group over day); the label's width is its
    widest line, not its total length.
    """
    lines = [
        line
        for label in labels
        for line in str(label).split("<br>")
    ]
    if len(labels) <= 1:
        return 0
    return angle if (len(labels) > 6 or max(map(len, lines), default=0) > 8) else 0


def apply_category_ticks(fig, labels, **kwargs):
    """Rotate a categorical x axis, and let it claim the margin it needs.

    ``automargin`` matters as much as the angle: without it a rotated label is
    drawn into whatever bottom margin the figure already had and is clipped
    rather than overlapped, which trades one unreadable axis for another.
    """
    fig.update_xaxes(
        tickangle=category_tickangle(labels), automargin=True, **kwargs
    )
    return fig


def _rgba(hex_color, alpha):
    r, g, b = (int(hex_color.lstrip("#")[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def _empty(text):
    return go.Figure().add_annotation(
        text=text, xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False
    )


# The paper draws its scalograms on a jet-like ramp with a fixed 0-1.5 z range
# ("All scalograms have a z axis scale that ranges from 0 to 1.5"). Plotly's
# "Jet" is the same ramp, and pinning the range is what makes two panels — or
# our panel and the printed one — comparable at a glance.
SCALOGRAM_COLORSCALE = "Jet"
SCALOGRAM_ZMAX = 1.5


def _polar_layout(tick_prefix="", inner_hole=0.0, label_hours=(0, 6, 12, 18)):
    """Clock-face polar axis: 00 at the top, time increasing clockwise.

    Every circular figure in the paper is oriented this way — 00 top, 06 right,
    12 bottom, 18 left — with spokes every 3 h and no radial tick labels. In
    Plotly that is ``rotation=90`` (put theta=0 at the top) plus
    ``direction='clockwise'``; the authors' own ``phase::rosePlotsSleep`` sets
    the same direction and the same 8 tick labels.
    """
    # Spokes every 3 h but LABELS only where `label_hours` asks for them. The
    # paper prints 00 and 12 on every panel of a row and puts 18 on the leftmost
    # ring and 06 on the rightmost only, because at five panels wide the 06 of
    # one ring and the 18 of the next land on top of each other.
    return dict(
        hole=inner_hole,
        angularaxis=dict(
            tickmode="array",
            tickvals=list(range(0, 360, 45)),
            ticktext=[
                f"{tick_prefix}{h:02d}" if h in tuple(label_hours) else ""
                for h in range(0, 24, 3)
            ],
            direction="clockwise",
            rotation=90,
            gridcolor="rgba(0,0,0,0.35)",
            linecolor="black",
        ),
        radialaxis=dict(showticklabels=False, showgrid=False, showline=False, ticks=""),
    )


def _wedges(theta_start_deg, theta_end_deg, radius, n_points=2):
    """Closed polygon for one rose wedge, as ``phase`` draws them.

    ``phase::rosePlotsSleep`` adds each bin as a ``scatterpolar`` polygon
    ``r = c(0, x, x, 0)`` / ``theta = c(0, y - width, y, 0)`` and fills it,
    which is why the printed wedges are contiguous with no inter-bar gap.
    Barpolar leaves hairlines between bars at 48 bins, so this matches the
    reference implementation rather than approximating it.
    """
    thetas = np.linspace(theta_start_deg, theta_end_deg, n_points)
    r = [0.0] + [radius] * len(thetas) + [0.0]
    theta = [thetas[0]] + list(thetas) + [thetas[-1]]
    return r, theta


def _day_night_wedges(phase_label):
    """Background shading for the subjective/actual day and night halves.

    Figure 3A (DD) shades CT00-12 light grey and CT12-24 darker grey; Figure 3B
    (ramped light) has no shading, only min/max light annotations. Drawn as two
    full-radius wedges underneath the data.
    """
    if phase_label == "DD":
        return [(0.0, 180.0, "rgba(0,0,0,0.08)"), (180.0, 360.0, "rgba(0,0,0,0.21)")]
    if phase_label == "LD":
        return [(180.0, 360.0, "rgba(0,0,0,0.21)")]
    return []


def _add_background_wedge(fig, start_deg, end_deg, fillcolor):
    thetas = np.linspace(start_deg, end_deg, 60)
    fig.add_trace(
        go.Scatterpolar(
            r=[0.0] + [1.0] * len(thetas) + [0.0],
            theta=[thetas[0]] + list(thetas) + [thetas[-1]],
            mode="lines",
            fill="toself",
            fillcolor=fillcolor,
            line=dict(color="rgba(0,0,0,0)"),
            hoverinfo="skip",
            showlegend=False,
        )
    )
    # Plotly draws traces in insertion order, so the shading has to be moved
    # behind the wedges that were added first.
    fig.data = tuple(fig.data[-1:]) + tuple(fig.data[:-1])


def _add_rose_series(fig, sdf, state, bin_size_min, normalise, colors, opacity=1.0):
    """Add one series' wedges. Shared by rose_plot and rose_plot_with_activity.

    Collapses to ONE value per bin first. Handed a frame still covering several
    groups, the previous behaviour was to draw every group's wedge set on top of
    one another — 48 wedges became 192, all overlapping, which looks like a
    plausible rose and is not one. Pooling the groups is the interpretable
    reading of such a frame; pass one group at a time to plot them separately.
    """
    sdf = sdf.groupby("zt_bin_minute", as_index=False)["mean"].mean()
    sdf = sdf.sort_values("zt_bin_minute")
    values = np.asarray(sdf["mean"].values, dtype=float)
    values = np.where(np.isfinite(values), values, 0.0)
    peak = values.max() if values.size else 0.0
    scale = peak if (normalise and peak > 0) else 1.0
    width_deg = bin_size_min / 1440.0 * 360.0
    color = colors.get(state, "#333333")

    for start_min, value in zip(sdf["zt_bin_minute"].values, values):
        # Bins are right-labelled, matching the reference implementation's
        # theta = c(0, y - width, y, 0): the wedge ENDS at the labelled edge.
        end_deg = (float(start_min) + bin_size_min) / 1440.0 * 360.0
        r, theta = _wedges(end_deg - width_deg, end_deg, value / scale)
        fig.add_trace(
            go.Scatterpolar(
                r=r,
                theta=theta,
                mode="lines",
                fill="toself",
                fillcolor=_rgba(color, opacity),
                line=dict(color="rgba(40,40,40,0.85)", width=0.6),
                hovertemplate=(
                    f"{state}<br>%{{theta:.1f}}deg<br>{value:.2f}<extra></extra>"
                ),
                showlegend=False,
            )
        )


def _gate_arc(onset_h, offset_h, radius, color, width, alpha, name=None, show_legend=False):
    """One gate as a constant-radius arc, unwrapped so it never runs backwards."""
    start = onset_h % 24.0
    end = offset_h % 24.0
    if end <= start:
        end += 24.0  # gate crosses the origin: go forward through it
    n = max(8, int((end - start) * 6))
    hours = np.linspace(start, end, n)
    theta = (hours % 24.0) / 24.0 * 360.0
    return go.Scatterpolar(
        r=np.full(n, radius),
        theta=theta,
        mode="lines",
        line=dict(color=_rgba(color, alpha), width=width),
        name=name,
        showlegend=show_legend,
        hovertemplate=(
            f"{name or ''}<br>gate {start % 24:.1f}h to {end % 24:.1f}h<extra></extra>"
        ),
    )


def _period_ticks(pmin, pmax):
    """Log period ticks: the paper's values where they fit, denser when they don't.

    Figure 5 labels 1, 2, 4, 8, 12, 17, 24 and 35 h. Those are right for a full
    1-32 h axis but nearly all fall outside a cropped ultradian band — a 2-6 h
    panel keeps only ``4``, leaving a log axis with a single tick. So fall back
    to a finer ladder, and always keep at least the two endpoints.
    """
    coarse = [1, 2, 4, 8, 12, 17, 24, 35]
    vals = [c for c in coarse if pmin <= c <= pmax]
    if len(vals) < 3:
        fine = [1, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10, 12, 17, 24, 35]
        vals = [c for c in fine if pmin <= c <= pmax]
    if len(vals) < 2:
        vals = [round(pmin, 2), round(pmax, 2)]
    return vals

def _day_night_bar(fig, n_days, row, col, phase_label="DD"):
    """The light/dark strip the paper puts above each scalogram.

    Figure 5A carries a bar of alternating light-grey and black blocks across
    the top marking subjective day and night, plus dashed verticals at the
    transitions. Under DD both halves are subjective, which is why day is grey
    rather than white.
    """
    day_fill = "rgba(190,190,190,1)" if phase_label == "DD" else "rgba(250,250,210,1)"
    for day in range(int(np.ceil(n_days))):
        for half, fill in ((0.0, day_fill), (0.5, "rgba(0,0,0,1)")):
            x0 = day + half
            if x0 >= n_days:
                continue
            fig.add_shape(
                type="rect",
                x0=x0,
                x1=min(x0 + 0.5, n_days),
                y0=1.01,
                y1=1.06,
                yref="y domain",
                fillcolor=fill,
                line=dict(width=0.4, color="black"),
                row=row,
                col=col,
            )
        fig.add_vline(
            x=day + 0.5,
            line=dict(color="rgba(90,90,90,0.55)", width=1, dash="dash"),
            row=row,
            col=col,
        )

def _get_group_colors(groups):
    """Return a dict mapping group names to Plotly color strings."""
    palette = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]
    return {g: palette[i % len(palette)] for i, g in enumerate(sorted(groups))}


def normalized_waveform_overlay(
    waveform_df, phase_label="DD", title="Normalized Daily Sleep Profiles"
):
    """
    Overlay normalized daily sleep waveforms for multiple sleep states.

    Each state's mean ± SEM band is plotted on the same axes, normalized to
    [0, 1] so that waveform shapes can be compared across states and groups.
    Matches Abhilash et al. 2026 Figure 1.

    Parameters
    ----------
    waveform_df : pd.DataFrame
        Output of sleep_state_metrics.compute_normalized_waveforms().
        Columns: 'group', 'state', 'zt_bin_minute', 'mean_normalized',
                 'sem_normalized'.
    phase_label : {'DD', 'LD'}
        Sets the time axis label and the day/night shading. Under DD the axis
        is CT and both halves are subjective, so the "day" half is shaded too;
        calling it ZT there would name the wrong timescale.
    title : str

    Returns
    -------
    go.Figure
    """
    # Paper colours, shared with every other Abhilash-figure renderer. The
    # literals that used to sit here were Plotly's default cycle, which gave
    # short sleep the paper's long-sleep blue and long sleep its activity red.
    STATE_COLORS, STATE_LABELS = _state_palette()
    DASH_STYLES = ["solid", "dash", "dot", "dashdot", "longdash"]

    if waveform_df.empty:
        return go.Figure().add_annotation(
            text="No waveform data available.",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
        )

    fig = go.Figure()
    groups = sorted(waveform_df["group"].unique())

    for g_idx, group in enumerate(groups):
        dash = DASH_STYLES[g_idx % len(DASH_STYLES)]
        gdf = waveform_df[waveform_df["group"] == group]

        for state in ["standard", "short", "intermediate", "long"]:
            sdf = gdf[gdf["state"] == state].sort_values("zt_bin_minute")
            if sdf.empty:
                continue

            color = STATE_COLORS.get(state, "#333333")
            x = (sdf["zt_bin_minute"].values) / 60.0
            y = sdf["mean_normalized"].values
            sem = sdf["sem_normalized"].values

            label = STATE_LABELS.get(state, state)
            name = f"{label} ({group})" if len(groups) > 1 else label

            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    mode="lines",
                    name=name,
                    line=dict(color=color, dash=dash, width=2),
                    legendgroup=f"{state}_{group}",
                )
            )
            # SEM band
            x_fill = np.concatenate([x, x[::-1]])
            y_fill = np.concatenate([y + sem, (y - sem)[::-1]])
            r, g_c, b = tuple(int(color.lstrip("#")[i : i + 2], 16) for i in (0, 2, 4))
            fig.add_trace(
                go.Scatter(
                    x=x_fill,
                    y=y_fill,
                    fill="toself",
                    fillcolor=f"rgba({r},{g_c},{b},0.15)",
                    line=dict(color="rgba(0,0,0,0)"),
                    showlegend=False,
                    legendgroup=f"{state}_{group}",
                    hoverinfo="skip",
                )
            )

    # Two-tone under DD (both halves subjective), one dark block under LD.
    for x0, x1, fill in _day_night_spans(phase_label):
        fig.add_vrect(x0=x0, x1=x1, fillcolor=fill, layer="below", line_width=0)

    fig.update_layout(
        title=title,
        xaxis=dict(
            title=f"{'CT' if phase_label == 'DD' else 'ZT'} (hours)",
            range=[0, 24],
            dtick=4,
        ),
        yaxis=dict(title="Normalized sleep (fraction of max)", range=[0, 1.05]),
        hovermode="x unified",
        legend=dict(orientation="v"),
    )
    return fig


def _day_night_spans(phase_label):
    """Day/night shading for a 0-24 h Cartesian axis, as (x0, x1, fill) spans.

    One definition shared by every Cartesian panel here. Under DD BOTH halves
    are subjective, so the paper shades the day light grey and the night dark
    grey; under LD only the real dark phase is shaded. Returns nothing for a
    ramped light cycle, which the paper annotates instead of shading.
    """
    if phase_label == "DD":
        return [(0, 12, "rgba(0,0,0,0.05)"), (12, 24, "rgba(0,0,0,0.13)")]
    if phase_label == "LD":
        return [(12, 24, "rgba(0,0,0,0.16)")]
    return []


def _night_shading(fig, phase_label, row, col, n_rows):
    """Grey the dark phase on one panel, from the shared spans.

    Drawn per panel rather than with ``row="all"`` because a secondary y axis
    makes the all-rows form ambiguous.
    """
    for x0, x1, fill in _day_night_spans(phase_label):
        fig.add_vrect(
            x0=x0,
            x1=x1,
            fillcolor=fill,
            line_width=0,
            layer="below",
            row=row,
            col=col,
        )


def state_profile_plot(
    profile_stats,
    phase_label="DD",
    states=("short", "intermediate", "long"),
    show_standard_reference=True,
    title="Daily profiles",
):
    """Activity and per-state sleep profiles — the LEFT column of Figure 2.

    This panel had no implementation at all. The Abhilash section carried the
    initiation-probability half of Figure 2 and the rose plots of Figure 3, but
    not the profiles they are read against — and the paper's argument is
    precisely the relationship between them ("the gray line represents profiles
    of standard sleep... Note that the activity counts remain the same within
    each column").

    Four rows: locomotor activity in red, then each state in its own colour
    with **standard sleep drawn behind it in grey** as the reference the paper
    puts in every panel. Sleep is in min/h and activity in counts/h, each on
    its own row, so no dual axis is needed.

    Parameters
    ----------
    profile_stats : pd.DataFrame
        ``sleep_state_metrics.group_profiles`` output for ONE group, including
        the ``'activity'`` and ``'standard'`` states.
    show_standard_reference : bool
        Draw standard sleep behind each state. Turn off to see a state alone.
    """
    from plotly.subplots import make_subplots

    colors, labels = _state_palette()
    if profile_stats is None or profile_stats.empty:
        return _empty("No profile data available.")

    present = [s for s in states if s in set(profile_stats["state"])]
    rows = (["activity"] if "activity" in set(profile_stats["state"]) else []) + present
    if not rows:
        return _empty("No profiles available for the requested states.")

    fig = make_subplots(
        rows=len(rows),
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.045,
        subplot_titles=[labels.get(r, r) for r in rows],
    )

    standard = profile_stats[profile_stats["state"] == "standard"].sort_values("zt_bin_minute")
    has_standard = show_standard_reference and not standard.empty

    for row, state in enumerate(rows, start=1):
        if state != "activity" and has_standard:
            # The paper's grey reference carries its own SEM band, not just a
            # line, so the two states' spreads are comparable by eye.
            _add_profile_trace(
                fig, standard, "standard", colors, row, name="Standard sleep"
            )
        sdf = profile_stats[profile_stats["state"] == state].sort_values("zt_bin_minute")
        _add_profile_trace(fig, sdf, state, colors, row, name=labels.get(state, state))
        _night_shading(fig, phase_label, row, 1, len(rows))
        fig.update_yaxes(
            title_text="Activity (counts/h)" if state == "activity" else "Sleep (min/h)",
            rangemode="tozero",
            row=row,
            col=1,
        )

    fig.update_xaxes(dtick=6, range=[0, 24])
    fig.update_xaxes(
        title_text=f"{'Circadian' if phase_label == 'DD' else 'Zeitgeber'} time (h)",
        row=len(rows),
        col=1,
    )
    # `title or ""` for the same reason as in rose_plot_with_activity: a None
    # title renders as the word "undefined", not as no title.
    fig.update_layout(title=title or "", height=175 * len(rows) + 90, showlegend=False)
    # Colour each panel label by its state, the way the paper prints them —
    # with a grey reference line in every sleep panel, the label is what tells
    # you which trace is the subject.
    for note, state in zip(fig.layout.annotations, rows):
        note.update(font=dict(size=12, color=colors.get(state, "#333333")))
    return fig

def _add_profile_trace(fig, sdf, state, colors, row, name=None, band=True):
    """One mean +/- SEM profile trace."""
    color = colors.get(state, "#333333")
    x = np.asarray(sdf["zt_bin_minute"].values, dtype=float) / 60.0
    y = np.asarray(sdf["mean"].values, dtype=float)
    if band and "sem" in sdf:
        sem = np.nan_to_num(np.asarray(sdf["sem"].values, dtype=float))
        fig.add_trace(
            go.Scatter(
                x=np.concatenate([x, x[::-1]]),
                y=np.concatenate([y + sem, (y - sem)[::-1]]),
                fill="toself",
                fillcolor=_rgba(color, 0.20),
                line=dict(color="rgba(0,0,0,0)"),
                showlegend=False,
                hoverinfo="skip",
            ),
            row=row,
            col=1,
        )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="lines",
            line=dict(color=color, width=2 if band else 1.4),
            name=name or state,
            hovertemplate="%{x:.1f} h<br>%{y:.2f}<extra></extra>",
            showlegend=False,
        ),
        row=row,
        col=1,
    )

def initiation_probability_plot(
    init_stats,
    activity_stats=None,
    phase_label="DD",
    title="Probability of initiating a sleep bout",
):
    """Bars of P(initiation) per hour with the activity profile overlaid — Figure 2 right.

    The paper's panels are bar charts with SEM whiskers on a left axis, and the
    locomotor activity profile as a red line on a **secondary right axis** in
    counts/h. That pairing is the whole point of the figure — it is how you see
    that "P(long) was highest in the hour immediately following the evening
    peak of activity" — and the previous version had neither the bars nor the
    activity overlay, drawing per-fly spaghetti lines instead.

    Parameters
    ----------
    init_stats : pd.DataFrame
        ``sleep_state_metrics.group_initiation_probability`` output for one
        group: ``state``, ``bin_hour``, ``mean``, ``sem``.
    activity_stats : pd.DataFrame or None
        ``group_profiles`` rows for ``state == 'activity'`` (same group), giving
        the red overlay. Omitted, the panels just lose the overlay.
    """
    from plotly.subplots import make_subplots

    colors, labels = _state_palette()
    if init_stats is None or init_stats.empty:
        return _empty("No initiation probability data available.")

    states = [s for s in ("standard", "short", "intermediate", "long") if s in set(init_stats["state"])]
    if not states:
        return _empty("No initiation probability data available.")

    fig = make_subplots(
        rows=len(states),
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.045,
        subplot_titles=[labels.get(s, s) for s in states],
        specs=[[{"secondary_y": True}] for _ in states],
    )

    for row, state in enumerate(states, start=1):
        sdf = init_stats[init_stats["state"] == state].sort_values("bin_hour")
        color = colors.get(state, "#333333")
        fig.add_trace(
            go.Bar(
                x=sdf["bin_hour"].values,
                y=sdf["mean"].values,
                error_y=dict(
                    type="data",
                    array=sdf["sem"].values,
                    visible=True,
                    thickness=1,
                    width=2,
                    color="rgba(30,30,30,0.85)",
                ),
                marker=dict(color=color, line=dict(color="rgba(30,30,30,0.9)", width=0.6)),
                name=labels.get(state, state),
                showlegend=False,
                hovertemplate="hour %{x}<br>P = %{y:.4f}<extra></extra>",
            ),
            row=row,
            col=1,
            secondary_y=False,
        )

        if activity_stats is not None and not activity_stats.empty:
            adf = activity_stats.sort_values("zt_bin_minute")
            x_h = adf["zt_bin_minute"].values / 60.0
            fig.add_trace(
                go.Scatter(
                    x=x_h,
                    y=adf["mean"].values,
                    mode="lines",
                    line=dict(color=colors["activity"], width=1.8),
                    name="Activity",
                    showlegend=(row == 1),
                    hovertemplate="CT/ZT %{x:.1f} h<br>%{y:.1f} counts/h<extra></extra>",
                ),
                row=row,
                col=1,
                secondary_y=True,
            )
            fig.update_yaxes(
                title_text="Activity (counts/h)",
                color=colors["activity"],
                showgrid=False,
                secondary_y=True,
                row=row,
                col=1,
            )

        # Night shading: the actual dark phase under LD, the subjective night
        # under DD. Drawn per panel because a secondary axis makes add_vrect
        # with row='all' ambiguous.
        fig.add_vrect(
            x0=12,
            x1=24,
            fillcolor="rgba(0,0,0,0.10)" if phase_label == "DD" else "rgba(0,0,0,0.16)",
            line_width=0,
            layer="below",
            row=row,
            col=1,
        )
        fig.update_yaxes(title_text="P(initiation)", secondary_y=False, row=row, col=1)

    # Applied to every row: with shared_xaxes only the range travels through
    # `matches`, so per-row settings like dtick have to be set on all of them.
    fig.update_xaxes(dtick=3, range=[0.5, 24.5])
    fig.update_xaxes(
        title_text=f"{'Circadian' if phase_label == 'DD' else 'Zeitgeber'} time (h)",
        row=len(states),
        col=1,
    )
    fig.update_layout(title=title or "", height=185 * len(states) + 90, bargap=0.12)
    return fig


def rebound_bar_plot(rebound_df, title="Sleep Rebound per State"):
    """
    Bar chart of sleep rebound per sleep state, with individual fly points.

    Matches Abhilash et al. 2026 Figure 4.  Single-sample t-test vs 0 is
    computed inside this function and annotated above each bar.

    Parameters
    ----------
    rebound_df : pd.DataFrame
        Output of sleep_state_metrics.compute_rebound_per_state().
        Columns: 'id', 'group', 'state', 'rebound_min'.
    title : str

    Returns
    -------
    go.Figure
    """
    from scipy import stats as _stats

    STATE_ORDER = ["standard", "short", "intermediate", "long"]
    # Paper colours, shared with every other Abhilash-figure renderer. The
    # literals that used to sit here were Plotly's default cycle, which gave
    # short sleep the paper's long-sleep blue and long sleep its activity red.
    STATE_COLORS, _ = _state_palette()

    if rebound_df.empty:
        return go.Figure().add_annotation(
            text="No rebound data available.",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
        )

    groups = sorted(rebound_df["group"].unique())
    states_present = [s for s in STATE_ORDER if s in rebound_df["state"].unique()]

    fig = go.Figure()
    annotations = []
    x_labels = []

    for g_idx, group in enumerate(groups):
        gdf = rebound_df[rebound_df["group"] == group]

        for _s_idx, state in enumerate(states_present):
            sdf = gdf[gdf["state"] == state]["rebound_min"].dropna()
            if sdf.empty:
                continue

            color = STATE_COLORS.get(state, "#333333")
            mean_val = sdf.mean()
            sem_val = sdf.std(ddof=1) / np.sqrt(len(sdf)) if len(sdf) > 1 else 0.0
            label = f"{state}<br>{group}" if len(groups) > 1 else state
            x_labels.append(label)

            # Bar
            fig.add_trace(
                go.Bar(
                    x=[label],
                    y=[mean_val],
                    error_y=dict(type="data", array=[sem_val], visible=True),
                    marker_color=color,
                    name=state,
                    showlegend=(g_idx == 0),
                    legendgroup=state,
                )
            )

            # Individual fly points (no horizontal jitter: x is the categorical group
            # label, so spreading points would require mapping labels to numeric x).
            fig.add_trace(
                go.Scatter(
                    x=[label] * len(sdf),
                    y=sdf.values,
                    mode="markers",
                    marker=dict(color="black", size=4, opacity=0.5),
                    showlegend=False,
                )
            )

            # t-test annotation
            if len(sdf) >= 2:
                t_stat, p_val = _stats.ttest_1samp(sdf, popmean=0.0, nan_policy="omit")
                if p_val < 0.001:
                    p_str = "p<0.001"
                elif p_val < 0.01:
                    p_str = f"p={p_val:.3f}"
                else:
                    p_str = f"p={p_val:.2f}"
                annotations.append(
                    dict(
                        x=label,
                        y=mean_val + sem_val + 1,
                        text=p_str,
                        showarrow=False,
                        font=dict(size=10),
                    )
                )

    # Zero reference line
    fig.add_hline(y=0, line_dash="dash", line_color="black", line_width=1)
    fig.update_layout(
        title=title,
        yaxis_title="Sleep rebound (minutes)",
        xaxis_title="Sleep state",
        barmode="group",
        annotations=annotations,
    )
    # Each label stacks the state over the group, so it is the GROUP half that
    # runs long here.
    apply_category_ticks(fig, x_labels)
    return fig


def period_amplitude_plot(
    per_fly_power_dict,
    period_axis,
    n_bootstrap=1000,
    ultradian_band=None,
    title="Period vs. amplitude",
):
    """Time-averaged spectrum per state with bootstrap CI — Figure 5B.

    "Time-averaged period versus amplitude plots were generated to capture the
    overall contribution of specific period bands to the overall raw timeseries
    of the various sleep states. A 95% confidence interval for the
    time-averaged amplitude of different period components was generated by
    using the bootstrap method, using 1000 replications." The resample is over
    FLIES, which is what makes non-overlapping bands a between-state claim:
    "any amplitude values that have non-overlapping error regions may be
    considered statistically significantly different from each other".

    ``n_bootstrap`` defaults to the paper's 1000 rather than the 2000 used
    before, and the period axis is logarithmic with the paper's tick values,
    dashed 12-h and 24-h references, and the ultradian band marked.

    Parameters
    ----------
    per_fly_power_dict : dict
        state -> (n_flies, n_periods) normalised power per fly.
    period_axis : np.ndarray or dict
    ultradian_band : tuple or dict or None
        (min, max) hours to annotate, per state if a dict. The paper uses
        1-4 h for short and intermediate sleep and 2-6 h for long sleep.
    """
    from plotly.subplots import make_subplots

    colors, labels = _state_palette()
    states = [s for s in per_fly_power_dict if per_fly_power_dict[s] is not None]
    if not states:
        return _empty("No period-amplitude data available.")

    rng = np.random.default_rng(42)
    fig = make_subplots(
        rows=len(states),
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=[labels.get(s, s) for s in states],
    )
    # Make the x axes logarithmic BEFORE any add_vline / add_vrect below.
    # add_vline converts its x to log10 only when the axis is ALREADY log at
    # call time; adding the 12-h and 24-h markers first and switching the axis
    # afterwards leaves the raw 24 stored as a log coordinate, so the axis
    # autoranges out to 10^24 and every curve collapses against the left edge.
    fig.update_xaxes(type="log")

    pmin, pmax = np.inf, -np.inf
    for row, state in enumerate(states, start=1):
        mat = np.asarray(per_fly_power_dict[state], dtype=float)
        if mat.ndim == 1:
            mat = mat[np.newaxis, :]
        periods = period_axis[state] if isinstance(period_axis, dict) else period_axis
        periods = np.asarray(periods, dtype=float)
        pmin, pmax = min(pmin, periods.min()), max(pmax, periods.max())

        good = ~np.all(~np.isfinite(mat), axis=1)
        mat = mat[good]
        if mat.size == 0:
            continue
        mean = np.nanmean(mat, axis=0)

        n_flies = mat.shape[0]
        if n_flies > 1:
            boot = np.empty((n_bootstrap, mat.shape[1]))
            for b in range(n_bootstrap):
                boot[b] = np.nanmean(mat[rng.integers(0, n_flies, n_flies)], axis=0)
            lo = np.nanpercentile(boot, 2.5, axis=0)
            hi = np.nanpercentile(boot, 97.5, axis=0)
        else:
            lo = hi = mean

        color = colors.get(state, "#333333")
        fig.add_trace(
            go.Scatter(
                x=np.concatenate([periods, periods[::-1]]),
                y=np.concatenate([hi, lo[::-1]]),
                fill="toself",
                fillcolor=_rgba(color, 0.22),
                line=dict(color="rgba(0,0,0,0)"),
                showlegend=False,
                hoverinfo="skip",
            ),
            row=row,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=periods,
                y=mean,
                mode="lines",
                line=dict(color=color, width=2),
                name=labels.get(state, state),
                hovertemplate="%{x:.2f} h<br>%{y:.3f}<extra></extra>",
            ),
            row=row,
            col=1,
        )

        # On a log axis, SHAPES take data units (Plotly converts them when it
        # renders) but ANNOTATIONS take log10 units and are never converted.
        # Letting add_vline/add_vrect attach their own labels therefore parks an
        # annotation at x = 24, which autoranges as 10^24 and squashes every
        # curve into the left edge of the panel. So the lines are added with raw
        # periods and the labels separately, in log10.
        for ref in (12, 24):
            if periods.min() <= ref <= periods.max():
                fig.add_vline(
                    x=ref,
                    line=dict(color="rgba(120,120,120,0.7)", width=1, dash="dash"),
                    row=row,
                    col=1,
                )
                fig.add_annotation(
                    x=float(np.log10(ref)),
                    y=1.0,
                    yref="y domain",
                    text=f"{ref}-h",
                    showarrow=False,
                    font=dict(size=9, color="#666666"),
                    xanchor="left",
                    yanchor="top",
                    row=row,
                    col=1,
                )

        band = ultradian_band[state] if isinstance(ultradian_band, dict) else ultradian_band
        if band:
            fig.add_vrect(
                x0=band[0],
                x1=band[1],
                fillcolor="rgba(220,40,40,0.07)",
                line_width=0,
                row=row,
                col=1,
            )
            fig.add_annotation(
                x=float(np.log10(np.sqrt(band[0] * band[1]))),
                y=0.04,
                yref="y domain",
                text="ultradian",
                showarrow=False,
                font=dict(size=9, color="#B03030"),
                row=row,
                col=1,
            )
        fig.update_yaxes(title_text="Norm. amplitude", row=row, col=1)

    ticks = _period_ticks(pmin, pmax)
    # Ticks go on EVERY row: shared_xaxes links the rows through `matches`,
    # which carries the range but not per-axis settings like tickvals.
    fig.update_xaxes(
        tickmode="array",
        tickvals=ticks,
        ticktext=[str(t) for t in ticks],
    )
    fig.update_xaxes(title_text="Period (h; log scale)", row=len(states), col=1)
    fig.update_layout(title=title, height=190 * len(states) + 80, showlegend=False)
    return fig


_GROUP_PALETTE = [
    "#4477AA",
    "#EE6677",
    "#228833",
    "#CCBB44",
    "#66CCEE",
    "#AA3377",
    "#BBBBBB",
    "#000000",
    "#EE7733",
    "#009988",
]


def group_spectrum_plot(
    per_group_curves,
    x_axis,
    *,
    xlabel="Period (hours)",
    ylabel="Normalised power",
    title="",
    x_log=True,
    y_log=False,
    ref_period=24.0,
    zero_line=False,
    group_order=None,
    show_individual=False,
    return_data=False,
):
    """Overlay one group-averaged curve (mean ± SEM band) per group on a shared x-axis.

    Generic over the four period methods so each gets its OWN honest axes (the
    y-scales are not comparable across methods): LS / CWT / MESA feed *power vs
    period* (log x, a 24 h reference line); AC feeds *autocorrelation vs lag*
    (linear x, a horizontal zero line). The page does the per-algorithm data prep
    (frequency→period conversion, per-fly normalisation); this function only
    averages across flies and draws.

    Parameters
    ----------
    per_group_curves : dict {group_label: (n_flies, n_points) ndarray}
        Per-fly curves already resampled onto ``x_axis``. NaN entries (e.g. AC
        lags beyond a fly's record) are excluded point-wise, so the mean/SEM at
        each x use only the flies with real data there (§2a — missing ≠ 0).
    x_axis : ndarray (n_points,)
        Shared x values for every group's curve.
    xlabel, ylabel, title : str
    x_log : bool
        Log-scale x (period methods) vs linear (AC lag).
    y_log : bool
        Log-scale y. When True, the SEM band's lower edge is floored at a small
        positive epsilon (log of zero/negative is undefined) instead of going
        negative — cosmetic only, does not affect the plotted mean.
    ref_period : float or None
        Draw a vertical dashed reference line at this x (e.g. 24 h) when in range.
    zero_line : bool
        Draw a horizontal dashed line at y=0 (AC correlogram).
    group_order : list or None
        Explicit group ordering / colour assignment; defaults to sorted keys.
    show_individual : bool
        If True, draw each fly's own curve as a faint line under the group
        mean (same colour/legendgroup as its group, so a legend click hides
        the individual lines along with the mean and its band).

    Returns
    -------
    go.Figure
    """
    import warnings

    x_axis = np.asarray(x_axis, dtype=float)
    fig = go.Figure()

    if not per_group_curves or len(x_axis) == 0:
        fig.add_annotation(
            text="No spectrum data available.",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
        )
        return (fig, None) if return_data else fig

    groups = group_order if group_order is not None else sorted(per_group_curves)
    export_cols = {xlabel: x_axis}  # for the save-to-working-folder export

    for gi, gname in enumerate(groups):
        mat = per_group_curves.get(gname)
        if mat is None:
            continue
        mat = np.asarray(mat, dtype=float)
        if mat.ndim == 1:
            mat = mat[np.newaxis, :]
        if mat.shape[0] == 0 or mat.shape[1] != len(x_axis):
            continue

        # point-wise mean / SEM over flies, ignoring NaN (all-NaN columns → NaN,
        # suppressed; columns with <2 finite flies get SEM 0).
        n_valid = np.sum(np.isfinite(mat), axis=0)
        with np.errstate(invalid="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            mean = np.nanmean(mat, axis=0)
            sd = np.nanstd(mat, axis=0, ddof=1)
        sem = np.where(n_valid > 1, sd / np.sqrt(np.maximum(n_valid, 1)), 0.0)
        export_cols[f"{gname} mean (n={mat.shape[0]})"] = mean
        export_cols[f"{gname} sem"] = sem

        color = _GROUP_PALETTE[gi % len(_GROUP_PALETTE)]
        r, g_c, b_c = tuple(int(color.lstrip("#")[i : i + 2], 16) for i in (0, 2, 4))
        n_flies = mat.shape[0]

        lg = str(gname)

        # Individual fly lines, faint, drawn first so the bold mean sits on top.
        # Same legendgroup + showlegend=False: a legend click hides these together
        # with the mean and its band, not as a separately-toggleable item.
        if show_individual:
            for fly_row in mat:
                fly_valid = np.isfinite(fly_row)
                if not np.any(fly_valid):
                    continue
                fig.add_trace(
                    go.Scatter(
                        x=x_axis[fly_valid],
                        y=fly_row[fly_valid],
                        mode="lines",
                        line=dict(color=f"rgba({r},{g_c},{b_c},0.25)", width=1),
                        legendgroup=lg,
                        showlegend=False,
                        hoverinfo="skip",
                    )
                )

        fig.add_trace(
            go.Scatter(
                x=x_axis,
                y=mean,
                mode="lines",
                name=f"{gname} (n={n_flies})",
                legendgroup=lg,
                line=dict(color=color, width=2),
            )
        )
        # ± SEM band (only where the mean is finite). SAME legendgroup as the mean
        # line so a legend click toggles BOTH the mean and its SEM band together.
        valid = np.isfinite(mean) & np.isfinite(sem)
        if np.any(valid):
            xv = x_axis[valid]
            hi = (mean + sem)[valid]
            lo = (mean - sem)[valid]
            if y_log:
                lo = np.maximum(lo, 1e-6)
            fig.add_trace(
                go.Scatter(
                    x=np.concatenate([xv, xv[::-1]]),
                    y=np.concatenate([hi, lo[::-1]]),
                    fill="toself",
                    fillcolor=f"rgba({r},{g_c},{b_c},0.18)",
                    line=dict(color="rgba(0,0,0,0)"),
                    legendgroup=lg,
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

    finite_x = x_axis[np.isfinite(x_axis)]
    if x_log:
        x_lo = max(float(finite_x.min()) * 0.95, 0.1) if len(finite_x) else 0.1
        x_hi = float(finite_x.max()) * 1.05 if len(finite_x) else 48.0
        fig.update_xaxes(type="log", title=xlabel, range=[np.log10(x_lo), np.log10(x_hi)])
    else:
        fig.update_xaxes(title=xlabel)

    if zero_line:
        fig.add_hline(y=0.0, line_dash="dash", line_color="grey")
    if ref_period is not None and len(finite_x) and finite_x.min() <= ref_period <= finite_x.max():
        fig.add_vline(
            x=ref_period,
            line_dash="dot",
            line_color="grey",
            annotation_text=f"{ref_period:g}h",
            annotation_position="top right",
        )

    # Publication styling: transparent background + explicit BLACK text so the
    # "Download plot as PNG" export is a clear-background, legible figure (not the
    # theme's white text, invisible on a clear/light export). The Periodograms page
    # renders these with theme=None so this styling — not Streamlit's dark theme —
    # drives both the on-screen chart and the PNG.
    _BLACK = "black"
    fig.update_xaxes(title_font=dict(color=_BLACK), tickfont=dict(color=_BLACK), linecolor=_BLACK)
    fig.update_yaxes(
        title=ylabel,
        title_font=dict(color=_BLACK),
        tickfont=dict(color=_BLACK),
        linecolor=_BLACK,
        type="log" if y_log else "linear",
    )
    fig.update_layout(
        title=title,
        hovermode="x unified",
        legend=dict(title="Group", font=dict(color=_BLACK), title_font=dict(color=_BLACK)),
        font=dict(color=_BLACK),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    if return_data:
        return fig, pd.DataFrame(export_cols)
    return fig


def sleep_state_scalogram(
    avg_surface_dict,
    period_axis,
    bin_size_min=5,
    phase_label="DD",
    zmax=SCALOGRAM_ZMAX,
    title="Normalised average scalograms",
):
    """Group-averaged normalised scalograms, one row per state — Figure 5A.

    Four changes from the previous version, all of them about being comparable
    to the printed figure rather than merely plausible:

    - **Fixed z range 0 to 1.5** instead of the 98th percentile of whatever
      happened to be in the data. The paper states "All scalograms have a z
      axis scale that ranges from 0 to 1.5"; a data-dependent range rescales
      every panel differently, so two states — or our panel and the paper's —
      cannot be compared by colour at all.
    - **Jet colour ramp** instead of Viridis, matching the printed blue-to-red
      ramp. (Viridis is the better colourmap in general; here the whole point
      is to read our output against theirs.)
    - **x axis in days**, not "5-min bins", with the subjective day/night bar
      and transition markers above each panel.
    - **Stacked rows** with a shared x axis, the paper's arrangement, so the
      same time on the clock lines up vertically across states.

    Parameters
    ----------
    avg_surface_dict : dict
        state -> (n_periods, n_timepoints) group-averaged normalised power.
    period_axis : np.ndarray or dict
        Period values in hours; a dict is keyed like ``avg_surface_dict`` for
        states whose CWT ran on a different grid.
    bin_size_min : int
        Time-bin width of the surfaces, used to convert columns to days.
    zmax : float
        Upper end of the colour range. Keep at 1.5 to match the paper.
    """
    from plotly.subplots import make_subplots

    colors, labels = _state_palette()
    states = list(avg_surface_dict.keys())
    if not states:
        return _empty("No scalogram data available.")

    fig = make_subplots(
        rows=len(states),
        cols=1,
        shared_xaxes=True,
        # Roomier than the other stacked figures: each panel carries a
        # light/dark bar above it AND a title above that, and 0.055 left the
        # title sitting on the panel above.
        vertical_spacing=0.09,
        subplot_titles=[labels.get(s, s) for s in states],
    )

    for row, state in enumerate(states, start=1):
        surface = np.asarray(avg_surface_dict[state])
        periods = period_axis[state] if isinstance(period_axis, dict) else period_axis
        periods = np.asarray(periods)
        n_t = surface.shape[1]
        days = np.arange(n_t) * bin_size_min / 60.0 / 24.0

        fig.add_trace(
            go.Heatmap(
                z=surface,
                x=days,
                y=periods,
                colorscale=SCALOGRAM_COLORSCALE,
                zmin=0.0,
                zmax=zmax,
                zsmooth="best",
                showscale=(row == 1),
                colorbar=dict(
                    title="Normalised<br>amplitude",
                    len=0.9 / len(states),
                    y=1.0,
                    yanchor="top",
                )
                if row == 1
                else None,
                hovertemplate="day %{x:.2f}<br>period %{y:.2f} h<br>%{z:.2f}<extra></extra>",
            ),
            row=row,
            col=1,
        )
        ticks = _period_ticks(float(periods.min()), float(periods.max()))
        fig.update_yaxes(
            type="log",
            tickmode="array",
            tickvals=ticks,
            ticktext=[str(t) for t in ticks],
            title_text="Period (h)" if row == (len(states) + 1) // 2 else None,
            row=row,
            col=1,
        )
        _day_night_bar(fig, days[-1] if n_t else 1, row, 1, phase_label)

    fig.update_xaxes(
        title_text=f"Days since start of {'constant darkness' if phase_label == 'DD' else 'the light cycle'}",
        row=len(states),
        col=1,
    )
    fig.update_layout(title=title, height=250 * len(states) + 110)
    # Each panel carries a light/dark bar immediately above it, which the
    # default subplot-title position overprints.
    for note in fig.layout.annotations:
        note.update(yshift=26)
    return fig


def single_fly_scalogram_plotly(scalogram_data, fly_id=""):
    """
    Interactive Plotly scalogram for a single fly's CWT results.

    Parameters
    ----------
    scalogram_data : dict
        Output of periodograms.compute_single_fly_scalogram(). Expected keys:
        'power', 'periods_hours', 'activity_processed', 'smoothed_periods'.
    fly_id : str
        Fly identifier for the plot title.

    Returns
    -------
    go.Figure
    """
    from plotly.subplots import make_subplots

    if scalogram_data is None:
        return go.Figure().add_annotation(
            text="No scalogram data available for this fly.",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
        )

    power = scalogram_data.get("power")
    periods = scalogram_data.get("periods_hours")
    activity = scalogram_data.get("activity_processed")
    ridge = scalogram_data.get("smoothed_periods")

    if power is None or periods is None:
        return go.Figure().add_annotation(
            text="Incomplete scalogram data.",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
        )

    n_t = power.shape[1]
    t = np.arange(n_t)

    fig = make_subplots(
        rows=2,
        cols=1,
        row_heights=[0.25, 0.75],
        shared_xaxes=True,
        subplot_titles=[f"Activity — {fly_id}", "Scalogram"],
    )

    # Activity trace
    if activity is not None:
        fig.add_trace(
            go.Scatter(
                x=t,
                y=activity,
                mode="lines",
                line=dict(color="steelblue", width=1),
                name="Activity",
            ),
            row=1,
            col=1,
        )

    # Scalogram heatmap
    all_vals = power.ravel()
    zmax = float(np.nanpercentile(all_vals, 98))
    fig.add_trace(
        go.Heatmap(
            z=power,
            x=t,
            y=periods,
            colorscale="Viridis",
            zmin=0,
            zmax=zmax,
            colorbar=dict(title="Power", y=0.3, len=0.5),
        ),
        row=2,
        col=1,
    )

    # Ridge overlay
    if ridge is not None:
        fig.add_trace(
            go.Scatter(
                x=t,
                y=ridge,
                mode="lines",
                line=dict(color="white", width=1.5, dash="dot"),
                name="Ridge period",
            ),
            row=2,
            col=1,
        )

    # 24h reference line
    fig.add_hline(y=24, line_dash="dash", line_color="orange", annotation_text="24h", row=2, col=1)

    # Linear y-axis with integer-hour ticks so the chronobiology-relevant
    # range reads naturally (16, 18, …, 32) instead of `3 × 10¹` etc.
    # Plotly's Heatmap places each row at its actual period value, so a
    # linear axis is faithful even though the period grid is log-spaced.
    if periods is not None and len(periods):
        _ticks = _circadian_period_ticks(float(np.min(periods)), float(np.max(periods)))
        fig.update_yaxes(
            type="linear",
            title_text="Period (hours)",
            tickmode="array",
            tickvals=_ticks,
            ticktext=[str(t) for t in _ticks],
            row=2,
            col=1,
        )
    else:
        fig.update_yaxes(type="linear", title_text="Period (hours)", row=2, col=1)
    fig.update_xaxes(title_text="Time (minutes)", row=2, col=1)
    fig.update_layout(title=f"CWT Scalogram — {fly_id}", height=600)
    return fig


def save_group_average_scalogram_png(
    power_2d: np.ndarray,
    periods_h: np.ndarray,
    time_h: np.ndarray,
    group_label: str,
    n_flies: int,
    out_png_path: str,
    dpi: int = 150,
    period_range=None,
    vmax_percentile: float = 98.0,
    phase_label: str | None = None,
) -> str:
    """
    Render a per-group averaged CWT periodogram (period × time × power)
    to disk as a PNG via matplotlib.

    Why disk-PNG instead of a plotly heatmap: plotly figures get serialized
    as JSON containing the full underlying matrix when shipped to the
    browser via ``st.plotly_chart``. For a typical CWT (~100 periods ×
    ~3000 timepoints × multiple groups) this blows past Streamlit's
    ~200 MB browser-payload ceiling. Saving as PNG and displaying via
    ``st.image`` ships pixel bytes only, dropping payload by ~100×, and
    incidentally produces a persistable scientific artifact.

    Parameters
    ----------
    power_2d : ndarray, shape (n_periods, n_time)
        Group-averaged CWT power matrix.
    periods_h : ndarray, shape (n_periods,)
        Period axis in hours (must match ``power_2d`` row order).
    time_h : ndarray, shape (n_time,)
        Time axis in hours (must match ``power_2d`` column order).
    group_label : str
        Group identifier used in the figure title.
    n_flies : int
        Number of flies contributing to the mean (rendered in the title).
    out_png_path : str
        Destination filesystem path.
    dpi : int, default 150
        Output resolution. 150 dpi balances clarity with file size; do
        not exceed 200 dpi for the 4-panel screen view.
    period_range : (lo, hi) tuple of float in hours, optional
        Y-axis range. Defaults to the periods array extremes.
    vmax_percentile : float, default 98.0
        Color-scale upper bound percentile (clips long-tail outliers
        from compressing the mid-band signal).
    phase_label : str or None, default None
        Active phase label (e.g. ``'DD'``, ``'LD'``, ``'full'``). If
        provided, included in the title.

    Returns
    -------
    str
        ``out_png_path`` (for chaining / logging).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    finite = power_2d[np.isfinite(power_2d)]
    vmax = float(np.nanpercentile(finite, vmax_percentile)) if finite.size else 1.0
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0

    if period_range is None and periods_h.size:
        period_range = (float(periods_h.min()), float(periods_h.max()))

    # Period grid is log-spaced (CWT scales = 2^(j/n_voices)). Use
    # `pcolormesh` with explicit cell edges so non-uniform row heights
    # render correctly on a *linear* y-axis. `imshow` would assume
    # uniform spacing and visually distort. Linear-with-integer-ticks
    # is the chronobiology-friendly default for the 16–32h band.
    def _edges_from_centers(c):
        c = np.asarray(c, dtype=float)
        if c.size == 0:
            return c
        if c.size == 1:
            half = 0.5
            return np.array([c[0] - half, c[0] + half])
        edges = np.empty(c.size + 1, dtype=float)
        edges[1:-1] = 0.5 * (c[:-1] + c[1:])
        edges[0] = c[0] - 0.5 * (c[1] - c[0])
        edges[-1] = c[-1] + 0.5 * (c[-1] - c[-2])
        return edges

    period_edges = _edges_from_centers(periods_h)
    time_edges = _edges_from_centers(time_h)

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=dpi)
    im = ax.pcolormesh(
        time_edges,
        period_edges,
        power_2d,
        cmap="viridis",
        vmin=0.0,
        vmax=vmax,
        shading="auto",
    )
    if period_range is not None:
        ax.set_ylim(period_range[0], period_range[1])
        ticks = _circadian_period_ticks(period_range[0], period_range[1])
        if ticks:
            ax.set_yticks(ticks)
            ax.set_yticklabels([str(t) for t in ticks])
    ax.axhline(24.0, linestyle="--", color="orange", linewidth=1.0, label="24h")
    ax.set_xlabel("Time (h)")
    ax.set_ylabel("Period (h)")
    title_bits = [f"Group-averaged CWT periodogram — {group_label}", f"n = {n_flies}"]
    if phase_label:
        title_bits.append(f"phase = {phase_label}")
    ax.set_title(" | ".join(title_bits))
    ax.legend(loc="upper right", fontsize=8, framealpha=0.85)
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("Mean power")
    fig.tight_layout()
    fig.savefig(out_png_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_png_path


def group_ridge_density_plotly(
    ds,
    group_col="genotype",
    group_val=None,
    temperature=None,
    period_range=None,
):
    """
    Plotly 2D histogram of CWT ridge periods across all flies in a group.

    Parameters
    ----------
    ds : xr.Dataset
        Must have cwt_ridge_periods variable.
    group_col : str
        Column used to group flies.
    group_val : str, optional
        Value to filter on.
    temperature : str, optional
        Temperature filter.
    period_range : tuple of (lo, hi) float in hours, optional
        Y-axis range (period). When ``None`` (default), pulled from the
        active CWT analysis window via ``ds.attrs['cwt_min_period']`` /
        ``cwt_max_period``; falls back to ``(18, 30)`` if those attrs are
        missing. Default-from-attrs is preferred so the y-range tracks
        the user's chosen analysis range across reruns (Phase 2D).
    Returns
    -------
    go.Figure

    Notes
    -----
    **Pass an already-filtered dataset.** This used to take
    ``filter_nonrhythmic`` / ``filter_gate`` and defer-import
    ``rhythmicity_classification.apply_rhythmic_filter`` to do the filtering
    itself — a deferred import whose only purpose was dodging a circular one,
    which is the signal that the compute/render seam is in the wrong place.
    Arrhythmic flies still need excluding (off-rhythm ridge tracks are noise and
    distort the group density); that is now the caller's call, made explicitly::

        ds = apply_rhythmic_filter(ds, gate="ac", enabled=True)
        fig = group_ridge_density_plotly(ds, group_val="w1118")

    This function currently has NO callers — it is one of the unwired plotting
    functions listed under backlog item 6, which will decide whether the whole
    Abhilash sleep-state section gets a UI or gets deleted. The signature change
    therefore has no blast radius, and is here because item 7's requirement is
    that ``core/plotting.py`` carry no deferred analysis imports at all.
    """
    if "cwt_ridge_periods" not in ds.data_vars:
        return go.Figure().add_annotation(
            text="CWT ridge data not found. Run CWT analysis first.",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
        )

    if period_range is None:
        period_range = (
            float(ds.attrs.get("cwt_min_period", 18.0)),
            float(ds.attrs.get("cwt_max_period", 30.0)),
        )

    # Filter flies
    fly_ids = ds["id"].values
    if group_val is not None and group_col in ds.coords:
        mask = ds[group_col].values == group_val
        fly_ids = ds["id"].values[mask]

    all_periods = []
    all_times = []

    for fly_id in fly_ids:
        ridge = ds["cwt_ridge_periods"].sel(id=fly_id).values
        valid = ~np.isnan(ridge)
        all_periods.extend(ridge[valid].tolist())
        all_times.extend(np.where(valid)[0].tolist())

    if not all_periods:
        return go.Figure().add_annotation(
            text="No ridge data for the selected group.",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
        )

    fig = go.Figure(
        go.Histogram2d(
            x=all_times,
            y=all_periods,
            colorscale="Viridis",
            nbinsx=50,
            nbinsy=40,
            colorbar=dict(title="Count"),
        )
    )
    fig.add_hline(y=24, line_dash="dash", line_color="orange", annotation_text="24h")
    group_label = f"{group_col}={group_val}" if group_val else "All flies"
    fig.update_layout(
        title=f"CWT Ridge Period Density — {group_label}",
        xaxis_title="Time (minutes)",
        yaxis_title="Ridge period (hours)",
        yaxis=dict(range=list(period_range)),
    )
    return fig


def rose_plot(
    profile_stats,
    state,
    bin_size_min=30,
    phase_label="DD",
    title=None,
    normalise=True,
):
    """Rose plot of one state's daily profile, as in Abhilash et al. 2026 Fig 3A/B.

    **This function used to plot the wrong quantity.** It binned sleep-bout
    INITIATION times and drew the fraction of bouts starting in each bin. The
    paper's rose plots are profiles: "We calculated sleep time series, binned
    at 30-minute intervals... These were then averaged over days and across
    flies. The resulting average activity or sleep profiles were plotted as
    rose plots." The authors' own implementation confirms it —
    ``phase::rosePlotsSleep`` is handed ``binnedDataInput()``, the binned time
    series, and never sees a bout table. Bout initiation is a different figure
    entirely (Figure 2's Cartesian bar panels, see
    :func:`initiation_probability_plot`), so a rose plot built from it looked
    superficially similar and could never match the paper.

    Parameters
    ----------
    profile_stats : pd.DataFrame
        From ``sleep_state_metrics.group_profiles`` — needs ``state``,
        ``zt_bin_minute`` and ``mean``, already filtered to one group.
    state : str
        Which state's profile to draw.
    bin_size_min : int
        Must match the binning used to build ``profile_stats``. Sets the wedge
        width, so a mismatch silently draws over- or under-wide wedges.
    phase_label : {'DD', 'LD', 'ramp'}
        Chooses the day/night background shading.
    normalise : bool
        Scale the profile so its peak touches the outer circle. The paper does
        this implicitly by giving each panel its own radial range: sleep
        (min/h) and activity (counts/h) have no common unit, so overlaying
        them raw would make whichever has the larger numbers swamp the other.

    Returns
    -------
    go.Figure
    """
    colors, labels = _state_palette()
    if profile_stats is None or profile_stats.empty:
        return _empty("No profile data available.")
    sdf = profile_stats[profile_stats["state"] == state].sort_values("zt_bin_minute")
    if sdf.empty:
        return _empty(f"No profile data for {state!r}.")

    fig = go.Figure()
    _add_rose_series(fig, sdf, state, bin_size_min, normalise, colors)
    for start, end, fill in _day_night_wedges(phase_label):
        _add_background_wedge(fig, start, end, fill)

    fig.update_layout(
        title=title or labels.get(state, state),
        polar=_polar_layout("CT" if phase_label == "DD" else "ZT"),
        showlegend=False,
    )
    return fig


def rose_plot_with_activity(
    profile_stats,
    group="",
    bin_size_min=30,
    phase_label="DD",
    states=("standard", "short", "intermediate", "long"),
    title=None,
):
    """The full Figure 3A/B row: activity alone, then each state with activity overlaid.

    Five panels in the paper's order and colours — locomotor activity (red),
    then standard (grey), short (orange), intermediate (green) and long (blue)
    sleep, each with the activity profile drawn over it in translucent red "to
    facilitate visualization of their temporal inter-relationships".

    The previous version drew four panels with activity in grey and the states
    in Plotly's default cycle, which put short sleep in the paper's long-sleep
    blue and long sleep in the paper's activity red.

    Each series is scaled to its own maximum, so a panel shows the SHAPE and
    relative timing of the two profiles rather than their absolute magnitudes;
    activity in counts/h and sleep in min/h share no unit. The paper notes
    "the activity counts remain the same within each column", which is what
    per-series scaling reproduces.

    Parameters
    ----------
    profile_stats : pd.DataFrame
        ``sleep_state_metrics.group_profiles`` output for ONE group, including
        the ``'activity'`` state.
    """
    from plotly.subplots import make_subplots

    colors, labels = _state_palette()
    if profile_stats is None or profile_stats.empty:
        return _empty("No profile data available.")

    present = [s for s in states if s in set(profile_stats["state"])]
    if not present:
        return _empty("No sleep-state profiles available.")
    has_activity = "activity" in set(profile_stats["state"])
    panels = (["activity"] if has_activity else []) + present

    fig = make_subplots(
        rows=1,
        cols=len(panels),
        specs=[[{"type": "polar"}] * len(panels)],
        subplot_titles=[labels.get(p, p) for p in panels],
        horizontal_spacing=0.02,
    )

    act = profile_stats[profile_stats["state"] == "activity"].sort_values("zt_bin_minute")
    shading = _day_night_wedges(phase_label)

    for col, panel in enumerate(panels, start=1):
        sub = go.Figure()
        if panel != "activity" and has_activity:
            # Activity UNDER the state so the state stays readable, matching
            # the printed panels where the solid state colour reads on top.
            _add_rose_series(sub, act, "activity", bin_size_min, True, colors, opacity=0.45)
        sdf = profile_stats[profile_stats["state"] == panel].sort_values("zt_bin_minute")
        _add_rose_series(sub, sdf, panel, bin_size_min, True, colors, opacity=0.9)
        for start, end, fill in shading:
            _add_background_wedge(sub, start, end, fill)
        for trace in sub.data:
            fig.add_trace(trace, row=1, col=col)

    prefix = "CT" if phase_label == "DD" else "ZT"
    for i in range(len(panels)):
        hours = [0, 12]
        if i == 0:
            hours.append(18)
        if i == len(panels) - 1:
            hours.append(6)
        fig.update_layout(
            **{
                f"polar{i + 1}" if i else "polar": _polar_layout(
                    prefix, label_hours=hours
                )
            }
        )

    fig.update_layout(
        # "" not None: a None title serialises to an empty title OBJECT, whose
        # `text` is undefined, and plotly.js renders that as the word
        # "undefined" where the title belongs.
        title=title or (f"Temporal organisation of sleep states — {group}" if group else ""),
        showlegend=False,
        height=380,
        margin=dict(t=110, b=30),
    )
    # Lift the panel titles clear of the CT00/ZT00 tick, which sits at the top
    # of each ring and otherwise overprints them.
    for note in fig.layout.annotations:
        note.update(yshift=16, font=dict(size=12))
    return fig


def polar_gating_plot(
    stats_df,
    gates_df=None,
    phase_label="DD",
    states=("activity", "short", "intermediate", "long"),
    title="Circadian gating of sleep states",
):
    """Concentric per-fly gate arcs with the mean gate outermost — Figure 3C.

    Each fly contributes one arc per state spanning its circadian gate, drawn
    at its own radius so the arcs nest instead of piling up; the outermost ring
    carries the thick group-mean arcs. Radii are assigned fly-by-fly so the
    colours interleave the way the printed figure's rings do.

    Three things were wrong before:

    - **Every per-fly arc was drawn at r = 1.** The paper's "each concentric
      ring shows the gate of each sleep/wake state for each fly" became one
      overlapping ring, so nothing about the per-fly spread was legible.
    - **Gates came from bout onset/offset.** The paper derives them from
      circular dispersion instead: the centre of mass gives a mean phase, and
      angular deviation about it stands in for gate width, because there is no
      objective phase marker for the start of a sleep state. See
      ``sleep_state_metrics.circular_state_stats``.
    - **Arcs that crossed the origin were drawn backwards.** A ``linspace``
      from onset to offset in degrees traverses the long way round whenever
      offset < onset, i.e. for exactly the states whose gate straddles
      midnight. Here the offset is unwrapped forward past the onset first.

    Parameters
    ----------
    stats_df : pd.DataFrame
        ``circular_state_stats`` output (one row per fly and state), already
        filtered to one group.
    gates_df : pd.DataFrame or None
        ``group_gates`` output for the same group. Omit to skip the mean ring.
    """
    colors, labels = _state_palette()
    if stats_df is None or stats_df.empty:
        return _empty("No circular statistics available.")

    present = [s for s in states if s in set(stats_df["state"])]
    if not present:
        return _empty("No gates available for the requested states.")

    flies = list(dict.fromkeys(stats_df["id"].tolist()))
    n_rings = max(1, len(flies) * len(present))
    inner, outer = 0.45, 0.93  # leave the middle clear for the CT/ZT labels

    fig = go.Figure()
    ring = 0
    for fly in flies:
        for state in present:
            row = stats_df[(stats_df["id"] == fly) & (stats_df["state"] == state)]
            ring += 1
            if row.empty:
                continue
            onset, offset = float(row["onset_h"].iloc[0]), float(row["offset_h"].iloc[0])
            if not (np.isfinite(onset) and np.isfinite(offset)):
                continue
            radius = inner + (outer - inner) * (ring / n_rings)
            fig.add_trace(_gate_arc(onset, offset, radius, colors[state], width=1.4, alpha=0.30))

    if gates_df is not None and not gates_df.empty:
        for state in present:
            row = gates_df[gates_df["state"] == state]
            if row.empty:
                continue
            onset, offset = float(row["onset_h"].iloc[0]), float(row["offset_h"].iloc[0])
            if not (np.isfinite(onset) and np.isfinite(offset)):
                continue
            # Nudge each mean arc onto its own radius: the paper's gates are
            # allowed to overlap ("although sleep states cannot co-occur in
            # time within a fly, the circadian gates may overlap"), and one
            # shared radius would hide whichever was drawn first.
            offset_idx = present.index(state)
            radius = 1.0 + 0.055 * offset_idx
            fig.add_trace(
                _gate_arc(
                    onset,
                    offset,
                    radius,
                    colors[state],
                    width=7,
                    alpha=1.0,
                    name=labels.get(state, state),
                    show_legend=True,
                )
            )

    layout = _polar_layout("", inner_hole=0.0)
    layout["radialaxis"]["range"] = [0, 1.0 + 0.055 * len(present) + 0.05]
    # Pin the polar domain rather than letting the legend squeeze it, so the
    # CT/ZT label in the middle of the rings can be placed on the actual centre
    # of the circle. With an auto domain the legend shifts the plot left and a
    # paper-space x of 0.5 lands off-centre.
    domain_x = (0.0, 0.66)
    layout["domain"] = dict(x=list(domain_x), y=[0.0, 1.0])
    fig.update_layout(
        title=title,
        polar=layout,
        showlegend=True,
        legend=dict(x=0.70, y=0.9, yanchor="top"),
        annotations=[
            dict(
                text="CT" if phase_label == "DD" else "ZT",
                x=sum(domain_x) / 2,
                y=0.5,
                xref="paper",
                yref="paper",
                showarrow=False,
                font=dict(size=13),
            )
        ],
    )
    return fig


def ultradian_amplitude_plot(
    amplitude_dict,
    bin_size_min=5,
    phase_label="DD",
    n_bootstrap=1000,
    bands=None,
    title="Ultradian amplitude over time",
):
    """Ultradian-band amplitude against time, with bootstrap CI — Figure 6A/C/E.

    "Also, shown are amplitude values over time for each sleep state... The
    error regions for each time vs. amplitude trace represent 95% confidence
    intervals estimated through bootstrapping."

    Three changes from the previous version, all about being readable against
    the printed panel: the x axis is **hours since the start of the epoch**
    (the paper's 0-216 h) rather than an unlabelled bin index, the subjective
    day/night blocks are shaded so the circadian gating the figure exists to
    show is visible, and the resample count is the paper's 1000.

    Parameters
    ----------
    amplitude_dict : dict
        state -> (n_flies, n_timepoints) ultradian-band amplitude.
    bands : dict or None
        state -> (min, max) hours, shown in each panel's subtitle so a reader
        knows which band was averaged (1-4 h for short and intermediate sleep,
        2-6 h for long sleep).
    """
    from plotly.subplots import make_subplots

    colors, labels = _state_palette()
    states = [s for s in amplitude_dict if amplitude_dict[s] is not None]
    if not states:
        return _empty("No ultradian amplitude data available.")

    rng = np.random.default_rng(42)
    subtitles = []
    for state in states:
        band = (bands or {}).get(state)
        subtitles.append(
            f"{labels.get(state, state)}"
            + (f" — {band[0]}-{band[1]} h band" if band else "")
        )

    fig = make_subplots(
        rows=len(states),
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=subtitles,
    )

    max_hours = 0.0
    for row, state in enumerate(states, start=1):
        mat = np.asarray(amplitude_dict[state], dtype=float)
        if mat.ndim == 1:
            mat = mat[np.newaxis, :]
        keep = ~np.all(~np.isfinite(mat), axis=1)
        mat = mat[keep]
        if mat.size == 0:
            continue
        hours = np.arange(mat.shape[1]) * bin_size_min / 60.0
        max_hours = max(max_hours, float(hours[-1]))
        mean = np.nanmean(mat, axis=0)

        if mat.shape[0] > 1:
            boot = np.empty((n_bootstrap, mat.shape[1]))
            for b in range(n_bootstrap):
                boot[b] = np.nanmean(mat[rng.integers(0, mat.shape[0], mat.shape[0])], axis=0)
            lo = np.nanpercentile(boot, 2.5, axis=0)
            hi = np.nanpercentile(boot, 97.5, axis=0)
        else:
            lo = hi = mean

        color = colors.get(state, "#333333")
        fig.add_trace(
            go.Scatter(
                x=np.concatenate([hours, hours[::-1]]),
                y=np.concatenate([hi, lo[::-1]]),
                fill="toself",
                fillcolor=_rgba(color, 0.22),
                line=dict(color="rgba(0,0,0,0)"),
                showlegend=False,
                hoverinfo="skip",
            ),
            row=row,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=hours,
                y=mean,
                mode="lines",
                line=dict(color=color, width=1.8),
                name=labels.get(state, state),
                showlegend=False,
                hovertemplate="%{x:.1f} h<br>%{y:.3f}<extra></extra>",
            ),
            row=row,
            col=1,
        )
        fig.update_yaxes(title_text="Norm. amplitude", rangemode="tozero", row=row, col=1)

    # Subjective day/night blocks across the whole recording, so the gating is
    # readable rather than inferred from tick positions.
    for row in range(1, len(states) + 1):
        for day in range(int(np.ceil(max_hours / 24.0))):
            fig.add_vrect(
                x0=day * 24 + 12,
                x1=min(day * 24 + 24, max_hours),
                fillcolor="rgba(0,0,0,0.11)" if phase_label == "DD" else "rgba(0,0,0,0.16)",
                line_width=0,
                layer="below",
                row=row,
                col=1,
            )

    fig.update_xaxes(dtick=24, range=[0, max_hours])
    fig.update_xaxes(
        title_text=f"Hours since start of {'constant darkness' if phase_label == 'DD' else 'the light cycle'}",
        row=len(states),
        col=1,
    )
    fig.update_layout(title=title, height=185 * len(states) + 90)
    return fig

def chi_sq_periodogram_plot(
    chi_ds,
    states=("standard", "short", "intermediate", "long"),
    title="Chi-squared periodogram of ultradian amplitude",
):
    """Per-fly adjusted chi-squared periodograms with the group mean — Figure 6B/D/F.

    "Thin lines represent the periodogram results for each fly, and the thick
    line displays the power for each period value, averaged over all the flies.
    The red horizontal line at zero marks the critical value for statistical
    significance."

    The y quantity is **adjusted** power (Qp minus the significance threshold),
    which is what puts the critical value at a flat zero instead of on a
    period-dependent curve — see
    ``periodograms.ultradian_rhythmicity_chi_sq``.

    Parameters
    ----------
    chi_ds : xr.Dataset
        Output of ``periodograms.ultradian_rhythmicity_chi_sq``.
    """
    from plotly.subplots import make_subplots

    colors, labels = _state_palette()
    present = [s for s in states if f"ultra_chisq_adjusted_{s}" in getattr(chi_ds, "data_vars", {})]
    if not present:
        return _empty("No chi-squared periodogram results available.")

    fig = make_subplots(
        rows=len(present),
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=[labels.get(s, s) for s in present],
    )

    for row, state in enumerate(present, start=1):
        da = chi_ds[f"ultra_chisq_adjusted_{state}"]
        pdim = [d for d in da.dims if d != "id"][0]
        periods = np.asarray(da[pdim].values, dtype=float)
        values = np.asarray(da.transpose("id", pdim).values, dtype=float)
        color = colors.get(state, "#333333")

        for fly_row in values:
            fig.add_trace(
                go.Scatter(
                    x=periods,
                    y=fly_row,
                    mode="lines",
                    line=dict(color=_rgba(color, 0.16), width=1),
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=row,
                col=1,
            )
        fig.add_trace(
            go.Scatter(
                x=periods,
                y=np.nanmean(values, axis=0),
                mode="lines",
                line=dict(color=color, width=2.5),
                name=labels.get(state, state),
                showlegend=False,
                hovertemplate="%{x:.2f} h<br>adj. power %{y:.1f}<extra></extra>",
            ),
            row=row,
            col=1,
        )
        fig.add_hline(
            y=0,
            line=dict(color="#D62728", width=1.4),
            row=row,
            col=1,
        )
        fig.add_vline(
            x=24,
            line=dict(color="rgba(120,120,120,0.7)", width=1, dash="dash"),
            row=row,
            col=1,
        )
        fig.update_yaxes(title_text="Adj. power", row=row, col=1)

    fig.update_xaxes(dtick=4)
    fig.update_xaxes(title_text="Period (h)", row=len(present), col=1)
    fig.update_layout(title=title, height=180 * len(present) + 80)
    return fig



_RHYTH_VIOLIN_META = {
    "ls": {
        "label": "Lomb-Scargle",
        "period_var": "ls_period",
        "metric_var": "ls_power",
        "metric_label": "LS power (R^2 strength)",
        "period_label": "LS period (h)",
        "flag_coord": "ls_rhythmic",
        "threshold_attr": "ls_power_threshold",
        "metric_lower_is_rhythmic": False,  # higher power = more rhythmic (LS's RI)
        "metric_log_y": True,
    },
    "ac": {
        "label": "Autocorrelation",
        "period_var": "ac_period",
        "metric_var": "ac_power",
        "metric_label": "RI (peak autocorr.)",
        "period_label": "AC period (h)",
        "flag_coord": "ac_rhythmic",
        "threshold_attr": "ac_ri_threshold",
        "metric_lower_is_rhythmic": False,
        "metric_log_y": False,
    },
    "cwt": {
        "label": "CWT",
        "period_var": "cwt_period",
        "metric_var": "cwt_rhythmicity",
        "metric_label": "CWT rhythmicity",
        "period_label": "CWT period (h)",
        "flag_coord": "cwt_rhythmic",
        "threshold_attr": "cwt_rhythmicity_threshold",
        "metric_lower_is_rhythmic": False,
        "metric_log_y": False,
    },
    # MESA is a PERIOD method with NO significance test, so its rhythmic call
    # BORROWS the Autocorrelation RI (`ac_power`) as the strength metric: MESA
    # supplies the period (`mesa_period`), AC supplies the rhythmicity. The MESA
    # explorer tab therefore shows the MESA period distribution of the flies AC
    # calls rhythmic, gated by the AC RI threshold (MESA has no cutoff of its
    # own). `mesa_power` (peak/median PSD) is still reported in the page-3 summary
    # table as an informational SNR, just not used to gate here. This entry is
    # consumed only by threshold_coupled_figure (rhythmicity_violin_grid is unused
    # by any page), so borrowing ac_power here does not affect other plots.
    "mesa": {
        "label": "MESA",
        "period_var": "mesa_period",
        "metric_var": "ac_power",  # borrow AC's RI (MESA has no metric)
        "metric_label": "RI (autocorr. — MESA borrows AC)",
        "period_label": "MESA period (h)",
        "flag_coord": "ac_rhythmic",
        "threshold_attr": "ac_ri_threshold",
        "metric_lower_is_rhythmic": False,
        "metric_log_y": False,
    },
}


def _get_group_coord_values(ds: xr.Dataset, group_coord: str = "group") -> np.ndarray:
    """Return a per-id array of group labels, or 'all' if no group coord."""
    if group_coord in ds.coords:
        return np.asarray(ds[group_coord].values, dtype=object)
    return np.full(ds.sizes.get("id", 0), "all", dtype=object)


def _collect_violin_data(
    ds: xr.Dataset,
    mode: str,
    algorithms: list,
    group_coord: str = "group",
    filter_nonrhythmic: bool = True,
    filter_gate: str = "ac",
) -> pd.DataFrame:
    """
    Build the long-form dataframe feeding the violin grid.

    Columns: algorithm, algorithm_label, group, fly_id, metric_name,
             value, is_rhythmic

    Parameters
    ----------
    filter_nonrhythmic : bool, default True
        If True, drop flies flagged arrhythmic by ``filter_gate`` before
        plotting *period* values. Period averages are easily biased by
        arrhythmic flies (whose period values are essentially random),
        so this gate is on by default. Has no effect in ``mode='metric'``
        (where every fly's metric is shown).
    filter_gate : {'ls', 'ac', 'cwt'}, default 'ac'
        Which classifier's flag drives the filter. AC is the canonical
        gate; LS/CWT remain plumbed for completeness but the rest of the
        app no longer uses them as filters. **If the gate's
        ``<gate>_rhythmic`` coord is missing the filter is silently
        skipped — every fly with a finite period value is plotted.**
        This intentional behavior decouples period plotting from
        classifier execution: an LS or CWT period can be plotted even
        when those classifiers were never run, as long as the underlying
        analysis produced a period value.
    """
    if mode not in ("period", "metric"):
        raise ValueError(f"mode must be 'period' or 'metric', got {mode!r}")

    groups = _get_group_coord_values(ds, group_coord)
    fly_ids = np.asarray(ds["id"].values)
    rows = []

    # Unified rhythmic gate (period mode only). Gates every algorithm's
    # period column on the same flag, so group means reflect a consistent
    # fly subset rather than a different subset per column.
    gate_flag_name = f"{filter_gate}_rhythmic"
    if mode == "period" and filter_nonrhythmic and gate_flag_name in ds.coords:
        unified_gate = np.asarray(ds[gate_flag_name].values, dtype=bool).ravel()
    else:
        unified_gate = None  # filter off OR gate flag missing -> no filter

    for algo in algorithms:
        meta = _RHYTH_VIOLIN_META[algo]
        var = meta["period_var"] if mode == "period" else meta["metric_var"]
        if var not in ds:
            continue
        vals = np.asarray(ds[var].values, dtype=float).ravel()

        # Per-algo rhythmic flag — used only to colour/annotate points in
        # 'metric' mode. Never gates plotting; the universal AC gate above
        # is the only filter (Phase 2A: AC is canonical, LS/CWT diagnostic).
        if meta["flag_coord"] in ds.coords:
            flag = np.asarray(ds[meta["flag_coord"]].values, dtype=bool).ravel()
        else:
            flag = np.full(vals.shape, False, dtype=bool)

        # For log-scale metrics (e.g. LS FAP), clamp zeros to a small
        # positive floor so they survive log transformation in Plotly.
        use_log = mode == "metric" and meta.get("metric_log_y", False)

        for i, fly_id in enumerate(fly_ids):
            if i >= len(vals):
                break
            # excluded by unified AC rhythmic filter
            if (
                mode == "period"
                and unified_gate is not None
                and (i >= len(unified_gate) or not unified_gate[i])
            ):
                continue
            v = vals[i]
            if not np.isfinite(v):
                continue
            if use_log and v == 0.0:
                v = 1e-300  # clamp exact zero for log display
            rows.append(
                {
                    "algorithm": algo,
                    "algorithm_label": meta["label"],
                    "group": str(groups[i]) if i < len(groups) else "all",
                    "fly_id": fly_id,
                    "metric_name": meta["period_label"]
                    if mode == "period"
                    else meta["metric_label"],
                    "value": float(v),
                    "is_rhythmic": bool(flag[i]) if i < len(flag) else False,
                }
            )

    return pd.DataFrame(rows)


def _plotly_axis_ref(axis: str, idx: int) -> str:
    """Return a valid Plotly axis reference: 'x' for idx=1, 'x2' for idx=2, etc."""
    return axis if idx <= 1 else f"{axis}{idx}"


def _add_significance_brackets(
    fig, col: int, group_order: list, pairwise: list, y_top: float, y_step: float
):
    """Draw significance brackets on one subplot column."""
    sig = [p for p in pairwise if p.get("significant") and p.get("stars")]
    if not sig:
        return y_top

    # Sort brackets by distance between groups so shorter ones are drawn lower
    def dist(pr):
        try:
            return abs(group_order.index(pr["group_a"]) - group_order.index(pr["group_b"]))
        except ValueError:
            return 0

    sig.sort(key=dist)

    current_y = y_top
    for pr in sig:
        try:
            xa = group_order.index(pr["group_a"])
            xb = group_order.index(pr["group_b"])
        except ValueError:
            continue
        x_left, x_right = min(xa, xb), max(xa, xb)
        # Horizontal line
        xref = _plotly_axis_ref("x", col)
        yref = _plotly_axis_ref("y", col)
        fig.add_shape(
            type="line",
            xref=xref,
            yref=yref,
            x0=x_left,
            x1=x_right,
            y0=current_y,
            y1=current_y,
            line=dict(color="black", width=1),
        )
        # Tick down on each end
        for xt in (x_left, x_right):
            fig.add_shape(
                type="line",
                xref=xref,
                yref=yref,
                x0=xt,
                x1=xt,
                y0=current_y,
                y1=current_y - y_step * 0.3,
                line=dict(color="black", width=1),
            )
        fig.add_annotation(
            xref=xref,
            yref=yref,
            x=(x_left + x_right) / 2.0,
            y=current_y + y_step * 0.1,
            text=pr["stars"],
            showarrow=False,
            font=dict(size=14, color="black"),
        )
        current_y += y_step
    return current_y


def rhythmicity_violin_grid(
    ds: xr.Dataset,
    mode: str,
    algorithms: list,
    group_coord: str = "group",
    stats_fn=None,
    show_significance: bool = True,
    filter_nonrhythmic: bool = True,
    filter_gate: str = "ac",
    period_window=None,
    metric_padding_frac: float = 0.10,
):
    """
    Build a horizontally-stacked violin grid (one column per algorithm).

    Parameters
    ----------
    ds : xr.Dataset
        Must contain the period/metric variables for each algorithm. Rhythmic
        flag coords (e.g. ``ls_rhythmic``) are required for mode='period'.
    mode : {'period', 'metric'}
        'period' plots the calculated circadian period with arrhythmic flies
        EXCLUDED (subject to ``filter_nonrhythmic`` / ``filter_gate``).
        'metric' plots the rhythmicity metric (LS FAP, AC RI, CWT
        rhythmicity) for ALL flies with a dashed cutoff line.
    algorithms : list of str
        Subset of {'ls','ac','cwt'}. Columns are drawn in this order.
    group_coord : str
        Name of the per-id group coordinate.
    stats_fn : callable, optional
        Function(values_by_group_dict) -> compare_groups-style result. If
        None, no statistics are computed or drawn.
    filter_nonrhythmic : bool, default True
        If True (period mode only), drop flies flagged arrhythmic by
        ``filter_gate`` before computing both the violin distribution and
        the pairwise group statistics. Default True — keeping arrhythmic
        flies biases group period estimates because their period values
        are essentially random. Set False to include them (e.g. when the
        biology of interest is loss of rhythmicity).
    filter_gate : {'ls', 'ac', 'cwt'}, default 'ac'
        Which classifier's rhythmic flag drives the filter.
    period_window : (lo, hi) tuple of float in hours, optional
        Y-axis clamp for ``mode='period'``. ``(min_period, max_period)``
        from the active analysis range; ±5% padding is added inside this
        function. Stable across reruns and prevents plotly's autorange
        from compressing meaningful between-group differences. When None
        (default), data-driven min/max with 5% padding is used.
    metric_padding_frac : float, default 0.10
        Fractional padding around the data-driven y-range for
        ``mode='metric'`` panels (10% above and below). Each metric
        panel autoranges independently because their scales differ
        (FAP ∈ [0, 1] log; RI ∈ [-1, 1]; CWT rhythmicity unbounded).

    Returns
    -------
    fig : plotly.graph_objects.Figure
    long_df : pd.DataFrame  (raw plotted values; one row per fly per algo)
    stats_by_alg : dict     {algo: stats_result}   (empty if stats_fn is None)
    """
    from plotly.subplots import make_subplots

    algorithms = [a for a in algorithms if a in _RHYTH_VIOLIN_META]
    if not algorithms:
        return go.Figure(), pd.DataFrame(), {}

    long_df = _collect_violin_data(
        ds,
        mode,
        algorithms,
        group_coord=group_coord,
        filter_nonrhythmic=filter_nonrhythmic,
        filter_gate=filter_gate,
    )
    n_total_flies = ds.sizes.get("id", 0)
    n_plotted = long_df["fly_id"].nunique() if not long_df.empty else 0
    print(
        f"[violin_grid] mode={mode}, total flies={n_total_flies}, "
        f"plotted={n_plotted}, rows={len(long_df)}, algos={algorithms}"
    )
    if long_df.empty:
        fig = go.Figure()
        fig.update_layout(
            title="No data to display (no flies passed the filter / analysis not run)",
            height=300,
        )
        return fig, long_df, {}

    groups_present = sorted(long_df["group"].unique().tolist())
    try:
        colors = _get_group_colors(groups_present)  # defined elsewhere in this file
    except NameError:
        palette = [
            "#1f77b4",
            "#ff7f0e",
            "#2ca02c",
            "#d62728",
            "#9467bd",
            "#8c564b",
            "#e377c2",
            "#7f7f7f",
            "#bcbd22",
            "#17becf",
        ]
        colors = {g: palette[i % len(palette)] for i, g in enumerate(groups_present)}

    subplot_titles = [_RHYTH_VIOLIN_META[a]["label"] for a in algorithms]
    n_groups = len(groups_present)
    n_cols = len(algorithms)
    h_spacing = max(0.02, 0.08 - 0.005 * n_groups)
    fig = make_subplots(
        rows=1,
        cols=n_cols,
        subplot_titles=subplot_titles,
        horizontal_spacing=h_spacing,
    )

    marker_sz = max(2, 5 - 0.2 * n_groups)
    pt_jitter = min(0.5, 0.2 + 0.02 * n_groups)
    violin_width = max(0.15, min(0.7, 5.0 / n_groups))

    stats_by_alg: dict = {}

    for col_idx, algo in enumerate(algorithms, start=1):
        meta = _RHYTH_VIOLIN_META[algo]
        sub = long_df[long_df["algorithm"] == algo]
        if sub.empty:
            continue

        values_by_group = {
            g: sub.loc[sub["group"] == g, "value"].to_numpy() for g in groups_present
        }

        # Draw one violin (or box for small n) per group
        max_group_n = max((v.size for v in values_by_group.values()), default=0)
        use_box = max_group_n < 8  # box+strip is clearer for small samples
        for g in groups_present:
            vals = values_by_group.get(g, np.array([]))
            if vals.size == 0:
                continue
            if use_box:
                fig.add_trace(
                    go.Box(
                        x=[g] * len(vals),
                        y=vals,
                        name=g,
                        legendgroup=g,
                        showlegend=(col_idx == 1),
                        line_color=colors.get(g, "#444"),
                        fillcolor=colors.get(g, "#444"),
                        opacity=0.6,
                        boxpoints="all",
                        jitter=pt_jitter,
                        pointpos=0,
                        marker=dict(size=marker_sz, opacity=0.8),
                        width=violin_width,
                    ),
                    row=1,
                    col=col_idx,
                )
            else:
                fig.add_trace(
                    go.Violin(
                        x=[g] * len(vals),
                        y=vals,
                        name=g,
                        legendgroup=g,
                        showlegend=(col_idx == 1),
                        line_color=colors.get(g, "#444"),
                        fillcolor=colors.get(g, "#444"),
                        opacity=0.6,
                        box_visible=True,
                        meanline_visible=True,
                        points="all",
                        jitter=pt_jitter,
                        pointpos=0,
                        marker=dict(size=marker_sz, opacity=0.8),
                        width=violin_width,
                        scalemode="width",
                    ),
                    row=1,
                    col=col_idx,
                )

        # Axis labels
        y_label = meta["period_label"] if mode == "period" else meta["metric_label"]
        fig.update_yaxes(title_text=y_label, row=1, col=col_idx)
        fig.update_xaxes(
            title_text="Group",
            row=1,
            col=col_idx,
            categoryorder="array",
            categoryarray=groups_present,
        )

        # ----- Y-axis range (Phase 2D adaptive clamps) -----
        if mode == "period":
            # Clamp to active analysis window with 5% padding so between-
            # group differences aren't compressed by extreme outliers.
            if period_window is not None:
                lo, hi = float(period_window[0]), float(period_window[1])
            else:
                # Data-driven fallback
                _algo_vals = sub["value"].to_numpy()
                _finite = _algo_vals[np.isfinite(_algo_vals)]
                if _finite.size:
                    lo, hi = float(_finite.min()), float(_finite.max())
                else:
                    lo, hi = 16.0, 32.0
            pad = 0.05 * max(hi - lo, 1e-9)
            fig.update_yaxes(range=[lo - pad, hi + pad], row=1, col=col_idx)
        else:
            # Metric mode: data-driven autorange with explicit padding so
            # plotly doesn't squeeze the data into a thin band when the
            # cutoff line sits far from the bulk.
            _algo_vals = sub["value"].to_numpy()
            _finite = _algo_vals[np.isfinite(_algo_vals)]
            if _finite.size:
                if meta.get("metric_log_y", False):
                    _pos = _finite[_finite > 0]
                    if _pos.size:
                        _lo_log = np.log10(_pos.min())
                        _hi_log = np.log10(_pos.max())
                        _pad_log = max((_hi_log - _lo_log) * metric_padding_frac, 0.1)
                        fig.update_yaxes(
                            type="log",
                            range=[_lo_log - _pad_log, _hi_log + _pad_log],
                            row=1,
                            col=col_idx,
                        )
                else:
                    _lo = float(_finite.min())
                    _hi = float(_finite.max())
                    pad = metric_padding_frac * max(_hi - _lo, 1e-9)
                    fig.update_yaxes(range=[_lo - pad, _hi + pad], row=1, col=col_idx)
            elif meta.get("metric_log_y", False):
                fig.update_yaxes(type="log", row=1, col=col_idx)

        # Cutoff reference line (metric mode only)
        if mode == "metric":
            cutoff = ds.attrs.get(meta["threshold_attr"], None)
            if cutoff is not None and np.isfinite(cutoff):
                fig.add_hline(
                    y=float(cutoff),
                    line_dash="dash",
                    line_color="red",
                    annotation_text=f"cutoff={cutoff:g}",
                    annotation_position="top right",
                    row=1,
                    col=col_idx,
                )

        # Statistics
        if stats_fn is not None:
            try:
                result = stats_fn(values_by_group)
            except Exception as e:
                result = {
                    "test": "none",
                    "statistic": float("nan"),
                    "pvalue": float("nan"),
                    "pairwise": [],
                    "n_per_group": {},
                    "normality_passed": False,
                    "equal_variance_passed": False,
                    "notes": f"stats error: {e}",
                }
            stats_by_alg[algo] = result

            # Omnibus p annotation at top of subplot
            p_omni = result.get("pvalue", float("nan"))
            test_name = result.get("test", "none")
            if show_significance and np.isfinite(p_omni):
                p_str = "p<0.001" if p_omni < 0.001 else f"p={p_omni:.3f}"
                label = (
                    "ANOVA"
                    if test_name == "anova"
                    else ("Kruskal-Wallis" if test_name == "kruskal" else test_name)
                )
                fig.add_annotation(
                    xref=f"{_plotly_axis_ref('x', col_idx)} domain",
                    yref=f"{_plotly_axis_ref('y', col_idx)} domain",
                    x=0.5,
                    y=1.03,
                    xanchor="center",
                    yanchor="bottom",
                    text=f"{label} {p_str}",
                    showarrow=False,
                    font=dict(size=11, color="gray"),
                )

            # Significance brackets
            all_vals = sub["value"].to_numpy()
            all_vals = all_vals[np.isfinite(all_vals)]
            if (
                show_significance
                and all_vals.size > 0
                and not (mode == "metric" and meta["metric_log_y"])
            ):
                y_top = float(np.nanmax(all_vals))
                y_range = float(np.nanmax(all_vals) - np.nanmin(all_vals))
                y_step = max(y_range * 0.08, y_top * 0.05, 1e-6)
                _add_significance_brackets(
                    fig,
                    col_idx,
                    groups_present,
                    result.get("pairwise", []),
                    y_top=y_top + y_step,
                    y_step=y_step,
                )

    title = "Period (rhythmic flies only)" if mode == "period" else "Rhythmicity metric (all flies)"
    fig_width = max(700, n_groups * 140 * n_cols)
    fig_height = max(500, 450 + n_groups * 10)
    fig.update_layout(
        title=title,
        width=fig_width,
        height=fig_height,
        violinmode="overlay",
        boxmode="overlay",
        showlegend=True,
        legend=dict(title="Group"),
    )
    for col_idx in range(1, n_cols + 1):
        apply_category_ticks(fig, groups_present, row=1, col=col_idx)
    return fig, long_df, stats_by_alg


def _threshold_explorer_arrays(
    ds: xr.Dataset, algo: str, threshold: float, period_window=None, group_coord: str = "group"
):
    """Shared core of the threshold explorer for ONE algorithm: the per-fly period,
    strength, group labels, and the live masks (analyzed / in-window / rhythmic) at
    ``threshold``. Both :func:`threshold_coupled_figure` (the plot) and
    :func:`threshold_explorer_table` (the export) call this, so the rhythmic call is
    derived in EXACTLY ONE place — the exported call cannot drift from the plotted
    one.

    Returns ``None`` when the algorithm's ``period_var``/``metric_var`` are absent
    (analysis not run). Raises ``ValueError`` for an unknown ``algo``.
    """
    if algo not in _RHYTH_VIOLIN_META:
        raise ValueError(f"algo must be one of {list(_RHYTH_VIOLIN_META)}, got {algo!r}")
    meta = _RHYTH_VIOLIN_META[algo]
    period_var, metric_var = meta["period_var"], meta["metric_var"]
    if period_var not in ds or metric_var not in ds:
        return None
    if period_window is None:
        lo = ds.attrs.get(f"{algo}_period_window_min", 16.0)
        hi = ds.attrs.get(f"{algo}_period_window_max", 32.0)
        period_window = (float(lo), float(hi))
    lo, hi = period_window
    period = np.asarray(ds[period_var].values, dtype=float).ravel()
    strength = np.asarray(ds[metric_var].values, dtype=float).ravel()
    groups = _get_group_coord_values(ds, group_coord)
    analyzed = np.isfinite(strength)  # candidates (real computation)
    in_window = np.isfinite(period) & (period >= lo) & (period <= hi)
    rhythmic = analyzed & in_window & (strength > float(threshold))  # live call
    return {
        "meta": meta,
        "period": period,
        "strength": strength,
        "groups": groups,
        "analyzed": analyzed,
        "in_window": in_window,
        "rhythmic": rhythmic,
        "window": (lo, hi),
    }


def threshold_explorer_table(
    ds: xr.Dataset, algo: str, threshold: float, period_window=None, group_coord: str = "group"
):
    """Per-fly table underlying the threshold explorer's ONE algorithm tab at the
    live ``threshold`` — the export payload behind the explorer's "Export" button.

    One row per successfully-analysed fly (NaN-strength / failed-analysis flies are
    excluded, exactly as they are absent from both explorer panels), with columns:
    ``ID``, ``Group``, the algorithm's period, its strength metric, ``Rhythmic``
    (the live call at ``threshold``), and ``Threshold`` (the cutoff used, recorded
    so a saved file is self-documenting). Rows are sorted alphabetically by Group
    then ID. Uses the SAME masks as :func:`threshold_coupled_figure` via
    :func:`_threshold_explorer_arrays`, so the exported call matches the plot.

    Returns an EMPTY ``DataFrame`` when the analysis was not run.
    """
    arr = _threshold_explorer_arrays(ds, algo, threshold, period_window, group_coord)
    if arr is None:
        return pd.DataFrame()
    meta = arr["meta"]
    analyzed = arr["analyzed"]
    ids = np.asarray(ds["id"].values)

    # ASCII-safe column headers so the CSV opens cleanly in Excel / GraphPad on
    # Windows (the MESA metric label carries an em-dash; keep it in the plot, not
    # in the file). En/em dashes -> '-', anything else non-ASCII -> '?'.
    def _ascii(s):
        return str(s).replace("—", "-").replace("–", "-").encode("ascii", "replace").decode("ascii")

    df = pd.DataFrame(
        {
            "ID": ids[analyzed],
            "Group": arr["groups"][analyzed].astype(str),
            _ascii(meta["period_label"]): arr["period"][analyzed],
            _ascii(meta["metric_label"]): arr["strength"][analyzed],
            "Rhythmic": arr["rhythmic"][analyzed],
            "Threshold": float(threshold),
        }
    )
    sort_keys = [c for c in ("Group", "ID") if c in df.columns]
    return df.sort_values(sort_keys).reset_index(drop=True) if sort_keys else df


def threshold_coupled_figure(
    ds: xr.Dataset,
    algo: str,
    threshold: float,
    period_window=None,
    group_coord: str = "group",
    scamp_ref: float = None,
):
    """Two coupled panels for ONE algorithm, threshold-driven (Part 2b).

    LEFT  — PERIOD distribution (box + every per-fly point) of the flies CURRENTLY
            classified rhythmic at ``threshold`` (and period in ``period_window``).
    RIGHT — STRENGTH distribution (box + every per-fly point) of ALL successfully
            analysed flies, points coloured by the live call, with ``threshold``
            drawn as a horizontal line.

    Correctness contract:
      * Per-fly period and strength are READ from the dataset (precomputed
        ``(id)`` data_vars) — this function never re-runs analysis. ``threshold``
        only changes which flies appear on the left and the line/point-colours on
        the right: a pure display filter.
      * "Arrhythmic" is DERIVED here (strength vs ``threshold`` AND period in
        window), never a stored fact. A fly below threshold simply DISAPPEARS from
        the period panel — it is filtered, not zeroed (drag the threshold down and
        it returns). It is NOT plotted at the axis floor.
      * NaN strength means "no valid computation" (failed curation / degenerate
        signal) — those flies are excluded from BOTH panels (never candidates),
        kept visibly distinct from a real low value.

    All three algorithms use "higher strength = more rhythmic" (LS power, AC RI,
    CWT rhythmicity), so the gate is ``strength > threshold``. The strength line is
    necessary but not sufficient — the call also requires period ∈ window, so a
    point above the line but out-of-window shows as arrhythmic (correct).

    Parameters
    ----------
    threshold : float
        Live cutoff on this algorithm's strength metric (the slider value; its
        default comes from ``core/calibrations.py`` via the caller).
    period_window : (lo, hi), optional
        Hours; flies with period outside this are not rhythmic. Defaults to the
        recorded ``<algo>_period_window_*`` attrs or (16, 32).
    scamp_ref : float, optional
        For AC only — draws a faint reference line (e.g. the SCAMP 0.195 RI) so it
        is not conflated with the live 0.3 classification cutoff.

    Returns
    -------
    fig : plotly.graph_objects.Figure
    summary : dict  {group: {'n_analyzed': int, 'n_rhythmic': int}}  + '_total'
    """
    from plotly.subplots import make_subplots

    # Shared masks (analyzed / in-window / rhythmic) — computed once in
    # _threshold_explorer_arrays so the export cannot diverge from the plot (§2d).
    arr = _threshold_explorer_arrays(ds, algo, threshold, period_window, group_coord)
    if arr is None:
        meta = _RHYTH_VIOLIN_META[algo]
        return go.Figure(
            layout=dict(
                title=f"{meta['label']}: analysis not run "
                f"(missing {meta['period_var']}/{meta['metric_var']})",
                height=300,
            )
        ), {}
    meta = arr["meta"]
    period, strength, groups = arr["period"], arr["strength"], arr["groups"]
    analyzed, rhythmic = arr["analyzed"], arr["rhythmic"]
    lo, hi = arr["window"]

    groups_present = sorted({str(g) for g in groups[analyzed]}) if analyzed.any() else []
    palette = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]
    gcolor = {g: palette[i % len(palette)] for i, g in enumerate(groups_present)}
    gidx = {g: i for i, g in enumerate(groups_present)}
    rng = np.random.RandomState(0)

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=[
            f"{meta['period_label']} — rhythmic-called flies",
            f"{meta['metric_label']} — all flies (— = cutoff)",
        ],
        horizontal_spacing=0.12,
    )

    # Live-call point colours, identical on BOTH panels so a fly that reads
    # "arrhythmic" (red) on the strength panel is the same red point on the
    # period panel. Green = rhythmic (kept by the gate at this threshold),
    # red = arrhythmic (excluded). The call is the SAME `rhythmic` mask the
    # summary counts use — never recomputed.
    _RHYTH_COLOR, _ARRHYTH_COLOR = "#2ca02c", "#d62728"

    summary = {}
    for g in groups_present:
        gmask = np.array([str(x) == g for x in groups])
        i = gidx[g]
        col = gcolor[g]
        # ---- LEFT: periods of the RHYTHMIC-CALLED flies ONLY (B2, 2026-06-30).
        # The point-coloring that briefly showed arrhythmic flies (red) on this
        # panel did its job — it proved the long-period spread WAS the excluded
        # arrhythmic flies, not the algorithm reading 49 h on good flies. Now the
        # panel is simplified back to the rhythmic-only period distribution that
        # actually feeds the period aggregate: arrhythmic-called flies are FILTERED
        # OUT here (lower the slider and they return as their strength clears the
        # cutoff). The live colouring stays on the STRENGTH panel (right). This
        # mask is the SAME `rhythmic` mask the summary/aggregate use (§2d — never
        # recomputed): a fly cannot appear here yet be absent from the aggregate.
        pmask = gmask & rhythmic & np.isfinite(period)  # rhythmic-only
        rhy_p = period[pmask]
        if rhy_p.size:
            fig.add_trace(
                go.Box(
                    y=rhy_p,
                    x=[i] * rhy_p.size,
                    name=g,
                    legendgroup=g,
                    showlegend=False,
                    line_color=col,
                    fillcolor="rgba(0,0,0,0)",
                    boxpoints=False,
                    width=0.6,
                ),
                row=1,
                col=1,
            )
            jit_p = (rng.rand(rhy_p.size) - 0.5) * 0.4
            fig.add_trace(
                go.Scatter(
                    x=i + jit_p,
                    y=rhy_p,
                    mode="markers",
                    name="rhythmic",
                    legendgroup="rhythmic",
                    showlegend=False,  # legend (rhythmic/arrhythmic) shown on strength panel
                    marker=dict(size=5, opacity=0.8, color=_RHYTH_COLOR),
                    hovertext=[g] * rhy_p.size,
                    hoverinfo="y+text",
                ),
                row=1,
                col=1,
            )
        # ---- RIGHT: strengths of ALL analysed flies, points coloured by call -
        smask = gmask & analyzed
        svals = strength[smask]
        scall = rhythmic[smask]
        if svals.size:
            fig.add_trace(
                go.Box(
                    y=svals,
                    x=[i] * svals.size,
                    name=g,
                    legendgroup=g,
                    showlegend=False,
                    line_color=col,
                    fillcolor="rgba(0,0,0,0)",
                    boxpoints=False,
                    width=0.6,
                ),
                row=1,
                col=2,
            )
            jit = (rng.rand(svals.size) - 0.5) * 0.4
            for sub_mask, color, lbl in (
                (scall, _RHYTH_COLOR, "rhythmic"),
                (~scall, _ARRHYTH_COLOR, "arrhythmic"),
            ):
                if sub_mask.any():
                    fig.add_trace(
                        go.Scatter(
                            x=(i + jit)[sub_mask],
                            y=svals[sub_mask],
                            mode="markers",
                            name=lbl,
                            legendgroup=lbl,
                            showlegend=(i == 0),  # shared legend (rhythmic/arrhythmic), this panel
                            marker=dict(size=5, opacity=0.8, color=color),
                            hovertext=[g] * int(sub_mask.sum()),
                            hoverinfo="y+text",
                        ),
                        row=1,
                        col=2,
                    )
        # B2: the live PERIOD AGGREGATE is computed over the RHYTHMIC-CALLED flies
        # ONLY (the same `rhythmic` mask, recomputed at this threshold) — an
        # arrhythmic-called fly's period CANNOT enter it. NaN when no fly is rhythmic.
        _g_rhy = gmask & rhythmic
        summary[g] = {
            "n_analyzed": int(smask.sum()),
            "n_rhythmic": int(_g_rhy.sum()),
            "period_median_rhythmic": (
                float(np.median(period[_g_rhy])) if _g_rhy.any() else float("nan")
            ),
        }

    # cutoff line on the strength panel
    fig.add_hline(
        y=float(threshold),
        line_dash="dash",
        line_color="black",
        annotation_text=f"cutoff={threshold:g}",
        annotation_position="top right",
        row=1,
        col=2,
    )
    if scamp_ref is not None and np.isfinite(scamp_ref):
        fig.add_hline(
            y=float(scamp_ref),
            line_dash="dot",
            line_color="gray",
            annotation_text=f"SCAMP ref {scamp_ref:g}",
            annotation_position="bottom right",
            row=1,
            col=2,
        )

    ticks = list(range(len(groups_present)))
    for _col in (1, 2):
        apply_category_ticks(
            fig, groups_present, tickvals=ticks, ticktext=groups_present,
            row=1, col=_col,
        )
    # period panel y-range: scale to the RANGE OF THE PLOTTED (rhythmic-called)
    # periods, NOT the full search window — a tight ~24 h cluster is dwarfed by an
    # empty [16, 36] h axis. Only when no fly is called rhythmic (nothing to scale
    # to) do we fall back to the search window.
    finite_p = period[rhythmic & np.isfinite(period)]
    if finite_p.size:
        dmin, dmax = float(np.min(finite_p)), float(np.max(finite_p))
        span = dmax - dmin
        pad = 0.05 * span if span > 0 else 0.5  # all-equal → a small fixed pad
        y_lo, y_hi = dmin - pad, dmax + pad
    else:
        pad = 0.05 * (hi - lo)
        y_lo, y_hi = lo - pad, hi + pad
    # faint ~24 h reference, drawn only when 24 h falls inside the (data-scaled) view
    if y_lo <= 24.0 <= y_hi:
        fig.add_hline(
            y=24.0,
            line_dash="dot",
            line_color="rgba(120,120,120,0.5)",
            annotation_text="~24 h",
            annotation_position="top left",
            row=1,
            col=1,
        )
    fig.update_yaxes(range=[y_lo, y_hi], title_text="period (h)", row=1, col=1)
    if meta.get("metric_log_y", False):
        fig.update_yaxes(type="log", title_text=meta["metric_label"], row=1, col=2)
    else:
        fig.update_yaxes(title_text=meta["metric_label"], row=1, col=2)

    n_an = int(analyzed.sum())
    n_rh = int(rhythmic.sum())
    summary["_total"] = {
        "n_analyzed": n_an,
        "n_rhythmic": n_rh,
        # Live period aggregate over RHYTHMIC-CALLED flies only (recomputed at this
        # threshold). An arrhythmic-called fly's period is structurally excluded.
        "period_median_rhythmic": (float(np.median(period[rhythmic])) if n_rh else float("nan")),
    }
    fig.update_layout(
        title=f"{meta['label']} — {n_rh}/{n_an} rhythmic at cutoff {threshold:g} "
        f"(period {lo:g}-{hi:g} h)",
        height=460,
        boxmode="overlay",
        legend=dict(title=None),
    )
    return fig, summary


def rhythmicity_long_to_wide_csv(long_df: pd.DataFrame) -> str:
    """
    Pivot the long-form violin dataframe to wide format with one column per
    group per algorithm, plus group-level summary rows at the bottom of each
    algorithm block.

    The result is a single CSV string where:
      - Each algorithm is emitted as its own block (separated by a blank row).
      - Within a block, columns are the groups, rows are per-fly values
        padded with NaN to the longest group.
      - A small summary (mean / sd / n / median / IQR) is appended.
    """
    if long_df is None or long_df.empty:
        return ""

    from io import StringIO

    buf = StringIO()
    algorithms = list(dict.fromkeys(long_df["algorithm_label"].tolist()))

    for i, algo_label in enumerate(algorithms):
        block = long_df[long_df["algorithm_label"] == algo_label]
        if block.empty:
            continue
        groups = sorted(block["group"].unique().tolist())
        # Build a padded wide table
        max_n = int(block.groupby("group").size().max())
        data = {}
        for g in groups:
            vals = block.loc[block["group"] == g, "value"].tolist()
            vals = vals + [np.nan] * (max_n - len(vals))
            data[g] = vals
        wide = pd.DataFrame(data)

        # Summary rows
        summary = pd.DataFrame(
            {
                g: [
                    np.nanmean(wide[g]),
                    np.nanstd(wide[g], ddof=1) if wide[g].notna().sum() > 1 else np.nan,
                    int(wide[g].notna().sum()),
                    np.nanmedian(wide[g]),
                    np.nanpercentile(wide[g].dropna(), 75) - np.nanpercentile(wide[g].dropna(), 25)
                    if wide[g].notna().sum() > 0
                    else np.nan,
                ]
                for g in groups
            },
            index=["mean", "sd", "n", "median", "IQR"],
        )

        buf.write(f"# {algo_label}\n")
        wide.to_csv(buf, index=False)
        buf.write("# summary\n")
        summary.to_csv(buf)
        if i < len(algorithms) - 1:
            buf.write("\n")

    return buf.getvalue()


# ---------------------------------------------------------------------------
# Phase-shift actogram (page 11)
# ---------------------------------------------------------------------------


def _bin_preserving_nan(values, bin_size):
    """Bin a 1-D trace by mean, keeping an all-missing bin missing.

    An all-NaN bin stays NaN rather than becoming 0: 0 means "measured, no
    movement" and NaN means "not measured", and collapsing the two would draw
    fabricated inactivity into the actogram.
    """
    values = np.asarray(values, dtype=float)
    if bin_size <= 1:
        return values
    usable = (len(values) // bin_size) * bin_size
    if usable == 0:
        return np.array([], dtype=float)
    blocks = values[:usable].reshape(-1, bin_size)
    finite = np.isfinite(blocks)
    out = np.full(blocks.shape[0], np.nan, dtype=float)
    has_data = finite.any(axis=1)
    if has_data.any():
        counts = finite[has_data].sum(axis=1)
        out[has_data] = np.where(finite[has_data], blocks[has_data], 0.0).sum(axis=1) / counts
    return out


def phase_shift_actogram(
    minutes,
    values,
    *,
    day_markers=None,
    pre_fit=None,
    post_fit=None,
    pulse_minute=None,
    pulse_duration_minutes=None,
    reference_day_index=None,
    bin_minutes=15,
    title="Actogram",
    double_plotted=True,
):
    """Double-plotted actogram for one fly, with the phase-shift fits overlaid.

    Each row is one day; a double-plotted row also repeats the following day to its
    right, so a drifting rhythm reads as a continuous diagonal instead of wrapping
    at the row edge. Days run top to bottom.

    The two regression lines from ``phase_shift.compute_fly_phase_shift`` are drawn
    in the same coordinates. A fit ``marker = intercept + slope * day`` becomes a
    straight diagonal here, because the x position within row ``d`` is
    ``marker - 1440*d``, so its per-day drift is ``(slope - 1440)`` — the familiar
    actogram slant, vertical when entrained and leaning when free-running. The
    pre-pulse line is solid over the days it was fitted on and dashed where it is
    extrapolated across the pulse: the gap between that dashed extension and the
    post-pulse line at the reference day IS the reported phase shift, so the number
    in the results table can be read straight off the picture.

    Parameters
    ----------
    minutes, values : array-like
        Relative-minute time axis and the matching activity trace for one fly.
    day_markers : dict[int, float], optional
        ``{day_index: marker_minute}`` as returned by the marker detectors.
    pre_fit, post_fit : dict, optional
        ``{'slope', 'intercept', 'n', ...}`` fits (``result['fits'][fly_id]``).
    pulse_minute, pulse_duration_minutes : float, optional
        Light-pulse position and width; shaded when given.
    reference_day_index : int, optional
        Day the two fits are compared at. Defaults to the pulse day.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    minutes = np.asarray(minutes, dtype=float)
    values = np.asarray(values, dtype=float)
    span = 2 if double_plotted else 1

    n_days = max(1, int(np.floor((minutes[-1] + 1) / 1440.0)))

    fig = go.Figure()
    hours_per_bin = bin_minutes / 60.0

    # Scale bar heights against the max of the BINNED trace, not the raw per-minute
    # max. A single-minute spike is many times any 15-minute mean, so normalising to
    # it flattens every bar to a few percent of the row and the actogram reads as
    # empty.
    binned_by_day = {}
    for day in range(n_days):
        lo = day * 1440
        sel = (minutes >= lo) & (minutes < lo + span * 1440)
        if not sel.any():
            continue
        binned = _bin_preserving_nan(values[sel], int(bin_minutes))
        if binned.size:
            binned_by_day[day] = (binned, (minutes[sel][0] - lo) / 60.0)

    peak = max(
        (np.nanmax(b) for b, _ in binned_by_day.values() if np.isfinite(b).any()), default=1.0
    )
    if not np.isfinite(peak) or peak <= 0:
        peak = 1.0

    bar_x, bar_y, bar_base = [], [], []
    for day, (binned, offset_hours) in binned_by_day.items():
        x = offset_hours + np.arange(binned.size) * hours_per_bin
        # 0.9 leaves a gutter between rows; NaN bins are simply not drawn, so a
        # data gap shows as blank rather than as a flat "no activity" floor.
        height = 0.9 * np.clip(binned / peak, 0.0, 1.0)
        good = np.isfinite(height)
        bar_x.extend(x[good].tolist())
        bar_y.extend((-height[good]).tolist())
        bar_base.extend([day + 0.95] * int(good.sum()))

    if bar_x:
        fig.add_trace(
            go.Bar(
                x=bar_x,
                y=bar_y,
                base=bar_base,
                width=hours_per_bin,
                marker=dict(color="#333333", line_width=0),
                hovertemplate="ZT %{x:.1f} h<extra></extra>",
                showlegend=False,
                name="activity",
            )
        )

    def _row_positions(minute_value, day):
        """(x, row) pairs for a time in row `day` — twice when double-plotted."""
        out = [((minute_value - day * 1440) / 60.0, day)]
        if double_plotted and day >= 1:
            out.append(((minute_value - (day - 1) * 1440) / 60.0, day - 1))
        return out

    # --- light pulse ------------------------------------------------------
    if pulse_minute is not None and np.isfinite(pulse_minute):
        width_h = 0.0
        if pulse_duration_minutes is not None and np.isfinite(pulse_duration_minutes):
            width_h = float(pulse_duration_minutes) / 60.0
        pulse_day = int(np.floor(float(pulse_minute) / 1440.0))
        for x0, row in _row_positions(float(pulse_minute), pulse_day):
            if row < 0 or x0 > span * 24:
                continue
            fig.add_shape(
                type="rect",
                x0=x0,
                x1=min(x0 + max(width_h, 0.25), span * 24),
                y0=row,
                y1=row + 1,
                fillcolor="gold",
                opacity=0.55,
                line_width=0,
                layer="below",
            )
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(color="gold", size=10, symbol="square"),
                name="light pulse",
            )
        )

    # --- per-day markers --------------------------------------------------
    if day_markers:
        mx, my = [], []
        for day, minute_value in sorted(day_markers.items()):
            for x, row in _row_positions(float(minute_value), int(day)):
                if row < 0 or x > span * 24:
                    continue
                mx.append(x)
                my.append(row + 0.5)
        if mx:
            fig.add_trace(
                go.Scatter(
                    x=mx,
                    y=my,
                    mode="markers",
                    marker=dict(color="#d62728", size=7, symbol="diamond"),
                    name="daily marker",
                    hovertemplate="marker at %{x:.2f} h<extra></extra>",
                )
            )

    # --- fitted / extrapolated lines --------------------------------------
    ref_day = reference_day_index
    if ref_day is None and pulse_minute is not None and np.isfinite(pulse_minute):
        ref_day = int(np.floor(float(pulse_minute) / 1440.0))

    def _add_fit(fit, solid_days, color, name, dashed_days=()):
        if not fit:
            return
        slope, intercept = fit["slope"], fit["intercept"]
        for label, day_list, dash in (
            (name, list(solid_days), "solid"),
            (f"{name} extrapolated", list(dashed_days), "dash"),
        ):
            xs, ys = [], []
            for day in day_list:
                x = (intercept + slope * day - day * 1440) / 60.0
                if 0 <= x <= span * 24:
                    xs.append(x)
                    ys.append(day + 0.5)
            if len(xs) < 2:
                continue
            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=ys,
                    mode="lines",
                    line=dict(color=color, width=2, dash=dash),
                    name=label,
                    hovertemplate=f"{label}<extra></extra>",
                )
            )

    if ref_day is not None:
        if pre_fit:
            n_pre = int(pre_fit.get("n", 0))
            _add_fit(
                pre_fit,
                range(max(0, ref_day - n_pre), ref_day),
                "#1f77b4",
                "pre-pulse fit",
                dashed_days=range(ref_day, min(n_days, ref_day + 3)),
            )
        if post_fit:
            n_post = int(post_fit.get("n", 0))
            _add_fit(
                post_fit,
                range(ref_day + 1, min(n_days, ref_day + 1 + n_post)),
                "#2ca02c",
                "post-pulse fit",
                dashed_days=range(ref_day, ref_day + 2),
            )

    fig.update_layout(
        title=dict(text=title, y=0.98, yanchor="top"),
        barmode="overlay",
        bargap=0,
        height=max(420, 46 * n_days) + 60,
        plot_bgcolor="white",
        legend=dict(orientation="h", yanchor="top", y=-0.08, x=0),
        margin=dict(l=60, r=30, t=60, b=90),
    )
    fig.update_xaxes(
        title="hours from day start" + (" (day d, then d+1)" if double_plotted else ""),
        range=[0, span * 24],
        tickmode="array",
        tickvals=list(range(0, span * 24 + 1, 6)),
        showgrid=True,
        gridcolor="#eeeeee",
    )
    fig.update_yaxes(
        title="day",
        range=[n_days, 0],
        tickmode="array",
        tickvals=[d + 0.5 for d in range(n_days)],
        ticktext=[str(d) for d in range(n_days)],
        showgrid=False,
    )
    return fig
