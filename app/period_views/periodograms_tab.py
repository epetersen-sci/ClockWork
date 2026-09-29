"""Periodograms tab: group-averaged spectra from every method that has a curve.

Each method gets its OWN figure, because the y-axes are not comparable across
methods (LS power vs autocorrelation vs wavelet power vs AR power, each
normalised differently). Curves are averaged BY GROUP — or by compared level
inside each facet panel — mean ± SEM, from the per-fly arrays the analyses
stored on the dataset.
"""

import warnings

import numpy as np
import streamlit as st

import export_helpers as ex
import facets
import plotting
from dataset_meta import dataset_fingerprint
from ui import charts, facet_panels
from ui.filters import resolve_group_coord


@st.cache_data(show_spinner=False, max_entries=16)
# `fp`, NOT `_fp`: Streamlit's underscore rule is syntactic and drops ANY
# leading-underscore parameter from the cache key, not just the unhashable
# dataset. Named `_fp` it was the one argument that could not reach the key —
# so a reloaded or re-analysed dataset kept serving the previous run's curves.
# The remaining args (var, selected, normalize, ...) hid it: changing method or
# group selection did invalidate, so only a dataset swap went stale.
def _curve_matrix(fp, _ds, var, axis_name, normalize, freq_to_period):
    """Every fly's stored (id, axis) spectrum as ``(ids, matrix, x_axis)``.

    Per-fly max-normalisation (for the power methods) makes the group-average
    shape fair across flies of differing amplitude; the correlogram is already
    normalised so AC passes normalize=False. Normalising is per fly, so doing it
    once for the whole dataset gives each group — or each facet panel — exactly
    what normalising its own flies would. All-NaN flies (unanalyzable) are dropped
    by ``plotting.curves_by_label``; NaN entries within a row are handled
    point-wise downstream."""
    x = np.asarray(_ds[axis_name].values, dtype=float)
    mat = np.asarray(_ds[var].transpose("id", axis_name).values, dtype=float)
    if freq_to_period:
        with np.errstate(divide="ignore"):
            x = 1.0 / x  # cycles/hour -> period (hours)
        order = np.argsort(x)
        x = x[order]
        mat = mat[:, order]
    if normalize:
        with np.errstate(invalid="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            peaks = np.nanmax(mat, axis=1, keepdims=True)
        peaks = np.where(np.isfinite(peaks) & (peaks != 0), peaks, np.nan)
        mat = mat / peaks
    ids = [str(i) for i in _ds["id"].values]
    return ids, mat, x



# (blurb-key, spectrum var, axis coord, section title, plot kwargs, normalize, freq->period)
SECTIONS = [
    (
        "ls",
        "ls_periodogram",
        "ls_periodogram_periods",
        "Lomb-Scargle",
        dict(
            xlabel="Period (hours)",
            ylabel="Normalised LS power",
            x_log=True,
            ref_period=24.0,
            zero_line=False,
        ),
        True,
        False,
    ),
    (
        "ac",
        "ac_correlogram",
        "ac_lag",
        "Autocorrelation",
        dict(
            xlabel="Lag (hours)",
            ylabel="Autocorrelation (r)",
            x_log=False,
            ref_period=24.0,
            zero_line=True,
        ),
        False,
        False,
    ),
    (
        "cwt",
        "cwt_powerseries",
        "cwt_periodogram_periods",
        "Continuous Wavelet Transform (CWT)",
        dict(
            xlabel="Period (hours)",
            ylabel="Normalised CWT power",
            x_log=True,
            ref_period=24.0,
            zero_line=False,
        ),
        True,
        False,
    ),
    (
        "mesa",
        "mesa_display_periodogram",
        "mesa_periodogram_periods",
        "MESA (Maximum Entropy)",
        dict(
            xlabel="Period (hours)",
            ylabel="Normalised AR power",
            x_log=True,
            ref_period=24.0,
            zero_line=False,
        ),
        True,
        False,
    ),
]

# MESA renders an ADAPTIVE-order (FPE) display spectrum: the N/3 estimation order
# gives the sharpest peak (used for the reported period) but a high order also
# invents spurious low peaks on arrhythmic flies (Burg line-splitting) that show up
# as a wavy group-average floor. The lower FPE order removes that while keeping the
# rhythmic peak. Old datasets (no display var) fall back to the estimation spectrum.
_MESA_DISPLAY_NOTE = (
    "Shown at an **adaptive (FPE) AR order** so arrhythmic flies don't add "
    "spurious high-order humps; the reported MESA *period* still comes from the "
    "sharper N/3 fit. Re-run MESA on the **Analysis** tab to populate this curve."
)


def _curves_by_group(ds, ids, mat, selected):
    """``{group: (n_flies, n_points)}`` for the selected groups, averaged BY GROUP."""
    coord = resolve_group_coord(ds)
    gv = [str(v) for v in ds[coord].values] if coord else ["All flies"] * ds.sizes["id"]
    return plotting.curves_by_label(ids, mat, dict(zip(ids, gv)), order=list(selected))


def render(ctx):
    ds = ctx.period_ds
    selected_groups = ctx.selected_groups
    facet_spec = ctx.facet_spec
    facet_panels_ = None
    if facet_spec.active:
        # Only the flies the sidebar group filter leaves in.
        facet_panels_ = facets.resolve_panels(
            ctx.facet_table[ctx.facet_table["group"].isin(set(selected_groups))], facet_spec
        )

    st.caption(
        "Group-averaged spectra (mean ± SEM across flies). Each method has its own "
        "y-axis — **the scales are not comparable across methods**. Period methods "
        "share a period-in-hours x-axis; autocorrelation is shown against lag."
    )

    fp = dataset_fingerprint(ds)
    any_run = False
    for key, var, axis_name, title, plot_kw, normalize, freq_to_period in SECTIONS:
        st.subheader(title)
        # MESA: prefer the adaptive-order display spectrum; fall back to the
        # estimation periodogram for datasets saved before it existed.
        if key == "mesa":
            if var in ds.data_vars:
                st.caption(_MESA_DISPLAY_NOTE)
            elif "mesa_periodogram" in ds.data_vars:
                var = "mesa_periodogram"
        if var not in ds.data_vars or axis_name not in ds.coords:
            st.info(f"No {title} periodogram stored yet — run it on the **Analysis** tab.")
            st.divider()
            continue
        any_run = True
        ids, mat, x_axis = _curve_matrix(fp, ds, var, axis_name, normalize, freq_to_period)
        if facet_panels_:
            pairs = plotting.faceted_spectra(
                ids,
                mat,
                x_axis,
                facet_panels_,
                facets.facet_colours(facets.layout_levels(facet_panels_)),
                title=title,
                shared_y=facet_spec.shared_y,
                **plot_kw,
            )
            facet_panels.render_panels(
                pairs, facet_spec, ds, key=f"pg_{key}", filename=f"periodogram_{key}"
            )
            st.divider()
            continue
        per_group = _curves_by_group(ds, ids, mat, tuple(selected_groups))
        if not per_group:
            st.info("No flies in the selected groups have a stored spectrum for this method.")
        else:
            fig, spec_df = plotting.group_spectrum_plot(
                per_group,
                x_axis,
                group_order=[g for g in selected_groups if g in per_group],
                title=f"{title} — group-averaged",
                return_data=True,
                **plot_kw,
            )
            # theme=None: the figure's own styling (black text, transparent bg)
            # drives both the on-screen chart and its PNG export.
            charts.plotly_chart(fig, width="stretch", theme=None)
            if spec_df is not None and not spec_df.empty:
                ex.save_df_button(
                    f"Save {title} group-average to working folder",
                    spec_df,
                    ds,
                    f"periodogram_{key}_group_average.csv",
                    key=f"dl_pg_{key}",
                )
        st.divider()

    if not any_run:
        st.warning("No period method has been run yet — run one on the **Analysis** tab.")
