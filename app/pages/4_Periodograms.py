"""
Periodograms Page — group-averaged spectra from every period-analysis method
that produces a full curve: Lomb-Scargle, Autocorrelation, CWT, and MESA.

Each method gets its OWN section (mechanism blurb + a group-averaged spectrum),
because the y-axes are NOT comparable across methods (LS power vs autocorrelation
vs wavelet power vs AR power, each normalised differently). Curves are averaged
BY GROUP (not per-fly), mean ± SEM, from the per-fly arrays the Period Analysis
page already stored on the dataset. Run Period Analysis first to populate them.
"""

import os
import sys

import numpy as np
import streamlit as st

# Add core/ (analysis modules) and app/ to the import path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CORE_DIR = os.path.join(PROJECT_ROOT, "core")
APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in [CORE_DIR, APP_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

import export_helpers as ex
import plotting
from dataset_meta import dataset_fingerprint

st.header("Periodograms")

if st.session_state.get("dataset") is None:
    st.warning("No dataset loaded. Go to **Data Loading** first.")
    st.stop()

ds = st.session_state.dataset

# ------------------------------------------------------------------ #
# Group coordinate: prefer the experimental 'group', fall back to
# 'genotype', else a single pooled group. The overlays are BY GROUP.
# ------------------------------------------------------------------ #
if "group" in ds.coords:
    group_coord = "group"
elif "genotype" in ds.coords:
    group_coord = "genotype"
else:
    group_coord = None

if group_coord is not None:
    group_vals = np.asarray([str(v) for v in ds[group_coord].values])
    all_groups = sorted(set(group_vals.tolist()))
    st.sidebar.subheader("Filter Groups")
    selected_groups = st.sidebar.multiselect(
        f"Groups ({group_coord})", all_groups, default=all_groups, key="pgram_groups"
    )
    # multiselect returns a list in the app; guard the None the headless test
    # stub returns (and default to all groups if the widget hasn't populated).
    if selected_groups is None:
        selected_groups = list(all_groups)
else:
    group_vals = np.array(["All flies"] * ds.sizes["id"])
    all_groups = ["All flies"]
    selected_groups = ["All flies"]

st.caption(
    "Group-averaged spectra (mean ± SEM across flies). Each method has its own "
    "y-axis — **the scales are not comparable across methods** (LS power vs "
    "autocorrelation vs wavelet power vs AR power). Period methods share a common "
    "period-in-hours x-axis; autocorrelation is shown vs lag. Plain-language "
    "explanations of each method are on the **Period Analysis** page."
)


@st.cache_data(show_spinner=False)
def _curves_by_group(_fp, _ds, var, axis_name, selected, normalize, freq_to_period):
    """Build {group: (n_flies, n_points)} for a stored (id, axis) spectrum var.

    Averages BY GROUP. Per-fly max-normalisation (for the power methods) makes the
    group-average shape fair across flies of differing amplitude; the correlogram
    is already normalised so AC passes normalize=False. All-NaN rows (unanalyzable
    flies) are dropped; NaN entries within a row are handled point-wise downstream.
    Returns (per_group_curves, x_axis)."""
    x = np.asarray(_ds[axis_name].values, dtype=float)
    mat = np.asarray(_ds[var].transpose("id", axis_name).values, dtype=float)
    if freq_to_period:
        with np.errstate(divide="ignore"):
            x = 1.0 / x  # cycles/hour -> period (hours)
        order = np.argsort(x)
        x = x[order]
        mat = mat[:, order]
    gv = (
        np.asarray([str(v) for v in _ds[group_coord].values])
        if group_coord
        else np.array(["All flies"] * _ds.sizes["id"])
    )
    out = {}
    for g in selected:
        rows = mat[gv == g]
        if rows.shape[0] == 0:
            continue
        rows = rows[~np.all(~np.isfinite(rows), axis=1)]  # drop all-NaN flies
        if rows.shape[0] == 0:
            continue
        if normalize:
            with np.errstate(invalid="ignore"):
                peaks = np.nanmax(rows, axis=1, keepdims=True)
            peaks = np.where(np.isfinite(peaks) & (peaks != 0), peaks, np.nan)
            rows = rows / peaks
        out[g] = rows
    return out, x


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
    "sharper N/3 fit. Re-run MESA on **Period Analysis** to populate this curve."
)

fp = dataset_fingerprint(ds)
_any = False
for key, var, axis_name, title, plot_kw, normalize, freq_to_period in SECTIONS:
    st.subheader(title)
    # MESA: prefer the adaptive-order display spectrum; fall back to the estimation
    # periodogram for datasets saved before the display curve existed.
    if key == "mesa":
        if var in ds.data_vars:
            st.caption(_MESA_DISPLAY_NOTE)
        elif "mesa_periodogram" in ds.data_vars:
            var = "mesa_periodogram"
    if var not in ds.data_vars or axis_name not in ds.coords:
        st.info(f"No {title} periodogram stored yet — run it on the **Period Analysis** page.")
        st.divider()
        continue
    _any = True
    per_group, x_axis = _curves_by_group(
        fp, ds, var, axis_name, tuple(selected_groups), normalize, freq_to_period
    )
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
        # theme=None: the figure's own styling (black text, transparent bg) drives both
        # the on-screen chart and the "Download plot as PNG" export (see group_spectrum_plot).
        st.plotly_chart(fig, use_container_width=True, theme=None)
        if spec_df is not None and not spec_df.empty:
            ex.save_df_button(
                f"Save {title} group-average to working folder",
                spec_df,
                ds,
                f"periodogram_{key}_group_average.csv",
                key=f"dl_pg_{key}",
            )
    st.divider()

if not _any:
    st.warning(
        "None of the period methods have been run yet. Go to **Period Analysis**, "
        "run Lomb-Scargle / Autocorrelation / CWT / MESA, then return here."
    )
