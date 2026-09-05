"""
preprocessing_widgets.py
========================
Shared Streamlit UI for the canonical preprocessing pipeline.

Renders one expander with three columns (LS, AC, CWT), each showing the same
parameters with method-appropriate defaults. Returns a dict of
:class:`PreprocessConfig` keyed by method name (``'LS'``, ``'AC'``, ``'CWT'``).

Pages 3 (production) and 11 (parameter sweep) call this so they share an
identical UI surface and identical preprocessing semantics.
"""

from __future__ import annotations

import streamlit as st

from preprocessing import (
    PreprocessConfig,
    ac_default_config,
    cwt_default_config,
    ls_default_config,
)

_NORMALIZE_OPTIONS = ["none", "zscore", "robust", "envelope", "rolling"]
_DETREND_OPTIONS = ["none", "linear"]


# Documented per-method defaults — must match the SCAMP/Rethomics behavior
# of the pre-refactor pipeline so production page-3 numbers do not drift.
METHOD_DEFAULTS: dict[str, PreprocessConfig] = {
    "LS": ls_default_config(),
    "AC": ac_default_config(),
    "CWT": cwt_default_config(),
}


def _render_one_method(
    method: str, defaults: PreprocessConfig, key_prefix: str
) -> PreprocessConfig:
    """Render the per-method column. Returns a config built from the widgets.

    Each widget's default is the method's documented default. The user can
    deviate per method.
    """
    bin_minutes = st.number_input(
        "Bin (min)",
        min_value=0,
        max_value=60,
        value=int(defaults.bin_minutes),
        step=1,
        key=f"{key_prefix}_{method}_bin",
        help="Sum-bin activity along time. 0 or 1 disables.",
    )
    smooth_sigma_min = st.number_input(
        "Smoothing σ (min)",
        min_value=0.0,
        max_value=240.0,
        value=float(defaults.smooth_sigma_min),
        step=5.0,
        key=f"{key_prefix}_{method}_smooth",
        help="Per-fly NaN-preserving Gaussian σ in minutes. 0 disables.",
    )
    lopass_hours = st.number_input(
        "Low-pass cutoff (h)",
        min_value=0.0,
        max_value=24.0,
        value=float(defaults.lopass_hours),
        step=0.5,
        key=f"{key_prefix}_{method}_lopass",
        help="2nd-order zero-phase Butterworth low-pass. 0 disables. AC default = 4.0 (SCAMP).",
    )
    detrend = st.selectbox(
        "Detrend",
        options=_DETREND_OPTIONS,
        index=_DETREND_OPTIONS.index(defaults.detrend),
        key=f"{key_prefix}_{method}_detrend",
        help="`linear` = scipy.signal.detrend. AC default = linear (SCAMP).",
    )
    normalize = st.selectbox(
        "Normalize",
        options=_NORMALIZE_OPTIONS,
        index=_NORMALIZE_OPTIONS.index(defaults.normalize),
        key=f"{key_prefix}_{method}_norm",
        help="Per-fly amplitude normalization. None for production defaults.",
    )
    rolling_window_h = st.number_input(
        "Rolling window (h)",
        min_value=1.0,
        max_value=72.0,
        value=float(defaults.rolling_window_h),
        step=1.0,
        key=f"{key_prefix}_{method}_rollwin",
        help="Used only when Normalize = rolling.",
        disabled=(normalize != "rolling"),
    )
    return PreprocessConfig(
        bin_minutes=int(bin_minutes),
        smooth_sigma_min=float(smooth_sigma_min),
        lopass_hours=float(lopass_hours),
        detrend=str(detrend),
        normalize=str(normalize),
        rolling_window_h=float(rolling_window_h),
    )


def render_preprocess_expander(
    key_prefix: str,
    methods: list | None = None,
    method_defaults: dict[str, PreprocessConfig] | None = None,
    expanded: bool = False,
) -> dict[str, PreprocessConfig]:
    """Render a single expander with one column per method.

    Parameters
    ----------
    key_prefix
        Page-local prefix to disambiguate widget keys (e.g. ``"page3"`` or
        ``"sweep"``).
    methods
        Subset of ``["LS", "AC", "CWT"]`` to render. Defaults to all three.
    method_defaults
        Override the documented defaults. Useful when the page wants to
        propagate sweep-specific overrides into the UI. Defaults to
        :data:`METHOD_DEFAULTS`.
    expanded
        Whether the expander starts open.

    Returns
    -------
    dict
        Keys are method names (``"LS"``, ``"AC"``, ``"CWT"``); values are
        :class:`PreprocessConfig`.
    """
    methods = methods or ["LS", "AC", "CWT"]
    method_defaults = method_defaults or METHOD_DEFAULTS
    out: dict[str, PreprocessConfig] = {}

    with st.expander("Preprocessing (per-method)", expanded=expanded):
        st.caption(
            "All three methods share the same five-step pipeline "
            "(bin → smooth → lopass → detrend → normalize). Defaults: "
            "LS uses the generalized (Zechmeister-Kürster) periodogram "
            "with floating-mean fit per trial frequency — z-scoring is "
            "recommended for cross-recording amplitude comparability but "
            "not required for correctness. CWT: no preprocessing besides "
            "mean-centering. AC: 4 h Butterworth low-pass + linear "
            "detrend, SCAMP-style."
        )
        cols = st.columns(len(methods))
        for col, m in zip(cols, methods):
            with col:
                st.markdown(f"**{m}**")
                out[m] = _render_one_method(
                    m,
                    method_defaults.get(m, PreprocessConfig()),
                    key_prefix,
                )
    return out


__all__ = ["render_preprocess_expander", "METHOD_DEFAULTS"]
