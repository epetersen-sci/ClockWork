"""
SCAMP Export Page — write curated, LD/DD-split datasets into the legacy
"luc"-format files that SCAMP (`scamp.m`) ingests.

Requires the dataset to be curated and phase-split: the Data Loading +
Curate & split pages produce ``session_state.dataset_LD`` and
``session_state.dataset_DD``. The SCAMP loader needs equal-length files per
"board", so this page selects a common window per board (see
``scamp_export/README.md`` for details).
"""

import os

import streamlit as st

import dam_utilities  # noqa: E402
from scamp_export.scamp_exporter import export_dataset_to_scamp  # noqa: E402

st.caption(
    "Write curated, LD/DD-split data into legacy SCAMP files so the lab's "
    "existing MATLAB sleep/circadian analyses can ingest the cleaned data."
)

ds_ld = st.session_state.get("dataset_LD")
ds_dd = st.session_state.get("dataset_DD")

if ds_ld is None and ds_dd is None:
    st.warning(
        "No phase-split datasets in session. Run **Data Loading → "
        "Preprocessing** (which performs curation and the LD/DD split) before "
        "exporting."
    )
    st.stop()

available_phases = []
if ds_ld is not None:
    available_phases.append("LD")
if ds_dd is not None:
    available_phases.append("DD")

st.markdown(f"**Phases available:** {', '.join(available_phases)}")

with st.form("scamp_export_form"):
    col1, col2 = st.columns(2)
    with col1:
        out_root = st.text_input(
            "Output root directory",
            value=dam_utilities.resolve_export_dir(
                st.session_state.get("dataset"), st.session_state.get("working_dir")
            ),
            help="LD/ and DD/ subfolders are written underneath. Defaults to the "
            "folder your monitor data came from.",
        )
        prefix = st.text_input(
            "Filename prefix",
            value="PY",
            help="No 'C', no '.', no spaces. When the dataset combines "
            "multiple recording dates, the per-fly date is appended "
            "automatically so SCAMP sees distinct boards.",
        )
        lights_on = st.number_input(
            "Lights-on military time (HHMM)",
            min_value=0,
            max_value=2359,
            value=900,
            step=100,
            help="Single 'start' clock written into every file. SCAMP uses "
            "this only for axis labels — data are already ZT-aligned.",
        )
    with col2:
        ld_min_days = st.number_input(
            "LD minimum days",
            min_value=1.0,
            value=2.0,
            step=0.5,
            help="Flies with valid span below this are dropped from the LD export.",
        )
        dd_min_days = st.number_input(
            "DD minimum days",
            min_value=1.0,
            value=2.0,
            step=0.5,
            help="Flies with valid span below this are dropped from the DD export. "
            "Counted AFTER discard-first-DD-day if applied upstream.",
        )
        intervals_label = st.multiselect(
            "Sampling intervals",
            options=[1, 30],
            default=[1, 30],
            help="SCAMP prompts for both a 1-min and a 30-min folder; the "
            "30-min files are sums over 30 consecutive 1-min bins.",
        )
        interpolate_interior = st.checkbox(
            "Interpolate interior NaN",
            value=True,
            help="Linearly interpolate small interior NaN gaps inside each "
            "fly's window. The longest-segment trim has already removed "
            "any large gaps.",
        )

    submitted = st.form_submit_button("Write SCAMP files")

if submitted:
    if not out_root:
        st.error("Provide an output root directory.")
        st.stop()
    if not intervals_label:
        st.error("Pick at least one sampling interval.")
        st.stop()

    intervals = tuple(intervals_label)

    for phase, ds in (("LD", ds_ld), ("DD", ds_dd)):
        if ds is None:
            continue
        out_dir = os.path.join(out_root, phase)
        min_days = ld_min_days if phase == "LD" else dd_min_days
        try:
            with st.spinner(f"Writing {phase} export to {out_dir}…"):
                summary = export_dataset_to_scamp(
                    ds,
                    out_dir,
                    prefix=prefix,
                    interval_set=intervals,
                    lights_on_military=int(lights_on),
                    min_days=float(min_days),
                    interpolate_interior=interpolate_interior,
                    phase_label=phase,
                )
        except ValueError as e:
            st.error(f"{phase} export failed: {e}")
            continue

        st.success(
            f"{phase}: wrote **{summary['n_out']}** files across "
            f"**{len(summary['boards'])}** board(s) to `{summary['out_root']}`."
        )

        with st.expander(f"{phase} details"):
            import pandas as pd

            st.write(f"Group key: `{summary['group_key']}`")
            st.write(f"Manifest:  `{summary['manifest']}`")

            if summary["boards"]:
                st.dataframe(pd.DataFrame(summary["boards"]), width="stretch")
            if summary["dropped"]:
                st.write(f"Dropped {len(summary['dropped'])} flies:")
                st.dataframe(pd.DataFrame(summary["dropped"]), width="stretch")
