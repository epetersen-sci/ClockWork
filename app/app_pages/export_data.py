"""
Export Page - Bulk CSV summaries, NetCDF save with compression, download buttons.
"""

import os

import pandas as pd
import streamlit as st

import dam_utilities
from analysis_detection import detect_analyses
from load_and_save_datasets import save_dataset_to_netcdf
from ui.guards import require_dataset

ds = require_dataset()
analyses = detect_analyses(ds)

# Three kinds of output, three tabs: the dataset itself, the raw/binned
# tables, and the per-analysis result exports.
tab_dataset, tab_tables, tab_results = st.tabs(
    ["Dataset (.nc)", "Activity & ZT tables", "Analysis results"]
)

with tab_dataset:
    # ============================================================
    # Save NetCDF
    # ============================================================
    st.subheader("Save Dataset (NetCDF)")

    save_dir = st.text_input(
        "Output directory",
        value=dam_utilities.resolve_export_dir(ds, st.session_state.get("working_dir")),
        key="export_dir",
    )
    save_filename = st.text_input(
        "Filename (without .nc)",
        value="analyzed_dataset",
        key="export_filename",
    )

    _has_phase_datasets = (
        st.session_state.get("dataset_DD") is not None or st.session_state.get("dataset_LD") is not None
    )
    if _has_phase_datasets:
        save_phase_datasets = st.checkbox(
            "Also save DD and LD phase datasets separately",
            value=True,
            key="save_phase_datasets",
            help="Saves each phase as its own NetCDF file (e.g. analyzed_dataset_DD.nc, analyzed_dataset_LD.nc) "
            "alongside the master dataset.",
        )
    else:
        save_phase_datasets = False

    if st.button("Save NetCDF", key="save_nc"):
        if not save_dir:
            st.error("Please provide an output directory.")
        else:
            os.makedirs(save_dir, exist_ok=True)
            full_path = os.path.join(save_dir, f"{save_filename}.nc")
            with st.spinner(f"Saving to {full_path}..."):
                try:
                    save_dataset_to_netcdf(ds, f"{save_filename}.nc", output_dir=save_dir)
                    st.session_state.dataset_path = full_path
                    st.success(f"Saved master dataset to {full_path}")

                    if save_phase_datasets:
                        if st.session_state.get("dataset_DD") is not None:
                            dd_path = os.path.join(save_dir, f"{save_filename}_DD.nc")
                            save_dataset_to_netcdf(
                                st.session_state.dataset_DD,
                                f"{save_filename}_DD.nc",
                                output_dir=save_dir,
                            )
                            st.success(f"Saved DD dataset to {dd_path}")
                        if st.session_state.get("dataset_LD") is not None:
                            ld_path = os.path.join(save_dir, f"{save_filename}_LD.nc")
                            save_dataset_to_netcdf(
                                st.session_state.dataset_LD,
                                f"{save_filename}_LD.nc",
                                output_dir=save_dir,
                            )
                            st.success(f"Saved LD dataset to {ld_path}")
                except Exception as e:
                    st.error(f"Error saving: {e}")


with tab_tables:
    # ============================================================
    # Export Activity CSV
    # ============================================================
    st.subheader("Export Activity Data (CSV)")
    st.markdown("Exports raw activity data: time column + one column per fly.")

    if "activity" in ds.data_vars:
        if st.button("Generate Activity CSV", key="gen_activity_csv"):
            with st.spinner("Generating CSV..."):
                try:
                    activity_df = ds["activity"].to_pandas().reset_index()
                    if activity_df.columns[0] != "time":
                        activity_df.rename(columns={activity_df.columns[0]: "time"}, inplace=True)

                    csv_data = activity_df.to_csv(index=False)
                    st.download_button(
                        "Download Activity CSV",
                        csv_data,
                        "activity_data.csv",
                        "text/csv",
                        key="dl_activity_csv",
                    )
                    st.success(f"Ready: {activity_df.shape[0]} rows x {activity_df.shape[1]} columns")
                except Exception as e:
                    st.error(f"Error: {e}")
    else:
        st.info("No activity data in dataset.")

    st.divider()

    # ============================================================
    # Export ZT-Binned Averaged Data
    # ============================================================
    st.subheader("Export ZT-Binned Averaged Data (CSV)")
    st.markdown(
        "Exports data binned by Zeitgeber Time in **two sets**: a group-averaged CSV "
        "(zt_bin_minute, zt_hours, then per group alphabetically **Mean, SD, N** — "
        "GraphPad grouped-table order) and a **per-fly** CSV (one row per fly per bin) "
        "so you can compute other stats yourself."
    )

    export_var = st.selectbox(
        "Variable to export",
        [v for v in ["activity", "sleep"] if v in ds.data_vars],
        key="export_var",
    )

    export_bin_size = st.slider(
        "Bin size (minutes)", min_value=5, max_value=60, value=30, step=5, key="export_bin_size"
    )

    export_bin_func = st.radio(
        "Binning function", ["mean", "sum"], horizontal=True, key="export_bin_func"
    )

    if st.button("Generate Averaged CSV", key="gen_avg_csv"):
        with st.spinner("Generating averaged CSV..."):
            try:
                binned_df = dam_utilities.get_zt_binned_dataframe(
                    ds, export_var, export_bin_size, export_bin_func
                )

                if binned_df.empty:
                    st.error("No binned data generated.")
                else:
                    # Add group info
                    id_to_group = {}
                    for fly_id in binned_df["id"].unique():
                        try:
                            fly_ds = ds.sel(id=fly_id)
                            if "group" in ds.coords:
                                id_to_group[fly_id] = str(fly_ds["group"].item())
                            elif "genotype" in ds.coords and "temperature" in ds.coords:
                                id_to_group[fly_id] = (
                                    f"{fly_ds['genotype'].item()}-{fly_ds['temperature'].item()}"
                                )
                            else:
                                id_to_group[fly_id] = "All"
                        except (KeyError, IndexError):
                            id_to_group[fly_id] = "All"

                    binned_df["group"] = binned_df["id"].map(id_to_group)

                    agg_df = (
                        binned_df.groupby(["zt_bin_minute", "group"])[export_var]
                        .agg(
                            Mean="mean",
                            SD=lambda x: x.std(ddof=1),
                            N=lambda x: x.notna().sum(),
                        )
                        .reset_index()
                    )

                    # Pivot to a wide (group, stat) MultiIndex. Groups are ordered
                    # ALPHABETICALLY and the stats are ordered Mean, SD, N (GraphPad's
                    # grouped-table order) so the CSV pastes straight into Prism — not
                    # the alphabetical Mean/N/SD a plain sort would give.
                    pivot_df = agg_df.pivot_table(
                        index="zt_bin_minute", columns="group", values=["Mean", "SD", "N"]
                    )
                    pivot_df.columns = pd.MultiIndex.from_tuples(
                        [(grp, stat) for stat, grp in pivot_df.columns], names=["group", "stat"]
                    )
                    _ordered_cols = pd.MultiIndex.from_tuples(
                        [(g, s) for g in sorted(agg_df["group"].unique()) for s in ("Mean", "SD", "N")],
                        names=["group", "stat"],
                    )
                    pivot_df = pivot_df.reindex(columns=_ordered_cols)
                    pivot_df.insert(
                        0,
                        ("zt_hours", ""),
                        dam_utilities.zt_bin_to_hours(pivot_df.index, export_bin_size),
                    )
                    result_df = pivot_df

                    csv_data = result_df.to_csv()
                    st.download_button(
                        "Download Averaged CSV (group Mean/SD/N)",
                        csv_data,
                        f"{export_var}_averaged.csv",
                        "text/csv",
                        key="dl_avg_csv",
                    )

                    # Second set: PER-FLY binned values (long) so other stats can be
                    # calculated. One row per fly per ZT bin — ID, Group, zt_bin_minute,
                    # zt_hours, <variable> — sorted alphabetically by Group then ID.
                    per_fly_df = binned_df.rename(columns={"id": "ID", "group": "Group"}).copy()
                    per_fly_df["zt_hours"] = dam_utilities.zt_bin_to_hours(
                        per_fly_df["zt_bin_minute"], export_bin_size
                    )
                    per_fly_df = (
                        per_fly_df[["ID", "Group", "zt_bin_minute", "zt_hours", export_var]]
                        .sort_values(["Group", "ID", "zt_bin_minute"])
                        .reset_index(drop=True)
                    )
                    st.download_button(
                        "Download Per-Fly CSV (for stats)",
                        per_fly_df.to_csv(index=False),
                        f"{export_var}_per_fly.csv",
                        "text/csv",
                        key="dl_perfly_csv",
                    )
                    n_groups = len(agg_df["group"].unique())
                    st.success(
                        f"Ready: {len(result_df)} bins x {n_groups} groups "
                        f"(+ per-fly: {len(per_fly_df)} rows)"
                    )
            except Exception as e:
                st.error(f"Error: {e}")


with tab_results:
    # ============================================================
    # Export Period Analysis Summary
    # ============================================================
    if any(analyses[k] for k in ["cwt", "lomb_scargle", "autocorrelation"]):
        st.subheader("Export Period Analysis Summary (CSV)")

        summary_data = []
        for fly_id in ds["id"].values:
            row = {"ID": fly_id}
            if "group" in ds.coords:
                row["Group"] = str(ds["group"].sel(id=fly_id).values)

            if "cwt_period" in ds.data_vars:
                row["CWT_Period_h"] = float(ds["cwt_period"].sel(id=fly_id).values)
                row["CWT_Power"] = float(ds["cwt_power"].sel(id=fly_id).values)
            if "cwt_period_stability" in ds.data_vars:
                row["CWT_Period_Stability_h"] = float(ds["cwt_period_stability"].sel(id=fly_id).values)
            if "cwt_rhythmicity" in ds.data_vars:
                row["CWT_Rhythmicity"] = float(ds["cwt_rhythmicity"].sel(id=fly_id).values)
            if "ls_period" in ds.data_vars:
                row["LS_Period_h"] = float(ds["ls_period"].sel(id=fly_id).values)
                row["LS_Power"] = float(ds["ls_power"].sel(id=fly_id).values)
            if "ls_fap" in ds.data_vars:
                row["LS_FAP"] = float(ds["ls_fap"].sel(id=fly_id).values)
            if "ac_period" in ds.data_vars:
                row["AC_Period_h"] = float(ds["ac_period"].sel(id=fly_id).values)
                row["AC_Power_RI"] = float(ds["ac_power"].sel(id=fly_id).values)
            if "ac_rhythm_strength" in ds.data_vars:
                row["AC_Rhythm_Strength"] = float(ds["ac_rhythm_strength"].sel(id=fly_id).values)

            # Per-algorithm rhythmic flags (written by rhythmicity_classification.classify_*)
            if "ls_rhythmic" in ds.coords:
                row["LS_Rhythmic"] = bool(ds["ls_rhythmic"].sel(id=fly_id).values)
            if "ac_rhythmic" in ds.coords:
                row["AC_Rhythmic"] = bool(ds["ac_rhythmic"].sel(id=fly_id).values)
            if "cwt_rhythmic" in ds.coords:
                row["CWT_Rhythmic"] = bool(ds["cwt_rhythmic"].sel(id=fly_id).values)

            summary_data.append(row)

        summary_df = pd.DataFrame(summary_data)
        # Alphabetical order for a predictable, GraphPad-friendly layout: by Group
        # then ID (or just ID when there is no group coord).
        _sort_keys = [c for c in ("Group", "ID") if c in summary_df.columns]
        if _sort_keys:
            summary_df = summary_df.sort_values(_sort_keys).reset_index(drop=True)
        st.dataframe(summary_df, width="stretch", height=250)

        csv_summary = summary_df.to_csv(index=False)
        st.download_button(
            "Download Period Summary CSV",
            csv_summary,
            "period_analysis_summary.csv",
            "text/csv",
            key="dl_period_summary",
        )

        # Optional: rhythmic-only variant, gated on user-selected algorithm
        _rhyth_cols = [
            c for c in ("LS_Rhythmic", "AC_Rhythmic", "CWT_Rhythmic") if c in summary_df.columns
        ]
        if _rhyth_cols:
            _gate = st.selectbox(
                "Filter rhythmic-only export by",
                options=_rhyth_cols,
                key="dl_period_summary_gate",
            )
            _filtered = summary_df[summary_df[_gate].astype(bool)]
            st.caption(f"Rhythmic-only: {len(_filtered)} of {len(summary_df)} flies.")
            st.download_button(
                "Download Period Summary (rhythmic only) CSV",
                _filtered.to_csv(index=False),
                "period_analysis_summary_rhythmic.csv",
                "text/csv",
                key="dl_period_summary_rhythmic",
            )

    st.divider()

    # ============================================================
    # Export Sleep Bout Data
    # ============================================================
    if analyses["sleep"] and "duration" in ds.data_vars:
        st.subheader("Export Sleep Bout Data (CSV)")

        bout_vars = ["duration"]
        if "start_time" in ds.data_vars:
            bout_vars.append("start_time")
        if "end_time" in ds.data_vars:
            bout_vars.append("end_time")
        bout_df = ds[bout_vars].to_dataframe().reset_index().dropna(subset=["duration"])

        st.write(f"{len(bout_df)} sleep bouts across {bout_df['id'].nunique()} flies")

        csv_bouts = bout_df.to_csv(index=False)
        st.download_button(
            "Download Sleep Bout CSV",
            csv_bouts,
            "sleep_bouts.csv",
            "text/csv",
            key="dl_sleep_bouts",
        )

    st.divider()

    # ============================================================
    # Export HMM Results
    # ============================================================
    if analyses["hmm"] and "hmm_state" in ds.data_vars:
        st.subheader("Export HMM State Assignments (CSV)")

        try:
            from hmm_models import load_hmm_config_from_attrs
        except ImportError:  # HMM module unavailable — degrade to no-op
            load_hmm_config_from_attrs = None
        _export_cfg = load_hmm_config_from_attrs(ds) if load_hmm_config_from_attrs else None
        if _export_cfg is not None:
            st.caption(
                f"HMM run parameters: **{_export_cfg.n_states}** states, "
                f"**{_export_cfg.emission_model}** emission, "
                f"**{_export_cfg.training_scope}** scope, "
                f"**{_export_cfg.transition_constraints}** transitions, "
                f"**{_export_cfg.decoding_method}** decoding"
            )

        hmm_state_df = ds["hmm_state"].to_pandas().reset_index()
        if hmm_state_df.columns[0] != "time":
            hmm_state_df.rename(columns={hmm_state_df.columns[0]: "time"}, inplace=True)

        st.write(
            f"State assignments: {hmm_state_df.shape[0]} timepoints x {hmm_state_df.shape[1] - 1} flies"
        )

        csv_hmm = hmm_state_df.to_csv(index=False)
        st.download_button(
            "Download HMM State Assignments CSV",
            csv_hmm,
            "hmm_state_assignments.csv",
            "text/csv",
            key="dl_hmm_states",
        )

        # HMM ZT-binned state fractions
        st.subheader("Export HMM ZT-Binned State Fractions (CSV)")
        hmm_export_bin = st.slider(
            "Bin size (minutes)",
            min_value=15,
            max_value=60,
            value=30,
            step=15,
            key="hmm_export_bin",
        )

        if st.button("Generate HMM ZT Fractions CSV", key="gen_hmm_zt"):
            with st.spinner("Computing ZT fractions..."):
                try:
                    from hmm_models import get_hmm_zt_fractions

                    zt_df = get_hmm_zt_fractions(ds, bin_size_minutes=hmm_export_bin)
                    if not zt_df.empty:
                        # Add group info
                        for fid in zt_df["id"].unique():
                            if "group" in ds.coords:
                                zt_df.loc[zt_df["id"] == fid, "group"] = str(
                                    ds.sel(id=fid).coords["group"].values
                                )
                        csv_zt = zt_df.to_csv(index=False)
                        st.download_button(
                            "Download HMM ZT Fractions CSV",
                            csv_zt,
                            "hmm_zt_fractions.csv",
                            "text/csv",
                            key="dl_hmm_zt_export",
                        )
                        st.success(f"Ready: {len(zt_df)} rows")
                    else:
                        st.warning("No ZT fraction data generated.")
                except ImportError:  # HMM module unavailable — degrade to no-op
                    st.info("HMM ZT-fraction export unavailable (HMM add-in not installed).")
                except Exception as e:
                    st.error(f"Error: {e}")
