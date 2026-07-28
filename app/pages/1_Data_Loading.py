"""
Data Loading Page - Raw DAM import, NetCDF loading, dataset combining.
"""

import os
import sys

import numpy as np
import pandas as pd
import streamlit as st
import xarray as xr

# Add core/ and app/ directories to the import path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CORE_DIR = os.path.join(PROJECT_ROOT, "core")
APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in [CORE_DIR, APP_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

import dam_processor
import dam_utilities
from analysis_detection import detect_analyses, format_status_summary
from dataset_meta import (
    PHASE_DD,
    PHASE_FULL,
    PHASE_LD,
    VALID_PHASES,
    dataset_phase,
    has_split_evidence,
    stamp_phase,
)
from load_and_save_datasets import load_dataset_from_netcdf

st.header("Data Loading")


def _invalidate_derived_caches():
    """Drop every session-state key whose contents are derived from the
    *current* set of flies in ``st.session_state.dataset``.

    Called both when a fresh dataset is loaded (via ``_clear_dataset_state``)
    and when the user applies/resets a group-subset filter — in either case
    every cached analysis below is keyed off fly id and would be stale.
    """
    keys_to_clear = [
        # Phase splits
        "dataset_DD",
        "dataset_LD",
        # Sleep / waveform / rebound
        "wf_df",
        "ip_df",
        "rb_df",
        "bout_timing_df",
        "activity_zt",
        # Period / rhythmicity caches
        "sleep_cwt_ds",
        "ultra_ls_ds",
        "circ_df",
        "rhythmicity_df",
        "rhythmicity_summary",
        "rhythmicity_per_fly_df",
        "rhythmicity_summary_df",
        # HMM
        "hmm_results",
        "hmm_config",
        "_last_hmm_preset",
        "cv_fold_df",
        "cv_summary_df",
        # Sleep deprivation
        "sd_results",
        "sd_config",
        # Parameter sweep page
        "param_sweep_results",
        # Preprocessing UI bits
        "show_preprocessing_heatmap",
        "_heatmap_var_last",
        "curated_dead_data",
    ]
    for k in keys_to_clear:
        if k in st.session_state:
            del st.session_state[k]
    # Per-group ultra_ls_ds_<group> entries are dynamically named — sweep them
    # by prefix so renaming/removing groups doesn't leave stragglers behind.
    for k in [
        k
        for k in list(st.session_state.keys())
        if isinstance(k, str) and k.startswith("ultra_ls_ds_")
    ]:
        del st.session_state[k]


def _clear_dataset_state():
    """Reset all dataset-related session state to defaults."""
    _invalidate_derived_caches()
    # Plus the keys that only get cleared on a fresh load, not on a filter:
    for k in ("dataset", "dataset_full", "dataset_path", "analyses", "_raw_metadata", "_raw_data"):
        if k in st.session_state:
            del st.session_state[k]
    # Re-initialize required defaults
    st.session_state.dataset = None
    st.session_state.dataset_full = None
    st.session_state.dataset_path = None
    st.session_state.dataset_DD = None
    st.session_state.dataset_LD = None
    st.session_state.analyses = {}
    st.session_state.working_dir = None


def _stash_full(ds):
    """Store the unfiltered dataset alongside the working ``dataset`` so a
    Reset button can restore the full set without re-reading from disk."""
    st.session_state.dataset_full = ds


def _apply_group_filter(selected_groups):
    """Subset ``dataset_full`` by group and replace ``dataset`` with the
    result. Clears every derived cache so downstream pages re-derive from
    the filtered dataset."""
    ds_full = st.session_state.dataset_full
    if ds_full is None or "group" not in ds_full.coords:
        return
    selected = {str(g) for g in selected_groups}
    mask = np.array([str(g) in selected for g in ds_full["group"].values])
    keep_ids = ds_full["id"].values[mask]
    filtered = ds_full.sel(id=list(keep_ids))
    _invalidate_derived_caches()
    st.session_state.dataset = filtered
    st.session_state.dataset_DD = None
    st.session_state.dataset_LD = None
    st.session_state.analyses = detect_analyses(filtered)


def _reset_group_filter():
    """Restore the unfiltered dataset and clear derived caches."""
    ds_full = st.session_state.get("dataset_full")
    if ds_full is None:
        return
    _invalidate_derived_caches()
    st.session_state.dataset = ds_full.copy()
    st.session_state.dataset_DD = None
    st.session_state.dataset_LD = None
    st.session_state.analyses = detect_analyses(st.session_state.dataset)


_REPLACE_WARNING = (
    "A dataset is already loaded ({n_flies} flies, {n_time} timepoints). "
    "Loading a new dataset will **discard all current data and analysis "
    "results**. To analyse a second dataset in parallel, open another "
    "browser tab."
)


tab_fresh, tab_netcdf, tab_combine = st.tabs(
    [
        "Path A: Fresh Start (Raw DAM Files)",
        "Path B: Load NetCDF",
        "Combine Datasets",
    ]
)

# ============================================================
# PATH A: Fresh Start
# ============================================================
with tab_fresh:
    st.subheader("Import Raw DAM Data")

    data_dir = st.text_input(
        "Data directory (folder containing MonitorXXX.txt files)",
        value=st.session_state.get("working_dir", ""),
        key="data_dir_input",
    )
    metadata_path = st.text_input(
        "Metadata file path (CSV or Excel)",
        value="",
        key="metadata_path_input",
    )
    gap_threshold = st.number_input("Gap threshold (hours)", min_value=0.1, value=1.0, step=0.5)

    with st.expander("Metadata file format"):
        st.markdown(
            "**Required columns:**\n"
            "- `start_datetime` — Must be **ZT0 (lights-on time)** for the experiment. "
            "This defines the start of ZT binning.\n"
            "- `stop_datetime` — End of the recording period to analyze.\n"
            "- `Monitor` — Monitor ID number (matches MonitorXXX.txt filenames).\n"
            "- `genotype` — Fly genotype label.\n"
            "- `temperature` — Temperature condition label.\n\n"
            "**Optional columns:**\n"
            "- `region_id` — Channel/tube number (1-32). Handled **per row**: leave "
            "it blank (or omit the column) to use the whole monitor (auto-expanded "
            "to all 32 tubes); or give a single number (`5`), an inclusive range "
            "(`1-16`), or a comma list (`1,3,5`) to select specific tubes. This lets "
            "you **split one monitor across genotypes** — e.g. one row with "
            "`region_id` `1-16` and another with `17-32`, each carrying its own "
            "genotype/drug — while every other monitor stays a single blank-region "
            "row. Overlapping ranges on the same monitor are flagged.\n"
            "- `first_DD_day` — **CT0 (subjective morning)** of the first full day "
            "of constant darkness to analyze. This is NOT the last lights-off time; "
            "it is the time that corresponds to what would have been lights-on on the "
            "first full DD day. Required for LD/DD splitting.\n\n"
            "All datetime values should be in a format pandas can parse "
            "(e.g. `2024-01-15 09:00:00`)."
        )

    if st.button("Load & Validate Data", key="load_raw"):
        if not data_dir or not metadata_path:
            st.error("Please provide both a data directory and a metadata file path.")
        elif not os.path.isdir(data_dir):
            st.error(f"Data directory not found: {data_dir}")
        elif not os.path.exists(metadata_path):
            st.error(f"Metadata file not found: {metadata_path}")
        else:
            # The WORKING FOLDER (where saved files / exports default) is the folder
            # that holds the METADATA file — the user's experiment folder — not the
            # monitor-data directory and not the app's launch directory. Absolute, so
            # a relative input can't later resolve against the app's cwd. Persisted
            # under a dedicated key so the Create Dataset step (which clears session
            # state) can restore it.
            _meta_dir = os.path.abspath(os.path.dirname(metadata_path))
            st.session_state.working_dir = _meta_dir
            st.session_state["_metadata_dir"] = _meta_dir
            load_progress = st.progress(0, text="Loading monitor files...")
            with st.spinner("Loading and validating data..."):
                try:

                    def _load_cb(completed, total):
                        load_progress.progress(
                            completed / total, text=f"Validating: monitor combo {completed}/{total}"
                        )

                    processor = dam_processor.MetadataProcessor(
                        metadata_path, data_dir, gap_threshold_hours=gap_threshold
                    )
                    metadata, all_data = processor.run(progress_callback=_load_cb)
                    st.session_state._raw_metadata = metadata
                    st.session_state._raw_data = all_data
                    st.success(
                        f"Loaded {len(all_data.columns)} channels, {len(all_data)} timepoints"
                    )
                    # Surface the data-integrity report (status rule + gaps). A
                    # status!=1 row is no-data -> NaN, never zero (§2a). Cosmetic
                    # rows (a real reading survived) are quiet info; DATA-LOSS
                    # holes (NaN, no valid reading) are shown prominently.
                    for _sev, _text in processor.integrity_summary_lines():
                        if _sev == "warning":
                            st.warning(_text)
                        elif _sev == "error":
                            st.error(_text)
                        else:
                            st.caption(_text)
                except Exception as e:
                    st.error(f"Error loading data: {e}")
                finally:
                    load_progress.empty()

    # Create Dataset
    if "_raw_metadata" in st.session_state and "_raw_data" in st.session_state:
        st.divider()

        metadata = st.session_state._raw_metadata
        has_dd_info = "first_DD_day" in metadata.columns

        if has_dd_info:
            st.info(
                "LD and DD phases detected (`first_DD_day` column present). "
                "The full dataset will be loaded now — LD/DD splitting is available "
                "in the **Preprocessing** page after curation."
            )

        # --- Group definition: which metadata columns define the comparison "group" ---
        # The chosen columns drive every group-level comparison (period plots, HMM,
        # exports) via the single `group` coord. Candidates auto-exclude id/file/monitor/
        # region and datetime columns (dam_utilities.group_defining_columns). Default =
        # genotype+temperature (the historical behavior).
        _grp_candidates = dam_utilities.group_defining_columns(metadata)
        _grp_default = [c for c in ("genotype", "temperature") if c in _grp_candidates]
        group_columns = _grp_default
        if _grp_candidates:
            group_columns = st.multiselect(
                "Group-defining metadata columns",
                options=_grp_candidates,
                default=_grp_default,
                key="group_columns_select",
                help="Which metadata factors define the comparison 'group' used throughout "
                "the analysis (e.g. add 'sex', or use genotype alone). Datetime, monitor, "
                "region and id columns are excluded automatically. "
                "Default: genotype + temperature.",
            )
            _labels = dam_utilities.derive_group_labels(metadata, group_columns)
            if _labels is None:
                st.caption("No group columns selected — all flies form a single group.")
            else:
                _uniq = sorted({str(v) for v in _labels.values})
                st.caption(
                    f"{len(_uniq)} group(s): "
                    + ", ".join(_uniq[:8])
                    + (" …" if len(_uniq) > 8 else "")
                )

        if st.button("Create Dataset", key="create_dataset"):
            # If a dataset already exists, ask for confirmation first
            if st.session_state.get("dataset") is not None:
                st.session_state["_pending_create_dataset"] = True
                st.rerun()
            else:
                # No existing dataset — proceed directly
                st.session_state["_pending_create_dataset"] = "confirmed"
                st.rerun()

        # Confirmation step (shows after the button click triggered a rerun)
        if st.session_state.get("_pending_create_dataset") is True:
            ds_old = st.session_state.dataset
            st.warning(
                _REPLACE_WARNING.format(
                    n_flies=len(ds_old["id"]),
                    n_time=len(ds_old["time"]),
                )
            )
            col_yes, col_no = st.columns(2)
            with col_yes:
                if st.button("Replace current dataset", key="confirm_create"):
                    st.session_state["_pending_create_dataset"] = "confirmed"
                    st.rerun()
            with col_no:
                if st.button("Cancel", key="cancel_create"):
                    st.session_state.pop("_pending_create_dataset", None)
                    st.rerun()

        # Actual dataset creation (runs after confirmation or when no dataset existed)
        if st.session_state.get("_pending_create_dataset") == "confirmed":
            st.session_state.pop("_pending_create_dataset", None)
            # Preserve raw data + the working folder across the clear since we need
            # them immediately. _clear_dataset_state() nulls working_dir, so without
            # restoring it here source_data_dir below is stamped '' and every export
            # falls back to the app's launch directory (the bug this fixes).
            _raw_meta = st.session_state._raw_metadata
            _raw_data = st.session_state._raw_data
            _meta_dir = st.session_state.get("_metadata_dir")
            _clear_dataset_state()
            st.session_state._raw_metadata = _raw_meta
            st.session_state._raw_data = _raw_data
            st.session_state.working_dir = _meta_dir

            with st.spinner("Creating xarray dataset..."):
                try:
                    full_data = dam_utilities.convert_to_relative_time(_raw_data, _raw_meta)
                    # group_columns: chosen above (persisted via the multiselect key);
                    # None/[] falls back to the historical genotype-temperature default.
                    ds = dam_utilities.create_xarray_dataset(
                        full_data,
                        _raw_meta,
                        group_columns=st.session_state.get("group_columns_select"),
                    )

                    if ds is None:
                        st.error("Failed to create dataset (empty time dimension).")
                    else:
                        # Raw CSV loads have no partitioning — full recording.
                        stamp_phase(ds, PHASE_FULL, split_applied=False)
                        # Record the working folder (the metadata file's directory)
                        # so every export defaults next to the user's experiment
                        # files. A plain string, so it survives the NetCDF round-trip
                        # (a reloaded .nc still exports to that folder when it exists).
                        ds.attrs["source_data_dir"] = st.session_state.get("working_dir") or ""
                        _stash_full(ds)
                        st.session_state.dataset = ds
                        st.session_state.analyses = detect_analyses(ds)

                        st.success(
                            f"Dataset created: {len(ds['id'])} flies, {len(ds['time'])} timepoints"
                        )
                        st.caption(
                            f"Working folder (where saved files & exports go): "
                            f"`{ds.attrs['source_data_dir']}` — the metadata file's "
                            f"directory. Each save also lets you edit the destination."
                        )
                        st.rerun()
                except Exception as e:
                    st.error(f"Error creating dataset: {e}")


# ============================================================
# PATH B: Load NetCDF
# ============================================================
with tab_netcdf:
    st.subheader("Load Existing NetCDF File")

    nc_path = st.text_input(
        "Path to NetCDF file (.nc)",
        value="",
        key="nc_path_input",
    )

    if st.button("Load NetCDF", key="load_nc"):
        if not nc_path:
            st.error("Please provide a file path.")
        elif not os.path.exists(nc_path):
            st.error(f"File not found: {nc_path}")
        elif st.session_state.get("dataset") is not None:
            # Dataset exists — ask for confirmation
            st.session_state["_pending_nc_path"] = nc_path
            st.rerun()
        else:
            # No existing dataset — proceed directly
            st.session_state["_pending_nc_path"] = nc_path
            st.session_state["_nc_confirmed"] = True
            st.rerun()

    # Confirmation step
    if st.session_state.get("_pending_nc_path") and not st.session_state.get("_nc_confirmed"):
        ds_old = st.session_state.dataset
        st.warning(
            _REPLACE_WARNING.format(
                n_flies=len(ds_old["id"]),
                n_time=len(ds_old["time"]),
            )
        )
        col_yes, col_no = st.columns(2)
        with col_yes:
            if st.button("Replace current dataset", key="confirm_nc"):
                st.session_state["_nc_confirmed"] = True
                st.rerun()
        with col_no:
            if st.button("Cancel", key="cancel_nc"):
                st.session_state.pop("_pending_nc_path", None)
                st.session_state.pop("_nc_confirmed", None)
                st.rerun()

    # Actual load (runs after confirmation or when no dataset existed)
    if st.session_state.get("_nc_confirmed") and st.session_state.get("_pending_nc_path"):
        _nc_path = st.session_state.pop("_pending_nc_path")
        st.session_state.pop("_nc_confirmed", None)
        _clear_dataset_state()
        with st.spinner("Loading NetCDF..."):
            try:
                ds = load_dataset_from_netcdf(_nc_path)

                # Resolve canonical phase metadata. Three cases:
                #   1. File already carries 'phase' attr → use it.
                #   2. File carries legacy 'split_phase' attr → migrate.
                #   3. No phase metadata + no split evidence → default
                #      silently to 'full' (plain unsplit recording).
                #   4. Ambiguous evidence (rare) → defer the load and
                #      prompt the user to declare via a dropdown.
                _existing_phase = ds.attrs.get("phase")
                if isinstance(_existing_phase, str) and _existing_phase in VALID_PHASES:
                    stamp_phase(
                        ds,
                        _existing_phase,
                        split_applied=ds.attrs.get("split_applied", _existing_phase != PHASE_FULL),
                    )
                elif isinstance(ds.attrs.get("split_phase"), str):
                    _legacy = ds.attrs["split_phase"]
                    _resolved = (
                        PHASE_LD
                        if _legacy == PHASE_LD
                        else PHASE_DD
                        if _legacy == PHASE_DD
                        else PHASE_FULL
                    )
                    stamp_phase(
                        ds, _resolved, split_applied=(_resolved != PHASE_FULL or _legacy == "both")
                    )
                elif has_split_evidence(ds):
                    # Ambiguous — defer load until the user declares.
                    st.session_state["_nc_ambiguous_ds"] = ds
                    st.session_state["_nc_ambiguous_path"] = _nc_path
                    st.rerun()
                else:
                    # No evidence anywhere → silently treat as full.
                    stamp_phase(ds, PHASE_FULL, split_applied=False)

                _stash_full(ds)
                st.session_state.dataset = ds
                st.session_state.dataset_path = _nc_path
                st.session_state.working_dir = os.path.abspath(os.path.dirname(_nc_path))

                analyses = detect_analyses(ds)
                st.session_state.analyses = analyses

                st.success(
                    f"Loaded: {len(ds['id'])} flies, "
                    f"{len(ds['time'])} timepoints (phase={dataset_phase(ds)})"
                )

                st.subheader("Detected Analyses")
                st.text(format_status_summary(analyses))
                st.rerun()
            except Exception as e:
                st.error(f"Error loading NetCDF: {e}")

    # Ambiguous-evidence dropdown — triggered when a file shows split
    # evidence but lacks canonical phase metadata.
    if st.session_state.get("_nc_ambiguous_ds") is not None:
        st.warning(
            "This NetCDF appears to be a partitioned LD/DD file but doesn't "
            "carry a canonical `phase` attribute. Please declare the phase "
            "so downstream pages can branch correctly."
        )
        _amb_phase = st.selectbox(
            "Phase of the loaded file",
            options=[PHASE_FULL, PHASE_LD, PHASE_DD],
            format_func=lambda p: {
                PHASE_FULL: "Full recording (no LD/DD split)",
                PHASE_LD: "LD-only partition",
                PHASE_DD: "DD-only partition",
            }[p],
            key="_nc_ambiguous_phase_pick",
        )
        if st.button("Confirm phase and load", key="confirm_ambiguous"):
            ds = st.session_state.pop("_nc_ambiguous_ds")
            _nc_path = st.session_state.pop("_nc_ambiguous_path")
            stamp_phase(ds, _amb_phase, split_applied=(_amb_phase != PHASE_FULL))
            _stash_full(ds)
            st.session_state.dataset = ds
            st.session_state.dataset_path = _nc_path
            st.session_state.working_dir = os.path.abspath(os.path.dirname(_nc_path))
            st.session_state.analyses = detect_analyses(ds)
            st.success(f"Loaded: {len(ds['id'])} flies (phase={_amb_phase})")
            st.rerun()


# ============================================================
# COMBINE DATASETS
# ============================================================
with tab_combine:
    st.subheader("Combine Multiple Datasets")
    st.markdown(
        "Select 2+ NetCDF files to concatenate along the `id` dimension. "
        "Shared group values automatically merge. Shorter datasets are padded with NaN."
    )

    nc_paths_text = st.text_area(
        "NetCDF file paths (one per line)",
        height=120,
        key="combine_paths",
    )
    combined_name = st.text_input(
        "Combined file name (without .nc extension)",
        value="combined_dataset",
        key="combined_name",
    )

    if st.button("Combine Datasets", key="combine_btn"):
        paths = [p.strip() for p in nc_paths_text.strip().split("\n") if p.strip()]
        if len(paths) < 2:
            st.error("Please provide at least 2 file paths.")
        else:
            missing = [p for p in paths if not os.path.exists(p)]
            if missing:
                st.error(f"Files not found: {missing}")
            elif st.session_state.get("dataset") is not None:
                st.session_state["_pending_combine"] = True
                st.rerun()
            else:
                st.session_state["_pending_combine"] = "confirmed"
                st.rerun()

    # Confirmation step
    if st.session_state.get("_pending_combine") is True:
        ds_old = st.session_state.dataset
        st.warning(
            _REPLACE_WARNING.format(
                n_flies=len(ds_old["id"]),
                n_time=len(ds_old["time"]),
            )
        )
        col_yes, col_no = st.columns(2)
        with col_yes:
            if st.button("Replace current dataset", key="confirm_combine"):
                st.session_state["_pending_combine"] = "confirmed"
                st.rerun()
        with col_no:
            if st.button("Cancel", key="cancel_combine"):
                st.session_state.pop("_pending_combine", None)
                st.rerun()

    # Actual combine (runs after confirmation or when no dataset existed)
    if st.session_state.get("_pending_combine") == "confirmed":
        st.session_state.pop("_pending_combine", None)
        paths = [p.strip() for p in nc_paths_text.strip().split("\n") if p.strip()]
        _clear_dataset_state()
        combine_progress = st.progress(0, text="Loading files...")
        with st.spinner("Loading and combining datasets..."):
            try:
                datasets = []
                for i, p in enumerate(paths):
                    combine_progress.progress(
                        (i + 1) / len(paths),
                        text=f"Loading file {i + 1}/{len(paths)}: {os.path.basename(p)}",
                    )
                    ds = load_dataset_from_netcdf(p)
                    datasets.append(ds)
                    st.write(f"  Loaded {p}: {len(ds['id'])} flies")

                # Check for duplicate fly IDs across datasets
                all_ids = []
                for ds in datasets:
                    all_ids.extend(ds["id"].values.tolist())
                id_counts = pd.Series(all_ids).value_counts()
                duplicates = id_counts[id_counts > 1]

                if len(duplicates) > 0:
                    st.warning(
                        f"Found {len(duplicates)} duplicate IDs. Appending suffix to disambiguate."
                    )
                    seen_ids = set()
                    for ds_idx, ds in enumerate(datasets):
                        new_ids = []
                        for fly_id in ds["id"].values:
                            original_id = str(fly_id)
                            resolved_id = original_id
                            suffix_counter = 1
                            while resolved_id in seen_ids:
                                suffix_char = chr(ord("a") + suffix_counter)
                                resolved_id = f"{original_id}_{suffix_char}"
                                suffix_counter += 1
                            seen_ids.add(resolved_id)
                            new_ids.append(resolved_id)
                        datasets[ds_idx] = ds.assign_coords(id=new_ids)

                combined = xr.concat(datasets, dim="id")

                # Combined datasets inherit no clear partitioning from
                # their constituents; stamp 'full' unless every input
                # carried the same non-full phase.
                _phases_in = {dataset_phase(d) for d in datasets}
                if len(_phases_in) == 1 and PHASE_FULL not in _phases_in:
                    _combined_phase = next(iter(_phases_in))
                    stamp_phase(combined, _combined_phase, split_applied=True)
                else:
                    stamp_phase(combined, PHASE_FULL, split_applied=False)

                _stash_full(combined)
                st.session_state.dataset = combined
                st.session_state.analyses = detect_analyses(combined)

                out_dir = os.path.dirname(paths[0])
                out_path = os.path.join(out_dir, f"{combined_name}.nc")

                st.success(
                    f"Combined dataset: {len(combined['id'])} total flies. "
                    f"Navigate to Export to save."
                )
                st.session_state.dataset_path = out_path
                st.rerun()
            except Exception as e:
                st.error(f"Error combining datasets: {e}")
            finally:
                combine_progress.empty()


# ============================================================
# Current dataset summary (always shown at bottom)
# ============================================================
if st.session_state.dataset is not None:
    st.divider()
    st.subheader("Current Dataset Summary")
    ds = st.session_state.dataset
    ds_full = st.session_state.get("dataset_full")

    # ---- Active-filter status badge ----------------------------------
    if ds_full is not None and len(ds["id"]) < len(ds_full["id"]) and "group" in ds_full.coords:
        full_groups = {str(g) for g in ds_full["group"].values}
        cur_groups = {str(g) for g in ds["group"].values} if "group" in ds.coords else set()
        st.warning(
            f"**Group filter active** — using **{len(ds['id'])} of "
            f"{len(ds_full['id'])} flies**, "
            f"**{len(cur_groups)} of {len(full_groups)} groups**. "
            "Use *Reset to all groups* below to restore the full set."
        )

    col1, col2, col3 = st.columns(3)
    col1.metric("Flies", len(ds["id"]))
    col2.metric("Timepoints", len(ds["time"]))
    if "group" in ds.coords:
        col3.metric("Groups", len(set(ds["group"].values)))
    st.dataframe(
        pd.DataFrame(
            {
                "ID": ds["id"].values,
                "Group": ds["group"].values if "group" in ds.coords else "N/A",
            }
        ),
        width="stretch",
        height=200,
    )

    # ============================================================
    # Group selection — subset the dataset before any downstream
    # page sees it. Reversible via the Reset button below; the
    # unfiltered dataset is preserved in `dataset_full`.
    # ============================================================
    if ds_full is not None and "group" in ds_full.coords:
        with st.expander("Group selection (subset for downstream analyses)"):
            st.markdown(
                "Pick which groups to keep. Applying a selection **drops the "
                "other flies from the working dataset** and clears every "
                "cached analysis result, so downstream pages re-compute on "
                "the subset only. Click *Reset* to restore the full set."
            )

            # Per-group fly counts from the unfiltered dataset.
            full_groups_arr = np.asarray([str(g) for g in ds_full["group"].values])
            group_counts = (
                pd.Series(full_groups_arr)
                .value_counts()
                .sort_index()
                .rename_axis("group")
                .reset_index(name="n_flies")
            )
            cur_groups_set = {str(g) for g in ds["group"].values} if "group" in ds.coords else set()
            group_counts["currently_kept"] = group_counts["group"].isin(cur_groups_set)
            st.dataframe(group_counts, width="stretch", height=180)

            all_group_options = group_counts["group"].tolist()
            # Default the picker to whatever is currently kept; first-time
            # users see all groups selected.
            default_selection = sorted(cur_groups_set) if cur_groups_set else all_group_options
            selected = st.multiselect(
                "Groups to keep",
                options=all_group_options,
                default=default_selection,
                key="group_filter_select",
            )

            # Live preview of the selection's effect.
            preview_n = (
                int(group_counts.loc[group_counts["group"].isin(selected), "n_flies"].sum())
                if selected
                else 0
            )
            st.caption(
                f"Will keep **{preview_n} / {len(ds_full['id'])} flies** "
                f"({len(selected)} / {len(all_group_options)} groups)."
            )

            col_apply, col_reset = st.columns(2)
            with col_apply:
                apply_disabled = len(selected) == 0
                if st.button(
                    "Apply selection",
                    key="apply_group_filter",
                    disabled=apply_disabled,
                    help=(
                        "Pick at least one group"
                        if apply_disabled
                        else "Subset the dataset and clear cached analyses."
                    ),
                ):
                    _apply_group_filter(selected)
                    st.success(
                        f"Filter applied — {preview_n} flies across "
                        f"{len(selected)} groups. Cached analyses were cleared."
                    )
                    st.rerun()
            with col_reset:
                reset_disabled = len(ds["id"]) == len(ds_full["id"])
                if st.button(
                    "Reset to all groups",
                    key="reset_group_filter",
                    disabled=reset_disabled,
                    help=(
                        "No filter active"
                        if reset_disabled
                        else "Restore the full unfiltered dataset."
                    ),
                ):
                    _reset_group_filter()
                    st.success("Restored full dataset. Cached analyses were cleared.")
                    st.rerun()
