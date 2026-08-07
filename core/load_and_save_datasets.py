"""
load_and_save_datasets.py
=========================
NetCDF serialization and deserialization of xarray Datasets, plus CSV/Excel export helpers.

NetCDF does not natively support Python datetime objects or booleans in attributes.
save_dataset_to_netcdf() converts these to storable types and records the original
types so that load_dataset_from_netcdf() can restore them faithfully.

Functions
---------
save_dataset_to_netcdf()      — Save xarray Dataset to .nc with type-safe attribute handling
load_dataset_from_netcdf()    — Load .nc and restore datetime/boolean attributes
save_data_to_csv()            — Save one variable to a wide CSV (time × id)
save_data_to_excel()          — Save one variable to Excel (time × id)
export_activity_to_csv()      — Save raw activity data with time as first column
export_averaged_data_to_csv() — Save ZT-binned group averages (mean, SEM, n) to CSV
"""

import os
import warnings

import numpy as np
import pandas as pd
import xarray as xr

import dam_utilities


def save_dataset_to_netcdf(
    ds: xr.Dataset, output_filename: str, output_dir: str = None, engine="netcdf4"
):
    """
    Save an xarray Dataset to a NetCDF file.

    Handles attributes that NetCDF cannot store natively:
      - datetime objects → ISO 8601 strings (with a type-metadata attribute for restoration)
      - boolean values   → integers (0/1)

    Parameters
    ----------
    ds : xr.Dataset
    output_filename : str
        File name (e.g., 'analyzed_dataset.nc').
    output_dir : str, optional
        Directory to write into; joined with output_filename if provided.
    engine : str
        netCDF4 engine (default 'netcdf4').
    """
    print(f"Attempting to save Dataset to {output_filename}...")
    ds_to_save = ds.copy()
    attrs_to_add_original_type_info = {}

    for attr_name, attr_value in ds_to_save.attrs.items():
        if isinstance(attr_value, (bool, np.bool_)):
            ds_to_save.attrs[attr_name] = int(attr_value)
            attrs_to_add_original_type_info[attr_name] = "bool"
            print(f"Converted attribute '{attr_name}' (bool) to int.")
        elif isinstance(attr_value, pd.Timestamp):
            ds_to_save.attrs[attr_name] = attr_value.isoformat()
            attrs_to_add_original_type_info[attr_name] = "pd.Timestamp"
            print(f"Converted attribute '{attr_name}' (pd.Timestamp) to string.")
        elif isinstance(attr_value, np.ndarray) and attr_value.dtype == "datetime64[ns]":
            ds_to_save.attrs[attr_name] = [pd.Timestamp(dt).isoformat() for dt in attr_value]
            attrs_to_add_original_type_info[attr_name] = "np.datetime64_array"
            print(f"Converted attribute '{attr_name}' (np.datetime64 array) to list of strings.")
        elif isinstance(attr_value, (list, tuple)):
            if all(isinstance(x, (pd.Timestamp, np.datetime64)) for x in attr_value):
                ds_to_save.attrs[attr_name] = [pd.Timestamp(x).isoformat() for x in attr_value]
                attrs_to_add_original_type_info[attr_name] = "list_of_datetimes"
                print(f"Converted attribute '{attr_name}' (list of datetimes) to list of strings.")
            elif any(isinstance(x, (pd.Timestamp, np.datetime64)) for x in attr_value):
                print(
                    f"Warning: Attribute '{attr_name}' has mixed types including datetimes. Converting datetimes only."
                )
                ds_to_save.attrs[attr_name] = [
                    pd.Timestamp(x).isoformat()
                    if isinstance(x, (pd.Timestamp, np.datetime64))
                    else x
                    for x in attr_value
                ]

    # Second pass to catch any remaining booleans
    for attr_name, attr_value in list(ds_to_save.attrs.items()):
        if isinstance(attr_value, (bool, np.bool_)):
            ds_to_save.attrs[attr_name] = int(attr_value)
            attrs_to_add_original_type_info[attr_name] = "bool"
            print(f"Converted attribute '{attr_name}' (bool) to int.")

    # Store type metadata so load_dataset_from_netcdf() can restore original types
    for attr_name, original_type_str in attrs_to_add_original_type_info.items():
        ds_to_save.attrs[f"_{attr_name}_original_type"] = original_type_str

    if output_dir is not None:
        output_filename = os.path.join(output_dir, output_filename)
    try:
        ds_to_save.to_netcdf(output_filename, engine=engine)
    except Exception as e:
        # The Dataset is the single source of truth. A silent save
        # failure — no file written, no signal — is worse than any attribute issue,
        # so RAISE rather than print-and-continue. Callers must learn the save failed.
        raise RuntimeError(f"Failed to save Dataset to {output_filename}: {e}") from e
    print(f"Successfully saved Dataset to {output_filename}")


def load_dataset_from_netcdf(filepath: str) -> xr.Dataset:
    """
    Load an xarray Dataset from a NetCDF file and restore original attribute types.

    Reads the type-metadata attributes written by save_dataset_to_netcdf() to
    convert ISO strings back to datetime objects and integers back to booleans.

    Parameters
    ----------
    filepath : str

    Returns
    -------
    xr.Dataset
    """
    print(f"Attempting to load Dataset from {filepath}...")
    ds_loaded = xr.open_dataset(filepath)
    print(f"Successfully loaded Dataset from {filepath}")

    attrs_to_remove = []

    for attr_name, attr_value in list(ds_loaded.attrs.items()):
        if attr_name.startswith("_") and attr_name.endswith("_original_type"):
            # Strip leading '_' and trailing '_original_type' to get the real attribute name
            original_attr_name = attr_name[1 : -len("_original_type")]
            original_type_str = attr_value

            if original_attr_name in ds_loaded.attrs:
                current_attr_value = ds_loaded.attrs[original_attr_name]
                print(f"Restoring '{original_attr_name}' to {original_type_str}")

                if original_type_str == "bool":
                    try:
                        ds_loaded.attrs[original_attr_name] = bool(current_attr_value)
                    except Exception as e:
                        print(f"Could not restore '{original_attr_name}' to bool: {e}")
                elif original_type_str == "pd.Timestamp":
                    try:
                        ds_loaded.attrs[original_attr_name] = pd.to_datetime(current_attr_value)
                    except Exception as e:
                        print(f"Could not restore '{original_attr_name}' to pd.Timestamp: {e}")
                elif original_type_str in ("np.datetime64_array", "list_of_datetimes"):
                    try:
                        vals = current_attr_value
                        # netCDF stores a 1-element list attribute as a SCALAR on read,
                        # so a saved [Timestamp] comes back as a bare ISO string. Wrap any
                        # scalar back into a list before parsing — otherwise the
                        # comprehension iterates the STRING character-by-character
                        # (pd.to_datetime("2") raises) and the datetime attr is silently lost.
                        if isinstance(vals, (str, bytes)) or np.ndim(vals) == 0:
                            vals = [vals]
                        restored_list = [pd.to_datetime(s) for s in vals]
                        if original_type_str == "np.datetime64_array":
                            ds_loaded.attrs[original_attr_name] = np.array(
                                restored_list, dtype="datetime64[ns]"
                            )
                        else:
                            ds_loaded.attrs[original_attr_name] = restored_list
                    except Exception as e:
                        print(
                            f"Could not restore '{original_attr_name}' to datetime array/list: {e}"
                        )

            attrs_to_remove.append(attr_name)

    for attr_to_remove in attrs_to_remove:
        del ds_loaded.attrs[attr_to_remove]

    # Defense in depth: NetCDF normally yields numpy, but any pandas/Arrow-backed
    # remnant would break .sel/.isel — normalize before returning to the app.
    return dam_utilities.ensure_numpy_backed(ds_loaded)


def save_data_to_csv(ds: xr.Dataset, value_col: str, save_path: str):
    """
    Save one variable from the dataset to a wide CSV (time × id).

    Parameters
    ----------
    ds : xr.Dataset
    value_col : str
        Variable name to export.
    save_path : str
        Output file path.
    """
    if value_col not in ds.data_vars:
        raise ValueError(f"Variable '{value_col}' not found in the dataset.")
    df = ds[value_col].to_pandas()
    df.to_csv(save_path)


def save_data_to_excel(ds: xr.Dataset, value_col: str, save_path: str):
    """
    Save one variable from the dataset to an Excel file (time × id).

    Parameters
    ----------
    ds : xr.Dataset
    value_col : str
        Variable name to export.
    save_path : str
        Output file path (.xlsx).
    """
    if value_col not in ds.data_vars:
        raise ValueError(f"Variable '{value_col}' not found in the dataset.")
    df = ds[value_col].to_pandas()
    df.to_excel(save_path)


def export_activity_to_csv(ds: xr.Dataset, output_path: str, activity_var: str = "activity"):
    """
    Export raw activity data for every fly to a CSV file.

    The CSV has 'time' as the first column and one column per fly ID.

    Parameters
    ----------
    ds : xr.Dataset
    output_path : str
    activity_var : str
        Name of the activity variable (default 'activity').
    """
    if activity_var not in ds.data_vars:
        raise ValueError(
            f"Activity variable '{activity_var}' not found in the dataset. "
            f"Available: {list(ds.data_vars.keys())}"
        )
    activity_df = ds[activity_var].to_pandas().reset_index()
    if activity_df.columns[0] != "time":
        activity_df.rename(columns={activity_df.columns[0]: "time"}, inplace=True)
    activity_df.to_csv(output_path, index=False)
    print(f"Successfully exported activity data to {output_path}")
    print(f"  Shape: {activity_df.shape[0]} rows × {activity_df.shape[1]} columns")
    print(f"  Number of flies: {len(activity_df.columns) - 1}")


def export_averaged_data_to_csv(
    ds: xr.Dataset,
    output_path: str,
    value_col: str = "activity",
    bin_size_minutes: int = 30,
    bin_function: str = "mean",
    group_cols: list = None,
):
    """
    Export ZT-binned group-averaged data to a CSV file.

    Column format: zt_bin_minute, zt_hours, then for each group:
      <group>, SEM_<group>, n_<group>

    This exports the same aggregated data shown in the daily pattern plots.

    Parameters
    ----------
    ds : xr.Dataset
    output_path : str
    value_col : str
        Variable to bin and aggregate (default 'activity').
    bin_size_minutes : int
        ZT bin width in minutes (default 30).
    bin_function : str
        'mean' or 'sum' (default 'mean').
    group_cols : list, optional
        Columns to use as group identifier (default ['genotype', 'temperature']).
    """
    if group_cols is None:
        group_cols = ["genotype", "temperature"]

    if value_col not in ds.data_vars:
        raise ValueError(
            f"Variable '{value_col}' not found in the dataset. "
            f"Available: {list(ds.data_vars.keys())}"
        )

    binned_df = dam_utilities.get_zt_binned_dataframe(ds, value_col, bin_size_minutes, bin_function)

    if binned_df.empty:
        raise ValueError(f"No binned data available for variable '{value_col}'.")

    # Map each fly ID to a group label
    id_to_group = {}
    for fly_id in binned_df["id"].unique():
        try:
            fly_ds = ds.sel(id=fly_id)
            group_parts = []
            for col in group_cols:
                if col in ds.coords:
                    group_parts.append(str(fly_ds[col].item()))
                else:
                    warnings.warn(
                        f"Grouping column '{col}' not found in dataset coordinates. Skipping.",
                        stacklevel=2,
                    )
            id_to_group[fly_id] = "-".join(group_parts) if group_parts else "All Flies"
        except (KeyError, IndexError):
            id_to_group[fly_id] = "All Flies"

    binned_df["group"] = binned_df["id"].map(id_to_group)

    agg_df = (
        binned_df.groupby(["zt_bin_minute", "group"])
        .agg(
            mean=(value_col, "mean"),
            sem=(
                value_col,
                lambda x: x.std(ddof=1) / (len(x.dropna()) ** 0.5) if len(x.dropna()) > 1 else 0,
            ),
            n_flies=(value_col, lambda x: x.notna().sum()),
        )
        .reset_index()
    )

    agg_df["zt_hours"] = dam_utilities.zt_bin_to_hours(agg_df["zt_bin_minute"], bin_size_minutes)
    unique_groups = sorted(agg_df["group"].unique())

    result_df = agg_df[["zt_bin_minute", "zt_hours"]].drop_duplicates().sort_values("zt_bin_minute")

    for group in unique_groups:
        group_data = agg_df[agg_df["group"] == group][
            ["zt_bin_minute", "mean", "sem", "n_flies"]
        ].copy()
        result_df = result_df.merge(
            group_data[["zt_bin_minute", "mean"]].rename(columns={"mean": group}),
            on="zt_bin_minute",
            how="left",
        )
        result_df = result_df.merge(
            group_data[["zt_bin_minute", "sem"]].rename(columns={"sem": f"SEM_{group}"}),
            on="zt_bin_minute",
            how="left",
        )
        result_df = result_df.merge(
            group_data[["zt_bin_minute", "n_flies"]].rename(columns={"n_flies": f"n_{group}"}),
            on="zt_bin_minute",
            how="left",
        )

    column_order = ["zt_bin_minute", "zt_hours"]
    for group in unique_groups:
        column_order.extend([group, f"SEM_{group}", f"n_{group}"])

    result_df = result_df[column_order]
    result_df.to_csv(output_path, index=False)
    print(f"Successfully exported binned {value_col} data to {output_path}")
    print(
        f"  {result_df.shape[0]} bins × {len(unique_groups)} groups | bin size: {bin_size_minutes} min"
    )
