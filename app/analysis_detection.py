"""
Utility module to detect completed analyses from xarray Dataset variable names.
"""

import xarray as xr


def detect_analyses(ds: xr.Dataset) -> dict:
    """
    Inspect data_vars in a loaded xarray Dataset to determine which analyses
    have already been completed.

    Parameters
    ----------
    ds : xr.Dataset
        The loaded dataset to inspect.

    Returns
    -------
    dict
        Keys are analysis names, values are booleans indicating completion.
        Example: {'sleep': True, 'cwt': False, 'lomb_scargle': True, ...}
    """
    var_names = set(ds.data_vars)

    return {
        "preprocessing": "is_alive" in var_names,
        "sleep": "sleep" in var_names,
        "cwt": any(v.startswith("cwt_") for v in var_names),
        "lomb_scargle": any(v.startswith("ls_") for v in var_names),
        "autocorrelation": any(v.startswith("ac_") for v in var_names),
        "hmm": any(v.startswith("hmm_") for v in var_names),
        # SD records its parameters in ds.attrs (sd_day_number, sd_start_zt_minutes,
        # …), NOT as a data_var — so detect completion from the attrs, not var_names.
        "sleep_deprivation": any(str(k).startswith("sd_") for k in ds.attrs),
        # Phase shift also records only its parameters in attrs (same reasoning as SD).
        "phase_shift": any(str(k).startswith("phase_shift_") for k in ds.attrs),
        "sleep_states": "sleep_long" in var_names,
    }


def get_status_label(completed: bool) -> str:
    """Return a human-readable status string."""
    return "Completed" if completed else "Not run"


def format_status_summary(analyses: dict) -> str:
    """
    Format analysis detection results into a readable summary string.

    Parameters
    ----------
    analyses : dict
        Output of detect_analyses().

    Returns
    -------
    str
        Multi-line summary of analysis status.
    """
    display_names = {
        "preprocessing": "Preprocessing (curation)",
        "sleep": "Sleep Analysis",
        "cwt": "CWT (Continuous Wavelet Transform)",
        "lomb_scargle": "Lomb-Scargle Periodogram",
        "autocorrelation": "Autocorrelation",
        "hmm": "HMM (Hidden Markov Model)",
        "sleep_deprivation": "Sleep Deprivation",
        "phase_shift": "Phase Shift (light pulse)",
        "sleep_states": "Sleep State Classification",
    }

    lines = []
    for key, completed in analyses.items():
        name = display_names.get(key, key)
        status = get_status_label(completed)
        lines.append(f"  {name}: {status}")
    return "\n".join(lines)
