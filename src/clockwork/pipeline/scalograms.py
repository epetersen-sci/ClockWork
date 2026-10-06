"""CWT group-averaged scalograms: one PNG + CSV per group, written to disk.

A CWT run can average each group's scalogram (period x time x power), optionally
over only the flies autocorrelation calls rhythmic. ``periodograms.wavelet_analysis``
returns the arrays; this writes them, so the analysis never decides where files
go. The Period page (into ``Averaged Scalograms/`` in the working folder) and
``clockwork run`` (into ``scalograms/`` in the run's output folder) both call it.
"""

from __future__ import annotations

import datetime
import json
import os
import re


def save_group_average_scalograms(group_averages, out_dir, ds=None, *, timestamp=True):
    """Write one PNG + CSV per group-averaged scalogram, and return the manifest.

    ``ds``, if given, gets the manifest stamped onto its attrs, so the record of
    where the files went travels with the dataset.

    ``timestamp`` adds the time to each file name, so re-running on the Period
    page never overwrites an earlier average. A CLI run has a folder of its own,
    and plain names are easier to find there.
    """
    import pandas as pd

    from clockwork.core import plotting

    os.makedirs(out_dir, exist_ok=True)
    suffix = "_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") if timestamp else ""
    saved = []
    for avg in group_averages:
        # Group labels come from user metadata and end up in filenames.
        safe = re.sub(r"[^A-Za-z0-9._-]+", "_", str(avg["group"])).strip("_") or "group"
        stem = f"averaged_scalogram_{safe}_{avg['phase_label']}{suffix}"
        png_path = os.path.join(out_dir, stem + ".png")
        csv_path = os.path.join(out_dir, stem + ".csv")

        plotting.save_group_average_scalogram_png(
            avg["mean_power"],
            avg["period_axis"],
            avg["time_h"],
            group_label=str(avg["group"]),
            n_flies=avg["n_flies"],
            out_png_path=png_path,
            period_range=avg["period_range"],
            phase_label=avg["phase_label"],
        )
        # CSV: rows are periods (h), columns are time (h).
        frame = pd.DataFrame(avg["mean_power"], index=avg["period_axis"], columns=avg["time_h"])
        frame.index.name = "period_h"
        frame.columns.name = "time_h"
        frame.to_csv(csv_path)

        saved.append(
            {
                "group": str(avg["group"]),
                "n": avg["n_flies"],
                "png": png_path,
                "csv": csv_path,
                "phase": avg["phase_label"],
            }
        )

    if saved and ds is not None:
        ds.attrs["cwt_group_average_paths"] = json.dumps(saved)
        ds.attrs["cwt_group_average_dir"] = out_dir
    return saved
