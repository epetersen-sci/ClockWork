"""Home — what ClockWork is, how the modules fit together, and what has been run."""

import streamlit as st

from ui.status import render_status_grid

st.markdown("**Drosophila Activity Monitor (DAM) analysis pipeline**")

st.markdown("""
### Getting started

Load your data under **Data → Import**, then work down the sidebar. Every module adds
its results to one shared dataset that you can save and reload at any point.

The sidebar groups the modules into four sections.

**Data** — build the dataset every other module reads.

- **Import** — bring in raw Trikinetics `MonitorXXX.txt` files with a metadata table,
  reload a saved `.nc`, or combine several `.nc` files. This is also where you choose
  which metadata columns define your comparison groups.
- **Groups & subsets** — see what you loaded, and reversibly narrow the dataset to a
  subset of groups.
- **Curate & split** — flag dead flies and trim their records, then divide each record
  into its light–dark (LD) and constant-darkness (DD) epochs.

**Circadian analysis** — free-running period and light-pulse responses.

- **Period analysis** — per-fly period from four independent estimators:
  autocorrelation, Lomb–Scargle, CWT and MESA.
- **Periodograms** — group-averaged spectra for whichever of those you ran.
- **Rhythmicity** — the per-fly summary table, an interactive threshold explorer, and
  the rhythmic/arrhythmic classification that gates downstream group filtering.
- **Phase shift** — how far each fly's rhythm shifted after a light pulse
  (needs a `pulse_time` ZT column in your metadata).

**Sleep & activity** — sleep structure and state.

- **Sleep analysis** — the 5-minute immobility rule, with each bout classified short,
  intermediate or long. Everything below depends on this.
- **Sleep & activity** — daily activity and sleep profiles, bout-duration curves,
  sleep-state totals and day/night summaries, with group comparisons and CSV export.
- **Sleep deprivation** — baseline sleep against post-deprivation recovery (rebound).
- **HMM model selection** — cross-validate the state count and emission model *before*
  committing to a full run.
- **HMM analysis** — Hidden Markov Model sleep/wake state classification.

**Export** — save the dataset as NetCDF and write CSV summaries, or export into the
SCAMP MATLAB toolbox format.
""")

ds = st.session_state.get("dataset")

if ds is None:
    st.info("No dataset loaded. Go to **Data → Import** to get started.")
else:
    st.success(f"Dataset loaded: {len(ds['id'])} flies, {len(ds['time'])} timepoints")

    from analysis_detection import detect_analyses

    analyses = detect_analyses(ds)
    st.session_state.analyses = analyses

    cols = st.columns(3)
    cols[0].metric("Flies", len(ds["id"]))
    cols[1].metric("Timepoints", len(ds["time"]))
    groups = ds["group"].values if "group" in ds.coords else []
    cols[2].metric("Groups", len(set(groups)))

    st.subheader("Analysis status")
    st.caption("Green = results are on the dataset. Grey = not run yet.")
    render_status_grid(analyses)
