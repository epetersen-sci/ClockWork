# scamp_export

Export curated, LD/DD-split Pythomics datasets into the legacy text format
that the SCAMP MATLAB toolbox (`scamp.m`) ingests, so the lab's existing
sleep/circadian analyses keep running on clean data.

## What this does

End-to-end: raw Trikinetics `MonitorXXX.txt` files + metadata → automated
dead-fly curation → LD/DD phase separation → equal-length, ZT-aligned
per-fly files that drop straight into SCAMP's two-folder load prompt.

Pythomics handles the parts that are tedious or error-prone in SCAMP
(automated dead-fly detection; per-fly first_DD_day splitting). SCAMP keeps
running the sleep/circadian analyses the lab already trusts.

## How to use it

### Standalone CLI

```
python -m scamp_export.curate_and_export \
    --metadata path/to/metadata.csv \
    --data-folder path/to/monitors/ \
    --out path/to/scamp_ready/ \
    --prefix EXP240115 \
    --min-alive-days 2 \
    --time-window 24 \
    --prop-immobile 0.01 \
    --gap-threshold 60 \
    --discard-first-dd-day \
    --ld-min-days 2 --dd-min-days 3 \
    --lights-on-military 900
```

Run from the repository root so the package is importable.

To export an already-prepared NetCDF (e.g. saved by the Streamlit app's
Preprocessing page) without re-curating:

```
python -m scamp_export.curate_and_export --nc analyzed_dataset_LD.nc --out scamp_ready/ --phase LD
```

### Streamlit page

Once Preprocessing has populated `session_state.dataset_LD` /
`session_state.dataset_DD`, the **SCAMP Export** page (sidebar entry 12)
exposes the same parameters and writes the files to a chosen folder.

## Output layout

```
<out>/
    LD/
        1min/   EXP240115M42C1   EXP240115M42C2  ...
        30min/  EXP240115M42C1   EXP240115M42C2  ...
        scamp_group_key.csv          board/channel ↔ fly_id ↔ genotype/group
        export_manifest.json         all parameters + per-board counts
    DD/
        1min/  ...   30min/  ...   scamp_group_key.csv   export_manifest.json
```

Filenames follow SCAMP's `<prefix>M<board>C<channel>` convention with **no
extension**. The prefix must not contain `C`/`c` or `m` (SCAMP's `dam_names.m`
splits board/channel on `M`/`C` in **both** cases with order-fragile logic, so a
stray `c`/`m` makes it mis-parse and silently drop the file), nor `.`, `/`, `\`,
or spaces. When the dataset combines multiple recording dates,
each fly's `YYYYMMDD` is appended to the prefix automatically, so each
`(date, Monitor)` pair shows up as a distinct SCAMP "board".

## Per-board common-window selection

SCAMP's `dam_read_names.m` aborts the load if files in one board differ on
`(start, int, len)`. To satisfy this without padding the equal-length
constraint with fabricated data:

1. Drop flies whose valid span is below `--ld-min-days` / `--dd-min-days`.
2. `window = min(valid span)` over the retained flies.
3. Floor to whole 1440-minute days, so SCAMP day-binning and 30-min binning
   are both clean.
4. Slice each fly from its first valid index for that window.

Small interior NaN gaps inside the window are linearly interpolated in
Python before writing (so SCAMP's `dam_cleanup.m` never has to act on
in-window gaps).

## File format (`dam_load/dam_file.m`)

```
Line 1   : header        (free text; we record phase, id, monitor, region, date)
Line 2   : len           (integer: number of data points)
Line 3   : int           (integer: sampling interval in minutes — 1 or 30)
Line 4   : start         (integer: military time HHMM)
Lines 5..: data          (integer per line; negative = missing/error)
```

`dam_cleanup.m` interpolates interior negatives and chops leading negatives
globally across all flies in a board — that global leading-chop is the
reason this exporter never pads short flies with `-1` to a union window.

## Import verified against real SCAMP (2026-07-24)

The **import path** was checked by running SCAMP's own MATLAB loader
(`dam_read` → `dam_read_names` → `dam_file` → `dam_cleanup` → full `dam_load`,
R2021b) on files produced by this exporter. Confirmed to load correctly: the luc
file format, `<prefix>M<board>C<channel>` filename parsing, **sparse/
non-contiguous channels**, the per-board equal-`(start, int, len)` constraint,
both the 1-min and 30-min folders, and the `-1` missing-sentinel →
`dam_cleanup.m` interpolation. (The unit tests in `tests/` check the writer
against a Python re-implementation of `dam_file.m`; this MATLAB run is the first
check against the real SCAMP code.)

One thing still needs the interactive GUI (see below): for a board where some
channels were dropped in curation, `scamp.m` draws all 32 channel checkboxes
defaulted **ON** — **uncheck the dropped channels** before analyzing. Do *not*
pad the missing channels with `-1`: `dam_cleanup.m` errors on an all-negative
channel.

## MATLAB acceptance procedure (manual)

This is the real validation gate. From MATLAB:

1. `>> scamp` to launch the GUI.
2. At the "Select 1-minute folder" prompt, choose `<out>/LD/1min`.
3. At the "Select 30-minute folder" prompt, choose `<out>/LD/30min`.
4. Confirm the board list shows the expected `<prefix>M<board>` entries
   with the expected channel counts.
5. Run a sleep analysis (`sleepcalc`) and an actogram pass; compare counts
   and sleep totals against the Pythomics activity heatmap for the same
   flies/days.
6. Repeat for DD.
7. Use `<out>/<phase>/scamp_group_key.csv` to enter group labels in
   SCAMP's GUI (the luc format has no group field).
