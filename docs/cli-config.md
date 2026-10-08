# Running ClockWork from a config file

How the `clockwork` command and its YAML config are meant to work, and the
rules that keep the config, the GUI and the saved dataset from drifting apart.
Agreed 2026-09-30; this is the design the CLI is being built to.

`ARCHITECTURE.md` is the companion: its rules about the dataset (5, 6, 7, 8)
are what make a config file possible at all.

---

## What the CLI is for

**Explore in the GUI; record the choices; reproduce and batch from the CLI.**

Several of ClockWork's settings are decisions a person makes by looking — the
rhythmicity cutoff chosen on the cutoff tab, the number of HMM states chosen
after cross-validation, a curation that looked right on the heatmap. The GUI is
where the plots that inform those decisions live. A config file records the
decisions once they are made, so the same analysis can be re-run exactly, on
the next experiment, or on fifty experiments overnight.

Writing a config from scratch is supported, but it is not the main path. The
main path is the **Export settings** button.

---

## Where a config file comes from

One button, on the **Save & export** page, writes the settings of every
analysis that has **actually run** on the current dataset. Nothing else in the
GUI writes configs: settings buttons scattered across pages would each have to
decide what "the settings" are, and some would inevitably include analyses that
never ran.

The button reads the **dataset**, not the widgets. Every analysis records the
parameters that produced its result as attrs beside the result
(ARCHITECTURE rule 6), and `analysis_detection` already knows which analyses
ran. So:

```
GUI widgets ──► Config ──► run_step(ds, config) ──► results + config.to_attrs()
                                                              │
YAML  ◄── overrides only ◄── Config.from_attrs(ds) ◄──────────┘   (Export settings)
```

Consequences worth stating:

- An analysis that did not run cannot appear in the exported file.
- Settings export works on a `.nc` reloaded months later, with no session.
- If an analysis stops recording a parameter, the round trip breaks, and a test
  says so (below).

What is deliberately **not** exported: exploration tools that never write to the
dataset (the sleep-states CWT, HMM model selection, threshold sensitivity
sweeps). What they informed — the number of states, the cutoff — *is* recorded,
by the analysis that used it.

---

## The file

```yaml
clockwork_config: 1                 # schema version; old files get migrated, not broken
experiment: EXP240115               # names the output folder
extends: lab_defaults.yaml          # optional; this file's values win

inputs:                             # paths are relative to THIS file, not the terminal
  metadata: metadata.xlsx
  monitors: raw/                    # or instead:  dataset: saved.nc
  gap_threshold_hours: 1.0

groups:
  by: [genotype, pulse_time]
  keep: {genotype: [w1118, per0]}   # optional subset, by column value

curation:
  min_alive_days: 2
  rolling_window_hours: 24
  immobility_proportion: 0.01

split:
  gap_threshold_minutes: 60
  discard_first_dd_day: false

analyses:                           # an analysis that is absent does not run
  period: { ... }
  sleep: { ... }
  hmm: { ... }

outputs:
  dir: results/                     # the run writes into results/<experiment>/
  dataset: true
  tables: [period_summary, sleep_summary]   # default: every table there is
  qc_report: true
```

### Rules

1. **Overrides only.** A file lists what differs from the defaults and nothing
   else, so a short file stays readable. It is still exactly reproducible,
   because of rule 2.
2. **Every run writes its resolved config** — every value actually used,
   defaults included, plus the ClockWork and dependency versions — beside its
   outputs and into the saved dataset's attrs.
3. **Units are in every key name**: `_hours`, `_minutes`, `_seconds`. The code
   uses all three for similar-sounding things (the import gap threshold is in
   hours, the split's in minutes, the sleep threshold in seconds), so a bare
   `gap_threshold` would be a trap.
4. **Subsets are by column value**, never by the composite group label, which
   changes whenever the grouping does. Two forms:

   ```yaml
   keep: {genotype: [w1118, per0]}              # every listed value, per column
   keep:                                        # exact combinations, any of them
     - {genotype: w1118, temperature: 25C}
     - {genotype: per0,  temperature: 30C}
   ```

   The second exists because a hand-picked set of groups is not always a
   product of column values; it is what the GUI records. Keys are metadata
   column names, and values match by meaning, so `Monitor: [17]` and
   `Monitor: ["17"]` are the same.
5. **A missing upstream step is an error**, not a silent default. Asking for
   `hmm` without `sleep` fails validation and names the missing section: the
   sleep threshold changes every downstream number, so it must be a choice.
6. **One experiment per file.** Batches are `clockwork run a.yaml b.yaml ...`;
   settings shared across a lab go in a file named by `extends:`.
7. **Outputs are never overwritten** unless the command is given `--force`.
8. **YAML only** for writing. A JSON Schema is published for editors, which
   gives autocompletion and inline errors for YAML and JSON alike.
9. **Naming a section runs it.** `curation:` with nothing under it (or only
   comments) runs curation with its defaults; leaving the key out does not run
   it. YAML reads an empty entry as null, and without this rule a step the file
   names would silently not run. The same goes for each analysis and each
   period method. For the same reason the resolved config leaves out a step
   that did not run, rather than writing it as null.
10. **`extends:` adds and overrides; it cannot remove.** Mappings merge key by
   key and this file wins. Two exceptions, where merging would change the
   meaning: `groups.keep` is replaced whole, and naming any input source
   (`metadata`, `monitors` or `dataset`) replaces the base's source entirely.
11. **Key names are the contract.** Once released, a key's meaning never
   changes (ARCHITECTURE rule 3); a new meaning is a new key, and a renamed key
   is handled by the `clockwork_config` migration, not by breaking old files.

### Scope

Phase shift and sleep deprivation stay GUI-only at first; they need more
design thought than the core analyses. A config that names them fails
validation with a message saying so, rather than an unknown-key error.

---

## Commands

| command | does |
|---|---|
| `clockwork run config.yaml [...]` | run each file; write outputs, resolved config and QC report |
| `clockwork validate config.yaml` | check types and ranges, that files exist, that the metadata has the columns the requested analyses need, and that no step is missing its upstream |
| `clockwork init` | write a commented starting file, filled in from a real metadata file's column names |
| `clockwork schema` | print the JSON Schema |
| `clockwork gui` | the Streamlit app |

---

## Outputs

All in `<outputs.dir>/<experiment>/`; `experiment` defaults to the config
file's name.

- `<experiment>.nc` — the dataset, with the resolved config in its attrs
  (`clockwork_run_config`, JSON)
- `config.resolved.yaml` — every value used, defaults included, plus a
  `provenance` block (Python, platform, the versions of every package that can
  change a number). Loadable and runnable as it stands; `provenance` is ignored
  on reading.
- `tables/<name>.csv` — the tables the file asks for, or every one the analyses
  that ran can make: `period_summary`, `sleep_summary`, `sleep_states`,
  `sleep_bouts`, `hmm_occupancy`, `hmm_states`, `hmm_zt_fractions`. Time-of-day
  tables have one block of rows per LD/DD epoch. The Export page builds the
  period summary from the same function (`pipeline/tables.py`).
- `scalograms/` — each group's averaged CWT scalogram (PNG + CSV), when the
  `cwt` method sets `group_scalograms: true`; `scalogram_flies: rhythmic`
  (the default) averages only the flies autocorrelation calls rhythmic, so it
  needs autocorrelation's call (`validate` says so), and `all` averages every
  fly. Both settings are recorded on the dataset (`cwt_group_scalograms`,
  `cwt_scalogram_flies`), so Export settings carries the Period page's choice.
- `qc_report.html` — so an unattended run can be checked by eye afterwards: the
  curation heatmaps (kept and removed flies), the split's per-epoch record
  lengths, and for each analysis that ran the plot a person would have looked at
  before accepting it — the rhythmicity cutoff distributions, the sleep totals,
  HMM state occupancy. Self-contained (plotly.js embedded), so it opens offline.
- `run.log` — everything the analyses printed; the terminal shows one line per
  step instead (`--verbose` shows it all).

`clockwork run` checks the file first (as `validate` does) and refuses to start
on an error, and refuses an output folder that already has files in it before
doing any work.

Configurable figures are later work, not v1.

---

## How it is built

Each analysis gets one **config object** (a pydantic model) in
`clockwork/pipeline/`, and one **step** that takes a dataset and a config and
returns the dataset with the analysis's results and the config's attrs added.

- The GUI page builds the config from its widgets and calls the step.
- The CLI builds the config from the YAML and calls the same step.
- `Config.to_attrs()` writes the parameters under the analysis's existing attr
  names (ARCHITECTURE rule 1: the names already on saved datasets do not
  change); `Config.from_attrs(ds)` reads them back.

So there is one computation per analysis, one mapping between a parameter and
its recorded attr, and the page, the CLI and the settings export are three
callers of the same code.

### The round-trip test

For every analysis the CLI supports:

> run from a config → export settings from the resulting dataset → the exported
> config equals the one that went in.

That test is what makes the Export settings button trustworthy. An analysis that
records a parameter under the wrong name, forgets one, or reinterprets one fails
it.

---

## Status

- [x] Phase 0: installable package, `clockwork gui`, measured dependency ranges.
- [x] Phase 1: pipeline steps (`clockwork/pipeline/`).
  - [x] import → groups → curate → split (`pipeline/data.py`); the Data pages
        call them, and the subset is now recorded on the dataset (`subset_keep`).
  - [x] period & rhythmicity (`pipeline/period.py`): the four estimators, their
        rhythmic calls, and per-method overrides of the shared settings. Each
        method's preprocessing is now recorded on the master (`<method>_prep_*`);
        before, it never reached it.
  - [x] sleep (`pipeline/sleep.py`): detection and the short / intermediate /
        long boundaries, which the Sleep states page re-cuts on its own.
  - [x] HMM (`pipeline/hmm.py`): a published preset plus overrides of it, fitted
        on LD, DD or the whole recording. The GUI's "Both (separate)" is two
        single-phase fits, of which only the viewed one is on the dataset; a
        config describes what the dataset holds.
  - [x] Sleep states needs no step of its own: it re-cuts the sleep boundaries
        (the sleep step) and otherwise computes for display, which the CLI's
        `sleep_states` table and QC report cover.
  - [x] exports: the result tables are `pipeline/tables.py`, shared by the
        Export page and `clockwork run`; settings export is `pipeline/export.py`.
- [x] Phase 2: the config file (`pipeline/experiment.py`), `run` / `validate` /
      `init` / `schema` (`cli.py`), the tables and QC report
      (`pipeline/tables.py`, `pipeline/report.py`), and the **Export settings**
      tab on Save & export (`pipeline/export.py`). The round-trip test runs the
      whole chain on example_data: config → run → settings read off the saved
      `.nc` → YAML → the same config (`tests/test_cli_config.py`).
- [ ] Phase 3: PyPI and conda-forge.
- [x] CWT group-averaged scalograms from the CLI (`pipeline/scalograms.py`,
      shared with the Period page).
- [ ] Later: configurable figures.
