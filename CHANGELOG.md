# Changelog

## 0.1.0 — unreleased

The first release as an installable package, `clockwork-sci`, with a `clockwork`
command that runs the GUI or runs a whole analysis from a YAML file.

### Read this first: results that change

- **CWT periods and rhythmicity are different from earlier versions, and more
  accurate.** The wavelet is now tabulated at PyWavelets precision 16 and the
  transform uses FFT convolution in float64 (`periodograms.compute_cwt`). Earlier
  versions used PyWavelets' old built-in precision (10), which at ClockWork's
  scales on 1-minute data distorts the spectrum into a power floor that rises
  with period, so arrhythmic flies grew a spurious peak at the top of the search
  window. On the 192 example flies, against the flies Lomb-Scargle and
  autocorrelation agree on (108 rhythmic, 69 arrhythmic), the CWT call at the
  default cutoff now agrees on 177 of 177, against 70.6% before; the
  dsOpa1 + Ldh-mutant genotypes, previously called rhythmic at ~35 h on 29/32
  and 28/32 flies, are now 3/32 and 1/32, in line with Lomb-Scargle and
  autocorrelation. The default cutoff (2.0) stands; the measurement is in
  `core/calibrations.py`. The sleep-state CWT uses the same transform.
- **The GPU CWT is gone.** It computed in float32 and gave different numbers from
  the CPU path; there is now one implementation. (The GPU path for the HMM is
  kept, as the optional `gpu` extra.)
- **MESA stops with an error when its fit fails for every fly**, instead of
  returning no period for anybody. One fly's failed fit still just skips that fly.
- **A period run no longer reorders the flies** in the dataset or adds a
  `split_minute` coordinate to an unsplit one (BACKLOG 23).

Nothing published was computed with the earlier CWT numbers, which is why the
change was made rather than kept behind a switch.

### New: running from a config file

```bash
clockwork init my_experiment.yaml --metadata metadata.xlsx
clockwork validate my_experiment.yaml
clockwork run my_experiment.yaml
```

- **`clockwork run`** does import, groups, curation, the LD/DD split, period &
  rhythmicity, sleep and the HMM from one YAML file, and writes the analysed
  `.nc`, per-fly CSV tables (period summary, sleep totals per LD/DD, sleep
  states, sleep bouts, HMM occupancy, states and ZT fractions), CWT
  group-averaged scalograms when asked for, an offline QC report, the resolved
  config with the versions of every package that can change a number, and a log.
  It never overwrites an earlier run unless given `--force`, which replaces that
  run's outputs whole.
- **`clockwork validate`** checks a file without running it: misspelt settings,
  missing files, metadata columns the analyses need, and analyses whose upstream
  step is missing (the HMM needs sleep; sleep needs curation).
- **`clockwork init`** writes a commented starting file from your metadata's
  columns; **`clockwork schema`** prints a JSON Schema for editor completion.
- Config files list only what differs from the defaults, can share settings
  through `extends:`, and take paths relative to themselves. See
  [docs/cli-config.md](docs/cli-config.md).
- Phase shift and sleep deprivation are GUI-only for now.

### New in the GUI

- **Save & export → Settings (.yaml)** writes the config for every analysis that
  has run on the dataset, read off the dataset itself (so it works on a `.nc`
  saved long ago too). Run it with `clockwork run` to repeat the analysis, or
  point it at the next experiment.
- **Changing groups keeps your work.** A subset, a regroup or a reset on Groups &
  subsets used to throw away curation, the LD/DD split and every result. Now
  curation, the split and sleep are re-applied with their recorded settings and
  each fly's period results are kept; only results that describe whole groups
  (the HMM, CWT group averages, sleep deprivation, phase shift) are dropped, and
  the page says which (BACKLOG 22).
- **Group labels no longer depend on how the grouping was set.** `pulse_time` is
  stored as your metadata wrote it ("ZT15" stays "ZT15", and "ZT15" and "zt15"
  stay distinct), and Import, Redefine groups and the Import preview build
  labels the same way. Previously the same grouping read `dsmcherry-ZT21` from
  Import and `dsmcherry-21.0` from Redefine groups (BACKLOG 21).
- The CWT "group-averaged scalograms" choice is now recorded on the dataset, so
  it carries into exported settings.
- Each period method's preprocessing is recorded on the dataset (before, it was
  never saved), and the subset you chose is recorded by column value.

### Install

```bash
pip install clockwork-sci
clockwork gui
```

Python 3.11 or later. Image (PNG) export needs Chrome, which
`plotly_get_chrome` installs.

Dependency ranges were measured rather than pinned, by testing every dependency
at its minimum alone and together, and the newest versions, against the full
suite and an unrounded fingerprint of every estimator's output. The minimums
that this raised: astropy ≥ 7.2 (older fails to import beside NumPy 2.4),
xarray ≥ 2025.8 (older cannot save pandas 3's text columns), statsmodels ≥ 0.14.6
(older silently broke MESA beside pandas 3), PyWavelets ≥ 1.9 (the precision
setting), PyYAML ≥ 6.0.1. astropy 8 is excluded until its Lomb-Scargle changes
are reviewed.

### Compatibility

- **`.nc` files saved by earlier versions load as before.** The two pulse
  columns are renamed on load (`pulse_zt_hour` → `pulse_time`, written "ZT21"
  because those versions did not keep the original spelling;
  `pulse_duration_minutes` → `pulse_duration_min`).
- Periods and rhythmic calls stored in an older `.nc` were computed with the old
  CWT; re-run the CWT to get the new numbers.

### Known limitations

- Phase shift and sleep deprivation run only in the GUI.
- Figures from `clockwork run` are the QC report's; configurable publication
  figures are planned.
