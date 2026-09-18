# How a ClockWork page is allowed to behave

Ten rules for adding an analysis without breaking the ones already there.

Every rule carries the bug that earned it. That is deliberate: a rule without a
failure attached gets argued with, worked around, or quietly dropped by whoever
finds it inconvenient. The ones here each cost someone a day.

`BACKLOG.md` is the index of those bugs; `<n>` below is an item number in it.

---

## The shape

```
raw DAM files  ─┐
                ├─►  the CORE dataset  ─►  analysis  ─►  analysis  ─► figures, exports
metadata file  ─┘     (Import)              (a page)     (a page)
```

**The core** is what Import builds: per-minute activity from the monitor files,
plus everything the metadata says about each fly. It is built once and every
other page reads it.

**An analysis** is a page that reads the core, computes something, and writes the
result back onto the dataset under names it owns.

Everything below is about keeping the arrows pointing one way.

---

## 1. The core is append-only

An analysis may not change the meaning, the name, the dtype or the units of
anything the core already records. It **may** cause the core to record something
new, and that is not a loophole — it is the pressure valve that stops rules 2 and
3 from forcing the interpretation into a page.

> **Earned by:** the grouping bug (`7c5dbc5`). `attrs['group_columns']` recorded
> metadata COLUMN names; two of those columns are stored as coords under
> different names. Pages needed the coord names. The fix added
> `group_coord_names` *beside* `group_columns` and left the latter
> byte-identical. Had "no changes to import" been absolute, the alternative was a
> rename table copied into every page — worse, and unreviewable.

**Test for whether a change is append-only:** an older `.nc` still loads and every
page still means the same thing by every name it already used.

---

## 2. Derivation belongs to the first page that needs it — but lives in `core/`

Import records what the metadata *says*. Turning that into what an analysis
*needs* is the analysis's business, and should happen the first time somebody
asks for it.

But the derivation itself goes in `core/` as an idempotent `add_*(ds)` that
attaches the variable if it is missing and returns the dataset unchanged if it is
not. Not in the page. If it lives in the page, the second page that needs it
copies the logic, and pages are the least stable layer in the app.

`add_phase_metadata` and `add_pulse_metadata` are the pattern. Copy them.

> **The one that got away:** `pulse_zt_hour` is derived at Import, while
> `pulse_minute` — same feature, one step further on — is derived on demand. Two
> policies for one feature is how a reader ends up unsure which layer owns a
> number. New work follows `add_pulse_metadata`.

---

## 3. A derived variable has exactly one meaning

Once `pulse_zt_hour` is "the ZT hour parsed from `pulse_time`, as a float, NaN
when unpulsed", that is what it is for every page, forever. A page that wants a
different interpretation computes a **differently named** variable.

Reinterpreting a name in place is the worst failure available here, because
nothing errors. Every downstream number silently becomes a different quantity,
and the figures still render.

**If you need a variant:** name it for the variant, not for the concept —
`pulse_zt_hour_from_actual_lights_on`, not "a better `pulse_zt_hour`".

---

## 4. "Group" means the grouping chosen at Import

`ds['group']` is the labels built from the columns ticked on the Import page,
recorded in `attrs['group_columns']` (what was ticked) and
`attrs['group_coord_names']` (where to read them). A page that wants different
groups computes them locally, under a local name, and does not write them back.

**One sanctioned exception:** "Redefine groups" on Groups & subsets
(`regroup_dataset`, item 12) does rewrite `group` — as an explicit user action,
on the page that owns the question, clearing every derived cache as it goes.
Regrouping is a thing the user does, never a thing an analysis does in passing.

**A page that draws one panel per group defaults to `ds['group']`** —
`ui.filters.group_by_options`. Re-deriving the grouping from its parts is how the
Actograms page ended up drawing a different partition than the rest of the app
while looking correct.

> **Earned by:** `7c5dbc5`. A dataset grouped by genotype + pulse time + pulse
> duration drew actograms grouped by genotype alone. Nothing failed; the page was
> simply answering a different question. Note the author: this was introduced by
> the commit that was meant to *fix* the grouping default, not by an outside
> contribution. Rules aimed only at incoming work would not have caught it.

---

## 5. Everything a page needs lives on the dataset, not in session state

Session state does not survive a `.nc` reload, and a reloaded `.nc` is a
first-class way to use this app. If a page needs a fact, that fact is on the
dataset — as a coord, a variable, or an attr.

> **Earned twice.** Item 12: `group_defining_columns` asked its question of a
> metadata DataFrame, which a reloaded `.nc` never had, so regrouping was
> impossible after a reload — hence `group_defining_coords`, which asks the
> dataset. Item 5: the `dataset_LD` / `dataset_DD` session caches were deleted
> because they were absent on a fresh load and four files had to keep them in
> sync; consumers derive phase views on demand instead.

Corollary: **the dataset is the only thing worth saving.** If a result exists
only in `st.session_state`, it does not really exist.

---

## 6. An analysis owns a name prefix, and stores its parameters beside its results

Pick a prefix and put every variable and attr you write under it. Record the
parameters that produced the result as attrs, next to the result.

Without the prefix, two analyses writing `duration` collide silently. Without the
parameters, a figure six months old cannot be reproduced or even interpreted —
`sleep_threshold_seconds` is the difference between "these flies slept a lot" and
a number.

`sleep_threshold_seconds`, `cwt_method`, `phase_shift_filter_hours` already do
this. Follow them.

---

## 7. Re-running an analysis REPLACES its output, and drops it first

There is one result per analysis. Re-running with new settings overwrites the old
one; nothing is versioned and nothing accumulates. Downstream work always means
the latest run.

The rule is about **ordering**: drop your previous variables *before* computing,
not by merging over them afterwards.

> **Earned by:** item 15 (`604d8a4`). Re-running sleep analysis on a reloaded
> `.nc` merged the new bout table into a dataset still carrying the old run's
> `sleep_bout_number` dimension. That took the cartesian product of `time` and
> `sleep_bout_number` — a 134x row blow-up that ran for minutes and *then* died
> on a duplicate index. Worth remembering as one bug, not two: a slow step that
> subsequently fails is usually the same fault.

---

## 8. Declare what you depend on, and invalidate what follows you

Analyses legitimately depend on each other — sleep states needs sleep analysis;
phase shift needs curation and the phase boundary. Isolation is the wrong goal.
The goal is that nothing keeps serving a result computed from data that has since
changed.

- Read the dependency through `ui.guards.require_analysis`, so a missing one is a
  message naming the page that produces it, not a crash.
- Key every `@st.cache_data` helper on `dataset_meta.dataset_fingerprint`, so a
  changed cohort is a cache miss.
- Anything that changes the cohort calls `ui.state.invalidate_derived_caches`.

> **Earned by:** item 14. Every cached helper took its fingerprint as `_fp`, and
> Streamlit's underscore rule is syntactic — it drops *any* leading-underscore
> parameter from the key. So the one argument encoding which flies were in the
> dataset never reached the cache key. `rhythmicity._build_period_summary_df` had
> an empty key: the exported per-fly table was computed once per session and
> returned unchanged for every later dataset. **The fingerprint parameter takes no
> leading underscore; the dataset parameter does.**

---

## 9. Missing is never zero

A gap in the record, an out-of-phase minute and a fly that did not move are three
different things. Zero is a measurement. Use NaN — or the documented sentinel for
an integer variable, which in the sleep masks is `-1`, never `0`.

This is the rule well-meaning code breaks most often, because zero-filling makes
the arrays line up and the plots come out.

> **Earned by:** item 16, among others. A masking bug kept ~869 minutes per
> trimmed fly that the trim exists to discard, and gave every other fly a
> spurious missing marker. Also why `core/actograms.py` refuses to zero-pad a
> partial final day the way SCAMP's `actogram2.m` does: a flat floor claims the
> flies were still.

---

## 10. Pages render; `core/` computes

A plotting function takes computed frames and returns a figure. It does not run
the analysis it is drawing. The tell is a deferred import inside a plotting
function to dodge a circular one — that means the seam is in the wrong place.

The practical cost of breaking this is not architectural tidiness: it is that the
figure and the CSV exported next to it become two separate computations that can
disagree.

> **Earned by:** item 7. `sleep_bout_duration_lines` used to take `ds` and
> defer-import three `sleep_analysis` functions to compute the curves, the
> per-fly summary and the group test — a deferred import that existed only to
> dodge a circular one, which is the signal. Its docstring still records the
> diagnosis. It took `ds` and was the sole caller of all three, so the analysis
> effectively lived inside the plotting module.
>
> It nearly happened again during the 2026-09 merge, in a renderer that
> summarised its own input behind a deferred import — caught only because the
> reviewer knew item 7. `tests/test_plotting_seam.py` enforces it now; **add new
> analysis modules to its `ANALYSIS_MODULES` set**, because that miss was
> possible only because the set did not yet name the module.

---

## When a rule is in the way

Say so, in the commit, with what it cost to follow it. These are load-bearing but
not sacred — rule 1 exists precisely because an absolute version of "don't touch
the core" would have forced a worse fix. What is not acceptable is working around
a rule silently, because the next reader then has two mental models of the app
and no way to tell which one is current.

New rules go here the same way the existing ones did: with the bug attached.
