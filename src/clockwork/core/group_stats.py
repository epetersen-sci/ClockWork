"""Group comparisons behind the figure statistics (see docs/stats-inventory.md).

Every test runs on one value per fly, across the groups one figure (or one facet
panel) shows. Two families:

* ``"welch"`` — Welch ANOVA, then Games-Howell; Welch's t-test for two groups.
* ``"rank"`` — Kruskal-Wallis, then Dunn's test with Holm correction;
  Mann-Whitney U for two groups.

The arithmetic is statsmodels, scipy and scikit-posthocs. This module picks the
test, runs the post-hoc test only when the figure shows it, and lays the result
out for the figure (:func:`clockwork.core.plotting.add_group_stats`) and for the
``Stats`` sheet of the export (:func:`stats_table`).

What a figure shows, by the number of groups ``k`` tested:

* ``k == 2``: a bracket, labelled with stars when p < 0.05 and with the p-value
  otherwise.
* ``k == 3``: a bracket on every pair, labelled the same way, when the omnibus
  test is significant; otherwise the omnibus p-value.
* ``k >= 4``: a compact letter display when the omnibus test is significant;
  otherwise the omnibus p-value.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

WELCH = "welch"
RANK = "rank"
RECOMMENDED = "recommended"
#: Counts rather than per-fly values: the share of flies called rhythmic.
PROPORTION = "proportion"

#: The sidebar / CLI setting: ``recommended`` uses each measure's own family.
SETTINGS = {
    RECOMMENDED: "Recommended per measure",
    WELCH: "Welch ANOVA + Games-Howell for everything",
    RANK: "Kruskal-Wallis + Dunn for everything",
}

ALPHA = 0.05
#: Groups with fewer flies than this are left out of the test.
MIN_N = 3

_NAMES = {
    WELCH: ("Welch ANOVA", "Games-Howell", "Welch's t-test"),
    RANK: ("Kruskal-Wallis", "Dunn (Holm)", "Mann-Whitney U"),
    PROPORTION: ("χ² test", "Fisher's exact (Holm)", "Fisher's exact test"),
}

#: Monte Carlo resamples (fixed seed) for Fisher-Freeman-Halton on tables
#: larger than 2 x 2.
FFH_RESAMPLES = 99_999
FFH_SEED = 0

LETTERS_TEST = "Compact letter display"


def resolve_family(recommended, setting=RECOMMENDED):
    """The family a measure uses under ``setting``."""
    if setting in (WELCH, RANK):
        return setting
    if recommended not in (WELCH, RANK):
        raise ValueError(f"Unknown test family {recommended!r}")
    return recommended


def procedure_name(family, n_groups):
    """``"Welch ANOVA + Games-Howell"``, or the two-group test's name."""
    omnibus, posthoc, two = _NAMES[family]
    return two if n_groups == 2 else f"{omnibus} + {posthoc}"


def notice_text(family, n_groups, unit="one value per fly"):
    """The one line shown under every tested figure."""
    return (
        f"{procedure_name(family, n_groups)}, {unit} — check this test suits your "
        "data before relying on the result."
    )


def combined_notice(results):
    """One notice line for every test in ``results`` (a figure, or a set of
    panels shown together). None when no test ran."""
    done = [r for r in results if r is not None]
    if not done:
        return None
    names = list(dict.fromkeys(r.procedure for r in done))
    units = list(dict.fromkeys(r.unit for r in done))
    return (
        f"{' / '.join(names)}, {' / '.join(units)} — check this test suits your "
        "data before relying on the result."
    )


def stars(p):
    """``*`` p < 0.05, ``**`` p < 0.01, ``***`` p < 0.001; empty otherwise."""
    if not np.isfinite(p) or p >= ALPHA:
        return ""
    return "***" if p < 0.001 else "**" if p < 0.01 else "*"


def format_p(p):
    """``p = 0.213``, or ``p < 0.001``."""
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


def pair_label(p):
    """A bracket's label: stars when significant, the p-value otherwise."""
    return stars(p) or format_p(p)


@dataclass
class GroupStats:
    """One comparison across the groups of one figure or panel."""

    family: str
    groups: list
    test: str
    statistic: float
    df: tuple
    pvalue: float
    posthoc: str | None = None
    #: ``[(group_a, group_b, p)]`` — the post-hoc p-values (adjusted where the
    #: post-hoc test adjusts), or the two-group test's p for ``k == 2``.
    pairs: list = field(default_factory=list)
    #: ``{group: letters}`` when a compact letter display is shown.
    letters: dict | None = None
    measure: str | None = None
    panel: str | None = None
    unit: str = "one value per fly"
    #: Where the result is drawn, for figures not laid out by group (the
    #: phase response curve's pulse time and series, say).
    key: dict = field(default_factory=dict)
    #: The test's name in the notice, when it is not the family's own.
    name: str | None = None

    @property
    def k(self):
        return len(self.groups)

    @property
    def significant(self):
        return bool(self.pvalue < ALPHA)

    @property
    def procedure(self):
        """The test as named in the notice: ``"Welch ANOVA + Games-Howell"``."""
        if self.name:
            return self.name
        if self.k == 2:
            return _NAMES[self.family][2]
        return f"{self.test} + {self.posthoc}" if self.posthoc else self.test

    @property
    def notice(self):
        return combined_notice([self])

    @property
    def marks(self):
        """What the figure draws: ``"brackets"``, ``"letters"`` or ``"omnibus"``."""
        if self.k == 2 or (self.k == 3 and self.significant):
            return "brackets"
        if self.k >= 4 and self.significant and self.letters:
            return "letters"
        return "omnibus"

    @property
    def omnibus_label(self):
        """``Kruskal-Wallis p = 0.340`` — shown when no pairs are marked."""
        return f"{self.test} {format_p(self.pvalue)}"


def _clean(values_by_group, order):
    keys = [str(g) for g in (order if order is not None else values_by_group)]
    by_str = {str(g): v for g, v in values_by_group.items()}
    out = {}
    for g in keys:
        if g in out or g not in by_str:
            continue
        vals = pd.to_numeric(pd.Series(np.asarray(by_str[g]).ravel()), errors="coerce")
        vals = vals[np.isfinite(vals)].to_numpy(dtype=float)
        if len(vals) >= MIN_N:
            out[g] = vals
    return out


def _long(samples):
    return pd.DataFrame(
        {
            "group": np.concatenate([[g] * len(v) for g, v in samples.items()]),
            "value": np.concatenate(list(samples.values())),
        }
    )


def _posthoc(samples, family):
    """The pairwise p-value matrix, rows and columns in ``samples`` order."""
    import scikit_posthocs as sp

    long = _long(samples)
    if family == WELCH:
        mat = sp.posthoc_games_howell(long, val_col="value", group_col="group")
    else:
        mat = sp.posthoc_dunn(long, val_col="value", group_col="group", p_adjust="holm")
    names = list(samples)
    return mat.reindex(index=names, columns=names)


def _letters(matrix):
    import scikit_posthocs as sp

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            letters = sp.compact_letter_display(matrix, alpha=ALPHA)
    except (ValueError, RuntimeWarning):
        return None
    return {str(g): str(s).replace(" ", "") for g, s in letters.items()}


def compare_groups(values_by_group, family, *, order=None, measure=None, panel=None):
    """Compare ``{group: per-fly values}`` with ``family``.

    Groups keep ``order`` (default: the dict's order); non-finite values are
    dropped and groups with fewer than :data:`MIN_N` flies are left out. Returns
    None when fewer than two groups remain or the test cannot be computed (every
    value identical, say).
    """
    from scipy import stats as _stats

    if family not in (WELCH, RANK):
        raise ValueError(f"Unknown test family {family!r}")
    samples = _clean(values_by_group, order)
    if len(samples) < 2:
        return None
    names = list(samples)
    omnibus, posthoc_name, two_name = _NAMES[family]
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            if len(samples) == 2:
                a, b = samples.values()
                if family == WELCH:
                    res = _stats.ttest_ind(a, b, equal_var=False)
                    stat, p, df = float(res.statistic), float(res.pvalue), (float(res.df),)
                else:
                    res = _stats.mannwhitneyu(a, b, alternative="two-sided")
                    stat, p, df = float(res.statistic), float(res.pvalue), ()
                if not np.isfinite(p):
                    return None
                return GroupStats(
                    family=family,
                    groups=names,
                    test=two_name,
                    statistic=stat,
                    df=df,
                    pvalue=p,
                    pairs=[(names[0], names[1], p)],
                    measure=measure,
                    panel=panel,
                )
            if family == WELCH:
                from statsmodels.stats.oneway import anova_oneway

                res = anova_oneway(list(samples.values()), use_var="unequal")
                stat, p = float(res.statistic), float(res.pvalue)
                df = (float(res.df[0]), float(res.df[1]))
            else:
                res = _stats.kruskal(*samples.values())
                stat, p, df = float(res.statistic), float(res.pvalue), (float(len(samples) - 1),)
    except ValueError:
        return None
    if not np.isfinite(p):
        return None
    out = GroupStats(
        family=family,
        groups=names,
        test=omnibus,
        statistic=stat,
        df=df,
        pvalue=p,
        posthoc=posthoc_name,
        measure=measure,
        panel=panel,
    )
    if out.significant:
        matrix = _posthoc(samples, family)
        out.pairs = [
            (names[i], names[j], float(matrix.iloc[i, j]))
            for i in range(len(names))
            for j in range(i + 1, len(names))
        ]
        if len(names) >= 4:
            out.letters = _letters(matrix)
    return out


def compare_frame(df, value_col, group_col, family, *, order=None, measure=None, panel=None):
    """:func:`compare_groups` on a per-fly frame (one row per fly)."""
    if df is None or df.empty or value_col not in df.columns:
        return None
    groups = df[group_col].astype(str)
    if order is None:
        order = sorted(groups.unique())
    values = {str(g): df.loc[groups == str(g), value_col] for g in order}
    return compare_groups(values, family, order=order, measure=measure, panel=panel)


def per_panel(per_fly, value_col, panels, family, *, id_col="ID", measure=None):
    """One :class:`GroupStats` (or None) per facet panel, over that panel's own
    flies and compare levels — the flies :func:`plotting.faceted_violins` draws."""
    out = []
    if per_fly is None or per_fly.empty or value_col not in per_fly.columns:
        return [None] * len(panels)
    ids = per_fly[id_col].astype(str)
    for p in panels:
        lab = p.labels()
        sub = per_fly[ids.isin(lab)]
        levels = ids[ids.isin(lab)].map(lab).astype(str)
        values = {str(lvl): sub.loc[levels == str(lvl), value_col] for lvl in p.levels}
        out.append(
            compare_groups(
                values, family, order=[str(v) for v in p.levels], measure=measure, panel=p.title
            )
        )
    return out


def compare_proportions(counts, *, order=None, measure=None):
    """Compare the share of flies with a property across groups.

    ``counts`` is ``{group: (n_with, n_total)}``. Two groups: Fisher's exact
    test. Three or more: the χ² test of independence, or Fisher-Freeman-Halton
    (Monte Carlo, :data:`FFH_RESAMPLES` tables, fixed seed) when an expected
    count is under 5; then pairwise Fisher's exact tests, Holm-adjusted, when the
    omnibus test is significant. Groups with fewer than :data:`MIN_N` flies are
    left out; None when fewer than two groups remain, or every fly is alike.
    """
    from scipy import stats as _stats
    from statsmodels.stats.multitest import multipletests

    keys = [str(g) for g in (order if order is not None else counts)]
    by_str = {str(g): v for g, v in counts.items()}
    table = {
        g: (int(by_str[g][0]), int(by_str[g][1]) - int(by_str[g][0]))
        for g in keys
        if g in by_str and int(by_str[g][1]) >= MIN_N
    }
    if len(table) < 2:
        return None
    names = list(table)
    mat = np.array([table[g] for g in names])
    if mat[:, 0].sum() == 0 or mat[:, 1].sum() == 0:
        return None
    omnibus, posthoc_name, two_name = _NAMES[PROPORTION]
    unit = "counts of flies per group"
    if len(names) == 2:
        p = float(_stats.fisher_exact(mat).pvalue)
        return GroupStats(
            family=PROPORTION, groups=names, test=two_name, statistic=float("nan"), df=(),
            pvalue=p, pairs=[(names[0], names[1], p)], measure=measure, unit=unit,
        )
    chi = _stats.chi2_contingency(mat)
    test, stat, df, p = omnibus, float(chi.statistic), (float(chi.dof),), float(chi.pvalue)
    if (chi.expected_freq < 5).any():
        try:
            res = _stats.fisher_exact(
                mat,
                method=_stats.MonteCarloMethod(
                    n_resamples=FFH_RESAMPLES, rng=np.random.default_rng(FFH_SEED)
                ),
            )
        except (AttributeError, TypeError, ValueError):
            pass  # scipy < 1.15 has no R x C Fisher test; the χ² result stands
        else:
            test, stat, df, p = "Fisher-Freeman-Halton (Monte Carlo)", float("nan"), (), float(res.pvalue)
    out = GroupStats(
        family=PROPORTION, groups=names, test=test, statistic=stat, df=df, pvalue=p,
        posthoc=posthoc_name, measure=measure, unit=unit,
    )
    if out.significant:
        pairs = [(i, j) for i in range(len(names)) for j in range(i + 1, len(names))]
        raw = [float(_stats.fisher_exact(mat[[i, j]]).pvalue) for i, j in pairs]
        adjusted = multipletests(raw, method="holm")[1]
        out.pairs = [(names[i], names[j], float(q)) for (i, j), q in zip(pairs, adjusted)]
        if len(names) >= 4:
            pm = pd.DataFrame(1.0, index=names, columns=names)
            for (a, b, q) in out.pairs:
                pm.loc[a, b] = pm.loc[b, a] = q
            out.letters = _letters(pm)
    return out


PAIRED_NAMES = {WELCH: "Paired t-test", RANK: "Wilcoxon signed-rank"}
ZERO_NAMES = {WELCH: "One-sample t-test vs 0", RANK: "Wilcoxon signed-rank vs 0"}


def compare_to_zero(values, family, *, group="", measure=None):
    """One group's per-fly values against zero: a one-sample t-test
    (``"welch"``) or the Wilcoxon signed-rank test (``"rank"``). None with fewer
    than :data:`MIN_N` flies, or when the test cannot be computed."""
    from scipy import stats as _stats

    vals = pd.to_numeric(pd.Series(np.asarray(values).ravel()), errors="coerce")
    vals = vals[np.isfinite(vals)].to_numpy(dtype=float)
    if len(vals) < MIN_N:
        return None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            if family == WELCH:
                res = _stats.ttest_1samp(vals, 0.0)
                df = (float(res.df),)
            else:
                res = _stats.wilcoxon(vals)
                df = ()
    except ValueError:
        return None
    p = float(res.pvalue)
    if not np.isfinite(p):
        return None
    return GroupStats(
        family=family, groups=[str(group)], test=ZERO_NAMES[family],
        statistic=float(res.statistic), df=df, pvalue=p, measure=measure,
        name=ZERO_NAMES[family],
    )


def compare_paired(before, after, family, *, labels=("before", "after"), measure=None):
    """The same flies measured twice: a paired t-test (``"welch"``) or the
    Wilcoxon signed-rank test (``"rank"``). ``before`` and ``after`` are Series
    indexed by fly id; flies missing either value are left out. None with fewer
    than :data:`MIN_N` pairs, or when the test cannot be computed.
    """
    from scipy import stats as _stats

    pairs = pd.concat({"a": before, "b": after}, axis=1).apply(pd.to_numeric, errors="coerce")
    pairs = pairs[np.isfinite(pairs["a"]) & np.isfinite(pairs["b"])]
    if len(pairs) < MIN_N:
        return None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            if family == WELCH:
                res = _stats.ttest_rel(pairs["b"], pairs["a"])
                df = (float(res.df),)
            else:
                res = _stats.wilcoxon(pairs["b"], pairs["a"])
                df = ()
    except ValueError:
        return None
    p = float(res.pvalue)
    if not np.isfinite(p):
        return None
    a, b = (str(x) for x in labels)
    return GroupStats(
        family=family, groups=[a, b], test=PAIRED_NAMES[family], statistic=float(res.statistic),
        df=df, pvalue=p, pairs=[(a, b, p)], measure=measure, name=PAIRED_NAMES[family],
        unit="the same flies measured twice",
    )


def holm(results, label):
    """Holm-adjust the p-values of two-group ``results`` together, in place; the
    test names gain ``(Holm across <label>)``. Returns ``results``."""
    from statsmodels.stats.multitest import multipletests

    done = [r for r in results if r is not None]
    if len(done) < 2:
        return results
    adjusted = multipletests([r.pvalue for r in done], method="holm")[1]
    for r, p in zip(done, adjusted):
        r.pvalue = float(p)
        r.pairs = [(r.groups[0], r.groups[1], float(p))]
        r.test = f"{r.test} (Holm across {label})"
    return results


RAYLEIGH = "Rayleigh test"


def rayleigh(phases_h, *, period_h=24.0, group="", measure=None):
    """Are one group's per-fly phases concentrated (rather than uniform around
    the cycle)? The Rayleigh test (astropy). ``statistic`` is the mean resultant
    length. None with fewer than :data:`MIN_N` flies."""
    from astropy.stats import rayleightest

    vals = pd.to_numeric(pd.Series(np.asarray(phases_h).ravel()), errors="coerce")
    vals = vals[np.isfinite(vals)].to_numpy(dtype=float)
    if len(vals) < MIN_N:
        return None
    theta = 2 * np.pi * vals / period_h
    p = float(rayleightest(theta))
    if not np.isfinite(p):
        return None
    return GroupStats(
        family=RANK, groups=[str(group)], test=RAYLEIGH,
        statistic=float(np.abs(np.mean(np.exp(1j * theta)))), df=(), pvalue=p,
        measure=measure, name=RAYLEIGH, unit="each fly's mean phase",
    )


VS_CONTROL_UNIT = (
    "one value per fly, each pulsed group vs its unpulsed control, Holm across pulse times"
)


def vs_control(
    per_fly,
    value_col,
    family,
    *,
    group_col="group",
    control_col="control_group",
    zt_col="zt",
    series_cols=(),
):
    """Each treated group against its own control: one two-group test per treated
    group, Holm-adjusted across the pulse times (``zt_col``) of each series (one
    combination of ``series_cols``).

    ``per_fly`` holds treated and control flies alike; a treated fly names its
    control in ``control_col``, and a control's own rows are the flies of that
    group. The returned results carry the adjusted p-value, and ``key`` holds
    the treated group's ``zt_col`` and ``series_cols`` values.
    """
    if per_fly is None or per_fly.empty:
        return []
    groups = per_fly[group_col].astype(str)
    controls = per_fly[control_col].astype(str) if control_col in per_fly else None
    if controls is None:
        return []
    control_names = set(controls[per_fly[control_col].notna()])
    results = []
    for grp in sorted(set(groups) - control_names):
        rows = per_fly[groups == grp]
        ctrl = rows[control_col].dropna()
        if ctrl.empty:
            continue
        ctrl = str(ctrl.iloc[0])
        if ctrl == grp:
            continue
        result = compare_groups(
            {grp: rows[value_col], ctrl: per_fly.loc[groups == ctrl, value_col]},
            family,
            order=[grp, ctrl],
            measure=grp,
        )
        if result is None or result.k != 2:
            continue
        first = rows.iloc[0]
        result.key = {c: first[c] for c in (zt_col, *series_cols) if c in rows.columns}
        result.unit = VS_CONTROL_UNIT
        results.append(result)

    by_series = {}
    for r in results:
        by_series.setdefault(tuple(str(r.key.get(c)) for c in series_cols), []).append(r)
    for members in by_series.values():
        holm(members, "pulse times")
    return results


def _df_text(df):
    return ", ".join(f"{d:.4g}" for d in df)


def stats_table(results):
    """The ``Stats`` sheet: the statistics each figure shows, one row each.

    Columns: ``Panel`` and ``Measure`` (when any result has one), ``Test``,
    ``Comparison``, ``Statistic``, ``df``, ``p value``, ``Letters``.
    """
    rows = []
    for r in results:
        if r is None:
            continue
        base = {"Panel": r.panel, "Measure": r.measure}
        if r.k == 1:
            rows.append(
                {
                    **base,
                    "Test": r.test,
                    "Comparison": r.groups[0],
                    "Statistic": r.statistic,
                    "df": _df_text(r.df),
                    "p value": r.pvalue,
                }
            )
            continue
        if r.k == 2:
            rows.append(
                {
                    **base,
                    "Test": r.test,
                    "Comparison": f"{r.groups[0]} vs {r.groups[1]}",
                    "Statistic": r.statistic,
                    "df": _df_text(r.df),
                    "p value": r.pvalue,
                }
            )
            continue
        rows.append(
            {
                **base,
                "Test": r.test,
                "Comparison": "All groups",
                "Statistic": r.statistic,
                "df": _df_text(r.df),
                "p value": r.pvalue,
            }
        )
        for a, b, p in r.pairs:
            rows.append(
                {**base, "Test": r.posthoc, "Comparison": f"{a} vs {b}", "p value": p}
            )
        if r.marks == "letters":
            for g in r.groups:
                rows.append(
                    {**base, "Test": LETTERS_TEST, "Comparison": g, "Letters": r.letters.get(g, "")}
                )
    cols = ["Panel", "Measure", "Test", "Comparison", "Statistic", "df", "p value", "Letters"]
    table = pd.DataFrame(rows, columns=cols)
    for c in ("Panel", "Measure", "Statistic", "df", "Letters"):
        if c in table and table[c].isna().all():
            table = table.drop(columns=c)
    return table
