"""Faceted comparison layouts: one panel per genotype, temperatures within it.

A multi-factor experiment (genotype × temperature, perhaps × sex) has too many
groups to read when every group is one trace in one figure. The comparison the
design asks for is usually *within* one level of the other factors: how does THIS
genotype respond across temperatures? That is a small-multiples layout — one panel
per combination of the ``panel_by`` factors, and inside each panel one trace (or
one violin) per level of ``compare_by``.

Everything here is DISPLAY ONLY. Nothing is written to the dataset, no analysis is
re-run, and the canonical ``group`` coord (ARCHITECTURE rule 4) is untouched: a
facet layout is a way of arranging flies that have already been analysed.

The layout is described by a :class:`FacetSpec`, plain data that round-trips
through JSON. The Streamlit page builds one from its sidebar and a command-line
run will read one from a file, and both hand it to the same functions in
``plotting`` — so a figure set made in the app can be replayed without it.

Factor values are read off the per-fly coords, never by splitting the ``group``
label: values legitimately contain the ``-`` the label is joined with
(``W1118xper0`` is fine, ``dsOpa1(67159)+Ldh-mut`` is not).
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

import dam_utilities

#: The categorical palette (Paul Tol's "bright" plus extras). ``plotting`` uses the
#: same list for its group curves, so a factor drawn here matches the page's other
#: figures when there are no numeric levels.
CATEGORICAL_PALETTE = (
    "#4477AA",
    "#EE6677",
    "#228833",
    "#CCBB44",
    "#66CCEE",
    "#AA3377",
    "#BBBBBB",
    "#000000",
    "#EE7733",
    "#009988",
)

#: Cool-to-warm stops for an ordered numeric factor (temperature, age, dose): dark
#: blue, light blue, green, yellow, red — the lab's temperature-series colouring.
#: Sampled by RANK, not by value, so neighbouring levels (27 and 29 °C) stay
#: distinguishable however unevenly the levels are spaced.
NUMERIC_RAMP = (
    (0.00, "#3F4FA3"),
    (0.25, "#7FA8D8"),
    (0.50, "#5DBB63"),
    (0.75, "#E9C83B"),
    (1.00, "#E8636B"),
)

#: The control/reference flies drawn behind every panel.
REFERENCE_COLOUR = "#AFAFAF"

#: What a blank metadata cell reads as in a panel title or a legend.
BLANK = "(blank)"

# A level is numeric when it is a number with, at most, a ZT/CT prefix or a short
# unit suffix: "25C", "25 °C", "ZT21", "3d", "10min". A leading letter otherwise
# (R272E, W1118xper0) makes it a name, so genotypes never sort as numbers.
_NUMERIC_LEVEL = re.compile(
    r"^\s*(?:ZT|CT)?\s*(-?\d+(?:\.\d+)?)\s*(?:°?\s*[A-Za-z%]{0,3})?\s*$", re.IGNORECASE
)


# ---------------------------------------------------------------------------
# Levels: parsing, ordering, colour
# ---------------------------------------------------------------------------


def level_value(value):
    """The number a factor level stands for, or None if it is a name.

    ``"25C"`` → 25.0, ``"ZT21"`` → 21.0, ``"18"`` → 18.0, ``"R272E"`` → None.
    """
    if value is None:
        return None
    if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
        return None if pd.isna(value) else float(value)
    m = _NUMERIC_LEVEL.match(str(value))
    return float(m.group(1)) if m else None


def is_numeric_levels(values):
    """True when every level parses as a number (so it can sit on a numeric axis)."""
    values = [v for v in values if v != BLANK]
    return bool(values) and all(level_value(v) is not None for v in values)


def _natural_key(text):
    return [int(t) if t.isdigit() else t.casefold() for t in re.split(r"(\d+)", str(text))]


def order_levels(values, explicit=None):
    """Distinct ``values`` in display order.

    An ``explicit`` order wins for the values it names; anything it does not name
    follows, in the default order. The default is numeric when every level is a
    number (``18C < 25C < 29C``, not the string order), natural otherwise
    (``Rep2 < Rep10``). A blank cell always sorts last.
    """
    distinct = list(dict.fromkeys(str(v) for v in values))
    blanks = [v for v in distinct if v == BLANK]
    rest = [v for v in distinct if v != BLANK]
    if is_numeric_levels(rest):
        rest.sort(key=lambda v: (level_value(v), v))
    else:
        rest.sort(key=_natural_key)
    default = rest + blanks
    if not explicit:
        return default
    named = [str(v) for v in explicit if str(v) in set(default)]
    return named + [v for v in default if v not in set(named)]


def _hex_to_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))


def _ramp(t):
    """The colour at position ``t`` in [0, 1] along :data:`NUMERIC_RAMP`."""
    for (t0, c0), (t1, c1) in zip(NUMERIC_RAMP, NUMERIC_RAMP[1:]):
        if t <= t1:
            f = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
            a, b = _hex_to_rgb(c0), _hex_to_rgb(c1)
            return "#{:02X}{:02X}{:02X}".format(
                *(round(x + (y - x) * f) for x, y in zip(a, b))
            )
    return NUMERIC_RAMP[-1][1]


def facet_colours(levels):
    """``{level: colour}`` for an ORDERED list of levels.

    Computed once for the whole layout and passed to every panel, so 25 °C is the
    same green in every genotype's panel even when a panel lacks some levels.
    Numeric levels get the cool-to-warm ramp; names get the categorical palette.
    """
    levels = [str(v) for v in levels]
    if is_numeric_levels(levels):
        named = [v for v in levels if v != BLANK]
        n = len(named)
        out = {v: _ramp(0.5 if n == 1 else i / (n - 1)) for i, v in enumerate(named)}
        if BLANK in levels:
            out[BLANK] = "#7F7F7F"
        return out
    return {v: CATEGORICAL_PALETTE[i % len(CATEGORICAL_PALETTE)] for i, v in enumerate(levels)}


# ---------------------------------------------------------------------------
# The spec
# ---------------------------------------------------------------------------


@dataclass
class FacetSpec:
    """How to arrange one figure's flies into panels. Plain data; JSON round-trips.

    Column names are METADATA column names, the ones ticked on Import
    (``pulse_time``, not the ``pulse_zt_hour`` coord it is stored under).

    compare_by
        The factor compared inside each panel: one line per level on a profile,
        one violin per level on a distribution plot. ``None`` is the ungrouped,
        historical layout — one figure, one trace per ``group`` — and is the
        default, so a page that has not been asked to facet looks as it did.
    panel_by
        One panel per combination of these (``("genotype", "sex")`` gives a
        ``Mutant · F`` and a ``Mutant · M`` panel).
    include
        ``{column: [values to keep]}``. A column not named keeps every value.
    order
        ``{column: [values in order]}``, overriding the default level order.
    reference
        ``{column: value}``: flies drawn in grey behind every other panel, matched
        to the panel on the remaining ``panel_by`` factors (a ``Mutant · F``
        panel gets the female controls). Typically the control genotype.
    shared_y
        One y-range for every panel, so heights compare across panels.
    ncols
        Panels per row, on screen and in a combined export.
    """

    compare_by: str | None = None
    panel_by: tuple = ()
    include: dict = field(default_factory=dict)
    order: dict = field(default_factory=dict)
    reference: dict | None = None
    shared_y: bool = True
    ncols: int = 3

    def __post_init__(self):
        self.panel_by = tuple(str(c) for c in (self.panel_by or ()))
        if self.compare_by is not None:
            self.compare_by = str(self.compare_by)
        self.include = {str(k): [str(v) for v in vals] for k, vals in (self.include or {}).items()}
        self.order = {str(k): [str(v) for v in vals] for k, vals in (self.order or {}).items()}
        self.reference = (
            {str(k): str(v) for k, v in self.reference.items()} if self.reference else None
        )
        self.ncols = max(1, int(self.ncols))

    @property
    def active(self):
        """True when this spec actually facets (a compare factor is chosen)."""
        return self.compare_by is not None

    @property
    def series_col(self):
        """The column naming each trace: the compare factor, or ``group``."""
        return self.compare_by if self.compare_by is not None else "group"

    def columns_used(self):
        cols = list(self.panel_by)
        if self.compare_by is not None:
            cols.append(self.compare_by)
        cols += list(self.include) + list(self.order) + list(self.reference or {})
        return list(dict.fromkeys(cols))

    def validate(self, available):
        """Raise ``ValueError`` naming any column the dataset does not have."""
        available = set(available)
        missing = [c for c in self.columns_used() if c not in available and c != "group"]
        if missing:
            raise ValueError(
                f"Unknown column(s) {missing}; this dataset's factors are "
                f"{sorted(available)}."
            )
        if self.compare_by is not None and self.compare_by in self.panel_by:
            raise ValueError(
                f"{self.compare_by!r} cannot be both the compared factor and a panel factor."
            )
        return self

    def to_dict(self):
        d = asdict(self)
        d["panel_by"] = list(self.panel_by)
        return d

    @classmethod
    def from_dict(cls, d):
        d = dict(d or {})
        known = set(cls.__dataclass_fields__)
        unknown = sorted(set(d) - known)
        if unknown:
            raise ValueError(f"Unknown facet setting(s) {unknown}; expected {sorted(known)}.")
        return cls(**d)

    def to_json(self, **kwargs):
        return json.dumps(self.to_dict(), indent=kwargs.pop("indent", 2), **kwargs)

    @classmethod
    def from_json(cls, text):
        return cls.from_dict(json.loads(text))


# ---------------------------------------------------------------------------
# Flies → panels
# ---------------------------------------------------------------------------


def factor_columns(ds):
    """The dataset's grouping factors, in METADATA column names, Import order."""
    back = {v: k for k, v in dam_utilities.METADATA_COORD_RENAMES.items()}
    return [back.get(c, c) for c in dam_utilities.group_defining_coords(ds)]


def _cell(v):
    v = v.item() if hasattr(v, "item") else v
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return BLANK
    s = str(v)
    if s.strip() == "" or s.lower() == "nan":
        return BLANK
    # A parsed ZT hour (21.0) reads as the hour it was written as.
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return s


def fly_factor_table(ds):
    """One row per fly (index = id, as str): every grouping factor, plus ``group``.

    Read off the per-id coords in one pass, in metadata column names. Values are
    strings; a blank cell is :data:`BLANK`.
    """
    ids = [str(v.item() if hasattr(v, "item") else v) for v in ds["id"].values]
    cols = {}
    for col in factor_columns(ds):
        coord = dam_utilities.METADATA_COORD_RENAMES.get(col, col)
        cols[col] = [_cell(v) for v in ds[coord].values]
    if "group" in ds.coords:
        cols["group"] = [_cell(v) for v in ds["group"].values]
    else:
        cols["group"] = ["All flies"] * len(ids)
    return pd.DataFrame(cols, index=pd.Index(ids, name="id"))


@dataclass
class Panel:
    """One panel of a faceted layout.

    ``series`` lists every compare level in layout order, each with its fly ids —
    possibly none, so a level missing from this panel keeps its slot and the
    panels line up. ``reference_series`` is the same for the grey control flies
    (empty when there is no reference, or when this panel is the reference).
    """

    key: dict
    title: str
    series: list
    reference_series: list = field(default_factory=list)
    reference_label: str | None = None

    @property
    def ids(self):
        return [i for _, ids in self.series for i in ids]

    @property
    def reference_ids(self):
        return [i for _, ids in self.reference_series for i in ids]

    @property
    def levels(self):
        return [lvl for lvl, _ in self.series]

    @property
    def levels_present(self):
        """The levels this panel actually has flies at."""
        return [lvl for lvl, ids in self.series if ids]

    @property
    def line_reference_series(self):
        """The reference to draw on a LINE graph: only in a panel showing a single
        level. With several levels the grey lines lie on top of one another and
        can only be told apart by hovering; a violin keeps each level in its own
        slot, so it uses :attr:`reference_series` whatever the count."""
        if len(self.levels_present) != 1:
            return []
        (lvl,) = self.levels_present
        return [(v, ids) for v, ids in self.reference_series if v == lvl and ids]

    def labels(self):
        """``{fly id: compare level}`` for this panel's own flies."""
        return {i: lvl for lvl, ids in self.series for i in ids}

    def reference_labels(self, *, lines=False):
        series = self.line_reference_series if lines else self.reference_series
        return {i: lvl for lvl, ids in series for i in ids}


def _apply_include(table, include, skip=()):
    for col, keep in (include or {}).items():
        if col in skip or col not in table.columns or not keep:
            continue
        table = table[table[col].isin(set(keep))]
    return table


def _series(table, col, levels):
    groups = {k: list(v) for k, v in table.groupby(col, sort=False).groups.items()}
    return [(lvl, [str(i) for i in groups.get(lvl, [])]) for lvl in levels]


def resolve_panels(table, spec):
    """Arrange the flies in ``table`` (from :func:`fly_factor_table`) into panels.

    With an inactive spec this is the historical layout: one panel, one series per
    ``group``. Otherwise one panel per non-empty combination of ``spec.panel_by``
    (in level order), each holding every compare level.
    """
    spec.validate(table.columns)
    shown = _apply_include(table, spec.include)
    series_col = spec.series_col
    levels = order_levels(shown[series_col], spec.order.get(series_col))

    if spec.panel_by:
        panel_levels = [order_levels(shown[c], spec.order.get(c)) for c in spec.panel_by]
        combos = [()]
        for lv in panel_levels:
            combos = [c + (v,) for c in combos for v in lv]
    else:
        combos = [()]

    # The reference comes from the table filtered on everything EXCEPT its own
    # column: narrowing the display to one mutant must not hide its control.
    ref = spec.reference or None
    ref_pool = None
    if ref:
        ref_pool = _apply_include(table, spec.include, skip=set(ref))
        for col, val in ref.items():
            ref_pool = ref_pool[ref_pool[col] == val]

    panels = []
    for combo in combos:
        key = dict(zip(spec.panel_by, combo))
        sub = shown
        for col, val in key.items():
            sub = sub[sub[col] == val]
        if sub.empty:
            continue

        ref_series, ref_label = [], None
        is_ref_panel = bool(ref) and all(key.get(c) == v for c, v in ref.items() if c in key)
        if ref and not is_ref_panel and ref_pool is not None:
            rp = ref_pool
            for col, val in key.items():
                if col not in ref:
                    rp = rp[rp[col] == val]
            if not rp.empty:
                ref_series = _series(rp, series_col, levels)
                ref_label = " · ".join(str(v) for v in ref.values())

        panels.append(
            Panel(
                key=key,
                title=" · ".join(combo),
                series=_series(sub, series_col, levels),
                reference_series=ref_series,
                reference_label=ref_label,
            )
        )
    return panels


def arrange_groups(table, spec, groups):
    """Arrange whole ``group`` labels into panel rows, for pages that draw one
    figure per group and cannot overlay levels (colour already means something
    there — a sleep state, a recovery day).

    Returns ``[(panel title, [group, ...]), ...]``: one row per panel, in panel
    order, each row's groups in compare-level order, so a genotype's temperatures
    sit side by side. Only groups in ``groups`` are placed.

    Returns None when the spec is inactive, or when a group does not sit in one
    panel at one level — the grouping is coarser than the layout (grouped by
    genotype alone, compare temperature) — so the caller keeps its own order.
    """
    if not spec.active:
        return None
    spec.validate(table.columns)
    wanted = [str(g) for g in groups]
    shown = _apply_include(table, spec.include)
    shown = shown[shown["group"].isin(set(wanted))]
    if shown.empty:
        return None
    cols = list(spec.panel_by) + [spec.compare_by]
    if (shown.groupby("group")[cols].nunique() > 1).any().any():
        return None
    first = shown.groupby("group")[cols].first()

    rank = {c: {v: i for i, v in enumerate(order_levels(shown[c], spec.order.get(c)))} for c in cols}

    def _key(g):
        return tuple(rank[c][first.at[g, c]] for c in cols) + (_natural_key(g),)

    placed = sorted((g for g in wanted if g in first.index), key=_key)
    rows = []
    for g in placed:
        title = " · ".join(first.at[g, c] for c in spec.panel_by)
        if rows and rows[-1][0] == title:
            rows[-1][1].append(g)
        else:
            rows.append((title, [g]))
    return rows


def group_order(table, spec, groups):
    """``groups`` in panel-then-level order, or unchanged when they cannot be
    arranged. For figures that lay their groups out themselves."""
    rows = arrange_groups(table, spec, groups)
    if rows is None:
        return list(groups)
    ordered = [g for _, gs in rows for g in gs]
    return ordered + [g for g in groups if g not in set(ordered)]


def layout_levels(panels):
    """The compare levels shared by every panel (they all carry the same list)."""
    return panels[0].levels if panels else []


def suggested_spec(ds):
    """A sensible active spec for a multi-factor dataset, or None for one factor.

    Compares the LAST grouping column (conventionally the treatment — temperature
    after genotype) within panels of the others.
    """
    cols = list(dam_utilities.get_group_columns(ds))
    back = {v: k for k, v in dam_utilities.METADATA_COORD_RENAMES.items()}
    cols = [back.get(c, c) for c in cols]
    if len(cols) < 2:
        return None
    return FacetSpec(compare_by=cols[-1], panel_by=tuple(cols[:-1]))
