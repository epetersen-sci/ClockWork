"""Faceted layouts: one panel per genotype (× sex …), a factor compared inside.

The layout is display-only (core/facets.py), so the tests are about arrangement —
which flies land in which panel, in what order, in what colour — and about the
guarantee that makes it safe to ship: with no compare factor chosen, every page
draws exactly what it drew before.
"""

import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from clockwork.core import dam_utilities, facets, plotting


def _table():
    """genotype × temperature × sex, with one combination missing on purpose:
    'mut' has no 18C flies, so its panel must still keep an 18C slot."""
    rows = []
    for geno in ("ctrl", "mut", "W1118xper0"):
        for temp in ("29C", "18C", "25C"):
            if geno == "mut" and temp == "18C":
                continue
            for sex in ("F", "M"):
                for i in range(2):
                    fid = f"{geno}_{temp}_{sex}_{i}"
                    rows.append(
                        {"id": fid, "genotype": geno, "temperature": temp, "sex": sex,
                         "group": f"{geno}-{temp}"}
                    )
    return pd.DataFrame(rows).set_index("id")


class TestLevels:
    def test_numeric_levels_sort_by_value_not_string(self):
        assert facets.order_levels(["29C", "18C", "25C", "110C"]) == ["18C", "25C", "29C", "110C"]

    def test_names_are_not_numbers(self):
        # A genotype that happens to contain digits must never sort as a number.
        assert facets.level_value("R272E") is None
        assert facets.level_value("W1118xper0") is None
        assert facets.level_value("25C") == 25.0
        assert facets.level_value("ZT21") == 21.0
        assert not facets.is_numeric_levels(["R272A", "D267K"])

    def test_explicit_order_wins_and_the_rest_follow(self):
        assert facets.order_levels(["a", "b", "c"], explicit=["c"]) == ["c", "a", "b"]

    def test_blank_sorts_last(self):
        assert facets.order_levels([facets.BLANK, "M", "F"]) == ["F", "M", facets.BLANK]

    def test_numeric_ramp_runs_cool_to_warm(self):
        c = facets.facet_colours(["18C", "25C", "29C"])
        assert c["18C"] == facets.NUMERIC_RAMP[0][1]
        assert c["29C"] == facets.NUMERIC_RAMP[-1][1]
        assert len(set(c.values())) == 3

    def test_categorical_levels_use_the_palette(self):
        c = facets.facet_colours(["F", "M"])
        assert list(c.values()) == list(facets.CATEGORICAL_PALETTE[:2])


class TestSpec:
    def test_json_round_trip(self):
        spec = facets.FacetSpec(
            compare_by="temperature",
            panel_by=("genotype", "sex"),
            include={"genotype": ["mut"]},
            order={"genotype": ["mut", "ctrl"]},
            reference={"genotype": "ctrl"},
            shared_y=False,
            ncols=2,
        )
        back = facets.FacetSpec.from_json(spec.to_json())
        assert back == spec
        assert json.loads(spec.to_json())["panel_by"] == ["genotype", "sex"]

    def test_unknown_setting_is_an_error(self):
        with pytest.raises(ValueError, match="Unknown facet setting"):
            facets.FacetSpec.from_dict({"compare": "temperature"})

    def test_unknown_column_is_named(self):
        with pytest.raises(ValueError, match="age"):
            facets.resolve_panels(_table(), facets.FacetSpec(compare_by="age"))

    def test_compare_factor_cannot_also_split_panels(self):
        spec = facets.FacetSpec(compare_by="temperature", panel_by=("temperature",))
        with pytest.raises(ValueError):
            facets.resolve_panels(_table(), spec)


class TestPanels:
    def test_one_panel_per_genotype(self):
        spec = facets.FacetSpec(compare_by="temperature", panel_by=("genotype",))
        panels = facets.resolve_panels(_table(), spec)
        assert [p.title for p in panels] == ["ctrl", "mut", "W1118xper0"]
        assert panels[0].levels == ["18C", "25C", "29C"]

    def test_missing_combination_keeps_its_slot(self):
        spec = facets.FacetSpec(compare_by="temperature", panel_by=("genotype",))
        mut = facets.resolve_panels(_table(), spec)[1]
        assert mut.levels == ["18C", "25C", "29C"]
        assert {lvl: len(ids) for lvl, ids in mut.series}["18C"] == 0

    def test_two_panel_factors_give_a_panel_per_combination(self):
        spec = facets.FacetSpec(compare_by="temperature", panel_by=("genotype", "sex"))
        titles = [p.title for p in facets.resolve_panels(_table(), spec)]
        assert titles[:2] == ["ctrl · F", "ctrl · M"]
        assert len(titles) == 6

    def test_every_fly_lands_in_exactly_one_panel(self):
        spec = facets.FacetSpec(compare_by="temperature", panel_by=("genotype", "sex"))
        ids = [i for p in facets.resolve_panels(_table(), spec) for i in p.ids]
        assert sorted(ids) == sorted(_table().index)

    def test_reference_is_matched_on_the_other_panel_factors(self):
        spec = facets.FacetSpec(
            compare_by="temperature", panel_by=("genotype", "sex"), reference={"genotype": "ctrl"}
        )
        panels = {p.title: p for p in facets.resolve_panels(_table(), spec)}
        ref = panels["mut · F"].reference_ids
        assert ref and all(i.startswith("ctrl_") and "_F_" in i for i in ref)

    def test_reference_panel_has_no_overlay_of_itself(self):
        spec = facets.FacetSpec(
            compare_by="temperature", panel_by=("genotype",), reference={"genotype": "ctrl"}
        )
        panels = {p.title: p for p in facets.resolve_panels(_table(), spec)}
        assert panels["ctrl"].reference_series == []
        assert panels["mut"].reference_label == "ctrl"

    def test_include_filter_does_not_hide_the_reference(self):
        spec = facets.FacetSpec(
            compare_by="temperature",
            panel_by=("genotype",),
            include={"genotype": ["mut"]},
            reference={"genotype": "ctrl"},
        )
        panels = facets.resolve_panels(_table(), spec)
        assert [p.title for p in panels] == ["mut"]
        assert panels[0].reference_ids

    def test_inactive_spec_is_one_panel_of_groups(self):
        panels = facets.resolve_panels(_table(), facets.FacetSpec())
        assert len(panels) == 1
        assert panels[0].levels == facets.order_levels(_table()["group"])


class TestArrangeGroups:
    def test_rows_per_panel_levels_in_order(self):
        tab = _table()
        spec = facets.FacetSpec(compare_by="temperature", panel_by=("genotype",))
        rows = facets.arrange_groups(tab, spec, sorted(tab["group"].unique()))
        assert [t for t, _ in rows] == ["ctrl", "mut", "W1118xper0"]
        assert rows[0][1] == ["ctrl-18C", "ctrl-25C", "ctrl-29C"]
        assert rows[1][1] == ["mut-25C", "mut-29C"]

    def test_only_the_requested_groups(self):
        tab = _table()
        spec = facets.FacetSpec(compare_by="temperature", panel_by=("genotype",))
        rows = facets.arrange_groups(tab, spec, ["mut-29C", "ctrl-18C"])
        assert rows == [("ctrl", ["ctrl-18C"]), ("mut", ["mut-29C"])]

    def test_coarser_grouping_cannot_be_arranged(self):
        """Grouped by genotype alone, one group spans every temperature."""
        tab = _table().assign(group=lambda t: t["genotype"])
        spec = facets.FacetSpec(compare_by="temperature", panel_by=("genotype",))
        assert facets.arrange_groups(tab, spec, ["ctrl", "mut"]) is None
        assert facets.group_order(tab, spec, ["mut", "ctrl"]) == ["mut", "ctrl"]

    def test_inactive_spec_arranges_nothing(self):
        assert facets.arrange_groups(_table(), facets.FacetSpec(), ["ctrl-18C"]) is None


class TestFromDataset:
    def test_factor_table_reads_coords_not_labels(self, master_ds):
        tab = facets.fly_factor_table(master_ds)
        assert list(tab.index) == [str(i) for i in master_ds["id"].values]
        assert {"genotype", "temperature", "group"} <= set(tab.columns)
        assert tab["genotype"].tolist() == [str(v) for v in master_ds["genotype"].values]

    def test_suggested_spec_compares_the_last_grouping_column(self, master_ds):
        spec = facets.suggested_spec(master_ds)
        assert spec.compare_by == "temperature" and spec.panel_by == ("genotype",)

    def test_single_factor_dataset_suggests_nothing(self, master_ds):
        assert facets.suggested_spec(dam_utilities.regroup_dataset(master_ds, ["genotype"])) is None


# ---------------------------------------------------------------------------
# Renderers
# ---------------------------------------------------------------------------


def _panels(ds, **kw):
    spec = facets.FacetSpec(compare_by="temperature", panel_by=("genotype",), **kw)
    panels = facets.resolve_panels(facets.fly_factor_table(ds), spec)
    return panels, facets.facet_colours(facets.layout_levels(panels))


class TestRenderers:
    def test_unfaceted_profile_is_unchanged(self, master_ds):
        """The historical figure, rebuilt through the new aggregation path."""
        zt = dam_utilities.get_zt_binned_dataframe(master_ds, "activity", 30)
        fig = plotting.daily_pattern_line(master_ds, "activity", "t", bin_size_minutes=30)
        labels = {str(k): v for k, v in dam_utilities.fly_group_map(master_ds).items()}
        agg = plotting.zt_profile_agg(zt, "activity", labels)
        again = plotting.profile_lines(agg, "activity", "t")
        assert fig.to_plotly_json() == again.to_plotly_json()
        # one line + one band per group, groups in sorted order
        names = [t.name for t in fig.data if t.showlegend is not False]
        assert names == sorted({str(g) for g in master_ds["group"].values})

    def test_profile_panels_colour_levels_consistently(self, master_ds):
        panels, colours = _panels(master_ds)
        zt = dam_utilities.get_zt_binned_dataframe(master_ds, "activity", 30)
        pairs = plotting.faceted_profiles(zt, "activity", panels, colours, title="A")
        assert [p.title for p, _ in pairs] == ["ctrl", "mut"]
        for _, fig in pairs:
            lines = [t for t in fig.data if t.mode == "lines" and t.fill is None]
            assert [t.legendgroup for t in lines] == ["25C", "29C"]
            assert [t.line.color for t in lines] == [colours["25C"], colours["29C"]]

    def test_shared_y_covers_every_band(self, master_ds):
        panels, colours = _panels(master_ds)
        zt = dam_utilities.get_zt_binned_dataframe(master_ds, "activity", 30)
        pairs = plotting.faceted_profiles(zt, "activity", panels, colours)
        ranges = {tuple(f.layout.yaxis.range) for _, f in pairs}
        assert len(ranges) == 1
        lo, hi = ranges.pop()
        for _, f in pairs:
            for t in f.data:
                y = np.asarray(t.y, dtype=float)
                assert np.nanmin(y) >= lo and np.nanmax(y) <= hi

    def test_reference_is_grey_and_drawn_first(self, master_ds):
        panels, colours = _panels(master_ds, reference={"genotype": "ctrl"})
        per_fly = plotting.per_fly_summary_table(master_ds, "activity")
        pairs = plotting.faceted_violins(per_fly, "All Day", panels, colours)
        by_title = {p.title: f for p, f in pairs}
        mut = by_title["mut"]
        assert mut.data[0].fillcolor == facets.REFERENCE_COLOUR
        coloured = [t for t in mut.data if t.fillcolor != facets.REFERENCE_COLOUR]
        assert [t.fillcolor for t in coloured] == [colours["25C"], colours["29C"]]
        assert all(t.fillcolor != facets.REFERENCE_COLOUR for t in by_title["ctrl"].data)

    def test_numeric_levels_are_placed_by_value(self, master_ds):
        panels, colours = _panels(master_ds)
        per_fly = plotting.per_fly_summary_table(master_ds, "activity")
        (_, fig), _ = plotting.faceted_violins(per_fly, "All Day", panels, colours)
        assert list(fig.layout.xaxis.tickvals) == [25.0, 29.0]
        assert list(fig.layout.xaxis.ticktext) == ["25C", "29C"]

    def test_a_panel_with_nothing_left_is_titled_and_says_so(self, master_ds):
        """'Rhythmic flies only' can empty a genotype's panel. It used to come back
        as a bare default figure (grey grid, no title) that looked broken."""
        panels, colours = _panels(master_ds)
        per_fly = plotting.per_fly_summary_table(master_ds, "activity")
        per_fly = per_fly[~per_fly["ID"].astype(str).str.contains("_mut")]
        by_title = {
            p.title: f
            for p, f in plotting.faceted_violins(per_fly, "All Day", panels, colours, title="A")
        }
        mut = by_title["mut"]
        assert mut.layout.title.text == "A — mut"
        assert len(mut.data) == 0
        assert "No flies" in mut.layout.annotations[0].text
        assert mut.layout.plot_bgcolor == "white"
        assert len(by_title["ctrl"].data) > 0

    def test_unfaceted_violins_are_unchanged(self, master_ds):
        per_fly = plotting.per_fly_summary_table(master_ds, "activity")
        fig, _ = plotting.group_violins(per_fly, "All Day")
        assert len(fig.data) == 1 and isinstance(fig.data[0], go.Violin)

    def test_spectra_panels(self, master_ds):
        panels, colours = _panels(master_ds, reference={"genotype": "ctrl"})
        ids = [str(i) for i in master_ds["id"].values]
        x = np.linspace(16, 32, 20)
        mat = np.vstack([np.exp(-((x - 24 - k * 0.1) ** 2)) for k in range(len(ids))])
        pairs = plotting.faceted_spectra(ids, mat, x, panels, colours, title="LS")
        mut = {p.title: f for p, f in pairs}["mut"]
        # Two temperatures in the panel: grey lines would lie on top of each other.
        assert all(t.name != "ctrl (reference)" for t in mut.data)
        assert any((t.name or "").startswith("29C") for t in mut.data)
        # One temperature: a single grey line can be told apart, so it is drawn.
        panels, colours = _panels(
            master_ds, reference={"genotype": "ctrl"}, include={"temperature": ["29C"]}
        )
        pairs = plotting.faceted_spectra(ids, mat, x, panels, colours, title="LS")
        mut = {p.title: f for p, f in pairs}["mut"]
        assert mut.data[0].name == "ctrl (reference)"

    def test_profile_reference_only_with_a_single_level(self, master_ds):
        zt = dam_utilities.get_zt_binned_dataframe(master_ds, "activity", 30)

        def _mut(**kw):
            panels, colours = _panels(master_ds, reference={"genotype": "ctrl"}, **kw)
            pairs = plotting.faceted_profiles(zt, "activity", panels, colours)
            return {p.title: f for p, f in pairs}["mut"]

        assert not any(t.legendgroup == "__reference__" for t in _mut().data)
        one = _mut(include={"temperature": ["25C"]})
        ref = [t for t in one.data if t.legendgroup == "__reference__"]
        assert ref and ref[0].line.color == facets.REFERENCE_COLOUR

    def test_combined_grid_has_every_panel_and_one_legend_entry_per_level(self, master_ds):
        panels, colours = _panels(master_ds)
        zt = dam_utilities.get_zt_binned_dataframe(master_ds, "activity", 30)
        pairs = plotting.faceted_profiles(zt, "activity", panels, colours)
        grid = plotting.combine_panels(pairs, ncols=2)
        assert len(grid.data) == sum(len(f.data) for _, f in pairs)
        legend = [t.name for t in grid.data if t.showlegend]
        assert legend == ["25C", "29C"]


# ---------------------------------------------------------------------------
# The page
# ---------------------------------------------------------------------------


def _chart_titles(at):
    return [
        json.loads(e.proto.spec)["layout"].get("title", {}).get("text")
        for e in at.get("plotly_chart")
    ]


class TestSleepActivityPage:
    def test_default_is_one_graph(self, app, master_ds):
        at = app(ds=master_ds, page="sleep_activity")
        assert not at.exception
        assert "Daily Activity Pattern" in _chart_titles(at)

    def test_panels_replace_the_single_graph(self, app, master_ds):
        at = app(ds=master_ds, page="sleep_activity", facet_compare="temperature")
        assert not at.exception, at.exception
        titles = _chart_titles(at)
        assert "Daily Activity Pattern" not in titles
        assert "Daily Activity Pattern — ctrl" in titles
        assert "Daily Activity Pattern — mut" in titles
        # the totals follow: one violin panel per genotype for the chosen period
        assert sum(t.startswith("Activity — Total — ") for t in titles) == 2


def _with_periods(ds):
    """``ds`` with a stored AC period per fly: 24 h at 25C, 26 h at 29C."""
    temps = np.asarray([str(t) for t in ds["temperature"].values])
    period = np.where(temps == "29C", 26.0, 24.0) + np.linspace(0, 0.5, ds.sizes["id"])
    ds = ds.assign(
        ac_period=("id", period),
        ac_power=("id", np.full(ds.sizes["id"], 0.5)),
    )
    return ds.assign_coords(ac_rhythmic=("id", np.ones(ds.sizes["id"], dtype=bool)))


class TestRhythmicityPage:
    def test_period_violins_by_group(self, app, master_ds):
        at = app(ds=_with_periods(master_ds), page="period_rhythmicity", period_rhythm_tab="Period length")
        assert not at.exception, at.exception
        assert "Period (Autocorrelation)" in _chart_titles(at)

    def test_period_violins_per_genotype(self, app, master_ds):
        at = app(
            ds=_with_periods(master_ds), page="period_rhythmicity",
            period_rhythm_tab="Period length", facet_compare="temperature",
        )
        assert not at.exception, at.exception
        titles = _chart_titles(at)
        assert "Period (Autocorrelation) — ctrl" in titles
        assert "Period (Autocorrelation) — mut" in titles


class TestSleepStatesPage:
    def test_default_layout(self, app, states_ds):
        at = app(ds=states_ds, page="sleep_states")
        assert not at.exception, at.exception

    def test_panels_arrange_the_per_group_figures(self, app, states_ds):
        at = app(ds=states_ds, page="sleep_states", facet_compare="temperature")
        assert not at.exception, at.exception
        md = [m.value for m in at.markdown]
        assert "##### ctrl" in md and "##### mut" in md
        titles = _chart_titles(at)
        # ctrl's temperatures come before mut's, in level order
        wave = [t for t in titles if t and t.endswith(tuple(f" — {p}" for p in ("LD", "DD", "full")))]
        assert wave == sorted(wave, key=lambda t: (t.split("-")[0], t))


class TestSleepDeprivationPage:
    @pytest.fixture
    def sd_results(self, unsplit_ds):
        from clockwork.core import sleep_deprivation as sd_module

        return sd_module.compute_sd_analysis(unsplit_ds, 720, 360, 1, bin_size_minutes=30, phase="LD")

    def test_one_figure_per_panel(self, app, unsplit_ds, sd_results):
        at = app(
            ds=unsplit_ds, page="sleep_deprivation", sd_results=sd_results,
            facet_compare="temperature",
        )
        assert not at.exception, at.exception
        titles = _chart_titles(at)
        assert "Baseline vs recovery — ctrl" in titles
        assert "Baseline vs recovery — mut" in titles
        fig = next(
            json.loads(e.proto.spec) for e in at.get("plotly_chart")
            if json.loads(e.proto.spec)["layout"].get("title", {}).get("text")
            == "Baseline vs recovery — mut"
        )
        names = {t.get("name") for t in fig["data"] if t.get("name")}
        assert "25C — Baseline" in names and "29C — Baseline" in names

    def test_default_is_unchanged(self, app, unsplit_ds, sd_results):
        at = app(ds=unsplit_ds, page="sleep_deprivation", sd_results=sd_results)
        assert not at.exception, at.exception
        names = {
            t.get("name") for e in at.get("plotly_chart") for t in json.loads(e.proto.spec)["data"]
        }
        assert "ctrl-25C — Baseline" in names


class TestHmmArrangement:
    def test_zt_fractions_one_row_per_panel(self, master_ds):
        from clockwork.core import hmm_models

        rng = np.random.default_rng(0)
        ds = master_ds.assign(
            hmm_state=(("id", "time"), rng.integers(0, 4, (master_ds.sizes["id"], master_ds.sizes["time"])))
        )
        tab = facets.fly_factor_table(ds)
        spec = facets.FacetSpec(compare_by="temperature", panel_by=("genotype",))
        rows = facets.arrange_groups(tab, spec, sorted(tab["group"].unique()))
        fig = hmm_models.plot_zt_state_fractions(ds, n_states=4, panel_rows=rows)
        visible = [ax for ax in fig.axes if ax.get_visible() and ax.get_title()]
        assert [ax.get_title() for ax in visible] == ["ctrl-25C", "ctrl-29C", "mut-25C", "mut-29C"]
        # two rows: ctrl's axes share a row, above mut's
        assert visible[0].get_position().y0 == visible[1].get_position().y0
        assert visible[2].get_position().y0 < visible[0].get_position().y0
