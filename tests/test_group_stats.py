"""core.group_stats against reference values from R.

The reference values were computed in R 4.4.0 with base R only: ``oneway.test``
(var.equal = FALSE), ``kruskal.test``, ``t.test``, ``wilcox.test``, Games-Howell
from its definition with ``ptukey``, and Dunn's test from its definition (tie
correction) with ``p.adjust(method = "holm")``. The script is
``tests/reference/group_stats_reference.R``; its output is pasted here.
"""

import numpy as np
import pandas as pd
import pytest
import scipy

from clockwork.core import group_stats as gs

A = [12.1, 14.3, 11.8, 15.2, 13.9, 12.7, 14.8, 13.1]
B = [16.4, 18.9, 15.2, 17.7, 19.3, 16.8, 18.1]
C = [13.5, 21.2, 9.8, 17.4, 24.6, 11.3, 19.9, 15.7, 22.8]
D = [14.0, 15.1, 13.2, 16.3, 14.7, 15.8]
FOUR = {"A": A, "B": B, "C": C, "D": D}

R_WELCH_F, R_WELCH_DF2, R_WELCH_P = 10.4048722594, 14.150071628, 0.000708260239255
R_KW_H, R_KW_P = 10.4594633674, 0.0150384169242
R_GAMES_HOWELL = {
    ("A", "B"): 0.000514011819239,
    ("A", "C"): 0.203124730086,
    ("A", "D"): 0.20543585357,
    ("B", "C"): 0.999857034405,
    ("B", "D"): 0.0168593319689,
    ("C", "D"): 0.528853786206,
}
R_DUNN_HOLM = {
    ("A", "B"): 0.0151582930443,
    ("A", "C"): 0.102555430896,
    ("A", "D"): 0.872684855006,
    ("B", "C"): 0.872684855006,
    ("B", "D"): 0.297501508822,
    ("C", "D"): 0.872684855006,
}
R_WELCH_T, R_WELCH_T_DF, R_WELCH_T_P = -5.67325318434, 12.0251932757, 0.000102585207013
R_MWU_W, R_MWU_P = 0.5, 0.00176337944472

SCIPY = tuple(int(v) for v in scipy.__version__.split(".")[:2])


class TestAgainstR:
    def test_welch_anova_and_games_howell(self):
        r = gs.compare_groups(FOUR, gs.WELCH)
        assert r.test == "Welch ANOVA"
        assert r.statistic == pytest.approx(R_WELCH_F, rel=1e-6)
        assert r.df[1] == pytest.approx(R_WELCH_DF2, rel=1e-6)
        assert r.pvalue == pytest.approx(R_WELCH_P, rel=1e-5)
        got = {(a, b): p for a, b, p in r.pairs}
        for pair, p in R_GAMES_HOWELL.items():
            assert got[pair] == pytest.approx(p, rel=1e-4, abs=1e-8), pair

    def test_kruskal_wallis_and_dunn(self):
        r = gs.compare_groups(FOUR, gs.RANK)
        assert r.test == "Kruskal-Wallis"
        assert r.statistic == pytest.approx(R_KW_H, rel=1e-6)
        assert r.pvalue == pytest.approx(R_KW_P, rel=1e-5)
        got = {(a, b): p for a, b, p in r.pairs}
        for pair, p in R_DUNN_HOLM.items():
            assert got[pair] == pytest.approx(p, rel=1e-5), pair

    def test_welch_t_for_two_groups(self):
        r = gs.compare_groups({"A": A, "B": B}, gs.WELCH)
        assert r.test == "Welch's t-test"
        assert r.statistic == pytest.approx(R_WELCH_T, rel=1e-6)
        assert r.df[0] == pytest.approx(R_WELCH_T_DF, rel=1e-6)
        assert r.pvalue == pytest.approx(R_WELCH_T_P, rel=1e-5)
        assert r.pairs == [("A", "B", r.pvalue)]

    def test_mann_whitney_for_two_groups(self):
        r = gs.compare_groups({"A": A, "B": B}, gs.RANK)
        assert r.test == "Mann-Whitney U"
        assert r.statistic == pytest.approx(R_MWU_W)
        assert r.pvalue == pytest.approx(R_MWU_P, rel=1e-5)


class TestWhatIsShown:
    def test_two_groups_get_a_bracket(self):
        assert gs.compare_groups({"A": A, "B": B}, gs.WELCH).marks == "brackets"

    def test_three_significant_groups_get_brackets(self):
        r = gs.compare_groups({"A": A, "B": B, "D": D}, gs.WELCH)
        assert r.significant
        assert r.marks == "brackets"
        assert len(r.pairs) == 3

    def test_four_significant_groups_get_letters(self):
        r = gs.compare_groups(FOUR, gs.WELCH)
        assert r.marks == "letters"
        assert set(r.letters) == set(FOUR)
        # A and B differ under Games-Howell, so they share no letter.
        assert not set(r.letters["A"]) & set(r.letters["B"])
        # B and C do not differ, so they share one.
        assert set(r.letters["B"]) & set(r.letters["C"])

    def test_no_posthoc_when_the_omnibus_is_not_significant(self):
        rng = np.random.default_rng(1)
        same = {g: rng.normal(0, 1, 10) for g in "WXYZ"}
        r = gs.compare_groups(same, gs.RANK)
        assert not r.significant
        assert r.marks == "omnibus"
        assert r.pairs == []
        assert r.letters is None


class TestInputs:
    def test_small_groups_are_left_out(self):
        r = gs.compare_groups({"A": A, "B": B, "tiny": [1.0, 2.0]}, gs.WELCH)
        assert r.groups == ["A", "B"]

    def test_nan_values_are_dropped(self):
        r = gs.compare_groups({"A": A + [np.nan], "B": B}, gs.WELCH)
        assert r.statistic == pytest.approx(R_WELCH_T, rel=1e-6)

    def test_fewer_than_two_groups_is_none(self):
        assert gs.compare_groups({"A": A}, gs.WELCH) is None

    def test_identical_values_are_none(self):
        assert gs.compare_groups({"A": [1.0] * 5, "B": [1.0] * 5, "C": [1.0] * 5}, gs.RANK) is None

    def test_order_is_kept(self):
        r = gs.compare_groups(FOUR, gs.WELCH, order=["D", "C", "B", "A"])
        assert r.groups == ["D", "C", "B", "A"]

    def test_compare_frame(self):
        frame = pd.DataFrame({"Group": ["A"] * len(A) + ["B"] * len(B), "v": A + B})
        r = gs.compare_frame(frame, "v", "Group", gs.WELCH)
        assert r.pvalue == pytest.approx(R_WELCH_T_P, rel=1e-5)


class TestSettingAndText:
    def test_recommended_keeps_the_measures_family(self):
        assert gs.resolve_family(gs.RANK, gs.RECOMMENDED) == gs.RANK
        assert gs.resolve_family(gs.RANK, gs.WELCH) == gs.WELCH

    def test_notice_names_the_test(self):
        assert gs.notice_text(gs.RANK, 4).startswith("Kruskal-Wallis + Dunn (Holm), one value per fly")
        assert gs.notice_text(gs.WELCH, 2).startswith("Welch's t-test")

    def test_labels(self):
        assert gs.pair_label(0.0004) == "***"
        assert gs.pair_label(0.004) == "**"
        assert gs.pair_label(0.04) == "*"
        assert gs.pair_label(0.2134) == "p = 0.213"


class TestStatsTable:
    def test_holds_only_what_is_shown(self):
        r = gs.compare_groups(FOUR, gs.WELCH, measure="Night")
        table = gs.stats_table([r])
        assert list(table["Test"].unique()) == ["Welch ANOVA", "Games-Howell", gs.LETTERS_TEST]
        assert "Panel" not in table
        assert (table["Measure"] == "Night").all()

    def test_nonsignificant_has_only_the_omnibus_row(self):
        rng = np.random.default_rng(1)
        r = gs.compare_groups({g: rng.normal(0, 1, 10) for g in "WXYZ"}, gs.RANK)
        table = gs.stats_table([r])
        assert len(table) == 1
        assert table["Comparison"].iloc[0] == "All groups"

    def test_none_results_are_skipped(self):
        assert gs.stats_table([None]).empty


class TestProportions:
    def test_two_groups_use_fishers_exact(self):
        from scipy import stats

        r = gs.compare_proportions({"a": (18, 20), "b": (6, 20)})
        assert r.test == "Fisher's exact test"
        assert r.pvalue == pytest.approx(stats.fisher_exact([[18, 2], [6, 14]]).pvalue)
        assert r.marks == "brackets"

    def test_large_counts_use_chi_squared(self):
        r = gs.compare_proportions({"a": (40, 50), "b": (20, 50), "c": (25, 50)})
        assert r.test == "χ² test"
        assert r.significant
        assert len(r.pairs) == 3

    SMALL = {"a": (2, 5), "b": (1, 7), "c": (4, 5)}

    @pytest.mark.skipif(SCIPY < (1, 15), reason="R x C fisher_exact needs scipy >= 1.15")
    def test_small_counts_use_fisher_freeman_halton_reproducibly(self):
        first = gs.compare_proportions(self.SMALL)
        assert first.test.startswith("Fisher-Freeman-Halton")
        assert gs.compare_proportions(self.SMALL).pvalue == first.pvalue

    @pytest.mark.skipif(SCIPY >= (1, 15), reason="scipy >= 1.15 has R x C fisher_exact")
    def test_small_counts_report_chi_squared_on_older_scipy(self):
        assert gs.compare_proportions(self.SMALL).test == "χ² test"

    def test_every_fly_alike_is_none(self):
        assert gs.compare_proportions({"a": (5, 5), "b": (7, 7)}) is None

    def test_notice_names_the_test_that_ran(self):
        r = gs.compare_proportions({"a": (40, 50), "b": (20, 50), "c": (25, 50)})
        assert r.notice.startswith("χ² test + Fisher's exact (Holm), counts of flies per group")


class TestVsControl:
    def _frame(self):
        rng = np.random.default_rng(4)
        rows = []
        for zt, shift in ((3, 0.0), (15, -2.0), (21, 1.5)):
            grp = f"wt ZT{zt}"
            rows += [
                {"group": grp, "control_group": "wt ctrl", "zt": zt, "genotype": "wt", "r": v}
                for v in rng.normal(shift, 0.4, 10)
            ]
        rows += [
            {"group": "wt ctrl", "control_group": None, "zt": np.nan, "genotype": "wt", "r": v}
            for v in rng.normal(0.0, 0.4, 10)
        ]
        return pd.DataFrame(rows)

    def test_one_holm_adjusted_test_per_pulsed_group(self):
        from scipy import stats
        from statsmodels.stats.multitest import multipletests

        df = self._frame()
        results = gs.vs_control(df, "r", gs.WELCH, series_cols=["genotype"])
        assert [r.groups for r in results] == [
            ["wt ZT15", "wt ctrl"],
            ["wt ZT21", "wt ctrl"],
            ["wt ZT3", "wt ctrl"],
        ]
        ctrl = df.loc[df.group == "wt ctrl", "r"]
        raw = [
            stats.ttest_ind(df.loc[df.group == g, "r"], ctrl, equal_var=False).pvalue
            for g in ("wt ZT15", "wt ZT21", "wt ZT3")
        ]
        expected = multipletests(raw, method="holm")[1]
        assert [r.pvalue for r in results] == pytest.approx(list(expected))
        assert results[0].key == {"zt": 15, "genotype": "wt"}
        assert results[0].test == "Welch's t-test (Holm across pulse times)"
