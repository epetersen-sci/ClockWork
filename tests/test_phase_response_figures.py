"""The two phase-response figures, on a cohort pulsed at more than one time.

``tests/test_phase_response.py`` pins the quantity; this pins what is DRAWN from
it, which is a separate way to be wrong. A curve can plot the right numbers on
the wrong axis, connect points that were never measured, or order a dose axis
``100, 20`` because the values arrived as text.

The cohort here exists because the real one does not yet: the lab's exp7 pulsed
at a single circadian time, so on that data the curve is permanently the "not
enough timepoints" message and nothing about the line itself is exercised.
Every arm below has a DESIGNED response, so the figure can be checked against a
number rather than against itself.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from conftest import MINUTES_PER_DAY, PULSE_DD_DAY

import phase_shift as ps
import plotting

#: ``(pulse ZT, duration, intensity, hours the peak MOVES LATER after the pulse)``.
#: A later peak is a delay, and the response is reported control-minus-fly, so the
#: expected response is the NEGATIVE of the last field.
ARMS = (
    (9.0, 20.0, 20.0, 1.0),  # delay
    (15.0, 20.0, 20.0, 0.0),  # dead zone
    (21.0, 20.0, 20.0, -2.0),  # advance
    (21.0, 20.0, 100.0, -3.0),  # same dose, brighter: a bigger advance
    (np.nan, 0.0, np.nan, 0.0),  # unpulsed control
)
BASE_PHASE = {"A": 12.0, "B": 13.0}
GROUP_BY = ("genotype", "pulse_zt_hour", "pulse_duration_minutes", "pulse_intensity")
TOL = 0.5


def _build_prc_cohort(n_per_arm=6, n_days=8, seed=1):
    """Two genotypes x :data:`ARMS`, each arm with a designed phase response.

    Built like ``conftest._build_pulse_cohort`` — one Gaussian bump per day, sharp
    enough to survive the 12 h low-pass the peak method applies — but with several
    pulse times, which is what a phase response curve needs and that fixture does
    not have.
    """
    rng = np.random.default_rng(seed)
    ids, gene_of, zt_of, dur_of, int_of, move_of = [], [], [], [], [], []
    for gene in ("A", "B"):
        for zt, dur, intensity, move in ARMS:
            for k in range(n_per_arm):
                ids.append(f"{gene}_{'ctl' if dur == 0 else f'zt{zt:g}i{intensity:g}'}_{k}")
                gene_of.append(gene)
                zt_of.append(zt)
                dur_of.append(dur)
                int_of.append(intensity)
                move_of.append(move)

    n_id = len(ids)
    time = np.arange(n_days * MINUTES_PER_DAY)
    activity = np.zeros((time.size, n_id))
    for j in range(n_id):
        for day in range(n_days):
            # The pulse is given late on the last entrained day and moves the peak
            # from the first DD day onward.
            moved = move_of[j] if day >= PULSE_DD_DAY else 0.0
            centre = day * MINUTES_PER_DAY + (BASE_PHASE[gene_of[j]] + moved) * 60.0
            activity[:, j] += 20 * np.exp(-0.5 * ((time - centre) / 120.0) ** 2)
    activity += rng.normal(0, 0.05, activity.shape).clip(0)

    start = np.datetime64("2025-01-15T09:00:00")
    return xr.Dataset(
        {"activity": (("time", "id"), activity)},
        coords={
            "id": ids,
            "time": time.astype(np.int64),
            "genotype": ("id", np.array(gene_of)),
            "pulse_zt_hour": ("id", np.array(zt_of, dtype="float32")),
            "pulse_duration_minutes": ("id", np.array(dur_of, dtype="float32")),
            "pulse_intensity": ("id", np.array(int_of, dtype="float32")),
            "start_datetime": ("id", np.array([start] * n_id)),
            "first_DD_day": (
                "id",
                np.array([start + np.timedelta64(PULSE_DD_DAY * MINUTES_PER_DAY, "m")] * n_id),
            ),
        },
        attrs={
            "time_is_relative_minutes": 1,
            "phase": "full",
            "split_applied": 1,
            "group_columns": ["genotype", "pulse_time", "pulse_duration_min", "pulse_intensity"],
            "genotype": ["A", "B"],
            "metadata_coords": [
                "genotype",
                "pulse_duration_minutes",
                "pulse_intensity",
                "pulse_zt_hour",
            ],
            "group_coord_names": [
                "genotype",
                "pulse_zt_hour",
                "pulse_duration_minutes",
                "pulse_intensity",
            ],
        },
    )


@pytest.fixture(scope="module")
def prc_multi():
    return _build_prc_cohort()


@pytest.fixture(scope="module")
def response(prc_multi):
    return ps.compute_phase_response(prc_multi, group_by=GROUP_BY)


@pytest.fixture(scope="module")
def treated(response):
    """What the page draws: the pulsed flies only.

    Controls are computed — their spread is the noise floor — but a control is
    the reference, so drawing it is drawing zero against itself.
    """
    pf = response["per_fly"]
    return pf[~pf["group"].isin(set(response["controls"]))].copy()


SERIES = ("genotype", "pulse_duration_minutes", "pulse_intensity")


def fig_traces(summary):
    """Every trace of the drawn curve, as a list of ``(x, y)`` pairs."""
    fig = plotting.phase_response_curve(summary, series_cols=SERIES)
    return [
        list(zip(np.asarray(tr.x, dtype=float), np.asarray(tr.y, dtype=float)))
        for tr in fig.data
    ]


class TestTheCurve:
    def test_it_recovers_every_arms_designed_response(self, treated):
        """The whole chain, from activity to the y value on the line."""
        summ = ps.summarize_phase_response(treated, by=("zt", *SERIES))
        # The grouping columns arrive as TEXT — _group_meta casts the coords — so
        # comparing them to a number needs coercing first. The same fact is why the
        # dose axis has to be sorted numerically; see TestTheDoseAxisIsSortedAsNumbers.
        intensity_of = pd.to_numeric(summ["pulse_intensity"])
        drawn = {}
        for tr in fig_traces(summ):
            for x, y in tr:
                drawn.setdefault(x, []).append(y)

        for zt, _dur, intensity, move in ARMS:
            if not np.isfinite(zt):
                continue
            expected = -move
            row = summ[(summ["zt"] == zt) & (intensity_of == intensity)]
            assert len(row) == 2, f"one row per genotype at ZT{zt:g}/{intensity:g}"
            for _, r in row.iterrows():
                assert r["mean"] == pytest.approx(expected, abs=TOL), r.to_dict()
            assert any(
                abs(y - expected) < TOL for y in drawn.get(zt, [])
            ), f"nothing plotted near {expected} h at ZT{zt:g}"

    def test_a_delay_is_below_the_line_and_an_advance_above(self, treated):
        summ = ps.summarize_phase_response(treated, by=("zt", *SERIES))
        at = lambda z: summ[summ["zt"] == z]["mean"]  # noqa: E731
        assert (at(9.0) < 0).all(), "ZT9 was built as a delay"
        assert (at(21.0) > 0).all(), "ZT21 was built as an advance"

    def test_only_the_pulse_times_present_get_a_tick(self, treated):
        summ = ps.summarize_phase_response(treated, by=("zt", *SERIES))
        fig = plotting.phase_response_curve(summ, series_cols=SERIES)
        assert list(fig.layout.xaxis.tickvals) == [9.0, 15.0, 21.0]
        assert list(fig.layout.xaxis.ticktext) == ["ZT9", "ZT15", "ZT21"]
        # Nothing is drawn between the measured times either.
        for tr in fig.data:
            assert set(np.asarray(tr.x, dtype=float)) <= {9.0, 15.0, 21.0}

    def test_the_points_are_joined_in_pulse_time_order(self, treated):
        """A line through points sorted by y rather than by x is a different claim."""
        summ = ps.summarize_phase_response(treated, by=("zt", *SERIES))
        fig = plotting.phase_response_curve(summ, series_cols=SERIES)
        assert fig.data, "nothing drawn"
        for tr in fig.data:
            xs = np.asarray(tr.x, dtype=float)
            assert list(xs) == sorted(xs)
            assert "lines" in tr.mode

    def test_each_line_is_one_series(self, treated):
        summ = ps.summarize_phase_response(treated, by=("zt", *SERIES))
        fig = plotting.phase_response_curve(summ, series_cols=SERIES)
        # Two genotypes x (20 min at 20 int, 20 min at 100 int).
        assert len(fig.data) == 4
        assert len({tr.name for tr in fig.data}) == 4

    def test_a_genotype_keeps_one_colour_across_its_doses(self, treated):
        """Colour carries the first series column and dash the rest, so a genotype's
        arms read as one genotype rather than as unrelated lines."""
        summ = ps.summarize_phase_response(treated, by=("zt", *SERIES))
        fig = plotting.phase_response_curve(summ, series_cols=SERIES)
        by_colour = {}
        for tr in fig.data:
            by_colour.setdefault(tr.line.color, set()).add(tr.name.split(" · ")[0])
        assert all(len(v) == 1 for v in by_colour.values()), by_colour
        assert len(by_colour) == 2, "one colour per genotype"

    def test_error_bars_are_the_per_fly_spread(self, treated):
        """Not a bootstrap: the response is measured once per fly, so the flies are
        the replicates and the SEM is the ordinary one over them. The "Phase over
        Time" tab does resample, because a group-mean peak has no per-fly values
        underneath it — two different figures, two different reasons."""
        # Widen one group so the bars are not all zero on this exact-by-design cohort.
        noisy = treated.copy()
        sel = noisy.index[noisy["zt"] == 21.0][:6]
        noisy.loc[sel, "response_hours"] = np.linspace(1.0, 5.0, len(sel))

        summ = ps.summarize_phase_response(noisy, by=("zt", *SERIES))
        assert (summ["sem"] > 0).any(), "the fixture must have some spread to plot"
        assert np.allclose(summ["sem"], summ["sd"] / np.sqrt(summ["n"]))

        fig = plotting.phase_response_curve(summ, series_cols=SERIES, error="sem")
        plotted = sorted(
            v for tr in fig.data for v in np.asarray(tr.error_y.array, dtype=float)
        )
        assert plotted == pytest.approx(sorted(summ["sem"].to_numpy(dtype=float)))

    def test_no_error_bars_when_none_are_asked_for(self, treated):
        summ = ps.summarize_phase_response(treated, by=("zt", *SERIES))
        fig = plotting.phase_response_curve(summ, series_cols=SERIES, error=None)
        assert all(tr.error_y.array is None for tr in fig.data)

    def test_an_empty_summary_draws_nothing_rather_than_raising(self):
        fig = plotting.phase_response_curve(pd.DataFrame())
        assert fig.data == ()


class TestTheViolins:
    def test_one_column_per_genotype_time_and_dose(self, treated):
        fig, drawn = plotting.phase_response_violins(
            treated,
            major_cols=("genotype", "zt", "pulse_duration_minutes"),
            split_col="pulse_intensity",
        )
        cats = list(fig.layout.xaxis.categoryarray)
        # Two genotypes x three pulse times, all at 20 min.
        assert len(cats) == 6
        assert cats[0].startswith("A · 9")
        assert all(c.endswith(" · 20") for c in cats), cats

    def test_intensities_sit_side_by_side_in_one_column(self, treated):
        """The ZT21 column carries both intensities; they must not overlay."""
        fig, _ = plotting.phase_response_violins(
            treated,
            major_cols=("genotype", "zt", "pulse_duration_minutes"),
            split_col="pulse_intensity",
        )
        assert fig.layout.violinmode == "group"
        assert len(fig.data) == 2, "one trace per intensity"
        assert {tr.name for tr in fig.data} == {"20", "100"}
        shared = set(fig.data[0].x) & set(fig.data[1].x)
        assert any("21" in c for c in shared), "ZT21 must carry both intensities"

    def test_without_a_split_the_traces_are_the_first_major_column(self, treated):
        fig, _ = plotting.phase_response_violins(
            treated, major_cols=("genotype", "zt"), split_col=None
        )
        assert {tr.name for tr in fig.data} == {"A", "B"}

    def test_the_violin_centre_is_the_point_on_the_line(self, treated):
        """One computation behind both figures — the property that lets them be read
        together. If these ever drift apart, one of them is lying."""
        summ = ps.summarize_phase_response(treated, by=("zt", *SERIES))
        _, drawn = plotting.phase_response_violins(
            treated,
            major_cols=("genotype", "zt", "pulse_duration_minutes"),
            split_col="pulse_intensity",
        )
        for _, row in summ.iterrows():
            sel = drawn[
                (drawn["zt"] == row["zt"])
                & (drawn["genotype"] == row["genotype"])
                & (drawn["pulse_intensity"] == row["pulse_intensity"])
            ]["response_hours"]
            assert float(sel.mean()) == pytest.approx(row["mean"])

    def test_a_fly_with_no_response_is_dropped_not_zeroed(self, treated):
        """Missing is never zero (ARCHITECTURE rule 9). A NaN counted as "no shift"
        drags every group mean toward zero by however many flies were arrhythmic."""
        spiked = treated.copy()
        spiked.loc[spiked.index[:3], "response_hours"] = np.nan
        _, drawn = plotting.phase_response_violins(
            spiked, major_cols=("genotype", "zt"), split_col=None
        )
        assert len(drawn) == len(treated) - 3
        assert drawn["response_hours"].notna().all()

    def test_an_empty_frame_draws_nothing_rather_than_raising(self):
        fig, drawn = plotting.phase_response_violins(pd.DataFrame())
        assert fig.data == ()
        assert drawn.empty


class TestTheDoseAxisIsSortedAsNumbers:
    """Coord values reach a page as STRINGS, so ``100`` sorts before ``20`` unless
    something coerces them back. The axis then reads 100, 20, 60 and the dose
    response it is drawn to show is scrambled."""

    def test_durations_order_by_value(self):
        df = pd.DataFrame(
            {
                "genotype": ["A"] * 6,
                "dose": ["100", "20", "60", "100", "20", "60"],
                "response_hours": [3.0, 1.0, 2.0, 3.1, 1.1, 2.1],
            }
        )
        fig, _ = plotting.phase_response_violins(
            df, major_cols=("genotype", "dose"), split_col=None
        )
        assert list(fig.layout.xaxis.categoryarray) == ["A · 20", "A · 60", "A · 100"]

    def test_a_genuinely_textual_column_still_orders_as_text(self):
        df = pd.DataFrame(
            {
                "genotype": ["per", "Mito", "Sr"],
                "response_hours": [1.0, 2.0, 3.0],
            }
        )
        fig, _ = plotting.phase_response_violins(
            df, major_cols=("genotype",), split_col=None
        )
        assert list(fig.layout.xaxis.categoryarray) == ["Mito", "Sr", "per"]

    def test_a_whole_number_loses_its_decimal_point(self):
        assert plotting._prc_join(("Mito", "20.0", 60.0)) == "Mito · 20 · 60"
        assert plotting._prc_join(("Mito", 21.5)) == "Mito · 21.5"


class TestTooFewPulseTimes:
    """What the page must not do with a one- or two-timepoint experiment, which is
    what the lab's own data currently is."""

    def test_the_count_of_pulse_times_is_readable_from_the_frame(self, treated):
        one = treated[treated["zt"] == 21.0]
        assert len({float(z) for z in one["zt"]}) == 1, (
            "the page keys its 'not enough timepoints' message off exactly this count"
        )
        assert len({float(z) for z in treated["zt"]}) == 3, (
            "and three is the floor a curve is allowed to be drawn at"
        )

    def test_the_curve_degrades_to_points_rather_than_raising(self, treated):
        """Belt and braces: if a caller draws it anyway, it must not invent a line
        between times that were never measured."""
        one = treated[treated["zt"] == 21.0]
        summ = ps.summarize_phase_response(one, by=("zt", *SERIES))
        fig = plotting.phase_response_curve(summ, series_cols=SERIES)
        assert list(fig.layout.xaxis.tickvals) == [21.0]
        for tr in fig.data:
            assert len(tr.x) == 1
