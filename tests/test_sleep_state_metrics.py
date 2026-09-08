"""The sleep-state metrics layer, and the figure semantics it exists to feed.

``core/sleep_state_metrics.py`` is new, but four docstrings in ``plotting.py``
had named it as their data source for a long time — nothing built the frames
those renderers described, so none of the Abhilash figures could be drawn at
all. These tests pin the contracts the paper specifies, concentrating on the
places where a plausible-looking alternative gives a different answer.
"""

import numpy as np
import pytest

import sleep_state_metrics as ssm

STATES = ("standard", "short", "intermediate", "long")


class TestProfiles:
    def test_covers_every_state_and_bin(self, states_ds):
        prof = ssm.state_profiles(states_ds, bin_size_min=30)
        assert set(prof["state"]) == {"activity", *STATES}
        per_series = prof.groupby(["id", "state"]).size().unique()
        assert list(per_series) == [48], "expected 48 thirty-minute bins per series"

    def test_sleep_is_minutes_per_hour(self, states_ds):
        """The paper's axis is min/h, so a bin can never exceed 60."""
        prof = ssm.state_profiles(states_ds, bin_size_min=30)
        sleep = prof[prof["state"] == "standard"]["value"]
        assert sleep.min() >= 0
        assert sleep.max() <= 60.0 + 1e-6

    def test_bin_width_does_not_change_the_daily_mean(self, states_ds):
        """Values are per-minute means scaled to the hour, so re-binning
        changes the resolution and not the level."""
        coarse = ssm.state_profiles(states_ds, bin_size_min=60)
        fine = ssm.state_profiles(states_ds, bin_size_min=15)
        for state in STATES:
            a = coarse[coarse["state"] == state]["value"].mean()
            b = fine[fine["state"] == state]["value"].mean()
            assert a == pytest.approx(b, rel=0.02), f"{state} shifted with bin width"

    def test_missing_minutes_are_excluded_not_counted_as_zero(self, states_ds):
        """A -1 sentinel must not be averaged in as "awake"."""
        ds = states_ds.copy(deep=True)
        ds["sleep"][0, :720] = -1  # (id, time)
        prof = ssm.state_profiles(ds, bin_size_min=30, include_activity=False)
        fly = str(ds["id"].values[0])
        row = prof[(prof["id"] == fly) & (prof["state"] == "standard")]
        assert (row["value"] >= 0).all(), "a negative sentinel leaked into a profile"


class TestNormalisedWaveforms:
    def test_each_state_peaks_at_one(self, states_ds):
        wf = ssm.compute_normalized_waveforms(states_ds)
        peaks = wf.groupby(["group", "state"])["mean_normalized"].max()
        assert np.allclose(peaks.values, 1.0)

    def test_normalises_after_averaging_flies_not_before(self, states_ds):
        """The paper's order is average-then-normalise. Normalising each fly
        first makes every fly's own peak 1.0, which flattens between-fly
        differences in profile shape and gives a different group trace. This
        checks the implemented order really is the paper's, by computing the
        wrong order explicitly and requiring the results to differ.
        """
        wf = ssm.compute_normalized_waveforms(states_ds, bin_size_min=30)
        prof = ssm.state_profiles(states_ds, bin_size_min=30, include_activity=False)

        state, group = "standard", wf["group"].iloc[0]
        sub = prof[(prof["state"] == state) & (prof["group"] == group)]
        per_fly_first = (
            sub.assign(
                scaled=sub.groupby("id")["value"].transform(lambda v: v / v.max())
            )
            .groupby("zt_bin_minute")["scaled"]
            .mean()
            .values
        )
        implemented = (
            wf[(wf["state"] == state) & (wf["group"] == group)]
            .sort_values("zt_bin_minute")["mean_normalized"]
            .values
        )
        assert not np.allclose(implemented, per_fly_first, atol=1e-6), (
            "average-then-normalise and normalise-then-average agree here, so "
            "this fixture cannot tell the two orders apart"
        )
        assert implemented.max() == pytest.approx(1.0)

    def test_scale_recovers_absolute_values(self, states_ds):
        wf = ssm.compute_normalized_waveforms(states_ds)
        stats = ssm.group_profiles(
            ssm.state_profiles(states_ds, include_activity=False)
        )
        for (group, state), sub in wf.groupby(["group", "state"]):
            raw = stats[(stats["group"] == group) & (stats["state"] == state)]
            assert sub["scale"].iloc[0] == pytest.approx(raw["mean"].max())


class TestInitiationProbability:
    def test_each_fly_and_state_sums_to_one(self, states_ds):
        """The paper divides by that fly's own bout count for that state, so a
        fly's curve is a distribution over the day. This is what makes the
        between-fly SEM comparable across states that differ tenfold in how
        many bouts they contain."""
        init = ssm.compute_initiation_probability(states_ds)
        sums = init.groupby(["id", "state"])["probability"].sum()
        assert np.allclose(sums.values, 1.0)

    def test_bins_are_labelled_one_to_twentyfour(self, states_ds):
        init = ssm.compute_initiation_probability(states_ds)
        assert sorted(init["bin_hour"].unique()) == list(range(1, 25))

    def test_a_fly_with_no_bouts_of_a_state_is_omitted_not_zeroed(self, states_ds):
        """Its probability is undefined, not zero. Emitting zeros would drag a
        group's P(long) toward zero for every fly that never slept long."""
        init = ssm.compute_initiation_probability(states_ds)
        counts = init.groupby("state")["id"].nunique()
        assert (counts > 0).all()
        assert (init["n_bouts"] > 0).all()

    def test_standard_counts_every_bout(self, states_ds):
        """'standard' is sleep by the unitary definition, so its denominator is
        the sum of the three exclusive states'."""
        init = ssm.compute_initiation_probability(states_ds)
        per_fly = init.groupby(["id", "state"])["n_bouts"].first().unstack()
        exclusive = per_fly[["short", "intermediate", "long"]].fillna(0).sum(axis=1)
        assert np.allclose(per_fly["standard"].values, exclusive.values)


class TestCircularStatistics:
    def test_concentration_is_bounded(self, states_ds):
        stats = ssm.circular_state_stats(states_ds)
        r = stats["r"].dropna()
        assert ((r >= 0) & (r <= 1)).all(), "r is a resultant length, so 0 <= r <= 1"

    def test_a_concentrated_profile_gives_a_phase_at_its_peak(self):
        """Direct check of the centre-of-mass port against a known answer:
        all the weight in one bin puts the mean phase in that bin."""
        theta = np.arange(48) * 7.5 + 3.75  # 30-min bin centres, degrees
        weights = np.zeros(48)
        weights[24] = 1.0  # ZT12
        phase, r = ssm._weighted_circular_mean(theta, weights)
        assert phase == pytest.approx(183.75)  # 12.25 h
        assert r == pytest.approx(1.0)

    def test_a_flat_profile_has_no_concentration(self):
        theta = np.arange(48) * 7.5
        phase, r = ssm._weighted_circular_mean(theta, np.ones(48))
        assert r == pytest.approx(0.0, abs=1e-9)

    def test_angular_deviation_matches_batschelet(self):
        for r in (0.0, 0.25, 0.5, 1.0):
            expected = np.rad2deg(np.sqrt(2 * (1 - r)))
            assert ssm._angular_deviation_deg(r) == pytest.approx(expected)

    def test_angle_doubling_recovers_a_bimodal_phase(self):
        """A profile with peaks near dawn and dusk averages, untransformed, to a
        mean phase in the middle of the day where the fly is doing nothing.
        Doubling is the paper's fix for exactly this."""
        theta = np.arange(48) * 7.5 + 3.75
        weights = np.zeros(48)
        weights[[4, 5, 40, 41]] = 1.0  # ~ZT2.5 and ~ZT20.5
        assert ssm._is_bimodal(weights)
        plain, _ = ssm._weighted_circular_mean(theta, weights)
        doubled, _ = ssm._weighted_circular_mean((2 * theta) % 360, weights)
        # The doubled mean, halved, lands on one of the two lobes; the plain
        # mean lands between them.
        candidates = [(doubled / 2) % 360, (doubled / 2 + 180) % 360]
        near_lobe = min(
            min(abs((c - t + 180) % 360 - 180) for t in (theta[4], theta[40]))
            for c in candidates
        )
        plain_dist = min(abs((plain - t + 180) % 360 - 180) for t in (theta[4], theta[40]))
        assert near_lobe < plain_dist

    def test_unimodal_profiles_are_left_alone(self):
        weights = np.zeros(48)
        weights[20:28] = 1.0
        assert not ssm._is_bimodal(weights)

    def test_gate_is_the_mean_phase_plus_minus_the_deviation(self, states_ds):
        stats = ssm.circular_state_stats(states_ds).dropna(
            subset=["mean_phase_h", "angular_deviation_h"]
        )
        assert not stats.empty
        for _, row in stats.head(20).iterrows():
            assert row["onset_h"] == pytest.approx(
                (row["mean_phase_h"] - row["angular_deviation_h"]) % 24
            )
            assert row["offset_h"] == pytest.approx(
                (row["mean_phase_h"] + row["angular_deviation_h"]) % 24
            )


class TestGroupGates:
    def test_phases_are_averaged_circularly(self):
        """The old gating plot took a plain arithmetic mean of onset and offset
        minutes, so any gate straddling the origin — long sleep under DD, the
        state the figure is about — got a mean on the far side of the clock.
        Two flies at ZT23 and ZT01 must average to midnight, not to noon.
        """
        import pandas as pd

        stats = pd.DataFrame(
            {
                "id": ["a", "b"],
                "group": ["g", "g"],
                "state": ["long", "long"],
                "mean_phase_h": [23.0, 1.0],
                "r": [0.5, 0.5],
                "angular_deviation_h": [2.0, 2.0],
                "onset_h": [21.0, 23.0],
                "offset_h": [1.0, 3.0],
                "doubled": [False, False],
            }
        )
        gates = ssm.group_gates(stats)
        mean_phase = float(gates["mean_phase_h"].iloc[0])
        assert min(mean_phase, 24 - mean_phase) == pytest.approx(0.0, abs=1e-6), (
            f"circular mean of ZT23 and ZT01 came out at {mean_phase:.2f} h"
        )

    def test_rebuilds_the_gate_around_the_circular_mean(self, states_ds):
        stats = ssm.circular_state_stats(states_ds)
        gates = ssm.group_gates(stats)
        assert not gates.empty
        for _, row in gates.iterrows():
            width = row["offset_h"] - row["onset_h"]
            assert (width % 24) == pytest.approx(2 * row["angular_deviation_h"], abs=1e-6)


class TestFigureContracts:
    """The renderers must accept exactly what this module emits."""

    def test_rose_plot_uses_profiles_not_bout_initiations(self, states_ds):
        """The rose plot's radius is the binned PROFILE. It used to be the
        fraction of bouts initiated per bin, which is a different quantity and
        a different figure (Fig 2), and is why the published rose plots could
        never be matched."""
        import plotting

        stats = ssm.group_profiles(ssm.state_profiles(states_ds, bin_size_min=30))
        fig = plotting.rose_plot(stats, "long", bin_size_min=30, phase_label="DD")
        # 48 wedges plus the two day/night background wedges.
        assert len(fig.data) == 50
        assert fig.layout.polar.angularaxis.direction == "clockwise"
        assert fig.layout.polar.angularaxis.rotation == 90

    def test_paper_colours_are_used(self):
        """Short sleep is orange and long sleep is blue in the paper. The old
        palette was Plotly's default cycle, which gave short sleep the paper's
        long-sleep blue and long sleep its activity red — inverting the two
        states a reader most needs to tell apart."""
        assert ssm.STATE_COLORS["short"].lower() == "#e8730c"
        assert ssm.STATE_COLORS["long"].lower() == "#3b4da0"
        assert ssm.STATE_COLORS["activity"].lower() == "#e03127"

    def test_gating_plot_draws_concentric_rings(self, states_ds):
        """Every per-fly arc used to be drawn at r = 1, collapsing the paper's
        one-ring-per-fly figure into a single overlapping band."""
        import plotting

        stats = ssm.circular_state_stats(states_ds)
        gates = ssm.group_gates(stats)
        one = stats[stats["group"] == stats["group"].iloc[0]]
        fig = plotting.polar_gating_plot(one, gates)
        radii = {float(t.r[0]) for t in fig.data if t.r is not None and len(t.r)}
        assert len(radii) > 4, f"expected many distinct radii, got {sorted(radii)}"

    def test_gate_arcs_never_run_backwards_through_the_day(self):
        """A linspace from onset to offset traverses the long way round whenever
        offset < onset, which is precisely the gates that straddle midnight."""
        import plotting

        arc = plotting._gate_arc(22.0, 2.0, 1.0, "#123456", width=2, alpha=1.0)
        hours = np.asarray(arc.theta) / 360.0 * 24.0
        # 22h -> 2h the short way is 4 hours of arc, not 20.
        spanned = np.abs(np.diff(np.unwrap(np.deg2rad(arc.theta)))).sum()
        assert np.rad2deg(spanned) == pytest.approx(60.0, abs=1.0)
        assert hours.min() >= 0 and hours.max() < 24

    def test_scalogram_pins_the_published_z_range(self, states_ds):
        """"All scalograms have a z axis scale that ranges from 0 to 1.5." A
        data-dependent range rescales each panel differently, so nothing can be
        compared by colour."""
        import plotting

        surfaces = {"long": np.random.default_rng(0).random((40, 100)) * 3}
        periods = np.geomspace(1, 32, 40)
        fig = plotting.sleep_state_scalogram(surfaces, periods)
        assert fig.data[0].zmin == 0.0
        assert fig.data[0].zmax == 1.5
        assert fig.data[0].colorscale is not None
        assert fig.layout.yaxis.type == "log"

    def test_period_amplitude_axis_is_log_and_data_ranged(self):
        """Shapes on a log axis take data units but their annotations take
        log10; letting add_vline label itself parked an annotation at x = 24,
        which autoranged the axis out to 10^24 and flattened every curve."""
        import plotting

        spectra = {"long": np.random.default_rng(0).random((6, 60)) + 0.5}
        periods = np.geomspace(1, 32, 60)
        fig = plotting.period_amplitude_plot(
            spectra, periods, n_bootstrap=20, ultradian_band=(2, 6)
        )
        computed = fig.full_figure_for_development(warn=False)
        lo, hi = computed.layout.xaxis.range
        assert computed.layout.xaxis.type == "log"
        assert 10**hi < 100, f"axis runs to {10**hi:.3g} h — an annotation is in data units"
        assert 10**lo > 0.5
