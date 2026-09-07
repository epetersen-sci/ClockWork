"""Degenerate inputs the Sleep states page has to survive.

The page is reachable with whatever dataset happens to be loaded, so every one
of these is a state a user can actually put it in: one fly left after a group
filter, a state nobody entered, an epoch too short to transform. None of them
should raise — an empty or annotated figure is a fine answer, a traceback on a
page is not.
"""

import numpy as np
import pandas as pd
import pytest

import plotting
import sleep_state_metrics as ssm

STATES = ("standard", "short", "intermediate", "long")


@pytest.fixture
def one_fly(states_ds):
    return states_ds.isel(id=[0])


class TestSingleFly:
    """A group filter can narrow to one fly, which kills every SEM and every
    bootstrap resample that assumes more than one."""

    def test_profiles_and_waveforms(self, one_fly):
        prof = ssm.state_profiles(one_fly, bin_size_min=30)
        assert not prof.empty
        stats = ssm.group_profiles(prof)
        # pandas .sem() of a single value is NaN, not 0 — the figures have to
        # tolerate a NaN band rather than dropping the trace.
        assert stats["sem"].isna().all()
        assert not ssm.compute_normalized_waveforms(one_fly).empty

    def test_figures_render(self, one_fly):
        stats = ssm.group_profiles(ssm.state_profiles(one_fly, bin_size_min=30))
        assert plotting.state_profile_plot(stats).data
        assert plotting.rose_plot_with_activity(stats).data
        circular = ssm.circular_state_stats(one_fly)
        gates = ssm.group_gates(circular)
        assert gates["n"].max() == 1
        assert plotting.polar_gating_plot(circular, gates).data

    def test_bootstrap_figures_survive_n_of_one(self):
        """One fly means the CI collapses onto the mean; it must not raise."""
        spectra = {"long": np.linspace(0.5, 2.0, 40)[np.newaxis, :]}
        periods = np.geomspace(1, 32, 40)
        fig = plotting.period_amplitude_plot(spectra, periods, n_bootstrap=10)
        assert fig.data
        fig = plotting.ultradian_amplitude_plot(
            {"long": np.linspace(0.5, 2.0, 60)[np.newaxis, :]}, n_bootstrap=10
        )
        assert fig.data


class TestEmptyAndMissingInputs:
    def test_every_figure_handles_an_empty_frame(self):
        empty = pd.DataFrame(
            columns=["group", "state", "zt_bin_minute", "mean", "sem", "n"]
        )
        for fig in (
            plotting.state_profile_plot(empty),
            plotting.rose_plot_with_activity(empty),
            plotting.rose_plot(empty, "long"),
            plotting.initiation_probability_plot(
                pd.DataFrame(columns=["group", "state", "bin_hour", "mean", "sem"])
            ),
            plotting.polar_gating_plot(pd.DataFrame(columns=["id", "group", "state"])),
            plotting.sleep_state_scalogram({}, np.array([1.0])),
            plotting.period_amplitude_plot({}, np.array([1.0])),
            plotting.ultradian_amplitude_plot({}),
            plotting.chi_sq_periodogram_plot(None),
        ):
            # An annotation-only figure is the documented "nothing to show".
            assert fig.layout.annotations or fig.data

    def test_metrics_handle_a_dataset_with_no_states(self, master_ds):
        """master_ds has the masks but they are all zero and there is no bout
        table, which is what a dataset looks like before Sleep analysis runs."""
        stripped = master_ds.drop_vars(
            [v for v in ("duration", "start_time", "sleep_state") if v in master_ds.data_vars]
        )
        assert ssm.compute_initiation_probability(stripped).empty
        # Profiles still work; the states are simply flat.
        assert not ssm.state_profiles(stripped).empty

    def test_a_state_nobody_entered_is_dropped_not_zeroed(self, states_ds):
        """A flat-zero state has no maximum to normalise to. Emitting zeros
        would draw a line along the axis that reads as data."""
        ds = states_ds.copy(deep=True)
        ds["sleep_long"] = ds["sleep_long"] * 0
        waveforms = ssm.compute_normalized_waveforms(ds)
        assert "long" not in set(waveforms["state"]), (
            "a state with no sleep at all was normalised anyway"
        )
        assert "short" in set(waveforms["state"])

    def test_profiles_without_an_activity_variable(self, states_ds):
        ds = states_ds.drop_vars("activity")
        prof = ssm.state_profiles(ds, bin_size_min=30)
        assert "activity" not in set(prof["state"])
        stats = ssm.group_profiles(prof)
        # The rose row drops the activity panel and the overlay rather than
        # raising on a missing series.
        fig = plotting.rose_plot_with_activity(stats)
        assert fig.data


class TestCircularEdges:
    def test_a_fly_that_never_slept_gets_no_phase(self):
        phase, r = ssm._weighted_circular_mean(np.arange(48) * 7.5, np.zeros(48))
        assert np.isnan(phase) and np.isnan(r)

    def test_all_nan_weights(self):
        phase, r = ssm._weighted_circular_mean(np.arange(48) * 7.5, np.full(48, np.nan))
        assert np.isnan(phase) and np.isnan(r)

    def test_group_gates_ignores_flies_with_no_phase(self):
        stats = pd.DataFrame(
            {
                "id": ["a", "b"],
                "group": ["g", "g"],
                "state": ["long", "long"],
                "mean_phase_h": [np.nan, 6.0],
                "r": [np.nan, 0.5],
                "angular_deviation_h": [np.nan, 2.0],
                "onset_h": [np.nan, 4.0],
                "offset_h": [np.nan, 8.0],
                "doubled": [False, False],
            }
        )
        gates = ssm.group_gates(stats)
        assert len(gates) == 1
        assert gates["n"].iloc[0] == 1
        assert gates["mean_phase_h"].iloc[0] == pytest.approx(6.0, abs=0.01)

    def test_bimodal_detector_on_degenerate_input(self):
        assert not ssm._is_bimodal(np.zeros(48))
        assert not ssm._is_bimodal(np.array([1.0, 2.0]))  # too few bins
        assert not ssm._is_bimodal(np.full(48, np.nan))


class TestShortRecordings:
    def test_an_epoch_too_short_to_transform_returns_empty(self, states_ds):
        """Fewer bins than the CWT needs must yield an empty Dataset, not a
        crash — the page checks `len(data_vars)` and says so."""
        from periodograms import sleep_cwt_analysis

        tiny = states_ds.isel(time=slice(0, 30))
        out = sleep_cwt_analysis(tiny, states=("long",), phase="DD", full_range=(1, 32))
        assert len(out.data_vars) == 0

    def test_chi_squared_on_a_series_shorter_than_the_period(self):
        """Trial periods longer than the series have no complete cycle. They
        must come back NaN rather than as a spuriously huge Qp."""
        from periodograms import _chi_sq_periodogram

        short = np.sin(np.linspace(0, 4 * np.pi, 40))
        periods = np.array([16.0, 24.0, 32.0])
        res = _chi_sq_periodogram(short, periods, sampling_min=5)
        # 16 h at 5-min sampling is 192 samples, well past the 40 available.
        assert np.all(np.isnan(res["power"]))

    def test_chi_squared_on_an_empty_series(self):
        from periodograms import _chi_sq_periodogram

        res = _chi_sq_periodogram(np.array([]), np.array([24.0]), sampling_min=5)
        assert np.isnan(res["power"]).all()

    def test_chi_squared_on_a_flat_series(self):
        """Zero variance means the denominator is zero. No division warning,
        no inf."""
        from periodograms import _chi_sq_periodogram

        flat = np.ones(2000)
        res = _chi_sq_periodogram(flat, np.array([24.0]), sampling_min=5)
        assert not np.isinf(res["power"]).any()


class TestPageWithDegenerateData:
    """The page itself, on the same shapes."""

    def test_renders_with_a_single_fly(self, app, states_ds):
        at = app(ds=states_ds.isel(id=[0]), page="sleep_states")
        assert not at.exception, f"single fly raised: {at.exception}"

    def test_renders_with_no_bout_table(self, app, states_ds):
        stripped = states_ds.drop_vars(
            [v for v in ("duration", "start_time", "end_time", "sleep_state")
             if v in states_ds.data_vars]
        )
        at = app(ds=stripped, page="sleep_states")
        assert not at.exception
        at.session_state["sleep_states_tab"] = "Initiation (Fig 2)"
        at = at.run()
        assert not at.exception, "the initiation tab needs the bout table but must not crash"
        assert at.warning, "it should say why initiation probability is unavailable"

    def test_renders_on_the_ld_epoch(self, app, states_ds):
        """Every default here is DD, so LD is the branch nobody exercises.

        Selecting it needs the radio's explicit key — the page sets one for
        exactly this reason, since an auto-generated widget id is unreachable.
        """
        at = app(ds=states_ds, page="sleep_states")
        at.session_state["sleep_states_phase"] = "LD"
        at = at.run()
        assert not at.exception, f"the LD epoch raised: {at.exception}"
        # Confirm the switch actually took, or the test proves nothing.
        assert any("LD epoch" in c.value for c in at.caption), (
            "the page is still showing DD, so the LD branch was not exercised"
        )
        assert not at.error

    def test_ld_epoch_renders_every_tab(self, app, states_ds):
        at = app(ds=states_ds, page="sleep_states")
        at.session_state["sleep_states_phase"] = "LD"
        for tab in ("Initiation (Fig 2)", "Rose & gating (Fig 3)"):
            at.session_state["sleep_states_tab"] = tab
            at = at.run()
            assert not at.exception, f"{tab} raised on LD: {at.exception}"
