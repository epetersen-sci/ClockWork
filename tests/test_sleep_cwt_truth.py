"""Ground truth for the sleep-state CWT and the chi-squared periodogram.

The Abhilash page was abandoned partly because it was "impossible to figure out
if the CWT analysis was working and resulting in the same results as what is in
this paper". Nothing anywhere fed the transform a signal whose answer was known
in advance, so there was no way to tell a correct scalogram from a plausible
one. These tests are that missing check: synthesise signals with known periodic
content, push them through the same code path the page uses, and assert on what
comes back.

Two of them pin the fix for a real bias. ``WaveletComp::analyze.wavelet`` — the
function the paper's scalograms come from — computes
``Power = Mod(Wave)^2 / scale`` (Liu et al. 2007). Our CWT returned a bare
``|W|^2``, which is biased toward long periods, so ultradian rhythms were
reported roughly an order of magnitude weaker than circadian ones relative to
the truth, and peak periods were pulled long. That is exactly the kind of
discrepancy that would make the output irreconcilable with the paper.
"""

import numpy as np
import pytest

from periodograms import _chi_sq_periodogram, _preprocess_and_compute_cwt

BIN_MIN = 5  # the sleep-state path bins to 5 minutes
BINS_PER_DAY = 24 * 60 // BIN_MIN
N_DAYS = 9  # the paper's DD recording length


def _signal(periods_h, amps, n_days=N_DAYS, noise=0.0, seed=0):
    n = n_days * BINS_PER_DAY
    t_h = np.arange(n) * BIN_MIN / 60.0
    x = np.zeros(n)
    for period, amp in zip(periods_h, amps):
        x += amp * np.sin(2 * np.pi * t_h / period)
    if noise:
        x += np.random.default_rng(seed).normal(0, noise, n)
    return x.astype(np.float32), np.arange(n, dtype=np.int64) * BIN_MIN


def _spectrum(x, t, min_p=1, max_p=32, rectify=True):
    """Time-averaged normalised spectrum, mirroring sleep_cwt_analysis exactly."""
    result = _preprocess_and_compute_cwt(
        x, t, min_p, max_p, cwt_method="ridge", resolution=1 / 100, wavelet="cmor1.5-1.0"
    )
    assert result is not None
    power = result["power"].astype(np.float64)
    periods = result["periods_hours"]
    if rectify:
        power = power / (periods * 60.0 / BIN_MIN)[:, None]
    power = power / power.mean()
    return periods, power.mean(axis=1)


def _peak_period(periods, spectrum, near=None, window=0.35):
    """Period of the largest value, optionally restricted to near a target."""
    if near is not None:
        mask = np.abs(np.log(periods / near)) < window
        idx = np.argmax(np.where(mask, spectrum, -np.inf))
    else:
        idx = np.argmax(spectrum)
    return float(periods[idx])


class TestPeakRecovery:
    """A component at a known period must be found at that period."""

    @pytest.mark.parametrize("true_period", [3.0, 8.0, 12.0, 24.0])
    def test_single_component_is_found_within_two_percent(self, true_period):
        x, t = _signal([true_period], [1.0])
        periods, spec = _spectrum(x, t)
        found = _peak_period(periods, spec, near=true_period)
        assert found == pytest.approx(true_period, rel=0.02), (
            f"a pure {true_period} h signal peaked at {found:.2f} h"
        )

    def test_rectification_removes_a_long_period_bias(self):
        """Unrectified |W|^2 pulls the peak long; the divide-by-scale fixes it.

        Measured on a pure 24-h signal: 24.51 h unrectified against 24.00 h
        rectified. Half a percent sounds small until it is a circadian period
        estimate, and the same bias is what distorts the whole surface.
        """
        x, t = _signal([24.0], [1.0])
        p_raw, s_raw = _spectrum(x, t, rectify=False)
        p_fix, s_fix = _spectrum(x, t, rectify=True)
        err_raw = abs(_peak_period(p_raw, s_raw, near=24) - 24.0)
        err_fix = abs(_peak_period(p_fix, s_fix, near=24) - 24.0)
        assert err_fix < err_raw, "rectifying should move the peak toward the truth"
        assert err_fix < 0.1, f"rectified peak is still {err_fix:.2f} h off"

    def test_two_components_are_both_resolved(self):
        x, t = _signal([24.0, 3.0], [1.0, 1.0])
        periods, spec = _spectrum(x, t)
        assert _peak_period(periods, spec, near=24) == pytest.approx(24.0, rel=0.02)
        assert _peak_period(periods, spec, near=3) == pytest.approx(3.0, rel=0.05)


class TestUltradianVersusCircadianWeighting:
    """The discriminating case, and the reason the paper could not be matched."""

    def test_equal_amplitudes_get_comparable_power(self):
        """A 24-h and a 3-h component of EQUAL amplitude should read comparably.

        Unrectified they do not: the 24-h component comes out about 7x stronger,
        purely from the scale bias. That understates ultradian rhythms by close
        to an order of magnitude relative to circadian — and the ultradian band
        is the entire subject of the paper's Figures 5 and 6.
        """
        x, t = _signal([24.0, 3.0], [1.0, 1.0])

        p_raw, s_raw = _spectrum(x, t, rectify=False)
        ratio_raw = s_raw[np.argmin(abs(p_raw - 3))] / s_raw[np.argmin(abs(p_raw - 24))]

        p_fix, s_fix = _spectrum(x, t, rectify=True)
        ratio_fix = s_fix[np.argmin(abs(p_fix - 3))] / s_fix[np.argmin(abs(p_fix - 24))]

        assert ratio_raw < 0.3, (
            "expected the unrectified spectrum to under-weight the short period; "
            f"got a 3h/24h ratio of {ratio_raw:.2f}"
        )
        assert 0.5 < ratio_fix < 2.0, (
            "equal input amplitudes should give comparable rectified power; got a "
            f"3h/24h ratio of {ratio_fix:.2f}"
        )


class TestNormalisation:
    """What "normalised to the average amplitude of the surface" has to mean."""

    def test_surface_mean_is_one(self):
        x, t = _signal([24.0, 3.0], [1.0, 0.5])
        _, spec = _spectrum(x, t)
        # The time-average of a surface whose grand mean is 1 also averages to 1.
        assert spec.mean() == pytest.approx(1.0, rel=1e-6)

    def test_spectrum_is_invariant_to_signal_amplitude(self):
        """Normalising per fly is what lets flies be averaged together: a fly
        that slept twice as much must not dominate the group surface."""
        _, weak = _spectrum(*_signal([24.0], [1.0]))
        _, strong = _spectrum(*_signal([24.0], [5.0]))
        assert np.allclose(weak, strong, rtol=1e-6)


class TestChiSquaredPeriodogram:
    """The paper's rhythmicity test — ``zeitgebr::chi_sq_periodogram``."""

    PERIODS = np.arange(16, 32.0001, 20 / 60.0)  # phase's defaults: 16-32 h, 20 min

    def _run(self, x):
        return _chi_sq_periodogram(x, self.PERIODS, sampling_min=BIN_MIN)

    @pytest.mark.parametrize("true_period", [20.0, 24.0, 28.0])
    def test_finds_the_right_period(self, true_period):
        x, _ = _signal([true_period], [1.0])
        res = self._run(x)
        peak = self.PERIODS[int(np.nanargmax(res["adjusted"]))]
        assert peak == pytest.approx(true_period, abs=0.34)  # one 20-min step

    def test_a_real_rhythm_is_significant(self):
        x, _ = _signal([24.0], [1.0], noise=0.5, seed=3)
        res = self._run(x)
        assert np.nanmax(res["adjusted"]) > 0

    def test_white_noise_is_not_significant(self):
        """The negative control. Without it a periodogram that always fires
        looks like a working one, since real data is rhythmic."""
        x = np.random.default_rng(11).normal(0, 1, N_DAYS * BINS_PER_DAY)
        res = _chi_sq_periodogram(x, self.PERIODS, sampling_min=BIN_MIN)
        assert np.nanmax(res["adjusted"]) < 0, (
            "white noise cleared the significance threshold, so the threshold or "
            "its degrees of freedom are wrong"
        )

    def test_adjusted_is_power_minus_threshold(self):
        """The paper plots adjusted power, which is why its critical line is a
        flat zero rather than a period-dependent curve."""
        x, _ = _signal([24.0], [1.0])
        res = self._run(x)
        assert np.allclose(res["adjusted"], res["power"] - res["threshold"], equal_nan=True)

    def test_threshold_rises_with_period(self):
        """Degrees of freedom are the period IN SAMPLES, so a longer trial
        period is tested against a higher bar."""
        x, _ = _signal([24.0], [1.0])
        res = self._run(x)
        assert res["threshold"][-1] > res["threshold"][0]


class TestFullPipeline:
    """The wiring, on a small real-shaped dataset."""

    def test_produces_one_full_range_surface_per_state(self, states_ds):
        from periodograms import sleep_cwt_analysis

        out = sleep_cwt_analysis(
            states_ds, states=("standard", "long"), phase="DD", full_range=(1, 32)
        )
        for state in ("standard", "long"):
            surf = out[f"sleep_cwt_{state}_full_avg_surface"]
            periods = out[f"sleep_cwt_{state}_full_period_axis"].values
            assert surf.ndim == 2
            # One surface spanning the whole range, not two narrow bands: that
            # is what makes the z scale comparable to the paper's 0-1.5.
            assert periods.min() < 1.1 and periods.max() > 28
            assert float(np.nanmean(surf.values)) == pytest.approx(1.0, rel=0.02)

    def test_narrow_band_mode_still_available(self, states_ds):
        """full_range=None restores the old split-band behaviour, which is only
        useful for reproducing an older run."""
        from periodograms import sleep_cwt_analysis

        out = sleep_cwt_analysis(
            states_ds, states=("long",), phase="DD", full_range=None
        )
        assert "sleep_cwt_long_circadian_avg_surface" in out.data_vars
        assert "sleep_cwt_long_ultradian_avg_surface" in out.data_vars

    def test_chi_squared_runs_on_the_amplitude_series(self, states_ds):
        from periodograms import sleep_cwt_analysis, ultradian_rhythmicity_chi_sq

        cwt = sleep_cwt_analysis(states_ds, states=("long",), phase="DD")
        chi = ultradian_rhythmicity_chi_sq(cwt, states=("long",))
        assert "ultra_chisq_adjusted_long" in chi.data_vars
        assert chi["ultra_chisq_adjusted_long"].sizes["id"] == states_ds.sizes["id"]

    def test_a_phase_view_does_not_produce_a_nan_surface(self, states_ds):
        """A select_phase() view is float with NaN out of phase, not int8 with
        -1, and the old missing-value guard only looked for negatives — so
        every cell of the averaged surface came back NaN. Guarding both
        representations, and then taking each fly's longest clean run, is what
        keeps a pre-sliced dataset usable."""
        from dam_utilities import select_phase
        from periodograms import sleep_cwt_analysis

        view, used = select_phase(states_ds, phase="DD")
        assert np.isnan(view["sleep_long"].values).any(), (
            "fixture precondition: the view should carry NaN out of phase"
        )
        out = sleep_cwt_analysis(view, states=("long",), phase=used, full_range=(1, 32))
        surface = out["sleep_cwt_long_full_avg_surface"].values
        assert not np.isnan(surface).any(), "the averaged surface contains NaN"
        assert float(surface.mean()) == pytest.approx(1.0, rel=0.02)

    def test_the_masked_epoch_is_excluded_not_zero_filled(self, states_ds):
        """The surface should span the in-phase epoch only. Carrying the
        out-of-phase minutes through as zeros would both halve the effective
        signal and put a step at the boundary."""
        from dam_utilities import select_phase
        from periodograms import sleep_cwt_analysis

        view, used = select_phase(states_ds, phase="DD")
        out = sleep_cwt_analysis(view, states=("long",), phase=used, full_range=(1, 32))
        n_bins = out["sleep_cwt_long_full_avg_surface"].shape[1]
        in_phase_minutes = int(np.isfinite(view["sleep_long"].values[0]).sum())
        assert n_bins == pytest.approx(in_phase_minutes / 5, rel=0.02), (
            f"surface spans {n_bins} five-minute bins but only "
            f"{in_phase_minutes} minutes are in phase"
        )
