"""The performance rewrites give the answers the code they replaced gave.

Three recent commits traded a slow loop for a fast equivalent and said so in the
message: "identical on 3000 random series" (sleep gap filling), "equals xarray's
bit for bit" (curation's rolling mean), and the CWT's switch to PyWavelets' FFT
convolution at wavelet precision 16. None of the first two checks was kept, so the
next edit to any of them could break the equivalence without a test noticing.
These are those checks.

Where there is a reference implementation still in the tree it is the oracle:
``sleep_analysis._fill_short_gaps`` for the gap fill, xarray's own ``rolling`` for
the rolling mean, PyWavelets' direct convolution for the CWT. Inputs are real
movement and activity traces from example_data, with gaps punched in where the
real recording has none.
"""

import numpy as np
import pandas as pd
import pytest
import pywt
import xarray as xr

from clockwork.core import dam_utilities, periodograms, sleep_analysis


def _real_moving(example_ds, n_flies=8):
    """Real 0/1 movement traces, (fly, time), as sleep_analysis sees them."""
    act = example_ds["activity"].transpose("id", "time").values[:n_flies]
    return (act > 0).astype(float)


def _punch_gaps(trace, rng, n_gaps=40, max_len=8):
    """Missing-data runs (-1) of 1..max_len minutes, anywhere including the ends."""
    out = trace.copy()
    for _ in range(n_gaps):
        length = int(rng.integers(1, max_len + 1))
        start = int(rng.integers(0, len(out) - length + 1))
        out[start : start + length] = -1
    out[:3] = -1  # a gap touching the start is never filled
    out[-2:] = -1  # nor one touching the end
    return out


# ---------------------------------------------------------------------------
# Sleep gap filling (commit b686793)
# ---------------------------------------------------------------------------


class TestFillShortGaps:
    @staticmethod
    def _reference(values, max_gap):
        return sleep_analysis._fill_short_gaps(pd.Series(values), max_gap=max_gap).to_numpy()

    @pytest.mark.parametrize("max_gap", [1, 4, 6])
    def test_matches_the_loop_on_real_traces(self, example_ds, max_gap):
        rng = np.random.default_rng(max_gap)
        for trace in _real_moving(example_ds):
            gappy = _punch_gaps(trace, rng)
            np.testing.assert_array_equal(
                sleep_analysis._fill_short_gaps_array(gappy, max_gap=max_gap),
                self._reference(gappy, max_gap),
            )

    @pytest.mark.parametrize(
        "values",
        [
            [0, 0, 0, 0],
            [1, 1, 1],
            [0],
            [-1],
            [-1, -1, -1],
            [0, -1, 0],
            [0, -1, 1],  # different neighbours: left alone
            [1, -1, -1, -1, -1, 1],  # exactly max_gap: filled
            [1, -1, -1, -1, -1, -1, 1],  # one over: left alone
            [-1, 0, -1, -1, 0, -1],
        ],
        ids=repr,
    )
    def test_matches_the_loop_on_edge_cases(self, values):
        values = np.array(values, dtype=float)
        np.testing.assert_array_equal(
            sleep_analysis._fill_short_gaps_array(values, max_gap=4),
            self._reference(values, 4),
        )

    def test_does_not_modify_its_input(self):
        values = np.array([1, -1, 1], dtype=float)
        sleep_analysis._fill_short_gaps_array(values)
        np.testing.assert_array_equal(values, [1, -1, 1])


# ---------------------------------------------------------------------------
# Curation's rolling mean (commit b686793)
# ---------------------------------------------------------------------------


class TestCenteredRollingMean:
    @staticmethod
    def _xarray(values, window):
        da = xr.DataArray(values, dims=("time", "id"))
        return da.rolling(time=window, center=True, min_periods=1).mean().values

    @pytest.mark.parametrize("window", [1, 2, 5, 60, 1440, 1441])
    def test_equals_xarray_on_real_movement_with_nan(self, example_ds, window):
        rng = np.random.default_rng(window)
        moving = _real_moving(example_ds).T  # (time, fly)
        # Missing data is NaN here (curate_dead_animals maps -1 to NaN first).
        gappy = np.where(_punch_gaps(moving[:, 0], rng) == -1, np.nan, moving[:, 0])
        moving[:, 0] = gappy
        moving[5000:5300, 1] = np.nan  # a long run: windows entirely missing
        # Movement is 0/1, so the running sums are exact: bit for bit, as claimed.
        np.testing.assert_array_equal(
            dam_utilities._centered_rolling_mean(moving, window), self._xarray(moving, window)
        )

    def test_non_integer_values_agree_to_rounding(self, example_ds):
        act = example_ds["activity"].transpose("time", "id").values[:, :4].astype(float)
        np.testing.assert_allclose(
            dam_utilities._centered_rolling_mean(act, 90), self._xarray(act, 90), rtol=1e-9
        )

    def test_blocks_of_flies_do_not_change_the_answer(self, example_ds):
        moving = _real_moving(example_ds, n_flies=10).T
        np.testing.assert_array_equal(
            dam_utilities._centered_rolling_mean(moving, 120, block=3),
            dam_utilities._centered_rolling_mean(moving, 120, block=256),
        )


def test_curation_gives_the_same_answer_on_a_datetime_axis(example_ds):
    """The rewrite's other claim: curation used to crash on a datetime time axis."""
    small = example_ds.isel(id=slice(0, 6))
    start = pd.Timestamp(str(small["start_datetime"].values[0]))
    as_datetime = small.assign_coords(
        time=pd.date_range(start, periods=small.sizes["time"], freq="min")
    )
    live_int, dead_int, *_ = dam_utilities.curate_dead_animals(small)
    live_dt, dead_dt, *_ = dam_utilities.curate_dead_animals(as_datetime)
    assert list(live_dt["id"].values) == list(live_int["id"].values)
    assert list(dead_dt["id"].values) == list(dead_int["id"].values)


# ---------------------------------------------------------------------------
# The CWT transform: FFT convolution, wavelet precision 16, float64
# ---------------------------------------------------------------------------


def _real_signals(example_ds, n=6, n_time=3000):
    act = example_ds["activity"].transpose("id", "time").values[:n, :n_time]
    return [np.asarray(row, dtype=np.float64) for row in act]


SCALES = np.geomspace(20, 400, 24)


def _circadian_scales():
    """The scales the page-default circadian CWT uses on 1-minute data (~980-2140)."""
    scales, _periods = periodograms._cwt_period_grid(
        16.0, 36.0, 1.0, periodograms.DEFAULT_CWT_RESOLUTION
    )
    return scales


def test_cwt_powers_equal_per_fly_cwt_in_order(example_ds):
    """cwt_powers is compute_cwt on a thread pool: same values, same order."""
    signals = _real_signals(example_ds)
    powers = list(periodograms.cwt_powers(signals, SCALES, n_workers=4))
    assert len(powers) == len(signals)
    for sig, power in zip(signals, powers):
        assert power.dtype == np.float32
        expected = periodograms.compute_cwt(sig, SCALES)[2].astype(np.float32)
        np.testing.assert_array_equal(power, expected)


def test_the_transform_is_float64_whatever_the_input(example_ds):
    """The worker hands over float32; the transform must not inherit it."""
    sig = _real_signals(example_ds, n=1)[0].astype(np.float32)
    coef, _f, power = periodograms.compute_cwt(sig, SCALES)
    assert coef.dtype == np.complex128
    assert power.dtype == np.float64


def test_fft_equals_direct_convolution(example_ds):
    """method="fft" is a faster way to compute the same transform, not a new one."""
    sig = _real_signals(example_ds, n=1, n_time=9 * 1440)[0]
    scales = _circadian_scales()
    _c, _f, fast = periodograms.compute_cwt(sig, scales)
    direct, _ = pywt.cwt(
        sig, scales, "cmor1.5-1.0", method="conv",
        precision=periodograms.CWT_WAVELET_PRECISION,
    )
    direct = np.abs(direct) ** 2
    np.testing.assert_allclose(fast, direct, rtol=0, atol=1e-9 * direct.max())


def test_wavelet_precision_is_converged_at_circadian_scales(example_ds):
    """Why 16: below it, pywt's tabulated wavelet is too coarse for these scales.

    Against a 4x finer table (precision 18), the time-averaged spectrum of a real
    fly must agree to 1% at every scale. Precision 10 (PyWavelets <= 1.8) and 12
    (the 1.9 default) both miss this by a wide margin on these scales, which is
    what the second assertion holds, so the test cannot pass vacuously.
    """
    sig = _real_signals(example_ds, n=1, n_time=9 * 1440)[0]
    sig = sig - sig.mean()
    scales = _circadian_scales()

    def spectrum(precision):
        coef, _ = pywt.cwt(sig, scales, "cmor1.5-1.0", method="fft", precision=precision)
        return (np.abs(coef) ** 2).mean(axis=1)

    reference = spectrum(18)
    rel_err = lambda p: np.max(np.abs(spectrum(p) - reference) / reference)  # noqa: E731
    assert rel_err(periodograms.CWT_WAVELET_PRECISION) < 0.01
    assert rel_err(12) > 0.05


def test_worker_count_leaves_one_core_and_never_exceeds_tasks(monkeypatch):
    monkeypatch.setattr(periodograms.mp, "cpu_count", lambda: 16)
    assert periodograms.get_optimal_workers(100) == 15
    assert periodograms.get_optimal_workers(3) == 3
    monkeypatch.setattr(periodograms.mp, "cpu_count", lambda: 1)
    assert periodograms.get_optimal_workers(100) == 1


def test_there_is_exactly_one_cwt_implementation():
    """Every CWT in ClockWork (circadian, scalogram, sleep states) is compute_cwt.

    A second pywt.cwt call, or a ptwt/torch one, is how two sets of numbers for the
    same data came about before: the GPU path and the CPU path disagreed and the
    fallback between them was silent.
    """
    import ast

    from conftest import PKG_ROOT

    calls = []
    for path in PKG_ROOT.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        parents = {}
        for node in ast.walk(tree):
            for child in ast.iter_child_nodes(node):
                parents[child] = node
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert not any(a.name.split(".")[0] == "ptwt" for a in node.names), path
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "cwt"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id in ("pywt", "ptwt")
            ):
                fn = node
                while fn in parents and not isinstance(fn, ast.FunctionDef):
                    fn = parents[fn]
                calls.append((path.name, getattr(fn, "name", "<module>")))
    assert calls == [("periodograms.py", "compute_cwt")], calls
