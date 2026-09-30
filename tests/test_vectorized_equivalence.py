"""The performance rewrites give the answers the code they replaced gave.

Three recent commits traded a slow loop for a fast equivalent and said so in the
message: "identical on 3000 random series" (sleep gap filling), "equals xarray's
bit for bit" (curation's rolling mean), "matches per-fly cwt_gpu to float32 FFT
rounding" (batched CWT). None of those checks was kept, so the next edit to any of
them could break the equivalence without a test noticing. These are those checks.

Where there is a reference implementation still in the tree it is the oracle:
``sleep_analysis._fill_short_gaps`` for the gap fill, xarray's own ``rolling`` for
the rolling mean, per-fly ``cwt_gpu`` for the batched CWT. Inputs are real
movement and activity traces from example_data, with gaps punched in where the
real recording has none.
"""

import numpy as np
import pandas as pd
import pytest
import pywt
import xarray as xr

import dam_utilities
import periodograms
import sleep_analysis


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
# Batched CWT (commit 0a533f3) and the GPU worker count (commit 96c239a)
# ---------------------------------------------------------------------------


def _real_signals(example_ds, n=6, n_time=3000):
    act = example_ds["activity"].transpose("id", "time").values[:n, :n_time]
    return [np.asarray(row, dtype=np.float32) for row in act]


SCALES = np.geomspace(20, 400, 24)


def test_cpu_cwt_powers_equal_per_fly_cwt_in_order(example_ds, monkeypatch):
    """Off the GPU, cwt_powers is cwt_gpu on a thread pool — same values, same order."""
    monkeypatch.setattr(periodograms, "PTWT_AVAILABLE", False)
    signals = _real_signals(example_ds)
    powers = list(periodograms.cwt_powers(signals, SCALES, n_workers=4))
    assert len(powers) == len(signals)
    for sig, power in zip(signals, powers):
        assert power.dtype == np.float32
        np.testing.assert_array_equal(power, periodograms.cwt_gpu(sig, SCALES)[2].astype(np.float32))


class _FakeTensor:
    """Just enough of a torch tensor for cwt_powers, backed by numpy."""

    def __init__(self, a):
        self.a = np.asarray(a)

    def cuda(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.a

    def contiguous(self):
        return self

    def permute(self, *axes):
        return _FakeTensor(self.a.transpose(axes))

    def __pow__(self, k):
        return _FakeTensor(self.a**k)


class _FakeTorch:
    """torch + ptwt for a machine with no GPU: runs PyWavelets, fails big batches."""

    def __init__(self, max_batch, free_bytes):
        self.max_batch = max_batch
        self.batches = []
        self.cuda = self
        self._free = free_bytes

    # torch
    def from_numpy(self, a):
        return _FakeTensor(a)

    def abs(self, t):
        return _FakeTensor(np.abs(t.a))

    def mem_get_info(self):
        return (self._free, self._free)

    def empty_cache(self):
        pass

    # ptwt
    def cwt(self, data_t, scales_t, wavelet, sampling_period=1):
        batch = 1 if data_t.a.ndim == 1 else data_t.a.shape[0]
        self.batches.append(batch)
        if batch > self.max_batch:
            raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
        coef, freqs = pywt.cwt(data_t.a, scales_t.a, wavelet, sampling_period=sampling_period)
        return _FakeTensor(coef), freqs


def test_out_of_memory_halves_the_batch_and_keeps_every_fly(example_ds, monkeypatch):
    """The GPU path's memory handling, exercised without a GPU.

    The fake reports room for all six signals in one batch but fails any batch
    over two, so cwt_powers must halve 6 -> 3 -> 1... and still return every
    fly's surface, in order, equal to the per-fly transform.
    """
    fake = _FakeTorch(max_batch=2, free_bytes=10**12)
    monkeypatch.setattr(periodograms, "PTWT_AVAILABLE", True)
    monkeypatch.setattr(periodograms, "torch", fake)
    monkeypatch.setattr(periodograms, "ptwt", fake)
    signals = _real_signals(example_ds)

    powers = list(periodograms.cwt_powers(signals, SCALES))

    # Tried all six (sized from "free" memory), then halved until one fit.
    assert fake.batches[:3] == [6, 3, 1]
    assert sum(b for b in fake.batches if b <= fake.max_batch) == len(signals)
    assert len(powers) == len(signals)
    for sig, power in zip(signals, powers):
        expected = np.abs(pywt.cwt(sig, SCALES, "cmor1.5-1.0")[0]) ** 2
        np.testing.assert_allclose(power, expected, rtol=1e-6, atol=1e-6 * expected.max())


class _FakeDevice:
    def __init__(self, total_memory):
        self.total_memory = total_memory


def _fake_gpu_torch(total_memory):
    class _Cuda:
        @staticmethod
        def get_device_properties(_i):
            if total_memory is None:
                raise RuntimeError("no device")
            return _FakeDevice(total_memory)

    class _Torch:
        cuda = _Cuda()

    return _Torch()


@pytest.mark.parametrize(
    "gpu_bytes, expected",
    [
        (8 * 2**30, 4),  # a workstation card: four workers
        (4 * 2**30, 4),
        (2 * 2**30, 2),  # under GPU_MEMORY_FOR_FOUR_WORKERS: two
        (None, 4),  # the size could not be read: keep the default
    ],
)
def test_gpu_worker_count_depends_on_card_memory(monkeypatch, gpu_bytes, expected):
    monkeypatch.setattr(periodograms, "PTWT_AVAILABLE", True)
    monkeypatch.setattr(periodograms, "torch", _fake_gpu_torch(gpu_bytes))
    monkeypatch.setattr(periodograms.mp, "cpu_count", lambda: 16)
    assert periodograms.get_optimal_workers(100, use_gpu=True) == expected


def test_cpu_worker_count_leaves_one_core_and_never_exceeds_tasks(monkeypatch):
    monkeypatch.setattr(periodograms.mp, "cpu_count", lambda: 16)
    assert periodograms.get_optimal_workers(100, use_gpu=False) == 15
    assert periodograms.get_optimal_workers(3, use_gpu=False) == 3
    monkeypatch.setattr(periodograms, "PTWT_AVAILABLE", False)
    # Asking for the GPU without one is the CPU rule, not the GPU one.
    assert periodograms.get_optimal_workers(100, use_gpu=True) == 15


@pytest.mark.gpu
@pytest.mark.skipif(not periodograms.PTWT_AVAILABLE, reason="needs CUDA with torch + ptwt")
def test_real_gpu_batches_match_per_fly_transforms(example_ds):
    """On a real GPU, the batched FFTs round differently from per-fly ones; the
    docstring's bound is ~2e-7 of peak power. 1e-5 leaves room without hiding
    a real disagreement."""
    signals = _real_signals(example_ds, n=8, n_time=5000)
    batched = list(periodograms.cwt_powers(signals, SCALES))
    for sig, power in zip(signals, batched):
        single = periodograms.cwt_gpu(sig, SCALES)[2]
        np.testing.assert_allclose(power, single, atol=1e-5 * single.max(), rtol=0)
