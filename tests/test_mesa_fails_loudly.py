"""MESA stops with an error when its Burg fit fails for every fly.

statsmodels before 0.14.6 raised a TypeError on every Burg call beside pandas 3.
MESA skips a fly whose fit raises, which is right for one fly, so that showed up
as a run with no MESA period for anybody and no error. These pin the line
between the two: some flies failing is a per-fly skip; all of them failing
stops the analysis, naming the error.

The analysis runs in-process (a stand-in pool), so the patched fit reaches it.
"""

import numpy as np
import pytest

from clockwork.core import dam_utilities, periodograms
from clockwork.core.periodograms import MesaFitError
from clockwork.core.preprocessing import ac_default_config, preprocess_activity
from conftest import requires_example_data

pytestmark = requires_example_data

FLIES = ["20250115_17_1", "20250115_17_2", "20250115_18_1", "20250115_18_2"]


class _InProcess:
    """Enough of a multiprocessing pool to run the per-fly workers here."""

    def imap_unordered(self, fn, args):
        return map(fn, args)


@pytest.fixture(scope="module")
def mesa_input(example_ds):
    src, _ = dam_utilities.select_phase(example_ds.sel(id=FLIES), "DD")
    return preprocess_activity(src, ac_default_config())


def _mesa(ds):
    return periodograms.mesa_analysis(ds, phase="DD", min_num_days=4, pool=_InProcess())


def test_a_working_fit_gives_every_fly_a_period(mesa_input):
    out = _mesa(mesa_input)
    assert np.isfinite(out["mesa_period"].sel(id=FLIES).values).all()


def test_a_fit_that_fails_for_every_fly_stops_with_the_error(mesa_input, monkeypatch):
    def broken(*args, **kwargs):
        raise TypeError("deprecate_kwarg() missing 1 required positional argument")

    monkeypatch.setattr(periodograms, "_sm_burg", broken)
    with pytest.raises(MesaFitError) as err:
        _mesa(mesa_input)
    message = str(err.value)
    assert "every fly" in message
    assert "TypeError: deprecate_kwarg()" in message  # the real error, not just its type
    assert "statsmodels" in message


def test_a_fit_that_fails_for_some_flies_skips_just_those(mesa_input, monkeypatch):
    real = periodograms._sm_burg
    calls = {"n": 0}

    def flaky(x, order):
        # The first fit fails; the rest work. (One fly's fit failing is that
        # fly's problem, not the analysis's.)
        calls["n"] += 1
        if calls["n"] == 1:
            raise np.linalg.LinAlgError("singular")
        return real(x, order=order)

    monkeypatch.setattr(periodograms, "_sm_burg", flaky)
    out = _mesa(mesa_input)
    periods = out["mesa_period"].sel(id=FLIES).values
    assert np.isnan(periods).sum() == 1
    assert np.isfinite(periods).sum() == len(FLIES) - 1


def test_flies_skipped_for_too_little_data_do_not_count(mesa_input, monkeypatch):
    """No fly reaching the fit is "no data", not a broken fit."""
    monkeypatch.setattr(
        periodograms, "_sm_burg", lambda *a, **k: (_ for _ in ()).throw(TypeError("never called"))
    )
    out = periodograms.mesa_analysis(mesa_input, phase="DD", min_num_days=999, pool=_InProcess())
    assert np.isnan(out["mesa_period"].sel(id=FLIES).values).all()
