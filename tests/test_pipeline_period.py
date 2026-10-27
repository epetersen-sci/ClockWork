"""The period step: four estimators, their rhythmic calls, and the round trip.

EQUIVALENCE: each method run through the pipeline gives bit-identical output to
the core call the Analysis tab made before the pipeline existed (same view, same
preprocessing, same arguments). ROUND TRIP: PeriodConfig reads back off the
dataset as it went in, through a .nc too — including per-method overrides and
the per-method preprocessing, which before this was not recorded on the master
at all.

Everything runs on eight of the example flies: these hold the wiring, not what
the estimators say about a cohort (test_period_estimators_snapshot does that).
"""

import numpy as np
import pytest
import xarray as xr
from pydantic import ValidationError

from clockwork import pipeline
from clockwork.core import dam_utilities, periodograms
from clockwork.core.load_and_save_datasets import save_dataset_to_netcdf
from clockwork.core.preprocessing import (
    PreprocessConfig,
    ac_default_config,
    cwt_default_config,
    ls_default_config,
    preprocess_activity,
)
from clockwork.core.rhythmicity_classification import classify_all
from clockwork.pipeline import PeriodConfig
from conftest import requires_example_data

pytestmark = [requires_example_data, pytest.mark.usefixtures("shared_worker_pool")]

RANGE = (16.0, 36.0)
FLOOR = 4.0
CWT_FLIES = ["20250115_17_1", "20250115_17_2", "20250115_18_1", "20250115_18_2"]
#: Both monitors, both genotypes, and not in sorted order ("_10" < "_2").
FLIES = CWT_FLIES[:2] + ["20250115_17_10", "20250115_17_11"] + CWT_FLIES[2:] + [
    "20250115_18_10", "20250115_18_11",
]


def _same(a: xr.Dataset, b: xr.Dataset, prefix):
    """Every ``prefix`` output equal FLY BY FLY. The direct core call returns its
    flies sorted (its own merge's outer join); the pipeline keeps import order."""
    names = [v for v in a.data_vars if v.startswith(prefix)]
    assert names, f"no {prefix} outputs"
    b = b.sel(id=a["id"].values)
    for v in names:
        x, y = a[v].values, b[v].transpose(*a[v].dims).values
        assert x.shape == y.shape, v
        if x.dtype.kind in "fc":
            np.testing.assert_array_equal(x, y, err_msg=v)
        else:
            assert (x == y).all(), v


@pytest.fixture(scope="module")
def flies(example_ds):
    return example_ds.sel(id=FLIES)


@pytest.fixture(scope="module")
def dd_view(flies):
    return dam_utilities.select_phase(flies, "DD")[0]


class TestEquivalence:
    """The pipeline computes what the Analysis tab computed, call for call."""

    def test_lomb_scargle(self, flies, dd_view):
        cfg = PeriodConfig(methods={"lomb_scargle": {}})
        via, _ = pipeline.run_period_method(flies, cfg, "lomb_scargle")
        direct = periodograms.lomb_scargle_analysis(
            preprocess_activity(dd_view, ls_default_config()),
            min_period=RANGE[0], max_period=RANGE[1], phase="DD", min_num_days=FLOOR,
            oversampling=8, fap_method="baluev",
        )
        _same(via, direct, "ls_")

    def test_autocorrelation_with_its_preprocessing(self, flies, dd_view):
        cfg = PeriodConfig(methods={"autocorrelation": {"preprocessing": {"lowpass_hours": 3}}})
        via, _ = pipeline.run_period_method(flies, cfg, "autocorrelation")
        direct = periodograms.autocorrelation_analysis(
            preprocess_activity(dd_view, PreprocessConfig(lopass_hours=3.0, detrend="linear")),
            min_period=RANGE[0], max_period=RANGE[1], phase="DD", min_num_days=FLOOR,
            ac_peak=2, max_bridge_gap_minutes=60.0,
        )
        _same(via, direct, "ac_")

    def test_mesa(self, flies, dd_view):
        cfg = PeriodConfig(methods={"autocorrelation": {}, "mesa": {"order": "fpe"}})
        via, _ = pipeline.run_period_method(flies, cfg, "mesa")
        direct = periodograms.mesa_analysis(
            preprocess_activity(dd_view, ac_default_config()),
            min_period=RANGE[0], max_period=RANGE[1], phase="DD", min_num_days=FLOOR,
            bin_minutes=30, order="fpe", max_bridge_gap_minutes=60.0,
        )
        _same(via, direct, "mesa_")

    def test_cwt(self, flies):
        few = flies.sel(id=CWT_FLIES)
        cfg = PeriodConfig(methods={"cwt": {}})
        via, _ = pipeline.run_period_method(few, cfg, "cwt")
        direct, _ = periodograms.wavelet_analysis(
            preprocess_activity(dam_utilities.select_phase(few, "DD")[0], cwt_default_config()),
            min_period=RANGE[0], max_period=RANGE[1], phase="DD", min_num_days=FLOOR,
            cwt_method="global_rednoise", resolution=1 / 32, max_bridge_gap_minutes=60.0,
            compute_group_averages=False, group_coord="group",
            filter_nonrhythmic_for_average=True, phase_label=None,
        )
        _same(via, direct, "cwt_")

    def test_the_master_keeps_its_raw_activity(self, flies):
        """The estimators run on detrended activity; it must not come back."""
        cfg = PeriodConfig(methods={"autocorrelation": {}})
        via, _ = pipeline.run_period_method(flies, cfg, "autocorrelation")
        xr.testing.assert_identical(via["activity"], flies["activity"])


class TestTheMergeLeavesTheMasterAsItWas:
    """BACKLOG 23: a period run used to sort the master's flies (an outer join of
    two differently ordered id indexes sorts the union) and hand an unsplit
    master the split_minute coord its phase view carried."""

    @pytest.fixture(scope="class")
    def classified(self, flies):
        cfg = PeriodConfig(methods={"lomb_scargle": {}, "autocorrelation": {}})
        return pipeline.run_period(flies, cfg)

    def test_the_flies_stay_in_import_order(self, flies, classified):
        assert list(classified["id"].values) == list(flies["id"].values)
        assert list(flies["id"].values) != sorted(flies["id"].values), (
            "the example must not already be sorted, or this proves nothing"
        )

    def test_no_coord_arrives_that_is_not_the_method_s_own(self, flies, classified):
        added = set(classified.coords) - set(flies.coords)
        assert "split_minute" not in added
        assert added <= {"ls_rhythmic", "ac_rhythmic"} | {
            c for c in added if str(c).startswith(("ls_", "ac_"))
        }

    def test_the_rhythmic_flags_still_arrive_under_the_right_flies(self, flies, classified):
        assert {"ls_rhythmic", "ac_rhythmic"} <= set(classified.coords)
        direct = classify_all(classified, period_window=RANGE, run_cwt=False)
        for flag in ("ls_rhythmic", "ac_rhythmic"):
            np.testing.assert_array_equal(classified[flag].values, direct[flag].values)

    def test_everything_already_on_the_master_is_untouched(self, flies, classified):
        xr.testing.assert_identical(
            classified[list(flies.data_vars)].drop_vars(
                [c for c in classified.coords if c not in flies.coords], errors="ignore"
            ).drop_attrs(),
            flies.drop_attrs(),
        )

    def test_classification_is_classify_all(self, flies):
        cfg = PeriodConfig(methods={"autocorrelation": {}})
        ran, _ = pipeline.run_period_method(flies, cfg, "autocorrelation")
        via = pipeline.classify_period(ran, "autocorrelation", 0.35, RANGE)
        direct = classify_all(ran, ac_ri_threshold=0.35, period_window=RANGE,
                              run_ls=False, run_ac=True, run_cwt=False)
        np.testing.assert_array_equal(via["ac_rhythmic"].values, direct["ac_rhythmic"].values)


@pytest.fixture(scope="module")
def full_run(flies):
    """Every method, with something non-default in each, as a CLI run takes it."""
    config = PeriodConfig(
        period_range_hours=(18.0, 32.0),
        min_dd_days=3.0,
        methods={
            "lomb_scargle": {"oversampling": 6, "classify": False},
            "autocorrelation": {"preprocessing": {"lowpass_hours": 3}, "rhythmic_threshold": 0.35},
            "mesa": {"order": "fpe", "bin_minutes": 20},
        },
    )
    return pipeline.run_period(flies, config), config


class TestRoundTrip:
    def test_the_config_reads_back_as_it_went_in(self, full_run):
        ds, config = full_run
        assert PeriodConfig.from_attrs(ds.attrs, ds.coords) == config

    def test_and_through_a_netcdf(self, full_run, tmp_path):
        ds, config = full_run
        path = tmp_path / "period.nc"
        save_dataset_to_netcdf(ds, str(path))
        back = pipeline.load_netcdf(path)
        assert PeriodConfig.from_attrs(back.attrs, back.coords) == config

    def test_preprocessing_is_now_on_the_master(self, full_run):
        ds, _ = full_run
        assert ds.attrs["ac_prep_lopass_hours"] == 3.0
        assert ds.attrs["ac_prep_detrend"] == "linear"
        assert ds.attrs["ls_prep_lopass_hours"] == 0.0

    def test_unclassified_and_classified_are_told_apart(self, full_run):
        ds, _ = full_run
        back = PeriodConfig.from_attrs(ds.attrs, ds.coords)
        assert back.methods.lomb_scargle.classify is False
        assert back.methods.autocorrelation.rhythmic_threshold == 0.35

    def test_overrides_are_short(self, full_run):
        _, config = full_run
        assert config.overrides() == {
            "period_range_hours": [18.0, 32.0],
            "min_dd_days": 3.0,
            "methods": {
                "autocorrelation": {"preprocessing": {"lowpass_hours": 3.0}, "rhythmic_threshold": 0.35},
                "lomb_scargle": {"oversampling": 6, "classify": False},
                "mesa": {"bin_minutes": 20, "order": "fpe"},
            },
        }

    def test_a_method_rerun_with_another_range_keeps_its_own(self, flies):
        """The GUI lets one method be re-run on a different range. The exported
        config must say which range produced which result."""
        ls_cfg = PeriodConfig(methods={"lomb_scargle": {}})
        ac_cfg = PeriodConfig(period_range_hours=(20.0, 28.0), methods={"autocorrelation": {}})
        ds, _ = pipeline.run_period_method(flies, ls_cfg, "lomb_scargle")
        ds, _ = pipeline.run_period_method(ds, ac_cfg, "autocorrelation")
        back = PeriodConfig.from_attrs(ds.attrs, ds.coords)
        assert back.effective("lomb_scargle")["period_range_hours"] == (16.0, 36.0)
        assert back.effective("autocorrelation")["period_range_hours"] == (20.0, 28.0)

    def test_a_dataset_with_no_period_analysis_reads_back_as_none(self, flies):
        assert PeriodConfig.from_attrs(flies.attrs, flies.coords) is None

    def test_a_dataset_from_before_preprocessing_was_recorded(self, full_run):
        """Older .nc files have no <method>_prep_* attrs: read as the method default."""
        ds, _ = full_run
        attrs = {k: v for k, v in ds.attrs.items() if "_prep_" not in k}
        assert PeriodConfig.from_attrs(attrs, ds.coords).methods.autocorrelation.preprocessing is None


class TestPhases:
    def test_a_full_recording_is_phase_full(self, flies):
        ds = flies.drop_vars("first_DD_day")
        cfg = PeriodConfig(phase="full", methods={"lomb_scargle": {"classify": False}})
        out = pipeline.run_period(ds, cfg)
        assert out.attrs["ls_phase"] == "both"
        assert PeriodConfig.from_attrs(out.attrs, out.coords).phase == "full"

    def test_dd_on_a_recording_without_dd_says_why(self, flies):
        ds = flies.drop_vars("first_DD_day")
        with pytest.raises(ValueError, match="phase: full"):
            pipeline.phase_source(ds, "DD")

    def test_full_on_a_recording_with_dd_says_why(self, flies):
        with pytest.raises(ValueError, match="DD or LD"):
            pipeline.phase_source(flies, "full")


class TestValidation:
    def test_mesa_runs_without_autocorrelation(self):
        """As on the Analysis tab: only MESA's rhythmic call needs autocorrelation."""
        assert PeriodConfig(methods={"mesa": {}}).methods.autocorrelation is None

    def test_lomb_scargle_takes_no_gap_bridging(self):
        with pytest.raises(ValidationError, match="gap-native"):
            PeriodConfig(methods={"lomb_scargle": {"max_bridge_gap_minutes": 30}})

    def test_the_range_must_be_a_range(self):
        with pytest.raises(ValidationError):
            PeriodConfig(period_range_hours=(30, 20), methods={"lomb_scargle": {}})

    def test_no_methods_is_an_error(self):
        with pytest.raises(ValidationError, match="at least one"):
            PeriodConfig(methods={})

    def test_an_unknown_method_or_option_is_an_error(self):
        with pytest.raises(ValidationError):
            PeriodConfig(methods={"fourier": {}})
        with pytest.raises(ValidationError):
            PeriodConfig(methods={"cwt": {"voices_per_octave": 33}})
