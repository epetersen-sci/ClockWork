"""The HMM step: fit, decode, record — and the HMM tab, whose Run button no test
pressed before this.

Few restarts and few iterations throughout: these hold the WIRING (equivalence
with the old call, the round trip, the merge), not the quality of a fit.
"""

import numpy as np
import pytest
import xarray as xr
from pydantic import ValidationError

from clockwork import pipeline
from clockwork.core import dam_utilities
from clockwork.core.hmm_models import HMMConfig, run_genotype_workflow
from clockwork.core.load_and_save_datasets import save_dataset_to_netcdf
from clockwork.pipeline import CurationConfig, HmmConfig
from conftest import requires_example_data

pytestmark = requires_example_data

FLIES = ["20250115_17_1", "20250115_17_2", "20250115_18_1", "20250115_18_2"]
QUICK = {"n_restarts": 2, "n_iter": 15}


@pytest.fixture(scope="module")
def curated(example_ds):
    return pipeline.curate(example_ds.sel(id=FLIES), CurationConfig()).live


@pytest.fixture(scope="module")
def fitted(curated):
    config = HmmConfig(phase="DD", preset="wiggin", n_states=3, **QUICK)
    return pipeline.run_hmm(curated, config), config


def _old_tab(master, core):
    """What the HMM tab did before the pipeline: fit on the phase view, then
    merge only the hmm_ vars and attrs onto the master."""
    src, phase_used = dam_utilities.select_phase(master, "DD")
    res, _ = run_genotype_workflow(src, core, verbose=False)
    hmm_vars = [v for v in ("hmm_state", "hmm_sleep", "hmm_confidence") if v in res.data_vars]
    out = master.drop_vars([v for v in hmm_vars if v in master.data_vars], errors="ignore")
    out = out.merge(res[hmm_vars], compat="no_conflicts", join="outer")
    for k, v in res.attrs.items():
        if str(k).startswith("hmm_"):
            out.attrs[k] = v
    out.attrs["hmm_phase"] = phase_used
    return out


class TestEquivalence:
    def test_the_fit_is_the_tab_s_old_computation(self, curated, fitted):
        run, config = fitted
        old = _old_tab(curated, config.core())
        for v in ("hmm_state", "hmm_sleep"):
            new = run.master[v]
            np.testing.assert_array_equal(
                new.values, old[v].sel(id=new["id"].values).transpose(*new.dims).values, err_msg=v
            )
        assert {k: v for k, v in run.master.attrs.items() if k.startswith("hmm_")} == {
            k: v for k, v in old.attrs.items() if k.startswith("hmm_")
        }

    def test_the_master_is_otherwise_untouched(self, curated, fitted):
        run, _ = fitted
        assert list(run.master["id"].values) == list(curated["id"].values)
        assert set(run.master.coords) == set(curated.coords)
        xr.testing.assert_identical(run.master["activity"], curated["activity"])

    def test_the_preset_plus_overrides_is_what_was_fitted(self, fitted):
        run, _ = fitted
        assert run.config.emission_model == "binary"  # Wiggin's
        assert run.config.n_states == 3  # the override


class TestRoundTrip:
    def test_it_reads_back(self, fitted):
        run, config = fitted
        assert HmmConfig.from_attrs(run.master.attrs) == config

    def test_and_through_a_netcdf(self, fitted, tmp_path):
        run, config = fitted
        path = tmp_path / "hmm.nc"
        save_dataset_to_netcdf(run.master, str(path))
        back = pipeline.load_netcdf(path)
        assert HmmConfig.from_attrs(back.attrs) == config

    def test_exported_against_the_preset_needing_fewest_overrides(self):
        import dataclasses

        core = dataclasses.replace(HMMConfig.harbison(), n_restarts=20)
        assert HmmConfig.from_core(core, phase="DD").overrides() == {
            "phase": "DD",
            "preset": "harbison",
            "n_restarts": 20,
        }

    def test_no_fit_reads_back_as_none(self, curated):
        assert HmmConfig.from_attrs(curated.attrs) is None

    def test_the_whole_recording_is_phase_full(self, curated):
        out = pipeline.run_hmm(curated, HmmConfig(phase="full", preset="wiggin", **QUICK))
        assert out.master.attrs["hmm_phase"] == "both"
        assert HmmConfig.from_attrs(out.master.attrs).phase == "full"


class TestValidation:
    def test_restating_a_preset_value_is_no_override(self):
        assert HmmConfig(preset="wiggin", emission_model="binary").emission_model is None

    def test_unknown_values_are_errors(self):
        with pytest.raises(ValidationError):
            HmmConfig(preset="bayesian")
        with pytest.raises(ValidationError):
            HmmConfig(n_states=1)
        with pytest.raises(ValidationError):
            HmmConfig(emission="zip")


class TestTheHmmTab:
    def test_run_records_what_the_controls_mean(self, app, curated):
        at = app(
            ds=curated,
            page="hmm",
            hmm_tab="Analysis",
            hmm_phase_choice="DD",
            hmm_preset="Wiggin et al. 2020",
            # Marks the preset as already synced, so its widget defaults do not
            # overwrite the restarts and states set here.
            _last_hmm_preset="Wiggin et al. 2020",
            hmm_n_states=3,
            hmm_training_scope="per_genotype",
            hmm_emission="binary",
            hmm_trans="hard",
            hmm_restarts=5,
            hmm_decoding="posterior",
        )
        assert not at.exception, at.exception
        at.button(key="run_hmm").click()
        at.session_state["hmm_tab"] = "Analysis"
        at.run(timeout=600)
        assert not at.exception, at.exception
        ds = at.session_state["dataset"]
        assert "hmm_state" in ds.data_vars
        assert HmmConfig.from_attrs(ds.attrs) == HmmConfig(
            phase="DD", preset="wiggin", n_states=3, n_restarts=5
        )
