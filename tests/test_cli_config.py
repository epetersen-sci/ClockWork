"""The config file and the commands that read it: clockwork run / validate /
init / schema, and the Export settings button that writes one.

Three layers:

- reading a file — extends:, paths relative to the file, empty sections, the
  error messages a person sees;
- checking one — the cross-section rules (a missing upstream step is an error);
- running one — on the committed example_data, end to end, and then reading the
  settings back off what it saved: run → export settings → the same config.
"""

import json
import shutil
from pathlib import Path

import pandas as pd
import pytest
import yaml

from clockwork import cli, pipeline
from clockwork.pipeline.experiment import (
    ConfigError,
    check_config,
    load_config,
    resolved_config,
)
from clockwork.pipeline.export import config_from_dataset, settings_yaml
from clockwork.pipeline.run import RUN_CONFIG_ATTR, OutputsExist, run_experiment
from conftest import EXAMPLE_DIR, requires_example_data

MONITORS = (17, 18)


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


@pytest.fixture(scope="module")
def raw(tmp_path_factory):
    """Monitors 17 and 18 of example_data, with a metadata CSV for just those."""
    if not all((EXAMPLE_DIR / f"Monitor{m}.txt").is_file() for m in MONITORS):
        pytest.skip("example_data monitor files not present")
    d = tmp_path_factory.mktemp("raw")
    for m in MONITORS:
        shutil.copy(EXAMPLE_DIR / f"Monitor{m}.txt", d / f"Monitor{m}.txt")
    meta = pd.read_excel(EXAMPLE_DIR / "metadata.xlsx")
    meta = meta[meta["Monitor"].isin(MONITORS)]
    meta.to_csv(d / "metadata.csv", index=False)
    # The same recording with no LD/DD boundary, for the phase checks.
    meta.drop(columns=["first_DD_day"]).to_csv(d / "metadata_no_dd.csv", index=False)
    return d


def _minimal(raw, **extra) -> str:
    body = {
        "clockwork_config": 1,
        "inputs": {"metadata": str(raw / "metadata.csv"), "monitors": str(raw)},
        **extra,
    }
    return yaml.safe_dump(body, sort_keys=False)


# ---------------------------------------------------------------------------
# Reading a file
# ---------------------------------------------------------------------------


class TestReading:
    def test_paths_are_relative_to_the_file_not_the_terminal(self, tmp_path):
        cfg = _write(
            tmp_path / "configs" / "a.yaml",
            "clockwork_config: 1\ninputs: {metadata: ../data/m.csv, monitors: ../data}\n",
        )
        loaded = load_config(cfg)
        assert loaded.config.inputs.metadata == (tmp_path / "data" / "m.csv").resolve()
        # The default output folder too.
        assert loaded.output_dir == (tmp_path / "configs" / "results" / "a").resolve()

    def test_extends_starts_from_the_base_and_this_file_wins(self, tmp_path):
        _write(
            tmp_path / "lab" / "defaults.yaml",
            "curation: {min_alive_days: 3}\n"
            "analyses:\n  sleep: {threshold_seconds: 420, short_max_minutes: 20}\n"
            "outputs: {dir: lab_results}\n",
        )
        cfg = _write(
            tmp_path / "exp" / "e1.yaml",
            "clockwork_config: 1\nextends: ../lab/defaults.yaml\n"
            "inputs: {metadata: m.csv, monitors: .}\n"
            "analyses:\n  sleep: {threshold_seconds: 360}\n",
        )
        loaded = load_config(cfg)
        c = loaded.config
        assert c.curation.min_alive_days == 3
        assert c.analyses.sleep.threshold_seconds == 360  # this file
        assert c.analyses.sleep.short_max_minutes == 20  # the base, merged key by key
        # A path in the base is relative to the BASE.
        assert loaded.output_dir == (tmp_path / "lab" / "lab_results" / "e1").resolve()
        assert loaded.chain == [cfg.resolve(), (tmp_path / "lab" / "defaults.yaml").resolve()]

    def test_a_subset_and_an_input_source_are_replaced_not_merged(self, tmp_path):
        _write(
            tmp_path / "base.yaml",
            "inputs: {metadata: m.csv, monitors: .}\ngroups: {keep: {genotype: [a]}}\n",
        )
        cfg = _write(
            tmp_path / "e.yaml",
            "clockwork_config: 1\nextends: base.yaml\ninputs: {dataset: saved.nc}\n"
            "groups: {keep: {temperature: [25C]}}\n",
        )
        c = load_config(cfg).config
        assert c.inputs.metadata is None and c.inputs.dataset is not None
        assert c.groups.keep == {"temperature": ["25C"]}

    def test_extends_in_a_circle_is_an_error(self, tmp_path):
        _write(tmp_path / "a.yaml", "clockwork_config: 1\nextends: b.yaml\n")
        _write(tmp_path / "b.yaml", "extends: a.yaml\n")
        with pytest.raises(ConfigError, match="circle"):
            load_config(tmp_path / "a.yaml")

    def test_a_named_empty_section_runs_with_its_defaults(self, tmp_path):
        cfg = _write(
            tmp_path / "e.yaml",
            "clockwork_config: 1\ninputs: {metadata: m.csv, monitors: .}\n"
            "curation:\n  # min_alive_days: 2\nanalyses:\n  sleep:\n"
            "  period:\n    methods:\n      lomb_scargle:\n",
        )
        c = load_config(cfg).config
        assert c.curation == pipeline.CurationConfig()
        assert c.analyses.sleep == pipeline.SleepConfig()
        assert c.analyses.period.methods.lomb_scargle is not None
        assert c.split is None  # not named: does not run

    @pytest.mark.parametrize(
        "text, message",
        [
            ("inputs: {metadata: m.csv, monitors: .}\n", "clockwork_config: missing"),
            ("clockwork_config: 2\ninputs: {metadata: m.csv, monitors: .}\n", "Upgrade ClockWork"),
            (
                "clockwork_config: 1\ninputs: {metadata: m.csv, monitors: .}\ncuration: {min_alive_day: 2}\n",
                "curation.min_alive_day: not a setting here",
            ),
            (
                "clockwork_config: 1\ninputs: {metadata: m.csv, monitors: .}\nanalyses: {phase_shift: {}}\n",
                "Phase shift can only be run from the GUI",
            ),
            (
                "clockwork_config: 1\ninputs: {metadata: m.csv, monitors: .}\n"
                "analyses: {hmm: {preset: bayesian}}\n",
                "analyses.hmm.preset",
            ),
            ("clockwork_config: 1\ninputs: [m.csv\n", "not valid YAML"),
        ],
    )
    def test_errors_say_where_and_what(self, tmp_path, text, message):
        cfg = _write(tmp_path / "e.yaml", text)
        with pytest.raises(ConfigError) as err:
            load_config(cfg)
        assert any(message in p for p in err.value.problems), err.value.problems

    def test_the_provenance_a_run_writes_is_ignored_on_reading(self, tmp_path):
        cfg = _write(
            tmp_path / "e.yaml",
            "clockwork_config: 1\ninputs: {metadata: m.csv, monitors: .}\n"
            "provenance: {python: '3.11'}\n",
        )
        load_config(cfg)


# ---------------------------------------------------------------------------
# Checking one
# ---------------------------------------------------------------------------


def _check(tmp_path, text):
    return check_config(load_config(_write(tmp_path / "e.yaml", text)))


@requires_example_data
class TestChecks:
    def test_a_complete_file_passes(self, raw, tmp_path):
        r = _check(
            tmp_path,
            _minimal(
                raw,
                curation={},
                split={},
                analyses={"period": {"methods": {"lomb_scargle": {}}}, "sleep": {}, "hmm": {}},
            ),
        )
        assert r.ok, r.errors
        assert r.warnings == []

    def test_hmm_without_sleep_names_the_missing_section(self, raw, tmp_path):
        r = _check(tmp_path, _minimal(raw, curation={}, analyses={"hmm": {}}))
        assert any("analyses.hmm needs analyses.sleep" in e for e in r.errors)

    def test_sleep_without_curation_is_an_error(self, raw, tmp_path):
        r = _check(tmp_path, _minimal(raw, analyses={"sleep": {}}))
        assert any("analyses.sleep needs curation" in e for e in r.errors)

    def test_period_without_curation_is_a_warning(self, raw, tmp_path):
        r = _check(tmp_path, _minimal(raw, analyses={"period": {"methods": {"lomb_scargle": {}}}}))
        assert r.ok
        assert any("uncurated" in w for w in r.warnings)

    def test_mesa_without_autocorrelation_warns_it_goes_unclassified(self, raw, tmp_path):
        r = _check(tmp_path, _minimal(raw, curation={}, analyses={"period": {"methods": {"mesa": {}}}}))
        assert r.ok
        assert any("unclassified" in w for w in r.warnings)

    def test_a_group_column_the_metadata_lacks(self, raw, tmp_path):
        r = _check(tmp_path, _minimal(raw, groups={"by": ["genotyp"]}))
        assert any("'genotyp' is not a metadata column" in e for e in r.errors)
        assert any("genotype" in e for e in r.errors)  # and it lists the real ones

    def test_a_subset_column_the_metadata_lacks(self, raw, tmp_path):
        r = _check(tmp_path, _minimal(raw, groups={"keep": {"sex": ["F"]}}))
        assert any("groups.keep: 'sex'" in e for e in r.errors)

    def test_period_across_the_ld_dd_boundary_is_refused(self, raw, tmp_path):
        r = _check(
            tmp_path,
            _minimal(raw, curation={}, analyses={"period": {"phase": "full", "methods": {"lomb_scargle": {}}}}),
        )
        assert any("choose phase: DD or LD" in e for e in r.errors)

    def test_dd_without_a_boundary_is_refused(self, raw, tmp_path):
        text = _minimal(raw, curation={}, split={}, analyses={"period": {"methods": {"lomb_scargle": {}}}})
        text = text.replace("metadata.csv", "metadata_no_dd.csv")
        r = _check(tmp_path, text)
        assert any("split: the metadata has no first_DD_day" in e for e in r.errors)
        assert any("phase DD needs an LD/DD boundary" in e for e in r.errors)

    def test_rhythmic_only_scalograms_need_autocorrelation_s_call(self, raw, tmp_path):
        cwt = {"group_scalograms": True}
        r = _check(tmp_path, _minimal(raw, curation={}, analyses={"period": {"methods": {"cwt": cwt}}}))
        assert any("scalogram_flies: rhythmic" in e for e in r.errors)
        # Averaging every fly needs no call; nor does a run that makes one.
        for methods in (
            {"cwt": {**cwt, "scalogram_flies": "all"}},
            {"cwt": cwt, "autocorrelation": {}},
        ):
            r = _check(tmp_path, _minimal(raw, curation={}, analyses={"period": {"methods": methods}}))
            assert r.ok, r.errors

    def test_a_table_from_an_analysis_that_does_not_run(self, raw, tmp_path):
        r = _check(tmp_path, _minimal(raw, outputs={"tables": ["hmm_states"]}))
        assert any("hmm_states needs analyses.hmm" in e for e in r.errors)

    def test_missing_inputs(self, tmp_path):
        r = _check(tmp_path, "clockwork_config: 1\ninputs: {metadata: nope.csv, monitors: nowhere}\n")
        assert any("no such file" in e for e in r.errors)
        assert any("no such folder" in e for e in r.errors)


# ---------------------------------------------------------------------------
# The resolved config
# ---------------------------------------------------------------------------


class TestResolved:
    def test_it_spells_out_what_the_defaults_mean(self, tmp_path):
        cfg = _write(
            tmp_path / "e.yaml",
            "clockwork_config: 1\ninputs: {metadata: m.csv, monitors: .}\n"
            "analyses:\n  hmm: {preset: wiggin}\n  period: {methods: {cwt: {}, lomb_scargle: {}}}\n",
        )
        r = resolved_config(load_config(cfg).config)
        assert r["analyses"]["hmm"]["emission_model"] == "binary"  # Wiggin's
        cwt = r["analyses"]["period"]["methods"]["cwt"]
        assert cwt["rhythmic_threshold"] > 0
        assert cwt["phase"] == "DD"
        assert "max_bridge_gap_minutes" not in r["analyses"]["period"]["methods"]["lomb_scargle"]

    def test_it_reads_back_as_the_same_config(self, tmp_path):
        cfg = _write(
            tmp_path / "e.yaml",
            "clockwork_config: 1\ninputs: {metadata: m.csv, monitors: .}\ncuration: {}\n"
            "analyses:\n  hmm: {preset: harbison, n_states: 3}\n  sleep: {}\n"
            "  period: {period_range_hours: [18, 30], methods: {autocorrelation: {}, mesa: {order: fpe}}}\n",
        )
        first = resolved_config(load_config(cfg).config)
        again = _write(tmp_path / "r.yaml", yaml.safe_dump(first))
        assert resolved_config(load_config(again).config) == first


# ---------------------------------------------------------------------------
# Running one
# ---------------------------------------------------------------------------

QUICK_HMM = {"preset": "wiggin", "phase": "LD", "n_states": 3, "n_restarts": 2, "n_iter": 15}


@pytest.fixture(scope="module")
def ran(raw, tmp_path_factory):
    """One full run on monitors 17 + 18: every step and every output."""
    d = tmp_path_factory.mktemp("run")
    cfg = _write(
        d / "exp1.yaml",
        _minimal(
            raw,
            experiment="Exp One",
            groups={"by": ["genotype"]},
            curation={"min_alive_days": 3},
            split={"discard_first_dd_day": True},
            analyses={
                "period": {
                    "period_range_hours": [18, 30],
                    "methods": {
                        "autocorrelation": {},
                        "lomb_scargle": {"rhythmic_threshold": 0.2},
                        "cwt": {"voices_per_octave": 8, "group_scalograms": True},
                    },
                },
                "sleep": {"threshold_seconds": 360},
                "hmm": QUICK_HMM,
            },
        ),
    )
    loaded = load_config(cfg)
    lines = []
    out = run_experiment(loaded, log=lines.append)
    return loaded, out, lines


@requires_example_data
class TestRun:
    def test_every_output_is_written(self, ran):
        loaded, out, _ = ran
        assert out.out_dir == loaded.output_dir
        assert out.out_dir.name == "Exp_One"
        assert out.dataset == out.out_dir / "Exp_One.nc"
        for p in (out.dataset, out.resolved, out.report):
            assert p.is_file(), p
        assert set(out.tables) == {
            "period_summary",
            "sleep_summary",
            "sleep_states",
            "sleep_bouts",
            "hmm_occupancy",
            "hmm_states",
            "hmm_zt_fractions",
        }

    def test_the_tables_hold_every_fly(self, ran):
        _, out, _ = ran
        ds = pipeline.load_netcdf(out.dataset)
        period = pd.read_csv(out.tables["period_summary"])
        assert len(period) == ds.sizes["id"]
        assert {"AC_Rhythmic", "LS_Rhythmic", "CWT_Rhythmic"} <= set(period.columns)
        sleep = pd.read_csv(out.tables["sleep_summary"])
        assert set(sleep["Phase"]) == {"LD", "DD"}
        assert len(sleep) == 2 * ds.sizes["id"]

    def test_the_cwt_group_scalograms_are_written(self, ran):
        _, out, _ = ran
        ds = pipeline.load_netcdf(out.dataset)
        groups = sorted({str(g) for g in ds["group"].values})
        assert sorted(e["group"] for e in out.scalograms) == groups
        for entry in out.scalograms:
            assert Path(entry["png"]).parent == out.out_dir / "scalograms"
            assert Path(entry["png"]).is_file() and Path(entry["csv"]).is_file()
            assert entry["phase"] == "DD"
            # Only autocorrelation's rhythmic flies are averaged (the default).
            members = ds["ac_rhythmic"].values[ds["group"].values == entry["group"]]
            assert entry["n"] <= int(members.sum())
        # The dataset records where they went, and that they were asked for.
        assert json.loads(ds.attrs["cwt_group_average_paths"]) == out.scalograms
        assert int(ds.attrs["cwt_group_scalograms"]) == 1

    def test_the_qc_report_has_a_section_per_step(self, ran):
        _, out, _ = ran
        text = out.report.read_text(encoding="utf-8")
        for title in (
            "Curation",
            "LD/DD split",
            "Period &amp; rhythmicity",
            "CWT group-averaged scalograms",
            "Sleep",
            "HMM sleep states",
        ):
            assert f"<h2>{title}</h2>" in text, title
        assert "could not be drawn" not in text

    def test_the_resolved_config_is_beside_the_outputs_and_on_the_dataset(self, ran):
        loaded, out, _ = ran
        resolved = yaml.safe_load(out.resolved.read_text(encoding="utf-8"))
        assert resolved["provenance"]["packages"]["clockwork-sci"]
        assert resolved["groups"]["by"] == ["genotype"]
        # It runs as it stands: same settings as the file that made it.
        again = load_config(out.resolved)
        assert resolved_config(again.config) == resolved_config(loaded.config, ["genotype"])
        ds = pipeline.load_netcdf(out.dataset)
        on_ds = json.loads(ds.attrs[RUN_CONFIG_ATTR])
        assert on_ds["analyses"] == resolved["analyses"]

    def test_outputs_are_not_overwritten_without_force(self, ran):
        loaded, _, _ = ran
        with pytest.raises(OutputsExist):
            run_experiment(loaded)

    def test_export_settings_gives_back_the_config_that_ran(self, ran):
        """The round trip the Export settings button rests on (docs)."""
        loaded, out, _ = ran
        ds = pipeline.load_netcdf(out.dataset)
        exported, notes = config_from_dataset(ds)
        assert notes == []
        original = loaded.config
        assert exported.analyses == original.analyses
        assert exported.curation == original.curation
        assert exported.split == original.split
        assert exported.groups == original.groups
        assert exported.inputs == original.inputs

    def test_and_through_the_yaml_file(self, ran, tmp_path):
        loaded, out, _ = ran
        ds = pipeline.load_netcdf(out.dataset)
        settings = _write(tmp_path / "settings.yaml", settings_yaml(ds, relative_to=tmp_path))
        back = load_config(settings)
        assert back.config.analyses == loaded.config.analyses
        assert back.config.inputs == loaded.config.inputs
        assert check_config(back).ok


@requires_example_data
class TestRunFromASavedDataset:
    def test_a_later_analysis_on_the_saved_dataset(self, ran, tmp_path):
        """inputs.dataset: the curation and split already on it count as done."""
        _, out, _ = ran
        cfg = _write(
            tmp_path / "again.yaml",
            "clockwork_config: 1\n"
            f"inputs: {{dataset: {out.dataset.as_posix()}}}\n"
            "analyses:\n  period: {methods: {lomb_scargle: {}}, period_range_hours: [20, 28]}\n"
            "outputs: {tables: [period_summary], qc_report: false}\n",
        )
        loaded = load_config(cfg)
        assert check_config(loaded).ok, check_config(loaded).errors
        res = run_experiment(loaded)
        ds = pipeline.load_netcdf(res.dataset)
        assert float(ds.attrs["ls_min_period"]) == 20.0
        assert "hmm_state" in ds.data_vars  # what was on it stays

    def test_curating_it_again_is_refused(self, ran, tmp_path):
        _, out, _ = ran
        r = _check(
            tmp_path,
            f"clockwork_config: 1\ninputs: {{dataset: {out.dataset.as_posix()}}}\ncuration: {{}}\nsplit: {{}}\n",
        )
        assert any("already curated" in e for e in r.errors)
        assert any("already split" in e for e in r.errors)


    def test_tables_can_come_from_results_already_on_it(self, ran, tmp_path):
        """No analyses section: the period and HMM tables come from the results
        the saved dataset already holds, and validate must allow asking for them."""
        _, out, _ = ran
        r = _check(
            tmp_path,
            "clockwork_config: 1\n"
            f"inputs: {{dataset: {out.dataset.as_posix()}}}\n"
            "outputs: {tables: [period_summary, hmm_states]}\n",
        )
        assert r.ok, r.errors


@requires_example_data
class TestForce:
    def test_force_replaces_the_last_run_and_nothing_else(self, raw, tmp_path):
        cfg = _write(
            tmp_path / "f.yaml",
            _minimal(raw, curation={}, analyses={"period": {"methods": {"lomb_scargle": {}}}}),
        )
        loaded = load_config(cfg)
        first = run_experiment(loaded)
        # What an earlier, different run would have left behind, and a file of
        # the user's own.
        (first.out_dir / "tables" / "hmm_states.csv").write_text("stale")
        (first.out_dir / "scalograms").mkdir()
        (first.out_dir / "scalograms" / "averaged_scalogram_old.png").write_bytes(b"stale")
        (first.out_dir / "my_notes.txt").write_text("keep me")

        second = run_experiment(loaded, force=True)
        assert set(second.tables) == {"period_summary"}
        assert not (second.out_dir / "tables" / "hmm_states.csv").exists()
        assert not (second.out_dir / "scalograms").exists()
        assert (second.out_dir / "my_notes.txt").read_text() == "keep me"
        assert second.dataset.is_file() and second.report.is_file()

# ---------------------------------------------------------------------------
# The commands
# ---------------------------------------------------------------------------


@requires_example_data
class TestCommands:
    def test_init_writes_a_file_that_validates(self, raw, tmp_path, capsys):
        path = tmp_path / "new.yaml"
        assert cli.main(["init", str(path), "--metadata", str(raw / "metadata.csv")]) == 0
        text = path.read_text(encoding="utf-8")
        assert "by: [genotype, temperature]" in text
        assert "split:" in text and "phase: DD" in text
        assert cli.main(["validate", str(path)]) == 0
        assert "OK" in capsys.readouterr().out

    def test_init_without_a_boundary_offers_no_split(self, raw, tmp_path):
        path = tmp_path / "new.yaml"
        assert cli.main(["init", str(path), "--metadata", str(raw / "metadata_no_dd.csv")]) == 0
        text = path.read_text(encoding="utf-8")
        assert "\nsplit:" not in text and "phase: full" in text
        assert cli.main(["validate", str(path)]) == 0

    def test_init_does_not_overwrite(self, tmp_path):
        path = _write(tmp_path / "mine.yaml", "keep me")
        assert cli.main(["init", str(path)]) == 1
        assert path.read_text() == "keep me"

    def test_validate_fails_on_a_bad_file(self, raw, tmp_path, capsys):
        bad = _write(tmp_path / "bad.yaml", _minimal(raw, analyses={"hmm": {}}))
        assert cli.main(["validate", str(bad)]) == 1
        assert "analyses.hmm needs analyses.sleep" in capsys.readouterr().err

    def test_run_does_not_start_a_bad_file(self, raw, tmp_path, capsys):
        bad = _write(tmp_path / "bad.yaml", _minimal(raw, analyses={"sleep": {}}))
        assert cli.main(["run", str(bad)]) == 1
        assert not (tmp_path / "results").exists()

    def test_run_will_not_overwrite(self, ran, capsys):
        loaded, _, _ = ran
        assert cli.main(["run", str(loaded.path)]) == 1
        assert "--force" in capsys.readouterr().err

    def test_schema_is_json_schema(self, capsys):
        assert cli.main(["schema"]) == 0
        schema = json.loads(capsys.readouterr().out)
        assert "inputs" in schema["properties"]
        assert "extends" in schema["properties"]


# ---------------------------------------------------------------------------
# The Export settings button
# ---------------------------------------------------------------------------


@requires_example_data
class TestExportSettingsTab:
    def test_the_tab_shows_and_saves_the_settings(self, app, ran, tmp_path):
        _, out, _ = ran
        ds = pipeline.load_netcdf(out.dataset)
        at = app(ds=ds, page="export_data", export_tab="Settings (.yaml)", working_dir=str(tmp_path))
        assert not at.exception, at.exception
        shown = at.code[0].value
        assert "threshold_seconds: 360" in shown
        at.button(key="save_settings_yaml").click()
        at.session_state["export_tab"] = "Settings (.yaml)"
        at.run()
        assert not at.exception, at.exception
        saved = list(Path(ds.attrs["source_data_dir"]).glob("clockwork_settings*.yaml")) or list(
            tmp_path.glob("clockwork_settings*.yaml")
        )
        assert saved, "the settings file was not written"
        back = load_config(saved[0])
        assert back.config.analyses == config_from_dataset(ds)[0].analyses
