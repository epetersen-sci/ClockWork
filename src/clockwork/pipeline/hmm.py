"""The hidden Markov sleep-state model: fit, decode, and record what was fitted.

A config names a published PRESET and overrides any of its settings, the way
the HMM tab does (pick a preset, then adjust under Advanced options). An
override is relative to the preset, so ``preset: wiggin`` plus ``n_states: 3``
is Wiggin's model with three states, not three states and the defaults.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Literal

import numpy as np
import xarray as xr
from pydantic import BaseModel, ConfigDict, Field, model_validator

from clockwork.core import dam_utilities
from clockwork.core.hmm_models import HMMConfig as CoreHMMConfig
from clockwork.core.hmm_models import load_hmm_config_from_attrs, run_genotype_workflow

Preset = Literal["improved", "wiggin", "harbison"]
#: Tie-break order when choosing the preset an exported config is written against.
PRESETS: tuple[Preset, ...] = ("improved", "wiggin", "harbison")

#: The core HMMConfig fields a config can set (n_jobs is the machine's business).
TUNABLE = (
    "n_states",
    "training_scope",
    "emission_model",
    "observation_type",
    "transition_constraints",
    "decoding_method",
    "n_restarts",
    "n_iter",
    "tol",
    "soft_floor",
    "dirichlet_concentration",
)

#: What the core records as ``hmm_phase`` for each config phase.
_RECORDED_PHASE = {"LD": "LD", "DD": "DD", "full": "both"}
_PHASE_FROM_RECORDED = {v: k for k, v in _RECORDED_PHASE.items()}


def preset_config(name: str) -> CoreHMMConfig:
    return {
        "improved": CoreHMMConfig.improved,
        "wiggin": CoreHMMConfig.wiggin,
        "harbison": CoreHMMConfig.harbison,
    }[name]()


class HmmConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    #: Which lighting paradigm to fit: LD, DD, or the whole recording.
    phase: Literal["LD", "DD", "full"] = "LD"
    preset: Preset = "improved"

    # Overrides of the preset. None = the preset's value.
    n_states: int | None = Field(None, ge=2, le=10)
    training_scope: Literal["per_genotype", "all_pooled", "per_fly", "per_fly_per_day"] | None = None
    emission_model: Literal["zip", "gaussian", "binary"] | None = None
    observation_type: Literal["counts", "binary", "normalized"] | None = None
    transition_constraints: Literal["soft", "hard", "none"] | None = None
    decoding_method: Literal["both", "viterbi", "posterior"] | None = None
    n_restarts: int | None = Field(None, ge=1)
    n_iter: int | None = Field(None, ge=1)
    tol: float | None = Field(None, gt=0)
    soft_floor: float | None = Field(None, gt=0)
    dirichlet_concentration: float | None = Field(None, gt=0)

    @model_validator(mode="before")
    @classmethod
    def _overrides_relative_to_the_preset(cls, data):
        """An override equal to the preset's own value is no override."""
        # An unknown preset is left for field validation to reject with a proper
        # message; looking it up here would raise a bare KeyError instead.
        if isinstance(data, dict) and data.get("preset", "improved") in PRESETS:
            base = preset_config(data.get("preset", "improved"))
            data = {
                k: (None if k in TUNABLE and v is not None and v == getattr(base, k) else v)
                for k, v in data.items()
            }
        return data

    def core(self) -> CoreHMMConfig:
        """The core HMMConfig to fit with: the preset, then the overrides."""
        base = preset_config(self.preset)
        return dataclasses.replace(
            base, **{k: getattr(self, k) for k in TUNABLE if getattr(self, k) is not None}
        )

    @classmethod
    def from_core(cls, core: CoreHMMConfig, phase: str = "LD", preset: str | None = None) -> HmmConfig:
        """The config for a core HMMConfig, against ``preset`` — or, when None, the
        preset that needs the fewest overrides to say it."""
        values = {k: getattr(core, k) for k in TUNABLE}
        if preset is None:
            preset = min(
                PRESETS,
                key=lambda p: sum(values[k] != getattr(preset_config(p), k) for k in TUNABLE),
            )
        return cls(phase=phase, preset=preset, **values)

    @classmethod
    def from_attrs(cls, attrs) -> HmmConfig | None:
        """The HMM a dataset records, or None if none was fitted on it."""
        core = load_hmm_config_from_attrs(_AttrsOnly(attrs))
        if core is None:
            return None
        phase = _PHASE_FROM_RECORDED.get(str(attrs.get("hmm_phase", "LD")), "LD")
        return cls.from_core(core, phase=phase)

    def overrides(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude_defaults=True)


class _AttrsOnly:
    """Just enough of a Dataset for load_hmm_config_from_attrs."""

    def __init__(self, attrs):
        self.attrs = {k: (v.item() if isinstance(v, np.generic) else v) for k, v in attrs.items()}


def hmm_source(master: xr.Dataset, phase: str) -> xr.Dataset:
    """The masked view the model is fitted on: out-of-phase minutes are NaN,
    which the workflow drops, so each fit is paradigm-pure."""
    if phase == "full":
        has_boundary = "first_DD_day" in master.coords or "split_minute" in master.coords
        arg = "both" if has_boundary else "auto"
    else:
        arg = phase
    return dam_utilities.select_phase(master, arg)


def merge_hmm_outputs(master: xr.Dataset, result: xr.Dataset, phase_used: str) -> xr.Dataset:
    """Attach the decoded states to ``master``; nothing else of ``result`` comes.

    ``result`` is a select_phase() MASKED VIEW — its activity, moving and sleep
    are NaN-masked and float-upcast — so it must never replace the master. Only
    hmm_state / hmm_sleep / hmm_confidence (full-axis, -1 out of phase) and the
    hmm_ attrs come across, in the master's fly order and without the coords the
    phase view carries (the same two hazards as BACKLOG 23).
    """
    order = master["id"].values
    hmm_vars = [v for v in ("hmm_state", "hmm_sleep", "hmm_confidence") if v in result.data_vars]
    out = master.drop_vars([v for v in hmm_vars if v in master.data_vars], errors="ignore")
    if hmm_vars:
        outputs = result[hmm_vars]
        outputs = outputs.drop_vars([c for c in outputs.coords if c not in outputs.dims])
        out = out.merge(outputs, compat="no_conflicts", join="outer")
    for k, v in result.attrs.items():
        if str(k).startswith("hmm_"):
            out.attrs[k] = v
    out.attrs["hmm_phase"] = phase_used
    return out.reindex(id=order)


@dataclasses.dataclass
class HmmResult:
    master: xr.Dataset
    #: per-group fitted models (core GenotypeModelResult), for the result views
    models: dict
    #: the core config the models were fitted with
    config: CoreHMMConfig


def run_hmm(master: xr.Dataset, config: HmmConfig, progress=None, verbose=False) -> HmmResult:
    """Fit the HMM on ``config.phase`` and decode every fly. Replaces any earlier fit."""
    src, phase_used = hmm_source(master, config.phase)
    core = config.core()
    result, models = run_genotype_workflow(src, core, verbose=verbose, progress_callback=progress)
    return HmmResult(merge_hmm_outputs(master, result, phase_used), models, core)
