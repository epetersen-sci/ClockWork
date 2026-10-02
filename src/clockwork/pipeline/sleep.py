"""Sleep: detecting bouts, and sorting them into short / intermediate / long.

Two steps, and only the first is expensive. Detection finds each fly's
immobility bouts (governed by the threshold); classification re-cuts that bout
table at the state boundaries and moves no bout. One config covers both,
because both write the same three attrs, and the Sleep states page re-cuts the
boundaries on its own without re-detecting.
"""

from __future__ import annotations

import xarray as xr
from pydantic import Field, model_validator

from clockwork.core import sleep_analysis
from clockwork.core.dataset_meta import PHASE_DD, PHASE_LD, dataset_phase
from clockwork.pipeline._config import StepConfig

#: The standard Drosophila sleep definition (Shaw et al. 2000): five minutes of
#: immobility.
DEFAULT_THRESHOLD_SECONDS = 300


class SleepConfig(StepConfig):
    #: Seconds of continuous immobility that count as sleep.
    threshold_seconds: int = Field(DEFAULT_THRESHOLD_SECONDS, ge=60, le=1800)
    #: Bouts shorter than this are short sleep.
    short_max_minutes: float = Field(30.0, gt=0)
    #: Bouts shorter than this (and not short) are intermediate; the rest are long.
    intermediate_max_minutes: float = Field(60.0, gt=0)

    ATTRS = {
        "threshold_seconds": "sleep_threshold_seconds",
        "short_max_minutes": "sleep_short_max_min",
        "intermediate_max_minutes": "sleep_inter_max_min",
    }
    # Only detection writes the threshold; re-classifying writes the boundaries
    # alone. So the threshold is what says "sleep was detected".
    PRESENCE_ATTR = "sleep_threshold_seconds"
    RECORDED_BY_CORE = frozenset(ATTRS)

    @model_validator(mode="after")
    def _ordered(self):
        if not self.intermediate_max_minutes > self.short_max_minutes:
            raise ValueError(
                f"intermediate_max_minutes ({self.intermediate_max_minutes:g}) must be "
                f"above short_max_minutes ({self.short_max_minutes:g})"
            )
        return self


def detection_phase(ds: xr.Dataset) -> tuple[str, str]:
    """``(phase argument, label)``: which epoch detection runs on.

    Always the whole recording when there is one, so the epoch stays a VIEWING
    choice on each page. Detecting one epoch writes masks that are missing
    everywhere else, and every figure is then blank for the other epoch.
    Detection has no phase-dependent parameter, so this is not a different
    method: a bout straddling the LD/DD boundary just stays one bout.

    A file that IS a single epoch is passed its own phase: ``select_phase``
    returns a matching request unchanged, and raises if asked for the epoch the
    file is not, which is the failure worth having.
    """
    stamped = dataset_phase(ds)
    if stamped in (PHASE_LD, PHASE_DD):
        return stamped, stamped
    return "both", "LD+DD"


def detect_sleep(ds: xr.Dataset, config: SleepConfig, progress=None) -> xr.Dataset:
    """Find every fly's sleep bouts and classify them. Replaces any earlier run."""
    if "moving" not in ds.data_vars:
        raise ValueError("sleep needs the movement data curation computes; run curation first")
    phase, _ = detection_phase(ds)
    return sleep_analysis.sleep_analysis(
        ds,
        sleep_threshold_sec=config.threshold_seconds,
        short_max_min=config.short_max_minutes,
        inter_max_min=config.intermediate_max_minutes,
        phase=phase,
        progress_callback=progress,
    )


def reclassify_sleep(ds: xr.Dataset, config: SleepConfig) -> xr.Dataset:
    """Re-cut the existing bouts at ``config``'s state boundaries. No bout moves.

    The threshold in ``config`` is not used: changing it would need detection
    again, and :func:`detect_sleep` is the step that does that.
    """
    return sleep_analysis.reclassify_sleep_states(
        ds,
        short_max_min=config.short_max_minutes,
        inter_max_min=config.intermediate_max_minutes,
    )
