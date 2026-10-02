"""The analyses as steps: plain functions of a dataset and a config.

The Streamlit pages and the ``clockwork`` CLI both call these, so there is one
computation per analysis however it is started. Each config records itself on
the dataset it produced (``to_attrs``) and can be read back from it
(``from_attrs``), which is what the Export settings button is built on. See
docs/cli-config.md.
"""

from clockwork.pipeline._config import StepConfig
from clockwork.pipeline.data import (
    AmbiguousPhase,
    CurationConfig,
    CurationResult,
    GroupsConfig,
    ImportFailed,
    InputsConfig,
    RawImport,
    SplitConfig,
    apply_groups,
    build_dataset,
    curate,
    keep_for_groups,
    load_netcdf,
    read_monitors,
    split,
    split_report,
    subset,
)
from clockwork.pipeline.period import (
    CWT,
    MESA,
    Autocorrelation,
    LombScargle,
    Methods,
    PeriodConfig,
    Preprocessing,
    classify_period,
    merge_period_outputs,
    phase_source,
    run_period,
    run_period_method,
)

__all__ = [
    "CWT",
    "MESA",
    "AmbiguousPhase",
    "Autocorrelation",
    "LombScargle",
    "Methods",
    "PeriodConfig",
    "Preprocessing",
    "classify_period",
    "merge_period_outputs",
    "phase_source",
    "run_period",
    "run_period_method",
    "CurationConfig",
    "CurationResult",
    "GroupsConfig",
    "ImportFailed",
    "InputsConfig",
    "RawImport",
    "SplitConfig",
    "StepConfig",
    "apply_groups",
    "build_dataset",
    "curate",
    "keep_for_groups",
    "load_netcdf",
    "read_monitors",
    "split",
    "split_report",
    "subset",
]
