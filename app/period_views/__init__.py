"""The tabs of the Period & rhythmicity page, one module each.

The page (``app_pages/period_rhythmicity.py``) resolves the phase, the period
range and the sidebar once and hands every tab the same :class:`PeriodContext`,
so no tab can disagree with another about which epoch or which window a number
came from. That agreement used to be enforced across three pages by shadow
session keys; with one page it is just an argument.
"""

from dataclasses import dataclass


@dataclass
class PeriodContext:
    ds: object  # the master dataset
    period_ds: object  # where the period results are read from
    analysis_src: object  # the phase-masked view the analyses run on
    phase_selection: str  # "DD" / "LD" / "full"
    phase_arg: str  # what to pass as phase= to the core analyses
    min_period: float
    max_period: float
    min_days_floor: float
    max_bridge_gap: float
    facet_spec: object
    facet_table: object
    selected_groups: list
