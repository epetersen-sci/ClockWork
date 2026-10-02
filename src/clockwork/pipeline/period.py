"""Period & rhythmicity: the four estimators and their rhythmic calls.

One config (:class:`PeriodConfig`) holds the settings every method shares —
phase, period range, DD-days floor, gap-bridging ceiling — and one block per
method that runs. Shared settings can be overridden per method, because the GUI
lets a user re-run one method with a different range, and an exported config
has to say what actually produced each result.

The period range is also the classification window: the Rhythmicity cutoff tab
classifies against it, and so does :func:`classify_period`.
"""

from __future__ import annotations

import dataclasses
from typing import Any, ClassVar, Literal

import numpy as np
import xarray as xr
from pydantic import BaseModel, ConfigDict, Field, model_validator

from clockwork.core import dam_utilities, periodograms
from clockwork.core.calibrations import (
    DEFAULT_CWT_MAX_PERIOD,
    DEFAULT_CWT_METHOD,
    DEFAULT_CWT_MIN_PERIOD,
    DEFAULT_MAX_BRIDGE_GAP_MINUTES,
    DEFAULT_MIN_DD_DAYS_FLOOR,
)
from clockwork.core.dataset_meta import PHASE_DD, PHASE_LD, dataset_phase
from clockwork.core.preprocessing import (
    PreprocessConfig,
    ac_default_config,
    cwt_default_config,
    ls_default_config,
    preprocess_activity,
)
from clockwork.core.rhythmicity_classification import (
    DEFAULT_AC_RI_THRESHOLD,
    DEFAULT_LS_POWER_THRESHOLD,
    classify_all,
    cwt_threshold_for,
)

Phase = Literal["DD", "LD", "full"]
#: What the estimators record as ``<method>_phase`` for each config phase.
_RECORDED_PHASE = {"DD": "DD", "LD": "LD", "full": "both"}
_PHASE_FROM_RECORDED = {v: k for k, v in _RECORDED_PHASE.items()}

#: The GUI's offered CWT resolutions (voices per octave).
CWT_VOICES = (8, 10, 12, 16, 32, 64, 128, 256, 512, 1024)
CWT_REDUCTIONS = ("ar1", "global_rednoise", "global", "global_baseline", "global_neighbor", "ridge")


class _Part(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

#: YAML field -> core.preprocessing.PreprocessConfig field. The core names are
#: also the recorded attrs' suffixes (``<method>_prep_<core name>``).
_PREP_FIELDS = {
    "bin_minutes": "bin_minutes",
    "smooth_sigma_minutes": "smooth_sigma_min",
    "lowpass_hours": "lopass_hours",
    "detrend": "detrend",
    "normalize": "normalize",
    "rolling_window_hours": "rolling_window_h",
}


class Preprocessing(_Part):
    """bin → smooth → low-pass → detrend → normalize; each step off at its sentinel.

    In a config file this block lists only what differs from the METHOD's own
    defaults (autocorrelation's are a 4 h low-pass and a linear detrend; the
    others' are none), never from these field defaults.
    """

    bin_minutes: int = Field(0, ge=0)
    smooth_sigma_minutes: float = Field(0.0, ge=0)
    lowpass_hours: float = Field(0.0, ge=0)
    detrend: Literal["none", "linear"] = "none"
    normalize: Literal["none", "zscore", "robust", "envelope", "rolling"] = "none"
    rolling_window_hours: float = Field(24.0, gt=0)

    def to_core(self) -> PreprocessConfig:
        return PreprocessConfig(**{core: getattr(self, f) for f, core in _PREP_FIELDS.items()})

    @classmethod
    def from_core(cls, cfg: PreprocessConfig) -> Preprocessing:
        return cls(**{f: getattr(cfg, core) for f, core in _PREP_FIELDS.items()})


# ---------------------------------------------------------------------------
# The methods
# ---------------------------------------------------------------------------


class _Method(_Part):
    #: attr prefix the estimator records under (``ls_``, ``ac_``, ...)
    PREFIX: ClassVar[str]
    #: the PreprocessConfig the GUI starts this method from
    DEFAULT_PREPROCESSING: ClassVar[PreprocessConfig]
    #: does this method take the gap-bridging ceiling? (Lomb-Scargle is gap-native)
    BRIDGES_GAPS: ClassVar[bool] = True

    # Shared settings, overridable per method. None = use the period-level value.
    phase: Phase | None = None
    period_range_hours: tuple[float, float] | None = None
    min_dd_days: float | None = Field(None, ge=0)
    max_bridge_gap_minutes: float | None = Field(None, ge=0)

    preprocessing: Preprocessing | None = None

    @model_validator(mode="before")
    @classmethod
    def _preprocessing_relative_to_the_method(cls, data):
        """A partial preprocessing block changes the METHOD's defaults, not blanks."""
        if isinstance(data, dict) and isinstance(data.get("preprocessing"), Preprocessing):
            data = {**data, "preprocessing": data["preprocessing"].model_dump()}
        if isinstance(data, dict) and isinstance(data.get("preprocessing"), dict):
            base = Preprocessing.from_core(cls.DEFAULT_PREPROCESSING).model_dump()
            merged = {**base, **data["preprocessing"]}
            # The method's own defaults, spelt out, ARE no override: normalise to
            # None so a config read back off a dataset equals the one that ran.
            data = {**data, "preprocessing": None if merged == base else merged}
        return data

    @model_validator(mode="after")
    def _check(self):
        if not self.BRIDGES_GAPS and self.max_bridge_gap_minutes is not None:
            raise ValueError("Lomb-Scargle is gap-native and takes no max_bridge_gap_minutes")
        if self.period_range_hours is not None:
            _check_range(self.period_range_hours)
        return self

    def preprocess(self) -> PreprocessConfig:
        return (self.preprocessing.to_core() if self.preprocessing else self.DEFAULT_PREPROCESSING)

    def overrides(self) -> dict[str, Any]:
        out = {}
        for name, field in type(self).model_fields.items():
            value = getattr(self, name)
            if name == "preprocessing":
                if value is None:
                    continue
                base = Preprocessing.from_core(self.DEFAULT_PREPROCESSING)
                diff = {
                    k: v for k, v in value.model_dump(mode="json").items()
                    if v != getattr(base, k)
                }
                if diff:
                    out[name] = diff
            elif value != field.default:
                out[name] = list(value) if isinstance(value, tuple) else value
        return out


class _Classified(_Method):
    """A method with a rhythmic call of its own."""

    #: classify after estimating. False = leave the flies unclassified.
    classify: bool = True
    #: None = the calibrated default for this method.
    rhythmic_threshold: float | None = Field(None, ge=0)
    #: The period window a fly must fall in to be called rhythmic. None = this
    #: method's period range, which is what a fresh run uses. (The Rhythmicity
    #: cutoff tab classifies against the page's CURRENT range, which can differ
    #: from the range the method last ran with; this records that.)
    rhythmic_window_hours: tuple[float, float] | None = None

    THRESHOLD_ATTR: ClassVar[str]
    FLAG: ClassVar[str]

    def threshold(self) -> float:
        return self.rhythmic_threshold if self.rhythmic_threshold is not None else self.default_threshold()

    #: Attrs recording the classification window.
    WINDOW_ATTRS: ClassVar[tuple[str, str] | None] = None

    def default_threshold(self) -> float:
        raise NotImplementedError


class LombScargle(_Classified):
    PREFIX = "ls"
    DEFAULT_PREPROCESSING = ls_default_config()
    BRIDGES_GAPS = False
    THRESHOLD_ATTR = "ls_power_threshold"
    FLAG = "ls_rhythmic"
    WINDOW_ATTRS = ("ls_period_window_min", "ls_period_window_max")

    oversampling: int = Field(8, ge=1, le=64)
    fap: Literal["baluev", "naive", "bootstrap"] = "baluev"

    def default_threshold(self):
        return float(DEFAULT_LS_POWER_THRESHOLD)


class Autocorrelation(_Classified):
    PREFIX = "ac"
    DEFAULT_PREPROCESSING = ac_default_config()
    THRESHOLD_ATTR = "ac_ri_threshold"
    FLAG = "ac_rhythmic"
    WINDOW_ATTRS = ("ac_period_window_min", "ac_period_window_max")

    peak_day: int = Field(2, ge=1, le=5)

    def default_threshold(self):
        return float(DEFAULT_AC_RI_THRESHOLD)


class CWT(_Classified):
    PREFIX = "cwt"
    DEFAULT_PREPROCESSING = cwt_default_config()
    THRESHOLD_ATTR = "cwt_rhythmicity_threshold"
    FLAG = "cwt_rhythmic"

    voices_per_octave: Literal[CWT_VOICES] = periodograms.DEFAULT_CWT_VOICES_PER_OCTAVE  # type: ignore[valid-type]
    reduction: Literal[CWT_REDUCTIONS] = DEFAULT_CWT_METHOD  # type: ignore[valid-type]

    def default_threshold(self):
        return float(cwt_threshold_for(self.reduction))


class MESA(_Method):
    """Maximum entropy spectral analysis. It has no significance test, so its
    rhythmic call is autocorrelation's: classify through ``autocorrelation``."""

    PREFIX = "mesa"
    # The GUI runs MESA on autocorrelation's preprocessing.
    DEFAULT_PREPROCESSING = ac_default_config()

    bin_minutes: int = Field(int(periodograms.DEFAULT_MESA_BIN_MINUTES), ge=5, le=60)
    #: Burg AR order: Dowse's N/3 (default), N/4, Akaike FPE, or a fixed order.
    order: Literal["n_over_3", "n_over_4", "fpe"] | int = periodograms.DEFAULT_MESA_ORDER_RULE

    def core_order(self):
        return None if self.order == "n_over_3" else self.order


class Methods(_Part):
    """The methods to run. One that is absent does not run."""

    lomb_scargle: LombScargle | None = None
    autocorrelation: Autocorrelation | None = None
    cwt: CWT | None = None
    mesa: MESA | None = None

    @model_validator(mode="after")
    def _at_least_one(self):
        # mesa without autocorrelation is allowed, as on the Analysis tab: MESA
        # estimates periods on its own, and only its rhythmic CALL is
        # autocorrelation's. (clockwork validate warns that it goes unclassified.)
        if not any(getattr(self, k) for k in METHOD_KEYS):
            raise ValueError("name at least one method to run")
        return self


#: Run order. Autocorrelation goes first because CWT group averages can be
#: limited to AC-rhythmic flies, which needs the AC call to exist already.
METHOD_KEYS = ("autocorrelation", "lomb_scargle", "mesa", "cwt")
_CLASS_FOR = {"lomb_scargle": LombScargle, "autocorrelation": Autocorrelation, "cwt": CWT, "mesa": MESA}
#: The GUI's short method keys.
GUI_KEY = {"lomb_scargle": "ls", "autocorrelation": "ac", "cwt": "cwt", "mesa": "mesa"}
_FROM_GUI_KEY = {v: k for k, v in GUI_KEY.items()}


def _check_range(rng):
    lo, hi = rng
    if not 0 < lo < hi:
        raise ValueError(f"period range {list(rng)} must be 0 < min < max (hours)")


class PeriodConfig(_Part):
    phase: Phase = "DD"
    #: The period search range AND the rhythmic classification window.
    period_range_hours: tuple[float, float] = (float(DEFAULT_CWT_MIN_PERIOD), float(DEFAULT_CWT_MAX_PERIOD))
    #: Flies whose longest analysable block is shorter than this get no period.
    min_dd_days: float = Field(float(DEFAULT_MIN_DD_DAYS_FLOOR), ge=0)
    #: Interior gaps up to this are bridged (CWT, AC, MESA only). 0 = off.
    max_bridge_gap_minutes: float = Field(float(DEFAULT_MAX_BRIDGE_GAP_MINUTES), ge=0)
    methods: Methods

    @model_validator(mode="after")
    def _check(self):
        _check_range(self.period_range_hours)
        return self

    # -- what one method actually runs with ----------------------------------

    def method(self, key: str) -> _Method:
        key = _FROM_GUI_KEY.get(key, key)
        m = getattr(self.methods, key)
        if m is None:
            raise KeyError(f"{key} is not configured to run")
        return m

    def effective(self, key: str) -> dict[str, Any]:
        """The shared settings as ``key`` uses them: its overrides, else the period's."""
        m = self.method(key)
        return {
            "phase": m.phase or self.phase,
            "period_range_hours": m.period_range_hours or self.period_range_hours,
            "min_dd_days": self.min_dd_days if m.min_dd_days is None else m.min_dd_days,
            "max_bridge_gap_minutes": (
                self.max_bridge_gap_minutes if m.max_bridge_gap_minutes is None else m.max_bridge_gap_minutes
            ),
        }

    # -- to and from the dataset -----------------------------------------------

    @classmethod
    def from_attrs(cls, attrs, coords=()) -> PeriodConfig | None:
        """The period config a dataset records, or None if no method ran on it.

        ``coords`` is the dataset's coord names: a rhythmic call counts as made
        only when its flag coord exists, not merely its threshold attr.
        """
        coords = set(coords)
        per_method = {}
        for key in METHOD_KEYS:
            klass = _CLASS_FOR[key]
            p = klass.PREFIX
            if f"{p}_min_period" not in attrs:
                continue
            fields: dict[str, Any] = {
                "phase": _PHASE_FROM_RECORDED.get(str(attrs.get(f"{p}_phase")), "DD"),
                "period_range_hours": (float(attrs[f"{p}_min_period"]), float(attrs[f"{p}_max_period"])),
                "min_dd_days": float(attrs.get(f"{p}_min_num_days", DEFAULT_MIN_DD_DAYS_FLOOR)),
            }
            if klass.BRIDGES_GAPS:
                fields["max_bridge_gap_minutes"] = float(
                    attrs.get(f"{p}_max_bridge_gap_minutes", DEFAULT_MAX_BRIDGE_GAP_MINUTES)
                )
            prep = _preprocessing_from_attrs(attrs, p)
            if prep is not None:
                fields["preprocessing"] = prep
            fields.update(_method_fields_from_attrs(key, attrs, coords))
            per_method[key] = fields

        if not per_method:
            return None
        shared = _hoist_shared(per_method)
        methods = {}
        for key, fields in per_method.items():
            klass = _CLASS_FOR[key]
            for name, value in shared.items():
                if fields.get(name) == value:
                    fields.pop(name, None)
            if "preprocessing" in fields:
                fields["preprocessing"] = fields["preprocessing"].model_dump()
            methods[key] = klass(**fields)
        return cls(**shared, methods=Methods(**methods))

    def overrides(self) -> dict[str, Any]:
        out = {}
        for name in ("phase", "period_range_hours", "min_dd_days", "max_bridge_gap_minutes"):
            value, default = getattr(self, name), type(self).model_fields[name].default
            if value != default:
                out[name] = list(value) if isinstance(value, tuple) else value
        out["methods"] = {
            key: getattr(self.methods, key).overrides()
            for key in METHOD_KEYS
            if getattr(self.methods, key) is not None
        }
        return out


def _preprocessing_from_attrs(attrs, prefix):
    names = {core: f"{prefix}_prep_{core}" for core in _PREP_FIELDS.values()}
    if not all(a in attrs for a in names.values()):
        return None  # recorded before preprocessing was recorded: the method default
    return Preprocessing(**{f: _plain(attrs[names[core]]) for f, core in _PREP_FIELDS.items()})


def _method_fields_from_attrs(key, attrs, coords):
    out: dict[str, Any] = {}
    if key == "lomb_scargle":
        out["oversampling"] = int(attrs.get("ls_oversampling", 8))
        out["fap"] = str(attrs.get("ls_fap_method", "baluev"))
    elif key == "autocorrelation":
        out["peak_day"] = int(attrs.get("ac_peak", 2))
    elif key == "cwt":
        out["voices_per_octave"] = int(round(1 / float(attrs["cwt_resolution"])))
        out["reduction"] = str(attrs.get("cwt_method", DEFAULT_CWT_METHOD))
    elif key == "mesa":
        out["bin_minutes"] = int(round(float(attrs.get("mesa_bin_minutes", 30))))
        order = _plain(attrs.get("mesa_order", periodograms.DEFAULT_MESA_ORDER_RULE))
        out["order"] = order if isinstance(order, str) else int(order)
    klass = _CLASS_FOR[key]
    if issubclass(klass, _Classified):
        if klass.FLAG in coords and klass.THRESHOLD_ATTR in attrs:
            recorded = float(attrs[klass.THRESHOLD_ATTR])
            default = klass(**{k: v for k, v in out.items() if k == "reduction"}).default_threshold()
            out["rhythmic_threshold"] = None if np.isclose(recorded, default) else recorded
            if klass.WINDOW_ATTRS and all(a in attrs for a in klass.WINDOW_ATTRS):
                window = tuple(float(attrs[a]) for a in klass.WINDOW_ATTRS)
                p = klass.PREFIX
                ran_over = (float(attrs[f"{p}_min_period"]), float(attrs[f"{p}_max_period"]))
                if window != ran_over:
                    out["rhythmic_window_hours"] = window
        else:
            out["classify"] = False
    return out


def _hoist_shared(per_method):
    """The shared settings, at the period level when every method agrees.

    Where methods disagree (one re-run with a different range, say) the most
    common value goes up and the others stay as per-method overrides, so the
    exported file is as short as it can be while still saying what ran.
    """
    shared = {}
    for name in ("phase", "period_range_hours", "min_dd_days", "max_bridge_gap_minutes"):
        values = [f[name] for f in per_method.values() if name in f]
        if values:
            # max() keeps the FIRST of equally common values, in METHOD_KEYS
            # order, so a tie resolves the same way on every export.
            shared[name] = max(values, key=values.count)
    return shared


def _plain(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, bytes):
        return value.decode()
    return value


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------


def phase_source(ds: xr.Dataset, phase: str):
    """``(analysis_src, phase_arg)``: the view the estimators consume, and the
    ``phase=`` to pass them. The same cascade the Period page resolves.

    ``analysis_src`` is a per-fly NaN-masked view of the WHOLE dataset; feeding a
    pre-sliced, re-zeroed object instead dropped the first phase days.
    """
    loaded = dataset_phase(ds)
    if loaded in (PHASE_LD, PHASE_DD):
        if phase != loaded:
            raise ValueError(f"this dataset is a saved {loaded} partition; phase must be {loaded}")
        src = ds.drop_vars([c for c in ("first_DD_day", "split_minute") if c in ds.coords])
        return dam_utilities.select_phase(src, "auto")[0], "auto"
    if "first_DD_day" in ds.coords:
        if phase == "full":
            raise ValueError("this dataset has an LD/DD boundary; choose phase DD or LD")
        return dam_utilities.select_phase(ds, phase)[0], phase
    if phase != "full":
        raise ValueError("this dataset has no LD/DD boundary (no first_DD_day); use phase: full")
    return dam_utilities.select_phase(ds, "auto")[0], "auto"


def merge_period_outputs(master: xr.Dataset, result: xr.Dataset) -> xr.Dataset:
    """Attach an estimator's or classifier's outputs to ``master``.

    The estimators run on a PREPROCESSED copy and return a dataset whose
    ``activity`` is that copy. Assigning it back would put detrended activity on
    the master, and every later sum over it would be wrong (negative DD totals,
    for one). So only the per-fly outputs, the rhythmic flags and the method's
    own attrs come across.
    """
    prefixes = ("cwt_", "ls_", "ac_", "mesa_")
    # The master's fly order, restored at the end: the outer join below SORTS the
    # union of two differently ordered id indexes (BACKLOG 23).
    order = master["id"].values
    scalar_vars = [v for v in result.data_vars if result[v].dims == ("id",)]
    array_vars = [
        v
        for v in result.data_vars
        if v != "activity"
        and "id" in result[v].dims
        and result[v].dims != ("id",)
        and v.startswith(prefixes)
    ]
    # The method's OWN per-fly coords — its rhythmic flags. Not every per-fly
    # coord on the result: that also carries whatever the phase view added, and
    # copying it gave an unsplit master a split_minute coord (BACKLOG 23).
    scalar_coords = [
        c
        for c in result.coords
        if str(c).startswith(prefixes) and c in result and result[c].dims == ("id",)
    ]
    to_drop = [v for v in (scalar_vars + array_vars + scalar_coords) if v in master]
    if to_drop:
        master = master.drop_vars(to_drop, errors="ignore")
    if scalar_vars or array_vars:
        outputs = result[scalar_vars + array_vars]
        # The variables travel with every coord on their dims, phase-view
        # artefacts included; keep only their own axes and the method's coords.
        outputs = outputs.drop_vars(
            [c for c in outputs.coords if c not in outputs.dims and not str(c).startswith(prefixes)]
        )
        master = master.merge(outputs, compat="no_conflicts", join="outer")
    if scalar_coords:
        # Aligned to the master's ids explicitly, rather than trusting
        # assign_coords to line up two differently ordered indexes.
        ids = master["id"].values
        master = master.assign_coords(
            {c: ("id", result[c].reindex(id=ids).values) for c in scalar_coords}
        )
    for k, v in result.attrs.items():
        if k.startswith(prefixes):
            master.attrs[k] = v
    return master.reindex(id=order)


def run_period_method(
    master: xr.Dataset,
    config: PeriodConfig,
    key: str,
    progress=None,
    *,
    group_averages: bool = False,
    average_only_rhythmic: bool = True,
    phase_label: str | None = None,
):
    """Run one estimator and merge its outputs onto ``master``.

    Returns ``(master, averages)``: ``averages`` is the CWT's per-group scalogram
    averages when ``group_averages`` is set (writing them to disk is the
    caller's business), else ``[]``. Does not classify; see :func:`classify_period`.
    """
    key = _FROM_GUI_KEY.get(key, key)
    method = config.method(key)
    eff = config.effective(key)
    src, phase_arg = phase_source(master, eff["phase"])
    pre = method.preprocess()
    data = preprocess_activity(src, pre)
    common = dict(
        min_period=eff["period_range_hours"][0],
        max_period=eff["period_range_hours"][1],
        phase=phase_arg,
        min_num_days=eff["min_dd_days"],
        progress_callback=progress,
    )
    averages = []
    if key == "cwt":
        result, averages = periodograms.wavelet_analysis(
            data,
            cwt_method=method.reduction,
            resolution=1 / method.voices_per_octave,
            max_bridge_gap_minutes=eff["max_bridge_gap_minutes"],
            compute_group_averages=bool(group_averages),
            group_coord="group",
            filter_nonrhythmic_for_average=bool(average_only_rhythmic),
            phase_label=phase_label,
            **common,
        )
    elif key == "lomb_scargle":
        result = periodograms.lomb_scargle_analysis(
            data, oversampling=method.oversampling, fap_method=method.fap, **common
        )
    elif key == "autocorrelation":
        result = periodograms.autocorrelation_analysis(
            data, ac_peak=method.peak_day, max_bridge_gap_minutes=eff["max_bridge_gap_minutes"], **common
        )
    else:
        result = periodograms.mesa_analysis(
            data,
            bin_minutes=method.bin_minutes,
            order=method.core_order(),
            max_bridge_gap_minutes=eff["max_bridge_gap_minutes"],
            **common,
        )
    # The preprocessing that produced this result, under the method's own prefix.
    # preprocess_activity stamps an unprefixed copy on the preprocessed data, but
    # that never reached the master (only <method>_ attrs are merged), and four
    # methods with one shared name could only keep the last one's anyway.
    result.attrs.update(
        {f"{method.PREFIX}_prep_{core}": v for core, v in dataclasses.asdict(pre).items()}
    )
    return merge_period_outputs(master, result), averages


def classify_period(master: xr.Dataset, key: str, threshold: float, window) -> xr.Dataset:
    """Make (or remake) one method's rhythmic call at ``threshold``.

    MESA's call is autocorrelation's, exactly as on the Rhythmicity cutoff tab.
    """
    key = _FROM_GUI_KEY.get(key, key)
    target = "autocorrelation" if key == "mesa" else key
    out = classify_all(
        master,
        ls_power_threshold=threshold if target == "lomb_scargle" else DEFAULT_LS_POWER_THRESHOLD,
        ac_ri_threshold=threshold if target == "autocorrelation" else DEFAULT_AC_RI_THRESHOLD,
        cwt_rhythmicity_threshold=threshold if target == "cwt" else None,
        period_window=tuple(window),
        run_ls=target == "lomb_scargle",
        run_ac=target == "autocorrelation",
        run_cwt=target == "cwt",
    )
    return merge_period_outputs(master, out)


def run_period(master: xr.Dataset, config: PeriodConfig, progress=None) -> xr.Dataset:
    """Every configured method, then its rhythmic call: the whole analysis."""
    for key in METHOD_KEYS:
        method = getattr(config.methods, key)
        if method is None:
            continue
        master, _ = run_period_method(master, config, key, progress)
        if isinstance(method, _Classified) and method.classify:
            window = getattr(method, "rhythmic_window_hours", None) or config.effective(key)["period_range_hours"]
            master = classify_period(master, key, method.threshold(), window)
    return master
