"""
dataset_meta.py
================
Single-source-of-truth helpers for the dataset's phase / split state.

Every analysis or visualization page that branches on whether the dataset
is a full recording, an LD-only partition, or a DD-only partition should
read the phase from this module. The dataset itself is the source of truth.
Pages used to gate "has the split been applied" prompts on the existence of
the ``dataset_LD`` / ``dataset_DD`` session caches instead; those caches are
gone, and the attrs here are the answer — they also survive a NetCDF round
trip, which the caches never did.

Canonical attrs
---------------
``ds.attrs['phase']`` : str ∈ {``'full'``, ``'LD'``, ``'DD'``}
    What the dataset's `time` axis represents.
    - ``'full'`` — recording covers both LD and DD epochs (or there is no
      split evidence on the file at all).
    - ``'LD'`` — light/dark-cycle partition produced by
      :func:`dam_utilities.split_xarray_dataset` with ``phase='LD'``.
    - ``'DD'`` — constant-darkness partition produced by
      :func:`dam_utilities.split_xarray_dataset` with ``phase='DD'``.

``ds.attrs['split_applied']`` : bool
    Whether a partitioning step has been performed on this file. ``True``
    for derivative LD/DD partitions and for full datasets whose splits
    have been precomputed on the Preprocessing page (in which case
    ``phase='full'`` because the master still spans both epochs).

Legacy alias (read-only, kept for backward-compat with files saved before
this convention)::

    ds.attrs['split_phase']  ∈ {'LD', 'DD', 'both'}
        - 'LD' / 'DD' map directly to ``phase``.
        - 'both' maps to ``phase='full'`` with ``split_applied=True``.

**Nothing writes ``split_phase`` any more** — backlog item 4 removed the
two writers (``data_curate_split.py`` and ``dam_utilities.split_xarray_dataset``).
The reads below are deliberately NOT dead code: every ``.nc`` saved before
that change still carries the alias, and these fallbacks are the only thing
that resolves such a file to the right phase. Deleting them would silently
re-label old saved datasets as ``'full'``. Keep them until you are willing
to say old files are unsupported.

Use :func:`dataset_phase` and :func:`is_split_applied` rather than
reading ``ds.attrs`` directly so the legacy fallback stays in one place.
"""

from __future__ import annotations

import numbers

import numpy as np
import xarray as xr

PHASE_FULL = "full"
PHASE_LD = "LD"
PHASE_DD = "DD"

VALID_PHASES = (PHASE_FULL, PHASE_LD, PHASE_DD)


def dataset_phase(ds: xr.Dataset) -> str:
    """Return the canonical phase label for ``ds``.

    Resolution order:
      1. ``ds.attrs['phase']`` if set to a valid value.
      2. Legacy ``ds.attrs['split_phase']`` mapping
         (``'LD' → 'LD'``, ``'DD' → 'DD'``, ``'both' → 'full'``).
      3. Fallback: ``'full'``.
    """
    val = ds.attrs.get("phase") if hasattr(ds, "attrs") else None
    if isinstance(val, str) and val in VALID_PHASES:
        return val
    legacy = ds.attrs.get("split_phase") if hasattr(ds, "attrs") else None
    if isinstance(legacy, str):
        if legacy in (PHASE_LD, PHASE_DD):
            return legacy
        if legacy == "both":
            return PHASE_FULL
    return PHASE_FULL


def is_split_applied(ds: xr.Dataset) -> bool:
    """True iff any LD/DD partitioning has been performed on ``ds``.

    Looks at ``ds.attrs['split_applied']``; if that's missing, infers
    from the legacy ``split_phase`` attr (any of ``'LD'``, ``'DD'``,
    ``'both'`` ⇒ split applied) or from the canonical phase being
    LD/DD. Returns ``False`` when no split-state evidence is present.
    """
    if not hasattr(ds, "attrs"):
        return False
    explicit = ds.attrs.get("split_applied")
    if isinstance(explicit, bool):
        return explicit
    # NetCDF round-trips bool→int, and xarray hands that back as a NUMPY integer
    # (np.int64), which is NOT an instance of Python's int under NumPy 2. Testing
    # `isinstance(explicit, int)` therefore missed every reloaded file and fell
    # through to the checks below. That went unnoticed while a split master also
    # carried the legacy split_phase='both' alias, which rescued it; once item 4
    # stopped writing the alias, a reloaded split master reported "not split" —
    # which hides the phase pickers and makes the SCAMP export refuse to run.
    # numbers.Integral covers Python ints and every NumPy integer width.
    if isinstance(explicit, numbers.Integral):
        return bool(explicit)
    if isinstance(explicit, np.bool_):  # np.bool_ is NOT an Integral
        return bool(explicit)
    legacy = ds.attrs.get("split_phase")
    if isinstance(legacy, str) and legacy in (PHASE_LD, PHASE_DD, "both"):
        return True
    canonical = ds.attrs.get("phase")
    return bool(isinstance(canonical, str) and canonical in (PHASE_LD, PHASE_DD))


def has_split_evidence(ds: xr.Dataset) -> bool:
    """True iff the dataset *looks* partitioned even when no canonical
    ``phase`` attr is set — used by the loader to decide whether to
    prompt the user to declare the phase manually for ambiguous files.

    Evidence considered: legacy ``split_phase`` attr, or the presence of
    a ``first_DD_day`` coord (which only appears on the master full
    dataset; its *absence* combined with a non-trivial recording can
    suggest a DD-only partition). Conservative — returns False on plain
    unsplit data so the user isn't bothered with a phase dropdown.
    """
    if not hasattr(ds, "attrs"):
        return False
    if ds.attrs.get("split_phase") is not None:
        return True
    if ds.attrs.get("phase") is not None:
        return False  # canonical attr is set; not ambiguous
    # No canonical attr, no legacy attr. The presence of a
    # `first_DD_day` coord on a file is itself a master-dataset
    # signature, so its *absence* on a NetCDF that nonetheless has
    # 'start_datetime' / 'stop_datetime' may indicate a partitioned
    # file. We do not return True for that case here because plain
    # unsplit recordings also lack `first_DD_day` (e.g. LD-only
    # recordings that were never set up for DD). The loader should
    # only prompt when there's positive evidence.
    return False


def dataset_fingerprint(ds: xr.Dataset) -> tuple:
    """Return a hashable identity tuple for ``ds`` suitable for use as a
    cache key with :func:`streamlit.cache_data`.

    Streamlit's hasher recurses into xarray objects and is too slow for
    page-rerun usage. This helper builds a small, primitive tuple that
    captures the things plots and aggregates actually depend on:

      - the fly id list (tuple of strings)
      - the time-axis length (cheap to read, sensitive to phase splits)
      - canonical phase + split_applied
      - the analysis-relevant attrs (CWT/LS/AC/AC-classification keys)

    Pass the fingerprint as the first arg of a cached function so
    Streamlit hashes the tuple, not the Dataset:

    >>> @st.cache_data
    ... def cached_violin(fp, _ds, mode, algos):
    ...     ...
    >>> cached_violin(dataset_fingerprint(ds), ds, 'period', ('ac',))

    The two parameter names are both load-bearing, and copies of this example
    have gone wrong in both directions:

    - ``fp`` takes NO leading underscore. Streamlit's rule is purely syntactic
      and drops EVERY leading-underscore parameter from the cache key, not just
      the unhashable ones — so ``_fp`` silently removes the fingerprint from the
      key it exists to form, and the cache stops tracking the dataset at all.
    - ``_ds`` DOES take one. Without it Streamlit hashes the Dataset, which is
      the slowness this whole helper exists to avoid.
    """
    if ds is None:
        return ()
    try:
        ids = tuple(str(x) for x in ds["id"].values.tolist())
    except Exception:
        ids = ()
    try:
        n_time = int(ds.sizes.get("time", 0))
    except Exception:
        n_time = 0
    phase = dataset_phase(ds)
    split = is_split_applied(ds)
    attrs = ds.attrs if hasattr(ds, "attrs") else {}
    # Pull the small set of attrs that materially affect cached results.
    attr_keys = (
        "cwt_method",
        "cwt_min_period",
        "cwt_max_period",
        "cwt_resolution",
        "cwt_wavelet",
        "cwt_phase",
        "ls_period_window_min",
        "ls_period_window_max",
        "ac_period_window_min",
        "ac_period_window_max",
        "ac_ri_threshold",
        "ac_use_dynamic_ci",
        "ls_power_threshold",
        "cwt_rhythmicity_threshold",
        # Sleep detection parameters: re-running with a different threshold
        # changes every sleep-derived figure, and nothing else in this tuple
        # moves when it does.
        "sleep_threshold_seconds",
        "sleep_short_max_min",
        "sleep_inter_max_min",
        "sleep_phase",
        "group_columns",
    )
    attr_tuple = tuple((k, str(attrs.get(k))) for k in attr_keys if k in attrs)
    flag_coords = tuple(c for c in ("ac_rhythmic", "ls_rhythmic", "cwt_rhythmic") if c in ds.coords)
    # The per-fly GROUP LABELS, not just the group_columns attr. Regrouping
    # rewrites this coord while leaving the id list, the time axis and the phase
    # untouched, so without it a regroup is invisible here — and every cached
    # group-level figure keeps being served against labels that no longer exist.
    # O(n_id), the same order as the id tuple above.
    groups = tuple(str(g) for g in ds["group"].values) if "group" in ds.coords else ()
    return (ids, n_time, phase, bool(split), flag_coords, attr_tuple, groups)


def stamp_phase(ds: xr.Dataset, phase: str, split_applied: bool | None = None) -> xr.Dataset:
    """Set canonical phase metadata on ``ds`` (in-place; returns ``ds``).

    Parameters
    ----------
    ds : xr.Dataset
        Target dataset. Mutated in place; the same reference is returned
        for chaining.
    phase : str
        One of ``'full'``, ``'LD'``, ``'DD'``.
    split_applied : bool or None
        If None, inferred from ``phase`` (LD/DD ⇒ True, full ⇒ False).
        Pass ``True`` explicitly to mark a master full dataset whose
        partitioning has been decided without replacing the master itself
        (the Curate & split page's apply-split step, which records the
        split parameters on the master so consumers can re-slice on demand).
    """
    if phase not in VALID_PHASES:
        raise ValueError(f"phase must be one of {VALID_PHASES}; got {phase!r}")
    ds.attrs["phase"] = phase
    if split_applied is None:
        split_applied = phase != PHASE_FULL
    ds.attrs["split_applied"] = bool(split_applied)
    return ds
