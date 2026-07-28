"""
dataset_meta.py
================
Single-source-of-truth helpers for the dataset's phase / split state.

Every analysis or visualization page that branches on whether the dataset
is a full recording, an LD-only partition, or a DD-only partition should
read the phase from this module. The dataset itself is the source of
truth — session-state caches like ``dataset_LD`` / ``dataset_DD`` are
*derivative* and should not be the gating signal for "has the split been
applied" prompts.

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

Use :func:`dataset_phase` and :func:`is_split_applied` rather than
reading ``ds.attrs`` directly so the legacy fallback stays in one place.
"""

from __future__ import annotations

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
    if isinstance(explicit, (int,)):  # NetCDF round-trips bool→int
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
    ... def cached_violin(fp, ds, mode, algos):
    ...     ...
    >>> cached_violin(dataset_fingerprint(ds), ds, 'period', ('ac',))
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
    )
    attr_tuple = tuple((k, str(attrs.get(k))) for k in attr_keys if k in attrs)
    flag_coords = tuple(c for c in ("ac_rhythmic", "ls_rhythmic", "cwt_rhythmic") if c in ds.coords)
    return (ids, n_time, phase, bool(split), flag_coords, attr_tuple)


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
        partitioning has been precomputed without yet replacing the
        master itself (e.g. after the Preprocessing page's apply-split
        step that populates session_state.dataset_LD/DD).
    """
    if phase not in VALID_PHASES:
        raise ValueError(f"phase must be one of {VALID_PHASES}; got {phase!r}")
    ds.attrs["phase"] = phase
    if split_applied is None:
        split_applied = phase != PHASE_FULL
    ds.attrs["split_applied"] = bool(split_applied)
    return ds
