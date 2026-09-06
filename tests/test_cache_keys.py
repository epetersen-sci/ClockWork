"""Cached helpers must actually key on the dataset fingerprint (backlog item 14).

Streamlit's underscore rule is SYNTACTIC: every leading-underscore parameter is
excluded from a ``@st.cache_data`` key, not just the unhashable ones. Naming the
fingerprint ``_fp`` therefore removed the one argument that encodes which flies
are in the dataset, and the caches silently stopped tracking it. In the worst
case (``rhythmicity._build_period_summary_df``) the key was empty, so an exported
per-fly table was computed once per session and reused for every later dataset.

These tests are on parameter NAMES rather than behaviour on purpose: reproducing
the stale-cache symptom needs two datasets and a live cache, while the mistake
itself is visible in the signature and is the thing a future edit would repeat.
"""


import pytest

CACHED_HELPERS = [
    ("app_pages.sleep_activity", "_cached_zt_binned"),
    ("app_pages.sleep_activity", "_cached_summary_bars"),
    ("app_pages.sleep_activity", "_cached_daily_pattern"),
    ("app_pages.sleep_activity", "_cached_summary_table"),
    ("app_pages.sleep_activity", "_cached_bout_duration_lines"),
    ("app_pages.periodograms", "_curves_by_group"),
    ("app_pages.rhythmicity", "_build_period_summary_df"),
]


def _source_of(module_path, func_name):
    """Read the def line out of the page source.

    Pages are top-level scripts that call st.* at import time, so they cannot be
    imported here — parse the text instead.
    """
    from conftest import REPO_ROOT

    path = REPO_ROOT / "app" / (module_path.replace(".", "/") + ".py")
    src = path.read_text(encoding="utf-8")
    marker = f"def {func_name}("
    idx = src.index(marker)
    end = src.index("):", idx)
    return src[idx:end]  # signature body, no trailing ")"


def _params_of(module_path, func_name):
    sig = _source_of(module_path, func_name)
    inner = sig[sig.index("(") + 1 :]
    return [p.strip() for p in inner.replace("\n", " ").split(",") if p.strip()]


@pytest.mark.parametrize(("module_path", "func_name"), CACHED_HELPERS)
def test_fingerprint_param_has_no_leading_underscore(module_path, func_name):
    sig = _source_of(module_path, func_name)
    first_param = sig[sig.index("(") + 1 :].replace("\n", " ").split(",")[0].strip()
    assert first_param == "fp", (
        f"{module_path}.{func_name} takes its fingerprint as {first_param!r}. "
        "A leading underscore drops it from the cache key, so the cache stops "
        "tracking which flies are in the dataset."
    )


@pytest.mark.parametrize(("module_path", "func_name"), CACHED_HELPERS)
def test_dataset_param_keeps_its_underscore(module_path, func_name):
    """The other half of the rule: without it Streamlit tries to hash the Dataset,
    which is the slowness dataset_fingerprint exists to avoid."""
    sig = _source_of(module_path, func_name)
    params = [p.strip() for p in sig[sig.index("(") + 1 :].replace("\n", " ").split(",")]
    assert "_ds" in params, f"{module_path}.{func_name} should take the dataset as `_ds`"


def test_streamlit_underscore_rule_still_holds():
    """Pin the upstream behaviour these tests are written against.

    If Streamlit ever starts hashing leading-underscore params, this fails and the
    naming rule above stops being load-bearing.
    """
    import streamlit as st

    calls = []

    @st.cache_data
    def f(_excluded, included):
        calls.append((_excluded, included))
        return _excluded

    assert f((1,), "a") == (1,)
    assert f((2,), "a") == (1,), "a leading-underscore arg should NOT affect the key"
    assert len(calls) == 1

    assert f((2,), "b") == (2,), "a normal arg SHOULD affect the key"
    assert len(calls) == 2


def test_fingerprint_distinguishes_fly_subsets(master_ds):
    """dataset_fingerprint must change when the fly set changes — that is the whole
    reason it is passed to the cached helpers."""
    from dataset_meta import dataset_fingerprint

    full = dataset_fingerprint(master_ds)
    subset = dataset_fingerprint(master_ds.isel(id=slice(0, 2)))
    assert full != subset
