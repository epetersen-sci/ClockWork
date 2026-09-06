"""Sleep analysis must not reorder the fly dimension.

``groupby("id")`` iterates in SORTED id order and ``expand_dims`` gives each
per-fly mask an object-dtype id index, so the stacked masks disagree with the
dataset on both order and dtype whenever its ids are not already string-sorted.
``xr.merge``'s historical default (``join="outer"``) reconciled that by silently
re-sorting the merged dataset's flies — and xarray is changing that default to
``join="exact"``, which turns the same mismatch into a hard AlignmentError.

So this guards two things at once: a silent reordering today, and a crash later.
"""

import warnings

import numpy as np
import pytest
import xarray as xr

import sleep_analysis as sa


def _run(ds):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = sa.sleep_analysis(ds, phase="both")
    join_warnings = [
        w
        for w in caught
        if issubclass(w.category, FutureWarning) and "join" in str(w.message)
    ]
    return out, join_warnings


@pytest.fixture
def unsorted_ds(master_ds):
    """A dataset whose ids are deliberately NOT in string-sorted order."""
    scrambled = list(range(master_ds.sizes["id"]))[::-1]
    ds = master_ds.isel(id=scrambled).drop_vars(
        [v for v in ("sleep", "sleep_short", "sleep_intermediate", "sleep_long")
         if v in master_ds.data_vars]
    )
    assert [str(i) for i in ds.id.values] != sorted(str(i) for i in ds.id.values)
    return ds


def test_fly_order_is_preserved(unsorted_ds):
    before = [str(i) for i in unsorted_ds.id.values]
    out, _ = _run(unsorted_ds)
    assert [str(i) for i in out.id.values] == before, (
        "sleep analysis reordered the fly dimension; a later export would then "
        "list its flies differently for reasons nothing in the export changed"
    )


def test_no_xarray_join_futurewarning(unsorted_ds):
    """The merge must not rely on the join default that xarray is changing."""
    _, join_warnings = _run(unsorted_ds)
    assert not join_warnings, (
        f"xr.merge is relying on the changing join default: "
        f"{[str(w.message)[:120] for w in join_warnings]}"
    )


def test_results_follow_the_right_flies(unsorted_ds):
    """Order preservation is only worth having if the values move with it.

    Run on the scrambled dataset and on the same data sorted, then compare each
    fly's sleep trace by id — a reordering bug that also mislabelled would pass
    a pure order check.
    """
    out_scrambled, _ = _run(unsorted_ds)
    sorted_ids = sorted(str(i) for i in unsorted_ds.id.values)
    out_sorted, _ = _run(unsorted_ds.sel(id=sorted_ids))

    for fly in sorted_ids[:3]:
        np.testing.assert_array_equal(
            out_scrambled["sleep"].sel(id=fly).values,
            out_sorted["sleep"].sel(id=fly).values,
            err_msg=f"{fly}'s sleep trace depends on the input id order",
        )


def test_missing_fly_raises_rather_than_filling_nan(master_ds, monkeypatch):
    """The alignment uses .sel, not .reindex, so a genuinely missing fly is loud.

    Silently NaN-filling a row would look like a fly that simply never slept.
    """
    ds = master_ds.drop_vars(
        [v for v in ("sleep", "sleep_short", "sleep_intermediate", "sleep_long")
         if v in master_ds.data_vars]
    )
    real_concat = xr.concat

    def drop_one(objs, *args, **kwargs):
        out = real_concat(objs, *args, **kwargs)
        if "id" in out.dims and out.sizes["id"] > 1:
            return out.isel(id=slice(1, None))  # lose a fly on the way through
        return out

    monkeypatch.setattr(sa.xr, "concat", drop_one)
    with pytest.raises(KeyError):
        sa.sleep_analysis(ds, phase="both")
