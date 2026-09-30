"""Full-precision fingerprint of what every estimator says about example_data.

The snapshot tests hold results to rtol=1e-3 plus per-field allowances, which is
right for "did a refactor move anything that matters" and far too coarse for
"does this dependency version compute the SAME numbers as the pinned set". This
answers the second question: it runs the same paths the pages take and saves
every numeric output unrounded, so two environments can be diffed exactly.

    python tools/numeric_fingerprint.py run  out.npz [--cpu] [--quick]
    python tools/numeric_fingerprint.py diff a.npz b.npz

``--cpu`` hides the GPU (CUDA_VISIBLE_DEVICES=-1 and HMM_USE_GPU=0) before
anything imports torch, so the CWT and ZIP-HMM take their CPU paths in the
worker processes too; a monkeypatch would not reach spawned workers.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
EXAMPLE = REPO / "example_data"
MONITORS = [17, 18]
# Same four as tests/test_period_estimators_snapshot.py: a rhythmic and an
# arrhythmic record on each monitor.
CWT_FLIES = ["20250115_17_1", "20250115_17_2", "20250115_18_1", "20250115_18_2"]
VERSIONED = [
    "numpy", "scipy", "astropy", "PyWavelets", "pandas", "xarray", "statsmodels",
    "hmmlearn", "scikit-learn", "joblib", "netCDF4", "torch", "ptwt",
]


def _versions():
    from importlib.metadata import PackageNotFoundError, version

    out = {"python": sys.version.split()[0]}
    for name in VERSIONED:
        try:
            out[name] = version(name)
        except PackageNotFoundError:
            out[name] = None
    # What is actually imported, which is not always what the metadata says.
    import numpy
    import pywt
    import scipy

    out["numpy_imported"] = numpy.__version__
    out["scipy_imported"] = scipy.__version__
    # NOT a version check: the PyWavelets 1.9.0 wheel ships a stale version.py
    # and reports "1.8.0" here while running 1.9 code. The metadata version
    # above is the true one; this is kept to make that mismatch visible.
    out["pywt_self_reported"] = pywt.__version__
    return out


def _load_example(tmp):
    import pandas as pd

    from clockwork.core import dam_processor, dam_utilities

    for m in MONITORS:
        shutil.copy(EXAMPLE / f"Monitor{m}.txt", tmp / f"Monitor{m}.txt")
    meta = pd.read_excel(EXAMPLE / "metadata.xlsx")
    meta[meta["Monitor"].isin(MONITORS)].to_csv(tmp / "metadata.csv", index=False)
    proc = dam_processor.MetadataProcessor(
        str(tmp / "metadata.csv"), str(tmp), gap_threshold_hours=1.0
    )
    metadata, data = proc.run()
    return dam_utilities.create_xarray_dataset(
        dam_utilities.convert_to_relative_time(data, metadata), metadata
    )


def _collect(prefix, ds, out):
    """Every numeric data_var and coord of ``ds``, unrounded, keyed by prefix."""
    import numpy as np

    for name in list(ds.data_vars) + list(ds.coords):
        vals = ds[name].values
        if vals.dtype.kind in "fiub":
            out[f"{prefix}/{name}"] = np.asarray(vals)
        elif vals.dtype.kind == "M":
            out[f"{prefix}/{name}"] = vals.astype("datetime64[ns]").astype(np.int64)


def run(path, quick):
    import numpy as np

    from clockwork.core import dam_utilities, periodograms, sleep_analysis
    from clockwork.core import rhythmicity_classification as rc
    from clockwork.core.calibrations import (
        DEFAULT_CWT_MAX_PERIOD,
        DEFAULT_CWT_MIN_PERIOD,
        DEFAULT_MAX_BRIDGE_GAP_MINUTES,
        DEFAULT_MIN_DD_DAYS_FLOOR,
    )
    from clockwork.core.hmm_models import HMMConfig, run_genotype_workflow
    from clockwork.core.preprocessing import (
        ac_default_config,
        cwt_default_config,
        ls_default_config,
        preprocess_activity,
    )

    page_kw = {
        "min_period": DEFAULT_CWT_MIN_PERIOD,
        "max_period": DEFAULT_CWT_MAX_PERIOD,
        "min_num_days": DEFAULT_MIN_DD_DAYS_FLOOR,
    }
    bridge_kw = {"max_bridge_gap_minutes": DEFAULT_MAX_BRIDGE_GAP_MINUTES}
    window = (DEFAULT_CWT_MIN_PERIOD, DEFAULT_CWT_MAX_PERIOD)
    out, timings = {}, {}

    def timed(name, fn):
        t0 = time.perf_counter()
        result = fn()
        timings[name] = round(time.perf_counter() - t0, 1)
        print(f"  {name}: {timings[name]} s", flush=True)
        return result

    with tempfile.TemporaryDirectory() as tmp:
        ds = timed("import", lambda: _load_example(Path(tmp)))
    _collect("import", ds, out)

    dd, _ = dam_utilities.select_phase(ds, "DD")
    for name, cfg in (("ls", ls_default_config()), ("ac", ac_default_config()),
                      ("cwt", cwt_default_config())):
        pp = preprocess_activity(dd.sel(id=CWT_FLIES), cfg)
        out[f"preprocess_{name}/activity"] = pp["activity"].transpose("id", "time").values

    ls = timed("lomb_scargle", lambda: periodograms.lomb_scargle_analysis(
        preprocess_activity(dd, ls_default_config()), phase="DD", n_processes=2, **page_kw))
    _collect("ls", rc.classify_lomb_scargle(ls, period_window=window), out)

    ac = timed("autocorrelation", lambda: periodograms.autocorrelation_analysis(
        preprocess_activity(dd, ac_default_config()), phase="DD", n_processes=2,
        **page_kw, **bridge_kw))
    _collect("ac", rc.classify_autocorrelation(ac, period_window=window), out)

    mesa = timed("mesa", lambda: periodograms.mesa_analysis(
        preprocess_activity(dd, ac_default_config()), phase="DD", n_processes=2,
        **page_kw, **bridge_kw))
    _collect("mesa", mesa, out)

    cwt, _ = timed("cwt", lambda: periodograms.wavelet_analysis(
        preprocess_activity(dd.sel(id=CWT_FLIES), cwt_default_config()), phase="DD",
        n_processes=2, **page_kw, **bridge_kw))
    _collect("cwt", rc.classify_cwt(cwt), out)

    # Sleep reads `moving`, which curation derives; run it the way the Curate page
    # does (its defaults), so the sleep input is the curated cohort.
    live = timed("curate", lambda: dam_utilities.curate_dead_animals(ds))[0]
    _collect("curate", live, out)
    slept = timed("sleep", lambda: sleep_analysis.sleep_analysis(live, phase=None))
    _collect("sleep", slept, out)

    if not quick:
        # Few restarts: this is a determinism check on the numerics, not a fit
        # anyone should read. Each preset exercises a different emission model.
        for preset in ("wiggin", "improved"):
            cfg = getattr(HMMConfig, preset)()
            cfg.n_restarts, cfg.n_iter = 3, 50
            hmm_ds, _ = timed(f"hmm_{preset}", lambda cfg=cfg: run_genotype_workflow(
                slept, cfg, verbose=False))
            _collect(f"hmm_{preset}", hmm_ds[[v for v in hmm_ds.data_vars
                                             if v.startswith("hmm_")]], out)

    meta = {"versions": _versions(), "timings": timings,
            "cpu_forced": os.environ.get("CUDA_VISIBLE_DEVICES") == "-1",
            "gpu_cwt_available": bool(periodograms.PTWT_AVAILABLE)}
    np.savez_compressed(path, __meta__=np.array(json.dumps(meta)), **out)
    print(json.dumps(meta, indent=1))


def diff(a_path, b_path):
    import numpy as np

    a, b = np.load(a_path), np.load(b_path)
    for label, f in (("A", a), ("B", b)):
        m = json.loads(str(f["__meta__"]))
        v = m["versions"]
        print(f"{label}: " + ", ".join(f"{k}={v[k]}" for k in v if v[k]) +
              f" | gpu_cwt_available={m.get('gpu_cwt_available', m.get('gpu_cwt_used'))}")
    keys = sorted((set(a.files) | set(b.files)) - {"__meta__"})
    identical, rows = 0, []
    for k in keys:
        if k not in a.files or k not in b.files:
            rows.append((k, "only in " + ("A" if k in a.files else "B")))
            continue
        x, y = a[k], b[k]
        if x.shape != y.shape:
            rows.append((k, f"shape {x.shape} vs {y.shape}"))
            continue
        if x.dtype.kind in "fc":
            nan_x, nan_y = np.isnan(x), np.isnan(y)
            if (nan_x != nan_y).any():
                rows.append((k, f"NaN pattern differs at {(nan_x != nan_y).sum()} cells"))
                continue
            fin = ~nan_x
            if np.array_equal(x[fin], y[fin]):
                identical += 1
                continue
            d = np.abs(x[fin] - y[fin])
            scale = np.maximum(np.abs(x[fin]), 1e-300)
            rows.append((k, f"max abs {d.max():.3g}, max rel {(d / scale).max():.3g}, "
                            f"{(d > 0).sum()}/{d.size} cells differ"))
        elif np.array_equal(x, y):
            identical += 1
        else:
            rows.append((k, f"{(x != y).sum()}/{x.size} values differ"))
    print(f"\n{identical}/{len(keys)} arrays bit-identical")
    for k, msg in rows:
        print(f"  {k}: {msg}")
    return 0 if not rows else 1


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("out")
    r.add_argument("--cpu", action="store_true")
    r.add_argument("--quick", action="store_true", help="skip the HMM")
    d = sub.add_parser("diff")
    d.add_argument("a")
    d.add_argument("b")
    args = p.parse_args()
    if args.cmd == "run":
        if args.cpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # "" does NOT hide it on Windows
            os.environ["HMM_USE_GPU"] = "0"
        run(args.out, args.quick)
        return 0
    return diff(args.a, args.b)


if __name__ == "__main__":
    sys.exit(main())
