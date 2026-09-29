"""Pinned snapshots of what the analyses say about real data.

Real flies have no ground truth: nobody knows fly 17_3's "true" period. What the
suite CAN hold fixed is what the code currently says about it, once a person has
looked at the numbers and accepted them. After that, any change — a refactor, a
dependency bump, a recalibrated threshold — shows up as a named list of flies
whose numbers moved, instead of as nothing at all.

The workflow:

* ``pytest --update-snapshots`` writes ``tests/snapshots/<name>.json`` from the
  current code, with a human-readable ``_summary`` block at the top. Read the
  summary (and ``git diff`` the file) before committing it: accepting a snapshot
  is a decision, not a formality.
* A plain ``pytest`` compares against the committed file. A snapshot that is
  missing FAILS rather than writing itself — a test that silently creates its own
  expected values would pass on any output.
"""

import json
import math

from conftest import REPO_ROOT

SNAPSHOT_DIR = REPO_ROOT / "tests" / "snapshots"


def _jsonable(value):
    """Plain JSON types; NaN is written as null so the file stays valid JSON."""
    if hasattr(value, "item"):  # numpy scalar
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _differs(expected, actual, atol, rtol):
    if expected is None or actual is None:
        return (expected is None) != (actual is None)
    if isinstance(expected, bool) or isinstance(actual, bool):
        return bool(expected) != bool(actual)
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        return abs(expected - actual) > atol + rtol * abs(expected)
    if isinstance(expected, list) and isinstance(actual, list):
        return len(expected) != len(actual) or any(
            _differs(e, a, atol, rtol) for e, a in zip(expected, actual)
        )
    return expected != actual


def check_snapshot(name, records, update, *, atol=None, rtol=1e-3, summary=None):
    """Compare ``records`` with the committed snapshot ``name``, or rewrite it.

    ``records`` maps a key (usually a fly id) to a dict of fields. Numbers match
    when ``|a - e| <= atol[field] + rtol * |e|``: ``atol`` maps a field name to an
    absolute allowance (e.g. ``{"ls_period": 0.05}`` hours) for values that sit
    near zero or have a natural unit, and ``rtol`` covers the float noise of a
    different platform or BLAS. Booleans and strings must match exactly.

    ``summary`` is stored beside the records for the reviewer and never compared.
    """
    atol = atol or {}
    path = SNAPSHOT_DIR / f"{name}.json"
    current = {
        str(key): {field: _jsonable(v) for field, v in fields.items()}
        for key, fields in records.items()
    }

    if update:
        SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
        payload = {"_summary": _jsonable(summary) if summary else {}, "records": current}
        path.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
        print(f"\nsnapshot {name} written: {len(current)} records")
        for line in (summary or {}).get("lines", []):
            print("  " + line)
        return

    assert path.is_file(), (
        f"No snapshot at {path.relative_to(REPO_ROOT)}. Generate it with "
        "`python -m pytest --update-snapshots`, review it, and commit it."
    )
    expected = json.loads(path.read_text())["records"]

    problems = []
    for key in sorted(set(expected) - set(current)):
        problems.append(f"{key}: in the snapshot, missing from the output")
    for key in sorted(set(current) - set(expected)):
        problems.append(f"{key}: new in the output, not in the snapshot")
    for key in sorted(set(expected) & set(current)):
        exp, cur = expected[key], current[key]
        for field in sorted(set(exp) | set(cur)):
            e, a = exp.get(field), cur.get(field)
            if _differs(e, a, atol.get(field, 0.0), rtol):
                problems.append(f"{key}.{field}: snapshot {e!r}, now {a!r}")

    assert not problems, (
        f"{len(problems)} difference(s) from snapshot {name} "
        f"(regenerate with --update-snapshots ONLY if the change is intended):\n  "
        + "\n  ".join(problems[:60])
        + ("\n  ..." if len(problems) > 60 else "")
    )
