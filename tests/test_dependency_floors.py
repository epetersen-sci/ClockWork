"""ci/constraints-minimum.txt pins exactly the floors pyproject.toml declares.

CI installs that file to prove the floors are real. If the two drift apart, CI
tests some other set of versions and a false floor ships unnoticed.
"""

import re
import tomllib

from packaging.requirements import Requirement
from packaging.version import Version

from conftest import REPO_ROOT


def _declared_floors():
    project = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    floors = {}
    for spec in project["project"]["dependencies"]:
        req = Requirement(spec)
        lows = [s.version for s in req.specifier if s.operator in (">=", "==")]
        assert lows, f"{req.name} has no floor; every runtime dependency needs one"
        floors[req.name.lower()] = Version(lows[0])
    return floors


def _pinned():
    pins = {}
    for line in (REPO_ROOT / "ci" / "constraints-minimum.txt").read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            name, version = re.split(r"==", line)
            pins[name.strip().lower()] = Version(version.strip())
    return pins


def test_every_dependency_is_pinned_at_its_floor():
    floors, pins = _declared_floors(), _pinned()
    assert set(pins) == set(floors), (
        f"missing from constraints: {sorted(set(floors) - set(pins))}; "
        f"not a dependency: {sorted(set(pins) - set(floors))}"
    )
    for name, floor in floors.items():
        # A post-release is the floor itself (netCDF4 1.7.1 only ever shipped
        # as 1.7.1.post1/.post2), so compare the release part.
        assert pins[name].release[: len(floor.release)] == floor.release, (
            f"{name}: constraints pin {pins[name]}, pyproject's floor is {floor}"
        )
