"""The base every step's config is built on: validation, and the round trip
through dataset attrs that the Export settings button depends on.

See docs/cli-config.md. A config field has two names: the one in the YAML file
(units in the name, the user-facing contract) and the attr it is recorded under
on the dataset. The attr names are the ones saved datasets already carry, and
ARCHITECTURE rule 1 says those do not change, so :attr:`StepConfig.ATTRS` is the
translation table between the two, declared once per config.
"""

from __future__ import annotations

import json
import typing
from typing import Any, ClassVar

import numpy as np
from pydantic import BaseModel, ConfigDict


class StepConfig(BaseModel):
    """One analysis step's parameters.

    Frozen, so a config that produced a result cannot be edited afterwards and
    silently stop describing it; and ``extra="forbid"``, so a misspelt YAML key is
    an error rather than a setting that quietly never applies.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    #: field name -> the dataset attr it is recorded under.
    ATTRS: ClassVar[dict[str, str]] = {}
    #: Fields stored as a JSON string, because netCDF attrs cannot nest.
    JSON_FIELDS: ClassVar[frozenset[str]] = frozenset()
    #: Fields whose attr the core function behind the step already writes. They
    #: are READ back like any other, but to_attrs leaves them alone: writing the
    #: same fact from two places is how two records of it come to disagree.
    RECORDED_BY_CORE: ClassVar[frozenset[str]] = frozenset()
    #: Present on the dataset iff this step has run. from_attrs returns None when
    #: it is absent, so a step that never ran cannot be read back as having run
    #: with its defaults. Defaults to the first ATTRS entry.
    PRESENCE_ATTR: ClassVar[str | None] = None

    # --- to the dataset ----------------------------------------------------

    def to_attrs(self) -> dict[str, Any]:
        """The attrs recording this config, in netCDF-storable form.

        None is left out (netCDF has no null), booleans become 0/1 (netCDF has no
        bool either, and the existing attrs already use ints for these), and
        nested values become JSON.
        """
        out = {}
        for field, attr in self.ATTRS.items():
            value = getattr(self, field)
            if value is None or field in self.RECORDED_BY_CORE:
                continue
            if field in self.JSON_FIELDS:
                out[attr] = json.dumps(value, sort_keys=True, default=_json_default)
            elif isinstance(value, bool):
                out[attr] = int(value)
            elif isinstance(value, (list, tuple)):
                out[attr] = [str(v) if not isinstance(v, (int, float)) else v for v in value]
            else:
                out[attr] = str(value) if hasattr(value, "__fspath__") else value
        return out

    # --- from the dataset --------------------------------------------------

    @classmethod
    def presence_attr(cls) -> str | None:
        if cls.PRESENCE_ATTR is not None:
            return cls.PRESENCE_ATTR
        return next(iter(cls.ATTRS.values()), None)

    @classmethod
    def from_attrs(cls, attrs: typing.Mapping[str, Any]):
        """The config a dataset records, or None if this step never ran on it."""
        marker = cls.presence_attr()
        if marker is None or marker not in attrs:
            return None
        values = {}
        for field, attr in cls.ATTRS.items():
            if attr not in attrs:
                continue
            raw = _plain(attrs[attr])
            if field in cls.JSON_FIELDS:
                raw = json.loads(raw) if isinstance(raw, str) else raw
            elif _is_list_field(cls, field):
                # netCDF hands a one-element list back as a bare scalar.
                raw = [] if raw is None else list(np.atleast_1d(raw))
                raw = [_plain(v) for v in raw]
            values[field] = raw
        return cls.model_validate(cls._normalise_from_attrs(values))

    @classmethod
    def _normalise_from_attrs(cls, values: dict[str, Any]) -> dict[str, Any]:
        """Hook for a config whose attrs need translating before validation."""
        return values

    # --- to the YAML -------------------------------------------------------

    def overrides(self) -> dict[str, Any]:
        """The fields that differ from the defaults, as plain data (docs rule 1)."""
        return self.model_dump(mode="json", exclude_defaults=True)


def _plain(value):
    """numpy scalars/arrays and bytes, as the Python values pydantic expects."""
    if isinstance(value, bytes):
        return value.decode()
    if isinstance(value, np.ndarray):
        return [_plain(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"{type(value).__name__} is not JSON serialisable")


def _is_list_field(cls, field: str) -> bool:
    annotation = cls.model_fields[field].annotation
    origins = {typing.get_origin(annotation)} | {
        typing.get_origin(a) for a in typing.get_args(annotation)
    }
    return bool(origins & {list, tuple}) or annotation in (list, tuple)
