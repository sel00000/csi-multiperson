"""JSON-safe measurement contracts. Nanoseconds never travel as JSON numbers."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field, fields, is_dataclass
from types import MappingProxyType
from typing import Any, Mapping

METRICS_SCHEMA_VERSION = 1
MISSING_REASONS = frozenset({"disabled", "unsupported", "not_measured", "no_gt", "no_samples"})
REDUCTIONS = frozenset({"sum", "rate", "ratio", "last", "p5", "p50", "p95", "max", "mean"})
SCOPES = frozenset({"window", "cumulative", "instant", "evaluation"})
CLOCK_KINDS = frozenset({"monotonic", "unix_wall", "virtual"})


def json_value(value: Any, key: str = "") -> Any:
    if (key == "ns" or key.endswith("_ns")) and value is not None:
        if isinstance(value, bool) or not isinstance(value, (int, str)):
            raise ValueError(f"{key}: nanoseconds require an integer or decimal string")
        if not re.fullmatch(r"-?\d+", str(value)):
            raise ValueError(f"{key}: nanoseconds require a decimal string")
        return str(value)
    if is_dataclass(value):
        return {f.name: json_value(getattr(value, f.name), f.name) for f in fields(value)}
    if isinstance(value, Mapping):
        return {k: json_value(v, str(k)) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(v) for v in value]
    return value


def parse_ns(value: Any, name: str, *, optional: bool = False) -> int | None:
    if value is None and optional:
        return None
    if not isinstance(value, str) or not re.fullmatch(r"-?\d+", value):
        raise ValueError(f"{name}: JSON nanoseconds must be a decimal string")
    return int(value)


def _count(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name}: expected a nonnegative integer")


@dataclass(frozen=True)
class MetricSpec:
    key: str
    version: int
    unit: str
    group: str
    labels: tuple[str, ...]
    calculation_id: str
    clock_requirement: str
    prerequisites: tuple[str, ...] = ()
    default_window_s: float | None = 10.0
    default_sample_period_s: float | None = None
    default_publish_period_s: float | None = 1.0
    reductions: tuple[str, ...] = ("sum",)
    quantile_method: str | None = None
    min_samples: dict[str, int] = field(default_factory=dict)
    aggregation_scopes: tuple[str, ...] = ("cumulative", "window")

    def __post_init__(self) -> None:
        if not self.key or self.version < 1 or not self.calculation_id:
            raise ValueError("MetricSpec: key, positive version and calculation_id are required")
        if self.group not in {"core", "diagnostic", "offline"}:
            raise ValueError("MetricSpec.group: expected core, diagnostic or offline")
        if not self.reductions or not set(self.reductions) <= REDUCTIONS:
            raise ValueError("MetricSpec.reductions: unknown reduction")
        if not self.aggregation_scopes or not set(self.aggregation_scopes) <= SCOPES:
            raise ValueError("MetricSpec.aggregation_scopes: unknown scope")
        if len(set(self.labels)) != len(self.labels):
            raise ValueError("MetricSpec.labels: duplicate label")
        quantiles = set(self.reductions) & {"p5", "p50", "p95"}
        if quantiles and self.quantile_method != "nearest_rank":
            raise ValueError("MetricSpec.quantile_method: expected nearest_rank")
        for reduction in quantiles:
            if self.min_samples.get(reduction, 0) < 1:
                raise ValueError(f"MetricSpec.min_samples.{reduction}: must be positive")
        object.__setattr__(self, "labels", tuple(self.labels))
        object.__setattr__(self, "prerequisites", tuple(self.prerequisites))
        object.__setattr__(self, "reductions", tuple(self.reductions))
        object.__setattr__(self, "aggregation_scopes", tuple(self.aggregation_scopes))
        object.__setattr__(self, "min_samples", MappingProxyType(dict(self.min_samples)))

    def to_dict(self) -> dict:
        return json_value(self)


@dataclass(frozen=True)
class MetricValue:
    key: str
    labels: dict[str, str]
    reduction: str
    aggregation_scope: str
    value: float | int | None
    availability_reason: str | None = None
    detail: str | None = None
    begin_ns: int | None = None
    end_ns: int | None = None
    sample_count: int = 0
    excluded_count: int = 0
    excluded_by_reason: dict[str, int] = field(default_factory=dict)
    exposure_ns: int | None = None
    counter_epoch: str | None = None
    numerator: float | int | None = None
    denominator: float | int | None = None
    provenance_id: str = ""

    def __post_init__(self) -> None:
        if self.reduction not in REDUCTIONS or self.aggregation_scope not in SCOPES:
            raise ValueError("MetricValue: unknown reduction or aggregation_scope")
        if self.value is None:
            if self.availability_reason not in MISSING_REASONS:
                raise ValueError("MetricValue.availability_reason: missing value requires a reason")
        elif self.availability_reason is not None:
            raise ValueError(
                "MetricValue.availability_reason: measured value cannot have a missing reason"
            )
        elif (
            isinstance(self.value, bool)
            or not isinstance(self.value, (int, float))
            or not math.isfinite(self.value)
        ):
            raise ValueError("MetricValue.value: expected a finite number or null")
        _count(self.sample_count, "sample_count")
        _count(self.excluded_count, "excluded_count")
        for name in ("begin_ns", "end_ns", "exposure_ns"):
            if getattr(self, name) is not None:
                _count(getattr(self, name), name)
        for reason, count in self.excluded_by_reason.items():
            _count(count, f"excluded_by_reason.{reason}")
        if sum(self.excluded_by_reason.values()) != self.excluded_count:
            raise ValueError("excluded_count must equal sum(excluded_by_reason)")
        if self.begin_ns is not None and self.end_ns is not None and self.begin_ns > self.end_ns:
            raise ValueError("MetricValue: begin_ns must not exceed end_ns")
        if self.exposure_ns is not None:
            _count(self.exposure_ns, "exposure_ns")
        if (
            self.aggregation_scope == "cumulative"
            and self.value is not None
            and not self.counter_epoch
        ):
            raise ValueError("MetricValue.counter_epoch: cumulative measurement requires an epoch")
        if self.reduction in {"ratio", "rate"} and self.value is not None:
            if self.denominator is None or self.denominator <= 0 or self.numerator is None:
                raise ValueError(
                    "MetricValue: measured rate/ratio requires numerator and positive denominator"
                )
        for name in ("numerator", "denominator"):
            number = getattr(self, name)
            if number is not None and (
                isinstance(number, bool)
                or not isinstance(number, (int, float))
                or not math.isfinite(number)
            ):
                raise ValueError(f"MetricValue.{name}: expected finite number or null")
        if any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in self.labels.items()
        ):
            raise ValueError("MetricValue.labels: keys and values must be strings")
        object.__setattr__(self, "labels", MappingProxyType(dict(self.labels)))
        object.__setattr__(
            self, "excluded_by_reason", MappingProxyType(dict(self.excluded_by_reason))
        )

    @property
    def series_id(self) -> tuple:
        return (
            self.key,
            tuple(sorted(self.labels.items())),
            self.reduction,
            self.aggregation_scope,
        )

    def to_dict(self) -> dict:
        return json_value(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> MetricValue:
        value = dict(data)
        for key in ("begin_ns", "end_ns", "exposure_ns"):
            value[key] = parse_ns(value.get(key), key, optional=True)
        return cls(**value)


@dataclass(frozen=True)
class MetricsSnapshot:
    schema_version: int
    run_id: str
    session_id: str
    stream_epoch: str
    event_sequence: int
    profile_id: str
    selection_id: str
    spec_hash: str
    selection_hash: str
    as_of_ns: int
    clock_domain: str
    clock_kind: str
    time_basis: str
    values: tuple[MetricValue, ...]
    provider_hash: str = ""
    provenance: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.schema_version != METRICS_SCHEMA_VERSION:
            raise ValueError(f"MetricsSnapshot.schema_version: expected {METRICS_SCHEMA_VERSION}")
        if self.clock_kind not in CLOCK_KINDS or not self.clock_domain:
            raise ValueError("MetricsSnapshot: explicit clock domain and kind required")
        if self.time_basis not in {"live", "replay_virtual", "replay_source", "process_wall"}:
            raise ValueError("MetricsSnapshot.time_basis: unsupported basis")
        _count(self.event_sequence, "event_sequence")
        _count(self.as_of_ns, "as_of_ns")
        if len({v.series_id for v in self.values}) != len(self.values):
            raise ValueError("MetricsSnapshot.values: duplicate series identity")
        object.__setattr__(self, "values", tuple(self.values))

    def to_dict(self) -> dict:
        return json_value(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, allow_nan=False, separators=(",", ":"))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> MetricsSnapshot:
        value = dict(data)
        value["as_of_ns"] = parse_ns(value["as_of_ns"], "as_of_ns")
        value["values"] = tuple(MetricValue.from_dict(v) for v in value["values"])
        return cls(**value)

    @classmethod
    def from_json(cls, data: str) -> MetricsSnapshot:
        return cls.from_dict(json.loads(data))
