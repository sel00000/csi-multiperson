"""Browser-facing pose and diagnostic contracts, independent of models and UI."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Mapping, Protocol

from csi_mp.data.contracts import validate_coordinate_spec
from csi_mp.data.schema import CoordinateSpec
from csi_mp.telemetry.schema import CLOCK_KINDS, MetricsSnapshot, json_value, parse_ns

DASHBOARD_SCHEMA_ID = "csi_mp.dashboard.v1"


@dataclass(frozen=True)
class DisplayPerson:
    candidate_key: str
    track_id: str | None
    estimate_kind: str
    joints: tuple
    joint_valid: tuple
    presence_score: float | None
    evidence_age_ms: float | None

    def __post_init__(self) -> None:
        if not isinstance(self.candidate_key, str) or not self.candidate_key:
            raise ValueError(
                "candidate_key: required; a query index is a candidate, not a track ID"
            )
        if self.track_id is not None and (not isinstance(self.track_id, str) or not self.track_id):
            raise ValueError("track_id: expected explicit nonempty tracking identifier or null")
        if self.estimate_kind not in {"observed", "predicted"}:
            raise ValueError("estimate_kind: expected observed or predicted")
        if len(self.joints) != 18 or len(self.joint_valid) != 18:
            raise ValueError("joints/joint_valid: BODY18 requires 18 joints")
        dimensions = {len(row) for row in self.joints}
        if dimensions not in ({2}, {3}):
            raise ValueError("joints: expected uniform [18,2] or [18,3]")
        canonical = []
        for index, (row, valid) in enumerate(zip(self.joints, self.joint_valid)):
            if not isinstance(valid, bool):
                raise ValueError(f"joint_valid[{index}]: expected bool")
            if valid:
                if any(
                    isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
                    for v in row
                ):
                    raise ValueError(f"joints[{index}]: valid joint requires finite coordinates")
                canonical.append(tuple(row))
            else:
                canonical.append(tuple(None for _ in row))
        object.__setattr__(self, "joints", tuple(canonical))
        object.__setattr__(self, "joint_valid", tuple(self.joint_valid))
        if self.presence_score is not None and (
            isinstance(self.presence_score, bool)
            or not math.isfinite(self.presence_score)
            or not 0 <= self.presence_score <= 1
        ):
            raise ValueError("presence_score: expected model score in [0,1] or null")
        if self.evidence_age_ms is not None and (
            not math.isfinite(self.evidence_age_ms) or self.evidence_age_ms < 0
        ):
            raise ValueError("evidence_age_ms: expected nonnegative finite age or null")
        if self.estimate_kind == "predicted" and self.evidence_age_ms is None:
            raise ValueError("predicted person requires evidence_age_ms")

    def to_dict(self) -> dict:
        return json_value(self)

    @classmethod
    def from_dict(cls, data: Mapping) -> DisplayPerson:
        value = dict(data)
        value["joints"] = tuple(tuple(row) for row in value["joints"])
        value["joint_valid"] = tuple(value["joint_valid"])
        return cls(**value)


@dataclass(frozen=True)
class DashboardFrame:
    schema_id: str
    run_id: str
    session_id: str
    stream_epoch: str
    event_sequence: int
    mode: str
    status: str
    coordinate_spec: dict
    source_time_ns: str
    source_clock_domain: str
    source_clock_kind: str
    emitted_monotonic_ns: str
    emitter_clock_domain: str
    availability_basis: str
    current_count: int | None
    predicted_count: int | None
    persons: tuple[DisplayPerson, ...]
    quality: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.schema_id != DASHBOARD_SCHEMA_ID:
            raise ValueError(f"DashboardFrame.schema_id: expected {DASHBOARD_SCHEMA_ID}")
        if self.mode not in {"ingest", "live", "replay", "demo"}:
            raise ValueError("DashboardFrame.mode: unknown mode")
        if self.status not in {
            "unknown",
            "warming_up",
            "observed",
            "predicted",
            "stale",
            "disconnected",
        }:
            raise ValueError("DashboardFrame.status: unknown status")
        if (
            self.source_clock_kind not in CLOCK_KINDS
            or not self.source_clock_domain
            or not self.emitter_clock_domain
        ):
            raise ValueError(
                "DashboardFrame: explicit source/emitter clock domains and source kind required"
            )
        parse_ns(self.source_time_ns, "source_time_ns")
        parse_ns(self.emitted_monotonic_ns, "emitted_monotonic_ns")
        if self.availability_basis not in {"measured_infer", "legacy_bridge_proxy", "synthetic"}:
            raise ValueError("DashboardFrame.availability_basis: unsupported basis")
        if (
            isinstance(self.event_sequence, bool)
            or not isinstance(self.event_sequence, int)
            or self.event_sequence < 0
        ):
            raise ValueError("DashboardFrame.event_sequence: expected nonnegative integer")
        for name in ("current_count", "predicted_count"):
            count = getattr(self, name)
            if count is not None and (
                isinstance(count, bool) or not isinstance(count, int) or count < 0
            ):
                raise ValueError(f"DashboardFrame.{name}: expected nonnegative integer or null")
        if len({p.candidate_key for p in self.persons}) != len(self.persons):
            raise ValueError("DashboardFrame.persons: duplicate candidate_key")
        observed = sum(p.estimate_kind == "observed" for p in self.persons)
        predicted = sum(p.estimate_kind == "predicted" for p in self.persons)
        if self.status in {"unknown", "warming_up"}:
            if self.current_count is not None or self.predicted_count is not None or self.persons:
                raise ValueError(
                    "unknown/warming_up must have null counts and no persons; empty room is observed/0"
                )
        elif self.status == "observed":
            if self.current_count != observed or self.predicted_count != predicted:
                raise ValueError("observed counts must match current/predicted person subsets")
        elif self.status == "predicted":
            if self.current_count is not None or observed or self.predicted_count != predicted:
                raise ValueError(
                    "predicted output must have null current_count and only predicted persons"
                )
        elif self.current_count is not None:
            raise ValueError("stale/disconnected output cannot claim a current person count")
        coordinates = CoordinateSpec.from_dict(self.coordinate_spec)
        validate_coordinate_spec(coordinates)
        dimension = coordinates.dimension
        object.__setattr__(self, "coordinate_spec", coordinates.to_dict())
        for person in self.persons:
            if len(person.joints[0]) != dimension:
                raise ValueError("person joints dimension disagrees with coordinate_spec")
            if person.track_id is not None and (
                not self.quality.get("tracking_enabled") or not self.quality.get("track_epoch")
            ):
                raise ValueError(
                    "track_id requires tracking_enabled and track_epoch; candidates have no track ID"
                )
        if self.mode == "ingest" and self.status != "unknown":
            raise ValueError("ingest mode has no model output; status must be unknown")
        object.__setattr__(self, "persons", tuple(self.persons))

    def to_dict(self) -> dict:
        return json_value(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, allow_nan=False, separators=(",", ":"))

    @classmethod
    def from_dict(cls, data: Mapping) -> DashboardFrame:
        value = dict(data)
        value["persons"] = tuple(DisplayPerson.from_dict(p) for p in value["persons"])
        return cls(**value)

    @classmethod
    def from_json(cls, data: str) -> DashboardFrame:
        return cls.from_dict(json.loads(data))


@dataclass(frozen=True)
class DiagnosticEvent:
    schema_id: str
    event_id: str
    run_id: str
    session_id: str
    stream_epoch: str
    event_sequence: int
    kind: str
    source_id: str | None
    available_ns: str
    clock_domain: str
    clock_kind: str
    affected_links: tuple[int, ...]
    detail: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.schema_id != "csi_mp.diagnostic.v1":
            raise ValueError("DiagnosticEvent.schema_id: expected csi_mp.diagnostic.v1")
        parse_ns(self.available_ns, "available_ns")
        if self.clock_kind not in CLOCK_KINDS or not self.clock_domain:
            raise ValueError("DiagnosticEvent: explicit clock required")
        if any(link not in range(9) for link in self.affected_links):
            raise ValueError("DiagnosticEvent.affected_links: expected indices 0..8")

    def to_dict(self) -> dict:
        return json_value(self)


class DashboardPublisher(Protocol):
    def publish_pose(self, frame: DashboardFrame) -> None: ...
    def publish_metrics(self, snapshot: MetricsSnapshot) -> None: ...
    def publish_event(self, event: DiagnosticEvent) -> None: ...
    def close(self) -> None: ...


def to_dashboard_frame(output: Mapping | None, context: Mapping) -> DashboardFrame:
    """Merge explicit pose state with clock/run context; None is ingest unknown.

    This function accepts already contracted display values. Model decoding,
    tracking and transport belong to S19/S22/V01 and do not run here.
    """
    data = dict(context)
    data.setdefault("schema_id", DASHBOARD_SCHEMA_ID)
    if output is None:
        data.update(status="unknown", current_count=None, predicted_count=None, persons=())
    else:
        data.update(output)
    if hasattr(data.get("coordinate_spec"), "to_dict"):
        data["coordinate_spec"] = data["coordinate_spec"].to_dict()
    data["persons"] = tuple(
        p if isinstance(p, DisplayPerson) else DisplayPerson.from_dict(p) for p in data["persons"]
    )
    return DashboardFrame(**data)
