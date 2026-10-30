"""Versioned, immutable values shared by the ingest and model boundaries.

Array storage is copied into a bytes-backed view, so callers cannot mutate a
published value through the original array or by re-enabling its write flag.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal

import numpy as np
from numpy.typing import NDArray


RAW_SCHEMA_ID = "CSIRAW01"
WIRE_SCHEMA_ID = "csi130.v1"
DERIVED_SCHEMA_ID = "csi_mp.window.v1"
LABEL_SCHEMA_ID = "csi_mp.labels.v1"
MODEL_SCHEMA_ID = "csi_mp.model.v1"
FEATURE_SPEC_ID = "logamp_delta.v1"
JOINT_SPEC_ID = "BODY18"

ClockKind = Literal["monotonic", "unix_wall", "virtual"]
AvailabilityBasis = Literal["measured_infer", "legacy_bridge_proxy", "synthetic"]


class ContractError(ValueError):
    """A named field violates a versioned data contract."""


def _integer(value: int, name: str, lower: int = 0, upper: int | None = None) -> None:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ContractError(f"{name}: expected integer, got {type(value).__name__}")
    if value < lower or (upper is not None and value > upper):
        raise ContractError(
            f"{name}: expected {lower} <= value" + (f" <= {upper}" if upper is not None else "")
        )


def immutable_array(value: NDArray, *, dtype: np.dtype | type, name: str) -> NDArray:
    """Copy an array without silently converting its dtype or sign convention."""
    array = np.asarray(value)
    if array.dtype != np.dtype(dtype):
        raise ContractError(f"{name}.dtype: expected {np.dtype(dtype)}, got {array.dtype}")
    return np.frombuffer(array.tobytes(order="C"), dtype=array.dtype).reshape(array.shape)


@dataclass(frozen=True)
class Stamp:
    ns: int
    domain: str
    kind: ClockKind

    def __post_init__(self) -> None:
        _integer(self.ns, "Stamp.ns")
        if not isinstance(self.domain, str) or not self.domain:
            raise ContractError("Stamp.domain: a nonempty clock domain is required")
        if self.kind not in {"monotonic", "unix_wall", "virtual"}:
            raise ContractError(f"Stamp.kind: unsupported clock kind {self.kind!r}")
        object.__setattr__(self, "ns", int(self.ns))

    def to_dict(self) -> dict[str, str]:
        return {"ns": str(self.ns), "domain": self.domain, "kind": self.kind}

    @classmethod
    def from_dict(cls, value: dict) -> Stamp:
        ns = value["ns"]
        if not isinstance(ns, str) or not ns.isdecimal():
            raise ContractError("Stamp.ns: JSON nanoseconds must be a decimal string")
        return cls(int(ns), value["domain"], value["kind"])


@dataclass(frozen=True)
class SourceSpan:
    source_sha256: str
    source_id: str
    chunk_index: int
    file_offset: int
    byte_count: int

    def __post_init__(self) -> None:
        for name in ("chunk_index", "file_offset", "byte_count"):
            _integer(getattr(self, name), f"SourceSpan.{name}")
        if not self.source_id or not self.source_sha256:
            raise ContractError("SourceSpan: source_id and source_sha256 are required")


def _availability(bridge: Stamp, infer: Stamp | None) -> AvailabilityBasis:
    if bridge.kind == "virtual" or (infer is not None and infer.kind == "virtual"):
        return "synthetic"
    return "measured_infer" if infer is not None else "legacy_bridge_proxy"


def _validate_infer_stamp(stamp: Stamp | None) -> None:
    if stamp is not None and stamp.kind not in {"monotonic", "virtual"}:
        raise ContractError("infer_available.kind: expected monotonic or explicitly virtual")


@dataclass(frozen=True)
class RawChunk:
    span: SourceSpan
    payload: bytes
    bridge_read: Stamp
    infer_available: Stamp | None
    ingest_ordinal: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "payload", bytes(self.payload))
        if len(self.payload) != self.span.byte_count:
            raise ContractError("RawChunk.span.byte_count: must equal payload byte length")
        _integer(self.ingest_ordinal, "RawChunk.ingest_ordinal")
        _validate_infer_stamp(self.infer_available)

    @property
    def availability_basis(self) -> AvailabilityBasis:
        return _availability(self.bridge_read, self.infer_available)


@dataclass(frozen=True)
class WireFrame:
    spans: tuple[SourceSpan, ...]
    wire_bytes: bytes
    rx_id: int
    tx_id: int
    seq_u32: int
    rx_timer_u32_us: int
    rx_boot_u8: int
    iq: NDArray[np.int8]
    rssi_dbm: int
    noise_dbm: int
    signal_mode: int | None
    bridge_complete: Stamp
    infer_available: Stamp | None
    ingest_ordinal: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "spans", tuple(self.spans))
        object.__setattr__(self, "wire_bytes", bytes(self.wire_bytes))
        for name in ("rx_id", "tx_id"):
            _integer(getattr(self, name), f"WireFrame.{name}", upper=2)
        for name in ("seq_u32", "rx_timer_u32_us"):
            _integer(getattr(self, name), f"WireFrame.{name}", upper=2**32 - 1)
        _integer(self.rx_boot_u8, "WireFrame.rx_boot_u8", upper=255)
        _integer(self.ingest_ordinal, "WireFrame.ingest_ordinal")
        _validate_infer_stamp(self.infer_available)
        for name in ("rssi_dbm", "noise_dbm"):
            _integer(getattr(self, name), f"WireFrame.{name}", lower=-128, upper=127)
        iq = immutable_array(self.iq, dtype=np.int8, name="WireFrame.iq")
        if iq.shape != (56, 2):
            raise ContractError(f"WireFrame.iq.shape: expected (56, 2), got {iq.shape}")
        object.__setattr__(self, "iq", iq)

    @property
    def availability_basis(self) -> AvailabilityBasis:
        return _availability(self.bridge_complete, self.infer_available)

    @property
    def link_index(self) -> int:
        return 3 * self.rx_id + self.tx_id


@dataclass(frozen=True)
class ParseIssue:
    kind: str
    layer: Literal["raw_record", "wire_frame"]
    spans: tuple[SourceSpan, ...]
    byte_count: int
    detail: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "spans", tuple(self.spans))
        _integer(self.byte_count, "ParseIssue.byte_count")
        if self.layer not in {"raw_record", "wire_frame"}:
            raise ContractError(f"ParseIssue.layer: unsupported layer {self.layer!r}")


@dataclass(frozen=True)
class EpochEvidence:
    kind: Literal["rx_boot", "tx_start", "tx_reboot", "operator_boundary"]
    device_id: int
    available: Stamp
    source_ref: str

    def __post_init__(self) -> None:
        _integer(self.device_id, "EpochEvidence.device_id", upper=2)
        if self.kind not in {"rx_boot", "tx_start", "tx_reboot", "operator_boundary"}:
            raise ContractError(f"EpochEvidence.kind: unsupported kind {self.kind!r}")
        if not self.source_ref:
            raise ContractError("EpochEvidence.source_ref: evidence source is required")


@dataclass(frozen=True)
class EpochEvent:
    event_id: str
    scope: Literal["rx", "tx"]
    device_id: int
    kind: Literal["rx_reset", "tx_reset", "timer_wrap", "seq_wrap", "timer_regression"]
    certainty: Literal["confirmed", "inferred", "uncertain"]
    available: Stamp
    affected_links: tuple[int, ...]
    evidence_refs: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "affected_links", tuple(self.affected_links))
        object.__setattr__(self, "evidence_refs", tuple(self.evidence_refs))
        _integer(self.device_id, "EpochEvent.device_id", upper=2)
        if self.scope not in {"rx", "tx"}:
            raise ContractError("EpochEvent.scope: expected rx or tx")
        if self.kind not in {"rx_reset", "tx_reset", "timer_wrap", "seq_wrap", "timer_regression"}:
            raise ContractError(f"EpochEvent.kind: unsupported kind {self.kind!r}")
        if self.certainty not in {"confirmed", "inferred", "uncertain"}:
            raise ContractError("EpochEvent.certainty: expected confirmed, inferred or uncertain")
        for link in self.affected_links:
            _integer(link, "EpochEvent.affected_links", upper=8)


@dataclass(frozen=True)
class EpochResult:
    rx_epoch: str
    tx_epoch: str | None
    link_segment: str
    timer_unwrapped_us: int | None
    classification: Literal[
        "normal",
        "duplicate",
        "reorder",
        "rx_reset",
        "tx_reset",
        "timer_wrap",
        "seq_wrap",
        "ambiguous",
    ]
    usable: bool
    evidence_refs: tuple[str, ...]
    events: tuple[EpochEvent, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "evidence_refs", tuple(self.evidence_refs))
        object.__setattr__(self, "events", tuple(self.events))
        if self.classification not in {
            "normal",
            "duplicate",
            "reorder",
            "rx_reset",
            "tx_reset",
            "timer_wrap",
            "seq_wrap",
            "ambiguous",
        }:
            raise ContractError(
                f"EpochResult.classification: unsupported value {self.classification!r}"
            )
        if self.timer_unwrapped_us is not None:
            _integer(self.timer_unwrapped_us, "EpochResult.timer_unwrapped_us")


@dataclass(frozen=True)
class CoordinateSpec:
    dimension: int = 2
    spec_id: str = "camera_normalized_v1"
    unit: str = "normalized"
    axes: tuple[str, ...] = ("x_right", "y_down")
    normalization: str = "x/W,y/H"
    image_width: int | None = None
    image_height: int | None = None
    coordinate_system: str = "camera"
    calibration_id: str | None = None
    gt_source_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "axes", tuple(self.axes))

    def to_dict(self) -> dict:
        result = asdict(self)
        result["axes"] = list(self.axes)
        return result

    @classmethod
    def from_dict(cls, value: dict) -> CoordinateSpec:
        return cls(**value)


@dataclass(frozen=True)
class DataSpec:
    D: int = 2
    T: int = 100
    L: int = 9
    F: int = 56
    C: int = 2
    K: int = 18
    Q: int = 6
    patch_size: int = 5
    grid_step_ns: int = 10_000_000
    joint_spec_id: str = JOINT_SPEC_ID
    joint_names: tuple[str, ...] = ()
    coordinate_spec: CoordinateSpec = field(default_factory=CoordinateSpec)
    raw_schema_id: str = RAW_SCHEMA_ID
    wire_schema_id: str = WIRE_SCHEMA_ID
    derived_schema_id: str = DERIVED_SCHEMA_ID
    label_schema_id: str = LABEL_SCHEMA_ID
    model_schema_id: str = MODEL_SCHEMA_ID
    feature_id: str = FEATURE_SPEC_ID

    def __post_init__(self) -> None:
        if not self.joint_names:
            from .joints import BODY18_NAMES

            object.__setattr__(self, "joint_names", BODY18_NAMES)
        else:
            object.__setattr__(self, "joint_names", tuple(self.joint_names))


@dataclass(frozen=True)
class WindowPacket:
    session_id: str
    target_ns: int
    decision_ns: int
    grid_ns: NDArray[np.int64]
    raw_x: NDArray[np.float32]
    feature_valid: NDArray[np.bool_]
    observed_mask: NDArray[np.bool_]
    link_state: NDArray[np.uint8]
    age_ns: NDArray[np.int64]
    schema_id: str
    feature_id: str
    provenance_id: str
    availability_basis: str
    status: Literal["ready", "warming_up", "insufficient_coverage"]

    def __post_init__(self) -> None:
        for name, dtype in (
            ("grid_ns", np.int64),
            ("raw_x", np.float32),
            ("feature_valid", np.bool_),
            ("observed_mask", np.bool_),
            ("link_state", np.uint8),
            ("age_ns", np.int64),
        ):
            object.__setattr__(
                self,
                name,
                immutable_array(getattr(self, name), dtype=dtype, name=f"WindowPacket.{name}"),
            )


@dataclass(frozen=True)
class PoseFrame:
    """Ragged labels for one frame; person rows are never truncated to Q."""

    frame_idx: int
    time: Stamp
    camera_id: str
    joints: NDArray[np.float32]
    joint_valid: NDArray[np.bool_]
    joint_weight: NDArray[np.float32]
    gt_track_id: NDArray[np.int64]
    annotation_complete: bool
    scene_status: Literal["empty_confirmed", "persons_labeled", "incomplete"]
    coordinate_spec: CoordinateSpec
    joint_spec_id: str = JOINT_SPEC_ID
    joint_names: tuple[str, ...] = ()
    schema_id: str = LABEL_SCHEMA_ID

    def __post_init__(self) -> None:
        for name, dtype in (
            ("joints", np.float32),
            ("joint_valid", np.bool_),
            ("joint_weight", np.float32),
            ("gt_track_id", np.int64),
        ):
            object.__setattr__(
                self,
                name,
                immutable_array(getattr(self, name), dtype=dtype, name=f"PoseFrame.{name}"),
            )
        if not self.joint_names:
            from .joints import BODY18_NAMES

            object.__setattr__(self, "joint_names", BODY18_NAMES)
        else:
            object.__setattr__(self, "joint_names", tuple(self.joint_names))


@dataclass(frozen=True)
class LabelValidation:
    person_count: int
    capacity_status: Literal["supported", "capacity_overflow"]
    capacity: int | None
    annotation_complete: bool


class CapacityOverflow(ValueError):
    """Valid labels exceed a model's query capacity; this is not corrupt data."""

    def __init__(self, person_count: int, capacity: int):
        self.person_count = person_count
        self.capacity = capacity
        super().__init__(
            f"CapacityOverflow: N={person_count} exceeds Q={capacity}; labels remain intact"
        )
