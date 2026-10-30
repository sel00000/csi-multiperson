"""Validation at version, shape, coordinate and clock boundaries."""

from __future__ import annotations

import numpy as np

from .joints import validate_joint_order
from .schema import (
    DERIVED_SCHEMA_ID,
    FEATURE_SPEC_ID,
    LABEL_SCHEMA_ID,
    MODEL_SCHEMA_ID,
    RAW_SCHEMA_ID,
    WIRE_SCHEMA_ID,
    CapacityOverflow,
    ContractError,
    CoordinateSpec,
    DataSpec,
    LabelValidation,
    PoseFrame,
    Stamp,
    WindowPacket,
)


def validate_coordinate_spec(spec: CoordinateSpec, *, for_labels: bool = False) -> None:
    if spec.dimension not in (2, 3) or isinstance(spec.dimension, bool):
        raise ContractError("coordinate_spec.dimension: expected 2 or 3")
    if not spec.spec_id:
        raise ContractError("coordinate_spec.spec_id: required")
    if (
        len(spec.axes) != spec.dimension
        or len(set(spec.axes)) != spec.dimension
        or not all(spec.axes)
    ):
        raise ContractError(
            "coordinate_spec.axes: one distinct named axis per dimension is required"
        )
    if spec.dimension == 2:
        if spec.unit != "normalized" or spec.normalization != "x/W,y/H":
            raise ContractError(
                "coordinate_spec.normalization: D2 uses unit=normalized and x/W,y/H"
            )
        if spec.axes != ("x_right", "y_down") or spec.coordinate_system != "camera":
            raise ContractError("coordinate_spec.axes: D2 requires camera x_right,y_down")
        known = spec.image_width is not None or spec.image_height is not None
        if known or for_labels:
            for field in ("image_width", "image_height"):
                value = getattr(spec, field)
                if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                    raise ContractError(f"coordinate_spec.{field}: positive image size required")
    else:
        if spec.unit != "m":
            raise ContractError("coordinate_spec.unit: D3 uses meters (m)")
        if spec.normalization != "none":
            raise ContractError("coordinate_spec.normalization: D3 requires none")
        if spec.coordinate_system not in {"world", "camera"}:
            raise ContractError("coordinate_spec.coordinate_system: D3 requires world or camera")
        if not spec.calibration_id:
            raise ContractError("coordinate_spec.calibration_id: required for D3")
        if not spec.gt_source_id:
            raise ContractError(
                "coordinate_spec.gt_source_id: D3 ground-truth device/source is required"
            )


def validate_data_spec(spec: DataSpec) -> None:
    for name in ("D", "T", "L", "F", "C", "K", "Q", "patch_size", "grid_step_ns"):
        value = getattr(spec, name)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ContractError(f"{name}: expected positive integer")
    if spec.D not in (2, 3):
        raise ContractError("D: expected 2 or 3")
    for name, expected in (("L", 9), ("F", 56), ("C", 2), ("K", 18)):
        if getattr(spec, name) != expected:
            raise ContractError(f"{name}: v1 expects {expected}")
    if spec.T % spec.patch_size:
        raise ContractError("patch_size: must divide T exactly")
    for name, expected in (
        ("raw_schema_id", RAW_SCHEMA_ID),
        ("wire_schema_id", WIRE_SCHEMA_ID),
        ("derived_schema_id", DERIVED_SCHEMA_ID),
        ("label_schema_id", LABEL_SCHEMA_ID),
        ("model_schema_id", MODEL_SCHEMA_ID),
        ("feature_id", FEATURE_SPEC_ID),
    ):
        if getattr(spec, name) != expected:
            raise ContractError(f"{name}: unsupported schema/version; expected {expected}")
    validate_joint_order(spec.joint_names, spec_id=spec.joint_spec_id)
    validate_coordinate_spec(spec.coordinate_spec)
    if spec.D != spec.coordinate_spec.dimension:
        raise ContractError("D: does not match coordinate_spec.dimension")


def validate_compatibility(model_spec: DataSpec, data_spec: DataSpec) -> None:
    validate_data_spec(model_spec)
    validate_data_spec(data_spec)
    for name in (
        "D",
        "T",
        "L",
        "F",
        "C",
        "K",
        "patch_size",
        "grid_step_ns",
        "joint_spec_id",
        "joint_names",
        "raw_schema_id",
        "wire_schema_id",
        "derived_schema_id",
        "label_schema_id",
        "model_schema_id",
        "feature_id",
    ):
        if getattr(model_spec, name) != getattr(data_spec, name):
            raise ContractError(
                f"{name}: incompatible model/data contracts ({getattr(model_spec, name)!r} != {getattr(data_spec, name)!r})"
            )
    if model_spec.coordinate_spec != data_spec.coordinate_spec:
        raise ContractError("coordinate_spec: incompatible model/data coordinate contracts")
    # Q is model capacity, never a storage limit or a reason to truncate labels.


def elapsed_ns(end: Stamp, start: Stamp) -> int:
    """Subtract only comparable monotonic/virtual stamps; no implicit mapping."""
    if end.domain != start.domain or end.kind != start.kind:
        raise ContractError("clock_domain: timestamps require an explicit clock transform")
    if end.kind == "unix_wall":
        raise ContractError(
            "clock_kind: wall-clock elapsed time needs a discontinuity-aware mapping"
        )
    if end.ns < start.ns:
        raise ContractError("clock: negative elapsed time")
    return end.ns - start.ns


def validate_window(window: WindowPacket, spec: DataSpec | None = None) -> None:
    spec = spec or DataSpec()
    validate_data_spec(spec)
    for name in ("target_ns", "decision_ns"):
        value = getattr(window, name)
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 0:
            raise ContractError(f"WindowPacket.{name}: expected nonnegative integer nanoseconds")
    if window.schema_id != spec.derived_schema_id or window.schema_id != DERIVED_SCHEMA_ID:
        raise ContractError(f"WindowPacket.schema_id: expected {spec.derived_schema_id}")
    if window.feature_id != spec.feature_id:
        raise ContractError(f"WindowPacket.feature_id: expected {spec.feature_id}")
    full_shape = (spec.T, spec.L, spec.F, spec.C)
    link_shape = (spec.T, spec.L)
    for name, expected in (
        ("grid_ns", (spec.T,)),
        ("raw_x", full_shape),
        ("feature_valid", full_shape),
        ("observed_mask", link_shape),
        ("link_state", link_shape),
        ("age_ns", link_shape),
    ):
        value = getattr(window, name)
        if value.shape != expected:
            raise ContractError(
                f"WindowPacket.{name}.shape: expected {expected}, got {value.shape}"
            )
    expected_grid = (
        window.target_ns - np.arange(spec.T - 1, -1, -1, dtype=np.int64) * spec.grid_step_ns
    )
    if not np.array_equal(window.grid_ns, expected_grid):
        raise ContractError("WindowPacket.grid_ns: expected target-(T-1)*step ... target")
    if window.decision_ns < window.target_ns:
        raise ContractError("WindowPacket.decision_ns: cannot precede target_ns")
    if window.status not in {"ready", "warming_up", "insufficient_coverage"}:
        raise ContractError("WindowPacket.status: unsupported status")
    if window.availability_basis not in {"measured_infer", "legacy_bridge_proxy", "synthetic"}:
        raise ContractError("WindowPacket.availability_basis: unsupported basis")
    if not window.session_id or not window.provenance_id:
        raise ContractError("WindowPacket: session_id and provenance_id are required")
    if not np.isfinite(window.raw_x).all():
        raise ContractError("WindowPacket.raw_x: values must be finite; invalid positions use zero")
    if np.any(window.raw_x[~window.feature_valid] != 0):
        raise ContractError("WindowPacket.raw_x: invalid features must be zero")
    if np.any(window.link_state > 2):
        raise ContractError("WindowPacket.link_state: expected 0=invalid, 1=observed, 2=held")
    if not np.array_equal(window.observed_mask, window.link_state == 1):
        raise ContractError("WindowPacket.observed_mask: must match link_state=observed")
    invalid = window.link_state == 0
    if np.any(window.feature_valid[invalid]) or np.any(window.age_ns[invalid] != -1):
        raise ContractError("WindowPacket: invalid links require feature_valid=False and age_ns=-1")
    if np.any(window.age_ns[~invalid] < 0):
        raise ContractError("WindowPacket.age_ns: valid/held links require a nonnegative age")
    if np.any(window.feature_valid[..., 1][window.link_state == 2]):
        raise ContractError("WindowPacket.feature_valid: held delta features must be invalid")
    if np.any(window.feature_valid[..., 1] & ~window.feature_valid[..., 0]):
        raise ContractError("WindowPacket.feature_valid: valid delta requires valid amplitude")
    if window.status == "ready" and not window.feature_valid.any():
        raise ContractError("WindowPacket.status: no usable observations cannot be ready")


def validate_labels(frame: PoseFrame, *, capacity: int | None = None) -> LabelValidation:
    if (
        isinstance(frame.frame_idx, bool)
        or not isinstance(frame.frame_idx, int)
        or frame.frame_idx < 0
    ):
        raise ContractError("PoseFrame.frame_idx: expected nonnegative integer")
    if not isinstance(frame.annotation_complete, bool):
        raise ContractError("PoseFrame.annotation_complete: expected boolean")
    validate_coordinate_spec(frame.coordinate_spec, for_labels=True)
    validate_joint_order(frame.joint_names, spec_id=frame.joint_spec_id)
    if frame.schema_id != LABEL_SCHEMA_ID:
        raise ContractError(f"PoseFrame.schema_id: expected {LABEL_SCHEMA_ID}")
    if frame.joints.ndim != 3 or frame.joints.shape[1:] != (18, frame.coordinate_spec.dimension):
        raise ContractError("PoseFrame.joints.shape: expected [N,18,D]")
    n = frame.joints.shape[0]
    for name, expected in (
        ("joint_valid", (n, 18)),
        ("joint_weight", (n, 18)),
        ("gt_track_id", (n,)),
    ):
        if getattr(frame, name).shape != expected:
            raise ContractError(f"PoseFrame.{name}.shape: expected {expected}")
    if not np.isfinite(frame.joints[frame.joint_valid]).all():
        raise ContractError("PoseFrame.joints: valid joints must have finite coordinates")
    if np.isinf(frame.joints).any():
        raise ContractError("PoseFrame.joints: missing joints may be NaN but never infinity")
    if not np.isfinite(frame.joint_weight).all() or np.any(frame.joint_weight < 0):
        raise ContractError("PoseFrame.joint_weight: expected finite nonnegative weights")
    if np.any(frame.gt_track_id < -1):
        raise ContractError("PoseFrame.gt_track_id: unknown=-1; otherwise IDs are nonnegative")
    known_ids = frame.gt_track_id[frame.gt_track_id >= 0]
    if len(np.unique(known_ids)) != len(known_ids):
        raise ContractError("PoseFrame.gt_track_id: known IDs must be unique within a frame")
    if frame.coordinate_spec.dimension == 2:
        coords = frame.joints[frame.joint_valid]
        if np.any(coords < 0) or np.any(coords > 1):
            raise ContractError(
                "PoseFrame.joints: valid normalized D2 coordinates must be in [0,1]"
            )
    if frame.scene_status not in {"empty_confirmed", "persons_labeled", "incomplete"}:
        raise ContractError("PoseFrame.scene_status: unsupported status")
    if frame.scene_status == "empty_confirmed" and (n != 0 or not frame.annotation_complete):
        raise ContractError(
            "PoseFrame.scene_status: empty_confirmed requires complete annotation and N=0"
        )
    if frame.scene_status == "persons_labeled" and n == 0:
        raise ContractError("PoseFrame.scene_status: persons_labeled requires at least one person")
    if frame.scene_status == "incomplete" and frame.annotation_complete:
        raise ContractError("PoseFrame.annotation_complete: incomplete scene cannot be complete")
    if not frame.annotation_complete and frame.scene_status != "incomplete":
        raise ContractError("PoseFrame.scene_status: partial annotation must be incomplete")
    if capacity is not None and (
        isinstance(capacity, bool) or not isinstance(capacity, int) or capacity <= 0
    ):
        raise ContractError("capacity: expected positive integer Q")
    status = "capacity_overflow" if capacity is not None and n > capacity else "supported"
    return LabelValidation(n, status, capacity, frame.annotation_complete)


def require_training_capacity(frame: PoseFrame, capacity: int) -> LabelValidation:
    result = validate_labels(frame, capacity=capacity)
    if result.capacity_status == "capacity_overflow":
        raise CapacityOverflow(result.person_count, capacity)
    return result
