"""BODY18 order and explicit COCO17 mapping; masks are True when valid."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .schema import ContractError, immutable_array


BODY18_NAMES = (
    "nose",
    "neck",
    "right_shoulder",
    "right_elbow",
    "right_wrist",
    "left_shoulder",
    "left_elbow",
    "left_wrist",
    "right_hip",
    "right_knee",
    "right_ankle",
    "left_hip",
    "left_knee",
    "left_ankle",
    "right_eye",
    "left_eye",
    "right_ear",
    "left_ear",
)
BODY18_EDGES = (
    (1, 0),
    (1, 2),
    (2, 3),
    (3, 4),
    (1, 5),
    (5, 6),
    (6, 7),
    (1, 8),
    (8, 9),
    (9, 10),
    (1, 11),
    (11, 12),
    (12, 13),
    (0, 14),
    (14, 16),
    (0, 15),
    (15, 17),
)
# BODY18 target index -> COCO17 source index; neck has no direct source.
COCO17_TO_BODY18 = (
    (0, 0),
    (2, 6),
    (3, 8),
    (4, 10),
    (5, 5),
    (6, 7),
    (7, 9),
    (8, 12),
    (9, 14),
    (10, 16),
    (11, 11),
    (12, 13),
    (13, 15),
    (14, 2),
    (15, 1),
    (16, 4),
    (17, 3),
)


def validate_joint_order(names: tuple[str, ...], *, spec_id: str = "BODY18") -> None:
    if spec_id != "BODY18":
        raise ContractError(f"joint_spec_id: expected BODY18, got {spec_id!r}")
    if tuple(names) != BODY18_NAMES:
        raise ContractError("joint_names: order does not match BODY18; map joints explicitly")


def synthesize_neck(
    joints: NDArray[np.float32], joint_valid: NDArray[np.bool_]
) -> tuple[NDArray[np.float32], NDArray[np.bool_]]:
    points = np.asarray(joints)
    valid = np.asarray(joint_valid)
    if (
        points.dtype != np.float32
        or points.ndim < 2
        or points.shape[-2] != 18
        or points.shape[-1] not in (2, 3)
    ):
        raise ContractError("joints: expected float32 [...,18,D] with D=2 or 3")
    if valid.dtype != np.bool_ or valid.shape != points.shape[:-1]:
        raise ContractError("joint_valid: expected bool [...,18] matching joints")
    points = points.copy()
    valid = valid.copy()
    neck_valid = valid[..., 2] & valid[..., 5]
    neck_valid &= np.isfinite(points[..., 2, :]).all(axis=-1)
    neck_valid &= np.isfinite(points[..., 5, :]).all(axis=-1)
    neck = (points[..., 2, :] + points[..., 5, :]) * np.float32(0.5)
    points[..., 1, :] = np.where(neck_valid[..., None], neck, np.nan)
    valid[..., 1] = neck_valid
    return (
        immutable_array(points, dtype=np.float32, name="joints"),
        immutable_array(valid, dtype=np.bool_, name="joint_valid"),
    )


def coco17_to_body18(
    joints: NDArray[np.float32], joint_valid: NDArray[np.bool_]
) -> tuple[NDArray[np.float32], NDArray[np.bool_]]:
    """Map every person and synthesize neck only when both shoulders are valid."""
    points = np.asarray(joints)
    valid = np.asarray(joint_valid)
    if (
        points.dtype != np.float32
        or points.ndim < 2
        or points.shape[-2] != 17
        or points.shape[-1] not in (2, 3)
    ):
        raise ContractError("joints: expected float32 [...,17,D] with D=2 or 3")
    if valid.dtype != np.bool_ or valid.shape != points.shape[:-1]:
        raise ContractError("joint_valid: expected bool [...,17] matching joints")
    out = np.full((*points.shape[:-2], 18, points.shape[-1]), np.nan, dtype=np.float32)
    mask = np.zeros((*points.shape[:-2], 18), dtype=np.bool_)
    for target, source in COCO17_TO_BODY18:
        out[..., target, :] = points[..., source, :]
        mask[..., target] = valid[..., source] & np.isfinite(points[..., source, :]).all(axis=-1)
    return synthesize_neck(out, mask)


def _image_scale(image_width: int, image_height: int) -> NDArray[np.float32]:
    for name, value in (("image_width", image_width), ("image_height", image_height)):
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or value <= 0
        ):
            raise ContractError(f"{name}: expected a positive integer")
    return np.asarray((image_width, image_height), dtype=np.float32)


def normalize_image_coordinates(
    joints_px: NDArray[np.float32], *, image_width: int, image_height: int
) -> NDArray[np.float32]:
    """Use x/W,y/H without clipping; the label validator checks valid ranges."""
    points = np.asarray(joints_px)
    if points.dtype != np.float32 or points.shape[-1:] != (2,):
        raise ContractError("joints_px: expected float32 [...,2]")
    out = points / _image_scale(image_width, image_height)
    return immutable_array(out, dtype=np.float32, name="normalized_joints")


def denormalize_image_coordinates(
    joints: NDArray[np.float32], *, image_width: int, image_height: int
) -> NDArray[np.float32]:
    points = np.asarray(joints)
    if points.dtype != np.float32 or points.shape[-1:] != (2,):
        raise ContractError("joints: expected float32 [...,2]")
    out = points * _image_scale(image_width, image_height)
    return immutable_array(out, dtype=np.float32, name="pixel_joints")
