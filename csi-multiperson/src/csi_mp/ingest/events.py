"""Immutable S04 timing values; frame/epoch provenance stays in shared types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from csi_mp.data.schema import (
    AvailabilityBasis,
    ClockKind,
    ContractError,
    EpochResult,
    Stamp,
    WireFrame,
)


def integer(value: int, name: str, *, minimum: int = 0) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ContractError(f"{name}: expected an integer >= {minimum}")


@dataclass(frozen=True)
class ClockTransform:
    """An explicit, previously available calibration between host clocks.

    The positive rational rate maps source nanoseconds to target nanoseconds.
    ``available`` is when the calibration became known, in the target domain.
    A fit over a whole recording must not be presented as a prior calibration.
    """

    transform_id: str
    source_reference: Stamp
    target_reference: Stamp
    available: Stamp
    evidence_ref: str
    error_ns: int | None = None
    rate_numerator: int = 1
    rate_denominator: int = 1

    def __post_init__(self) -> None:
        if not self.transform_id or not self.evidence_ref:
            raise ContractError("ClockTransform: transform_id and evidence_ref are required")
        if (self.available.domain, self.available.kind) != (
            self.target_reference.domain,
            self.target_reference.kind,
        ):
            raise ContractError("ClockTransform.available: expected the target clock")
        for name in ("rate_numerator", "rate_denominator"):
            integer(getattr(self, name), f"ClockTransform.{name}", minimum=1)
        if self.error_ns is not None:
            integer(self.error_ns, "ClockTransform.error_ns")

    def map_ns(self, stamp: Stamp) -> int:
        if (stamp.domain, stamp.kind) != (
            self.source_reference.domain,
            self.source_reference.kind,
        ):
            raise ContractError("ClockTransform.source_reference: incompatible clock")
        return self.target_reference.ns + (
            (stamp.ns - self.source_reference.ns) * self.rate_numerator // self.rate_denominator
        )

    def to_dict(self) -> dict:
        return {
            "transform_id": self.transform_id,
            "source_reference": self.source_reference.to_dict(),
            "target_reference": self.target_reference.to_dict(),
            "available": self.available.to_dict(),
            "evidence_ref": self.evidence_ref,
            "error_ns": None if self.error_ns is None else str(self.error_ns),
            "rate_numerator": str(self.rate_numerator),
            "rate_denominator": str(self.rate_denominator),
        }


@dataclass(frozen=True)
class ClockModel:
    """The model used at publication, never a pointer to the next online fit.

    Error is the empirical spread of bridge residuals, not a bound on physical
    acquisition error. Unknown fixed transport delay cannot be inferred here.
    """

    model_id: str
    rx_epoch: str
    segment_id: str
    reference_timer_us: int
    reference_ns: int
    rate_numerator: int
    rate_denominator: int
    sample_count: int
    window_count: int
    span_us: int
    status: Literal["warming_up", "ready"]
    residual_ns: int | None
    rate_limited: bool
    calibration_error_ns: int | None
    transform_ids: tuple[str, ...]

    def estimate(self, timer_us: int) -> int:
        return self.reference_ns + (
            (timer_us - self.reference_timer_us) * self.rate_numerator // self.rate_denominator
        )

    def to_dict(self) -> dict:
        return {
            "model_id": self.model_id,
            "rx_epoch": self.rx_epoch,
            "segment_id": self.segment_id,
            "reference_timer_us": str(self.reference_timer_us),
            "reference_ns": str(self.reference_ns),
            "rate_numerator": str(self.rate_numerator),
            "rate_denominator": str(self.rate_denominator),
            "sample_count": self.sample_count,
            "window_count": self.window_count,
            "span_us": str(self.span_us),
            "status": self.status,
            "residual_ns": None if self.residual_ns is None else str(self.residual_ns),
            "rate_limited": self.rate_limited,
            "calibration_error_ns": (
                None if self.calibration_error_ns is None else str(self.calibration_error_ns)
            ),
            "transform_ids": list(self.transform_ids),
            "error_basis": "empirical_bridge_residual_not_acquisition_bound",
        }


@dataclass(frozen=True)
class TimedObservation:
    observation_id: str
    frame: WireFrame
    epochs: EpochResult
    acquisition_ns: int | None
    availability_ns: int
    availability_basis: AvailabilityBasis
    clock_domain: str
    clock_kind: ClockKind
    clock_reference: Stamp
    clock_model: ClockModel | None
    clock_error_ns: int | None
    clock_status: Literal["warming_up", "ready", "excluded"]
    exclusion_reason: str | None
    transform_ids: tuple[str, ...] = ()
    legacy_delay_ns: int = 0

    def __post_init__(self) -> None:
        integer(self.availability_ns, "TimedObservation.availability_ns")
        integer(self.legacy_delay_ns, "TimedObservation.legacy_delay_ns")
        if self.acquisition_ns is not None:
            integer(self.acquisition_ns, "TimedObservation.acquisition_ns")
            if self.acquisition_ns > self.availability_ns:
                raise ContractError("TimedObservation: acquisition exceeds availability")
        if self.clock_error_ns is not None:
            integer(self.clock_error_ns, "TimedObservation.clock_error_ns")
        if (self.clock_domain, self.clock_kind) != (
            self.clock_reference.domain,
            self.clock_reference.kind,
        ):
            raise ContractError("TimedObservation.clock_reference: incompatible clock")
        if self.clock_status not in {"warming_up", "ready", "excluded"}:
            raise ContractError("TimedObservation.clock_status: unsupported status")
        if self.availability_basis not in {"measured_infer", "legacy_bridge_proxy", "synthetic"}:
            raise ContractError("TimedObservation.availability_basis: unsupported basis")
        if self.clock_status == "excluded":
            if self.acquisition_ns is not None or not self.exclusion_reason:
                raise ContractError("TimedObservation: excluded input needs a reason and null time")
        elif (
            not self.epochs.usable
            or self.acquisition_ns is None
            or self.clock_model is None
            or self.exclusion_reason is not None
            or self.clock_status != self.clock_model.status
        ):
            raise ContractError("TimedObservation: usable input needs epochs, model and time")
        object.__setattr__(self, "transform_ids", tuple(self.transform_ids))

    @property
    def clock_model_id(self) -> str | None:
        return None if self.clock_model is None else self.clock_model.model_id

    @property
    def clock_segment_id(self) -> str | None:
        return None if self.clock_model is None else self.clock_model.segment_id

    @property
    def usable(self) -> bool:
        # Warmup estimates are explicitly preliminary; consumers can require ready.
        return self.clock_status != "excluded"

    def to_dict(self) -> dict:
        """Timing metadata only; the enclosing frame record stores frame/epochs."""
        return {
            "schema_id": "csi_mp.timing.v1",
            "observation_id": self.observation_id,
            "acquisition_ns": None if self.acquisition_ns is None else str(self.acquisition_ns),
            "availability_ns": str(self.availability_ns),
            "availability_basis": self.availability_basis,
            "clock_domain": self.clock_domain,
            "clock_kind": self.clock_kind,
            "clock_reference": self.clock_reference.to_dict(),
            "clock_model_id": self.clock_model_id,
            "clock_segment_id": self.clock_segment_id,
            "clock_model": None if self.clock_model is None else self.clock_model.to_dict(),
            "clock_error_ns": None if self.clock_error_ns is None else str(self.clock_error_ns),
            "clock_status": self.clock_status,
            "exclusion_reason": self.exclusion_reason,
            "transform_ids": list(self.transform_ids),
            "legacy_delay_ns": str(self.legacy_delay_ns),
            "usable": self.usable,
        }
