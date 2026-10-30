"""Bounded storage and immutable, zero-padded windows of committed grid slots."""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING

import numpy as np

from .contracts import validate_data_spec, validate_window
from .schema import ContractError, DataSpec, WindowPacket, _integer

if TYPE_CHECKING:
    from .grid import GridSlot


class WindowBuffer:
    """Retain T committed slots; callers retain published windows independently.

    The origin is the first real slot, not the first received packet. Padding
    before that origin is unavailable and never counts toward warmup. A past
    target is readable only while all of its real slots remain in the buffer.
    """

    def __init__(
        self,
        *,
        session_id: str,
        start_ns: int,
        settle_ns: int,
        spec: DataSpec,
        provenance_id: str,
        availability_basis: str,
    ):
        validate_data_spec(spec)
        for name, value in (("start_ns", start_ns), ("settle_ns", settle_ns)):
            _integer(value, f"window.{name}", upper=np.iinfo(np.int64).max)
        for name, value in (("session_id", session_id), ("provenance_id", provenance_id)):
            if not isinstance(value, str) or not value.strip():
                raise ContractError(f"window.{name}: required")
        if availability_basis not in {"synthetic", "measured_infer", "legacy_bridge_proxy"}:
            raise ContractError("window.availability_basis: unsupported basis")
        if (spec.T - 1) * spec.grid_step_ns > np.iinfo(np.int64).max:
            raise ContractError("window: context span exceeds int64 nanoseconds")
        self.session_id = session_id
        self.start_ns = int(start_ns)
        self.settle_ns = int(settle_ns)
        self.spec = spec
        self.provenance_id = provenance_id
        self.availability_basis = availability_basis
        self._slots: deque[GridSlot] = deque(maxlen=spec.T)
        self._next_ns = self.start_ns

    def append(self, slot: GridSlot) -> None:
        if slot.grid_ns != self._next_ns:
            raise ContractError("window.grid_ns: slots must be consecutive and committed once")
        if slot.deadline_ns != slot.grid_ns + self.settle_ns:
            raise ContractError("window.deadline_ns: must equal grid_ns + settle_ns")
        self._slots.append(slot)
        self._next_ns += self.spec.grid_step_ns

    def window_at(self, target_ns: int) -> WindowPacket:
        _integer(target_ns, "window.target_ns", upper=np.iinfo(np.int64).max)
        target_ns = int(target_ns)
        step = self.spec.grid_step_ns
        if target_ns < self.start_ns or (target_ns - self.start_ns) % step:
            raise ContractError("window.target_ns: expected a grid tick at or after start")
        if target_ns >= self._next_ns:
            raise ContractError("window.target_ns: target has not been committed")
        first_ns = target_ns - (self.spec.T - 1) * step
        if not self._slots or max(first_ns, self.start_ns) < self._slots[0].grid_ns:
            raise ContractError("window.target_ns: required history has been evicted")
        shape = (self.spec.T, self.spec.L, self.spec.F, self.spec.C)
        raw_x = np.zeros(shape, dtype=np.float32)
        valid = np.zeros(shape, dtype=np.bool_)
        states = np.zeros(shape[:2], dtype=np.uint8)
        ages = np.full(shape[:2], -1, dtype=np.int64)
        for slot in self._slots:
            if first_ns <= slot.grid_ns <= target_ns:
                index = (slot.grid_ns - first_ns) // step
                raw_x[index] = slot.raw_x
                valid[index] = slot.feature_valid
                states[index] = slot.link_state
                ages[index] = slot.age_ns
        count = (target_ns - self.start_ns) // step + 1
        # Empty/stale current input remains unknown even if the context still
        # contains an old observation. No pose/count or empty-room claim here.
        status = (
            "warming_up"
            if count < self.spec.T
            else ("ready" if valid[-1, ..., 0].any() else "insufficient_coverage")
        )
        packet = WindowPacket(
            session_id=self.session_id,
            target_ns=target_ns,
            decision_ns=target_ns + self.settle_ns,
            grid_ns=target_ns - np.arange(self.spec.T - 1, -1, -1, dtype=np.int64) * step,
            raw_x=raw_x,
            feature_valid=valid,
            observed_mask=states == 1,
            link_state=states,
            age_ns=ages,
            schema_id=self.spec.derived_schema_id,
            feature_id=self.spec.feature_id,
            provenance_id=self.provenance_id,
            availability_basis=self.availability_basis,
            status=status,
        )
        validate_window(packet, self.spec)
        return packet
