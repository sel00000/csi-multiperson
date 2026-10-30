"""S06: arrival-causal, single-commit grid with bounded past-only hold.

Numeric features are supplied explicitly; data.features.FeatureBuilder provides
the S07 log-amplitude/delta formula. This module does not infer acquisition/arrival times.
All queued arrivals and boundaries are applied only at their decision prefix,
so ingesting a file ahead of commit has the same result as streaming it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
import heapq
from typing import TYPE_CHECKING, Protocol

import numpy as np
from numpy.typing import NDArray

from csi_mp.ingest.events import TimedObservation

from .schema import ContractError, DataSpec, Stamp, _integer, immutable_array
from .window import WindowBuffer

if TYPE_CHECKING:
    from csi_mp.config import AppConfig
    from .schema import WindowPacket


def _ns(value: int, name: str, *, positive: bool = False) -> None:
    _integer(value, name, lower=1 if positive else 0, upper=np.iinfo(np.int64).max)


@dataclass(frozen=True)
class GridPolicy:
    """Development timing policy; these defaults are not measured guarantees."""

    step_ns: int = 10_000_000
    output_step_ns: int = 50_000_000
    settle_ns: int = 0
    max_hold_age_ns: int = 30_000_000

    def __post_init__(self) -> None:
        for name in ("step_ns", "output_step_ns", "settle_ns", "max_hold_age_ns"):
            _ns(getattr(self, name), f"GridPolicy.{name}", positive=name.endswith("step_ns"))
            object.__setattr__(self, name, int(getattr(self, name)))
        if self.output_step_ns % self.step_ns:
            raise ContractError("GridPolicy.output_step_ns: must be a multiple of step_ns")

    @classmethod
    def from_config(cls, config: AppConfig) -> GridPolicy:
        values = config.as_dict()["input"]

        def nanoseconds(name: str) -> int:
            value = Decimal(str(values[name])) * 1_000_000
            if value != value.to_integral_value():
                raise ContractError(f"input.{name}: must represent whole nanoseconds")
            return int(value)

        return cls(
            step_ns=nanoseconds("grid_step_ms"),
            output_step_ns=nanoseconds("output_period_ms"),
            settle_ns=nanoseconds("settle_ms"),
            max_hold_age_ns=nanoseconds("max_hold_age_ms"),
        )


@dataclass(frozen=True)
class SlotFeatures:
    raw_x: NDArray[np.float32]
    feature_valid: NDArray[np.bool_]

    def __post_init__(self) -> None:
        for name, dtype in (("raw_x", np.float32), ("feature_valid", np.bool_)):
            value = immutable_array(getattr(self, name), dtype=dtype, name=f"SlotFeatures.{name}")
            if value.shape != (56, 2):
                raise ContractError(f"SlotFeatures.{name}.shape: expected (56, 2)")
            object.__setattr__(self, name, value)
        if not np.isfinite(self.raw_x).all() or np.any(self.raw_x[~self.feature_valid] != 0):
            raise ContractError("SlotFeatures.raw_x: finite values and invalid=0 required")
        if np.any(self.feature_valid[:, 1] & ~self.feature_valid[:, 0]):
            raise ContractError("SlotFeatures.feature_valid: delta requires amplitude")


class FeatureBuilder(Protocol):
    """S07 injection boundary. Reorders must not replace the forward baseline.

    The grid snapshots new-observation features immediately after on_observation
    and masks held deltas itself. Reset must break differentiation history; it
    must not rewrite any already published features.
    """

    def on_observation(self, observation: TimedObservation) -> None: ...

    def values_for_slot(self, observation_id: str, is_new: bool) -> SlotFeatures: ...

    def reset(self, links: tuple[int, ...], reason: str) -> None: ...


@dataclass(frozen=True)
class GridSlot:
    grid_ns: int
    deadline_ns: int
    raw_x: NDArray[np.float32]
    feature_valid: NDArray[np.bool_]
    observed_mask: NDArray[np.bool_]
    link_state: NDArray[np.uint8]
    age_ns: NDArray[np.int64]
    observation_ids: tuple[str | None, ...]
    segment_ids: tuple[str | None, ...]

    def __post_init__(self) -> None:
        _ns(self.grid_ns, "GridSlot.grid_ns")
        _ns(self.deadline_ns, "GridSlot.deadline_ns")
        if self.deadline_ns < self.grid_ns:
            raise ContractError("GridSlot.deadline_ns: cannot precede grid_ns")
        for name, dtype, shape in (
            ("raw_x", np.float32, (9, 56, 2)),
            ("feature_valid", np.bool_, (9, 56, 2)),
            ("observed_mask", np.bool_, (9,)),
            ("link_state", np.uint8, (9,)),
            ("age_ns", np.int64, (9,)),
        ):
            value = immutable_array(getattr(self, name), dtype=dtype, name=f"GridSlot.{name}")
            if value.shape != shape:
                raise ContractError(f"GridSlot.{name}.shape: expected {shape}")
            object.__setattr__(self, name, value)
        for name in ("observation_ids", "segment_ids"):
            ids = tuple(getattr(self, name))
            if len(ids) != 9 or any(
                v is not None and (not isinstance(v, str) or not v) for v in ids
            ):
                raise ContractError(f"GridSlot.{name}: expected nine optional nonempty IDs")
            object.__setattr__(self, name, ids)
        if not np.isfinite(self.raw_x).all() or np.any(self.raw_x[~self.feature_valid] != 0):
            raise ContractError("GridSlot.raw_x: finite values and invalid=0 required")
        if np.any(self.link_state > 2) or not np.array_equal(
            self.observed_mask, self.link_state == 1
        ):
            raise ContractError("GridSlot.link_state: expected 0=unavailable, 1=observed, 2=held")
        missing = self.link_state == 0
        if np.any(self.age_ns[missing] != -1) or np.any(self.feature_valid[missing]):
            raise ContractError("GridSlot: unavailable requires age=-1 and invalid features")
        if np.any(self.age_ns[~missing] < 0):
            raise ContractError("GridSlot.age_ns: available links require nonnegative ages")
        if np.any(self.feature_valid[self.link_state == 2, :, 1]):
            raise ContractError("GridSlot.feature_valid: held delta must be invalid")
        if np.any(self.feature_valid[..., 1] & ~self.feature_valid[..., 0]):
            raise ContractError("GridSlot.feature_valid: delta requires amplitude")
        for link, invalid in enumerate(missing):
            if invalid != (self.observation_ids[link] is None) or invalid != (
                self.segment_ids[link] is None
            ):
                raise ContractError("GridSlot: IDs must be present exactly for available links")


@dataclass
class _Candidate:
    observation: TimedObservation
    features: SlotFeatures
    order: int
    segment_id: str
    selected: bool = False


@dataclass(frozen=True)
class _Arrival:
    observation: TimedObservation
    late_ingest: bool
    availability_regression: bool


@dataclass
class _Link:
    segment: tuple[str, str | None] | None = None
    retired: set[tuple[str, str | None]] = field(default_factory=set)
    candidates: list[_Candidate] = field(default_factory=list)
    high_acquisition_ns: int | None = None


class CausalGridBuilder:
    """One builder per session, common clock and declared availability basis.

    ``start`` explicitly anchors the first grid slot (it need not be UTC-aligned).
    Feed TimedObservations in actual arrival order, then advance a nondecreasing
    availability watermark. A regressed recorded clock is queued after the last
    observation admission; original timestamps are never rewritten.
    Ties follow ingest call order. A post-watermark arrival is admitted only
    after that watermark, even if its recorded timestamp is older.

    A new builder is the full-session reset. ``invalidate`` is a link boundary.
    Window retention is bounded to T; keep returned immutable packets for replay.
    The output cadence is policy metadata for the runtime; window_at accepts
    any committed grid tick and never commits or waits implicitly.
    """

    def __init__(
        self,
        *,
        session_id: str,
        start: Stamp,
        feature_builder: FeatureBuilder,
        provenance_id: str,
        availability_basis: str,
        policy: GridPolicy | None = None,
        spec: DataSpec | None = None,
    ):
        if not isinstance(start, Stamp):
            raise ContractError("grid.start: expected a Stamp naming the common clock")
        self.policy = policy or GridPolicy()
        self.spec = spec or DataSpec()
        if self.spec.grid_step_ns != self.policy.step_ns:
            raise ContractError("grid.step_ns: must match DataSpec.grid_step_ns")
        if any(
            not callable(getattr(feature_builder, name, None))
            for name in ("on_observation", "values_for_slot", "reset")
        ):
            raise ContractError("grid.feature_builder: an explicit FeatureBuilder is required")
        if getattr(feature_builder, "feature_id", self.spec.feature_id) != self.spec.feature_id:
            raise ContractError("grid.feature_builder.feature_id: must match DataSpec.feature_id")
        _ns(start.ns + self.policy.settle_ns, "grid.first_deadline_ns")
        self.start = start
        self.availability_basis = availability_basis
        self.features = feature_builder
        self._window = WindowBuffer(
            session_id=session_id,
            start_ns=start.ns,
            settle_ns=self.policy.settle_ns,
            spec=self.spec,
            provenance_id=provenance_id,
            availability_basis=availability_basis,
        )
        self._next_grid_ns = start.ns
        self._watermark_ns = -1
        self._last_availability_ns = -1
        self._last_admission_ns = -1
        self._order = 0
        self._pending: list[tuple[int, int, _Arrival | tuple[tuple[int, ...], str]]] = []
        self._links = [_Link() for _ in range(self.spec.L)]
        self._seen_ids: set[str] = set()
        self._seen_events: set[str] = set()
        self._epochs: dict[tuple[str, int], str] = {}
        self._retired_epochs: dict[tuple[str, int], set[str]] = {}
        self._counts = dict.fromkeys(
            (
                "received_observations",
                "duplicate_observations",
                "excluded_observations",
                "retired_observations",
                "late_observations",
                "late_ingest_observations",
                "availability_regressions",
                "committed_slots",
                "observed_slots",
                "held_slots",
                "unavailable_slots",
                "invalidations",
            ),
            0,
        )

    @classmethod
    def from_config(
        cls,
        config: AppConfig,
        *,
        start: Stamp,
        feature_builder: FeatureBuilder,
        provenance_id: str,
        availability_basis: str,
    ) -> CausalGridBuilder:
        return cls(
            session_id=config.as_dict()["capture"]["session_id"],
            start=start,
            feature_builder=feature_builder,
            provenance_id=provenance_id,
            availability_basis=availability_basis,
            policy=GridPolicy.from_config(config),
            spec=config.model_spec,
        )

    def ingest(self, observation: TimedObservation) -> None:
        if not isinstance(observation, TimedObservation):
            raise ContractError("grid.observation: expected TimedObservation")
        if (observation.clock_domain, observation.clock_kind) != (
            self.start.domain,
            self.start.kind,
        ):
            raise ContractError(
                "grid.clock: observation requires the declared common clock domain/kind"
            )
        if observation.availability_basis != self.availability_basis:
            raise ContractError("grid.availability_basis: mixed availability bases are not allowed")
        if not isinstance(observation.observation_id, str) or not observation.observation_id:
            raise ContractError("grid.observation_id: required")
        _ns(observation.availability_ns, "grid.availability_ns")
        regressed = observation.availability_ns < self._last_availability_ns
        arrival = _Arrival(
            observation, observation.availability_ns <= self._watermark_ns, regressed
        )
        # Admission is an ordering bound, not a repaired acquisition/arrival
        # timestamp. A backward clock must not move its exclusion/reset before
        # observations that have already arrived in this same input trace.
        effective_ns = max(observation.availability_ns, self._last_admission_ns + int(regressed))
        self._last_admission_ns = self._enqueue(effective_ns, arrival)
        self._last_availability_ns = observation.availability_ns

    def _enqueue(self, available_ns, event) -> int:
        effective_ns = max(available_ns, self._watermark_ns + 1)
        _ns(effective_ns, "grid.effective_availability_ns")
        heapq.heappush(self._pending, (effective_ns, self._order, event))
        self._order += 1
        return effective_ns

    def invalidate(
        self,
        links: tuple[int, ...],
        reason: str,
        *,
        available_ns: int | None = None,
    ) -> None:
        """Queue a boundary in the common clock, defaulting to after the watermark.

        Already committed slots remain intact. Specify available_ns when queuing
        controls alongside observations ahead of time; no wall clock is invented.
        """
        links = tuple(links)
        for link in links:
            _integer(link, "grid.invalidate.links", upper=self.spec.L - 1)
        if not links or len(set(links)) != len(links):
            raise ContractError("grid.invalidate.links: nonempty distinct links required")
        if not isinstance(reason, str) or not reason:
            raise ContractError("grid.invalidate.reason: required")
        if available_ns is None:
            available_ns = self._watermark_ns + 1
        _ns(available_ns, "grid.invalidate.available_ns")
        self._enqueue(available_ns, (tuple(sorted(links)), reason))

    def _clear(self, links: tuple[int, ...], reason: str) -> None:
        if not links:
            return
        self.features.reset(links, reason)
        for link in links:
            self._links[link].candidates.clear()
            self._links[link].high_acquisition_ns = None
        self._counts["invalidations"] += len(links)

    @staticmethod
    def _scope_links(scope: str, device: int) -> tuple[int, ...]:
        return (
            tuple(3 * device + tx for tx in range(3))
            if scope in {"rx", "clock"}
            else (tuple(3 * rx + device for rx in range(3)))
        )

    def _observe(self, arrival: _Arrival, effective_ns: int, order: int) -> None:
        obs = arrival.observation
        self._counts["received_observations"] += 1
        if obs.observation_id in self._seen_ids:
            self._counts["duplicate_observations"] += 1
            return
        self._seen_ids.add(obs.observation_id)
        if arrival.late_ingest:
            self._counts["late_ingest_observations"] += 1
        if arrival.availability_regression:
            self._counts["availability_regressions"] += 1
        if obs.acquisition_ns is not None:
            first_tick = (
                self.start.ns
                + max(
                    0,
                    (obs.acquisition_ns - self.start.ns + self.policy.step_ns - 1)
                    // self.policy.step_ns,
                )
                * self.policy.step_ns
            )
            if effective_ns > first_tick + self.policy.settle_ns:
                self._counts["late_observations"] += 1
        link = obs.frame.link_index
        state = self._links[link]
        segment = (obs.epochs.link_segment, obs.clock_segment_id)
        identities = {
            ("rx", obs.frame.rx_id): obs.epochs.rx_epoch,
            ("tx", obs.frame.tx_id): obs.epochs.tx_epoch,
            ("clock", obs.frame.rx_id): obs.clock_segment_id,
        }
        if any(
            v is not None and v in self._retired_epochs.get(k, ()) for k, v in identities.items()
        ) or (obs.usable and segment in state.retired):
            self._counts["retired_observations"] += 1
            return
        reset_links: set[int] = set()
        reset_epochs: dict[tuple[str, int], str | None] = {}
        for event in obs.epochs.events:
            if event.kind in {"rx_reset", "tx_reset"} and event.event_id not in self._seen_events:
                self._seen_events.add(event.event_id)
                reset_links.update(self._scope_links(event.scope, event.device_id))
                key = (event.scope, event.device_id)
                reset_epochs[key] = self._epochs.get(key)
        # Boundaries attached to a timed frame become known at its mapped
        # availability; EpochEvent.available may name a different source clock.
        for key, identity in identities.items():
            if identity is None:
                continue
            old = self._epochs.get(key)
            if old is not None and identity != old:
                self._retired_epochs.setdefault(key, set()).add(old)
                reset_links.update(self._scope_links(*key))
            self._epochs[key] = identity
        if not obs.usable:
            if obs.exclusion_reason in {
                "availability_clock_regression",
                "bridge_clock_regression",
                "clock_fit_regression",
            }:
                reset_links.update(self._scope_links("rx", obs.frame.rx_id))
                key = ("clock", obs.frame.rx_id)
                reset_epochs[key] = self._epochs.get(key)
            elif obs.epochs.classification != "duplicate":
                reset_links.add(link)
            # The next usable frame may not arrive immediately. Retire the
            # pre-boundary identities now, so a delayed old frame cannot restore
            # them during that gap. A new identity already learned above stays.
            for key, old in reset_epochs.items():
                if old is not None:
                    self._retired_epochs.setdefault(key, set()).add(old)
                    if self._epochs.get(key) == old:
                        del self._epochs[key]
            for affected in reset_links:
                old_segment = self._links[affected].segment
                if old_segment is not None:
                    self._links[affected].retired.add(old_segment)
                    self._links[affected].segment = None
            self._clear(tuple(sorted(reset_links)), obs.exclusion_reason or "excluded")
            self._counts["excluded_observations"] += 1
            return
        if state.segment is not None and state.segment != segment:
            state.retired.add(state.segment)
            reset_links.add(link)
        self._clear(tuple(sorted(reset_links)), "epoch_or_clock_boundary")
        state.segment = segment
        acquisition = obs.acquisition_ns
        if (
            state.high_acquisition_ns is not None
            and acquisition - state.high_acquisition_ns > self.policy.max_hold_age_ns
        ):
            # Preserve any still-selectable earlier candidate, but sever the
            # feature baseline before computing the first observation after a gap.
            self.features.reset((link,), "long_gap")
        self.features.on_observation(obs)
        features = self.features.values_for_slot(obs.observation_id, True)
        if not isinstance(features, SlotFeatures):
            raise ContractError("grid.feature_builder: values_for_slot must return SlotFeatures")
        # Two immutable segment components, length-prefixed to avoid ambiguity.
        segment_id = f"{len(segment[0])}:{segment[0]}{segment[1]}"
        state.candidates.append(_Candidate(obs, features, order, segment_id))
        state.high_acquisition_ns = max(acquisition, state.high_acquisition_ns or acquisition)

    def _apply_through(self, deadline_ns: int) -> None:
        while self._pending and self._pending[0][0] <= deadline_ns:
            effective_ns, order, event = heapq.heappop(self._pending)
            if isinstance(event, _Arrival):
                self._observe(event, effective_ns, order)
            else:
                self._clear(*event)

    def _slot(self, grid_ns: int) -> GridSlot:
        values = np.zeros((9, 56, 2), dtype=np.float32)
        valid = np.zeros((9, 56, 2), dtype=np.bool_)
        states = np.zeros(9, dtype=np.uint8)
        ages = np.full(9, -1, dtype=np.int64)
        observation_ids: list[str | None] = [None] * 9
        segment_ids: list[str | None] = [None] * 9
        for link, state in enumerate(self._links):
            state.candidates[:] = [
                c
                for c in state.candidates
                if (c.observation.acquisition_ns >= grid_ns - self.policy.max_hold_age_ns)
            ]
            eligible = [c for c in state.candidates if c.observation.acquisition_ns <= grid_ns]
            if not eligible:
                continue
            candidate = max(eligible, key=lambda c: (c.observation.acquisition_ns, c.order))
            is_new = not candidate.selected
            values[link] = candidate.features.raw_x
            valid[link] = candidate.features.feature_valid
            if not is_new:
                values[link, :, 1] = 0
                valid[link, :, 1] = False
            states[link] = 1 if is_new else 2
            ages[link] = grid_ns - candidate.observation.acquisition_ns
            observation_ids[link] = candidate.observation.observation_id
            segment_ids[link] = candidate.segment_id
            candidate.selected = True
            # A selected newer acquisition permanently supersedes older ones.
            state.candidates[:] = [
                c
                for c in state.candidates
                if c is candidate or (c.observation.acquisition_ns > grid_ns)
            ]
        return GridSlot(
            grid_ns,
            grid_ns + self.policy.settle_ns,
            values,
            valid,
            states == 1,
            states,
            ages,
            tuple(observation_ids),
            tuple(segment_ids),
        )

    def commit_through(self, deadline_ns: int) -> list[GridSlot]:
        _ns(deadline_ns, "grid.deadline_ns")
        if deadline_ns < self._watermark_ns:
            raise ContractError("grid.deadline_ns: watermark cannot move backward")
        committed = []
        while self._next_grid_ns + self.policy.settle_ns <= deadline_ns:
            self._apply_through(self._next_grid_ns + self.policy.settle_ns)
            slot = self._slot(self._next_grid_ns)
            self._window.append(slot)
            committed.append(slot)
            self._counts["committed_slots"] += 1
            for state, key in enumerate(("unavailable_slots", "observed_slots", "held_slots")):
                self._counts[key] += int(np.count_nonzero(slot.link_state == state))
            self._next_grid_ns += self.policy.step_ns
        self._apply_through(deadline_ns)
        self._watermark_ns = int(deadline_ns)
        return committed

    def window_at(self, target_ns: int) -> WindowPacket:
        return self._window.window_at(target_ns)

    def metrics(self) -> dict[str, int]:
        """Raw counters for the processed prefix, not rates or live telemetry."""
        return dict(self._counts)
