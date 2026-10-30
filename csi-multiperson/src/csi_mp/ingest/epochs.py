"""Causal RX timer and TX sequence epochs for the legacy wire protocol.

The wire boot byte belongs to the receiver. Sequence resets have no equivalent
wire flag, so one low sequence is quarantined; a configured progressing prefix
may establish an *inferred* reset. Only a trusted control boundary confirms it.
The default thresholds are fixture policy, not measured device characteristics.

No earlier result is edited when subsequent evidence resolves uncertainty.
Caches are bounded; observations older than the retained/reorder horizon are
excluded rather than silently treated as new. Exact previous-epoch bytes are
also quarantined. The legacy protocol cannot prove the epoch of every delayed
packet, particularly after a counter or boot byte has been reused.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
import hashlib

from csi_mp.data.schema import EpochEvidence, EpochEvent, EpochResult, Stamp, WireFrame

U32 = 1 << 32
HALF_U32 = 1 << 31


@dataclass(frozen=True)
class EpochPolicy:
    policy_id: str = "epochs_fixture_v1"
    timer_reorder_us: int = 100_000
    seq_reorder_packets: int = 16
    restart_confirm_packets: int = 3
    restart_max_initial_seq: int = 16
    restart_max_step: int = 16
    wrap_guard: int = 65_536
    max_forward_timer_us: int = 10_000_000
    max_seen_per_link: int = 4096

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            if name == "policy_id":
                continue
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"EpochPolicy.{name}: expected a positive integer")
        if not self.policy_id:
            raise ValueError("EpochPolicy.policy_id: required")
        if self.restart_confirm_packets < 2:
            raise ValueError("EpochPolicy.restart_confirm_packets: must be at least 2")
        if self.max_seen_per_link <= self.seq_reorder_packets:
            raise ValueError("EpochPolicy.max_seen_per_link: must exceed seq_reorder_packets")
        if self.wrap_guard >= HALF_U32:
            raise ValueError("EpochPolicy.wrap_guard: must be below the modular half range")


def _stamp(frame: WireFrame) -> Stamp:
    return frame.infer_available or frame.bridge_complete


def _reference(frame: WireFrame) -> str:
    spans = ",".join(
        f"{s.source_id}:{s.source_sha256}:{s.file_offset}+{s.byte_count}" for s in frame.spans
    )
    return f"wire:{spans}:ordinal={frame.ingest_ordinal}"


def _elapsed_us(before: Stamp, after: Stamp) -> int | None:
    if (before.domain, before.kind) != (after.domain, after.kind) or after.ns < before.ns:
        return None
    return (after.ns - before.ns) // 1000


def _delta(value: int, previous: int) -> int:
    return (value - previous + HALF_U32) % U32 - HALF_U32


@dataclass
class _RxState:
    epoch: int
    boot: int
    high_timer: int
    high_stamp: Stamp
    segment: int = 0
    uncertain: bool = False
    seen: OrderedDict[bytes, None] = field(default_factory=OrderedDict)


class RxTimerUnwrapper:
    """Keep independent receiver epochs and a forward-only timer watermark."""

    def __init__(self, policy: EpochPolicy | None = None):
        self.policy = policy or EpochPolicy()
        self._states: dict[int, _RxState] = {}
        self._next_epoch: dict[int, int] = {}
        self._event_numbers: dict[int, int] = {}
        self._pending_reset: dict[int, str] = {}
        self._retired: dict[int, OrderedDict[bytes, None]] = {}

    def _retire(self, rx_id: int, state: _RxState) -> None:
        retired = self._retired.setdefault(rx_id, OrderedDict())
        retired.update(state.seen)
        while len(retired) > self.policy.max_seen_per_link:
            retired.popitem(last=False)

    def reset(self, rx_id: int, reason: str) -> None:
        if isinstance(rx_id, bool) or rx_id not in range(3):
            raise ValueError("rx_id: expected 0, 1 or 2")
        if not reason:
            raise ValueError("reset.reason: required")
        old = self._states.pop(rx_id, None)
        if old is not None:
            self._retire(rx_id, old)
            self._next_epoch[rx_id] = old.epoch + 1
        self._pending_reset[rx_id] = reason

    def _event(
        self, frame: WireFrame, kind: str, certainty: str, refs: tuple[str, ...]
    ) -> EpochEvent:
        number = self._event_numbers.get(frame.rx_id, 0) + 1
        self._event_numbers[frame.rx_id] = number
        return EpochEvent(
            f"rx:{frame.rx_id}:event:{number}",
            "rx",
            frame.rx_id,
            kind,
            certainty,
            _stamp(frame),
            tuple(3 * frame.rx_id + tx for tx in range(3)),
            refs,
        )

    def update(self, frame: WireFrame) -> EpochResult:
        state = self._states.get(frame.rx_id)
        current_stamp = _stamp(frame)
        reference = _reference(frame)
        refs = (reference,)
        events: tuple[EpochEvent, ...] = ()
        classification = "normal"
        usable = True
        value: int | None = frame.rx_timer_u32_us
        fingerprint = hashlib.sha256(frame.wire_bytes).digest()
        if state is not None and fingerprint in self._retired.get(frame.rx_id, {}):
            if not state.uncertain:
                state.segment += 1
                state.uncertain = True
            epoch = f"rx:{frame.rx_id}:epoch:{state.epoch}"
            return EpochResult(
                epoch,
                None,
                f"{epoch}:segment:{state.segment}",
                None,
                "ambiguous",
                False,
                refs + ("reason:prior_rx_epoch_replay",),
                (),
            )
        if state is None:
            state = _RxState(
                self._next_epoch.get(frame.rx_id, 0), frame.rx_boot_u8, value, current_stamp
            )
            self._states[frame.rx_id] = state
            if frame.rx_id in self._pending_reset:
                refs += (self._pending_reset.pop(frame.rx_id),)
                classification = "rx_reset"
                events = (self._event(frame, "rx_reset", "confirmed", refs),)
        elif frame.rx_boot_u8 != state.boot:
            self._retire(frame.rx_id, state)
            state = _RxState(state.epoch + 1, frame.rx_boot_u8, value, current_stamp)
            self._states[frame.rx_id] = state
            classification = "rx_reset"
            events = (self._event(frame, "rx_reset", "confirmed", refs),)
        else:
            high_raw = state.high_timer % U32
            difference = _delta(frame.rx_timer_u32_us, high_raw)
            candidate = state.high_timer + difference
            elapsed = _elapsed_us(state.high_stamp, current_stamp)
            crossing = difference > 0 and frame.rx_timer_u32_us < high_raw
            if crossing:
                boundary = (
                    high_raw >= U32 - self.policy.wrap_guard
                    and frame.rx_timer_u32_us <= self.policy.wrap_guard
                )
                # Arrival is only a consistency bound here, never a clock fit.
                timing = (
                    elapsed is not None
                    and elapsed <= self.policy.max_forward_timer_us
                    and difference <= self.policy.max_forward_timer_us
                )
                if boundary and timing:
                    classification = "timer_wrap"
                    events = (self._event(frame, "timer_wrap", "inferred", refs),)
                else:
                    classification, usable, value = "ambiguous", False, None
                    refs += ("reason:timer_wrap_without_consistent_elapsed",)
            elif difference < 0:
                events = (self._event(frame, "timer_regression", "uncertain", refs),)
                if -difference <= self.policy.timer_reorder_us and candidate >= 0:
                    classification = "reorder"
                else:
                    classification, usable, value = "ambiguous", False, None
                    refs += ("reason:timer_regression_outside_grace",)
            if usable:
                value = candidate
                if difference >= 0:
                    state.high_timer, state.high_stamp = candidate, current_stamp
                if state.uncertain:
                    state.segment += 1
                    state.uncertain = False
            elif not state.uncertain:
                state.segment += 1
                state.uncertain = True
        if usable:
            state.seen[fingerprint] = None
            while len(state.seen) > self.policy.max_seen_per_link:
                state.seen.popitem(last=False)
        epoch = f"rx:{frame.rx_id}:epoch:{state.epoch}"
        return EpochResult(
            epoch,
            None,
            f"{epoch}:segment:{state.segment}",
            value,
            classification,
            usable,
            refs,
            events,
        )


@dataclass
class _LinkState:
    rx_epoch: str
    high_seq: int | None = None
    high_timer: int | None = None
    timer_floor: int | None = None
    seen: OrderedDict[int, bytes] = field(default_factory=OrderedDict)
    retired: OrderedDict[bytes, None] = field(default_factory=OrderedDict)
    segment: int = 0
    uncertain: bool = False


@dataclass
class _Restart:
    last_seq: int
    count: int
    refs: list[str]
    timers: dict[int, int]


@dataclass
class _TxState:
    epoch: int = 0
    high_seq: int | None = None
    high_stamp: Stamp | None = None
    pending: _Restart | None = None


class TxSequenceEpochs:
    """Share TX evidence across receivers, but deduplicate each link separately.

    ``operator_boundary`` is explicitly TX-scoped. RX controls use
    :meth:`RxTimerUnwrapper.reset`; ``rx_boot`` is rejected by this class.
    Control stamps must share the selected availability domain with frames.
    A control older than a processed frame is rejected because S03 cannot
    retroactively reclassify that prefix. Same-time controls are accepted before
    the first frame at that timestamp and rejected if submitted afterward.
    """

    def __init__(self, policy: EpochPolicy | None = None):
        self.policy = policy or EpochPolicy()
        self._states = {tx: _TxState() for tx in range(3)}
        self._links: dict[tuple[int, int], _LinkState] = {}
        self._rx_epochs: dict[int, str] = {}
        self._controls: list[EpochEvidence] = []
        self._control_refs: OrderedDict[str, EpochEvidence] = OrderedDict()
        self._last_stamp: dict[int, Stamp] = {}
        self._event_numbers: dict[int, int] = {}

    def observe_control(self, evidence: EpochEvidence) -> None:
        if evidence.kind not in {"tx_start", "tx_reboot", "operator_boundary"}:
            raise ValueError("control.kind: RX controls require RxTimerUnwrapper.reset")
        previous = self._control_refs.get(evidence.source_ref)
        if previous is not None:
            if previous != evidence:
                raise ValueError(
                    "control.source_ref: conflicting evidence for an existing reference"
                )
            return
        last = self._last_stamp.get(evidence.device_id)
        if last is not None:
            if _elapsed_us(last, evidence.available) is None:
                raise ValueError(
                    "control.available: stale control or incompatible clock domain/kind"
                )
            if evidence.available.ns <= last.ns:
                raise ValueError(
                    "control.available: stale control cannot rewrite a processed prefix"
                )
        if len(self._controls) >= self.policy.max_seen_per_link:
            raise ValueError("control queue capacity exceeded")
        self._controls.append(evidence)
        self._control_refs[evidence.source_ref] = evidence
        while len(self._control_refs) > self.policy.max_seen_per_link:
            self._control_refs.popitem(last=False)

    def _event(
        self, tx: int, kind: str, certainty: str, available: Stamp, refs: tuple[str, ...]
    ) -> EpochEvent:
        number = self._event_numbers.get(tx, 0) + 1
        self._event_numbers[tx] = number
        return EpochEvent(
            f"tx:{tx}:event:{number}",
            "tx",
            tx,
            kind,
            certainty,
            available,
            tuple(3 * rx + tx for rx in range(3)),
            refs,
        )

    def _bounded(self, mapping: OrderedDict) -> None:
        while len(mapping) > self.policy.max_seen_per_link:
            mapping.popitem(last=False)

    def _new_tx_epoch(self, tx: int) -> None:
        state = self._states[tx]
        state.epoch += 1
        state.high_seq = None
        state.high_stamp = None
        state.pending = None
        for (rx_id, link_tx), link in self._links.items():
            if link_tx != tx:
                continue
            for fingerprint in link.seen.values():
                link.retired[fingerprint] = None
            self._bounded(link.retired)
            link.seen.clear()
            link.timer_floor = link.high_timer
            link.high_seq = None
            link.segment += 1
            link.uncertain = False

    def _apply_controls(self, frame: WireFrame) -> tuple[EpochEvent, ...]:
        stamp = _stamp(frame)
        due, remaining = [], []
        for control in self._controls:
            if control.device_id != frame.tx_id:
                remaining.append(control)
                continue
            if (control.available.domain, control.available.kind) != (stamp.domain, stamp.kind):
                raise ValueError("control.available: frame uses an incompatible clock domain/kind")
            if control.available.ns > stamp.ns:
                remaining.append(control)
            else:
                due.append(control)
        events = []
        for control in sorted(due, key=lambda evidence: evidence.available.ns):
            self._new_tx_epoch(frame.tx_id)
            events.append(
                self._event(
                    frame.tx_id, "tx_reset", "confirmed", control.available, (control.source_ref,)
                )
            )
        self._controls = remaining
        return tuple(events)

    def _link(self, frame: WireFrame, rx: EpochResult) -> _LinkState:
        if self._rx_epochs.get(frame.rx_id) != rx.rx_epoch:
            # One RX reboot cuts that receiver's three link histories only.
            for key in [key for key in self._links if key[0] == frame.rx_id]:
                del self._links[key]
            self._rx_epochs[frame.rx_id] = rx.rx_epoch
        return self._links.setdefault((frame.rx_id, frame.tx_id), _LinkState(rx.rx_epoch))

    def _result(
        self,
        frame: WireFrame,
        rx: EpochResult,
        link: _LinkState,
        classification: str,
        usable: bool,
        refs: tuple[str, ...],
        events: tuple[EpochEvent, ...],
    ) -> EpochResult:
        uncertain = classification == "ambiguous"
        if uncertain and not link.uncertain:
            link.segment += 1
            link.uncertain = True
        elif usable and link.uncertain:
            link.segment += 1
            link.uncertain = False
        epoch = f"tx:{frame.tx_id}:epoch:{self._states[frame.tx_id].epoch}"
        segment = f"{rx.link_segment}:{epoch}:link:{frame.link_index}:segment:{link.segment}"
        return EpochResult(
            rx.rx_epoch,
            None if uncertain else epoch,
            segment,
            rx.timer_unwrapped_us,
            classification,
            usable,
            tuple(dict.fromkeys(rx.evidence_refs + refs)),
            rx.events + events,
        )

    def _restart_evidence(
        self, frame: WireFrame, rx: EpochResult, link: _LinkState
    ) -> tuple[str, ...] | None:
        state = self._states[frame.tx_id]
        timer = rx.timer_unwrapped_us
        # A new receiver with no prior same-TX timer cannot prove progress.
        if timer is None or link.high_timer is None or timer <= link.high_timer:
            return None
        pending = state.pending
        reference = _reference(frame)
        if pending is None:
            if frame.seq_u32 > self.policy.restart_max_initial_seq:
                return None
            state.pending = _Restart(frame.seq_u32, 1, [reference], {frame.rx_id: timer})
            return None
        previous_timer = pending.timers.get(frame.rx_id)
        if previous_timer is not None and timer <= previous_timer:
            return None
        pending.timers[frame.rx_id] = timer
        step = frame.seq_u32 - pending.last_seq
        if step == 0:
            return None  # Same TX packet at three RXs is only one sequence step.
        if not 0 < step <= self.policy.restart_max_step:
            state.pending = None
            if frame.seq_u32 <= self.policy.restart_max_initial_seq:
                state.pending = _Restart(frame.seq_u32, 1, [reference], {frame.rx_id: timer})
            return None
        pending.count += 1
        pending.last_seq = frame.seq_u32
        pending.refs.append(reference)
        if pending.count >= self.policy.restart_confirm_packets:
            return tuple(pending.refs)
        return None

    def classify(self, frame: WireFrame, rx: EpochResult) -> EpochResult:
        link = self._link(frame, rx)
        events = self._apply_controls(frame)
        state = self._states[frame.tx_id]
        stamp = _stamp(frame)
        previous_stamp = self._last_stamp.get(frame.tx_id)
        if previous_stamp is None or _elapsed_us(previous_stamp, stamp) is not None:
            self._last_stamp[frame.tx_id] = stamp
        refs: tuple[str, ...] = (_reference(frame),)
        if not rx.usable or rx.timer_unwrapped_us is None:
            return self._result(
                frame, rx, link, "ambiguous", False, refs + ("reason:rx_timer_uncertain",), events
            )
        fingerprint = hashlib.sha256(frame.wire_bytes).digest()
        if fingerprint in link.retired:
            return self._result(
                frame, rx, link, "ambiguous", False, refs + ("reason:prior_epoch_replay",), events
            )
        if link.timer_floor is not None and rx.timer_unwrapped_us < link.timer_floor:
            return self._result(
                frame,
                rx,
                link,
                "ambiguous",
                False,
                refs + ("reason:timer_before_tx_boundary",),
                events,
            )
        candidate = (
            frame.seq_u32
            if state.high_seq is None
            else state.high_seq + _delta(frame.seq_u32, state.high_seq % U32)
        )
        old_fingerprint = link.seen.get(candidate)
        if old_fingerprint == fingerprint:
            return self._result(frame, rx, link, "duplicate", False, refs, events)
        high = link.high_seq if link.high_seq is not None else state.high_seq
        behind = high is not None and candidate < high
        timer_forward = link.high_timer is not None and rx.timer_unwrapped_us > link.high_timer
        # Once a low-sequence branch is isolated, its evidence may progress past
        # the former high sequence. Otherwise a restart from head 1 or 2 would
        # silently merge into the previous epoch before reaching three samples.
        continuing_restart = (
            state.pending is not None
            and 0 <= frame.seq_u32 - state.pending.last_seq <= self.policy.restart_max_step
        )
        low_head_conflict = (
            old_fingerprint is not None
            and candidate == high
            and frame.seq_u32 <= self.policy.restart_max_initial_seq
        )
        if timer_forward and (behind or continuing_restart or low_head_conflict):
            evidence = self._restart_evidence(frame, rx, link)
            if evidence is not None:
                self._new_tx_epoch(frame.tx_id)
                candidate = frame.seq_u32
                events += (self._event(frame.tx_id, "tx_reset", "inferred", stamp, evidence),)
                refs += evidence
                old_fingerprint = None
                behind = False
            else:
                reason = (
                    "sequence_payload_conflict"
                    if old_fingerprint is not None
                    else "sequence_reset_unconfirmed"
                )
                return self._result(
                    frame, rx, link, "ambiguous", False, refs + (f"reason:{reason}",), events
                )
        if old_fingerprint is not None:
            return self._result(
                frame,
                rx,
                link,
                "ambiguous",
                False,
                refs + ("reason:sequence_payload_conflict",),
                events,
            )
        if candidate < 0:
            return self._result(
                frame,
                rx,
                link,
                "ambiguous",
                False,
                refs + ("reason:sequence_before_epoch",),
                events,
            )
        if behind and high - candidate > self.policy.seq_reorder_packets:
            return self._result(
                frame,
                rx,
                link,
                "ambiguous",
                False,
                refs + ("reason:sequence_outside_reorder_grace",),
                events,
            )
        if (
            link.high_seq is not None
            and candidate > link.high_seq
            and link.high_timer is not None
            and rx.timer_unwrapped_us < link.high_timer
        ):
            return self._result(
                frame,
                rx,
                link,
                "ambiguous",
                False,
                refs + ("reason:sequence_timer_conflict",),
                events,
            )
        crossing = state.high_seq is not None and candidate // U32 > state.high_seq // U32
        if crossing:
            high_raw = state.high_seq % U32
            elapsed = _elapsed_us(state.high_stamp, stamp) if state.high_stamp else None
            boundary = (
                high_raw >= U32 - self.policy.wrap_guard and frame.seq_u32 <= self.policy.wrap_guard
            )
            timing = elapsed is not None and elapsed <= self.policy.max_forward_timer_us
            if (
                not boundary
                or not timing
                or (link.high_timer is not None and rx.timer_unwrapped_us < link.high_timer)
            ):
                return self._result(
                    frame,
                    rx,
                    link,
                    "ambiguous",
                    False,
                    refs + ("reason:sequence_wrap_without_consistent_elapsed",),
                    events,
                )
            events += (self._event(frame.tx_id, "seq_wrap", "inferred", stamp, refs),)
        if state.high_seq is None or candidate > state.high_seq:
            state.high_seq, state.high_stamp = candidate, stamp
            state.pending = None
        if link.high_seq is None or candidate > link.high_seq:
            link.high_seq = candidate
            link.high_timer = rx.timer_unwrapped_us
        link.seen[candidate] = fingerprint
        self._bounded(link.seen)
        kinds = {event.kind for event in events}
        if "tx_reset" in kinds:
            classification = "tx_reset"
        elif "seq_wrap" in kinds:
            classification = "seq_wrap"
        elif behind or rx.classification == "reorder":
            classification = "reorder"
        else:
            classification = rx.classification
        return self._result(frame, rx, link, classification, True, refs, events)
