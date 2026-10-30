"""Causal, receiver-local RX microseconds to a declared common host clock.

Only the current usable forward observation enters a bounded fit. All time
arithmetic is integer/rational, including centering before the affine fit.
The intercept is a lower envelope of bridge arrival residuals: acquisition
remains an estimate with unknown fixed transport latency. No whole-session fit,
lookahead, retrospective timestamp rewrite, or monotonic +1ns clamp is used.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from fractions import Fraction
import hashlib
import json

from csi_mp.data.schema import ContractError, EpochResult, Stamp, WireFrame
from csi_mp.ingest.events import ClockModel, ClockTransform, TimedObservation, integer


def _json(value: dict) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


@dataclass(frozen=True)
class ClockPolicy:
    """Development parameters, not calibrated hardware accuracy guarantees."""

    policy_id: str = "clock_fixture_v1"
    window_size: int = 64
    warmup_samples: int = 3
    min_span_us: int = 10_000
    max_drift_ppm: int = 1000
    legacy_delay_ns: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.policy_id, str) or not self.policy_id:
            raise ContractError("ClockPolicy.policy_id: required")
        for name in ("window_size", "warmup_samples", "min_span_us"):
            integer(getattr(self, name), f"ClockPolicy.{name}", minimum=1)
        for name in ("max_drift_ppm", "legacy_delay_ns"):
            integer(getattr(self, name), f"ClockPolicy.{name}")
        if not 2 <= self.warmup_samples <= self.window_size:
            raise ContractError("ClockPolicy.warmup_samples: expected 2 <= samples <= window_size")
        if self.max_drift_ppm >= 1_000_000:
            raise ContractError("ClockPolicy.max_drift_ppm: must be below 1000000")


@dataclass
class _State:
    rx_epoch: str
    segment_id: str
    samples: list[tuple[int, int]]
    sample_errors: list[int | None]
    sample_transforms: list[str | None]
    model: ClockModel
    high_acquisition_ns: int


class OnlineClockMapper:
    """Observe already classified frames in arrival order, emitting immediately.

    ``reference`` names the common output clock; its ns value is an audit anchor,
    not a value subtracted from output timestamps. Without it, the first frame's
    inference clock (or legacy bridge clock) establishes the common clock.
    Different domains/kinds require an explicit ClockTransform known by arrival.
    Duplicate/ambiguous frames never train the fit. Reordered timers use the
    existing model without changing its samples or forward watermark.
    """

    def __init__(
        self,
        *,
        reference: Stamp | None = None,
        policy: ClockPolicy | None = None,
        synthetic: bool = False,
        transforms: tuple[ClockTransform, ...] = (),
    ):
        if reference is not None and not isinstance(reference, Stamp):
            raise ContractError("clock.reference: expected Stamp")
        if not isinstance(synthetic, bool):
            raise ContractError("clock.synthetic: expected boolean")
        self.policy = policy or ClockPolicy()
        self.reference = reference
        self.synthetic = synthetic
        self.transforms = tuple(transforms)
        keys = [
            (
                t.source_reference.domain,
                t.source_reference.kind,
                t.target_reference.domain,
                t.target_reference.kind,
            )
            for t in self.transforms
        ]
        if len(set(keys)) != len(keys):
            raise ContractError("clock.transforms: duplicate source/target calibration")
        if len({t.transform_id for t in self.transforms}) != len(self.transforms):
            raise ContractError("clock.transforms: duplicate transform_id")
        self._states: dict[int, _State] = {}
        self._generations: dict[int, int] = {}
        self._arrivals: dict[int, tuple[int, int]] = {}

    def _map(self, stamp: Stamp, reference: Stamp) -> tuple[int, int | None, ClockTransform | None]:
        if (stamp.domain, stamp.kind) == (reference.domain, reference.kind):
            return stamp.ns, 0, None
        for transform in self.transforms:
            if (stamp.domain, stamp.kind) == (
                transform.source_reference.domain,
                transform.source_reference.kind,
            ) and (reference.domain, reference.kind) == (
                transform.target_reference.domain,
                transform.target_reference.kind,
            ):
                value = transform.map_ns(stamp)
                if value < 0:
                    raise ContractError("clock.transform: mapped timestamp is negative")
                return value, transform.error_ns, transform
        raise ContractError(
            "clock.domain: an explicit clock transform is required for different domains/kinds"
        )

    def _fit(
        self,
        samples: list[tuple[int, int]],
        sample_errors: list[int | None],
        sample_transforms: list[str | None],
        rx_epoch: str,
        segment: str,
        count: int,
    ) -> ClockModel:
        origin_timer, origin_host = samples[0]
        xs = [timer - origin_timer for timer, _ in samples]
        ys = [host - origin_host for _, host in samples]
        n = len(samples)
        denominator = n * sum(x * x for x in xs) - sum(xs) ** 2
        rate = Fraction(1000)
        if denominator:
            rate = Fraction(n * sum(x * y for x, y in zip(xs, ys)) - sum(xs) * sum(ys), denominator)
        low = Fraction(1000 * (1_000_000 - self.policy.max_drift_ppm), 1_000_000)
        high = Fraction(1000 * (1_000_000 + self.policy.max_drift_ppm), 1_000_000)
        limited = not low <= rate <= high
        rate = min(high, max(low, rate))
        projected = [x * rate.numerator // rate.denominator for x in xs]
        reference_ns = min(host - delta for (_, host), delta in zip(samples, projected))
        span = xs[-1]
        ready = n >= self.policy.warmup_samples and span >= self.policy.min_span_us
        residual = max(host - reference_ns - delta for (_, host), delta in zip(samples, projected))
        calibration_error = None if None in sample_errors else max(sample_errors)
        transform_ids = tuple(sorted({t for t in sample_transforms if t is not None}))
        model = ClockModel(
            "",
            rx_epoch,
            segment,
            origin_timer,
            reference_ns,
            rate.numerator,
            rate.denominator,
            count,
            n,
            span,
            "ready" if ready else "warming_up",
            residual if ready else None,
            limited,
            calibration_error,
            transform_ids,
        )
        identity = {
            "policy": asdict(self.policy),
            "model": model.to_dict(),
            "reference": self.reference.to_dict(),
        }
        return replace(model, model_id="clock:" + hashlib.sha256(_json(identity)).hexdigest())

    def observe(self, frame: WireFrame, epochs: EpochResult) -> TimedObservation:
        reference = self.reference or frame.infer_available or frame.bridge_complete
        bridge_ns, bridge_error, bridge_transform = self._map(frame.bridge_complete, reference)
        selected = frame.infer_available or frame.bridge_complete
        availability_ns, _, arrival_transform = self._map(selected, reference)
        if frame.infer_available is None:
            availability_ns += self.policy.legacy_delay_ns
        transforms = tuple(t for t in (bridge_transform, arrival_transform) if t is not None)
        if any(t.available.ns > availability_ns for t in transforms):
            raise ContractError(
                "clock.transform.available: future calibration cannot time this prefix"
            )
        self.reference = reference
        basis = "synthetic" if self.synthetic else frame.availability_basis
        # Whole-file SHA changes on append. It remains in frame.spans as provenance,
        # but cannot participate in causal model or observation identity.
        identity = {
            "source": [
                (s.source_id, s.chunk_index, s.file_offset, s.byte_count) for s in frame.spans
            ],
            "wire": frame.wire_bytes.hex(),
            "rx": frame.rx_id,
            "tx": frame.tx_id,
        }
        observation_id = "observation:" + hashlib.sha256(_json(identity)).hexdigest()

        def emit(model=None, acquisition=None, reason=None):
            error = None
            if (
                model is not None
                and model.residual_ns is not None
                and model.calibration_error_ns is not None
            ):
                error = model.residual_ns + model.calibration_error_ns
            model_transforms = () if model is None else model.transform_ids
            return TimedObservation(
                observation_id,
                frame,
                epochs,
                acquisition,
                availability_ns,
                basis,
                reference.domain,
                reference.kind,
                reference,
                model,
                error,
                "excluded" if reason else model.status,
                reason,
                tuple(sorted(set(model_transforms) | {t.transform_id for t in transforms})),
                self.policy.legacy_delay_ns if frame.infer_available is None else 0,
            )

        state = self._states.get(frame.rx_id)
        if bridge_ns > availability_ns:
            return emit(reason="availability_before_bridge")
        # Track delivery independently of acquisition/epoch usability: even an
        # excluded duplicate is evidence of the host clock at that arrival.
        previous_arrival = self._arrivals.get(frame.rx_id)
        self._arrivals[frame.rx_id] = (bridge_ns, availability_ns)
        if previous_arrival is not None:
            reason = None
            if availability_ns < previous_arrival[1]:
                reason = "availability_clock_regression"
            elif bridge_ns < previous_arrival[0]:
                reason = "bridge_clock_regression"
            if reason:
                self._states.pop(frame.rx_id, None)
                return emit(model=None if state is None else state.model, reason=reason)
        if not epochs.usable or epochs.timer_unwrapped_us is None:
            return emit(reason="epoch_" + epochs.classification)
        if state is not None and state.rx_epoch != epochs.rx_epoch:
            state = None  # RX reset is local; a TX epoch change does not enter this branch.
        timer = epochs.timer_unwrapped_us
        if state is not None and timer <= state.samples[-1][0]:
            acquisition = state.model.estimate(timer)
            if acquisition < 0:
                return emit(model=state.model, reason="acquisition_before_clock_origin")
            if acquisition > availability_ns:
                return emit(model=state.model, reason="acquisition_after_availability")
            return emit(state.model, acquisition)
        transform_id = None if bridge_transform is None else bridge_transform.transform_id
        if state is None:
            generation = self._generations.get(frame.rx_id, 0) + 1
            segment = f"{epochs.rx_epoch}:clock:{generation}"
            samples, count = [(timer, bridge_ns)], 1
            sample_errors, sample_transforms = [bridge_error], [transform_id]
        else:
            generation = self._generations[frame.rx_id]
            segment = state.segment_id
            samples = (state.samples + [(timer, bridge_ns)])[-self.policy.window_size :]
            sample_errors = (state.sample_errors + [bridge_error])[-self.policy.window_size :]
            sample_transforms = (state.sample_transforms + [transform_id])[
                -self.policy.window_size :
            ]
            count = state.model.sample_count + 1
        model = self._fit(
            samples, sample_errors, sample_transforms, epochs.rx_epoch, segment, count
        )
        acquisition = model.estimate(timer)
        if acquisition < 0:
            return emit(model=model, reason="acquisition_before_clock_origin")
        if state is not None and acquisition < state.high_acquisition_ns:
            del self._states[frame.rx_id]
            return emit(model=model, reason="clock_fit_regression")
        self._states[frame.rx_id] = _State(
            epochs.rx_epoch, segment, samples, sample_errors, sample_transforms, model, acquisition
        )
        self._generations[frame.rx_id] = generation
        return emit(model, acquisition)

    def reset(self, rx_epoch: str) -> None:
        """Invalidate only this RX epoch's fit; published observations stay intact."""
        if not isinstance(rx_epoch, str) or not rx_epoch:
            raise ContractError("clock.reset.rx_epoch: required")
        for rx, state in tuple(self._states.items()):
            if state.rx_epoch == rx_epoch:
                del self._states[rx]

    def snapshot(self) -> bytes:
        """Deterministic, bounded audit state from the processed prefix only."""
        return _json(
            {
                "schema_id": "csi_mp.online_clock.v1",
                "policy": asdict(self.policy),
                "reference": None if self.reference is None else self.reference.to_dict(),
                "synthetic": self.synthetic,
                "transforms": [t.to_dict() for t in self.transforms],
                "generations": self._generations,
                "arrivals": {
                    str(rx): {"bridge_ns": str(bridge), "availability_ns": str(available)}
                    for rx, (bridge, available) in sorted(self._arrivals.items())
                },
                "receivers": {
                    str(rx): {
                        "rx_epoch": state.rx_epoch,
                        "segment_id": state.segment_id,
                        "samples": [[str(timer), str(host)] for timer, host in state.samples],
                        "sample_errors_ns": [
                            None if e is None else str(e) for e in state.sample_errors
                        ],
                        "sample_transform_ids": state.sample_transforms,
                        "model": state.model.to_dict(),
                        "high_acquisition_ns": str(state.high_acquisition_ns),
                    }
                    for rx, state in sorted(self._states.items())
                },
            }
        )
