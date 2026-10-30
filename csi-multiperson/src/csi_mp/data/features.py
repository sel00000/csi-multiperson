"""S07 log-amplitude and causal temporal differences, before normalization.

Use one builder per grid/session. The grid owns arrival admission, duplicate
filtering and device-scoped boundaries. This provider owns per-link forward
baselines and immutable feature snapshots; it never fits statistics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from csi_mp.ingest.events import TimedObservation

from .grid import GridPolicy, SlotFeatures
from .schema import FEATURE_SPEC_ID, ContractError, _integer

if TYPE_CHECKING:
    from csi_mp.config import AppConfig


@dataclass(frozen=True)
class FeatureSpec:
    """Fixed v1 formula plus the declared, inclusive differentiation gap cap.

    The logarithm uses an offset of exactly one (log1p), with no additional
    epsilon or clipping. A zero gap cap disables all temporal differences.
    Persist to_dict() and spec_hash alongside the grid policy in provenance.
    """

    max_delta_gap_ns: int = 30_000_000
    feature_id: str = field(default=FEATURE_SPEC_ID, init=False)

    def __post_init__(self) -> None:
        _integer(
            self.max_delta_gap_ns,
            "FeatureSpec.max_delta_gap_ns",
            upper=np.iinfo(np.int64).max,
        )
        object.__setattr__(self, "max_delta_gap_ns", int(self.max_delta_gap_ns))

    def to_dict(self) -> dict:
        return {
            "feature_id": self.feature_id,
            "channels": ["amp", "delta_amp"],
            "amp": {
                "formula": "log1p(hypot(I,Q))",
                "log_base": "e",
                "log_offset": 1.0,
                "additional_epsilon": 0.0,
                "zero_iq_valid": True,
            },
            "delta_amp": {
                "formula": "(amp_current - amp_previous) / (dt_ns / 1e9)",
                "time_basis": "acquisition_ns",
                "max_gap_ns": str(self.max_delta_gap_ns),
                "valid_interval": "0 < dt_ns <= max_gap_ns",
                "baseline": "last_strictly_forward_observation_in_same_segment",
                "segment_fields": [
                    "rx_epoch",
                    "tx_epoch",
                    "link_segment",
                    "clock_segment_id",
                    "clock_domain",
                    "clock_kind",
                ],
                "held_valid": False,
            },
            "input_dtype": "int8",
            "compute_dtype": "float64",
            "output_dtype": "float32",
            "mask_dtype": "bool",
            "mask_true": "usable_feature",
            "invalid_value": 0.0,
            "link_order": "3*rx_id+tx_id",
            "tone_order": "wire_order_56",
            "iq_order": ["I", "Q"],
            "normalization": "none",
        }

    @property
    def spec_hash(self) -> str:
        payload = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class _Baseline:
    acquisition_ns: int
    segment: tuple
    amp: NDArray[np.float64]


class FeatureBuilder:
    """Concrete provider for data.grid.FeatureBuilder's injection protocol.

    Call on_observation only for admitted, usable observations, then immediately
    snapshot values_for_slot(id, True), as CausalGridBuilder does. At most the
    latest processed snapshot for each of nine links is retained. Older IDs are
    rejected; callers keep returned SlotFeatures for longer retention/holds.

    A reordered or equal-time observation has valid amplitude but no delta and
    cannot replace the forward baseline. A changed segment or a long gap starts
    a new baseline. reset clears selected baselines without modifying snapshots.
    Full-session reset uses a new builder, together with a new grid.
    """

    def __init__(self, spec: FeatureSpec | None = None):
        if spec is not None and not isinstance(spec, FeatureSpec):
            raise ContractError("features.spec: expected FeatureSpec")
        self._spec = spec if spec is not None else FeatureSpec()
        self._previous: dict[int, _Baseline] = {}
        self._latest: dict[int, tuple[str, SlotFeatures]] = {}

    @property
    def spec(self) -> FeatureSpec:
        return self._spec

    @property
    def feature_id(self) -> str:
        return self.spec.feature_id

    @classmethod
    def from_config(cls, config: AppConfig) -> FeatureBuilder:
        """Use the same gap limit as the configured grid's bounded hold policy."""
        policy = GridPolicy.from_config(config)
        return cls(FeatureSpec(max_delta_gap_ns=policy.max_hold_age_ns))

    def on_observation(self, observation: TimedObservation) -> None:
        if not isinstance(observation, TimedObservation) or not observation.usable:
            raise ContractError("features.observation: expected usable TimedObservation")
        if not isinstance(observation.observation_id, str) or not observation.observation_id:
            raise ContractError("features.observation_id: required")
        link = observation.frame.link_index
        segment = (
            observation.epochs.rx_epoch,
            observation.epochs.tx_epoch,
            observation.epochs.link_segment,
            observation.clock_segment_id,
            observation.clock_domain,
            observation.clock_kind,
        )
        # Cast before magnitude arithmetic: squaring signed int8 would overflow.
        iq = observation.frame.iq.astype(np.float64)
        amp = np.log1p(np.hypot(iq[:, 0], iq[:, 1]))
        values = np.zeros((56, 2), dtype=np.float32)
        valid = np.zeros((56, 2), dtype=np.bool_)
        values[:, 0] = amp
        valid[:, 0] = True
        previous = self._previous.get(link)
        same_segment = previous is not None and previous.segment == segment
        # Subtract integers BEFORE float conversion, including for Unix-sized ns.
        dt_ns = observation.acquisition_ns - previous.acquisition_ns if same_segment else 0
        if same_segment and 0 < dt_ns <= self.spec.max_delta_gap_ns:
            values[:, 1] = (amp - previous.amp) / (dt_ns / 1_000_000_000)
            valid[:, 1] = True
        snapshot = SlotFeatures(values, valid)
        if not same_segment or dt_ns > 0:
            self._previous[link] = _Baseline(observation.acquisition_ns, segment, amp)
        self._latest[link] = (observation.observation_id, snapshot)

    def values_for_slot(self, observation_id: str, is_new: bool) -> SlotFeatures:
        if not isinstance(is_new, (bool, np.bool_)):
            raise ContractError("features.is_new: expected boolean")
        if not isinstance(observation_id, str) or not observation_id:
            raise ContractError("features.observation_id: required")
        snapshot = next(
            (value for key, value in self._latest.values() if key == observation_id), None
        )
        if snapshot is None:
            raise ContractError("features.observation_id: unknown or evicted snapshot")
        if is_new:
            return snapshot
        values = snapshot.raw_x.copy()
        valid = snapshot.feature_valid.copy()
        values[:, 1] = 0
        valid[:, 1] = False
        return SlotFeatures(values, valid)

    def reset(self, links: tuple[int, ...], reason: str) -> None:
        links = tuple(links)
        for link in links:
            _integer(link, "features.reset.links", upper=8)
        if not links or len(set(links)) != len(links):
            raise ContractError("features.reset.links: nonempty distinct links required")
        if not isinstance(reason, str) or not reason:
            raise ContractError("features.reset.reason: required")
        for link in links:
            self._previous.pop(link, None)
