"""Ingest counters and bounded rolling signal samples; no inferred exposure."""

from __future__ import annotations

import math
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Mapping

from csi_mp.data.schema import EpochEvent, EpochResult, ParseIssue, RawChunk, Stamp, WireFrame

from .registry import (
    INGEST_V1_KEYS,
    MetricRegistry,
    availability_reason,
    content_hash,
    default_registry,
)
from .schema import METRICS_SCHEMA_VERSION, MetricValue, MetricsSnapshot

PROVIDER_HASH = content_hash({"provider": "csi_mp.telemetry.IngestMetrics", "version": 1})


@dataclass
class _Series:
    key: str
    labels: dict[str, str]
    total: int = 0
    count: int = 0
    events: deque = field(default_factory=deque)
    exclusions: Counter = field(default_factory=Counter)
    exclusion_events: deque = field(default_factory=deque)
    unclocked: bool = False
    first_event_ns: int | None = None


class IngestMetrics:
    """Call each on_* hook once per original item, in arrival/reader order.

    ``on_raw_chunk`` is the legacy raw-record adapter: one call equals one complete
    CSIRAW01 record. A future serial source must use a dedicated adapter because it
    has no raw records. All time-bearing items must share one explicit clock.
    Rates/gaps stay unavailable until S04 supplies capture exposure/mature intervals.
    """

    def __init__(
        self,
        run_id: str,
        session_id: str,
        stream_epoch: str = "ingest-0",
        *,
        registry: MetricRegistry | None = None,
        selected=None,
        selection_id: str = "ingest_v1",
        profile_id: str = "custom",
        window_s: float = 10.0,
        clock_domain: str | None = None,
        clock_kind: str | None = None,
        provenance: Mapping | None = None,
        counter_epoch: str | None = None,
    ) -> None:
        if not math.isfinite(window_s) or window_s <= 0:
            raise ValueError("window_s: must be positive and finite")
        if (clock_domain is None) != (clock_kind is None):
            raise ValueError("clock_domain and clock_kind must be supplied together")
        self.registry = registry or default_registry()
        self.selected = frozenset(
            self.registry.validate_selection(INGEST_V1_KEYS if selected is None else selected)
        )
        self.run_id, self.session_id, self.stream_epoch = run_id, session_id, stream_epoch
        self.selection_id, self.profile_id = selection_id, profile_id
        self.window_ns = int(window_s * 1_000_000_000)
        self.clock_domain, self.clock_kind = clock_domain, clock_kind
        self.counter_epoch = counter_epoch or f"{run_id}:{session_id}:{stream_epoch}:counter-0"
        self.selection_hash = content_hash(
            {
                "profile_id": profile_id,
                "selection_id": selection_id,
                "keys": sorted(self.selected),
                "window_ns": str(self.window_ns),
            }
        )
        self.provenance = dict(provenance or {})
        self.provenance_id = content_hash(
            {
                "run_id": run_id,
                "session_id": session_id,
                "spec_hash": self.registry.spec_hash,
                "selection_hash": self.selection_hash,
                "provider_hash": PROVIDER_HASH,
                **self.provenance,
            }
        )
        self._series: dict[tuple, _Series] = {}
        self._last_ns: int | None = None
        self._first_ns: int | None = None
        self._max_ns: int | None = None
        self._snapshot_ns: int | None = None
        self._event_sequence = 0
        self._discontinuous = False
        self._sources: set[str] = set()
        self._epoch_connected = False
        self._seen_events: set[str] = set()
        self._uncertain_segments: set[tuple[int, int, str]] = set()
        self._crc_candidates: set[tuple] = set()
        self.diagnostics: list[dict] = []

    def _touch(self, stamp: Stamp) -> None:
        if self.clock_domain is None:
            self.clock_domain, self.clock_kind = stamp.domain, stamp.kind
        if (stamp.domain, stamp.kind) != (self.clock_domain, self.clock_kind):
            raise ValueError(
                "metrics clock mismatch: split snapshots by domain/kind; a verified S04 mapping is required"
            )
        if self._last_ns is not None and stamp.ns < self._last_ns:
            self._discontinuous = True
            self.diagnostics.append(
                {
                    "kind": "time_discontinuity",
                    "previous_ns": str(self._last_ns),
                    "current_ns": str(stamp.ns),
                    "clock_domain": stamp.domain,
                }
            )
            for series in self._series.values():
                series.events.clear()
                series.exclusion_events.clear()
        self._first_ns = stamp.ns if self._first_ns is None else self._first_ns
        self._max_ns = stamp.ns if self._max_ns is None else max(self._max_ns, stamp.ns)
        self._last_ns = stamp.ns

    def _get(self, key: str, labels: Mapping[str, object]) -> _Series:
        normalized = {k: str(v) for k, v in labels.items()}
        spec = self.registry[key]
        if set(normalized) != set(spec.labels):
            raise ValueError(f"{key}: expected labels {spec.labels}, got {tuple(normalized)}")
        identity = key, tuple(sorted(normalized.items()))
        if identity not in self._series:
            self._series[identity] = _Series(key, normalized)
        return self._series[identity]

    def _prune(self, series: _Series, as_of_ns: int) -> None:
        begin = as_of_ns - self.window_ns
        while series.events and series.events[0][0] < begin:
            series.events.popleft()
        while series.exclusion_events and series.exclusion_events[0][0] < begin:
            series.exclusion_events.popleft()

    def increment(
        self, key: str, *, labels: Mapping[str, object], amount: int = 1, stamp: Stamp | None = None
    ) -> None:
        if isinstance(amount, bool) or not isinstance(amount, int) or amount < 0:
            raise ValueError("counter amount: expected a nonnegative integer")
        if stamp is not None:
            self._touch(stamp)
        series = self._get(key, labels)
        if "sum" not in self.registry[key].reductions:
            raise ValueError(f"{key}: increment requires a counter metric")
        if key not in self.selected:
            return
        series.total += amount
        series.count += 1
        if self._last_ns is None:
            series.unclocked = True
        elif not self._discontinuous:
            series.events.append((self._last_ns, amount))
            self._prune(series, self._last_ns)

    def observe(
        self, key: str, value: float, *, labels: Mapping[str, object], stamp: Stamp | None = None
    ) -> None:
        if not math.isfinite(value):
            raise ValueError(f"{key}: sample must be finite")
        if stamp is not None:
            self._touch(stamp)
        series = self._get(key, labels)
        if not set(self.registry[key].reductions) & {"p5", "p50", "p95"}:
            raise ValueError(f"{key}: observe requires a distribution metric")
        if key not in self.selected:
            return
        series.count += 1
        if self._last_ns is None:
            series.unclocked = True
        elif not self._discontinuous:
            series.events.append((self._last_ns, value))
            self._prune(series, self._last_ns)

    def _exclude(self, key: str, labels: Mapping[str, object], reason: str) -> None:
        series = self._get(key, labels)
        if key not in self.selected:
            return
        series.exclusions[reason] += 1
        if self._last_ns is not None and not self._discontinuous:
            series.exclusion_events.append((self._last_ns, reason))
            self._prune(series, self._last_ns)

    def _source(self, source_id: str) -> None:
        if source_id in self._sources:
            return
        self._sources.add(source_id)
        labels = {"source_id": source_id}
        for key in INGEST_V1_KEYS:
            if key.startswith("source.") and self.registry[key].labels == ("source_id",):
                self._get(key, labels)
        for reason in (
            "rx_id_out_of_range",
            "tx_id_out_of_range",
            "sc_count_mismatch",
            "source_rx_mismatch",
        ):
            self._get("source.semantic_reject_count", {**labels, "reason": reason})
        for reason in ("junk", "resync"):
            self._get("source.discarded_bytes_count", {**labels, "reason": reason})
        for layer in ("raw_record", "wire_frame"):
            self._get("source.truncated_tail_bytes", {**labels, "layer": layer})

    def connect_source(self, source_id: str) -> None:
        """Declare an active CSIRAW01 provider, including a valid empty source.

        Connection initializes measured zero counters without recording an event,
        sample, source timestamp or exposure. Call only once the source format
        has been accepted; a missing provider must retain ``not_measured``.
        """
        if not isinstance(source_id, str) or not source_id:
            raise ValueError("source_id: expected a nonempty string")
        self._source(source_id)

    def on_raw_chunk(self, chunk: RawChunk) -> None:
        self._touch(chunk.infer_available or chunk.bridge_read)
        self._source(chunk.span.source_id)
        labels = {"source_id": chunk.span.source_id}
        self.increment("source.bytes_received_count", labels=labels, amount=len(chunk.payload))
        self.increment("source.raw_record_count", labels=labels)

    def on_parse_issue(self, issue: ParseIssue) -> None:
        source_ids = {span.source_id for span in issue.spans}
        if len(source_ids) > 1:
            raise ValueError("ParseIssue spans: a source counter cannot mix source IDs")
        source_id = next(iter(source_ids), "unknown")
        self._source(source_id)
        labels = {"source_id": source_id}
        if issue.kind == "crc":
            candidate = (
                ()
                if not issue.spans
                else (source_id, issue.spans[0].source_sha256, issue.spans[0].file_offset)
            )
            if not issue.spans or candidate not in self._crc_candidates:
                self._crc_candidates.add(candidate)
                self.increment("source.crc_error_count", labels=labels)
        elif issue.kind == "semantic":
            self.increment(
                "source.semantic_reject_count", labels={**labels, "reason": issue.detail}
            )
        elif issue.kind == "discarded_bytes":
            self.increment(
                "source.discarded_bytes_count",
                labels={**labels, "reason": issue.detail},
                amount=issue.byte_count,
            )
        elif issue.kind == "truncated":
            self.increment(
                "source.truncated_tail_bytes",
                labels={**labels, "layer": issue.layer},
                amount=issue.byte_count,
            )
        elif issue.kind == "time_discontinuity":
            self._discontinuous = True
        self.diagnostics.append(
            {
                "kind": issue.kind,
                "layer": issue.layer,
                "source_id": source_id,
                "detail": issue.detail,
                "byte_count": issue.byte_count,
            }
        )

    def on_frame(self, frame: WireFrame) -> None:
        self._touch(frame.infer_available or frame.bridge_complete)
        source_ids = {span.source_id for span in frame.spans}
        if len(source_ids) != 1:
            raise ValueError("WireFrame spans: one trusted source ID is required")
        source_id = next(iter(source_ids))
        self._source(source_id)
        self.increment("source.valid_frame_count", labels={"source_id": source_id})

    def _connect_epochs(self) -> None:
        if self._epoch_connected:
            return
        self._epoch_connected = True
        for rx_id in range(3):
            for tx_id in range(3):
                labels = {"rx_id": rx_id, "tx_id": tx_id}
                for key in INGEST_V1_KEYS:
                    if key.startswith("link."):
                        extra = (
                            {"reason": "ambiguous"}
                            if key == "link.epoch_excluded_frame_count"
                            else {}
                        )
                        self._get(key, {**labels, **extra})
        for scope, kinds in (
            ("rx", ("rx_reset", "timer_wrap", "timer_regression")),
            ("tx", ("tx_reset", "seq_wrap")),
        ):
            for device_id in range(3):
                for kind in kinds:
                    for certainty in ("confirmed", "inferred", "uncertain"):
                        self._get(
                            "epoch.event_count",
                            {
                                "scope": scope,
                                "device_id": device_id,
                                "kind": kind,
                                "certainty": certainty,
                            },
                        )

    def connect_epochs(self) -> None:
        """Declare the enabled epoch provider without inventing frame samples."""
        self._connect_epochs()

    def _epoch_event(self, event: EpochEvent) -> None:
        if event.event_id in self._seen_events:
            return
        same_clock = (event.available.domain, event.available.kind) == (
            self.clock_domain,
            self.clock_kind,
        )
        if same_clock and self._last_ns is not None and event.available.ns > self._last_ns:
            raise ValueError(
                "EpochEvent.available: future evidence cannot be counted in the current prefix"
            )
        self._seen_events.add(event.event_id)
        series = self._get(
            "epoch.event_count",
            {
                "scope": event.scope,
                "device_id": event.device_id,
                "kind": event.kind,
                "certainty": event.certainty,
            },
        )
        if "epoch.event_count" not in self.selected:
            return
        series.total += 1
        series.count += 1
        if not same_clock:
            # Counting a received device event needs no clock conversion. Placing
            # that event in a time window does, and must remain unavailable.
            series.unclocked = True
            return
        ns = event.available.ns
        series.first_event_ns = (
            ns if series.first_event_ns is None else min(series.first_event_ns, ns)
        )
        if not self._discontinuous:
            # Trusted control evidence can predate the frame carrying the event.
            # Keep its availability without moving the input stream clock back.
            series.events.append((ns, 1))
            series.events = deque(sorted(series.events))
            self._prune(series, self._last_ns)

    def on_epoch(self, frame: WireFrame, result: EpochResult) -> None:
        self._touch(frame.infer_available or frame.bridge_complete)
        self._connect_epochs()
        labels = {"rx_id": frame.rx_id, "tx_id": frame.tx_id}
        for event in result.events:
            self._epoch_event(event)
        if result.classification == "duplicate":
            self.increment("link.duplicate_count", labels=labels)
            for key in ("link.rssi_dbm", "link.noise_dbm"):
                self._exclude(key, labels, "duplicate")
        elif result.usable:
            self.increment("link.rx_unique_count", labels=labels)
            if result.classification == "reorder":
                self.increment("link.reorder_count", labels=labels)
            self.observe("link.rssi_dbm", frame.rssi_dbm, labels=labels)
            self.observe("link.noise_dbm", frame.noise_dbm, labels=labels)
        else:
            self.increment(
                "link.epoch_excluded_frame_count",
                labels={**labels, "reason": result.classification},
            )
            segment = frame.rx_id, frame.tx_id, result.link_segment
            if segment not in self._uncertain_segments:
                self._uncertain_segments.add(segment)
                self.increment("link.epoch_uncertain_count", labels=labels)
            for key in ("link.rx_unique_count", "link.rssi_dbm", "link.noise_dbm"):
                self._exclude(key, labels, result.classification)

    def _value(self, series: _Series, reduction: str, scope: str, as_of_ns: int) -> MetricValue:
        spec = self.registry[series.key]
        beginnings = [ns for ns in (self._first_ns, series.first_event_ns) if ns is not None]
        first = min(beginnings) if beginnings else None
        begin = (
            first
            if scope == "cumulative" or first is None
            else max(first, as_of_ns - self.window_ns)
        )
        reason = availability_reason(selected=series.key in self.selected, provider_connected=True)
        detail = None
        if (
            reason is None
            and scope == "window"
            and (self._discontinuous or series.unclocked or self._first_ns is None)
        ):
            reason = "unsupported"
            detail = "clock_discontinuity" if self._discontinuous else "missing_event_clock"
        self._prune(series, as_of_ns)
        events = [(ns, value) for ns, value in series.events if ns < as_of_ns]
        exclusions = (
            dict(series.exclusions)
            if scope == "cumulative"
            else dict(Counter(reason for ns, reason in series.exclusion_events if ns < as_of_ns))
        )
        count = series.count if scope == "cumulative" else len(events)
        value = None
        if reason is None:
            if reduction == "sum":
                value = series.total if scope == "cumulative" else sum(v for _, v in events)
            else:
                minimum = spec.min_samples[reduction]
                if len(events) < minimum:
                    reason = "no_samples"
                    detail = "insufficient_samples"
                else:
                    ordered = sorted(v for _, v in events)
                    value = ordered[math.ceil(int(reduction[1:]) / 100 * len(ordered)) - 1]
        return MetricValue(
            key=series.key,
            labels=dict(series.labels),
            reduction=reduction,
            aggregation_scope=scope,
            value=value,
            availability_reason=reason,
            detail=detail,
            begin_ns=begin,
            end_ns=as_of_ns,
            sample_count=count,
            excluded_count=sum(exclusions.values()),
            excluded_by_reason=exclusions,
            exposure_ns=None,
            counter_epoch=self.counter_epoch if scope == "cumulative" else None,
            provenance_id=self.provenance_id,
        )

    def snapshot(
        self, as_of_ns: int | None = None, *, time_basis: str = "replay_virtual"
    ) -> MetricsSnapshot:
        if as_of_ns is None:
            as_of_ns = 0 if self._max_ns is None else self._max_ns + 1
        if self._max_ns is not None and as_of_ns < self._max_ns:
            raise ValueError(
                "as_of_ns: cannot reconstruct an earlier snapshot from later observations"
            )
        if self._snapshot_ns is not None and as_of_ns < self._snapshot_ns:
            raise ValueError(
                "as_of_ns: snapshot clock cannot move backwards; create a new counter epoch"
            )
        self._snapshot_ns = as_of_ns
        basis = self.provenance.get(
            "availability_basis",
            (
                "legacy_bridge_proxy"
                if self.clock_kind == "unix_wall"
                else "synthetic"
                if self.clock_kind in {None, "virtual"}
                else "measured_infer"
            ),
        )
        clock_provenance = {
            **self.provenance,
            "provider_hash": PROVIDER_HASH,
            "counter_epoch": self.counter_epoch,
            "clock_discontinuity": self._discontinuous,
            "clock_domain": self.clock_domain or "unavailable:no_observations",
            "clock_kind": self.clock_kind or "virtual",
            "exposure_basis": "not_measured",
            "availability_basis": basis,
        }
        self.provenance_id = content_hash(
            {
                "run_id": self.run_id,
                "session_id": self.session_id,
                "spec_hash": self.registry.spec_hash,
                "selection_hash": self.selection_hash,
                **clock_provenance,
            }
        )
        values: list[MetricValue] = []
        populated = {s.key for s in self._series.values()}
        for series in sorted(
            self._series.values(), key=lambda s: (s.key, tuple(sorted(s.labels.items())))
        ):
            spec = self.registry[series.key]
            for scope in spec.aggregation_scopes:
                for reduction in spec.reductions:
                    values.append(self._value(series, reduction, scope, as_of_ns))
        for key in self.registry:
            if key in populated:
                continue
            spec = self.registry[key]
            reason = availability_reason(
                selected=key in self.selected,
                supported=key != "source.reconnect_count" or self.clock_kind != "unix_wall",
                requires_gt="ground_truth" in spec.prerequisites,
                provider_connected=False,
            )
            for scope in spec.aggregation_scopes:
                for reduction in spec.reductions:
                    values.append(
                        MetricValue(
                            key=key,
                            labels={},
                            reduction=reduction,
                            aggregation_scope=scope,
                            value=None,
                            availability_reason=reason,
                            detail="provider_not_connected" if reason == "not_measured" else None,
                            provenance_id=self.provenance_id,
                        )
                    )
        self._event_sequence += 1
        snapshot = MetricsSnapshot(
            schema_version=METRICS_SCHEMA_VERSION,
            run_id=self.run_id,
            session_id=self.session_id,
            stream_epoch=self.stream_epoch,
            event_sequence=self._event_sequence,
            profile_id=self.profile_id,
            selection_id=self.selection_id,
            spec_hash=self.registry.spec_hash,
            selection_hash=self.selection_hash,
            as_of_ns=as_of_ns,
            clock_domain=self.clock_domain or "unavailable:no_observations",
            clock_kind=self.clock_kind or "virtual",
            time_basis=time_basis,
            values=tuple(values),
            provider_hash=PROVIDER_HASH,
            provenance=clock_provenance,
        )
        self.registry.validate_snapshot(snapshot)
        return snapshot
