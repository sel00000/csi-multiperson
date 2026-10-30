from dataclasses import replace
import json

import numpy as np
import pytest

from csi_mp.data.schema import (
    EpochEvent,
    EpochResult,
    ParseIssue,
    RawChunk,
    SourceSpan,
    Stamp,
    WireFrame,
)
from csi_mp.telemetry import (
    INGEST_V1_KEYS,
    IngestMetrics,
    MetricsSnapshot,
    default_registry,
    validate_selection,
)
from csi_mp.telemetry.registry import availability_reason
from csi_mp.telemetry.schema import MetricValue, json_value


def chunk(ns=0, payload=b"x" * 130, source="rx0", *, kind="virtual", index=0):
    return RawChunk(
        SourceSpan("sha256-fixture", source, index, index * 140 + 18, len(payload)),
        payload,
        Stamp(ns, "fixture", kind),
        None,
        index,
    )


def frame(ns=0, seq=0, rx=0, tx=0, rssi=-50):
    raw = chunk(ns, index=seq, source=f"rx{rx}")
    return WireFrame(
        (raw.span,),
        raw.payload,
        rx,
        tx,
        seq,
        ns // 1000,
        0,
        np.zeros((56, 2), dtype=np.int8),
        rssi,
        -95,
        None,
        raw.bridge_read,
        None,
        seq,
    )


def result(classification="normal", *, usable=True, segment="s0", events=()):
    return EpochResult(
        "rx0.0", "tx0.0" if usable else None, segment, 0, classification, usable, (), events
    )


def value(snapshot, key, labels=None, reduction="sum", scope="cumulative"):
    labels = {} if labels is None else {k: str(v) for k, v in labels.items()}
    return next(
        v
        for v in snapshot.values
        if v.key == key
        and v.labels == labels
        and v.reduction == reduction
        and v.aggregation_scope == scope
    )


def test_initial_selection_is_exactly_fifteen_defined_ids():
    registry = default_registry()
    assert len(INGEST_V1_KEYS) == 15
    assert validate_selection(INGEST_V1_KEYS) == INGEST_V1_KEYS
    assert registry.spec_hash == default_registry().spec_hash
    with pytest.raises(ValueError, match="unknown metric"):
        validate_selection(["link.invented"])
    with pytest.raises(ValueError, match="duplicate"):
        validate_selection([INGEST_V1_KEYS[0]] * 2)
    with pytest.raises(ValueError, match="unit"):
        registry.validate_request("link.rssi_dbm", unit="mm")
    with pytest.raises(ValueError, match="reduction"):
        registry.validate_request("link.rssi_dbm", reduction="sum")
    with pytest.raises(ValueError, match="aggregation_scope"):
        registry.validate_request("link.rssi_dbm", aggregation_scope="cumulative")


@pytest.mark.parametrize(
    "kwargs,expected",
    [
        ({"selected": False, "supported": False, "requires_gt": True}, "disabled"),
        ({"selected": True, "supported": False, "requires_gt": True}, "unsupported"),
        ({"selected": True, "requires_gt": True}, "no_gt"),
        ({"selected": True, "requires_gt": True, "has_gt": True}, "not_measured"),
        ({"selected": True, "provider_connected": True, "sample_count": 0}, "no_samples"),
        ({"selected": True, "provider_connected": True, "denominator": 0}, "no_samples"),
        ({"selected": True, "provider_connected": True, "sample_count": 20}, None),
    ],
)
def test_missing_reason_precedence(kwargs, expected):
    assert availability_reason(**kwargs) == expected


def test_missing_vs_zero_and_unconnected_providers():
    metrics = IngestMetrics(
        "run",
        "session",
        selected=(*INGEST_V1_KEYS, "eval.pck", "latency.model_ready_ms", "link.rx_unique_pps"),
    )
    before = metrics.snapshot()
    assert value(before, "source.bytes_received_count").availability_reason == "not_measured"
    assert (
        value(before, "eval.pck", reduction="ratio", scope="evaluation").availability_reason
        == "no_gt"
    )
    assert (
        value(before, "latency.model_ready_ms", reduction="p95", scope="window").availability_reason
        == "not_measured"
    )
    assert (
        value(before, "link.rx_unique_pps", reduction="rate", scope="window").availability_reason
        == "not_measured"
    )
    assert (
        value(
            before, "resource.process_rss_bytes", reduction="last", scope="window"
        ).availability_reason
        == "disabled"
    )
    metrics.on_raw_chunk(chunk(1))
    after = metrics.snapshot()
    assert value(after, "source.crc_error_count", {"source_id": "rx0"}).value == 0
    assert value(after, "source.bytes_received_count", {"source_id": "rx0"}).value == 130


def test_explicit_empty_provider_connections_measure_zero_without_samples_or_time():
    metrics = IngestMetrics("run", "session", clock_domain="fixture", clock_kind="unix_wall")
    metrics.connect_source("rx0")
    metrics.connect_source("rx0")
    metrics.connect_epochs()
    metrics.connect_epochs()
    snap = metrics.snapshot(500, time_basis="replay_source")
    for key in (
        "source.raw_record_count",
        "source.bytes_received_count",
        "source.valid_frame_count",
    ):
        counter = value(snap, key, {"source_id": "rx0"})
        assert counter.value == 0 and counter.availability_reason is None
        assert counter.sample_count == 0 and counter.begin_ns is None
        assert counter.exposure_ns is None
        window = value(snap, key, {"source_id": "rx0"}, scope="window")
        assert window.value is None and window.availability_reason == "unsupported"
        assert window.detail == "missing_event_clock" and window.begin_ns is None
    unique = value(snap, "link.rx_unique_count", {"rx_id": 0, "tx_id": 0})
    assert unique.value == 0 and unique.sample_count == 0 and unique.begin_ns is None
    signal = value(snap, "link.rssi_dbm", {"rx_id": 0, "tx_id": 0}, "p50", "window")
    assert signal.value is None and signal.availability_reason == "unsupported"
    assert signal.sample_count == 0 and signal.begin_ns is None
    assert not metrics.diagnostics


def test_connecting_empty_second_source_does_not_add_events_to_active_clock():
    metrics = IngestMetrics("run", "session")
    metrics.on_raw_chunk(chunk(0))
    metrics.connect_source("rx1")
    snap = metrics.snapshot()
    first = value(snap, "source.raw_record_count", {"source_id": "rx0"})
    empty = value(snap, "source.raw_record_count", {"source_id": "rx1"})
    assert first.value == 1 and first.sample_count == 1
    assert empty.value == 0 and empty.sample_count == 0
    assert value(snap, "source.raw_record_count", {"source_id": "rx1"}, scope="window").value == 0


def test_parser_counters_do_not_trust_crc_header_or_double_count_tail():
    metrics = IngestMetrics("run", "session")
    raw = chunk(100, payload=b"x" * 270)
    metrics.on_raw_chunk(raw)
    crc_span = SourceSpan("sha256-fixture", "rx0", 0, 18, 130)
    crc = ParseIssue("crc", "wire_frame", (crc_span,), 130, "crc_mismatch")
    metrics.on_parse_issue(crc)
    metrics.on_parse_issue(crc)
    metrics.on_parse_issue(
        replace(
            crc,
            spans=(
                replace(crc_span, byte_count=1),
                replace(crc_span, chunk_index=1, file_offset=19, byte_count=129),
            ),
        )
    )
    metrics.on_parse_issue(
        ParseIssue("semantic", "wire_frame", (crc_span,), 130, "tx_id_out_of_range")
    )
    metrics.on_parse_issue(ParseIssue("discarded_bytes", "wire_frame", (crc_span,), 7, "junk"))
    metrics.on_parse_issue(
        ParseIssue("truncated", "raw_record", (crc_span,), 3, "incomplete_record_header")
    )
    metrics.on_parse_issue(
        ParseIssue("truncated", "wire_frame", (crc_span,), 10, "incomplete_frame")
    )
    metrics.on_frame(frame(100))
    snap = metrics.snapshot()
    assert value(snap, "source.raw_record_count", {"source_id": "rx0"}).value == 1
    assert value(snap, "source.valid_frame_count", {"source_id": "rx0"}).value == 1
    assert value(snap, "source.crc_error_count", {"source_id": "rx0"}).value == 1
    assert (
        value(
            snap,
            "source.semantic_reject_count",
            {"source_id": "rx0", "reason": "tx_id_out_of_range"},
        ).value
        == 1
    )
    assert (
        value(snap, "source.discarded_bytes_count", {"source_id": "rx0", "reason": "junk"}).value
        == 7
    )
    assert (
        value(
            snap, "source.truncated_tail_bytes", {"source_id": "rx0", "layer": "raw_record"}
        ).value
        == 3
    )
    assert (
        value(
            snap, "source.truncated_tail_bytes", {"source_id": "rx0", "layer": "wire_frame"}
        ).value
        == 10
    )


def test_epoch_segments_device_events_and_signal_population():
    metrics = IngestMetrics("run", "session")
    event = EpochEvent(
        "tx0-reset1",
        "tx",
        0,
        "tx_reset",
        "inferred",
        Stamp(1, "fixture", "virtual"),
        (0, 3, 6),
        ("fixture",),
    )
    metrics.on_epoch(frame(1), result(events=(event,)))
    metrics.on_epoch(frame(2, rx=1), result(events=(replace(event, certainty="confirmed"),)))
    metrics.on_epoch(frame(3), result("duplicate", usable=False))
    metrics.on_epoch(frame(4, seq=2), result("ambiguous", usable=False, segment="u1"))
    metrics.on_epoch(frame(5, seq=3), result("ambiguous", usable=False, segment="u1"))
    metrics.on_epoch(frame(6, seq=4), result("reorder"))
    snap = metrics.snapshot()
    link = {"rx_id": 0, "tx_id": 0}
    assert value(snap, "link.rx_unique_count", link).value == 2
    assert value(snap, "link.rx_unique_count", {"rx_id": 1, "tx_id": 0}).value == 1
    assert value(snap, "link.reorder_count", link).value == 1
    assert value(snap, "link.duplicate_count", link).value == 1
    assert value(snap, "link.epoch_uncertain_count", link).value == 1
    assert (
        value(snap, "link.epoch_excluded_frame_count", {**link, "reason": "ambiguous"}).value == 2
    )
    events = [
        v
        for v in snap.values
        if v.key == "epoch.event_count" and v.aggregation_scope == "cumulative"
    ]
    assert sum(v.value for v in events) == 1
    assert next(v for v in events if v.value == 1).labels["certainty"] == "inferred"
    signal = value(snap, "link.rssi_dbm", link, "p50", "window")
    assert signal.sample_count == 2 and signal.excluded_count == 3
    assert signal.excluded_by_reason == {"duplicate": 1, "ambiguous": 2}


def test_nearest_rank_quantiles_minimum_samples_and_half_open_window():
    metrics = IngestMetrics("run", "session", window_s=10)
    link = {"rx_id": 0, "tx_id": 0}
    for i in range(19):
        metrics.on_epoch(frame(i * 1000, seq=i, rssi=-100 + i), result())
    before = metrics.snapshot(18_001)
    assert value(before, "link.rssi_dbm", link, "p50", "window").value == -91
    p95 = value(before, "link.rssi_dbm", link, "p95", "window")
    assert (
        p95.value is None
        and p95.availability_reason == "no_samples"
        and p95.detail == "insufficient_samples"
    )
    metrics.on_epoch(frame(19_000, seq=19, rssi=-81), result())
    at_boundary = metrics.snapshot(19_000)
    assert value(at_boundary, "link.rssi_dbm", link, "p95", "window").value is None
    after = metrics.snapshot(19_001)
    assert value(after, "link.rssi_dbm", link, "p5", "window").value == -100
    assert value(after, "link.rssi_dbm", link, "p50", "window").value == -91
    assert value(after, "link.rssi_dbm", link, "p95", "window").value == -82


def test_cumulative_and_window_delta_are_distinct_with_no_exposure_fabrication():
    metrics = IngestMetrics("run", "session", window_s=10)
    metrics.on_raw_chunk(chunk(0, payload=b"a"))
    metrics.on_raw_chunk(chunk(11_000_000_000, payload=b"bc", index=1))
    snap = metrics.snapshot(11_000_000_001)
    cumulative = value(snap, "source.bytes_received_count", {"source_id": "rx0"})
    window = value(snap, "source.bytes_received_count", {"source_id": "rx0"}, scope="window")
    assert cumulative.value == 3 and window.value == 2
    assert cumulative.counter_epoch and window.counter_epoch is None
    assert window.exposure_ns is None
    assert window.begin_ns == 1_000_000_001


def test_wall_regression_keeps_original_counts_but_not_window_statistics():
    metrics = IngestMetrics("run", "session")
    metrics.on_raw_chunk(chunk(100, kind="unix_wall"))
    metrics.on_raw_chunk(chunk(90, kind="unix_wall", index=1))
    snap = metrics.snapshot(time_basis="replay_source")
    assert value(snap, "source.raw_record_count", {"source_id": "rx0"}).value == 2
    win = value(snap, "source.raw_record_count", {"source_id": "rx0"}, scope="window")
    assert (
        win.value is None
        and win.availability_reason == "unsupported"
        and win.detail == "clock_discontinuity"
    )
    assert snap.clock_kind == "unix_wall" and snap.provenance["clock_discontinuity"]
    assert metrics.diagnostics


def test_explicit_synthetic_provenance_survives_legacy_wall_storage():
    metrics = IngestMetrics("run", "session", provenance={"availability_basis": "synthetic"})
    metrics.on_raw_chunk(chunk(0, kind="unix_wall"))
    snap = metrics.snapshot()
    assert snap.clock_kind == "unix_wall"
    assert snap.provenance["availability_basis"] == "synthetic"
    window = value(snap, "source.raw_record_count", {"source_id": "rx0"}, scope="window")
    assert window.begin_ns == 0


def test_clock_domains_cannot_be_subtracted_or_mixed():
    metrics = IngestMetrics("run", "session")
    metrics.on_raw_chunk(chunk(100))
    with pytest.raises(ValueError, match="clock mismatch"):
        metrics.on_raw_chunk(replace(chunk(200), bridge_read=Stamp(200, "other-host", "monotonic")))


def test_snapshot_roundtrip_preserves_large_ns_series_ids_and_hashes():
    metrics = IngestMetrics("run", "session")
    metrics.on_raw_chunk(chunk(1_800_000_000_000_000_001))
    snap = metrics.snapshot()
    encoded = json.loads(snap.to_json())
    assert isinstance(encoded["as_of_ns"], str)
    assert isinstance(encoded["values"][0]["begin_ns"], str)
    restored = MetricsSnapshot.from_json(snap.to_json())
    assert restored == snap
    assert restored.provider_hash and restored.selection_hash and restored.spec_hash
    assert len({v.series_id for v in restored.values}) == len(restored.values)
    encoded["as_of_ns"] = int(encoded["as_of_ns"])
    with pytest.raises(ValueError, match="decimal string"):
        MetricsSnapshot.from_dict(encoded)
    with pytest.raises(ValueError, match="registry mismatch"):
        metrics.registry.validate_snapshot(replace(snap, spec_hash="wrong"))
    with pytest.raises(ValueError, match="schema_version"):
        replace(snap, schema_version=999)


def test_nested_epoch_event_stamp_serializes_ns_as_decimal_string():
    event = EpochEvent(
        "tx0-control1",
        "tx",
        0,
        "tx_reset",
        "confirmed",
        Stamp(1_800_000_000_000_000_001, "source:wall", "unix_wall"),
        (0, 3, 6),
        ("trusted_start",),
    )
    encoded = json.loads(json.dumps(json_value(result(events=(event,)))))
    available = encoded["events"][0]["available"]
    assert available["ns"] == "1800000000000000001"
    assert Stamp.from_dict(available) == event.available


def test_device_event_uses_evidence_clock_without_rewinding_frame_clock():
    metrics = IngestMetrics("run", "session", window_s=1)
    event = EpochEvent(
        "start-old",
        "tx",
        0,
        "tx_reset",
        "confirmed",
        Stamp(1, "fixture", "virtual"),
        (0, 3, 6),
        ("start_log",),
    )
    metrics.on_epoch(frame(2_000_000_000), result(events=(event,)))
    snap = metrics.snapshot()
    labels = {"scope": "tx", "device_id": 0, "kind": "tx_reset", "certainty": "confirmed"}
    cumulative = value(snap, "epoch.event_count", labels)
    assert cumulative.value == 1 and cumulative.begin_ns == 1
    assert value(snap, "epoch.event_count", labels, scope="window").value == 0
    assert not snap.provenance["clock_discontinuity"]
    signal = value(snap, "link.rssi_dbm", {"rx_id": 0, "tx_id": 0}, "p50", "window")
    assert signal.value == -50


def test_device_event_other_clock_has_count_without_fabricated_window():
    metrics = IngestMetrics("run", "session")
    event = EpochEvent(
        "start-other-clock",
        "tx",
        0,
        "tx_reset",
        "confirmed",
        Stamp(1, "other", "unix_wall"),
        (0, 3, 6),
        ("start_log",),
    )
    metrics.on_epoch(frame(100), result(events=(event,)))
    snap = metrics.snapshot()
    labels = {"scope": "tx", "device_id": 0, "kind": "tx_reset", "certainty": "confirmed"}
    assert value(snap, "epoch.event_count", labels).value == 1
    assert (
        value(snap, "epoch.event_count", labels, scope="window").availability_reason
        == "unsupported"
    )


def test_null_requires_reason_and_zero_denominator_cannot_claim_ratio():
    with pytest.raises(ValueError, match="requires a reason"):
        MetricValue("test", {}, "last", "instant", None)
    with pytest.raises(ValueError, match="positive denominator"):
        MetricValue("test", {}, "ratio", "window", 0, numerator=0, denominator=0)
    with pytest.raises(ValueError, match="missing reason"):
        MetricValue("test", {}, "last", "instant", 0, availability_reason="no_samples")
