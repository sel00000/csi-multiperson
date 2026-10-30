"""Read-only S00–S04 composition with causal receiver clock mapping."""

from __future__ import annotations

from collections import Counter
from contextlib import ExitStack
from dataclasses import asdict, replace
import heapq
import importlib.metadata
import json
from pathlib import Path
import platform
import time
import uuid

from csi_mp.config import AppConfig, selected_metrics
from csi_mp.data.provenance import file_sha256, implementation_sha256
from csi_mp.data.schema import EpochEvidence, Stamp
from csi_mp.ingest.clock import ClockPolicy, OnlineClockMapper
from csi_mp.ingest.epochs import EpochPolicy, RxTimerUnwrapper, TxSequenceEpochs
from csi_mp.ingest.protocol import FrameParser
from csi_mp.ingest.rawlog import iter_raw_chunks
from csi_mp.telemetry.aggregate import IngestMetrics
from csi_mp.telemetry.schema import json_value
from csi_mp.viz.schema import to_dashboard_frame


def _frame_record(frame, epochs):
    return {
        "schema_id": "csi_mp.inspected_frame.v1",
        "spans": [asdict(span) for span in frame.spans],
        "wire_hex": frame.wire_bytes.hex(),
        "rx_id": frame.rx_id,
        "tx_id": frame.tx_id,
        "seq_u32": frame.seq_u32,
        "rx_timer_u32_us": frame.rx_timer_u32_us,
        "rx_boot_u8": frame.rx_boot_u8,
        "iq": frame.iq.tolist(),
        "rssi_dbm": frame.rssi_dbm,
        "noise_dbm": frame.noise_dbm,
        "signal_mode": frame.signal_mode,
        "signal_mode_reason": "absent_in_legacy_wire",
        "length_valid": True,
        "crc_valid": True,
        "bridge_complete": frame.bridge_complete.to_dict(),
        "infer_available": None
        if frame.infer_available is None
        else frame.infer_available.to_dict(),
        "availability_basis": frame.availability_basis,
        "ingest_ordinal": frame.ingest_ordinal,
        "epochs": json_value(epochs),
    }


def inspect_capture(cfg: AppConfig, *, emit=None) -> dict:
    """Inspect comparable source streams, sending optional JSON records to emit.

    The one-record lookahead preserves each file's order even across backwards
    wall-clock jumps. Ties use source ID then file offset. Distinct bridge domains
    need explicit cross-host calibration, which this legacy source merger does
    not infer. Only popped, parsed frames update the clock mapper; lookahead
    records never train it. Complete frames are emitted immediately.
    """
    emit = emit or (lambda kind, value: None)
    config = cfg.as_dict()
    sources = config["capture"]["sources"]
    if not sources:
        raise ValueError(
            "capture.sources: inspect requires at least one trusted file/source mapping"
        )
    if any(source.get("format", "csiraw01") != "csiraw01" for source in sources):
        raise ValueError("inspect supports CSIRAW01 only; use import for legacy HDF5 /links")
    domains = {source["clock_domain"] for source in sources}
    if len(domains) != 1:
        raise ValueError(
            "capture.sources.clock_domain: multi-source inspect needs one comparable domain; "
            "distinct bridge hosts require explicit calibration, not automatic RX clock fitting"
        )
    domain = next(iter(domains))
    for evidence in config["capture"]["controls"]:
        if evidence["clock_domain"] != domain:
            raise ValueError(
                "capture.controls.clock_domain: control evidence must share the raw source domain"
            )
    source_manifest = []
    for source in sources:
        digest = file_sha256(source["path"])
        if source.get("sha256") is not None and source["sha256"] != digest:
            raise ValueError(f"capture.sources.{source['source_id']}.sha256: source hash mismatch")
        source_manifest.append(
            {
                **source,
                "sha256": digest,
                "size_bytes": Path(source["path"]).stat().st_size,
                "raw_schema_id": "CSIRAW01",
                "wire_schema_id": "csi130.v1",
                "clock_kind": "unix_wall",
            }
        )
    source_hashes = {s["source_id"]: s["sha256"] for s in source_manifest}
    stream_epoch = f"ingest-{uuid.uuid4().hex}"
    synthetic = all(s.get("provenance") == "synthetic" for s in sources)
    basis = "synthetic" if synthetic else "legacy_bridge_proxy"
    measurement = config["measurement"]
    metrics = IngestMetrics(
        config["run_id"],
        config["capture"]["session_id"],
        stream_epoch,
        selected=selected_metrics(cfg),
        selection_id=measurement["selection_id"],
        profile_id=measurement["profile"],
        window_s=measurement["window_s"],
        clock_domain=domain,
        clock_kind="unix_wall",
        provenance={
            "config_hash": cfg.config_hash,
            "source_hashes": source_hashes,
            "availability_basis": basis,
            "synthetic": synthetic,
            "clock_model_id": None,
            "clock_error_ns": None,
        },
    )
    policy = EpochPolicy(**config["epoch"])
    rx_epochs, tx_epochs = RxTimerUnwrapper(policy), TxSequenceEpochs(policy)
    clock_mapper = OnlineClockMapper(policy=ClockPolicy(**config["clock"]), synthetic=synthetic)
    for evidence in config["capture"]["controls"]:
        tx_epochs.observe_control(
            EpochEvidence(
                evidence["kind"],
                evidence["device_id"],
                Stamp(
                    int(evidence["available_ns"]), evidence["clock_domain"], evidence["clock_kind"]
                ),
                evidence["source_ref"],
            )
        )
    counts = Counter(raw_records=0, payload_bytes=0, valid_frames=0, usable_frames=0)
    classifications, issue_counts, epoch_counts = Counter(), Counter(), Counter()
    clock_statuses, clock_exclusions = Counter(), Counter()
    timed_usable_frames = 0
    last_source_stamp = {}
    seen_events = set()
    max_ns = None
    ordinal = 0
    diagnostic_sequence = 0

    def issue(item):
        nonlocal diagnostic_sequence
        metrics.on_parse_issue(item)
        issue_counts[f"{item.layer}:{item.kind}:{item.detail}"] += 1
        diagnostic_sequence += 1
        source_id = item.spans[0].source_id if item.spans else None
        stamp = last_source_stamp.get(source_id)
        emit(
            "issues",
            {
                "event_sequence": diagnostic_sequence,
                **json_value(item),
                "available": stamp.to_dict() if stamp else None,
                "time_basis": "last_source_chunk" if stamp else "unavailable",
            },
        )

    parsers = {
        s["source_id"]: FrameParser(source_id=s["source_id"], expected_rx_id=s["rx_id"])
        for s in sources
    }
    pending = {s["source_id"]: [] for s in sources}
    heap = []
    next_snapshot = None
    coalesced_snapshot_boundaries = 0
    publish_ns = int(measurement["publish_period_s"] * 1_000_000_000)

    with ExitStack() as stack:
        readers = {}
        for source in sources:
            source_id = source["source_id"]
            reader = iter_raw_chunks(
                source["path"],
                source_id=source_id,
                clock_domain=source["clock_domain"],
                on_issue=pending[source_id].append,
            )
            stack.callback(reader.close)
            readers[source_id] = reader

        def advance(source_id):
            try:
                chunk = next(readers[source_id])
            except StopIteration:
                for item in pending[source_id]:
                    issue(item)
                pending[source_id].clear()
                for item in parsers[source_id].finish():
                    issue(item)
                return
            except ValueError:
                for item in pending[source_id]:
                    issue(item)
                pending[source_id].clear()
                raise
            diagnostics = tuple(pending[source_id])
            pending[source_id].clear()
            heapq.heappush(
                heap, (chunk.bridge_read.ns, source_id, chunk.span.file_offset, chunk, diagnostics)
            )

        for source in sources:
            advance(source["source_id"])
        while heap:
            _, source_id, _, original, diagnostics = heapq.heappop(heap)
            if original.span.source_sha256 != source_hashes[source_id]:
                raise ValueError(f"source {source_id} changed after manifest hashing")
            chunk = replace(original, ingest_ordinal=ordinal)
            ordinal += 1
            last_source_stamp[source_id] = chunk.bridge_read
            max_ns = chunk.bridge_read.ns if max_ns is None else max(max_ns, chunk.bridge_read.ns)
            # A periodic snapshot closes [begin,end); records exactly at end
            # belong to the following window. Gaps do not fabricate samples.
            if next_snapshot is None:
                next_snapshot = chunk.bridge_read.ns + publish_ns
            elif chunk.bridge_read.ns >= next_snapshot:
                skipped = (chunk.bridge_read.ns - next_snapshot) // publish_ns
                coalesced_snapshot_boundaries += skipped
                boundary = next_snapshot + skipped * publish_ns
                emit("metrics", metrics.snapshot(boundary, time_basis="replay_source").to_dict())
                next_snapshot = boundary + publish_ns
            metrics.on_raw_chunk(chunk)
            for item in diagnostics:
                issue(item)
            counts["raw_records"] += 1
            counts["payload_bytes"] += len(chunk.payload)
            frames, problems = parsers[source_id].feed(chunk)
            for item in problems:
                issue(item)
            for frame in frames:
                metrics.on_frame(frame)
                result = tx_epochs.classify(frame, rx_epochs.update(frame))
                metrics.on_epoch(frame, result)
                counts["valid_frames"] += 1
                counts["usable_frames"] += int(result.usable)
                classifications[result.classification] += 1
                for event in result.events:
                    if event.event_id not in seen_events:
                        seen_events.add(event.event_id)
                        epoch_counts[f"{event.scope}:{event.kind}:{event.certainty}"] += 1
                        emit("epochs", json_value(event))
                record = _frame_record(frame, result)
                timing = clock_mapper.observe(frame, result).to_dict()
                record["timing"] = timing
                clock_statuses[timing["clock_status"]] += 1
                timed_usable_frames += int(timing["usable"])
                if timing["exclusion_reason"] is not None:
                    clock_exclusions[timing["exclusion_reason"]] += 1
                # Generated CSIRAW01 fixture timestamps retain format semantics;
                # synthetic origin is an independent provenance declaration.
                record["availability_basis"] = basis
                emit("frames", record)
            advance(source_id)
    for parser in parsers.values():
        for item in parser.finish():
            issue(item)
    # Successful empty sources and an enabled epoch engine measured zero events.
    # Connecting them does not invent a chunk, signal sample, clock or exposure.
    for source in sources:
        metrics.connect_source(source["source_id"])
    metrics.connect_epochs()
    snapshot = metrics.snapshot(time_basis="replay_source")
    emit("metrics", snapshot.to_dict())
    dashboard = to_dashboard_frame(
        None,
        {
            "run_id": config["run_id"],
            "session_id": config["capture"]["session_id"],
            "stream_epoch": stream_epoch,
            "event_sequence": snapshot.event_sequence,
            "mode": "ingest",
            "coordinate_spec": cfg.model_spec.coordinate_spec,
            "source_time_ns": str(max_ns if max_ns is not None else 0),
            "source_clock_domain": domain,
            "source_clock_kind": "unix_wall",
            "emitted_monotonic_ns": str(time.monotonic_ns()),
            "emitter_clock_domain": f"inspect-process:{stream_epoch}",
            "availability_basis": basis,
            "quality": {
                "model_id": None,
                "config_hash": cfg.config_hash,
                "reason": "model_provider_not_connected",
                "tracking_enabled": False,
                "source_time_available": max_ns is not None,
                "acceptance_status": "not_evaluated",
            },
        },
    )
    manifest = {
        "schema_id": "csi_mp.inspection.v1",
        "status": "completed",
        "run_id": config["run_id"],
        "session_id": config["capture"]["session_id"],
        "stream_epoch": stream_epoch,
        "config_hash": cfg.config_hash,
        "implementation_sha256": implementation_sha256(),
        "metric_spec_hash": snapshot.spec_hash,
        "metric_selection_hash": snapshot.selection_hash,
        "metric_provider_hash": snapshot.provider_hash,
        "sources": source_manifest,
        "synthetic": synthetic,
        "availability_basis": basis,
        "legacy_assumptions": [
            "CSIRAW01 timestamps are Unix wall clock",
            "no measured inference arrival",
            "merge preserves source order; ties use source_id and byte offset",
            "RX timers mapped causally to the shared bridge clock; acquisition is estimated",
            "legacy availability uses bridge completion plus configured legacy_delay_ns",
            "no measured capture exposure or inference/network latency",
        ],
        "epoch_policy": config["epoch"],
        "clock_policy": config["clock"],
        "clock_snapshot": json.loads(clock_mapper.snapshot()),
        "timing_summary": {
            "observations": sum(clock_statuses.values()),
            "usable_observations": timed_usable_frames,
            "statuses": dict(clock_statuses),
            "excluded_by_reason": dict(clock_exclusions),
        },
        "counts": dict(counts),
        "metric_snapshot_policy": {
            "period_ns": str(publish_ns),
            "basis": "replay_source",
            "coalesced_empty_boundaries": coalesced_snapshot_boundaries,
            "note": "latest closed source-time boundary on arrival plus EOF; not measured runtime publish rate",
        },
        "classifications": dict(classifications),
        "issue_counts": dict(issue_counts),
        "epoch_events": dict(epoch_counts),
        "acceptance_status": "not_evaluated",
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "hostname": platform.node(),
            "package_version": importlib.metadata.version("csi-multiperson")
            if _installed()
            else "0.1.0",
            "numpy": importlib.metadata.version("numpy"),
            "PyYAML": importlib.metadata.version("PyYAML"),
        },
    }
    return {"manifest": manifest, "metrics": snapshot.to_dict(), "dashboard": dashboard.to_dict()}


def _installed():
    try:
        importlib.metadata.version("csi-multiperson")
        return True
    except importlib.metadata.PackageNotFoundError:
        return False
