"""Clock publication through the actual raw reader, epoch engine, and CLI."""

from collections import Counter, defaultdict
import binascii
import json
from pathlib import Path
import struct

import pytest

from csi_mp.cli import main
from csi_mp.config import load_config
from csi_mp.ingest import inspect as inspect_module


CAPTURE = Path(__file__).resolve().parents[1] / "configs/capture.yaml"
BASE_NS = 1_700_000_000_000_000_123


def wire(*, seq, timer, rx=0, boot=1):
    body = struct.pack("<HBBIIbbBB112s", 0xC51D, rx, 0, seq, timer, -60, -90, 56, boot, bytes(112))
    return body + struct.pack("<H", binascii.crc_hqx(body, 0xFFFF))


def record(ns, payload):
    return struct.pack("<QH", ns, len(payload)) + payload


def source(path, rx=0, *, synthetic=False):
    value = {
        "path": str(path),
        "source_id": f"rx{rx}",
        "rx_id": rx,
        "clock_domain": "test:bridge",
        "clock_kind": "unix_wall",
    }
    if synthetic:
        value["provenance"] = "synthetic"
    return value


def collect(sources, **overrides):
    cfg = load_config(CAPTURE, overrides={"capture": {"sources": sources}, **overrides})
    records = defaultdict(list)
    report = inspect_module.inspect_capture(
        cfg, emit=lambda kind, value: records[kind].append(value)
    )
    return report, records


def test_cli_writes_clock_metadata_without_inventing_infer_arrival(tmp_path, capsys):
    output = tmp_path / "s04-report"
    assert main(["inspect", "--config", str(CAPTURE), "--output", str(output)]) == 0
    manifest = json.loads(capsys.readouterr().out)
    frames = [json.loads(line) for line in (output / "frames.jsonl").read_text().splitlines()]
    assert manifest["clock_policy"]["policy_id"] == "clock_fixture_v1"
    assert isinstance(manifest["clock_snapshot"], dict)
    summary = manifest["timing_summary"]
    assert summary["observations"] == len(frames) == 13
    assert summary["statuses"] == dict(Counter(row["timing"]["clock_status"] for row in frames))
    assert summary["usable_observations"] == sum(row["timing"]["usable"] for row in frames)
    assert manifest["counts"]["usable_frames"] == 10
    for row in frames:
        timing = row["timing"]
        assert row["infer_available"] is None
        assert timing["availability_basis"] == "synthetic"
        assert timing["availability_ns"] == row["bridge_complete"]["ns"]
        assert timing["clock_domain"] == "fixture:bridge"
        assert timing["clock_kind"] == "unix_wall"
        assert timing["clock_reference"]["kind"] == "unix_wall"
        assert isinstance(timing["availability_ns"], str)
        if timing["acquisition_ns"] is not None:
            assert isinstance(timing["acquisition_ns"], str)
            assert int(timing["acquisition_ns"]) <= int(timing["availability_ns"])
        if timing["clock_error_ns"] is not None:
            assert isinstance(timing["clock_error_ns"], str)
        if timing["clock_status"] == "excluded":
            assert not timing["usable"] and timing["exclusion_reason"]
    metrics = json.loads((output / "metrics.json").read_text())
    for value in metrics["values"]:
        if value["key"] in {"link.rx_unique_pps", "link.seq_gap_estimate"}:
            assert value["value"] is None


def test_legacy_proxy_delay_is_explicit_and_keeps_nanosecond_precision(tmp_path):
    raw = tmp_path / "legacy.csiraw"
    raw.write_bytes(
        b"CSIRAW01"
        + b"".join(
            record(BASE_NS + i * 10_000_000, wire(seq=100 + i, timer=500_000 + i * 10_000))
            for i in range(4)
        )
    )
    delay = 2_000_001
    report, records = collect([source(raw)], clock={"legacy_delay_ns": delay})
    assert report["manifest"]["availability_basis"] == "legacy_bridge_proxy"
    assert report["manifest"]["clock_policy"]["legacy_delay_ns"] == delay
    for i, row in enumerate(records["frames"]):
        timing = row["timing"]
        assert timing["availability_basis"] == "legacy_bridge_proxy"
        assert timing["availability_ns"] == str(BASE_NS + i * 10_000_000 + delay)
        assert timing["acquisition_ns"] == str(BASE_NS + i * 10_000_000)
        assert row["infer_available"] is None


def test_appending_different_raw_futures_preserves_published_prefix(tmp_path):
    raw = tmp_path / "growing.csiraw"
    prefix = b"CSIRAW01" + b"".join(
        record(BASE_NS + i * 10_000_000, wire(seq=100 + i, timer=900_000 + i * 10_000))
        for i in range(5)
    )
    raw.write_bytes(prefix)
    before, records = collect([source(raw)])
    published = [row["timing"] for row in records["frames"]]
    original_bytes = json.dumps(published, sort_keys=True)
    suffixes = [
        record(BASE_NS + 50_000_000, wire(seq=105, timer=950_000)),
        record(BASE_NS + 900_000_000, wire(seq=105, timer=1_000, boot=2))
        + record(BASE_NS + 910_000_000, wire(seq=106, timer=11_000, boot=2)),
    ]
    hashes = {before["manifest"]["sources"][0]["sha256"]}
    for suffix in suffixes:
        raw.write_bytes(prefix + suffix)
        after, extended = collect([source(raw)])
        hashes.add(after["manifest"]["sources"][0]["sha256"])
        assert [row["timing"] for row in extended["frames"][:5]] == published
        assert json.dumps(published, sort_keys=True) == original_bytes
    assert len(hashes) == 3


def test_rx_reboot_does_not_reset_other_receivers_common_timeline(tmp_path):
    paths = [tmp_path / f"rx{rx}.csiraw" for rx in range(2)]
    for rx, path in enumerate(paths):
        payloads = []
        for i in range(5):
            restarted = rx == 0 and i >= 3
            timer = (
                1_000 + (i - 3) * 10_000 if restarted else 900_000 + rx * 20_000_000 + i * 10_000
            )
            payloads.append(
                record(
                    BASE_NS + i * 10_000_000,
                    wire(seq=100 + i, timer=timer, rx=rx, boot=2 if restarted else 1),
                )
            )
        path.write_bytes(b"CSIRAW01" + b"".join(payloads))
    _, records = collect([source(path, rx) for rx, path in enumerate(paths)])
    by_rx = {
        rx: [row["timing"] for row in records["frames"] if row["rx_id"] == rx] for rx in range(2)
    }
    for timings in by_rx.values():
        assert [row["acquisition_ns"] for row in timings] == [
            str(BASE_NS + i * 10_000_000) for i in range(5)
        ]
    assert by_rx[0][3]["clock_status"] == "warming_up"
    assert by_rx[0][3]["clock_segment_id"] != by_rx[0][2]["clock_segment_id"]
    assert by_rx[1][3]["clock_status"] == "ready"
    assert by_rx[1][3]["clock_segment_id"] == by_rx[1][2]["clock_segment_id"]


@pytest.mark.parametrize("receiver_count", [1, 3])
@pytest.mark.parametrize("frames_per_record", [1, 2])
def test_inspect_emits_current_frame_before_reading_next_source_record(
    tmp_path, monkeypatch, receiver_count, frames_per_record
):
    sources = []
    for rx in range(receiver_count):
        raw = tmp_path / f"rx{rx}.csiraw"
        raw.write_bytes(
            b"CSIRAW01"
            + b"".join(
                record(
                    BASE_NS + rx * 1_000_000 + i * 10_000_000,
                    b"".join(
                        wire(
                            seq=100 + i * frames_per_record + j,
                            timer=800_000 + rx * 100_000 + i * 10_000,
                            rx=rx,
                        )
                        for j in range(frames_per_record)
                    ),
                )
                for i in range(4)
            )
        )
        sources.append(source(raw, rx))
    actual_reader = inspect_module.iter_raw_chunks
    published = []
    published_by_source = defaultdict(list)

    def guarded_reader(*args, **kwargs):
        source_id = kwargs["source_id"]
        for i, chunk in enumerate(actual_reader(*args, **kwargs)):
            assert len(published_by_source[source_id]) == i * frames_per_record, (
                "source read ahead of causal clock publication"
            )
            yield chunk

    def emit(kind, value):
        if kind == "frames":
            published.append(value)
            published_by_source[value["spans"][0]["source_id"]].append(value)

    monkeypatch.setattr(inspect_module, "iter_raw_chunks", guarded_reader)
    cfg = load_config(CAPTURE, overrides={"capture": {"sources": sources}})
    inspect_module.inspect_capture(cfg, emit=emit)
    assert len(published) == 4 * receiver_count * frames_per_record
    for i, row in enumerate(published):
        # One pending raw record per source is allowed by the merger, but only
        # the currently published receiver/timer may train its model.
        tick = i // (receiver_count * frames_per_record)
        timing = row["timing"]
        assert timing["acquisition_ns"] == str(
            BASE_NS + row["rx_id"] * 1_000_000 + tick * 10_000_000
        )
        assert timing["clock_model"]["sample_count"] == tick + 1
        assert timing["clock_status"] == ("warming_up" if tick < 2 else "ready")
