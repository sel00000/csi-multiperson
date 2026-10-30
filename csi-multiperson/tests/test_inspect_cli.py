"""End-to-end fixtures assert losses and epochs independently of implementation."""

from collections import defaultdict
import json
import os
from pathlib import Path
import struct
import subprocess
import sys

import pytest
import yaml

from csi_mp.cli import _output_directory, main
from csi_mp.config import load_config
from csi_mp.ingest.inspect import file_sha256, inspect_capture
from csi_mp.telemetry.schema import MetricsSnapshot
from csi_mp.viz.schema import DashboardFrame

PROJECT = Path(__file__).resolve().parents[1]
CAPTURE = PROJECT / "configs/capture.yaml"
FIXTURES = PROJECT / "fixtures/ingest_v1"


def collect(cfg=None):
    records = defaultdict(list)
    report = inspect_capture(
        cfg or load_config(CAPTURE), emit=lambda kind, value: records[kind].append(value)
    )
    return report, records


def total(snapshot, key, **labels):
    return sum(
        v["value"]
        for v in snapshot["values"]
        if v["key"] == key
        and v["aggregation_scope"] == "cumulative"
        and v["value"] is not None
        and all(v["labels"].get(k) == str(value) for k, value in labels.items())
    )


def test_fixture_byte_accounting_source_hashes_and_serialized_contracts():
    expected = json.loads((FIXTURES / "expected.json").read_text())
    before = {name: file_sha256(FIXTURES / f"{name}.csiraw") for name in ("rx0", "rx1")}
    report, records = collect()
    counts = report["manifest"]["counts"]
    assert counts == {
        "raw_records": 16,
        "payload_bytes": 1990,
        "valid_frames": 13,
        "usable_frames": 10,
    }
    snapshot = report["metrics"]
    mapping = {
        "source.raw_record_count": "raw_records",
        "source.bytes_received_count": "payload_bytes",
        "source.valid_frame_count": "valid_frames",
        "source.crc_error_count": "crc_errors",
        "source.semantic_reject_count": "semantic_rejects",
    }
    for source_id, source in expected["sources"].items():
        assert before[source_id] == source["sha256"] == file_sha256(FIXTURES / source["path"])
        for key, count_key in mapping.items():
            assert total(snapshot, key, source_id=source_id) == source["counts"][count_key]
    assert total(snapshot, "source.discarded_bytes_count", reason="junk") == 23
    assert total(snapshot, "source.discarded_bytes_count", reason="resync") == 130
    assert total(snapshot, "source.truncated_tail_bytes", layer="raw_record") == 15
    assert total(snapshot, "source.truncated_tail_bytes", layer="wire_frame") == 17
    # CRC candidates overlap discarded bytes; do not add them as separate byte loss.
    assert 1990 == (13 + 1) * 130 + 23 + 130 + 17
    assert total(snapshot, "link.rx_unique_count") == 10
    assert total(snapshot, "link.duplicate_count") == 1
    assert total(snapshot, "link.reorder_count") == 1
    assert total(snapshot, "link.epoch_uncertain_count") == 1
    assert total(snapshot, "link.epoch_excluded_frame_count") == 2
    assert total(snapshot, "epoch.event_count", kind="tx_reset", certainty="inferred") == 1
    assert total(snapshot, "epoch.event_count", kind="rx_reset", certainty="confirmed") == 1
    for row in records["frames"]:
        pieces = []
        for span in row["spans"]:
            raw = (FIXTURES / f"{span['source_id']}.csiraw").read_bytes()
            pieces.append(raw[span["file_offset"] : span["file_offset"] + span["byte_count"]])
        assert b"".join(pieces).hex() == row["wire_hex"]
        assert row["bridge_complete"]["kind"] == "unix_wall"
        assert isinstance(row["bridge_complete"]["ns"], str)
        assert row["availability_basis"] == "synthetic"
        assert row["infer_available"] is None
    assert MetricsSnapshot.from_dict(snapshot).to_dict() == snapshot
    dashboard = DashboardFrame.from_dict(report["dashboard"])
    assert dashboard.mode == "ingest" and dashboard.status == "unknown"
    assert dashboard.current_count is None and dashboard.persons == ()
    assert report["manifest"]["acceptance_status"] == "not_evaluated"
    assert snapshot["provenance"]["availability_basis"] == "synthetic"
    for value in snapshot["values"]:
        if value["key"] in {"link.rx_unique_pps", "link.seq_gap_estimate", "eval.pck"}:
            assert value["value"] is None and value["availability_reason"] == "disabled"


def test_tx_restart_recovers_and_other_receiver_clock_is_independent():
    _, records = collect()
    frames = records["frames"]
    rx0 = [row for row in frames if row["rx_id"] == 0]
    low = [row for row in rx0 if row["seq_u32"] <= 3]
    assert [row["epochs"]["classification"] for row in low] == [
        "ambiguous",
        "ambiguous",
        "tx_reset",
        "normal",
    ]
    assert [row["epochs"]["usable"] for row in low] == [False, False, True, True]
    assert low[0]["epochs"]["tx_epoch"] is None
    assert low[2]["epochs"]["tx_epoch"] != rx0[0]["epochs"]["tx_epoch"]
    rx1 = [row for row in frames if row["rx_id"] == 1]
    assert len({row["epochs"]["rx_epoch"] for row in rx1}) == 1
    assert all(row["epochs"]["usable"] for row in rx1)
    first100 = [row for row in frames if row["seq_u32"] == 100]
    assert first100[0]["epochs"]["usable"] and first100[1]["epochs"]["usable"]
    assert first100[0]["rx_id"] != first100[1]["rx_id"]


def test_trusted_start_produces_confirmed_event_and_no_inferred_reset():
    cfg = load_config(
        CAPTURE,
        overrides={
            "capture": {
                "controls": [
                    {
                        "kind": "tx_start",
                        "device_id": 0,
                        "available_ns": "1700000000039000000",
                        "clock_domain": "fixture:bridge",
                        "clock_kind": "unix_wall",
                        "source_ref": "fixture:trusted-start-log:1",
                    }
                ]
            }
        },
    )
    report, records = collect(cfg)
    assert (
        total(report["metrics"], "epoch.event_count", kind="tx_reset", certainty="confirmed") == 1
    )
    assert total(report["metrics"], "epoch.event_count", kind="tx_reset", certainty="inferred") == 0
    assert total(report["metrics"], "link.epoch_excluded_frame_count") == 0
    resets = [row for row in records["epochs"] if row["kind"] == "tx_reset"]
    assert resets[0]["affected_links"] == [0, 3, 6]
    assert resets[0]["available"]["ns"] == "1700000000039000000"


def test_cli_report_collision_and_originals_unchanged(tmp_path, capsys):
    before = {p: file_sha256(p) for p in FIXTURES.glob("*.csiraw")}
    output = tmp_path / "report"
    assert main(["inspect", "--config", str(CAPTURE), "--output", str(output)]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "completed"
    assert {p.name for p in output.iterdir()} == {
        "config.resolved.json",
        "manifest.json",
        "metrics.json",
        "dashboard.json",
        "frames.jsonl",
        "issues.jsonl",
        "epochs.jsonl",
        "metrics.jsonl",
    }
    manifest_bytes = (output / "manifest.json").read_bytes()
    assert main(["inspect", "--config", str(CAPTURE), "--output", str(output)]) == 2
    assert "File exists" in capsys.readouterr().err
    assert (output / "manifest.json").read_bytes() == manifest_bytes
    assert before == {p: file_sha256(p) for p in before}
    assert len((output / "frames.jsonl").read_text().splitlines()) == 13


def test_cli_config_and_help_work_from_unrelated_directory_without_ml(tmp_path):
    env = {**os.environ, "PYTHONPATH": str(PROJECT / "src")}
    help_result = subprocess.run(
        [sys.executable, "-m", "csi_mp", "--help"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "validate-config" in help_result.stdout and "inspect" in help_result.stdout
    assert "S00–S05" in help_result.stdout and "import" in help_result.stdout
    validate = subprocess.run(
        [sys.executable, "-m", "csi_mp", "validate-config", "--config", str(CAPTURE)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(validate.stdout)["valid"]
    code = "from csi_mp.cli import main; import sys; main(['validate-config','--config',sys.argv[1]]); assert not {'torch','cv2','serial'} & sys.modules.keys()"
    subprocess.run(
        [sys.executable, "-c", code, str(CAPTURE)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )


def test_bad_header_failure_keeps_issue_record_and_source(tmp_path, capsys):
    raw = tmp_path / "bad.csiraw"
    raw.write_bytes(b"notraw")
    cfg = tmp_path / "bad.yaml"
    cfg.write_text(
        yaml.safe_dump(
            {
                "capture": {
                    "sources": [
                        {
                            "source_id": "rx0",
                            "rx_id": 0,
                            "path": "bad.csiraw",
                            "clock_domain": "test:wall",
                        }
                    ]
                }
            }
        )
    )
    output = tmp_path / "failed-run"
    assert main(["inspect", "--config", str(cfg), "--output", str(output)]) == 2
    assert "invalid CSIRAW01 header" in capsys.readouterr().err
    issue = json.loads((output / "issues.jsonl").read_text())
    assert issue["kind"] == "invalid_header" and issue["byte_count"] == 6
    assert json.loads((output / "failure.json").read_text())["status"] == "failed"
    assert not (output / "manifest.json").exists()
    assert raw.read_bytes() == b"notraw"


def test_different_domains_and_wrong_source_hash_rejected():
    sources = load_config(CAPTURE).as_dict()["capture"]["sources"]
    sources[1]["clock_domain"] = "other:host"
    with pytest.raises(ValueError, match="comparable domain"):
        collect(load_config(CAPTURE, overrides={"capture": {"sources": sources}}))
    sources[1]["clock_domain"] = sources[0]["clock_domain"]
    sources[0]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="source hash mismatch"):
        collect(load_config(CAPTURE, overrides={"capture": {"sources": sources}}))


def test_backwards_wall_clock_retains_file_order_and_invalidates_windows(tmp_path):
    _, records = collect()
    frames = [row for row in records["frames"] if row["rx_id"] == 1][:2]
    path = tmp_path / "regression.csiraw"
    path.write_bytes(
        b"CSIRAW01"
        + b"".join(
            struct.pack("<QH", stamp, 130) + bytes.fromhex(frame["wire_hex"])
            for stamp, frame in zip([2_000_000, 1_000_000], frames)
        )
    )
    cfg = load_config(
        CAPTURE,
        overrides={
            "capture": {
                "sources": [
                    {"path": str(path), "source_id": "rx1", "rx_id": 1, "clock_domain": "test:wall"}
                ]
            }
        },
    )
    report, output = collect(cfg)
    assert [row["seq_u32"] for row in output["frames"]] == [100, 103]
    assert output["frames"][1]["timing"]["clock_status"] == "excluded"
    assert output["frames"][1]["timing"]["exclusion_reason"]
    assert any(row["kind"] == "time_discontinuity" for row in output["issues"])
    assert total(report["metrics"], "source.valid_frame_count") == 2
    window = [
        v
        for v in report["metrics"]["values"]
        if v["key"] == "source.valid_frame_count" and v["aggregation_scope"] == "window"
    ]
    assert window[0]["value"] is None and window[0]["availability_reason"] == "unsupported"


def test_same_fixture_reinspection_changes_counter_epoch_but_not_results():
    first, _ = collect()
    second, _ = collect()
    assert first["manifest"]["config_hash"] == second["manifest"]["config_hash"]
    assert first["manifest"]["counts"] == second["manifest"]["counts"]
    assert first["manifest"]["stream_epoch"] != second["manifest"]["stream_epoch"]
    assert (
        first["metrics"]["provenance"]["counter_epoch"]
        != second["metrics"]["provenance"]["counter_epoch"]
    )


def test_source_time_snapshots_use_fixed_boundaries_with_half_open_windows():
    cfg = load_config(CAPTURE, overrides={"measurement": {"publish_period_s": 0.02}})
    report, records = collect(cfg)
    base = 1_700_000_000_000_000_000
    assert [int(s["as_of_ns"]) - base for s in records["metrics"]] == [
        20_000_000,
        40_000_000,
        60_000_000,
        80_000_000,
        100_000_000,
        110_000_001,
    ]
    # At 20ms the frame arriving exactly at 20ms has not been admitted.
    assert total(records["metrics"][0], "source.valid_frame_count") == 3
    assert report["manifest"]["metric_snapshot_policy"]["coalesced_empty_boundaries"] == 0


@pytest.mark.parametrize("name", ["examples", "ex)robot"])
def test_reference_tree_write_is_rejected_before_any_filesystem_mutation(
    tmp_path, monkeypatch, name
):
    workspace = tmp_path / "workspace"
    (workspace / "examples").mkdir(parents=True)
    (workspace / "docs/multiperson-csi-design").mkdir(parents=True)
    config = workspace / "config.yaml"
    config.write_text(f"includes: [{CAPTURE}]\n")

    def forbid_mkdir(*args, **kwargs):
        pytest.fail("reference path must be rejected before mkdir")

    monkeypatch.setattr(Path, "mkdir", forbid_mkdir)
    with pytest.raises(ValueError, match="read-only"):
        _output_directory(workspace / name / "forbidden-fixture-output", load_config(config))
