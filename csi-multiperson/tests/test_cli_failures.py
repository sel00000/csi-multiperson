"""Regressions for reporting failures and known-empty source measurements."""

import json
from pathlib import Path

import pytest
import yaml

import csi_mp.cli as cli
from csi_mp.config import ConfigError, load_config
from csi_mp.ingest.inspect import inspect_capture

PROJECT = Path(__file__).resolve().parents[1]
CAPTURE = PROJECT / "configs/capture.yaml"


@pytest.mark.parametrize("failure_stage", ["flush", "close"])
def test_jsonl_writer_failure_cannot_publish_completed_run(
    tmp_path, monkeypatch, capsys, failure_stage
):
    output = tmp_path / f"failed-{failure_stage}"
    native_open = Path.open

    class FailingStream:
        def __init__(self, stream):
            self.stream = stream

        def __enter__(self):
            return self

        def __getattr__(self, name):
            return getattr(self.stream, name)

        def flush(self):
            if failure_stage == "flush":
                raise OSError("simulated JSONL flush failure")
            return self.stream.flush()

        def close(self):
            try:
                self.flush()
            finally:
                self.stream.close()
            if failure_stage == "close":
                raise OSError("simulated JSONL close failure")

        def __exit__(self, *args):
            self.close()

    def failing_open(path, *args, **kwargs):
        stream = native_open(path, *args, **kwargs)
        return FailingStream(stream) if path.suffix == ".jsonl" else stream

    monkeypatch.setattr(Path, "open", failing_open)
    result = cli.main(["inspect", "--config", str(CAPTURE), "--output", str(output)])
    captured = capsys.readouterr()
    assert result == 2
    assert f"JSONL {failure_stage} failure" in captured.err
    assert captured.out == ""
    assert not (output / "manifest.json").exists()
    failure = json.loads((output / "failure.json").read_text())
    assert failure["status"] == "failed"


def test_failure_report_write_failure_retains_primary_error(tmp_path, monkeypatch, capsys):
    output = tmp_path / "unwritable-report"
    attempted = []

    def failed_write(path, value):
        attempted.append(path.name)
        if path.name == "failure.json":
            raise OSError("secondary failure-report disk error")
        raise OSError("primary output disk error")

    monkeypatch.setattr(cli, "_write", failed_write)
    assert cli.main(["inspect", "--config", str(CAPTURE), "--output", str(output)]) == 2
    captured = capsys.readouterr()
    assert "primary output disk error" in captured.err
    assert captured.out == ""
    assert attempted == ["config.resolved.json", "failure.json"]
    assert not (output / "manifest.json").exists()


def test_header_only_source_has_measured_zero_counters_with_source_label(tmp_path):
    raw = tmp_path / "empty.csiraw"
    raw.write_bytes(b"CSIRAW01")
    cfg = load_config(
        CAPTURE,
        overrides={
            "capture": {
                "sources": [
                    {
                        "source_id": "empty-rx0",
                        "path": str(raw),
                        "rx_id": 0,
                        "clock_domain": "fixture:empty",
                    }
                ]
            }
        },
    )
    report = inspect_capture(cfg)
    assert report["manifest"]["counts"] == {
        "raw_records": 0,
        "payload_bytes": 0,
        "valid_frames": 0,
        "usable_frames": 0,
    }
    for key in (
        "source.raw_record_count",
        "source.bytes_received_count",
        "source.valid_frame_count",
        "source.crc_error_count",
    ):
        values = [
            value
            for value in report["metrics"]["values"]
            if value["key"] == key and value["aggregation_scope"] == "cumulative"
        ]
        assert len(values) == 1
        assert values[0]["labels"] == {"source_id": "empty-rx0"}
        assert values[0]["value"] == 0
        assert values[0]["availability_reason"] is None
        assert values[0]["exposure_ns"] is None
    assert raw.read_bytes() == b"CSIRAW01"


@pytest.mark.parametrize(
    ("section", "field"),
    [
        ("model", "d"),
        ("input", "grid_step_ms"),
        ("measurement", "window_s"),
        ("measurement", "publish_period_s"),
    ],
)
def test_huge_numeric_configuration_is_a_named_config_error(section, field):
    with pytest.raises(ConfigError, match=rf"{section}\.{field}"):
        load_config(CAPTURE, overrides={section: {field: 10**400}})


@pytest.mark.parametrize("sinks", [[], ["jsonl"]])
def test_ingest_sink_selection_controls_streaming_metrics_only(tmp_path, capsys, sinks):
    config_path = tmp_path / "capture.yaml"
    config_path.write_text(
        yaml.safe_dump({"includes": [str(CAPTURE)], "measurement": {"sinks": sinks}})
    )
    output = tmp_path / "report"
    assert cli.main(["inspect", "--config", str(config_path), "--output", str(output)]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "completed"
    assert (output / "metrics.jsonl").exists() == ("jsonl" in sinks)
    assert (output / "metrics.json").is_file()
    assert (output / "dashboard.json").is_file()
    assert (output / "frames.jsonl").is_file()


def test_unimplemented_dashboard_sink_is_rejected():
    with pytest.raises(ConfigError, match="measurement.sinks"):
        load_config(CAPTURE, overrides={"measurement": {"sinks": ["dashboard"]}})
