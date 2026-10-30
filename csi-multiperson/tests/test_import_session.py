"""S05 archive reproducibility, original bytes, and publication failure boundaries."""

from collections import Counter
import binascii
import builtins
import hashlib
import json
from pathlib import Path
import sqlite3
import struct

import pytest

from csi_mp.cli import main
from csi_mp.config import load_config
from csi_mp.data.provenance import COPY_BLOCK_BYTES, file_sha256
from csi_mp.data.store import SessionStore
from csi_mp.ingest import import_legacy as importer


PROJECT = Path(__file__).resolve().parents[1]
CAPTURE = PROJECT / "configs/capture.yaml"
BASE_NS = 1_700_000_000_000_000_123


def staging_for(output):
    return output.parent / f".{output.name}.importing"


def local_config(tmp_path):
    sources = load_config(CAPTURE).as_dict()["capture"]["sources"]
    for source in sources:
        original = Path(source["path"])
        copied = tmp_path / original.name
        copied.write_bytes(original.read_bytes())
        source["path"] = str(copied)
    return load_config(CAPTURE, overrides={"capture": {"sources": sources}})


def wire_record(ns, *, seq, timer, boot=1):
    body = struct.pack("<HBBIIbbBB112s", 0xC51D, 0, 0, seq, timer, -60, -90, 56, boot, bytes(112))
    frame = body + struct.pack("<H", binascii.crc_hqx(body, 0xFFFF))
    return struct.pack("<QH", ns, len(frame)) + frame


def test_reimport_preserves_observations_lineage_and_every_original_byte(tmp_path):
    cfg = load_config(CAPTURE)
    sources = cfg.as_dict()["capture"]["sources"]
    original = {item["source_id"]: Path(item["path"]).read_bytes() for item in sources}
    hashes = {key: hashlib.sha256(value).hexdigest() for key, value in original.items()}
    expected = json.loads((PROJECT / "fixtures/ingest_v1/expected.json").read_text())
    outputs = [tmp_path / "first.csisession", tmp_path / "second.csisession"]
    reports = [importer.import_session(None, output, cfg) for output in outputs]
    with (
        SessionStore.open_readonly(outputs[0]) as first,
        SessionStore.open_readonly(outputs[1]) as second,
    ):
        assert first.manifest == second.manifest
        assert list(first.observations()) == list(second.observations())
        assert list(first.epochs()) == list(second.epochs())
        assert list(first.issues()) == list(second.issues())
        assert reports[0].to_dict() == reports[1].to_dict()
        manifest = first.manifest
        rows = list(first.observations())
        assert len(rows) == reports[0].stored_frames == expected["totals"]["valid_frames"] == 13
        assert reports[0].accepted_frames == 13
        assert reports[0].usable_frames == sum(row["timing"]["usable"] for row in rows) == 10
        assert reports[0].unresolved_epochs == 2
        assert Counter(row["epochs"]["classification"] for row in rows)["duplicate"] == 1
        assert manifest["record_counts"]["epochs"] == len(list(first.epochs()))
        assert not manifest["features_materialized"]
        assert not manifest["windows_materialized"]
        assert manifest["signal"]["iq_order"]["value"] is None
        assert manifest["clock_policy"] == cfg.as_dict()["clock"]
        assert manifest["config_hash"] == cfg.config_hash
        for source in sources:
            key = source["source_id"]
            assert b"".join(first.source_bytes(key)) == original[key]
            assert file_sha256(source["path"]) == hashes[key] == expected["sources"][key]["sha256"]
        for row in rows:
            reconstructed = b"".join(
                original[span["source_id"]][
                    span["file_offset"] : span["file_offset"] + span["byte_count"]
                ]
                for span in row["spans"]
            )
            assert reconstructed == bytes.fromhex(row["wire_hex"])
            assert row["infer_available"] is None
            assert isinstance(row["timing"]["availability_ns"], str)
        tails = {
            (issue["layer"], issue["byte_count"])
            for issue in first.issues()
            if issue["kind"] == "truncated"
        }
        assert tails == {("raw_record", 15), ("wire_frame", 17)}
    assert reports[0].output_sha256 == file_sha256(outputs[0]) == file_sha256(outputs[1])
    assert not any(staging_for(output).exists() for output in outputs)


def test_archive_remains_readonly_and_self_contained_after_sources_are_removed(tmp_path):
    cfg = local_config(tmp_path)
    output = tmp_path / "portable name?#.csisession"
    report = importer.import_session(None, output, cfg)
    for source in cfg.as_dict()["capture"]["sources"]:
        Path(source["path"]).unlink()
    with SessionStore.open_readonly(output) as store:
        assert len(list(store.observations())) == 13
        for source in store.manifest["sources"]:
            assert (
                hashlib.sha256(b"".join(store.source_bytes(source["source_id"]))).hexdigest()
                == source["sha256"]
            )
        changed = store.manifest
        changed["session_id"] = "attempted mutation"
        row = next(store.observations())
        original_pair = list(row["iq"][0])
        row["iq"][0][0] = 99
        assert next(store.observations())["iq"][0] == original_pair
        assert store.manifest["session_id"] != changed["session_id"]
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            store._connection.execute("DELETE FROM records")
        with pytest.raises(KeyError):
            list(store.source_bytes("not-a-source"))
    assert file_sha256(output) == report.output_sha256


def test_future_source_suffix_changes_lineage_but_not_published_timing(tmp_path):
    raw = tmp_path / "growing.csiraw"
    prefix = b"CSIRAW01" + b"".join(
        wire_record(BASE_NS + i * 10_000_000, seq=100 + i, timer=900_000 + i * 10_000)
        for i in range(5)
    )
    source = {
        "path": str(raw),
        "source_id": "rx0",
        "rx_id": 0,
        "clock_domain": "test:bridge",
        "clock_kind": "unix_wall",
    }
    cfg = load_config(
        CAPTURE,
        overrides={"capture": {"sources": [source]}, "clock": {"legacy_delay_ns": 2_000_001}},
    )
    suffixes = [
        b"",
        wire_record(BASE_NS + 50_000_000, seq=105, timer=950_000),
        wire_record(BASE_NS + 900_000_000, seq=105, timer=1_000, boot=2),
    ]
    timings, reports = [], []
    for index, suffix in enumerate(suffixes):
        raw.write_bytes(prefix + suffix)
        output = tmp_path / f"suffix-{index}.csisession"
        reports.append(importer.import_session(raw, output, cfg))
        with SessionStore.open_readonly(output) as store:
            rows = list(store.observations())
            timings.append([row["timing"] for row in rows[:5]])
    assert timings[0] == timings[1] == timings[2]
    assert len({report.source_hashes for report in reports}) == 3
    assert len({report.provenance_id for report in reports}) == 3
    assert timings[0][0]["availability_ns"] == str(BASE_NS + 2_000_001)
    assert timings[0][0]["acquisition_ns"] == str(BASE_NS)


@pytest.mark.parametrize("target_kind", ["file", "dangling_symlink", "source_alias"])
def test_existing_destination_or_source_alias_is_never_overwritten(tmp_path, target_kind):
    cfg = local_config(tmp_path)
    source = Path(cfg.as_dict()["capture"]["sources"][0]["path"])
    original = source.read_bytes()
    output = tmp_path / "occupied.csisession"
    if target_kind == "file":
        output.write_bytes(b"existing result")
    elif target_kind == "dangling_symlink":
        output.symlink_to(tmp_path / "missing")
    else:
        output.symlink_to(source)
    with pytest.raises((FileExistsError, ValueError)):
        importer.import_session(None, output, cfg)
    assert source.read_bytes() == original
    assert not staging_for(output).exists()
    if target_kind == "file":
        assert output.read_bytes() == b"existing result"
    else:
        assert output.is_symlink()


def test_racing_publication_cannot_replace_the_winner(tmp_path, monkeypatch):
    output = tmp_path / "raced.csisession"
    native_publish = importer.publish_noreplace

    def race(temporary, destination):
        destination.write_bytes(b"another writer won")
        native_publish(temporary, destination)

    monkeypatch.setattr(importer, "publish_noreplace", race)
    with pytest.raises(FileExistsError):
        importer.import_session(None, output, load_config(CAPTURE))
    assert output.read_bytes() == b"another writer won"
    state = json.loads((staging_for(output) / "state.json").read_text())
    assert state["phase"] == "failed"


@pytest.mark.parametrize(
    "error_type,phase", [(OSError, "failed"), (KeyboardInterrupt, "interrupted")]
)
def test_failed_or_interrupted_import_is_not_published_and_retry_is_explicit(
    tmp_path, monkeypatch, error_type, phase
):
    output = tmp_path / "interrupted.csisession"

    def abort(self, sources, emit):
        raise error_type("injected conversion failure")

    monkeypatch.setattr(importer.RawLogImporter, "_convert", abort)
    with pytest.raises(error_type):
        importer.import_session(None, output, load_config(CAPTURE))
    assert not output.exists()
    state_path = staging_for(output) / "state.json"
    state = json.loads(state_path.read_text())
    assert state["phase"] == phase
    assert "injected conversion failure" in state["error"]
    with pytest.raises(ValueError, match="manifest"):
        SessionStore.open_readonly(staging_for(output) / "session.tmp")
    with pytest.raises(FileExistsError, match="active or interrupted"):
        importer.import_session(None, output, load_config(CAPTURE))
    assert json.loads(state_path.read_text()) == state


def test_validation_failure_keeps_final_destination_absent(tmp_path, monkeypatch):
    output = tmp_path / "invalid.csisession"

    def reject(self):
        raise ValueError("injected schema/hash validation failure")

    monkeypatch.setattr(SessionStore, "validate", reject)
    with pytest.raises(ValueError, match="validation failure"):
        importer.import_session(None, output, load_config(CAPTURE))
    assert not output.exists()
    assert json.loads((staging_for(output) / "state.json").read_text())["phase"] == "failed"


def test_source_mutation_after_snapshot_aborts_publication(tmp_path, monkeypatch):
    cfg = local_config(tmp_path)
    source = Path(cfg.as_dict()["capture"]["sources"][0]["path"])
    output = tmp_path / "changed-source.csisession"
    native_convert = importer.RawLogImporter._convert

    def mutate(self, sources, emit):
        summary = native_convert(self, sources, emit)
        with source.open("ab") as stream:
            stream.write(b"unexpected new bytes")
        return summary

    monkeypatch.setattr(importer.RawLogImporter, "_convert", mutate)
    with pytest.raises(ValueError, match="changed during import"):
        importer.import_session(None, output, cfg)
    assert not output.exists()


@pytest.mark.parametrize("damage", ["source", "observation", "manifest", "missing_record"])
def test_reader_rejects_archive_corruption(tmp_path, damage):
    output = tmp_path / f"corrupt-{damage}.csisession"
    importer.import_session(None, output, load_config(CAPTURE))
    with sqlite3.connect(output) as connection:
        if damage == "source":
            blob = connection.execute(
                "SELECT data FROM source_chunks WHERE source_id='rx0' AND ordinal=0"
            ).fetchone()[0]
            connection.execute(
                "UPDATE source_chunks SET data=? WHERE source_id='rx0' AND ordinal=0",
                (bytes([blob[0] ^ 1]) + blob[1:],),
            )
        elif damage == "observation":
            row = json.loads(
                connection.execute(
                    "SELECT payload FROM records WHERE kind='observations' AND ordinal=0"
                ).fetchone()[0]
            )
            row["iq"][0][0] = (row["iq"][0][0] + 1 + 128) % 256 - 128
            connection.execute(
                "UPDATE records SET payload=? WHERE kind='observations' AND ordinal=0",
                (json.dumps(row, sort_keys=True, separators=(",", ":")),),
            )
        elif damage == "manifest":
            manifest = json.loads(
                connection.execute("SELECT value FROM metadata WHERE key='manifest'").fetchone()[0]
            )
            manifest["session_id"] = "a different session"
            connection.execute(
                "UPDATE metadata SET value=? WHERE key='manifest'", (json.dumps(manifest),)
            )
        else:
            connection.execute("DELETE FROM records WHERE kind='observations' AND ordinal=12")
    with pytest.raises(ValueError, match="hash|count"):
        SessionStore.open_readonly(output)


def test_unrelated_sqlite_database_is_not_a_session(tmp_path):
    output = tmp_path / "unrelated.sqlite"
    with sqlite3.connect(output) as connection:
        connection.execute("CREATE TABLE sample (value INTEGER)")
    with pytest.raises(ValueError, match="application ID"):
        SessionStore.open_readonly(output)


def test_protected_reference_tree_cannot_be_an_output(tmp_path):
    # No file in the real reference tree is created, even on a failed assertion.
    fake_root = tmp_path / "workspace"
    (fake_root / "examples").mkdir(parents=True)
    (fake_root / "docs/multiperson-csi-design").mkdir(parents=True)
    config = fake_root / "config.yaml"
    config.write_text(f"includes: [{CAPTURE}]\n")
    output = fake_root / "examples" / "forbidden.csisession"
    with pytest.raises(ValueError, match="read-only"):
        importer.import_session(None, output, load_config(config))
    assert list((fake_root / "examples").iterdir()) == []


def test_cli_import_reports_only_completed_archive_and_refuses_repeat(tmp_path, capsys):
    output = tmp_path / "cli.csisession"
    args = ["import", "--config", str(CAPTURE), "--output", str(output)]
    assert main(args) == 0
    captured = capsys.readouterr()
    report = json.loads(captured.out)
    assert captured.err == ""
    assert report["status"] == "completed"
    assert report["stored_frames"] == 13
    assert report["output_sha256"] == file_sha256(output)
    assert main(args) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "exists" in captured.err
    assert file_sha256(output) == report["output_sha256"]


def test_samples_only_hdf5_cannot_be_claimed_as_raw_session(tmp_path):
    h5py = pytest.importorskip("h5py")
    np = pytest.importorskip("numpy")
    source = tmp_path / "samples-only.h5"
    with h5py.File(source, "w") as file:
        file.create_dataset("samples/X", data=np.zeros((2, 9, 280), dtype=np.float32))
    before = file_sha256(source)
    cfg = load_config(
        CAPTURE,
        overrides={
            "capture": {
                "sources": [
                    {
                        "source_id": "legacy",
                        "path": str(source),
                        "format": "legacy_hdf5",
                    }
                ]
            }
        },
    )
    output = tmp_path / "samples.csisession"
    with pytest.raises(ValueError, match="links|samples"):
        importer.import_session(None, output, cfg)
    assert not output.exists()
    assert file_sha256(source) == before


def test_hdf5_missing_fields_survive_session_storage_as_unknown(tmp_path):
    h5py = pytest.importorskip("h5py")
    np = pytest.importorskip("numpy")
    source = tmp_path / "legacy.h5"
    iq = np.arange(112, dtype=np.int8).reshape(1, 56, 2)
    with h5py.File(source, "w") as file:
        file.create_dataset("links/00/iq", data=iq)
    before = file_sha256(source)
    cfg = load_config(
        CAPTURE,
        overrides={
            "capture": {
                "sources": [
                    {
                        "source_id": "legacy",
                        "path": str(source),
                        "format": "legacy_hdf5",
                    }
                ]
            }
        },
    )
    output = tmp_path / "legacy.csisession"
    report = importer.import_session(None, output, cfg)
    assert report.stored_frames == report.unresolved_epochs == 1
    assert report.usable_frames == 0
    with SessionStore.open_readonly(output) as store:
        row = next(store.observations())
        assert row["iq"] == iq[0].tolist()
        assert row["wire_hex"] is None
        assert row["epochs"]["rx_epoch"] is None
        assert row["epochs"]["tx_epoch"] is None
        assert row["timing"]["availability_ns"] is None
        assert row["timing"]["acquisition_ns"] is None
        assert row["source_ref"]["dataset_path"] == "/links/00"
        assert row["source_ref"]["row_index"] == 0
        assert b"".join(store.source_bytes("legacy")) == source.read_bytes()
    assert file_sha256(source) == before


@pytest.mark.parametrize("stage", ["finish", "close"])
def test_writer_completion_failure_never_publishes_a_session(tmp_path, monkeypatch, stage):
    output = tmp_path / f"writer-{stage}.csisession"
    writer_class = importer._SessionWriter
    native_close = writer_class.close
    raised = False

    def fail_finish(self, manifest):
        raise OSError("injected commit failure")

    def fail_close_once(self):
        nonlocal raised
        native_close(self)
        if not raised:
            raised = True
            raise OSError("injected close failure")

    if stage == "finish":
        monkeypatch.setattr(writer_class, "finish", fail_finish)
    else:
        monkeypatch.setattr(writer_class, "close", fail_close_once)
    with pytest.raises(OSError, match="injected .* failure"):
        importer.import_session(None, output, load_config(CAPTURE))
    assert not output.exists()
    state = json.loads((staging_for(output) / "state.json").read_text())
    assert state["phase"] == "failed"
    assert "failure" in state["error"]


def test_original_sources_are_opened_only_for_binary_reading(tmp_path, monkeypatch):
    cfg = local_config(tmp_path)
    paths = {Path(source["path"]) for source in cfg.as_dict()["capture"]["sources"]}
    before = {path: path.read_bytes() for path in paths}
    native_open = Path.open
    source_modes = []

    def guarded_open(path, mode="r", *args, **kwargs):
        if path in paths:
            source_modes.append((path, mode))
            assert mode == "rb", "an original source was opened for mutation or text conversion"
        return native_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    importer.import_session(None, tmp_path / "readonly.csisession", cfg)
    assert {path for path, _ in source_modes} == paths
    assert all(mode == "rb" for _, mode in source_modes)
    assert {path: path.read_bytes() for path in paths} == before


def test_configured_source_hash_mismatch_fails_before_publication(tmp_path):
    cfg = local_config(tmp_path)
    sources = cfg.as_dict()["capture"]["sources"]
    original = {source["path"]: Path(source["path"]).read_bytes() for source in sources}
    sources[0]["sha256"] = "0" * 64
    cfg = load_config(CAPTURE, overrides={"capture": {"sources": sources}})
    output = tmp_path / "bad-source-hash.csisession"
    with pytest.raises(ValueError, match="source hash mismatch"):
        importer.import_session(None, output, cfg)
    assert not output.exists()
    assert {path: Path(path).read_bytes() for path in original} == original


def test_header_only_raw_source_is_a_valid_empty_archive(tmp_path):
    raw = tmp_path / "empty.csiraw"
    raw.write_bytes(b"CSIRAW01")
    cfg = load_config(
        CAPTURE,
        overrides={
            "capture": {
                "sources": [
                    {
                        "path": str(raw),
                        "source_id": "empty-rx0",
                        "rx_id": 0,
                        "clock_domain": "empty:bridge",
                    }
                ]
            }
        },
    )
    output = tmp_path / "empty.csisession"
    report = importer.import_session(raw, output, cfg)
    assert report.stored_frames == report.accepted_frames == report.usable_frames == 0
    assert report.unresolved_epochs == 0
    with SessionStore.open_readonly(output) as store:
        assert list(store.observations()) == list(store.epochs()) == list(store.issues()) == []
        assert b"".join(store.source_bytes("empty-rx0")) == b"CSIRAW01"
        assert store.manifest["counts"]["raw_records"] == 0
        assert store.manifest["counts"]["payload_bytes"] == 0


def test_control_records_and_confirmed_epoch_evidence_are_preserved(tmp_path):
    raw = tmp_path / "controlled.csiraw"
    raw.write_bytes(
        b"CSIRAW01"
        + b"".join(
            wire_record(BASE_NS + i * 10_000_000, seq=seq, timer=900_000 + i * 10_000)
            for i, seq in enumerate([100, 101, 0, 1])
        )
    )
    control = {
        "kind": "tx_start",
        "device_id": 0,
        "available_ns": str(BASE_NS + 15_000_000),
        "clock_domain": "control:bridge",
        "clock_kind": "unix_wall",
        "source_ref": "operator-log:tx0-start:17",
    }
    cfg = load_config(
        CAPTURE,
        overrides={
            "capture": {
                "sources": [
                    {
                        "path": str(raw),
                        "source_id": "rx0",
                        "rx_id": 0,
                        "clock_domain": "control:bridge",
                    }
                ],
                "controls": [control],
            }
        },
    )
    output = tmp_path / "controls.csisession"
    report = importer.import_session(raw, output, cfg)
    assert report.unresolved_epochs == 0
    with SessionStore.open_readonly(output) as store:
        assert list(store.records("controls")) == [control]
        rows = list(store.observations())
        assert rows[0]["epochs"]["tx_epoch"] == rows[1]["epochs"]["tx_epoch"]
        assert rows[2]["epochs"]["tx_epoch"] == rows[3]["epochs"]["tx_epoch"]
        assert rows[1]["epochs"]["tx_epoch"] != rows[2]["epochs"]["tx_epoch"]
        assert rows[2]["epochs"]["classification"] == "tx_reset"
        events = [event for event in store.epochs() if event["kind"] == "tx_reset"]
        assert len(events) == 1
        assert events[0]["certainty"] == "confirmed"
        assert events[0]["available"]["ns"] == control["available_ns"]
        assert events[0]["evidence_refs"] == [control["source_ref"]]


def test_wire_source_span_crosses_embedded_source_chunk_boundary(tmp_path):
    raw = tmp_path / "large.csiraw"
    contents = bytearray(b"CSIRAW01")
    target_header = COPY_BLOCK_BYTES - 50
    timestamp = BASE_NS
    while len(contents) < target_header:
        payload_size = min(65535, target_header - len(contents) - 10)
        assert payload_size >= 0
        contents.extend(struct.pack("<QH", timestamp, payload_size))
        contents.extend(b"x" * payload_size)
        timestamp += 1_000
    frame = wire_record(timestamp, seq=100, timer=900_000)[10:]
    # The first 100-byte source span starts 40 bytes before the storage boundary.
    contents.extend(struct.pack("<QH", timestamp, 100) + frame[:100])
    timestamp += 1_000
    contents.extend(struct.pack("<QH", timestamp, 30) + frame[100:])
    raw.write_bytes(contents)
    cfg = load_config(
        CAPTURE,
        overrides={
            "capture": {
                "sources": [
                    {
                        "path": str(raw),
                        "source_id": "large-rx0",
                        "rx_id": 0,
                        "clock_domain": "large:bridge",
                    }
                ]
            }
        },
    )
    output = tmp_path / "large.csisession"
    report = importer.import_session(raw, output, cfg)
    assert report.stored_frames == report.usable_frames == 1
    with SessionStore.open_readonly(output) as store:
        chunks = list(store.source_bytes("large-rx0"))
        assert len(chunks) == 2
        assert b"".join(chunks) == contents
        rows = list(store.observations())
        assert len(rows) == 1
        assert len(rows[0]["spans"]) == 2
        span = rows[0]["spans"][0]
        assert span["file_offset"] < COPY_BLOCK_BYTES < span["file_offset"] + span["byte_count"]
        reconstructed = b"".join(
            contents[part["file_offset"] : part["file_offset"] + part["byte_count"]]
            for part in rows[0]["spans"]
        )
        assert reconstructed == frame == bytes.fromhex(rows[0]["wire_hex"])
        assert rows[0]["bridge_complete"]["ns"] == str(timestamp)


def test_raw_import_does_not_require_h5py(tmp_path, monkeypatch):
    native_import = builtins.__import__

    def without_h5py(name, *args, **kwargs):
        if name == "h5py" or name.startswith("h5py."):
            raise ModuleNotFoundError("h5py deliberately unavailable")
        return native_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_h5py)
    output = tmp_path / "raw-without-h5py.csisession"
    report = importer.import_session(None, output, load_config(CAPTURE))
    assert report.stored_frames == 13
    with SessionStore.open_readonly(output) as store:
        assert len(list(store.observations())) == 13
