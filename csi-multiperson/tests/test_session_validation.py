"""Schema checks reject importer regressions even when every digest is resealed."""

import copy
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3

import pytest

from csi_mp.config import canonical_json, content_hash, load_config
from csi_mp.data.provenance import seal_manifest
from csi_mp.data.store import RECORD_KINDS, SessionStore
from csi_mp.ingest.import_legacy import import_session


CAPTURE = Path(__file__).resolve().parents[1] / "configs/capture.yaml"


@pytest.fixture(scope="module")
def raw_archive(tmp_path_factory):
    path = tmp_path_factory.mktemp("schema-raw") / "valid.csisession"
    import_session(None, path, load_config(CAPTURE))
    return path


@pytest.fixture(scope="module")
def legacy_archive(tmp_path_factory):
    h5py = pytest.importorskip("h5py")
    import numpy as np

    directory = tmp_path_factory.mktemp("schema-legacy")
    original = directory / "legacy.h5"
    with h5py.File(original, "w") as h5:
        h5.create_dataset("links/00/iq", data=np.zeros((2, 56, 2), dtype="i1"))
        h5.create_dataset("links/00/t_ns", data=np.array([10, 20], dtype="u8"))
    config = load_config(
        CAPTURE,
        overrides={
            "capture": {
                "sources": [
                    {
                        "source_id": "legacy",
                        "path": str(original),
                        "format": "legacy_hdf5",
                        "clock_domain": "legacy:bridge",
                    }
                ]
            }
        },
    )
    path = directory / "valid.csisession"
    import_session(None, path, config)
    return path


def rewritten(archive, destination, mutate):
    """Simulate faulty importer output, not unsealed on-disk corruption."""
    shutil.copyfile(archive, destination)
    with sqlite3.connect(destination) as connection:
        manifest = json.loads(
            connection.execute("SELECT value FROM metadata WHERE key='manifest'").fetchone()[0]
        )
        records = {
            kind: [
                json.loads(row[0])
                for row in connection.execute(
                    "SELECT payload FROM records WHERE kind=? ORDER BY ordinal", (kind,)
                )
            ]
            for kind in RECORD_KINDS
        }
        mutate(manifest, records, connection)
        connection.execute("DELETE FROM records")
        for kind in RECORD_KINDS:
            digest = hashlib.sha256()
            for ordinal, value in enumerate(records[kind]):
                encoded = canonical_json(value).encode()
                digest.update(encoded + b"\n")
                connection.execute(
                    "INSERT INTO records VALUES (?, ?, ?, ?)",
                    (kind, ordinal, encoded.decode(), hashlib.sha256(encoded).hexdigest()),
                )
            manifest["record_hashes"][kind] = digest.hexdigest()
        manifest.pop("provenance_id")
        connection.execute(
            "UPDATE metadata SET value=? WHERE key='manifest'",
            (canonical_json(seal_manifest(manifest)),),
        )
    return destination


def set_nested(value, path, replacement):
    for key in path[:-1]:
        value = value[key]
    value[path[-1]] = replacement


@pytest.mark.parametrize(
    "field,replacement",
    [
        (("epochs",), {}),
        (("timing",), {}),
        (("wire_hex",), "00" * 130),
        (("seq_u32",), 8888),
        (("iq", 0, 0), 127),
        (("spans", 0, "byte_count"), 1),
        (("spans", 0, "file_offset"), 0),
        (("spans", 0, "chunk_index"), -1),
        (("timing", "availability_ns"), 1),
        (("timing", "clock_model_id"), "different-model"),
        (("timing", "clock_model", "model_id"), "different-model"),
        (("timing", "clock_model", "rate_denominator"), "0"),
        (("timing", "clock_model", "sample_count"), True),
        (("timing", "usable"), False),
        (("timing", "clock_error_ns"), "100"),
        (("timing", "observation_id"), "different-observation"),
        (("epochs", "usable"), False),
        (("epochs", "timer_unwrapped_us"), 1),
        (("epochs", "rx_epoch"), "rx:2:epoch:0"),
        (("bridge_complete", "ns"), "00012"),
        (("bridge_complete", "kind"), "monotonic"),
        (("crc_valid",), False),
        (("infer_available",), {"ns": "1", "domain": "made-up", "kind": "monotonic"}),
        (("ingest_ordinal",), True),
    ],
)
def test_resealed_raw_schema_regressions_are_rejected(raw_archive, tmp_path, field, replacement):
    def mutate(manifest, records, connection):
        set_nested(records["observations"][0], field, copy.deepcopy(replacement))

    path = rewritten(raw_archive, tmp_path / "broken.csisession", mutate)
    with pytest.raises(ValueError):
        SessionStore.open_readonly(path)


@pytest.mark.parametrize(
    "count", ["stored_frames", "accepted_frames", "usable_frames", "unresolved_epochs"]
)
def test_resealed_summary_counts_must_agree_with_observations(raw_archive, tmp_path, count):
    def mutate(manifest, records, connection):
        owner = manifest if count == "unresolved_epochs" else manifest["counts"]
        owner[count] += 1

    path = rewritten(raw_archive, tmp_path / "counts.csisession", mutate)
    with pytest.raises(ValueError, match="count mismatch"):
        SessionStore.open_readonly(path)


def test_raw_source_receiver_mapping_cannot_disagree_with_wire(raw_archive, tmp_path):
    def mutate(manifest, records, connection):
        manifest["sources"][0]["rx_id"] = 2
        manifest["config"]["capture"]["sources"][0]["rx_id"] = 2
        manifest["config_hash"] = content_hash(manifest["config"])

    path = rewritten(raw_archive, tmp_path / "mapping.csisession", mutate)
    with pytest.raises(ValueError, match="trusted source/RX/clock"):
        SessionStore.open_readonly(path)


def test_epoch_archive_cannot_omit_observation_events(raw_archive, tmp_path):
    def mutate(manifest, records, connection):
        records["epochs"].clear()
        manifest["record_counts"]["epochs"] = 0

    path = rewritten(raw_archive, tmp_path / "missing-epochs.csisession", mutate)
    with pytest.raises(ValueError, match="epoch events differ"):
        SessionStore.open_readonly(path)


def test_source_chunks_require_fixed_block_offsets_even_when_bytes_hash_matches(
    raw_archive, tmp_path
):
    def mutate(manifest, records, connection):
        source_id = manifest["sources"][0]["source_id"]
        data = connection.execute(
            "SELECT data FROM source_chunks WHERE source_id=? AND ordinal=0", (source_id,)
        ).fetchone()[0]
        split = len(data) // 2
        connection.execute("DELETE FROM source_chunks WHERE source_id=?", (source_id,))
        connection.executemany(
            "INSERT INTO source_chunks VALUES (?, ?, ?, ?)",
            [(source_id, 0, 0, data[:split]), (source_id, 1, split, data[split:])],
        )

    path = rewritten(raw_archive, tmp_path / "chunks.csisession", mutate)
    with pytest.raises(ValueError, match="fixed-size blocks"):
        SessionStore.open_readonly(path)


@pytest.mark.parametrize(
    "field,replacement",
    [
        (("wire_hex",), "00" * 130),
        (("crc_valid",), True),
        (("epochs", "rx_epoch"), "rx:0:epoch:0"),
        (("epochs", "usable"), 0),
        (("timing", "acquisition_ns"), "10"),
        (("timing", "availability_ns"), "10"),
        (("timing", "clock_status"), "ready"),
        (("timing", "usable"), True),
        (("source_ref", "row_index"), -1),
        (("source_ref", "dataset_path"), "/links/22"),
        (("seq",), 0),
        (("t_ns",), None),
        (("unknown_fields",), []),
        (("bridge_complete", "ns"), "999"),
    ],
)
def test_legacy_resealed_rows_cannot_invent_missing_data(
    legacy_archive, tmp_path, field, replacement
):
    def mutate(manifest, records, connection):
        set_nested(records["observations"][0], field, copy.deepcopy(replacement))

    path = rewritten(legacy_archive, tmp_path / "broken-legacy.csisession", mutate)
    with pytest.raises(ValueError):
        SessionStore.open_readonly(path)


@pytest.mark.parametrize("bad_value", [None, [], "record"])
def test_invalid_record_json_shapes_fail_as_value_errors(raw_archive, tmp_path, bad_value):
    def mutate(manifest, records, connection):
        records["observations"][0] = bad_value

    path = rewritten(raw_archive, tmp_path / "shape.csisession", mutate)
    with pytest.raises(ValueError, match="mapping"):
        SessionStore.open_readonly(path)


def test_readonly_reader_retains_the_validated_snapshot(raw_archive, tmp_path):
    path = tmp_path / "snapshot.csisession"
    shutil.copyfile(raw_archive, path)
    with sqlite3.connect(path) as writer:
        writer.execute("PRAGMA journal_mode=WAL")
        with SessionStore.open_readonly(path) as reader:
            manifest = reader.manifest
            observations = list(reader.observations())
            writer.execute("DELETE FROM records WHERE kind='observations'")
            writer.commit()
            assert reader.manifest == manifest
            assert list(reader.observations()) == observations
            assert reader.validate() == manifest
    with pytest.raises(ValueError, match="count/hash mismatch"):
        SessionStore.open_readonly(path)
