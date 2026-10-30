"""Self-contained SQLite sessions with canonical JSON records and source bytes.

Only the importer owns a writer. Readers use SQLite's mode=ro and query_only,
verify the completed manifest, record hashes/counts and embedded source hashes,
and return detached values. No h5py dependency is needed for raw sessions.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import struct

import numpy as np

from csi_mp.config import canonical_json, content_hash
from csi_mp.data.schema import EpochEvidence, EpochEvent, EpochResult, SourceSpan, Stamp, WireFrame
from csi_mp.ingest.crc import crc16_ccitt
from csi_mp.ingest.events import ClockModel, TimedObservation
from csi_mp.ingest.protocol import FRAME_BODY, FRAME_LEN, FRAME_MAGIC
from csi_mp.telemetry.schema import json_value
from .provenance import (
    COPY_BLOCK_BYTES,
    OBSERVATION_SCHEMA_ID,
    seal_manifest,
    verify_manifest,
)


APPLICATION_ID = 0x4353494D
RECORD_KINDS = ("observations", "epochs", "issues", "controls")


def _payload(value):
    return canonical_json(value).encode("utf-8")


class _SessionWriter:
    def __init__(self, path):
        self.path = Path(path)
        with self.path.open("xb"):
            pass
        self.connection = sqlite3.connect(self.path)
        self.counts = Counter({kind: 0 for kind in RECORD_KINDS})
        self.hashes = {kind: hashlib.sha256() for kind in RECORD_KINDS}
        self.connection.executescript(f"""
            PRAGMA application_id={APPLICATION_ID};
            PRAGMA user_version=1;
            PRAGMA journal_mode=DELETE;
            PRAGMA synchronous=FULL;
            CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL) WITHOUT ROWID;
            CREATE TABLE records (
                kind TEXT NOT NULL, ordinal INTEGER NOT NULL, payload TEXT NOT NULL,
                sha256 TEXT NOT NULL, PRIMARY KEY (kind, ordinal)
            ) WITHOUT ROWID;
            CREATE TABLE source_chunks (
                source_id TEXT NOT NULL, ordinal INTEGER NOT NULL,
                byte_offset INTEGER NOT NULL, data BLOB NOT NULL,
                PRIMARY KEY (source_id, ordinal)
            ) WITHOUT ROWID;
        """)

    def emit(self, kind, record):
        if kind not in RECORD_KINDS:
            raise ValueError(f"unknown session record kind: {kind}")
        payload = _payload(record)
        self.connection.execute(
            "INSERT INTO records VALUES (?, ?, ?, ?)",
            (kind, self.counts[kind], payload.decode(), hashlib.sha256(payload).hexdigest()),
        )
        self.hashes[kind].update(payload + b"\n")
        self.counts[kind] += 1

    def add_source(self, source):
        digest = hashlib.sha256()
        offset, ordinal = 0, 0
        with Path(source["path"]).open("rb") as stream:
            while block := stream.read(COPY_BLOCK_BYTES):
                self.connection.execute(
                    "INSERT INTO source_chunks VALUES (?, ?, ?, ?)",
                    (source["source_id"], ordinal, offset, block),
                )
                digest.update(block)
                offset += len(block)
                ordinal += 1
        if offset != source["size_bytes"] or digest.hexdigest() != source["sha256"]:
            raise ValueError(f"snapshot {source['source_id']} changed before storage")

    def finish(self, manifest):
        manifest = seal_manifest(
            {
                **manifest,
                "record_counts": dict(self.counts),
                "record_hashes": {kind: value.hexdigest() for kind, value in self.hashes.items()},
            }
        )
        self.connection.execute(
            "INSERT INTO metadata VALUES ('manifest', ?)", (canonical_json(manifest),)
        )
        self.connection.commit()
        self.close()
        return manifest

    def close(self):
        if self.connection is not None:
            self.connection.close()
            self.connection = None


class SessionStore:
    def __init__(self, path, connection):
        self.path = path
        self._connection = connection

    @classmethod
    def open_readonly(cls, path):
        path = Path(path).resolve(strict=True)
        if not path.is_file():
            raise ValueError("session path must be a regular file")
        connection = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)
        store = cls(path, connection)
        try:
            connection.execute("PRAGMA query_only=ON")
            # Hold the same read snapshot for validation and every later read.
            connection.execute("BEGIN")
            store.validate()
        except (sqlite3.Error, KeyError, TypeError, IndexError) as exc:
            connection.close()
            raise ValueError(f"invalid session store: {exc}") from exc
        except BaseException:
            connection.close()
            raise
        return store

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        self._connection.close()

    @property
    def manifest(self):
        row = self._connection.execute("SELECT value FROM metadata WHERE key='manifest'").fetchone()
        if row is None:
            raise ValueError("session has no completed manifest")
        return json.loads(row[0])

    def records(self, kind):
        if kind not in RECORD_KINDS:
            raise ValueError(f"unknown session record kind: {kind}")
        for (payload,) in self._connection.execute(
            "SELECT payload FROM records WHERE kind=? ORDER BY ordinal", (kind,)
        ):
            yield json.loads(payload)

    def observations(self):
        return self.records("observations")

    def epochs(self):
        return self.records("epochs")

    def issues(self):
        return self.records("issues")

    def source_bytes(self, source_id):
        """Yield exact original bytes in bounded blocks, including rejected tails."""
        if source_id not in {source["source_id"] for source in self.manifest["sources"]}:
            raise KeyError(source_id)
        for (data,) in self._connection.execute(
            "SELECT data FROM source_chunks WHERE source_id=? ORDER BY ordinal", (source_id,)
        ):
            yield data

    def validate(self):
        try:
            return self._validate()
        except (
            sqlite3.Error,
            KeyError,
            TypeError,
            IndexError,
            AttributeError,
            OverflowError,
        ) as exc:
            raise ValueError(f"invalid session store: {exc}") from exc

    def _validate(self):
        connection = self._connection
        if connection.execute("PRAGMA application_id").fetchone()[0] != APPLICATION_ID:
            raise ValueError("unsupported session application ID")
        if connection.execute("PRAGMA user_version").fetchone()[0] != 1:
            raise ValueError("unsupported session storage version")
        if connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise ValueError("session SQLite integrity check failed")
        manifest = self.manifest
        if not isinstance(manifest, dict):
            raise ValueError("session manifest must be a mapping")
        verify_manifest(manifest)
        sources = _validate_sources(manifest)
        unknown_sources = {
            row[0] for row in connection.execute("SELECT DISTINCT source_id FROM source_chunks")
        } - sources.keys()
        if unknown_sources:
            raise ValueError("session has unmanifested source bytes")
        for source_id, source in sources.items():
            digest = hashlib.sha256()
            offset, expected_ordinal = 0, 0
            for ordinal, byte_offset, data in connection.execute(
                "SELECT ordinal, byte_offset, data FROM source_chunks "
                "WHERE source_id=? ORDER BY ordinal",
                (source_id,),
            ):
                if (
                    type(ordinal) is not int
                    or type(byte_offset) is not int
                    or ordinal != expected_ordinal
                    or byte_offset != offset
                    or byte_offset != ordinal * COPY_BLOCK_BYTES
                    or not isinstance(data, bytes)
                    or not 0 < len(data) <= COPY_BLOCK_BYTES
                    or len(data) != min(COPY_BLOCK_BYTES, source["size_bytes"] - offset)
                ):
                    raise ValueError("session source chunks are not contiguous fixed-size blocks")
                digest.update(data)
                offset += len(data)
                expected_ordinal += 1
            if offset != source["size_bytes"] or digest.hexdigest() != source["sha256"]:
                raise ValueError(f"embedded source {source_id} hash/size mismatch")
        kinds = {row[0] for row in connection.execute("SELECT DISTINCT kind FROM records")}
        if kinds - set(RECORD_KINDS):
            raise ValueError("session has unsupported record kinds")
        if set(manifest["record_counts"]) != set(RECORD_KINDS) or set(
            manifest["record_hashes"]
        ) != set(RECORD_KINDS):
            raise ValueError("session record count/hash schema mismatch")
        totals = Counter(stored_frames=0, accepted_frames=0, usable_frames=0, unresolved_epochs=0)
        classifications = Counter()
        expected_events, recorded_events = {}, {}
        legacy_rows = Counter()
        previous_ingest = -1
        previous_legacy = None
        for kind in RECORD_KINDS:
            _integer(manifest["record_counts"][kind], "record count")
            digest = hashlib.sha256()
            count = 0
            for ordinal, payload, sha256 in connection.execute(
                "SELECT ordinal, payload, sha256 FROM records WHERE kind=? ORDER BY ordinal",
                (kind,),
            ):
                value = json.loads(payload)
                if not isinstance(value, dict):
                    raise ValueError(f"session {kind} record must be a mapping")
                encoded = _payload(value)
                if ordinal != count or encoded.decode() != payload:
                    raise ValueError("session records are not contiguous canonical JSON")
                if hashlib.sha256(encoded).hexdigest() != sha256:
                    raise ValueError(f"session {kind} record hash mismatch")
                if kind == "observations":
                    _validate_observation(value, sources, manifest, connection)
                    ingest = value["ingest_ordinal"]
                    if ingest < previous_ingest:
                        raise ValueError("session observations regress in ingest order")
                    previous_ingest = ingest
                    totals["stored_frames"] += 1
                    totals["accepted_frames"] += 1
                    totals["usable_frames"] += value["timing"]["usable"]
                    classification = value["epochs"]["classification"]
                    classifications[classification] += 1
                    totals["unresolved_epochs"] += classification in {"ambiguous", "unknown"}
                    for event in value["epochs"]["events"]:
                        identity = event["event_id"]
                        if identity in expected_events and expected_events[identity] != event:
                            raise ValueError("conflicting observation epoch event identity")
                        expected_events[identity] = event
                    if value["adapter"] == "legacy_links":
                        ref = value["source_ref"]
                        key = (ref["source_id"], ref["dataset_path"])
                        location = (*key, ref["row_index"])
                        if ref["row_index"] != legacy_rows[key] or ingest != count:
                            raise ValueError("legacy rows are not contiguous in archive order")
                        if previous_legacy is not None and location <= previous_legacy:
                            raise ValueError("legacy archive order is not canonical")
                        previous_legacy = location
                        legacy_rows[key] += 1
                elif kind == "epochs":
                    _epoch_event(value)
                    if value["event_id"] in recorded_events:
                        raise ValueError("duplicate archived epoch event")
                    recorded_events[value["event_id"]] = value
                elif kind == "controls":
                    _validate_control(value, manifest, count)
                elif kind == "issues":
                    _validate_issue(value, sources, manifest)
                digest.update(encoded + b"\n")
                count += 1
            if (
                count != manifest["record_counts"][kind]
                or digest.hexdigest() != manifest["record_hashes"][kind]
            ):
                raise ValueError(f"session {kind} count/hash mismatch")
        for name, expected in totals.items():
            actual = manifest[name] if name == "unresolved_epochs" else manifest["counts"][name]
            _integer(actual, name)
            if actual != expected:
                raise ValueError(f"session {name} count mismatch")
        for value in manifest["counts"].values():
            _integer(value, "session count")
        if expected_events != recorded_events:
            raise ValueError("archived epoch events differ from observation evidence")
        if manifest["record_counts"]["controls"] != len(manifest["config"]["capture"]["controls"]):
            raise ValueError("session control count mismatch")
        if manifest["importer"] == "csiraw01":
            if dict(classifications) != manifest["classifications"]:
                raise ValueError("session epoch classification counts mismatch")
        else:
            expected_rows = {
                (source_id, f"/links/{name}"): group["row_count"]
                for source_id, metadata in manifest["source_metadata"].items()
                for name, group in metadata["link_metadata"].items()
                if group["row_count"]
            }
            if dict(legacy_rows) != expected_rows:
                raise ValueError("legacy observation rows differ from source metadata")
        return manifest


def _integer(value, name, *, minimum=0, maximum=None):
    if type(value) is not int or value < minimum or (maximum is not None and value > maximum):
        raise ValueError(f"{name}: expected integer in the supported range")
    return value


def _text(value, name):
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name}: expected nonempty string")
    return value


def _strings(value, name):
    if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
        raise ValueError(f"{name}: expected a string list")
    return tuple(value)


def _decimal(value, name, *, optional=False, signed=False):
    if optional and value is None:
        return None
    pattern = r"-?(0|[1-9][0-9]*)" if signed else r"0|[1-9][0-9]*"
    if not isinstance(value, str) or re.fullmatch(pattern, value) is None:
        raise ValueError(f"{name}: expected a decimal integer string")
    result = int(value)
    if str(result) != value:
        raise ValueError(f"{name}: expected a canonical decimal integer string")
    return result


def _stamp(value):
    if not isinstance(value, dict) or set(value) != {"ns", "domain", "kind"}:
        raise ValueError("clock stamp schema mismatch")
    _decimal(value["ns"], "stamp.ns")
    return Stamp.from_dict(value)


def _validate_sources(manifest):
    if manifest["config_hash"] != content_hash(manifest["config"]):
        raise ValueError("session config hash mismatch")
    if manifest["importer"] not in {"csiraw01", "legacy_hdf5"}:
        raise ValueError("unsupported session importer")
    if (
        manifest["clock_policy"] != manifest["config"]["clock"]
        or manifest["epoch_policy"] != manifest["config"]["epoch"]
    ):
        raise ValueError("session policy differs from resolved configuration")
    if not isinstance(manifest["sources"], list):
        raise ValueError("session sources must be a list")
    sources = {}
    for source in manifest["sources"]:
        source_id = _text(source["source_id"], "source_id")
        if source_id in sources:
            raise ValueError("session sources must be unique")
        _integer(source["size_bytes"], "source.size_bytes")
        if re.fullmatch(r"[0-9a-f]{64}", source["sha256"]) is None:
            raise ValueError("source.sha256: expected lowercase SHA256")
        if source.get("format", "csiraw01") != manifest["importer"]:
            raise ValueError("source format differs from the session importer")
        if source.get("clock_kind", "unix_wall") != "unix_wall":
            raise ValueError("legacy sources require unix_wall timestamps")
        if manifest["importer"] == "csiraw01":
            _integer(source["rx_id"], "source.rx_id", maximum=2)
            _text(source["clock_domain"], "source.clock_domain")
        elif source.get("clock_domain") is not None:
            _text(source["clock_domain"], "source.clock_domain")
        sources[source_id] = source
    mappings = manifest["config"]["capture"]["sources"]
    if not sources or len(mappings) != len(sources):
        raise ValueError("session source/config mappings must be nonempty and one-to-one")
    mapped_ids = set()
    for mapping in mappings:
        source_id = mapping["source_id"]
        source = sources[source_id]
        if source_id in mapped_ids:
            raise ValueError("session config has duplicate source identifiers")
        mapped_ids.add(source_id)
        for name, value in mapping.items():
            if name == "sha256" and value is None:
                continue
            if source.get(name) != value:
                raise ValueError(f"session source/config mapping differs at {source_id}.{name}")
    if manifest["importer"] == "csiraw01":
        synthetic = all(source.get("provenance") == "synthetic" for source in sources.values())
        if manifest["synthetic"] is not synthetic or manifest["availability_basis"] != (
            "synthetic" if synthetic else "legacy_bridge_proxy"
        ):
            raise ValueError("session availability provenance differs from source declarations")
    if manifest["importer"] == "legacy_hdf5":
        if set(manifest["source_metadata"]) != set(sources):
            raise ValueError("legacy source metadata identifiers mismatch")
        for source_id, metadata in manifest["source_metadata"].items():
            if (
                metadata["source_sha256"] != sources[source_id]["sha256"]
                or metadata["original_path"] != sources[source_id]["path"]
            ):
                raise ValueError("legacy source metadata lineage mismatch")
            for name, group in metadata["link_metadata"].items():
                if re.fullmatch(r"[0-2][0-2]", name) is None:
                    raise ValueError("legacy source metadata link name mismatch")
                _integer(group["row_count"], "legacy row count")
    return sources


def _source_slice(connection, source_id, offset, length):
    """Read only requested byte spans using the fixed source-chunk primary key."""
    parts = []
    while length:
        ordinal, local = divmod(offset, COPY_BLOCK_BYTES)
        take = min(length, COPY_BLOCK_BYTES - local)
        row = connection.execute(
            "SELECT substr(data, ?, ?) FROM source_chunks WHERE source_id=? AND ordinal=?",
            (local + 1, take, source_id, ordinal),
        ).fetchone()
        if row is None or not isinstance(row[0], bytes) or len(row[0]) != take:
            raise ValueError("observation source span cannot be read from embedded bytes")
        parts.append(row[0])
        offset += take
        length -= take
    return b"".join(parts)


def _source_span(value, sources, *, allow_empty=False):
    span = SourceSpan(**value)
    source = sources[span.source_id]
    if (
        span.source_sha256 != source["sha256"]
        or (span.byte_count == 0 and not allow_empty)
        or span.file_offset + span.byte_count > source["size_bytes"]
    ):
        raise ValueError("observation source span is invalid")
    return span


def _epoch_event(value):
    event = EpochEvent(**{**value, "available": _stamp(value["available"])})
    _text(event.event_id, "epoch.event_id")
    _strings(value["evidence_refs"], "epoch.evidence_refs")
    expected_links = (
        [3 * event.device_id + index for index in range(3)]
        if event.scope == "rx"
        else [3 * index + event.device_id for index in range(3)]
    )
    allowed = (
        {"rx_reset", "timer_wrap", "timer_regression"}
        if event.scope == "rx"
        else {"tx_reset", "seq_wrap"}
    )
    if value["affected_links"] != expected_links or event.kind not in allowed:
        raise ValueError("epoch event scope/link mismatch")
    if json_value(event) != value:
        raise ValueError("epoch event schema mismatch")
    return event


def _raw_epochs(value, frame):
    events = tuple(_epoch_event(item) for item in value["events"])
    epochs = EpochResult(**{**value, "events": events})
    _text(epochs.rx_epoch, "epochs.rx_epoch")
    _text(epochs.link_segment, "epochs.link_segment")
    _strings(value["evidence_refs"], "epochs.evidence_refs")
    if epochs.tx_epoch is not None:
        _text(epochs.tx_epoch, "epochs.tx_epoch")
    if type(epochs.usable) is not bool:
        raise ValueError("epochs.usable must be boolean")
    if epochs.usable != (epochs.classification not in {"duplicate", "ambiguous"}):
        raise ValueError("epoch classification/usability mismatch")
    if epochs.usable and (epochs.tx_epoch is None or epochs.timer_unwrapped_us is None):
        raise ValueError("usable epoch requires TX epoch and unwrapped timer")
    if not epochs.rx_epoch.startswith(f"rx:{frame.rx_id}:epoch:"):
        raise ValueError("epoch belongs to another RX")
    if epochs.tx_epoch is not None and not epochs.tx_epoch.startswith(f"tx:{frame.tx_id}:epoch:"):
        raise ValueError("epoch belongs to another TX")
    if (
        epochs.timer_unwrapped_us is not None
        and epochs.timer_unwrapped_us % (1 << 32) != frame.rx_timer_u32_us
    ):
        raise ValueError("unwrapped epoch timer differs from wire timer")
    if json_value(epochs) != value:
        raise ValueError("epoch result schema mismatch")
    return epochs


def _clock_model(value, reference, policy):
    if value is None:
        return None
    decoded = dict(value)
    decoded.pop("error_basis")
    for name in (
        "reference_timer_us",
        "reference_ns",
        "rate_numerator",
        "rate_denominator",
        "span_us",
        "residual_ns",
        "calibration_error_ns",
    ):
        decoded[name] = _decimal(
            value[name],
            f"clock_model.{name}",
            optional=name in {"residual_ns", "calibration_error_ns"},
            signed=name == "reference_ns",
        )
    for name in ("model_id", "rx_epoch", "segment_id"):
        _text(decoded[name], f"clock_model.{name}")
    for name in ("rate_numerator", "rate_denominator", "sample_count", "window_count"):
        _integer(decoded[name], f"clock_model.{name}", minimum=1)
    _strings(decoded["transform_ids"], "clock_model.transform_ids")
    if type(decoded["rate_limited"]) is not bool:
        raise ValueError("clock_model.rate_limited must be boolean")
    model = ClockModel(**decoded)
    if not model.window_count <= min(model.sample_count, policy["window_size"]):
        raise ValueError("clock model window/sample count mismatch")
    ready = (
        model.window_count >= policy["warmup_samples"] and model.span_us >= policy["min_span_us"]
    )
    if model.status != ("ready" if ready else "warming_up"):
        raise ValueError("clock model warmup status mismatch")
    if (model.residual_ns is None) == ready:
        raise ValueError("clock model residual/warmup mismatch")
    if model.to_dict() != value:
        raise ValueError("clock model schema mismatch")
    identity = {
        "policy": policy,
        "model": replace(model, model_id="").to_dict(),
        "reference": reference.to_dict(),
    }
    if model.model_id != "clock:" + content_hash(identity):
        raise ValueError("clock model identity mismatch")
    return model


def _raw_timing(value, frame, epochs, manifest):
    if value["schema_id"] != "csi_mp.timing.v1":
        raise ValueError("unsupported observation timing schema")
    reference = _stamp(value["clock_reference"])
    model = _clock_model(value["clock_model"], reference, manifest["clock_policy"])
    _text(value["observation_id"], "timing.observation_id")
    _strings(value["transform_ids"], "timing.transform_ids")
    if type(value["usable"]) is not bool:
        raise ValueError("timing.usable must be boolean")
    if value["exclusion_reason"] is not None:
        _text(value["exclusion_reason"], "timing.exclusion_reason")
    timing = TimedObservation(
        observation_id=value["observation_id"],
        frame=frame,
        epochs=epochs,
        acquisition_ns=_decimal(value["acquisition_ns"], "acquisition_ns", optional=True),
        availability_ns=_decimal(value["availability_ns"], "availability_ns"),
        availability_basis=value["availability_basis"],
        clock_domain=value["clock_domain"],
        clock_kind=value["clock_kind"],
        clock_reference=reference,
        clock_model=model,
        clock_error_ns=_decimal(value["clock_error_ns"], "clock_error_ns", optional=True),
        clock_status=value["clock_status"],
        exclusion_reason=value["exclusion_reason"],
        transform_ids=tuple(value["transform_ids"]),
        legacy_delay_ns=_decimal(value["legacy_delay_ns"], "legacy_delay_ns"),
    )
    if timing.to_dict() != value:
        raise ValueError("timing derived fields/schema mismatch")
    if (timing.clock_domain, timing.clock_kind) != (
        frame.bridge_complete.domain,
        frame.bridge_complete.kind,
    ) or timing.transform_ids:
        raise ValueError("raw importer cannot invent clock transforms")
    if (
        timing.legacy_delay_ns != manifest["clock_policy"]["legacy_delay_ns"]
        or timing.availability_ns != frame.bridge_complete.ns + timing.legacy_delay_ns
        or timing.availability_basis != manifest["availability_basis"]
    ):
        raise ValueError("raw availability policy mismatch")
    expected_error = None
    if (
        model is not None
        and model.residual_ns is not None
        and model.calibration_error_ns is not None
    ):
        expected_error = model.residual_ns + model.calibration_error_ns
    if timing.clock_error_ns != expected_error:
        raise ValueError("clock error differs from published model")
    if timing.usable and (
        model.rx_epoch != epochs.rx_epoch
        or timing.acquisition_ns != model.estimate(epochs.timer_unwrapped_us)
    ):
        raise ValueError("acquisition estimate differs from published epoch/model")
    identity = {
        "source": [(s.source_id, s.chunk_index, s.file_offset, s.byte_count) for s in frame.spans],
        "wire": frame.wire_bytes.hex(),
        "rx": frame.rx_id,
        "tx": frame.tx_id,
    }
    if timing.observation_id != "observation:" + content_hash(identity):
        raise ValueError("observation identity differs from source/wire lineage")


def _validate_observation(value, sources, manifest, connection):
    if value.get("schema_id") != OBSERVATION_SCHEMA_ID:
        raise ValueError("unsupported session observation schema")
    if value.get("adapter") not in {"csiraw01", "legacy_links"}:
        raise ValueError("unsupported session observation adapter")
    for field in ("rx_id", "tx_id"):
        if type(value.get(field)) is not int or value[field] not in range(3):
            raise ValueError(f"observation {field} outside 0..2")
    iq = value.get("iq")
    if (
        not isinstance(iq, list)
        or len(iq) != 56
        or any(
            not isinstance(pair, list)
            or len(pair) != 2
            or any(type(v) is not int or not -128 <= v <= 127 for v in pair)
            for pair in iq
        )
    ):
        raise ValueError("observation iq must be signed int8 [56,2]")
    for field in ("epochs", "timing"):
        if not isinstance(value.get(field), dict):
            raise ValueError(f"observation {field} is required")
    _integer(value["ingest_ordinal"], "observation.ingest_ordinal")
    if value["adapter"] == "csiraw01":
        if manifest["importer"] != "csiraw01":
            raise ValueError("raw observation in a legacy links session")
        wire = bytes.fromhex(value["wire_hex"])
        if len(wire) != FRAME_LEN or not value["spans"]:
            raise ValueError("raw observation requires wire bytes and source spans")
        if value["wire_hex"] != wire.hex():
            raise ValueError("wire_hex must use canonical lowercase hexadecimal")
        spans = tuple(_source_span(span, sources) for span in value["spans"])
        if sum(span.byte_count for span in spans) != FRAME_LEN:
            raise ValueError("raw observation spans must total one wire frame")
        if len({span.source_id for span in spans}) != 1:
            raise ValueError("raw observation cannot span multiple sources")
        embedded = b"".join(
            _source_slice(connection, s.source_id, s.file_offset, s.byte_count) for s in spans
        )
        if embedded != wire:
            raise ValueError("wire bytes differ from embedded source spans")
        if crc16_ccitt(wire[:-2]) != struct.unpack_from("<H", wire, FRAME_LEN - 2)[0]:
            raise ValueError("raw observation wire CRC mismatch")
        magic, rx, tx, seq, timer, rssi, noise, length, boot, iq_bytes = FRAME_BODY.unpack(
            wire[:-2]
        )
        if magic != FRAME_MAGIC or length != 56:
            raise ValueError("raw observation wire header mismatch")
        source = sources[spans[0].source_id]
        bridge = _stamp(value["bridge_complete"])
        if (
            rx != source["rx_id"]
            or bridge.domain != source["clock_domain"]
            or bridge.kind != "unix_wall"
        ):
            raise ValueError("raw observation differs from trusted source/RX/clock mapping")
        expected = {
            "rx_id": rx,
            "tx_id": tx,
            "seq_u32": seq,
            "rx_timer_u32_us": timer,
            "rssi_dbm": rssi,
            "noise_dbm": noise,
            "rx_boot_u8": boot,
        }
        if any(type(value[name]) is not int or value[name] != expected[name] for name in expected):
            raise ValueError("observation fields differ from wire values")
        if np.frombuffer(iq_bytes, dtype=np.int8).reshape(56, 2).tolist() != value["iq"]:
            raise ValueError("observation IQ differs from wire values")
        if (
            value["length_valid"] is not True
            or value["crc_valid"] is not True
            or value["signal_mode"] is not None
            or value["infer_available"] is not None
            or value["availability_basis"] != manifest["availability_basis"]
            or value["frame_schema_id"] != "csi_mp.inspected_frame.v1"
        ):
            raise ValueError("raw observation validity/provenance mismatch")
        frame = WireFrame(
            spans=spans,
            wire_bytes=wire,
            **expected,
            iq=np.asarray(value["iq"], dtype=np.int8),
            signal_mode=None,
            bridge_complete=bridge,
            infer_available=None,
            ingest_ordinal=value["ingest_ordinal"],
        )
        epochs = _raw_epochs(value["epochs"], frame)
        _raw_timing(value["timing"], frame, epochs, manifest)
    else:
        _validate_legacy(value, sources, manifest)


def _validate_legacy(value, sources, manifest):
    if manifest["importer"] != "legacy_hdf5":
        raise ValueError("legacy observation in a raw session")
    ref = value["source_ref"]
    source = sources[ref["source_id"]]
    name = f"{value['rx_id']}{value['tx_id']}"
    metadata = manifest["source_metadata"][ref["source_id"]]["link_metadata"][name]
    if ref["source_sha256"] != source["sha256"] or ref["dataset_path"] != f"/links/{name}":
        raise ValueError("legacy observation source lineage mismatch")
    _integer(ref["row_index"], "legacy row_index", maximum=metadata["row_count"] - 1)
    if (
        value["wire_hex"] is not None
        or value["spans"] != []
        or value["infer_available"] is not None
        or value["signal_mode"] is not None
        or value.get("crc_valid") is not None
        or value.get("length_valid") is not None
    ):
        raise ValueError("legacy links cannot invent wire, CRC or inference arrival")
    expected_epochs = {
        "rx_epoch": None,
        "tx_epoch": None,
        "link_segment": None,
        "timer_unwrapped_us": None,
        "classification": "unknown",
        "usable": False,
        "evidence_refs": [],
        "events": [],
    }
    if value["epochs"].get("usable") is not False or value["epochs"] != expected_epochs:
        raise ValueError("legacy links cannot invent epochs")
    domain = source.get("clock_domain")
    expected_timing = {
        "acquisition_ns": None,
        "availability_ns": None,
        "availability_basis": "unknown",
        "clock_domain": domain,
        "clock_kind": "unix_wall" if domain else None,
        "clock_reference": None,
        "clock_model_id": None,
        "clock_segment_id": None,
        "clock_error_ns": None,
        "clock_status": "unknown",
        "usable": False,
        "exclusion_reason": "legacy_epoch_and_availability_unknown",
    }
    if value["timing"].get("usable") is not False or value["timing"] != expected_timing:
        raise ValueError("legacy links cannot invent acquisition/availability/clock models")
    for name, lower, upper in (
        ("seq", 0, 2**32 - 1),
        ("boot_id", 0, 255),
        ("rssi", -128, 127),
        ("noise", -128, 127),
    ):
        if value[name] is not None:
            _integer(value[name], f"legacy {name}", minimum=lower, maximum=upper)
    for name in ("t_ns", "esp_us"):
        number = _decimal(value[name], f"legacy {name}", optional=True)
        if number is not None and number > 2**64 - 1:
            raise ValueError(f"legacy {name} exceeds uint64")
    bridge = (
        Stamp(int(value["t_ns"]), domain, "unix_wall").to_dict()
        if value["t_ns"] is not None and domain
        else None
    )
    if value["bridge_complete"] != bridge:
        raise ValueError("legacy bridge stamp differs from stored t_ns/known clock domain")
    unknown = _strings(value["unknown_fields"], "legacy unknown_fields")
    reasons = value["unknown_reasons"]
    if list(unknown) != sorted(set(unknown)) or set(unknown) != set(reasons):
        raise ValueError("legacy unknown fields/reasons mismatch")
    for reason in reasons.values():
        _text(reason, "legacy unknown reason")
    required_unknown = {
        "wire_hex",
        "infer_available",
        "rx_epoch",
        "tx_epoch",
        "acquisition_ns",
        "availability_ns",
        "signal_mode",
    }
    required_unknown.update(
        name
        for name in ("t_ns", "esp_us", "seq", "boot_id", "rssi", "noise")
        if value[name] is None
    )
    if not domain:
        required_unknown.add("clock_domain")
    if bridge is None:
        required_unknown.add("bridge_complete")
    if required_unknown != set(unknown):
        raise ValueError("legacy missing fields need explicit unknown reasons")
    present = set(metadata["dataset_attributes"])
    for name in ("t_ns", "esp_us", "seq", "boot_id", "rssi", "noise"):
        if (value[name] is None) == (name in present):
            raise ValueError("legacy field presence differs from source metadata")


def _validate_control(value, manifest, ordinal):
    if manifest["importer"] != "csiraw01":
        raise ValueError("legacy links cannot contain raw epoch controls")
    if value != manifest["config"]["capture"]["controls"][ordinal]:
        raise ValueError("archived control differs from source configuration")
    EpochEvidence(
        kind=value["kind"],
        device_id=value["device_id"],
        available=_stamp(
            {
                "ns": value["available_ns"],
                "domain": value["clock_domain"],
                "kind": value["clock_kind"],
            }
        ),
        source_ref=value["source_ref"],
    )


def _validate_issue(value, sources, manifest):
    _text(value["kind"], "issue.kind")
    _text(value["detail"], "issue.detail")
    if manifest["importer"] == "csiraw01":
        if value["layer"] not in {"raw_record", "wire_frame"}:
            raise ValueError("unsupported parse issue layer")
        _integer(value["byte_count"], "issue.byte_count")
        _integer(value["event_sequence"], "issue.event_sequence", minimum=1)
        spans = [_source_span(span, sources, allow_empty=True) for span in value["spans"]]
        diagnostic = value["kind"] == "time_discontinuity"
        if diagnostic and value["byte_count"] != 0:
            raise ValueError("time discontinuity is a diagnostic, not discarded bytes")
        if not diagnostic and sum(span.byte_count for span in spans) != value["byte_count"]:
            raise ValueError("parse issue byte count differs from source spans")
        if value["available"] is not None:
            _stamp(value["available"])
    else:
        source = sources[value["source_id"]]
        if value["source_sha256"] != source["sha256"]:
            raise ValueError("legacy issue source hash mismatch")
        if value["kind"] != "legacy_unknown_fields":
            raise ValueError("unsupported legacy issue kind")
        _strings(value["fields"], "legacy issue fields")


def publish_noreplace(temporary, destination):
    """Atomic same-filesystem publication, including racing/broken symlink refusal.

    link+unlink has rename's visibility semantics without POSIX rename's
    overwrite race. Unsupported filesystems fail closed; no copying fallback.
    """
    with Path(temporary).open("rb") as stream:
        os.fsync(stream.fileno())
    os.link(temporary, destination)
