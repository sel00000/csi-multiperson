"""Canonical provenance and streaming, read-only source snapshots for S05."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import os
from pathlib import Path

from csi_mp.config import canonical_json, content_hash


SESSION_SCHEMA_ID = "csi_mp.session.v1"
OBSERVATION_SCHEMA_ID = "csi_mp.session_observation.v1"
COPY_BLOCK_BYTES = 1024 * 1024


def file_sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def implementation_sha256():
    package = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for path in sorted(package.rglob("*.py")):
        digest.update(path.relative_to(package).as_posix().encode() + b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def file_identity(stat):
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def snapshot_source(source, destination):
    """Copy once through an rb descriptor, hashing exactly the copied bytes."""
    path = Path(source["path"])
    digest = hashlib.sha256()
    with path.open("rb") as reader, Path(destination).open("xb") as writer:
        initial = file_identity(os.fstat(reader.fileno()))
        while block := reader.read(COPY_BLOCK_BYTES):
            digest.update(block)
            writer.write(block)
        writer.flush()
        os.fsync(writer.fileno())
        if file_identity(os.fstat(reader.fileno())) != initial:
            raise ValueError(f"source {source['source_id']} changed while snapshotting")
    if file_identity(path.stat()) != initial:
        raise ValueError(f"source {source['source_id']} changed while snapshotting")
    sha256 = digest.hexdigest()
    if source.get("sha256") is not None and source["sha256"] != sha256:
        raise ValueError(f"source {source['source_id']}: source hash mismatch")
    return {
        **source,
        "path": str(destination),
        "original_path": str(path),
        "sha256": sha256,
        "size_bytes": initial[2],
    }, initial


def verify_original(source, identity):
    path = Path(source["original_path"])
    with path.open("rb") as stream:
        before = file_identity(os.fstat(stream.fileno()))
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
        after = file_identity(os.fstat(stream.fileno()))
    if before != identity or after != identity or file_identity(path.stat()) != identity:
        raise ValueError(f"source {source['source_id']} changed during import")
    if digest != source["sha256"]:
        raise ValueError(f"source {source['source_id']} changed during import (hash mismatch)")


@dataclass(frozen=True)
class MigrationReport:
    source_hashes: tuple[str, ...]
    schema_id: str
    output_sha256: str
    accepted_frames: int
    rejected_by_reason: dict[str, int]
    unresolved_epochs: int
    legacy_assumptions: tuple[str, ...]
    stored_frames: int
    usable_frames: int
    provenance_id: str

    def to_dict(self):
        return asdict(self)


def seal_manifest(manifest):
    """Hash deterministic lineage, without runtime IDs or destination paths."""
    manifest = dict(manifest)
    manifest["provenance_id"] = content_hash(manifest)
    return manifest


def verify_manifest(manifest):
    body = dict(manifest)
    claimed = body.pop("provenance_id", None)
    if claimed != content_hash(body):
        raise ValueError("session manifest provenance hash mismatch")
    if body.get("schema_id") != SESSION_SCHEMA_ID or body.get("status") != "completed":
        raise ValueError("session is incomplete or has an unsupported schema")
    # Also reject non-finite values instead of accepting JSON NaN extensions.
    canonical_json(body)
