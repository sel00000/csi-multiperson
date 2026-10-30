"""Read-only, atomic migration from CSIRAW01 or legacy HDF5 /links."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import shutil
import sqlite3

from csi_mp.config import AppConfig, canonical_json, validate_config
from csi_mp.data.provenance import (
    MigrationReport,
    OBSERVATION_SCHEMA_ID,
    SESSION_SCHEMA_ID,
    file_sha256,
    implementation_sha256,
    snapshot_source,
    verify_original,
)
from csi_mp.data.store import SessionStore, _SessionWriter, publish_noreplace


def _destination(path, spec, sources):
    # Do not resolve the final component: a broken symlink is an existing target.
    path = Path(path).absolute()
    destination = path.parent.resolve() / path.name
    roots = {
        parent
        for location in (Path(__file__).resolve(), spec.path.resolve())
        for parent in location.parents
        if (parent / "docs/multiperson-csi-design").is_dir() and (parent / "examples").is_dir()
    }
    for root in roots:
        for name in ("examples", "ex)robot"):
            protected = root / name
            if destination == protected or protected in destination.parents:
                raise ValueError("output: reference examples and papers are read-only")
    if any(destination.resolve() == Path(source["path"]).resolve() for source in sources):
        raise ValueError("output: cannot replace a source")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"output already exists; never overwritten: {destination}")
    return destination


def _state(directory, phase, *, error=None):
    value = {"schema_id": "csi_mp.import_state.v1", "phase": phase}
    if error is not None:
        value["error"] = error
    temporary = directory / "state.new"
    temporary.write_text(canonical_json(value) + "\n", encoding="utf-8")
    temporary.replace(directory / "state.json")


def _source_manifest(source, source_format):
    return {
        "format": source_format,
        "raw_schema_id": "CSIRAW01" if source_format == "csiraw01" else "legacy.links.v1",
        "wire_schema_id": "csi130.v1" if source_format == "csiraw01" else None,
        "clock_kind": "unix_wall",
        "clock_domain": source.get("clock_domain"),
        **{key: value for key, value in source.items() if key not in {"path", "original_path"}},
        "path": source["original_path"],
    }


def _raw_import(sources, spec, emit):
    from .inspect import inspect_capture

    config = spec.as_dict()
    config["capture"]["sources"] = [
        {key: value for key, value in source.items() if key not in {"original_path", "size_bytes"}}
        for source in sources
    ]
    snapshot_config = AppConfig(spec.path, canonical_json(config))

    def on_record(kind, record):
        if kind == "frames":
            emit(
                "observations",
                {
                    **record,
                    "frame_schema_id": record["schema_id"],
                    "schema_id": OBSERVATION_SCHEMA_ID,
                    "adapter": "csiraw01",
                },
            )
        elif kind in {"issues", "epochs"}:
            emit(kind, record)

    report = inspect_capture(snapshot_config, emit=on_record)["manifest"]
    # Inspection stream UUID, host execution clock, telemetry and temporary
    # source/config paths are intentionally not part of archival lineage.
    return {
        "counts": {
            "stored_frames": report["counts"]["valid_frames"],
            "accepted_frames": report["counts"]["valid_frames"],
            "usable_frames": report["timing_summary"]["usable_observations"],
            "raw_records": report["counts"]["raw_records"],
            "payload_bytes": report["counts"]["payload_bytes"],
        },
        "rejected_by_reason": report["issue_counts"],
        "unresolved_epochs": report["classifications"].get("ambiguous", 0),
        "legacy_assumptions": report["legacy_assumptions"],
        "classifications": report["classifications"],
        "epoch_events": report["epoch_events"],
        "clock_snapshot": report["clock_snapshot"],
        "timing_summary": report["timing_summary"],
        "availability_basis": report["availability_basis"],
        "synthetic": report["synthetic"],
        "ordering": "source-preserving merge by bridge ns, source_id, raw byte offset",
    }


class LegacyImporter:
    """Common snapshot/validation/commit lifecycle; subclasses own parsing."""

    source_format = None

    def __init__(self, spec: AppConfig):
        if not isinstance(spec, AppConfig):
            raise TypeError("import spec must be a validated AppConfig")
        validate_config(spec)
        self.spec = spec

    def import_atomic(self, sources, output) -> MigrationReport:
        config = self.spec.as_dict()
        mappings = sorted(config["capture"]["sources"], key=lambda source: source["source_id"])
        if not mappings:
            raise ValueError("import requires at least one trusted source mapping")
        if {source.get("format", "csiraw01") for source in mappings} != {self.source_format}:
            raise ValueError(f"{type(self).__name__} requires only {self.source_format} sources")
        paths = [str(Path(path).resolve()) for path in sources]
        if Counter(paths) != Counter(source["path"] for source in mappings):
            raise ValueError("source paths must exactly match capture.sources in the import spec")
        for source in mappings:
            if not Path(source["path"]).is_file():
                raise ValueError(f"source must be a regular file: {source['path']}")
        destination = _destination(output, self.spec, mappings)
        destination.parent.mkdir(parents=True, exist_ok=True)
        staging = destination.parent / f".{destination.name}.importing"
        try:
            staging.mkdir()
        except FileExistsError as exc:
            raise FileExistsError(
                f"active or interrupted import at {staging}; inspect state.json and preserve "
                "or remove that staging directory before retrying"
            ) from exc
        writer = None
        published = False
        try:
            _state(staging, "snapshotting")
            snapshots, identities = [], []
            for index, source in enumerate(mappings):
                snapshot, identity = snapshot_source(source, staging / f"source-{index:04d}.bin")
                snapshots.append(snapshot)
                identities.append(identity)
            _state(staging, "writing")
            temporary = staging / "session.tmp"
            writer = _SessionWriter(temporary)
            for source in snapshots:
                writer.add_source(source)
            for control in config["capture"]["controls"]:
                writer.emit("controls", control)
            summary = self._convert(snapshots, writer.emit)
            manifest = writer.finish(
                {
                    "schema_id": SESSION_SCHEMA_ID,
                    "status": "completed",
                    "session_id": config["capture"]["session_id"],
                    "storage": "sqlite_source_chunks_and_canonical_json_v1",
                    "importer": self.source_format,
                    "implementation_sha256": implementation_sha256(),
                    "config_hash": self.spec.config_hash,
                    "config": config,
                    "sources": [
                        _source_manifest(source, self.source_format) for source in snapshots
                    ],
                    "epoch_policy": config["epoch"],
                    "clock_policy": config["clock"],
                    "signal": {
                        **{
                            key: {
                                "value": None,
                                "provenance": "unknown",
                                "evidence": "not supplied; original IQ pair and SC order preserved",
                            }
                            for key in ("iq_order", "sc_map_id")
                        },
                        **config["capture"]["signal"],
                    },
                    "feature_id": config["model"]["feature_id"],
                    "features_materialized": False,
                    "windows_materialized": False,
                    "acceptance_status": "not_evaluated",
                    **summary,
                }
            )
            _state(staging, "validating")
            with SessionStore.open_readonly(temporary) as store:
                if store.manifest != manifest:
                    raise ValueError("validated manifest differs from the written manifest")
            _state(staging, "verifying_sources")
            for source, identity in zip(snapshots, identities, strict=True):
                verify_original(source, identity)
            digest = file_sha256(temporary)
            report = MigrationReport(
                source_hashes=tuple(source["sha256"] for source in snapshots),
                schema_id=SESSION_SCHEMA_ID,
                output_sha256=digest,
                accepted_frames=manifest["counts"]["accepted_frames"],
                rejected_by_reason=manifest["rejected_by_reason"],
                unresolved_epochs=manifest["unresolved_epochs"],
                legacy_assumptions=tuple(manifest["legacy_assumptions"]),
                stored_frames=manifest["counts"]["stored_frames"],
                usable_frames=manifest["counts"]["usable_frames"],
                provenance_id=manifest["provenance_id"],
            )
            _state(staging, "validated")
            publish_noreplace(temporary, destination)
            published = True
        except BaseException as exc:
            try:
                _state(
                    staging,
                    "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                    error=f"{type(exc).__name__}: {exc}",
                )
            except OSError:
                pass
            if isinstance(exc, sqlite3.Error):
                raise ValueError(f"session storage error: {exc}") from exc
            raise
        finally:
            if writer is not None:
                writer.close()
            if published:
                # Publication is the commit point. A cleanup failure cannot
                # turn an already published complete session into a failed one.
                try:
                    _state(staging, "published")
                    shutil.rmtree(staging)
                except OSError:
                    pass
        return report

    def _convert(self, sources, emit):
        raise NotImplementedError


class RawLogImporter(LegacyImporter):
    source_format = "csiraw01"

    def _convert(self, sources, emit):
        return _raw_import(sources, self.spec, emit)


class Hdf5LinksImporter(LegacyImporter):
    source_format = "legacy_hdf5"

    def _convert(self, sources, emit):
        from .legacy_links import import_links

        if self.spec.as_dict()["capture"]["controls"]:
            raise ValueError("legacy /links cannot apply raw epoch control evidence")
        return import_links(sources, emit=emit)


def import_session(source, destination, spec: AppConfig) -> MigrationReport:
    """Import explicitly mapped source path(s) into a new .csisession file.

    ``source=None`` selects all ``spec.capture.sources`` (the CLI behavior).
    Mixing raw and /links in one session is refused because their ordering and
    timing evidence are different. Use separate sessions for those sources.
    """
    if not isinstance(spec, AppConfig):
        raise TypeError("import spec must be a validated AppConfig")
    mappings = spec.as_dict()["capture"]["sources"]
    formats = {item.get("format", "csiraw01") for item in mappings}
    if len(formats) != 1:
        raise ValueError(
            "import requires sources of one explicit format; raw and /links cannot mix"
        )
    source_format = next(iter(formats))
    importer = {"csiraw01": RawLogImporter, "legacy_hdf5": Hdf5LinksImporter}.get(source_format)
    if importer is None:
        raise ValueError(f"unsupported import source format: {source_format}")
    if source is None:
        source = [item["path"] for item in mappings]
    elif isinstance(source, (str, Path)):
        source = [source]
    return importer(spec).import_atomic(tuple(source), destination)
