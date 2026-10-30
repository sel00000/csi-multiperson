"""Read-only, loss-aware import of legacy HDF5 ``/links`` observations.

The legacy writer kept row order within each link, but discarded raw bytes,
CRC, raw-file offsets, and global arrival order. Its ``esp_us`` is already
unwrapped by a policy that treated any timer regression as a wrap. This
adapter therefore archives values without inventing wire frames, verified
epochs, acquisition estimates, or a replay availability stream.
"""

from __future__ import annotations

import math
from pathlib import Path
import re

import numpy as np

from csi_mp.data.schema import Stamp


_FIELDS = {
    "iq": (np.dtype("int8"), (56, 2)),
    "t_ns": (np.dtype("uint64"), ()),
    "esp_us": (np.dtype("uint64"), ()),
    "seq": (np.dtype("uint32"), ()),
    "rssi": (np.dtype("int8"), ()),
    "noise": (np.dtype("int8"), ()),
    "boot_id": (np.dtype("uint8"), ()),
}
_OPTIONAL_FIELDS = tuple(name for name in _FIELDS if name != "iq")
_UNKNOWN_REASONS = {
    "wire_hex": "wire_bytes_and_crc_not_preserved_in_legacy_links",
    "infer_available": "inference_arrival_not_recorded",
    "rx_epoch": "verified_rx_epoch_not_recorded",
    "tx_epoch": "verified_tx_epoch_not_recorded",
    "acquisition_ns": "verified_epoch_and_causal_clock_model_unavailable",
    "availability_ns": "global_arrival_order_and_inference_arrival_not_recorded",
    "signal_mode": "not_recorded_in_legacy_links",
}
_ASSUMPTIONS = (
    "legacy_links rows are archived in source_id, link name, row index order; "
    "this is not a replay arrival order",
    "legacy t_ns is a bridge clock stamp, not measured inference availability; "
    "a clock domain is used only when explicitly supplied",
    "legacy esp_us is preserved as stored; its old unwrap policy may count "
    "timer reordering as wrapping and is not verified epoch evidence",
    "legacy links contain no original wire bytes, CRC, raw byte offsets, or "
    "global ingest ordinal; dataset path and row index provide source lineage",
    "RX and TX epochs, acquisition time, and replay availability remain unknown; "
    "retained rows are not usable by the causal input engine",
    "IQ pairs retain stored order; real/imaginary meaning and subcarrier mapping "
    "are not established by this import",
    "derived samples and grids are not used to reconstruct raw CSI or a T100 window",
)


def _metadata_value(value, *, location):
    """Keep HDF5 metadata JSON-safe without silently losing bytes or array dtype."""
    if isinstance(value, str):
        return value
    if isinstance(value, (bytes, np.bytes_)):
        return {"type": "bytes", "hex": bytes(value).hex()}
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"{location}: nonfinite HDF5 metadata is unsupported")
        return number
    if isinstance(value, np.ndarray):
        return {
            "type": "ndarray",
            "dtype": value.dtype.str,
            "shape": list(value.shape),
            "values": _metadata_value(value.tolist(), location=location),
        }
    if isinstance(value, np.void):
        return {"type": "void", "dtype": value.dtype.str, "hex": value.tobytes().hex()}
    if isinstance(value, (tuple, list)):
        return [_metadata_value(item, location=location) for item in value]
    raise ValueError(f"{location}: unsupported HDF5 metadata type {type(value).__name__}")


def _attributes(obj):
    return {
        name: _metadata_value(obj.attrs[name], location=f"{obj.name}.attrs[{name!r}]")
        for name in sorted(obj.attrs)
    }


def _check_self_contained(h5, h5py):
    """Reject references to data outside the hashed file, including ignored groups."""
    visited = set()

    def visit(obj):
        address = h5py.h5o.get_info(obj.id).addr
        if address in visited:
            return
        visited.add(address)
        if isinstance(obj, h5py.Dataset):
            if obj.is_virtual or obj.external:
                raise ValueError(f"{obj.name}: virtual/external datasets are unsupported")
            return
        if not isinstance(obj, h5py.Group):
            raise ValueError(f"{obj.name}: unsupported HDF5 object")
        for name in sorted(obj):
            link = obj.get(name, getlink=True)
            if not isinstance(link, h5py.HardLink):
                raise ValueError(f"{obj.name}/{name}: soft/external HDF5 links are unsupported")
            visit(obj[name])

    visit(h5)


def _validate_links(h5, h5py):
    if "links" not in h5:
        if "samples" in h5:
            raise ValueError("legacy HDF5: /samples alone cannot reconstruct raw CSI or T100")
        raise ValueError("legacy HDF5: missing /links group")
    links = h5["links"]
    if not isinstance(links, h5py.Group) or not len(links):
        raise ValueError("legacy HDF5: /links must be a nonempty group")
    counts = {}
    for name in sorted(links):
        if re.fullmatch(r"[0-2][0-2]", name) is None:
            raise ValueError(f"/links/{name}: expected a two-digit RX/TX link name in 00..22")
        group = links[name]
        if not isinstance(group, h5py.Group):
            raise ValueError(f"{group.name}: expected a link group")
        extras = sorted(set(group) - _FIELDS.keys())
        if extras:
            raise ValueError(f"{group.name}: unsupported legacy link fields {extras}")
        if "iq" not in group or not isinstance(group["iq"], h5py.Dataset):
            raise ValueError(f"{group.name}: required iq dataset is missing")
        iq = group["iq"]
        if iq.ndim != 3 or iq.shape[1:] != (56, 2):
            raise ValueError(f"{iq.name}: expected shape [N,56,2], got {iq.shape}")
        count = iq.shape[0]
        for field in sorted(group):
            dataset = group[field]
            dtype, tail = _FIELDS[field]
            if not isinstance(dataset, h5py.Dataset):
                raise ValueError(f"{group.name}/{field}: expected a dataset")
            actual = dataset.dtype
            if actual.kind != dtype.kind or actual.itemsize != dtype.itemsize or actual.metadata:
                raise ValueError(f"{dataset.name}: expected dtype {dtype}, got {actual}")
            if dataset.shape != (count, *tail):
                raise ValueError(
                    f"{dataset.name}: expected shape {(count, *tail)}, got {dataset.shape}; "
                    "legacy capture may be incomplete"
                )
        counts[name] = count
    if not sum(counts.values()):
        raise ValueError("legacy HDF5: /links contains no observations")
    return counts


def _observation(source, name, row_index, fields, ordinal, missing):
    domain = source.get("clock_domain")
    clock_kind = source.get("clock_kind", "unix_wall")
    t_ns = fields["t_ns"]
    bridge = Stamp(t_ns, domain, clock_kind).to_dict() if t_ns is not None and domain else None
    unknown_reasons = {**_UNKNOWN_REASONS, **missing}
    if not domain:
        unknown_reasons["clock_domain"] = "clock_domain_not_supplied"
    if bridge is None:
        unknown_reasons["bridge_complete"] = "t_ns_or_clock_domain_unavailable"
    return {
        "schema_id": "csi_mp.session_observation.v1",
        "adapter": "legacy_links",
        "source_ref": {
            "source_id": source["source_id"],
            "source_sha256": source["sha256"],
            "dataset_path": f"/links/{name}",
            "row_index": row_index,
        },
        "ingest_ordinal": ordinal,
        "rx_id": int(name[0]),
        "tx_id": int(name[1]),
        "iq": fields["iq"],
        **{
            field: str(value) if field in {"t_ns", "esp_us"} and value is not None else value
            for field, value in fields.items()
            if field != "iq"
        },
        "signal_mode": None,
        "wire_hex": None,
        "spans": [],
        "bridge_complete": bridge,
        "infer_available": None,
        "unknown_fields": sorted(unknown_reasons),
        "unknown_reasons": dict(sorted(unknown_reasons.items())),
        "epochs": {
            "rx_epoch": None,
            "tx_epoch": None,
            "link_segment": None,
            "timer_unwrapped_us": None,
            "classification": "unknown",
            "usable": False,
            "evidence_refs": [],
            "events": [],
        },
        "timing": {
            "acquisition_ns": None,
            "availability_ns": None,
            "availability_basis": "unknown",
            "clock_domain": domain,
            "clock_kind": clock_kind if domain else None,
            "clock_reference": None,
            "clock_model_id": None,
            "clock_segment_id": None,
            "clock_error_ns": None,
            "clock_status": "unknown",
            "usable": False,
            "exclusion_reason": "legacy_epoch_and_availability_unknown",
        },
    }


def import_links(sources: list[dict], *, emit) -> dict:
    """Emit retained legacy rows and deterministic lineage from read-only snapshots.

    ``sources`` identify already-hashed snapshots; this adapter never hashes a
    different path or resolves original source metadata as a data dependency.
    Unknown optional fields are retained as ``None``. Malformed datasets fail
    the migration rather than being cast, truncated, or zero-filled.
    """
    try:
        import h5py
    except ImportError as exc:
        raise ValueError("legacy_links requires the optional dataset dependency: h5py") from exc

    if not sources:
        raise ValueError("legacy_links requires at least one source")
    identifiers = [source.get("source_id") for source in sources]
    if any(not isinstance(value, str) or not value for value in identifiers):
        raise ValueError("legacy_links source_id must be a nonempty string")
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("legacy_links source_id values must be unique")
    ordinal = 0
    metadata = {}
    for source in sorted(sources, key=lambda item: item["source_id"]):
        domain = source.get("clock_domain")
        if domain is not None and (not isinstance(domain, str) or not domain):
            raise ValueError("legacy_links clock_domain must be a nonempty string or null")
        kind = source.get("clock_kind", "unix_wall")
        if kind != "unix_wall":
            raise ValueError("legacy_links t_ns requires clock_kind='unix_wall'")
        if not isinstance(source.get("sha256"), str) or not re.fullmatch(
            r"[0-9a-f]{64}", source["sha256"]
        ):
            raise ValueError("legacy_links source sha256 must be a lowercase SHA256 digest")
        with h5py.File(source["path"], "r") as h5:
            _check_self_contained(h5, h5py)
            counts = _validate_links(h5, h5py)
            if "meta" in h5 and not isinstance(h5["meta"], h5py.Group):
                raise ValueError("legacy HDF5: /meta must be a group")
            metadata[source["source_id"]] = {
                "original_path": str(Path(source.get("original_path", source["path"]))),
                "source_sha256": source["sha256"],
                "root_attributes": _attributes(h5),
                "meta_attributes": _attributes(h5["meta"]) if "meta" in h5 else None,
                "links_attributes": _attributes(h5["links"]),
                "link_metadata": {
                    name: {
                        "row_count": count,
                        "attributes": _attributes(h5[f"links/{name}"]),
                        "dataset_attributes": {
                            field: _attributes(h5[f"links/{name}/{field}"])
                            for field in sorted(h5[f"links/{name}"])
                        },
                    }
                    for name, count in counts.items()
                },
            }
            for name, count in counts.items():
                group = h5[f"links/{name}"]
                missing = {
                    field: "field_absent_in_legacy_links"
                    for field in _OPTIONAL_FIELDS
                    if field not in group
                }
                if missing:
                    emit(
                        "issues",
                        {
                            "kind": "legacy_unknown_fields",
                            "source_id": source["source_id"],
                            "source_sha256": source["sha256"],
                            "dataset_path": f"/links/{name}",
                            "fields": sorted(missing),
                            "detail": "missing fields are null; no values have been inferred",
                        },
                    )
                for start in range(0, count, 4096):
                    block = {field: group[field][start : start + 4096] for field in group}
                    for local in range(min(4096, count - start)):
                        fields = {
                            field: int(block[field][local]) if field in block else None
                            for field in _OPTIONAL_FIELDS
                        }
                        fields["iq"] = block["iq"][local].tolist()
                        emit(
                            "observations",
                            _observation(source, name, start + local, fields, ordinal, missing),
                        )
                        ordinal += 1
    return {
        "counts": {"stored_frames": ordinal, "accepted_frames": ordinal, "usable_frames": 0},
        "rejected_by_reason": {},
        "unresolved_epochs": ordinal,
        "legacy_assumptions": list(_ASSUMPTIONS),
        "source_metadata": metadata,
        "observation_order": "source_id_link_name_row_index_not_replay_arrival",
    }
