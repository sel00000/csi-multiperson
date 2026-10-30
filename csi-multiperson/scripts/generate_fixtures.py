"""Rebuild the small, deterministic S00-S03 fixture without device access.

Run ``python scripts/generate_fixtures.py`` from the project, or pass
``--output DIR`` to generate an independent copy for verification. This uses
only Python's standard library and constructs bytes independently of csi_mp.
"""

from __future__ import annotations

import argparse
import binascii
import hashlib
import json
from pathlib import Path
import random
import struct
import tempfile

SEED = 0xC5103
BASE_NS = 1_700_000_000_000_000_000
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _wire(iq: bytes, *, rx: int, seq: int, timer: int, boot: int = 1, tx: int = 0) -> bytes:
    body = struct.pack("<HBBIIbbBB112s", 0xC51D, rx, tx, seq, timer, -61 - rx, -94, 56, boot, iq)
    return body + struct.pack("<H", binascii.crc_hqx(body, 0xFFFF))


def _record(offset_ns: int, payload: bytes) -> bytes:
    return struct.pack("<QH", BASE_NS + offset_ns, len(payload)) + payload


def _atomic_bytes(path: Path, content: bytes) -> None:
    # Replacing an existing symlink cannot follow it into reference inputs.
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(content)
    temporary.replace(path)


def generate(output: Path) -> dict:
    output = output.resolve()
    workspace = PROJECT_ROOT.parent
    for reference in (workspace / "examples", workspace / "ex)robot"):
        if output.is_relative_to(reference.resolve()):
            raise ValueError("fixture output must not be inside reference inputs")
    rng = random.Random(SEED)
    # Avoid magic bytes within the signal payload so a damaged candidate has
    # exactly one expected CRC failure, independent of parser implementation.
    levels = (-120, -96, -80, -64, -32, -16, -8, -1, 0, 1, 8, 16, 32, 64, 80, 96, 120)
    iq = bytes(rng.choice(levels) & 0xFF for _ in range(112))
    first = _wire(iq, rx=0, seq=100, timer=100_000)
    banner = b"CSI synthetic fixture\r\n"
    corrupt = bytearray(_wire(iq, rx=0, seq=900, timer=135_000))
    corrupt[20] ^= 1
    semantic = _wire(iq, rx=0, tx=3, seq=901, timer=136_000)
    wire_tail = _wire(iq, rx=0, seq=7, timer=21_000, boot=2)[:17]
    # This incomplete record is not yielded; its 15 actually received bytes
    # are accounted at the raw layer, never again as a wire tail.
    raw_tail = struct.pack("<QH", BASE_NS + 120_000_000, 200) + b"lost!"
    rx0_records = [
        (0, banner + first[:37], "banner_and_split_seq100_start"),
        (100_000, first[37:], "split_seq100_completion"),
        (2_000_000, first, "exact_duplicate_seq100"),
        (20_000_000, _wire(iq, rx=0, seq=102, timer=120_000), "normal_seq102"),
        (21_000_000, _wire(iq, rx=0, seq=101, timer=110_000), "reorder_seq101"),
        (35_000_000, bytes(corrupt) + semantic, "crc_error_and_semantic_tx3"),
        (40_000_000, _wire(iq, rx=0, seq=0, timer=140_000), "tx_restart_candidate0"),
        (50_000_000, _wire(iq, rx=0, seq=1, timer=150_000), "tx_restart_candidate1"),
        (60_000_000, _wire(iq, rx=0, seq=2, timer=160_000), "tx_restart_inferred2"),
        (70_000_000, _wire(iq, rx=0, seq=3, timer=170_000), "tx_restart_recovered3"),
        (90_000_000, _wire(iq, rx=0, seq=5, timer=1_000, boot=2), "rx0_boot_change"),
        (110_000_000, wire_tail, "incomplete_wire_tail"),
    ]
    rx1_records = [
        (1_000_000, _wire(iq, rx=1, seq=100, timer=100_000), "same_tx_seq_different_rx"),
        (30_000_000, _wire(iq, rx=1, seq=103, timer=130_000), "normal_seq103"),
        (80_000_000, _wire(iq, rx=1, seq=4, timer=180_000), "new_tx_epoch_other_rx"),
        (100_000_000, _wire(iq, rx=1, seq=6, timer=200_000), "rx1_clock_not_reset"),
    ]
    manifest = {
        "schema_id": "csi_mp.fixture.ingest_v1",
        "generator": "scripts/generate_fixtures.py",
        "generator_version": 1,
        "seed": SEED,
        "synthetic": True,
        "availability_basis": "synthetic",
        "raw_schema_id": "CSIRAW01",
        "wire_schema_id": "csi130.v1",
        "clock_domain": "fixture:bridge",
        "clock_kind": "unix_wall",
        "base_ns": str(BASE_NS),
        "sc_map_verified": False,
        "iq_order_verified": False,
        "provenance": "Constructed bytes; no measured RF, device clock, exposure, or GT.",
        "arrival_order": "merge complete records by timestamp, source_id, record_index",
        "expectation_method": "Lengths and candidate counts from construction, without decoding.",
        "sources": {},
    }
    source_cases = (
        ("rx0", 0, rx0_records, raw_tail, 9, 1, 1, len(banner), 130, len(wire_tail)),
        ("rx1", 1, rx1_records, b"", 4, 0, 0, 0, 0, 0),
    )
    output.mkdir(parents=True, exist_ok=True)
    for (
        source_id,
        rx,
        records,
        tail,
        valid,
        crc,
        semantic_count,
        junk,
        resync,
        wire_tail_count,
    ) in source_cases:
        data = b"CSIRAW01" + b"".join(_record(ns, payload) for ns, payload, _ in records) + tail
        filename = f"{source_id}.csiraw"
        counts = {
            "raw_records": len(records),
            "payload_bytes": sum(len(payload) for _, payload, _ in records),
            "valid_frames": valid,
            "crc_errors": crc,
            "semantic_rejects": semantic_count,
            "discarded_junk_bytes": junk,
            "discarded_resync_bytes": resync,
            "raw_tail_bytes": len(tail),
            "wire_tail_bytes": wire_tail_count,
        }
        # Partition is independent from any csi_mp parser/collector code.
        assert (
            counts["payload_bytes"]
            == valid * 130 + semantic_count * 130 + junk + resync + wire_tail_count
        )
        manifest["sources"][source_id] = {
            "path": filename,
            "source_id": source_id,
            "expected_rx_id": rx,
            "clock_domain": "fixture:bridge",
            "sha256": hashlib.sha256(data).hexdigest(),
            "size_bytes": len(data),
            "counts": counts,
            "records": [
                {
                    "record_index": index,
                    "timestamp_ns": str(BASE_NS + ns),
                    "payload_bytes": len(payload),
                    "scenario": scenario,
                }
                for index, (ns, payload, scenario) in enumerate(records)
            ],
        }
        _atomic_bytes(output / filename, data)
    count_keys = manifest["sources"]["rx0"]["counts"]
    manifest["totals"] = {
        key: sum(source["counts"][key] for source in manifest["sources"].values())
        for key in count_keys
    }
    _atomic_bytes(
        output / "expected.json", (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    )
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "fixtures" / "ingest_v1")
    args = parser.parse_args(argv)
    manifest = generate(args.output)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "totals": manifest["totals"],
                "sha256": {key: source["sha256"] for key, source in manifest["sources"].items()},
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
