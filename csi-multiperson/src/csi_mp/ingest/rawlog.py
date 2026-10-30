"""Read legacy CSIRAW01 records without altering their order or source file.

The timestamp in this format was recorded with ``time.time_ns()``. It is a
Unix wall clock, not an inference host monotonic arrival timestamp. Source
identity and clock domain must therefore be supplied by the caller.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import struct
from typing import Callable, Iterator

from csi_mp.data.schema import ParseIssue, RawChunk, SourceSpan, Stamp

HEADER = b"CSIRAW01"
RECORD_HEADER = struct.Struct("<QH")


def _file_identity(stat: os.stat_result) -> tuple[int, int, int]:
    return stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def iter_raw_chunks(
    path: str | Path,
    *,
    source_id: str,
    clock_domain: str,
    on_issue: Callable[[ParseIssue], None],
) -> Iterator[RawChunk]:
    """Yield complete records with exact payload offsets and a source SHA256.

    A truncated record is never yielded. Its issue counts only bytes present
    on disk, including any record header, not its advertised missing payload.
    The input is hashed through the same read-only file descriptor before
    parsing; mutation detected during a full read raises ``ValueError``.
    """
    if not isinstance(source_id, str) or not source_id:
        raise ValueError("source_id must be a nonempty trusted source identifier")
    if not isinstance(clock_domain, str) or not clock_domain:
        raise ValueError("clock_domain must identify the legacy wall clock")
    with Path(path).open("rb") as stream:
        initial_stat = _file_identity(os.fstat(stream.fileno()))
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if _file_identity(os.fstat(stream.fileno())) != initial_stat:
            raise ValueError("raw source changed while computing source_sha256")
        stream.seek(0)

        def span(index: int, offset: int, count: int) -> SourceSpan:
            return SourceSpan(digest, source_id, index, offset, count)

        header = stream.read(len(HEADER))
        if header != HEADER:
            on_issue(
                ParseIssue(
                    "invalid_header",
                    "raw_record",
                    (span(0, 0, len(header)),),
                    len(header),
                    "invalid_raw_header",
                )
            )
            raise ValueError(f"invalid CSIRAW01 header: {path}")

        record_index = 0
        previous_ns: int | None = None
        while True:
            offset = stream.tell()
            raw_header = stream.read(RECORD_HEADER.size)
            if not raw_header:
                break
            if len(raw_header) != RECORD_HEADER.size:
                on_issue(
                    ParseIssue(
                        "truncated",
                        "raw_record",
                        (span(record_index, offset, len(raw_header)),),
                        len(raw_header),
                        "incomplete_record_header",
                    )
                )
                break
            stamp_ns, payload_size = RECORD_HEADER.unpack(raw_header)
            payload = stream.read(payload_size)
            if len(payload) != payload_size:
                actual = RECORD_HEADER.size + len(payload)
                on_issue(
                    ParseIssue(
                        "truncated",
                        "raw_record",
                        (span(record_index, offset, actual),),
                        actual,
                        "incomplete_record_payload",
                    )
                )
                break
            payload_span = span(record_index, offset + RECORD_HEADER.size, len(payload))
            if previous_ns is not None and stamp_ns < previous_ns:
                on_issue(
                    ParseIssue(
                        "time_discontinuity", "raw_record", (payload_span,), 0, "clock_regression"
                    )
                )
            yield RawChunk(
                span=payload_span,
                payload=payload,
                bridge_read=Stamp(stamp_ns, clock_domain, "unix_wall"),
                infer_available=None,
                ingest_ordinal=record_index,
            )
            previous_ns = stamp_ns
            record_index += 1
        if _file_identity(os.fstat(stream.fileno())) != initial_stat:
            raise ValueError("raw source changed while reading records")
