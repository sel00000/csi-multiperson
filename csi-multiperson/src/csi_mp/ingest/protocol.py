"""Streaming decoder for the 130-byte little-endian legacy CSI format.

CRC failures are candidate counts; discarded-byte issues describe bytes
actually removed to find the next candidate. They are separate measurements,
so their spans may overlap. Valid, semantic-rejected, discarded and trailing
bytes themselves form a disjoint partition of the input stream.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
import struct

import numpy as np

from csi_mp.data.schema import ParseIssue, RawChunk, SourceSpan, Stamp, WireFrame
from csi_mp.ingest.crc import crc16_ccitt

FRAME_MAGIC = 0xC51D
FRAME_LEN = 130
FRAME_BODY = struct.Struct("<HBBIIbbBB112s")
MAGIC_BYTES = struct.pack("<H", FRAME_MAGIC)


@dataclass
class _Piece:
    span: SourceSpan
    bridge_read: Stamp
    infer_available: Stamp | None
    ingest_ordinal: int


class FrameParser:
    """One trusted source/RX stream; never multiplex different sources.

    Unsupported RAW-dump frames are not interpreted as 130-byte CSI frames.
    Non-CSI bytes (including ASCII boot banners) are counted as junk. During
    CRC recovery discarded bytes have the distinct ``resync`` reason.
    """

    def __init__(self, *, source_id: str, expected_rx_id: int):
        if not isinstance(source_id, str) or not source_id:
            raise ValueError("source_id must be a nonempty trusted source identifier")
        if type(expected_rx_id) is not int or expected_rx_id not in range(3):
            raise ValueError("expected_rx_id must be one of 0, 1, 2")
        self.source_id = source_id
        self.expected_rx_id = expected_rx_id
        self._buffer = bytearray()
        self._pieces: deque[_Piece] = deque()
        self._resync = False
        self._finished = False
        self._source_sha256: str | None = None

    def _spans(self, count: int) -> tuple[SourceSpan, ...]:
        result = []
        for piece in self._pieces:
            take = min(count, piece.span.byte_count)
            result.append(replace(piece.span, byte_count=take))
            count -= take
            if not count:
                break
        if count:
            raise RuntimeError("parser provenance buffer is inconsistent")
        return tuple(result)

    def _consume(self, count: int) -> _Piece:
        del self._buffer[:count]
        last: _Piece | None = None
        while count:
            piece = self._pieces[0]
            last = piece
            take = min(count, piece.span.byte_count)
            if take == piece.span.byte_count:
                self._pieces.popleft()
            else:
                self._pieces[0] = replace(
                    piece,
                    span=replace(
                        piece.span,
                        file_offset=piece.span.file_offset + take,
                        byte_count=piece.span.byte_count - take,
                    ),
                )
            count -= take
        if last is None:
            raise RuntimeError("cannot consume an empty byte range")
        return last

    def _discard(self, count: int, issues: list[ParseIssue]) -> None:
        if count:
            issues.append(
                ParseIssue(
                    "discarded_bytes",
                    "wire_frame",
                    self._spans(count),
                    count,
                    "resync" if self._resync else "junk",
                )
            )
            self._consume(count)

    def feed(self, chunk: RawChunk) -> tuple[list[WireFrame], list[ParseIssue]]:
        if self._finished:
            raise ValueError("cannot feed a finished FrameParser; create a new parser")
        if chunk.span.source_id != self.source_id:
            raise ValueError("chunk source_id differs from the trusted parser source")
        if self._source_sha256 is None:
            self._source_sha256 = chunk.span.source_sha256
        elif chunk.span.source_sha256 != self._source_sha256:
            raise ValueError("source_sha256 changed; create a parser for the new source")
        if chunk.payload:
            self._buffer.extend(chunk.payload)
            self._pieces.append(
                _Piece(chunk.span, chunk.bridge_read, chunk.infer_available, chunk.ingest_ordinal)
            )
        frames: list[WireFrame] = []
        issues: list[ParseIssue] = []
        while self._buffer:
            index = self._buffer.find(MAGIC_BYTES)
            if index < 0:
                # The final byte may be the first half of split magic.
                keep = int(self._buffer[-1] == MAGIC_BYTES[0])
                self._discard(len(self._buffer) - keep, issues)
                break
            self._discard(index, issues)
            if len(self._buffer) < FRAME_LEN:
                break
            candidate = bytes(self._buffer[:FRAME_LEN])
            spans = self._spans(FRAME_LEN)
            expected_crc = struct.unpack_from("<H", candidate, FRAME_LEN - 2)[0]
            if crc16_ccitt(candidate[:-2]) != expected_crc:
                issues.append(ParseIssue("crc", "wire_frame", spans, FRAME_LEN, "crc_mismatch"))
                self._resync = True
                # Neither byte of the magic can start an overlapping magic.
                self._discard(len(MAGIC_BYTES), issues)
                continue
            _, rx, tx, seq, timer, rssi, noise, sc_count, boot, iq = FRAME_BODY.unpack(
                candidate[:-2]
            )
            reason = None
            if rx not in range(3):
                reason = "rx_id_out_of_range"
            elif tx not in range(3):
                reason = "tx_id_out_of_range"
            elif sc_count != 56:
                reason = "sc_count_mismatch"
            elif rx != self.expected_rx_id:
                reason = "source_rx_mismatch"
            complete = self._consume(FRAME_LEN)
            self._resync = False
            if reason is not None:
                issues.append(ParseIssue("semantic", "wire_frame", spans, FRAME_LEN, reason))
                continue
            frames.append(
                WireFrame(
                    spans=spans,
                    wire_bytes=candidate,
                    rx_id=rx,
                    tx_id=tx,
                    seq_u32=seq,
                    rx_timer_u32_us=timer,
                    rx_boot_u8=boot,
                    iq=np.frombuffer(iq, dtype=np.int8).reshape(56, 2),
                    rssi_dbm=rssi,
                    noise_dbm=noise,
                    signal_mode=None,
                    bridge_complete=complete.bridge_read,
                    infer_available=complete.infer_available,
                    ingest_ordinal=complete.ingest_ordinal,
                )
            )
        return frames, issues

    def finish(self) -> list[ParseIssue]:
        """Report only bytes held by this parser; repeated calls are harmless."""
        if self._finished:
            return []
        self._finished = True
        if not self._buffer:
            return []
        count = len(self._buffer)
        issue = ParseIssue("truncated", "wire_frame", self._spans(count), count, "incomplete_frame")
        self._consume(count)
        return [issue]
