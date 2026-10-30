"""Protocol fixtures derive wire bytes independently from the parser."""

import binascii
from dataclasses import replace
import hashlib
import struct

import numpy as np
import pytest

from csi_mp.data.schema import RawChunk, SourceSpan, Stamp
from csi_mp.ingest.crc import crc16_ccitt
from csi_mp.ingest.protocol import FrameParser
from csi_mp.ingest.rawlog import iter_raw_chunks


def wire(*, rx=0, tx=1, seq=123, timer=456, boot=7, sc=56, iq=None):
    pairs = bytes([128, 255, 0, 1, 127, 2, 3, 4]) * 14 if iq is None else iq
    body = struct.pack("<HBBIIbbBB112s", 0xC51D, rx, tx, seq, timer, -62, -94, sc, boot, pairs)
    return body + struct.pack("<H", binascii.crc_hqx(body, 0xFFFF))


def chunk(payload, *, index=0, offset=100, ns=1_000, infer_ns=None):
    return RawChunk(
        span=SourceSpan("ab" * 32, "rx0", index, offset, len(payload)),
        payload=payload,
        bridge_read=Stamp(ns, "bridge:test", "unix_wall"),
        infer_available=(None if infer_ns is None else Stamp(infer_ns, "infer:test", "monotonic")),
        ingest_ordinal=index,
    )


def parser():
    return FrameParser(source_id="rx0", expected_rx_id=0)


def test_crc_standard_check_vector_and_seed():
    assert crc16_ccitt(b"123456789") == 0x29B1
    assert crc16_ccitt(b"56789", crc16_ccitt(b"1234")) == 0x29B1
    with pytest.raises(ValueError, match="uint16"):
        crc16_ccitt(b"a", -1)


@pytest.mark.parametrize("cut", range(1, 130))
def test_every_frame_boundary_preserves_bytes_spans_and_completion(cut):
    encoded = wire()
    decoder = parser()
    frames, issues = decoder.feed(chunk(encoded[:cut]))
    assert frames == issues == []
    frames, issues = decoder.feed(
        chunk(encoded[cut:], index=1, offset=100 + cut, ns=2_000, infer_ns=9_000)
    )
    assert issues == [] and len(frames) == 1
    frame = frames[0]
    assert frame.wire_bytes == encoded
    assert [(s.file_offset, s.byte_count) for s in frame.spans] == [
        (100, cut),
        (100 + cut, 130 - cut),
    ]
    assert frame.bridge_complete.ns == 2_000
    assert frame.bridge_complete.kind == "unix_wall"
    assert frame.infer_available.ns == 9_000
    assert frame.infer_available.kind == "monotonic"
    assert frame.ingest_ordinal == 1
    assert frame.iq.dtype == np.int8 and frame.iq.shape == (56, 2)
    assert frame.iq[0].tolist() == [-128, -1]
    assert not frame.iq.flags.writeable
    assert (frame.seq_u32, frame.rx_timer_u32_us, frame.rx_boot_u8) == (123, 456, 7)
    assert frame.signal_mode is None
    assert decoder.finish() == []


def test_one_byte_chunks_and_multiple_frames_share_source_span_correctly():
    encoded = wire()
    decoder = parser()
    out = []
    for index, byte in enumerate(encoded):
        frames, issues = decoder.feed(
            chunk(bytes([byte]), index=index, offset=100 + index, ns=1_000 + index)
        )
        assert not issues
        out.extend(frames)
    assert len(out) == 1 and len(out[0].spans) == 130
    assert out[0].bridge_complete.ns == 1129
    assert sum(s.byte_count for s in out[0].spans) == 130
    decoder = parser()
    frames, issues = decoder.feed(chunk(encoded + wire(seq=124)))
    assert not issues and len(frames) == 2
    assert frames[0].spans[0].file_offset == 100
    assert frames[1].spans[0].file_offset == 230
    assert all(frame.spans[0].byte_count == 130 for frame in frames)


@pytest.mark.parametrize(
    ("field", "value", "detail"),
    [
        ("rx", 3, "rx_id_out_of_range"),
        ("tx", 3, "tx_id_out_of_range"),
        ("sc", 55, "sc_count_mismatch"),
        ("rx", 1, "source_rx_mismatch"),
    ],
)
def test_crc_valid_semantic_reject_is_one_complete_frame(field, value, detail):
    decoder = parser()
    bad = wire(**{field: value}, iq=b"\x1d\xc5" * 56)
    frames, issues = decoder.feed(chunk(bad + wire()))
    assert len(frames) == 1
    assert [(issue.kind, issue.byte_count, issue.detail) for issue in issues] == [
        ("semantic", 130, detail)
    ]
    assert frames[0].spans[0].file_offset == 230


@pytest.mark.parametrize("chunk_size", [1, 2, 17, 129, 130, 131, 10_000])
def test_banner_corrupt_candidate_semantic_and_tail_byte_accounting(chunk_size):
    banner = b"ESP boot\r\nready\n"
    corrupted = bytearray(wire(tx=255))
    corrupted[20] ^= 1  # CRC must fail before untrusted TX semantics are read.
    invalid = wire(sc=55)
    tail = wire()[:23]
    content = banner + bytes(corrupted) + invalid + wire(seq=124) + tail
    decoder = parser()
    all_frames, all_issues = [], []
    for index, offset in enumerate(range(0, len(content), chunk_size)):
        frames, issues = decoder.feed(
            chunk(
                content[offset : offset + chunk_size],
                index=index,
                offset=100 + offset,
                ns=1_000 + index,
            )
        )
        all_frames.extend(frames)
        all_issues.extend(issues)
    all_issues.extend(decoder.finish())
    assert len(all_frames) == 1 and all_frames[0].seq_u32 == 124
    assert sum(i.kind == "crc" for i in all_issues) == 1
    assert [i.detail for i in all_issues if i.kind == "semantic"] == ["sc_count_mismatch"]
    discarded = [i for i in all_issues if i.kind == "discarded_bytes"]
    assert sum(i.byte_count for i in discarded if i.detail == "junk") == len(banner)
    assert sum(i.byte_count for i in discarded if i.detail == "resync") == 130
    assert [(i.layer, i.byte_count) for i in all_issues if i.kind == "truncated"] == [
        ("wire_frame", len(tail))
    ]
    # CRC candidate bytes are a diagnostic overlap, not additional lost bytes.
    accounted = sum(i.byte_count for i in all_issues if i.kind != "crc") + 130
    assert accounted == len(content)
    assert decoder.finish() == []
    with pytest.raises(ValueError, match="finished"):
        decoder.feed(chunk(b""))


def test_crc_recovery_preserves_valid_frame_inside_failed_candidate_range():
    decoder = parser()
    # Losing bytes in one frame means the first CRC candidate overlaps the
    # next valid frame; discarding the whole candidate would lose that frame.
    frames, issues = decoder.feed(chunk(wire()[:20] + wire(seq=124)))
    assert len(frames) == 1 and frames[0].seq_u32 == 124
    assert sum(i.kind == "crc" for i in issues) == 1
    assert sum(i.byte_count for i in issues if i.kind == "discarded_bytes") == 20
    assert decoder.finish() == []


def test_partial_magic_and_nonmagic_final_byte_are_distinct():
    decoder = parser()
    assert decoder.feed(chunk(b"\x1d")) == ([], [])
    assert decoder.finish()[0].byte_count == 1
    decoder = parser()
    frames, issues = decoder.feed(chunk(b"x"))
    assert not frames
    assert issues[0].kind == "discarded_bytes" and issues[0].byte_count == 1
    assert decoder.finish() == []


def test_trusted_source_must_not_change():
    with pytest.raises(ValueError, match="expected_rx_id"):
        FrameParser(source_id="rx0", expected_rx_id=3)
    decoder = parser()
    foreign = replace(chunk(wire()), span=replace(chunk(wire()).span, source_id="rx1"))
    with pytest.raises(ValueError, match="source_id"):
        decoder.feed(foreign)
    decoder.feed(chunk(wire()[:1]))
    changed = replace(
        chunk(wire()[1:]), span=replace(chunk(wire()[1:]).span, source_sha256="cd" * 32)
    )
    with pytest.raises(ValueError, match="source_sha256"):
        decoder.feed(changed)


def raw_record(ns, payload):
    return struct.pack("<QH", ns, len(payload)) + payload


def test_raw_records_provenance_zero_length_and_clock_regression(tmp_path):
    content = b"CSIRAW01" + raw_record(2_000, wire()[:1])
    content += raw_record(1_000, wire()[1:]) + raw_record(1_000, b"")
    path = tmp_path / "capture.raw"
    path.write_bytes(content)
    issues = []
    chunks = list(
        iter_raw_chunks(path, source_id="rx0", clock_domain="legacy:test", on_issue=issues.append)
    )
    assert [c.bridge_read.ns for c in chunks] == [2_000, 1_000, 1_000]
    assert [c.ingest_ordinal for c in chunks] == [0, 1, 2]
    assert [c.span.file_offset for c in chunks] == [18, 29, 168]
    assert [c.span.byte_count for c in chunks] == [1, 129, 0]
    assert all(c.bridge_read.kind == "unix_wall" and c.infer_available is None for c in chunks)
    assert all(c.span.source_sha256 == hashlib.sha256(content).hexdigest() for c in chunks)
    assert [(i.kind, i.detail, i.byte_count) for i in issues] == [
        ("time_discontinuity", "clock_regression", 0)
    ]
    decoder = parser()
    frames = []
    for item in chunks:
        found, parse_issues = decoder.feed(item)
        assert not parse_issues
        frames.extend(found)
    assert len(frames) == 1
    assert [(s.file_offset, s.byte_count) for s in frames[0].spans] == [(18, 1), (29, 129)]
    assert frames[0].bridge_complete.ns == 1_000
    assert path.read_bytes() == content


@pytest.mark.parametrize(
    ("tail", "detail"),
    [
        (b"\x01\x02\x03", "incomplete_record_header"),
        (struct.pack("<QH", 2_000, 500) + b"\x1d\xc5", "incomplete_record_payload"),
    ],
)
def test_raw_tail_counts_received_bytes_once_and_preserves_good_frame(tmp_path, tail, detail):
    complete = b"CSIRAW01" + raw_record(1_000, wire())
    path = tmp_path / "partial.raw"
    path.write_bytes(complete + tail)
    issues = []
    chunks = list(
        iter_raw_chunks(path, source_id="rx0", clock_domain="legacy:test", on_issue=issues.append)
    )
    assert len(chunks) == 1
    assert len(issues) == 1
    issue = issues[0]
    assert (issue.kind, issue.layer, issue.byte_count, issue.detail) == (
        "truncated",
        "raw_record",
        len(tail),
        detail,
    )
    assert issue.spans[0].file_offset == len(complete)
    assert issue.spans[0].chunk_index == 1
    decoder = parser()
    frames, wire_issues = decoder.feed(chunks[0])
    assert len(frames) == 1 and not wire_issues
    assert decoder.finish() == []  # Raw-reader tail was never given to parser.


@pytest.mark.parametrize("header", [b"", b"CSI", b"CSIRAW02", b"notrawxx"])
def test_invalid_raw_header_reports_actual_bytes_before_failure(tmp_path, header):
    path = tmp_path / "bad.raw"
    path.write_bytes(header)
    issues = []
    with pytest.raises(ValueError, match="CSIRAW01"):
        list(
            iter_raw_chunks(
                path, source_id="rx0", clock_domain="legacy:test", on_issue=issues.append
            )
        )
    assert len(issues) == 1
    assert issues[0].kind == "invalid_header"
    assert issues[0].byte_count == len(header)


def test_raw_source_mutation_is_detected(tmp_path):
    path = tmp_path / "changing.raw"
    path.write_bytes(b"CSIRAW01" + raw_record(1_000, wire()))
    reader = iter_raw_chunks(
        path, source_id="rx0", clock_domain="legacy:test", on_issue=lambda _: None
    )
    next(reader)
    with path.open("ab") as stream:
        stream.write(raw_record(2_000, wire(seq=124)))
    with pytest.raises(ValueError, match="changed"):
        list(reader)
