"""Prefix fixtures exercise independent receiver clocks and TX reset evidence."""

from dataclasses import replace
import struct

import numpy as np
import pytest

from csi_mp.data.schema import EpochEvidence, SourceSpan, Stamp, WireFrame
from csi_mp.ingest.epochs import EpochPolicy, RxTimerUnwrapper, TxSequenceEpochs, U32


def frame(
    seq,
    timer,
    *,
    rx=0,
    tx=0,
    boot=1,
    ordinal=0,
    ns=None,
    payload=0,
    domain="fixture",
    kind="virtual",
):
    wire = struct.pack("<BBIIBb", rx, tx, seq, timer, boot, payload)
    stamp = Stamp(ordinal * 10_000_000 if ns is None else ns, domain, kind)
    span = SourceSpan("fixture-sha", f"rx{rx}", ordinal, ordinal * 130, len(wire))
    return WireFrame(
        (span,),
        wire,
        rx,
        tx,
        seq,
        timer,
        boot,
        np.full((56, 2), payload, dtype=np.int8),
        -40,
        -90,
        None,
        stamp,
        None,
        ordinal,
    )


class Machine:
    def __init__(self, policy=None):
        self.rx = RxTimerUnwrapper(policy)
        self.tx = TxSequenceEpochs(policy)

    def __call__(self, value):
        return self.tx.classify(value, self.rx.update(value))


def test_same_tx_sequence_at_three_receivers_is_three_unique_observations():
    engine = Machine()
    results = [engine(frame(100, 100_000, rx=rx, ordinal=rx)) for rx in range(3)]
    assert [r.classification for r in results] == ["normal"] * 3
    assert all(r.usable for r in results)
    assert len({r.tx_epoch for r in results}) == 1
    assert len({r.link_segment for r in results}) == 3


def test_duplicate_requires_wire_consistency_and_conflict_is_quarantined():
    engine = Machine()
    value = frame(7, 70_000)
    first = engine(value)
    duplicate = engine(replace(value, bridge_complete=Stamp(1, "fixture", "virtual")))
    conflict = engine(frame(7, 70_000, payload=1, ordinal=1))
    assert first.usable
    assert duplicate.classification == "duplicate" and not duplicate.usable
    assert conflict.classification == "ambiguous" and not conflict.usable
    assert conflict.tx_epoch is None
    assert "reason:sequence_payload_conflict" in conflict.evidence_refs


def test_low_single_packet_never_confirms_reset_and_segment_is_stable():
    engine = Machine()
    original = engine(frame(100, 100_000))
    first = engine(frame(0, 110_000, ordinal=1))
    second = engine(frame(0, 120_000, ordinal=2))
    assert not first.usable and first.classification == "ambiguous"
    assert not second.usable and second.link_segment == first.link_segment
    assert not first.events and not second.events
    resumed = engine(frame(101, 130_000, ordinal=3))
    assert resumed.usable and resumed.tx_epoch == original.tx_epoch
    assert resumed.link_segment != first.link_segment


def test_tx_only_restart_recovers_from_progressing_prefix_without_rewriting_past():
    engine = Machine()
    original = engine(frame(100, 100_000))
    first = engine(frame(0, 110_000, ordinal=1))
    second = engine(frame(1, 120_000, ordinal=2))
    recovered = engine(frame(2, 130_000, ordinal=3))
    normal = engine(frame(3, 140_000, ordinal=4))
    assert not first.usable and not second.usable
    assert recovered.usable and recovered.classification == "tx_reset"
    assert recovered.tx_epoch != original.tx_epoch
    assert recovered.events[-1].certainty == "inferred"
    assert recovered.events[-1].affected_links == (0, 3, 6)
    assert normal.usable and normal.tx_epoch == recovered.tx_epoch
    assert first.tx_epoch is None and first.classification == "ambiguous"


def test_same_low_sequence_from_multiple_receivers_does_not_inflate_restart_evidence():
    engine = Machine()
    for rx in range(3):
        engine(frame(100, 100_000, rx=rx, ordinal=rx))
    results = [engine(frame(0, 110_000, rx=rx, ordinal=3 + rx)) for rx in range(3)]
    assert all(not result.usable for result in results)
    assert all(not result.events for result in results)
    first_progress = engine(frame(1, 120_000, rx=0, ordinal=6))
    recovered = engine(frame(2, 130_000, rx=1, ordinal=7))
    assert not first_progress.usable and recovered.usable
    assert recovered.events[-1].certainty == "inferred"


@pytest.mark.parametrize("old_head", [0, 1, 2])
def test_restart_evidence_continues_across_low_pre_restart_high_sequence(old_head):
    engine = Machine()
    original = engine(frame(old_head, 100_000))
    results = [engine(frame(seq, 110_000 + seq * 10_000, ordinal=seq + 1)) for seq in range(4)]
    assert [result.usable for result in results] == [False, False, True, True]
    assert results[2].classification == "tx_reset"
    assert results[2].events[-1].certainty == "inferred"
    assert results[2].tx_epoch != original.tx_epoch
    assert results[3].tx_epoch == results[2].tx_epoch


def test_old_delayed_low_burst_with_older_timer_cannot_infer_restart():
    engine = Machine()
    engine(frame(100, 100_000))
    results = [engine(frame(seq, 10_000 + seq * 10_000, ordinal=seq + 1)) for seq in range(3)]
    assert all(not result.usable for result in results)
    assert all(all(event.kind != "tx_reset" for event in result.events) for result in results)


def test_small_timer_and_sequence_reorder_does_not_move_forward_cursor():
    engine = Machine()
    engine(frame(100, 100_000))
    engine(frame(102, 120_000, ordinal=1))
    reordered = engine(frame(101, 110_000, ordinal=2))
    next_value = engine(frame(103, 130_000, ordinal=3))
    assert reordered.usable and reordered.classification == "reorder"
    assert reordered.timer_unwrapped_us == 110_000
    assert [event.kind for event in reordered.events] == ["timer_regression"]
    assert next_value.classification == "normal" and next_value.timer_unwrapped_us == 130_000


def test_timer_and_sequence_wrap_events_survive_together_and_emit_once_per_device():
    engine = Machine()
    before = frame(U32 - 2, U32 - 2, ns=0)
    engine(before)
    wrapped = engine(frame(1, 1, ordinal=1, ns=3_000))
    second_rx = engine(frame(1, 100, rx=1, ordinal=2, ns=4_000))
    assert wrapped.usable and wrapped.timer_unwrapped_us == U32 + 1
    assert wrapped.classification == "seq_wrap"
    assert [event.kind for event in wrapped.events] == ["timer_wrap", "seq_wrap"]
    assert len({event.event_id for event in wrapped.events}) == 2
    assert second_rx.usable and not second_rx.events
    assert second_rx.classification == "normal"


def test_pre_wrap_late_packet_does_not_create_second_wrap_or_poison_cursors():
    engine = Machine()
    engine(frame(U32 - 3, U32 - 30, ns=0))
    engine(frame(1, 10, ordinal=1, ns=40_000))
    late = engine(frame(U32 - 2, U32 - 20, ordinal=2, ns=50_000))
    after = engine(frame(2, 20, ordinal=3, ns=60_000))
    assert late.usable and late.classification == "reorder"
    assert late.timer_unwrapped_us == U32 - 20
    assert [event.kind for event in late.events] == ["timer_regression"]
    assert after.usable and after.timer_unwrapped_us == U32 + 20
    assert not after.events


@pytest.mark.parametrize(
    "kind,domain,ns",
    [
        ("virtual", "other", 1_000),
        ("unix_wall", "fixture", 1_000),
        ("virtual", "fixture", 0),
        ("virtual", "fixture", 20_000_000_000),
    ],
)
def test_timer_wrap_needs_comparable_nonregressing_reasonable_elapsed(kind, domain, ns):
    receiver = RxTimerUnwrapper()
    receiver.update(frame(0, U32 - 1, ns=1))
    result = receiver.update(frame(1, 0, ordinal=1, ns=ns, kind=kind, domain=domain))
    assert not result.usable and result.classification == "ambiguous"
    assert not any(event.kind == "timer_wrap" for event in result.events)


def test_large_timer_regression_is_not_wrap_and_does_not_change_other_receiver():
    receiver = RxTimerUnwrapper()
    receiver.update(frame(100, 500_000))
    other_before = receiver.update(frame(100, 200_000, rx=1))
    regression = receiver.update(frame(0, 1000, ordinal=1))
    other_after = receiver.update(frame(101, 210_000, rx=1, ordinal=2))
    assert regression.classification == "ambiguous" and regression.timer_unwrapped_us is None
    assert [event.kind for event in regression.events] == ["timer_regression"]
    assert other_after.rx_epoch == other_before.rx_epoch and other_after.usable


def test_rx_boot_change_resets_only_its_receiver_and_never_reuses_epoch_id():
    engine = Machine()
    first = engine(frame(100, 100_000, boot=255))
    other = engine(frame(100, 100_000, rx=1))
    reboot = engine(frame(101, 1, boot=0, ordinal=1))
    other_after = engine(frame(101, 110_000, rx=1, ordinal=2))
    repeated_boot = engine(frame(102, 1, boot=255, ordinal=3))
    assert reboot.classification == "rx_reset" and reboot.usable
    assert reboot.tx_epoch == first.tx_epoch
    assert reboot.events[-1].affected_links == (0, 1, 2)
    assert other_after.rx_epoch == other.rx_epoch
    assert other_after.timer_unwrapped_us == 110_000
    assert len({first.rx_epoch, reboot.rx_epoch, repeated_boot.rx_epoch}) == 3


def test_explicit_rx_reset_emits_event_at_next_frame_and_is_local():
    receiver = RxTimerUnwrapper()
    original = receiver.update(frame(1, 10))
    other = receiver.update(frame(1, 20, rx=1))
    receiver.reset(0, "control:rx0:boot")
    reset = receiver.update(frame(2, 1, ordinal=1))
    assert reset.rx_epoch != original.rx_epoch and reset.classification == "rx_reset"
    assert reset.events[0].certainty == "confirmed"
    assert "control:rx0:boot" in reset.events[0].evidence_refs
    assert receiver.update(frame(2, 21, rx=1, ordinal=2)).rx_epoch == other.rx_epoch


def test_known_delayed_old_boot_frame_cannot_reboot_receiver_back_to_old_epoch():
    engine = Machine()
    original = frame(100, 100_000, boot=1)
    engine(original)
    reboot = engine(frame(101, 1, boot=2, ordinal=1))
    delayed = engine(replace(original, bridge_complete=Stamp(20_000_000, "fixture", "virtual")))
    resumed = engine(frame(102, 10_000, boot=2, ordinal=3))
    assert delayed.classification == "ambiguous" and not delayed.usable
    assert delayed.rx_epoch == reboot.rx_epoch == resumed.rx_epoch
    assert "reason:prior_rx_epoch_replay" in delayed.evidence_refs
    assert not any(event.kind == "rx_reset" for event in delayed.events + resumed.events)
    assert resumed.usable


def test_control_start_is_confirmed_scoped_and_not_applied_to_earlier_prefix():
    engine = Machine()
    control = EpochEvidence(
        "tx_start", 0, Stamp(30_000_000, "fixture", "virtual"), "control:start:0"
    )
    engine.tx.observe_control(control)
    originals = [engine(frame(100, 100_000, rx=rx, ordinal=rx)) for rx in range(3)]
    other_tx = engine(frame(50, 125_000, tx=1, ordinal=2, ns=25_000_000))
    reset = engine(frame(0, 130_000, ordinal=3))
    second_rx = engine(frame(0, 130_000, rx=1, ordinal=4))
    other_after = engine(frame(51, 150_000, tx=1, ordinal=5))
    assert all(result.tx_epoch == originals[0].tx_epoch for result in originals)
    assert reset.usable and reset.classification == "tx_reset"
    assert reset.events[-1].certainty == "confirmed"
    assert reset.events[-1].evidence_refs == ("control:start:0",)
    assert second_rx.usable and second_rx.tx_epoch == reset.tx_epoch
    assert not second_rx.events
    assert other_after.tx_epoch == other_tx.tx_epoch and other_after.usable


def test_previous_epoch_exact_bytes_are_excluded_after_control_reset():
    engine = Machine()
    original = frame(100, 100_000)
    engine(original)
    engine.tx.observe_control(
        EpochEvidence("tx_start", 0, Stamp(1_000_000, "fixture", "virtual"), "start")
    )
    engine(frame(0, 110_000, ordinal=1))
    delayed = engine(
        replace(original, bridge_complete=Stamp(20_000_000, "fixture", "virtual"), ingest_ordinal=2)
    )
    assert delayed.classification == "ambiguous" and not delayed.usable
    assert "reason:prior_epoch_replay" in delayed.evidence_refs


def test_control_replay_deduplicates_and_stale_or_incompatible_control_fails():
    engine = Machine()
    event = EpochEvidence("tx_start", 0, Stamp(1_000_000, "fixture", "virtual"), "start")
    engine.tx.observe_control(event)
    engine.tx.observe_control(event)
    result = engine(frame(0, 10, ns=1_000_000))
    assert len(result.events) == 1
    engine.tx.observe_control(event)
    with pytest.raises(ValueError, match="stale"):
        engine.tx.observe_control(replace(event, source_ref="late"))
    with pytest.raises(ValueError, match="clock"):
        engine.tx.observe_control(
            replace(event, available=Stamp(2_000_000, "other", "virtual"), source_ref="bad-clock")
        )
    with pytest.raises(ValueError, match="RX controls"):
        engine.tx.observe_control(replace(event, kind="rx_boot", source_ref="rx-boot"))


def test_queued_control_cannot_use_different_clock_at_first_frame():
    engine = Machine()
    engine.tx.observe_control(EpochEvidence("tx_start", 0, Stamp(1, "other", "unix_wall"), "start"))
    with pytest.raises(ValueError, match="incompatible clock"):
        engine(frame(0, 1))


def test_bounded_history_never_reclassifies_very_old_sequence_as_unique():
    policy = EpochPolicy(max_seen_per_link=20)
    engine = Machine(policy)
    for seq in range(100):
        engine(frame(seq, seq * 1000, ordinal=seq))
    old = engine(frame(0, 0, ordinal=100))
    assert not old.usable and old.classification == "ambiguous"
    assert len(engine.tx._links[(0, 0)].seen) == 20


def test_different_future_suffixes_leave_same_prefix_results_unchanged():
    prefix = [frame(100, 100_000), frame(0, 110_000, ordinal=1)]
    engines = [Machine(), Machine()]
    results = [[engine(value) for value in prefix] for engine in engines]
    engines[0](frame(101, 120_000, ordinal=2))
    engines[1](frame(1, 120_000, ordinal=2))
    engines[1](frame(2, 130_000, ordinal=3))
    assert results[0] == results[1]
    assert results[0][-1].tx_epoch is None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"restart_confirm_packets": 1},
        {"timer_reorder_us": 0},
        {"max_seen_per_link": 16},
        {"wrap_guard": 1 << 31},
        {"max_forward_timer_us": True},
    ],
)
def test_policy_rejects_unusable_thresholds(kwargs):
    with pytest.raises(ValueError, match="EpochPolicy"):
        EpochPolicy(**kwargs)
