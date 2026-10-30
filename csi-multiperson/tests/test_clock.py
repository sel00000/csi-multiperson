"""S04 arrival-prefix invariance, shared clocks and independent RX fits."""

from dataclasses import FrozenInstanceError, replace
import json
import struct

import numpy as np
import pytest

from csi_mp.data.schema import ContractError, SourceSpan, Stamp, WireFrame
from csi_mp.ingest.clock import ClockPolicy, OnlineClockMapper
from csi_mp.ingest.epochs import RxTimerUnwrapper, TxSequenceEpochs, U32
from csi_mp.ingest.events import ClockTransform


BASE = 1_800_000_000_000_000_001  # Deliberately not exactly representable as float64.


def frame(seq, timer, ns, *, rx=0, tx=0, boot=1, infer=None, domain="bridge", kind="unix_wall"):
    wire = struct.pack("<BBIIB", rx, tx, seq, timer, boot)
    return WireFrame(
        (SourceSpan("whole-file-hash", f"rx{rx}", seq, 8 + seq * 130, len(wire)),),
        wire,
        rx,
        tx,
        seq,
        timer,
        boot,
        np.zeros((56, 2), dtype=np.int8),
        -40,
        -90,
        None,
        Stamp(ns, domain, kind),
        infer,
        seq,
    )


class Engine:
    def __init__(self, **kwargs):
        self.rx = RxTimerUnwrapper()
        self.tx = TxSequenceEpochs()
        self.clock = OnlineClockMapper(**kwargs)

    def __call__(self, packet):
        return self.clock.observe(packet, self.tx.classify(packet, self.rx.update(packet)))


def test_affine_rate_reference_warmup_and_large_integer_precision():
    engine = Engine()
    # Known +1000 ppm synthetic slope and a stable 3ns transport offset.
    results = [engine(frame(i, 10_000 + i * 10_000, BASE + 3 + i * 10_010_000)) for i in range(4)]
    assert [r.clock_status for r in results] == ["warming_up", "warming_up", "ready", "ready"]
    assert [r.acquisition_ns for r in results] == [BASE + 3 + i * 10_010_000 for i in range(4)]
    model = results[-1].clock_model
    assert model.rate_numerator == 1001 and model.rate_denominator == 1
    assert model.reference_timer_us == 10_000 and model.reference_ns == BASE + 3
    assert model.sample_count == model.window_count == 4
    assert results[-1].clock_error_ns == 0
    assert results[0].clock_error_ns is None
    assert results[0].clock_reference == Stamp(BASE + 3, "bridge", "unix_wall")
    assert all(r.availability_basis == "legacy_bridge_proxy" for r in results)
    serial = json.loads(json.dumps(results[-1].to_dict()))
    assert serial["acquisition_ns"] == str(BASE + 3 + 3 * 10_010_000)
    assert serial["clock_model"]["reference_ns"] == str(BASE + 3)
    assert serial["clock_model"]["error_basis"] == "empirical_bridge_residual_not_acquisition_bound"


def test_same_prefix_with_different_futures_has_identical_times_and_snapshots():
    prefix = [
        frame(i, 50_000 + i * 10_000, BASE + i * 10_000_000 + (i % 2) * 100) for i in range(6)
    ]
    futures = [
        [],
        [frame(i, 50_000 + i * 10_000, BASE + i * 10_020_000) for i in range(6, 120)],
        [
            frame(6, 500, BASE + 100_000_000, boot=2),
            frame(7, 600, BASE + 100_100_000, boot=2),
            frame(8, 700, BASE + 100_200_000, boot=2),
        ],
    ]
    expected, expected_snapshot = None, None
    for suffix in futures:
        engine = Engine()
        published = [engine(f) for f in prefix]
        serialized = [r.to_dict() for r in published]
        snapshot = engine.clock.snapshot()
        if expected is None:
            expected, expected_snapshot = serialized, snapshot
        assert serialized == expected
        assert snapshot == expected_snapshot
        for f in suffix:
            engine(f)
        assert [r.to_dict() for r in published] == expected
        assert snapshot == expected_snapshot
        with pytest.raises(FrozenInstanceError):
            published[0].acquisition_ns = 0
        with pytest.raises(FrozenInstanceError):
            published[0].clock_model.reference_ns = 0


def test_mapper_emits_without_consuming_future_or_storing_frame_objects():
    engine = Engine()

    def source():
        yield frame(1, 100_000, BASE)
        raise AssertionError("future must remain unread")

    stream = source()
    result = engine(next(stream))
    assert result.acquisition_ns == BASE
    snapshot = json.loads(engine.clock.snapshot())
    assert snapshot["receivers"]["0"]["model"]["sample_count"] == 1
    assert "wire" not in engine.clock.snapshot().decode()


def test_small_timer_reorder_preserves_fit_and_advances_arrival_watermark():
    engine = Engine()
    engine(frame(100, 100_000, BASE))
    previous = engine(frame(102, 120_000, BASE + 20_000_000))
    late = engine(frame(101, 110_000, BASE + 30_000_000))
    assert late.epochs.classification == "reorder" and late.usable
    assert late.acquisition_ns == BASE + 10_000_000
    assert late.clock_model is previous.clock_model
    assert late.availability_ns == BASE + 30_000_000
    assert json.loads(engine.clock.snapshot())["receivers"]["0"]["samples"] == [
        ["100000", str(BASE)],
        ["120000", str(BASE + 20_000_000)],
    ]
    resumed = engine(frame(103, 140_000, BASE + 40_000_000))
    assert resumed.acquisition_ns == BASE + 40_000_000
    assert resumed.clock_model.sample_count == 3


def test_host_regression_after_reorder_is_quarantined_without_clamp():
    engine = Engine()
    engine(frame(100, 100_000, BASE))
    engine(frame(102, 120_000, BASE + 20_000_000))
    late = engine(frame(101, 110_000, BASE + 100_000_000))
    backward = engine(frame(103, 130_000, BASE + 90_000_000))
    assert backward.clock_status == "excluded"
    assert backward.acquisition_ns is None
    assert backward.exclusion_reason == "availability_clock_regression"
    resumed = engine(frame(104, 140_000, BASE + 110_000_000))
    assert resumed.clock_status == "warming_up"
    assert resumed.clock_segment_id != late.clock_segment_id
    assert resumed.clock_model.sample_count == 1
    assert late.acquisition_ns == BASE + 10_000_000


def test_fit_regression_is_quarantined_without_rewriting_prefix_or_resetting_other_rx():
    engine = Engine()
    # A delayed burst shares a bridge timestamp. Refitting that burst can move
    # acquisition backwards while host time never decreases and RX time advances.
    points = [
        (0, 0),
        (3, 1_000_000_000),
        (20_000, 2_000_000_000),
        (1_020_000, 2_000_000_000),
        (1_020_003, 2_000_000_000),
    ]
    published = []
    for seq, (timer, elapsed_ns) in enumerate(points):
        published.append(engine(frame(seq, timer, BASE + elapsed_ns)))
        if seq < 3:
            other = engine(frame(seq, 5_000_000 + seq * 1_000_000, BASE + elapsed_ns, rx=1, tx=1))
    assert all(result.usable for result in published)
    assert other.clock_status == "ready"
    serialized_prefix = [result.to_dict() for result in published]
    prefix_snapshot = engine.clock.snapshot()
    other_state = json.loads(prefix_snapshot)["receivers"]["1"]

    excluded = engine(frame(5, 1_020_005, BASE + 2_000_000_000))
    assert excluded.epochs.usable
    assert excluded.availability_ns >= published[-1].availability_ns
    assert excluded.clock_status == "excluded" and not excluded.usable
    assert excluded.exclusion_reason == "clock_fit_regression"
    assert excluded.acquisition_ns is None
    assert excluded.clock_model.estimate(1_020_005) < published[-1].acquisition_ns
    assert json.loads(engine.clock.snapshot())["receivers"] == {"1": other_state}

    resumed = engine(frame(6, 1_030_005, BASE + 2_010_000_000))
    assert resumed.clock_status == "warming_up"
    assert resumed.clock_segment_id != published[-1].clock_segment_id
    assert resumed.clock_model.sample_count == 1
    assert resumed.acquisition_ns == BASE + 2_010_000_000
    assert json.loads(engine.clock.snapshot())["receivers"]["1"] == other_state
    assert [result.to_dict() for result in published] == serialized_prefix
    assert (
        json.loads(prefix_snapshot)["receivers"]["0"]["model"]
        == published[-1].clock_model.to_dict()
    )


def test_reorder_before_clock_origin_is_excluded_without_clamping_or_resetting_fit():
    engine = Engine()
    first = engine(frame(100, 100_000, 5_000, kind="monotonic"))
    previous = engine(frame(102, 100_010, 15_000, kind="monotonic"))
    fit = json.loads(engine.clock.snapshot())["receivers"]

    late = engine(frame(101, 99_990, 16_000, kind="monotonic"))
    assert late.epochs.classification == "reorder" and late.epochs.usable
    assert late.exclusion_reason == "acquisition_before_clock_origin"
    assert late.clock_status == "excluded" and late.acquisition_ns is None
    assert late.clock_model is previous.clock_model
    assert json.loads(engine.clock.snapshot())["receivers"] == fit

    resumed = engine(frame(103, 100_020, 25_000, kind="monotonic"))
    assert resumed.clock_segment_id == first.clock_segment_id
    assert resumed.clock_model.sample_count == 3
    assert resumed.acquisition_ns == 25_000
    assert previous.acquisition_ns == 15_000


def test_true_timer_and_sequence_wrap_and_pre_wrap_late_packet():
    engine = Engine()
    before = engine(frame(U32 - 3, U32 - 30, BASE))
    wrapped = engine(frame(1, 10, BASE + 40_000))
    late = engine(frame(U32 - 2, U32 - 20, BASE + 50_000))
    after = engine(frame(2, 20, BASE + 60_000))
    assert wrapped.epochs.timer_unwrapped_us == U32 + 10
    assert {e.kind for e in wrapped.epochs.events} == {"timer_wrap", "seq_wrap"}
    assert wrapped.acquisition_ns == BASE + 40_000
    assert late.acquisition_ns == BASE + 10_000
    assert late.clock_model is wrapped.clock_model
    assert after.clock_segment_id == before.clock_segment_id
    assert after.clock_model.sample_count == 3
    assert after.acquisition_ns >= wrapped.acquisition_ns


def test_multiple_receiver_offsets_share_one_time_axis_and_reset_is_local():
    engine = Engine()
    offsets = [10_000, 500_000, 8_000_000]
    for seq in range(3):
        observations = [
            engine(frame(seq, timer + seq * 10_000, BASE + seq * 10_000_000, rx=rx))
            for rx, timer in enumerate(offsets)
        ]
        assert {o.acquisition_ns for o in observations} == {BASE + seq * 10_000_000}
        assert {o.clock_domain for o in observations} == {"bridge"}
    previous = [o.clock_model for o in observations]
    rebooted = engine(frame(3, 100, BASE + 30_000_000, rx=0, boot=2))
    continuing = engine(frame(3, offsets[1] + 30_000, BASE + 30_000_000, rx=1))
    assert rebooted.clock_status == "warming_up"
    assert rebooted.clock_model.rx_epoch != previous[0].rx_epoch
    assert continuing.clock_status == "ready"
    assert continuing.clock_segment_id == previous[1].segment_id
    snapshot = json.loads(engine.clock.snapshot())
    assert snapshot["receivers"]["2"]["model"] == previous[2].to_dict()


def test_tx_restart_does_not_reset_rx_clock_and_uncertain_frames_never_train():
    engine = Engine()
    before = engine(frame(100, 100_000, BASE))
    snapshot = json.loads(engine.clock.snapshot())["receivers"]
    first = engine(frame(0, 110_000, BASE + 10_000_000))
    second = engine(frame(1, 120_000, BASE + 20_000_000))
    assert first.clock_status == second.clock_status == "excluded"
    assert first.acquisition_ns is second.acquisition_ns is None
    assert first.exclusion_reason == "epoch_ambiguous"
    assert json.loads(engine.clock.snapshot())["receivers"] == snapshot
    reset = engine(frame(2, 130_000, BASE + 30_000_000))
    assert reset.epochs.classification == "tx_reset"
    assert reset.clock_segment_id == before.clock_segment_id
    assert reset.clock_model.sample_count == 2


def test_duplicate_and_ambiguous_timer_do_not_poison_fit():
    engine = Engine()
    packet = frame(100, 1_000_000, BASE)
    before = engine(packet)
    snapshot = json.loads(engine.clock.snapshot())["receivers"]
    duplicate = engine(replace(packet, bridge_complete=Stamp(BASE + 1_000, "bridge", "unix_wall")))
    invalid = engine(frame(101, 1, BASE + 10_000_000))
    assert duplicate.exclusion_reason == "epoch_duplicate"
    assert invalid.exclusion_reason == "epoch_ambiguous"
    assert invalid.epochs.timer_unwrapped_us is None
    assert json.loads(engine.clock.snapshot())["receivers"] == snapshot
    resumed = engine(frame(102, 1_020_000, BASE + 20_000_000))
    assert resumed.clock_segment_id == before.clock_segment_id
    assert resumed.clock_model.sample_count == 2


def test_excluded_duplicate_advances_arrival_clock_without_training_fit():
    engine = Engine()
    packet = frame(100, 100_000, BASE)
    first = engine(packet)
    duplicate = engine(
        replace(packet, bridge_complete=Stamp(BASE + 100_000_000, "bridge", "unix_wall"))
    )
    assert duplicate.exclusion_reason == "epoch_duplicate"
    state = json.loads(engine.clock.snapshot())
    assert state["receivers"]["0"]["model"] == first.clock_model.to_dict()
    backward = engine(frame(101, 110_000, BASE + 90_000_000))
    assert backward.exclusion_reason == "availability_clock_regression"
    assert backward.acquisition_ns is None
    resumed = engine(frame(102, 120_000, BASE + 110_000_000))
    assert resumed.clock_segment_id != first.clock_segment_id


def test_explicit_reset_is_local_and_never_reuses_a_clock_segment():
    engine = Engine()
    first = engine(frame(1, 100_000, BASE))
    other = engine(frame(1, 500_000, BASE, rx=1))
    engine.clock.reset(first.epochs.rx_epoch)
    second = engine(frame(2, 110_000, BASE + 10_000_000))
    assert second.clock_segment_id != first.clock_segment_id
    assert second.clock_model.sample_count == 1
    state = json.loads(engine.clock.snapshot())
    assert state["receivers"]["1"]["model"] == other.clock_model.to_dict()
    unchanged = engine.clock.snapshot()
    engine.clock.reset("rx:2:epoch:never-seen")
    assert engine.clock.snapshot() == unchanged


def test_measured_availability_is_distinct_from_estimated_acquisition():
    engine = Engine(policy=ClockPolicy(legacy_delay_ns=999))
    result = engine(
        frame(
            1, 100_000, BASE, kind="monotonic", infer=Stamp(BASE + 5_000_000, "bridge", "monotonic")
        )
    )
    assert result.acquisition_ns == BASE
    assert result.availability_ns == BASE + 5_000_000
    assert result.availability_basis == "measured_infer"
    assert result.legacy_delay_ns == 0
    # Fixed-lag consumer must check BOTH inequalities, even before S06 exists.
    grid = BASE + 1_000_000
    assert result.acquisition_ns <= grid
    assert result.availability_ns > grid + 1_000_000


def calibration(*, error_ns=7, available_ns=0):
    return ClockTransform(
        "bridge-to-infer.v1",
        Stamp(BASE, "bridge", "unix_wall"),
        Stamp(5_000_000_001, "infer", "monotonic"),
        Stamp(available_ns, "infer", "monotonic"),
        "calibration:prior",
        error_ns,
    )


def test_explicit_cross_clock_transform_preserves_offsets_errors_and_evidence():
    transform = calibration()
    engine = Engine(reference=Stamp(0, "infer", "monotonic"), transforms=(transform,))
    for i in range(3):
        observed = engine(
            frame(
                i,
                i * 10_000,
                BASE + i * 10_000_000,
                infer=Stamp(5_005_000_001 + i * 10_000_000, "infer", "monotonic"),
            )
        )
    assert observed.acquisition_ns == 5_020_000_001
    assert observed.availability_ns == 5_025_000_001
    assert observed.clock_error_ns == 7
    assert observed.clock_domain == "infer" and observed.clock_kind == "monotonic"
    assert observed.transform_ids == (transform.transform_id,)
    assert json.loads(engine.clock.snapshot())["transforms"] == [transform.to_dict()]


@pytest.mark.parametrize("kind,domain", [("monotonic", "bridge"), ("unix_wall", "other")])
def test_domain_or_kind_mismatch_requires_transform_without_mutating_state(kind, domain):
    engine = Engine(reference=Stamp(0, "bridge", "unix_wall"))
    engine(frame(1, 100_000, BASE))
    before = engine.clock.snapshot()
    with pytest.raises(ContractError, match="explicit clock transform"):
        engine(frame(2, 110_000, BASE + 10_000_000, kind=kind, domain=domain))
    assert engine.clock.snapshot() == before


def test_bridge_wall_and_infer_monotonic_are_never_implicitly_subtracted():
    engine = Engine()
    before = engine.clock.snapshot()
    with pytest.raises(ContractError, match="explicit clock transform"):
        engine(frame(0, 0, BASE, infer=Stamp(10_000, "infer", "monotonic")))
    assert engine.clock.snapshot() == before


def test_future_calibration_cannot_leak_into_prefix():
    engine = Engine(
        reference=Stamp(0, "infer", "monotonic"),
        transforms=(calibration(available_ns=6_000_000_000),),
    )
    before = engine.clock.snapshot()
    with pytest.raises(ContractError, match="future calibration"):
        engine(frame(0, 0, BASE, infer=Stamp(5_100_000_000, "infer", "monotonic")))
    assert engine.clock.snapshot() == before


def test_missing_calibration_error_is_unknown_even_after_warmup():
    engine = Engine(
        reference=Stamp(0, "infer", "monotonic"), transforms=(calibration(error_ns=None),)
    )
    for i in range(3):
        observation = engine(frame(i, i * 10_000, BASE + i * 10_000_000))
    assert observation.clock_status == "ready"
    assert observation.clock_model.residual_ns == 0
    assert observation.clock_error_ns is None
    assert observation.availability_basis == "legacy_bridge_proxy"


@pytest.mark.parametrize("error_ns", [None, 5_000_000])
def test_fit_retains_contributing_calibration_uncertainty_when_bridge_source_changes(error_ns):
    transform = calibration(error_ns=error_ns)
    engine = Engine(
        reference=Stamp(0, "infer", "monotonic"),
        transforms=(transform,),
        policy=ClockPolicy(window_size=3),
    )
    for i in range(3):
        previous = engine(frame(i * 2, i * 10_000, BASE + i * 10_000_000))
    # Late frame is already in the common domain, but its estimate still comes
    # from a calibrated model: current-frame error=0 cannot erase that evidence.
    late = engine(frame(3, 15_000, 5_025_000_001, domain="infer", kind="monotonic"))
    assert late.clock_model is previous.clock_model
    assert late.clock_error_ns == error_ns
    assert transform.transform_id in late.transform_ids
    for i in range(3, 6):
        result = engine(
            frame(
                i * 2, i * 10_000, 5_000_000_001 + i * 10_000_000, domain="infer", kind="monotonic"
            )
        )
        if i < 5:
            assert result.clock_error_ns == error_ns
            assert transform.transform_id in result.transform_ids
    # The old calibration stops contributing once its last sample leaves the window.
    assert result.clock_error_ns == 0 and not result.transform_ids
    assert previous.clock_error_ns == error_ns
    assert previous.clock_model.transform_ids == (transform.transform_id,)


def test_infer_arrival_before_mapped_bridge_is_excluded():
    engine = Engine(reference=Stamp(0, "infer", "monotonic"), transforms=(calibration(),))
    before = engine.clock.snapshot()
    result = engine(frame(0, 0, BASE, infer=Stamp(5_000_000_000, "infer", "monotonic")))
    assert result.exclusion_reason == "availability_before_bridge"
    assert result.acquisition_ns is None
    assert engine.clock.snapshot() == before


def test_bridge_regression_is_visible_when_infer_arrival_is_monotonic():
    engine = Engine(reference=Stamp(0, "infer", "monotonic"), transforms=(calibration(),))
    first = engine(frame(0, 100_000, BASE, infer=Stamp(6_000_000_000, "infer", "monotonic")))
    backward = engine(frame(1, 110_000, BASE - 1, infer=Stamp(6_000_001_000, "infer", "monotonic")))
    assert backward.exclusion_reason == "bridge_clock_regression"
    resumed = engine(frame(2, 120_000, BASE + 1, infer=Stamp(6_000_002_000, "infer", "monotonic")))
    assert resumed.clock_segment_id != first.clock_segment_id
    assert resumed.clock_model.sample_count == 1


@pytest.mark.parametrize("virtual,declared_synthetic", [(True, False), (False, True)])
def test_synthetic_origin_is_explicit(virtual, declared_synthetic):
    engine = Engine(synthetic=declared_synthetic)
    observed = engine(frame(1, 1, BASE, kind="virtual" if virtual else "unix_wall"))
    assert observed.availability_basis == "synthetic"
    assert observed.clock_kind == ("virtual" if virtual else "unix_wall")


def test_fit_history_is_bounded_and_rate_limit_is_reported():
    engine = Engine(policy=ClockPolicy(window_size=4, max_drift_ppm=100))
    for i in range(100):
        result = engine(frame(i, i * 10_000, BASE + i * 10_020_000))
    model = result.clock_model
    assert model.rate_limited and model.sample_count == 100 and model.window_count == 4
    assert model.rate_numerator == 10001 and model.rate_denominator == 10
    assert result.clock_error_ns > 0
    assert len(json.loads(engine.clock.snapshot())["receivers"]["0"]["samples"]) == 4


def test_warmup_requires_elapsed_device_span_and_snapshot_bytes_are_detached():
    engine = Engine()
    for i in range(3):
        result = engine(frame(i, i, BASE + i * 1000))
    assert result.clock_status == "warming_up" and result.clock_error_ns is None
    snapshot = engine.clock.snapshot()
    decoded = json.loads(snapshot)
    decoded["receivers"]["0"]["samples"].clear()
    assert snapshot == engine.clock.snapshot()
    later = engine(frame(3, 10_000, BASE + 10_000_000))
    assert later.clock_status == "ready"


@pytest.mark.parametrize(
    "changes",
    [
        {"window_size": 0},
        {"warmup_samples": 1},
        {"warmup_samples": 65},
        {"min_span_us": 0},
        {"max_drift_ppm": -1},
        {"max_drift_ppm": 1_000_000},
        {"legacy_delay_ns": -1},
        {"window_size": True},
        {"min_span_us": 1.5},
        {"policy_id": ""},
    ],
)
def test_invalid_clock_policies_report_the_contract_field(changes):
    with pytest.raises(ContractError, match="ClockPolicy"):
        ClockPolicy(**changes)


def test_clock_transform_validation_and_rational_rate():
    transform = replace(calibration(), rate_numerator=3, rate_denominator=2)
    assert transform.map_ns(Stamp(BASE + 2, "bridge", "unix_wall")) == 5_000_000_004
    for changes in (
        {"error_ns": -1},
        {"rate_numerator": 0},
        {"rate_denominator": 0},
        {"evidence_ref": ""},
        {"available": Stamp(0, "other", "monotonic")},
    ):
        with pytest.raises(ContractError, match="ClockTransform"):
            replace(transform, **changes)
    with pytest.raises(ContractError, match="duplicate source/target"):
        OnlineClockMapper(transforms=(transform, transform))
