"""S06 deadlines, discontinuities and T100 windows with an explicit fixture transform.

The feature double copies I and takes an unscaled difference. It is deliberately
not the S07 log-amplitude formula and is never an inference implementation.
"""

from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest

from csi_mp.config import ConfigError, load_config
from csi_mp.data.contracts import validate_window
from csi_mp.data.grid import CausalGridBuilder, GridPolicy, GridSlot, SlotFeatures
from csi_mp.data.schema import (
    FEATURE_SPEC_ID,
    ContractError,
    DataSpec,
    EpochEvent,
    EpochResult,
    SourceSpan,
    Stamp,
    WireFrame,
)
from csi_mp.ingest.clock import OnlineClockMapper
from csi_mp.ingest.epochs import RxTimerUnwrapper, TxSequenceEpochs
from csi_mp.ingest.events import ClockModel, TimedObservation


MS = 1_000_000
STEP = 10 * MS
BASE = 1_800_000_000_000_000_001  # Grid arithmetic must not round through float64.


class FixtureFeatures:
    """A transparent stateful spy; reset preserves previously published features."""

    feature_id = FEATURE_SPEC_ID

    def __init__(self):
        self.previous = {}
        self.features = {}
        self.calls = []

    def on_observation(self, observation):
        self.calls.append(("observation", observation.observation_id))
        link = observation.frame.link_index
        value = observation.frame.iq[:, 0].astype(np.float32)
        values = np.zeros((56, 2), dtype=np.float32)
        valid = np.zeros((56, 2), dtype=np.bool_)
        values[:, 0], valid[:, 0] = value, True
        previous = self.previous.get(link)
        if previous is not None and previous.acquisition_ns < observation.acquisition_ns:
            values[:, 1] = value - previous.frame.iq[:, 0].astype(np.float32)
            valid[:, 1] = True
        self.features[observation.observation_id] = SlotFeatures(values, valid)
        if previous is None or previous.acquisition_ns < observation.acquisition_ns:
            self.previous[link] = observation

    def values_for_slot(self, observation_id, is_new):
        # Returning the same delta even for holds tests grid-side enforcement.
        return self.features[observation_id]

    def reset(self, links, reason):
        self.calls.append(("reset", tuple(links), reason))
        for link in links:
            self.previous.pop(link, None)


def observation(
    number,
    acquisition_ms,
    availability_ms=None,
    *,
    link=0,
    segment="a",
    clock_segment="a",
    usable=True,
    classification="normal",
    reason=None,
    events=(),
    value=None,
):
    """Independent timed input: all supplied millisecond values are BASE offsets."""
    if availability_ms is None:
        availability_ms = acquisition_ms
    rx, tx = divmod(link, 3)
    available = Stamp(BASE + availability_ms * MS, "grid", "virtual")
    acquired = BASE + acquisition_ms * MS
    wire = f"fixture:{number}:{link}".encode()
    frame = WireFrame(
        (SourceSpan("fixture-source", f"rx{rx}", number, number * 130, len(wire)),),
        wire,
        rx,
        tx,
        number,
        1_000_000 + acquisition_ms * 1000,
        1,
        np.full((56, 2), number % 100 if value is None else value, dtype=np.int8),
        -40,
        -90,
        None,
        available,
        None,
        number,
    )
    epochs = EpochResult(
        f"rx:{rx}:epoch:0",
        f"tx:{tx}:epoch:0" if classification != "ambiguous" else None,
        f"link:{link}:segment:{segment}",
        frame.rx_timer_u32_us if usable else None,
        classification,
        usable,
        ("fixture-only",),
        events,
    )
    model = ClockModel(
        f"model:{number}:{link}",
        epochs.rx_epoch,
        f"rx:{rx}:clock:{clock_segment}",
        frame.rx_timer_u32_us,
        acquired,
        1000,
        1,
        3,
        3,
        20_000,
        "ready",
        0,
        False,
        0,
        (),
    )
    return TimedObservation(
        f"observation:{number}:{link}",
        frame,
        epochs,
        acquired if usable else None,
        available.ns,
        "synthetic",
        "grid",
        "virtual",
        Stamp(BASE, "grid", "virtual"),
        model if usable else None,
        0 if usable else None,
        "ready" if usable else "excluded",
        None if usable else reason or f"epoch_{classification}",
    )


def builder(*, features=None, policy=None, **kwargs):
    return CausalGridBuilder(
        session_id="fixture",
        start=Stamp(BASE, "grid", "virtual"),
        feature_builder=FixtureFeatures() if features is None else features,
        provenance_id="fixture-only",
        availability_basis="synthetic",
        policy=GridPolicy() if policy is None else policy,
        **kwargs,
    )


def slot_fingerprint(slot):
    return (
        slot.grid_ns,
        slot.deadline_ns,
        slot.raw_x.tobytes(),
        slot.feature_valid.tobytes(),
        slot.observed_mask.tobytes(),
        slot.link_state.tobytes(),
        slot.age_ns.tobytes(),
        slot.observation_ids,
        slot.segment_ids,
    )


def test_default_policy_and_complete_100_slot_shape_and_integer_grid():
    policy = GridPolicy()
    assert policy.step_ns == 10 * MS
    assert policy.output_step_ns == 50 * MS
    assert policy.settle_ns == 0
    assert policy.max_hold_age_ns == 30 * MS
    grid = builder()
    for number in range(100):
        grid.ingest(observation(number, 10 * number))
    slots = grid.commit_through(BASE + 990 * MS)
    assert len(slots) == 100
    assert all(isinstance(slot, GridSlot) for slot in slots)
    window = grid.window_at(BASE + 990 * MS)
    validate_window(window)
    assert window.status == "ready"
    assert window.raw_x.shape == window.feature_valid.shape == (100, 9, 56, 2)
    np.testing.assert_array_equal(window.grid_ns, BASE + np.arange(100, dtype=np.int64) * STEP)
    assert window.observed_mask[:, 0].all()
    assert np.all(window.age_ns[:, 0] == 0)
    assert np.all(window.age_ns[:, 1:] == -1)
    assert not window.feature_valid[0, 0, :, 1].any()
    assert window.feature_valid[1:, 0, :, 1].all()
    assert grid.metrics()["observed_slots"] == 100
    assert grid.metrics()["held_slots"] == 0
    assert grid.metrics()["unavailable_slots"] == 800


def test_warmup_uses_invalid_left_padding_until_100_real_slots():
    grid = builder()
    grid.ingest(observation(1, 0))
    grid.commit_through(BASE)
    first = grid.window_at(BASE)
    validate_window(first)
    assert first.status == "warming_up"
    assert not first.feature_valid[:-1].any()
    assert np.all(first.age_ns[:-1] == -1)
    assert first.grid_ns[0] == BASE - 990 * MS
    assert first.observed_mask[-1, 0]
    grid.commit_through(BASE + 980 * MS)
    assert grid.window_at(BASE + 980 * MS).status == "warming_up"
    grid.ingest(observation(2, 990))
    grid.commit_through(BASE + 990 * MS)
    assert grid.window_at(BASE + 990 * MS).status == "ready"


def test_all_links_missing_has_no_features_and_insufficient_coverage_after_warmup():
    grid = builder()
    grid.commit_through(BASE + 990 * MS)
    window = grid.window_at(BASE + 990 * MS)
    validate_window(window)
    assert window.status == "insufficient_coverage"
    assert not window.raw_x.any()
    assert not window.feature_valid.any()
    assert not window.observed_mask.any()
    assert not window.link_state.any()
    assert np.all(window.age_ns == -1)
    assert grid.metrics()["unavailable_slots"] == 900


def test_missing_current_slot_is_insufficient_even_with_older_context():
    grid = builder()
    grid.ingest(observation(1, 900))
    grid.commit_through(BASE + 990 * MS)
    window = grid.window_at(BASE + 990 * MS)
    validate_window(window)
    assert window.feature_valid[:-1].any()
    assert not window.feature_valid[-1].any()
    assert window.status == "insufficient_coverage"


def test_observed_held_and_expired_missing_preserve_age_and_masks():
    grid = builder()
    sample = observation(1, 0)
    grid.ingest(sample)
    slots = grid.commit_through(BASE + 40 * MS)
    assert [int(slot.link_state[0]) for slot in slots] == [1, 2, 2, 2, 0]
    assert [int(slot.age_ns[0]) for slot in slots] == [0, 10 * MS, 20 * MS, 30 * MS, -1]
    assert [bool(slot.observed_mask[0]) for slot in slots] == [True, False, False, False, False]
    assert all(slot.observation_ids[0] == sample.observation_id for slot in slots[:-1])
    assert slots[-1].observation_ids[0] is None
    assert slots[-1].segment_ids[0] is None
    assert not slots[-1].raw_x[0].any()
    assert not slots[-1].feature_valid[0].any()
    assert grid.metrics()["observed_slots"] == 1
    assert grid.metrics()["held_slots"] == 3
    assert grid.metrics()["unavailable_slots"] == 41


def test_hold_cannot_repeat_a_previous_valid_delta_even_if_provider_returns_it():
    grid = builder()
    grid.ingest(observation(1, 0, value=2))
    grid.ingest(observation(2, 10, value=5))
    first, second, held = grid.commit_through(BASE + 20 * MS)
    assert not first.feature_valid[0, :, 1].any()
    assert second.feature_valid[0, :, 1].all()
    assert np.all(second.raw_x[0, :, 1] == 3)
    assert held.feature_valid[0, :, 0].all()
    assert np.all(held.raw_x[0, :, 0] == 5)
    assert not held.feature_valid[0, :, 1].any()
    assert not held.raw_x[0, :, 1].any()


def test_late_packet_cannot_fill_closed_slot_but_is_new_in_next_eligible_slot():
    grid = builder(policy=GridPolicy(settle_ns=10 * MS))
    sample = observation(1, 99, 111)
    grid.ingest(sample)
    earlier = grid.commit_through(BASE + 110 * MS)
    closed = earlier[-1]
    before = slot_fingerprint(closed)
    assert closed.grid_ns == BASE + 100 * MS
    assert closed.deadline_ns == BASE + 110 * MS
    assert closed.link_state[0] == 0
    new, held = grid.commit_through(BASE + 130 * MS)
    assert new.grid_ns == BASE + 110 * MS
    assert new.link_state[0] == 1
    assert new.age_ns[0] == 11 * MS
    assert new.observation_ids[0] == sample.observation_id
    assert held.link_state[0] == 2
    assert held.age_ns[0] == 21 * MS
    assert slot_fingerprint(closed) == before
    assert grid.metrics()["late_observations"] == 1
    assert grid.metrics()["late_ingest_observations"] == 0
    assert grid.window_at(BASE + 120 * MS).decision_ns == BASE + 130 * MS


def test_future_acquisition_is_excluded_even_when_arrival_is_before_deadline():
    grid = builder(policy=GridPolicy(settle_ns=10 * MS))
    grid.ingest(observation(1, 101, 105))
    slots = grid.commit_through(BASE + 110 * MS)
    assert slots[-1].grid_ns == BASE + 100 * MS
    assert slots[-1].link_state[0] == 0
    assert grid.commit_through(BASE + 120 * MS)[0].link_state[0] == 1


def test_deadline_is_inclusive_and_commit_before_deadline_emits_nothing():
    grid = builder(policy=GridPolicy(settle_ns=10 * MS))
    grid.ingest(observation(1, 99, 110))
    before = grid.commit_through(BASE + 110 * MS - 1)
    assert before[-1].grid_ns == BASE + 90 * MS
    boundary = grid.commit_through(BASE + 110 * MS)
    assert len(boundary) == 1
    assert boundary[0].grid_ns == BASE + 100 * MS
    assert boundary[0].link_state[0] == 1
    assert boundary[0].age_ns[0] == MS
    assert grid.metrics()["late_observations"] == 0
    assert grid.commit_through(BASE + 110 * MS) == []


def test_arrival_one_nanosecond_after_deadline_is_late():
    grid = builder(policy=GridPolicy(settle_ns=10 * MS))
    sample = observation(1, 99, 110)
    grid.ingest(replace(sample, availability_ns=sample.availability_ns + 1))
    assert grid.commit_through(BASE + 110 * MS)[-1].link_state[0] == 0
    assert grid.commit_through(BASE + 120 * MS)[0].link_state[0] == 1
    assert grid.metrics()["late_observations"] == 1


def test_late_ingest_does_not_rewrite_committed_history():
    grid = builder()
    grid.commit_through(BASE + 10 * MS)
    old = grid.window_at(BASE + 10 * MS)
    grid.ingest(observation(1, 10, 10))
    new = grid.commit_through(BASE + 20 * MS)[0]
    assert not old.observed_mask.any()
    assert new.link_state[0] == 1
    assert new.age_ns[0] == 10 * MS
    assert grid.metrics()["late_ingest_observations"] == 1


def test_reordered_older_acquisition_does_not_replace_latest_candidate():
    grid = builder()
    latest = observation(2, 20)
    grid.ingest(observation(1, 0))
    grid.ingest(latest)
    grid.ingest(observation(3, 10, 30, classification="reorder"))
    slots = grid.commit_through(BASE + 30 * MS)
    assert slots[2].observation_ids[0] == latest.observation_id
    assert slots[3].observation_ids[0] == latest.observation_id
    assert slots[3].link_state[0] == 2
    assert slots[3].age_ns[0] == 10 * MS


def test_equal_availability_uses_ingest_order_for_equal_acquisition():
    grid = builder()
    first = observation(2, 10, value=2)
    second = observation(1, 10, value=7)
    grid.ingest(first)
    grid.ingest(second)
    slot = grid.commit_through(BASE + 10 * MS)[-1]
    assert slot.observation_ids[0] == second.observation_id
    assert np.all(slot.raw_x[0, :, 0] == 7)


def test_batch_ingest_and_incremental_stream_have_identical_committed_slots():
    trace = [
        observation(1, 0),
        observation(2, 10),
        observation(3, 21, 25),
        observation(4, 15, 31, classification="reorder"),
        observation(5, 40, segment="b"),
        observation(6, 50, segment="b", clock_segment="b"),
        observation(7, 150, segment="b", clock_segment="b"),
        observation(8, 160, 180, segment="b", clock_segment="b"),
    ]
    policy = GridPolicy(settle_ns=5 * MS)
    batch_features, stream_features = FixtureFeatures(), FixtureFeatures()
    batch = builder(features=batch_features, policy=policy)
    streamed = builder(features=stream_features, policy=policy)
    for sample in trace:
        batch.ingest(sample)
    assert batch_features.calls == []
    batch_slots = batch.commit_through(BASE + 205 * MS)
    stream_slots = []
    position = 0
    for milliseconds in range(5, 206, 10):
        deadline = BASE + milliseconds * MS
        while position < len(trace) and trace[position].availability_ns <= deadline:
            streamed.ingest(trace[position])
            position += 1
        stream_slots.extend(streamed.commit_through(deadline))
    assert [slot_fingerprint(slot) for slot in batch_slots] == [
        slot_fingerprint(slot) for slot in stream_slots
    ]
    assert batch_features.calls == stream_features.calls


def test_recorded_arrival_regression_preserves_admission_order_in_batch_and_stream():
    first = observation(1, 0)
    second = observation(2, 20)
    regressed = replace(
        observation(3, 15),
        acquisition_ns=None,
        clock_status="excluded",
        exclusion_reason="availability_clock_regression",
    )
    recovered = observation(4, 25, clock_segment="b")
    trace = [first, second, regressed, recovered]
    original_timing = [sample.to_dict() for sample in trace]
    batch, streamed = builder(), builder()
    for sample in trace:
        batch.ingest(sample)
    batch_slots = batch.commit_through(BASE + 30 * MS)
    for sample in trace[:2]:
        streamed.ingest(sample)
    stream_slots = streamed.commit_through(BASE + 20 * MS)
    for sample in trace[2:]:
        streamed.ingest(sample)
    stream_slots.extend(streamed.commit_through(BASE + 30 * MS))
    assert [slot_fingerprint(slot) for slot in batch_slots] == [
        slot_fingerprint(slot) for slot in stream_slots
    ]
    assert batch_slots[2].observation_ids[0] == second.observation_id
    assert batch_slots[2].link_state[0] == 1
    assert batch_slots[3].observation_ids[0] == recovered.observation_id
    assert batch_slots[3].link_state[0] == 1
    assert not batch_slots[3].feature_valid[0, :, 1].any()
    assert batch.metrics()["availability_regressions"] == 1
    assert streamed.metrics()["availability_regressions"] == 1
    assert batch.metrics()["late_ingest_observations"] == 0
    assert streamed.metrics()["late_ingest_observations"] == 1
    assert [sample.to_dict() for sample in trace] == original_timing
    assert regressed.availability_ns == BASE + 15 * MS


def test_future_reset_and_future_feature_values_cannot_change_prefix():
    prefix = [observation(1, 0), observation(2, 10)]
    event = EpochEvent(
        "future-reset",
        "rx",
        0,
        "rx_reset",
        "confirmed",
        Stamp(BASE + 40 * MS, "grid", "virtual"),
        (0, 1, 2),
        ("fixture",),
    )
    future = observation(3, 40, events=(event,), segment="future", value=99)
    baseline, actual = builder(), builder()
    for sample in prefix:
        baseline.ingest(sample)
        actual.ingest(sample)
    actual.ingest(future)
    expected = baseline.commit_through(BASE + 20 * MS)
    published = actual.commit_through(BASE + 20 * MS)
    fingerprints = [slot_fingerprint(slot) for slot in published]
    assert fingerprints == [slot_fingerprint(slot) for slot in expected]
    actual.commit_through(BASE + 80 * MS)
    assert [slot_fingerprint(slot) for slot in published] == fingerprints


def test_features_are_snapshotted_before_later_callback_changes_provider_state():
    class LastValueFeatures:
        def on_observation(self, sample):
            self.latest = sample.frame.iq.astype(np.float32)

        def values_for_slot(self, observation_id, is_new):
            return SlotFeatures(self.latest, np.ones((56, 2), dtype=np.bool_))

        def reset(self, links, reason):
            pass

    grid = builder(features=LastValueFeatures(), policy=GridPolicy(settle_ns=50 * MS))
    grid.ingest(observation(1, 0, value=2))
    grid.ingest(observation(2, 40, value=7))
    slots = grid.commit_through(BASE + 90 * MS)
    assert np.all(slots[0].raw_x[0, :, 0] == 2)
    assert np.all(slots[1].raw_x[0, :, 0] == 2)
    assert np.all(slots[4].raw_x[0, :, 0] == 7)
    assert not slots[1].feature_valid[0, :, 1].any()


@pytest.mark.parametrize("scope,links", [("rx", (0, 1, 2)), ("tx", (0, 3, 6))])
def test_reset_event_invalidates_device_scope_even_on_excluded_observation(scope, links):
    features = FixtureFeatures()
    grid = builder(features=features)
    for link in range(9):
        grid.ingest(observation(link + 1, 0, link=link))
    event = EpochEvent(
        "reset",
        scope,
        0,
        f"{scope}_reset",
        "confirmed",
        Stamp(BASE + 10 * MS, "unmapped-source-clock", "monotonic"),
        links,
        ("fixture",),
    )
    grid.ingest(observation(20, 10, usable=False, classification="ambiguous", events=(event,)))
    slots = grid.commit_through(BASE + 10 * MS)
    for link in range(9):
        assert slots[-1].link_state[link] == (0 if link in links else 2)
    reset_links = set(link for call in features.calls if call[0] == "reset" for link in call[1])
    assert set(links) <= reset_links


def test_shared_reset_event_is_applied_once_across_receiver_observations():
    features = FixtureFeatures()
    grid = builder(features=features)
    for link in (0, 3, 6):
        grid.ingest(observation(link + 1, 0, link=link))
    event = EpochEvent(
        "one-tx-reset",
        "tx",
        0,
        "tx_reset",
        "confirmed",
        Stamp(BASE + 10 * MS, "grid", "virtual"),
        (0, 3, 6),
        ("fixture",),
    )
    first = observation(10, 10, events=(event,))
    grid.ingest(first)
    grid.ingest(observation(11, 20, link=3, events=(event,)))
    slots = grid.commit_through(BASE + 20 * MS)
    assert slots[-1].link_state[0] == 2
    assert slots[-1].observation_ids[0] == first.observation_id
    assert slots[-1].link_state[3] == 1
    assert slots[-1].link_state[6] == 0
    assert len([call for call in features.calls if call[0] == "reset"]) == 1


@pytest.mark.parametrize("kind", ["timer_wrap", "seq_wrap", "timer_regression"])
def test_wrap_and_normal_reorder_events_do_not_break_continuity(kind):
    features = FixtureFeatures()
    grid = builder(features=features)
    grid.ingest(observation(1, 0))
    event = EpochEvent(
        kind,
        "tx" if kind == "seq_wrap" else "rx",
        0,
        kind,
        "inferred",
        Stamp(BASE + 10 * MS, "grid", "virtual"),
        (0, 1, 2),
        ("fixture",),
    )
    classification = "reorder" if kind == "timer_regression" else kind
    grid.ingest(observation(2, 10, events=(event,), classification=classification))
    final = grid.commit_through(BASE + 10 * MS)[-1]
    assert final.link_state[0] == 1
    assert final.feature_valid[0, :, 1].all()
    assert not any(call[0] == "reset" for call in features.calls)


def test_epoch_segment_change_cuts_feature_history_and_hold_only_for_affected_link():
    grid = builder()
    grid.ingest(observation(1, 0, link=0))
    grid.ingest(observation(2, 0, link=1))
    grid.ingest(observation(3, 10, link=0, segment="b"))
    final = grid.commit_through(BASE + 10 * MS)[-1]
    assert final.link_state[0] == 1
    assert not final.feature_valid[0, :, 1].any()
    assert final.link_state[1] == 2


def test_clock_segment_change_cuts_receiver_links_and_feature_history():
    grid = builder()
    for link in range(4):
        grid.ingest(observation(link + 1, 0, link=link))
    grid.ingest(observation(10, 10, clock_segment="b"))
    final = grid.commit_through(BASE + 10 * MS)[-1]
    assert final.link_state[0] == 1
    assert not final.feature_valid[0, :, 1].any()
    assert final.link_state[1] == final.link_state[2] == 0
    assert final.link_state[3] == 2


def test_excluded_clock_regression_immediately_cuts_receiver_hold():
    grid = builder()
    for link in range(4):
        grid.ingest(observation(link + 1, 0, link=link))
    excluded = replace(
        observation(10, 10),
        acquisition_ns=None,
        clock_status="excluded",
        exclusion_reason="availability_clock_regression",
    )
    grid.ingest(excluded)
    final = grid.commit_through(BASE + 10 * MS)[-1]
    assert np.all(final.link_state[:3] == 0)
    assert final.link_state[3] == 2


def test_retired_link_segment_cannot_be_revived_by_late_observation():
    grid = builder()
    grid.ingest(observation(1, 0))
    active = observation(2, 10, segment="b")
    grid.ingest(active)
    grid.ingest(observation(3, 20, segment="a"))
    final = grid.commit_through(BASE + 20 * MS)[-1]
    assert final.observation_ids[0] == active.observation_id
    assert final.link_state[0] == 2
    assert grid.metrics()["retired_observations"] == 1


def test_retired_receiver_clock_cannot_be_revived_on_another_link():
    grid = builder()
    grid.ingest(observation(1, 0))
    grid.ingest(observation(2, 0, link=1))
    active = observation(3, 10, clock_segment="b")
    grid.ingest(active)
    grid.ingest(observation(4, 20, link=1, clock_segment="a"))
    final = grid.commit_through(BASE + 20 * MS)[-1]
    assert final.observation_ids[0] == active.observation_id
    assert final.link_state[1] == 0
    assert grid.metrics()["retired_observations"] == 1


def test_same_observation_id_is_idempotent_without_recomputing_features():
    features = FixtureFeatures()
    grid = builder(features=features)
    sample = observation(1, 0)
    grid.ingest(sample)
    grid.ingest(sample)
    grid.commit_through(BASE)
    assert features.calls == [("observation", sample.observation_id)]
    assert grid.metrics()["duplicate_observations"] == 1
    assert grid.metrics()["observed_slots"] == 1
    grid.ingest(sample)
    final = grid.commit_through(BASE + 10 * MS)[-1]
    assert final.link_state[0] == 2
    assert grid.metrics()["duplicate_observations"] == 2
    assert features.calls == [("observation", sample.observation_id)]


@pytest.mark.parametrize("gap_ms,delta_valid", [(30, True), (31, False)])
def test_forward_gap_boundary_controls_feature_reset(gap_ms, delta_valid):
    grid = builder()
    grid.ingest(observation(1, 0))
    grid.ingest(observation(2, gap_ms))
    slots = grid.commit_through(BASE + 40 * MS)
    selected = slots[3] if gap_ms == 30 else slots[4]
    assert selected.link_state[0] == 1
    assert bool(selected.feature_valid[0, :, 1].all()) is delta_valid


def test_ambiguous_segment_breaks_hold_but_duplicate_does_not():
    grid = builder()
    grid.ingest(observation(1, 0))
    grid.ingest(observation(2, 10, usable=False, classification="duplicate"))
    grid.ingest(observation(3, 20, usable=False, classification="ambiguous", segment="b"))
    grid.ingest(observation(4, 30, segment="c"))
    slots = grid.commit_through(BASE + 30 * MS)
    assert [int(slot.link_state[0]) for slot in slots] == [1, 2, 0, 1]
    assert not slots[-1].feature_valid[0, :, 1].any()


def test_ambiguous_boundary_retires_previous_segment_before_a_new_usable_segment():
    grid = builder()
    grid.ingest(observation(1, 0, segment="a"))
    grid.ingest(observation(2, 10, segment="b", usable=False, classification="ambiguous"))
    grid.ingest(observation(3, 5, 15, segment="a", classification="reorder"))
    slots = grid.commit_through(BASE + 20 * MS)
    assert [int(slot.link_state[0]) for slot in slots] == [1, 0, 0]
    assert grid.metrics()["retired_observations"] == 1
    recovered = observation(4, 30, segment="c")
    grid.ingest(recovered)
    final = grid.commit_through(BASE + 30 * MS)[0]
    assert final.observation_ids[0] == recovered.observation_id
    assert final.link_state[0] == 1
    assert not final.feature_valid[0, :, 1].any()


def test_clock_regression_without_model_retires_receiver_clock_before_recovery():
    grid = builder()
    grid.ingest(observation(1, 0, link=1, clock_segment="a"))
    grid.ingest(observation(2, 0, link=3, clock_segment="a"))
    excluded = replace(
        observation(3, 10, link=0),
        acquisition_ns=None,
        clock_model=None,
        clock_status="excluded",
        exclusion_reason="availability_clock_regression",
    )
    grid.ingest(excluded)
    grid.ingest(observation(4, 5, 15, link=1, clock_segment="a", classification="reorder"))
    final = grid.commit_through(BASE + 20 * MS)[-1]
    assert np.all(final.link_state[:3] == 0)
    assert final.link_state[3] == 2
    assert grid.metrics()["retired_observations"] == 1
    recovered = observation(5, 30, link=0, clock_segment="b")
    grid.ingest(recovered)
    final = grid.commit_through(BASE + 30 * MS)[0]
    assert final.observation_ids[0] == recovered.observation_id
    assert final.link_state[0] == 1
    assert not final.feature_valid[0, :, 1].any()
    assert final.link_state[1] == 0
    assert final.link_state[3] == 2


def test_excluded_tx_reset_retires_old_epoch_on_other_receiver_before_recovery():
    grid = builder()
    for link in (0, 1, 3, 6):
        grid.ingest(observation(link + 1, 0, link=link))
    event = EpochEvent(
        "excluded-tx-reset",
        "tx",
        0,
        "tx_reset",
        "confirmed",
        Stamp(BASE + 10 * MS, "grid", "virtual"),
        (0, 3, 6),
        ("fixture",),
    )
    excluded = observation(
        10,
        10,
        link=0,
        segment="uncertain",
        usable=False,
        classification="ambiguous",
        events=(event,),
    )
    assert excluded.epochs.tx_epoch is None
    grid.ingest(excluded)
    grid.ingest(observation(11, 5, 15, link=3, classification="reorder"))
    final = grid.commit_through(BASE + 20 * MS)[-1]
    assert all(final.link_state[link] == 0 for link in (0, 3, 6))
    assert final.link_state[1] == 2
    assert grid.metrics()["retired_observations"] == 1
    recovered = observation(12, 30, link=3, segment="new-epoch")
    recovered = replace(recovered, epochs=replace(recovered.epochs, tx_epoch="tx:0:epoch:1"))
    grid.ingest(recovered)
    final = grid.commit_through(BASE + 30 * MS)[0]
    assert final.observation_ids[3] == recovered.observation_id
    assert final.link_state[3] == 1
    assert not final.feature_valid[3, :, 1].any()
    assert final.link_state[0] == final.link_state[6] == 0
    assert final.link_state[1] == 2


def test_queued_invalidation_is_applied_at_its_availability_without_rewriting_prefix():
    grid = builder()
    grid.ingest(observation(1, 0))
    grid.invalidate((0,), "operator boundary", available_ns=BASE + 15 * MS)
    slots = grid.commit_through(BASE + 20 * MS)
    assert [int(slot.link_state[0]) for slot in slots] == [1, 2, 0]


def test_future_invalidation_reservation_does_not_raise_observation_admission_floor():
    grid = builder(policy=GridPolicy(max_hold_age_ns=200 * MS))
    grid.invalidate((0,), "scheduled boundary", available_ns=BASE + 100 * MS)
    first, second = observation(1, 0), observation(2, 10)
    grid.ingest(first)
    grid.ingest(second)
    slots = grid.commit_through(BASE + 10 * MS)
    assert [int(slot.link_state[0]) for slot in slots] == [1, 1]
    assert slots[0].observation_ids[0] == first.observation_id
    assert slots[1].observation_ids[0] == second.observation_id
    assert grid.commit_through(BASE + 90 * MS)[-1].link_state[0] == 2
    assert grid.commit_through(BASE + 100 * MS)[0].link_state[0] == 0
    assert grid.metrics()["availability_regressions"] == 0
    assert grid.metrics()["late_ingest_observations"] == 0


def test_immediate_invalidation_only_changes_future_slots():
    grid = builder()
    grid.ingest(observation(1, 0))
    first = grid.commit_through(BASE)[0]
    before = slot_fingerprint(first)
    grid.invalidate((0,), "operator boundary")
    next_slot = grid.commit_through(BASE + 10 * MS)[0]
    assert slot_fingerprint(first) == before
    assert next_slot.link_state[0] == 0


def test_output_arrays_are_frozen_and_own_their_bytes():
    values = np.ones((56, 2), dtype=np.float32)
    valid = np.ones((56, 2), dtype=np.bool_)
    feature = SlotFeatures(values, valid)
    values.fill(7)
    valid.fill(False)
    assert np.all(feature.raw_x == 1)
    assert feature.feature_valid.all()
    grid = builder()
    grid.ingest(observation(1, 0))
    slot = grid.commit_through(BASE)[0]
    window = grid.window_at(BASE)
    for array in (
        feature.raw_x,
        feature.feature_valid,
        slot.raw_x,
        slot.feature_valid,
        slot.age_ns,
        slot.link_state,
        slot.observed_mask,
        window.raw_x,
        window.feature_valid,
        window.grid_ns,
        window.age_ns,
    ):
        with pytest.raises(ValueError):
            array.flags.writeable = True
    with pytest.raises(FrozenInstanceError):
        slot.grid_ns = 0
    with pytest.raises(FrozenInstanceError):
        window.target_ns = 0


def test_query_requires_committed_aligned_target_and_preserved_history():
    grid = builder()
    with pytest.raises(ContractError):
        grid.window_at(BASE)
    grid.commit_through(BASE + 990 * MS)
    with pytest.raises(ContractError):
        grid.window_at(BASE + 990 * MS + 1)
    with pytest.raises(ContractError):
        grid.window_at(BASE + 1000 * MS)
    grid.commit_through(BASE + 3000 * MS)
    with pytest.raises(ContractError):
        grid.window_at(BASE + 990 * MS)
    validate_window(grid.window_at(BASE + 3000 * MS))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"step_ns": 0},
        {"step_ns": True},
        {"step_ns": 1.5},
        {"output_step_ns": 0},
        {"settle_ns": -1},
        {"max_hold_age_ns": -1},
    ],
)
def test_policy_rejects_noninteger_or_invalid_durations(kwargs):
    with pytest.raises(ContractError):
        GridPolicy(**kwargs)


def test_grid_policy_must_match_data_spec_step():
    with pytest.raises(ContractError):
        builder(policy=GridPolicy(step_ns=20 * MS, output_step_ns=40 * MS), spec=DataSpec())


def test_policy_from_config_converts_fractional_milliseconds_to_integer_nanoseconds(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("{}\n", encoding="utf-8")
    config = load_config(
        path,
        overrides={
            "input": {
                "grid_step_ms": 10.0,
                "output_period_ms": 50.0,
                "settle_ms": 0.5,
                "max_hold_age_ms": 30.25,
            },
        },
    )
    policy = GridPolicy.from_config(config)
    assert policy.step_ns == 10_000_000
    assert policy.output_step_ns == 50_000_000
    assert policy.settle_ns == 500_000
    assert policy.max_hold_age_ns == 30_250_000
    assert all(
        isinstance(value, int)
        for value in (
            policy.step_ns,
            policy.output_step_ns,
            policy.settle_ns,
            policy.max_hold_age_ns,
        )
    )
    grid = CausalGridBuilder.from_config(
        config,
        start=Stamp(BASE, "grid", "virtual"),
        feature_builder=FixtureFeatures(),
        provenance_id="fixture-only",
        availability_basis="synthetic",
    )
    grid.ingest(observation(1, 0))
    assert grid.commit_through(BASE + 499_999) == []
    slot = grid.commit_through(BASE + 500_000)[0]
    assert slot.deadline_ns == BASE + 500_000
    assert slot.link_state[0] == 1


@pytest.mark.parametrize("field", ["settle_ms", "max_hold_age_ms"])
def test_config_rejects_sub_nanosecond_grid_durations(tmp_path, field):
    path = tmp_path / "config.yaml"
    path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ConfigError, match=field):
        load_config(path, overrides={"input": {field: 0.0000001}})


def test_config_rejects_output_cadence_that_is_not_a_grid_multiple(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(path, overrides={"input": {"output_period_ms": 15}})


def test_watermark_cannot_move_backward():
    grid = builder()
    grid.commit_through(BASE + MS)
    with pytest.raises(ContractError):
        grid.commit_through(BASE)


def test_zero_hold_age_admits_only_exact_grid_acquisitions():
    grid = builder(policy=GridPolicy(max_hold_age_ns=0))
    grid.ingest(observation(1, 0))
    grid.ingest(observation(2, 19))
    slots = grid.commit_through(BASE + 20 * MS)
    assert [int(slot.link_state[0]) for slot in slots] == [1, 0, 0]


def test_clock_domain_and_availability_basis_cannot_be_silently_mixed():
    grid = builder()
    sample = observation(1, 0)
    with pytest.raises(ContractError):
        grid.ingest(
            replace(sample, clock_domain="other", clock_reference=Stamp(BASE, "other", "virtual"))
        )
    with pytest.raises(ContractError):
        grid.ingest(replace(sample, availability_basis="legacy_bridge_proxy"))


def test_real_epoch_and_clock_pipeline_preserves_reorder_and_rx_reset_scope():
    rx, tx, mapper = RxTimerUnwrapper(), TxSequenceEpochs(), OnlineClockMapper(synthetic=True)
    grid = builder()
    frames = [
        replace(observation(100, 0, link=0).frame, rx_timer_u32_us=100_000),
        replace(observation(100, 0, link=3).frame, rx_timer_u32_us=500_000),
        replace(observation(102, 20).frame, rx_timer_u32_us=120_000),
        replace(observation(101, 10, 30).frame, rx_timer_u32_us=110_000),
        replace(observation(103, 40).frame, rx_timer_u32_us=500, rx_boot_u8=2),
    ]
    timed = []
    for frame in frames:
        result = mapper.observe(frame, tx.classify(frame, rx.update(frame)))
        timed.append(result)
        grid.ingest(result)
    assert timed[3].epochs.classification == "reorder"
    assert timed[3].acquisition_ns == BASE + 10 * MS
    assert timed[4].epochs.classification == "rx_reset"
    slots = grid.commit_through(BASE + 40 * MS)
    assert slots[3].observation_ids[0] == timed[2].observation_id
    assert slots[3].link_state[0] == 2
    assert slots[3].link_state[3] == 2
    assert slots[4].observation_ids[0] == timed[4].observation_id
    assert not slots[4].feature_valid[0, :, 1].any()
