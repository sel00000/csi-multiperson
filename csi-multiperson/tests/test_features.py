"""S07 numerical features and causal grid parity with the production provider."""

from dataclasses import FrozenInstanceError, replace
import math

import numpy as np
import pytest

from csi_mp.config import load_config
from csi_mp.data.contracts import validate_window
from csi_mp.data.features import FeatureBuilder, FeatureSpec
from csi_mp.data.grid import GridPolicy
from csi_mp.data.schema import FEATURE_SPEC_ID, ContractError, Stamp

from test_causal_window import BASE, MS, builder, observation, slot_fingerprint


def sample(number, time_ns, available_ns=None, *, pair=(3, 4), **kwargs):
    """Construct coherent synthetic observations without rounding absolute ns."""
    if available_ns is None:
        available_ns = time_ns
    item = observation(number, 0, **kwargs)
    frame = replace(
        item.frame,
        iq=np.tile(np.asarray(pair, dtype=np.int8), (56, 1)),
        rx_timer_u32_us=1_000_000 + time_ns // 1_000,
        bridge_complete=Stamp(BASE + available_ns, "grid", "virtual"),
    )
    model = (
        None
        if item.clock_model is None
        else replace(
            item.clock_model,
            reference_ns=BASE + time_ns,
            reference_timer_us=frame.rx_timer_u32_us,
        )
    )
    return replace(
        item,
        frame=frame,
        clock_model=model,
        acquisition_ns=None if not item.usable else BASE + time_ns,
        availability_ns=BASE + available_ns,
    )


def take(features, item):
    features.on_observation(item)
    return features.values_for_slot(item.observation_id, True)


def production_grid(policy=None):
    policy = policy or GridPolicy()
    return builder(
        features=FeatureBuilder(FeatureSpec(max_delta_gap_ns=policy.max_hold_age_ns)),
        policy=policy,
    )


def test_log_amplitude_zero_and_signed_int8_extremes_are_valid_finite_values():
    pairs = [(0, 0), (3, 4), (-3, -4), (-128, 0), (-128, -128), (127, 127)]
    iq = np.asarray([pairs[index % len(pairs)] for index in range(56)], dtype=np.int8)
    item = sample(1, 0)
    item = replace(item, frame=replace(item.frame, iq=iq))
    result = take(FeatureBuilder(), item)
    expected = np.asarray([math.log1p(math.hypot(int(i), int(q))) for i, q in iq], dtype=np.float32)
    np.testing.assert_array_equal(result.raw_x[:, 0], expected)
    assert result.raw_x.shape == result.feature_valid.shape == (56, 2)
    assert result.raw_x.dtype == np.float32
    assert result.feature_valid.dtype == np.bool_
    assert np.isfinite(result.raw_x).all()
    assert result.feature_valid[:, 0].all()
    assert not result.feature_valid[:, 1].any()
    assert np.all(result.raw_x[:, 1] == 0)
    assert result.raw_x[0, 0] == 0 and result.feature_valid[0, 0]


def test_delta_uses_actual_observation_seconds_and_preserves_static_amplitude():
    features = FeatureBuilder()
    take(features, sample(1, 0, pair=(3, 4)))
    moving = take(features, sample(2, 7 * MS, pair=(0, 12)))
    stationary = take(features, sample(3, 20 * MS, pair=(0, 12)))
    expected = np.float32((math.log(13) - math.log(6)) / 0.007)
    np.testing.assert_allclose(moving.raw_x[:, 1], expected, rtol=1e-7)
    assert moving.feature_valid.all()
    assert stationary.feature_valid.all()
    assert np.all(stationary.raw_x[:, 0] == np.float32(math.log(13)))
    assert np.all(stationary.raw_x[:, 1] == 0)


def test_zero_iq_is_observed_signal_in_both_sides_of_a_difference():
    features = FeatureBuilder()
    take(features, sample(1, 0, pair=(0, 0)))
    rising = take(features, sample(2, 10 * MS, pair=(3, 4)))
    falling = take(features, sample(3, 20 * MS, pair=(0, 0)))
    expected = np.float32(math.log(6) / 0.01)
    assert rising.feature_valid.all() and falling.feature_valid.all()
    np.testing.assert_allclose(rising.raw_x[:, 1], expected, rtol=1e-7)
    np.testing.assert_allclose(falling.raw_x[:, 1], -expected, rtol=1e-7)
    assert np.all(falling.raw_x[:, 0] == 0)


def test_one_nanosecond_difference_survives_large_absolute_timestamps():
    features = FeatureBuilder()
    take(features, sample(1, 0, pair=(3, 4)))
    result = take(features, sample(2, 1, pair=(0, 12)))
    expected = np.float32((math.log(13) - math.log(6)) * 1e9)
    assert result.feature_valid[:, 1].all()
    np.testing.assert_allclose(result.raw_x[:, 1], expected, rtol=1e-7)


def test_delta_is_computed_before_amplitudes_are_rounded_for_storage():
    features = FeatureBuilder()
    take(features, sample(1, 0, pair=(127, 127)))
    result = take(features, sample(2, 1, pair=(127, 126)))
    expected = np.float32((math.log1p(math.sqrt(32005)) - math.log1p(math.sqrt(32258))) * 1e9)
    np.testing.assert_allclose(result.raw_x[:, 1], expected, rtol=1e-7)
    assert result.feature_valid[:, 1].all()


@pytest.mark.parametrize(
    "gap_ns,valid", [(30 * MS - 1, True), (30 * MS, True), (30 * MS + 1, False)]
)
def test_delta_gap_limit_is_inclusive_and_recovery_uses_post_gap_observation(gap_ns, valid):
    features = FeatureBuilder()
    take(features, sample(1, 0, pair=(3, 4)))
    result = take(features, sample(2, gap_ns, pair=(0, 12)))
    assert result.feature_valid[:, 0].all()
    assert np.all(result.feature_valid[:, 1] == valid)
    if not valid:
        assert np.all(result.raw_x[:, 1] == 0)
    recovered = take(features, sample(3, gap_ns + 10 * MS, pair=(0, 12)))
    assert recovered.feature_valid[:, 1].all()
    assert np.all(recovered.raw_x[:, 1] == 0)


def test_zero_max_gap_disables_delta_without_disabling_amplitude():
    features = FeatureBuilder(FeatureSpec(max_delta_gap_ns=0))
    take(features, sample(1, 0))
    result = take(features, sample(2, 1))
    assert result.feature_valid[:, 0].all()
    assert not result.feature_valid[:, 1].any()
    assert np.all(result.raw_x[:, 1] == 0)


@pytest.mark.parametrize("time_ns", [10 * MS, 20 * MS])
def test_reordered_and_equal_time_observations_do_not_replace_forward_baseline(time_ns):
    features = FeatureBuilder()
    take(features, sample(1, 0, pair=(0, 0)))
    forward = take(features, sample(2, 20 * MS, pair=(3, 4)))
    delayed = take(features, sample(3, time_ns, 21 * MS, pair=(100, 100)))
    assert delayed.feature_valid[:, 0].all()
    assert not delayed.feature_valid[:, 1].any()
    assert np.all(delayed.raw_x[:, 1] == 0)
    resumed = take(features, sample(4, 30 * MS, pair=(0, 12)))
    expected = np.float32((math.log(13) - math.log(6)) / 0.01)
    np.testing.assert_allclose(resumed.raw_x[:, 1], expected, rtol=1e-7)
    assert resumed.feature_valid[:, 1].all()
    assert forward.feature_valid[:, 1].all()


@pytest.mark.parametrize(
    "boundary", ["link", "rx", "tx", "clock_segment", "clock_domain", "clock_kind"]
)
def test_standalone_builder_cuts_differences_at_each_epoch_or_clock_boundary(boundary):
    features = FeatureBuilder()
    take(features, sample(1, 0))
    changed = sample(2, 10 * MS, pair=(0, 12))
    if boundary == "link":
        changed = replace(changed, epochs=replace(changed.epochs, link_segment="new-link-segment"))
    elif boundary in {"rx", "tx"}:
        changed = replace(
            changed, epochs=replace(changed.epochs, **{f"{boundary}_epoch": "new-epoch"})
        )
    elif boundary == "clock_segment":
        changed = replace(changed, clock_model=replace(changed.clock_model, segment_id="new-clock"))
    else:
        domain = "new-domain" if boundary == "clock_domain" else changed.clock_domain
        kind = "monotonic" if boundary == "clock_kind" else changed.clock_kind
        changed = replace(
            changed,
            clock_domain=domain,
            clock_kind=kind,
            clock_reference=Stamp(BASE, domain, kind),
        )
    result = take(features, changed)
    assert result.feature_valid[:, 0].all()
    assert not result.feature_valid[:, 1].any()
    assert np.all(result.raw_x[:, 1] == 0)


def test_links_have_independent_baselines_and_explicit_reset_affects_only_requested_links():
    features = FeatureBuilder()
    for link in range(9):
        take(features, sample(link + 1, 0, link=link, pair=(3, 4)))
    before = take(features, sample(10, 10 * MS, link=0, pair=(0, 12)))
    previous_bytes = before.raw_x.tobytes(), before.feature_valid.tobytes()
    features.reset((0, 3, 6), "tx_restart")
    for link in range(9):
        result = take(features, sample(link + 11, 20 * MS, link=link, pair=(0, 12)))
        expected_valid = link not in {0, 3, 6}
        assert np.all(result.feature_valid[:, 1] == expected_valid)
        if expected_valid:
            expected = np.float32((math.log(13) - math.log(6)) / 0.02)
            np.testing.assert_allclose(result.raw_x[:, 1], expected, rtol=1e-7)
        else:
            assert np.all(result.raw_x[:, 1] == 0)
    assert (before.raw_x.tobytes(), before.feature_valid.tobytes()) == previous_bytes


def test_held_lookup_and_later_reset_cannot_mutate_published_features():
    features = FeatureBuilder()
    take(features, sample(1, 0))
    latest = sample(2, 10 * MS, pair=(0, 12))
    fresh = take(features, latest)
    fingerprint = fresh.raw_x.tobytes(), fresh.feature_valid.tobytes()
    held = features.values_for_slot(latest.observation_id, False)
    np.testing.assert_array_equal(held.raw_x[:, 0], fresh.raw_x[:, 0])
    assert held.feature_valid[:, 0].all()
    assert not held.feature_valid[:, 1].any()
    assert np.all(held.raw_x[:, 1] == 0)
    assert features.values_for_slot(latest.observation_id, True).feature_valid[:, 1].all()
    features.reset((0,), "manual_boundary")
    take(features, sample(3, 20 * MS))
    assert (fresh.raw_x.tobytes(), fresh.feature_valid.tobytes()) == fingerprint
    for result in (fresh, held):
        with pytest.raises(ValueError):
            result.raw_x.setflags(write=True)
        with pytest.raises(ValueError):
            result.feature_valid.setflags(write=True)


def test_cache_retains_latest_snapshot_per_link_and_rejects_stale_or_unknown_ids():
    features = FeatureBuilder()
    items = [sample(link + 1, 0, link=link) for link in range(9)]
    snapshots = [take(features, item) for item in items]
    newer = sample(10, 10 * MS, link=0, pair=(0, 12))
    take(features, newer)
    with pytest.raises(ContractError):
        features.values_for_slot(items[0].observation_id, True)
    with pytest.raises(ContractError):
        features.values_for_slot("not-observed", True)
    for item, expected in zip(items[1:], snapshots[1:]):
        actual = features.values_for_slot(item.observation_id, True)
        np.testing.assert_array_equal(actual.raw_x, expected.raw_x)
        np.testing.assert_array_equal(actual.feature_valid, expected.feature_valid)


def test_excluded_observation_is_rejected_when_provider_is_used_directly():
    features = FeatureBuilder()
    excluded = sample(1, 0, usable=False, classification="ambiguous")
    with pytest.raises(ContractError):
        features.on_observation(excluded)


def test_spec_is_immutable_and_hash_tracks_gap_policy_without_exposing_mutable_state():
    original = FeatureSpec()
    assert original.feature_id == FEATURE_SPEC_ID
    assert original.max_delta_gap_ns == 30 * MS
    assert original.spec_hash == FeatureSpec().spec_hash
    assert len(original.spec_hash) == 64
    assert original.spec_hash != FeatureSpec(max_delta_gap_ns=0).spec_hash
    with pytest.raises(FrozenInstanceError):
        original.max_delta_gap_ns = 1
    description = original.to_dict()
    description.clear()
    assert original.to_dict()
    assert original.spec_hash == FeatureSpec().spec_hash


@pytest.mark.parametrize("value", [-1, True, 0.5])
def test_invalid_delta_gap_policy_is_rejected(value):
    with pytest.raises(ContractError):
        FeatureSpec(max_delta_gap_ns=value)


def test_builder_configuration_uses_the_same_nanosecond_gap_limit_as_grid(tmp_path):
    path = tmp_path / "features.yaml"
    path.write_text("input:\n  max_hold_age_ms: 7.5\n", encoding="utf-8")
    config = load_config(path)
    features = FeatureBuilder.from_config(config)
    policy = GridPolicy.from_config(config)
    assert features.feature_id == FEATURE_SPEC_ID
    assert features.spec.max_delta_gap_ns == policy.max_hold_age_ns == 7_500_000


def test_grid_rejects_provider_with_an_incompatible_declared_feature_version():
    class IncompatibleFeatures(FeatureBuilder):
        feature_id = "logamp_delta.v999"

    with pytest.raises(ContractError, match="feature_id"):
        builder(features=IncompatibleFeatures())


def test_grid_held_signal_and_all_missing_windows_preserve_validity_distinctions():
    grid = production_grid()
    grid.ingest(sample(1, 0, pair=(0, 0)))
    grid.ingest(sample(2, 10 * MS, pair=(3, 4)))
    slots = grid.commit_through(BASE + 50 * MS)
    assert slots[0].feature_valid[0, :, 0].all()
    assert np.all(slots[0].raw_x[0, :, 0] == 0)
    assert slots[1].feature_valid[0].all()
    for slot in slots[2:5]:
        assert slot.link_state[0] == 2
        assert slot.feature_valid[0, :, 0].all()
        assert not slot.feature_valid[0, :, 1].any()
        assert np.all(slot.raw_x[0, :, 1] == 0)
    assert not slots[5].feature_valid.any()
    assert np.all(slots[5].raw_x == 0)
    assert np.all(slots[5].age_ns == -1)
    missing = production_grid()
    missing.commit_through(BASE + 990 * MS)
    packet = missing.window_at(BASE + 990 * MS)
    validate_window(packet)
    assert packet.status == "insufficient_coverage"
    assert not packet.feature_valid.any()
    assert np.all(packet.raw_x == 0)


def test_production_features_fill_t100_without_changing_raw_amplitude_scale():
    grid = production_grid()
    for index in range(100):
        grid.ingest(sample(index + 1, index * 10 * MS, pair=(3, 4)))
    grid.commit_through(BASE + 990 * MS)
    packet = grid.window_at(BASE + 990 * MS)
    validate_window(packet)
    assert packet.status == "ready"
    assert packet.raw_x.shape == packet.feature_valid.shape == (100, 9, 56, 2)
    assert packet.feature_id == FEATURE_SPEC_ID
    assert packet.feature_valid[:, 0, :, 0].all()
    assert not packet.feature_valid[0, 0, :, 1].any()
    assert packet.feature_valid[1:, 0, :, 1].all()
    assert np.all(packet.raw_x[:, 0, :, 0] == np.float32(math.log(6)))
    assert np.all(packet.raw_x[:, 0, :, 1] == 0)
    assert not packet.feature_valid[:, 1:].any()


def test_production_features_match_between_batch_and_streamed_arrival_traces():
    trace = [
        sample(1, 0, pair=(0, 0)),
        sample(2, 7 * MS, pair=(3, 4)),
        sample(3, 18 * MS, 24 * MS, pair=(-128, -128)),
        sample(4, 10 * MS, 30 * MS, pair=(100, 0), classification="reorder"),
        sample(5, 35 * MS, pair=(0, 12)),
        sample(6, 45 * MS, pair=(3, 4), segment="b"),
        sample(7, 55 * MS, pair=(0, 12), segment="b", clock_segment="b"),
        sample(8, 150 * MS, pair=(3, 4), segment="b", clock_segment="b"),
        sample(9, 160 * MS, 180 * MS, pair=(0, 12), segment="b", clock_segment="b"),
    ]
    policy = GridPolicy(settle_ns=5 * MS)
    batch, streaming = production_grid(policy), production_grid(policy)
    for item in trace:
        batch.ingest(item)
    expected = batch.commit_through(BASE + 205 * MS)
    streamed = []
    position = 0
    for milliseconds in range(5, 206, 10):
        deadline = BASE + milliseconds * MS
        while position < len(trace) and trace[position].availability_ns <= deadline:
            streaming.ingest(trace[position])
            position += 1
        streamed.extend(streaming.commit_through(deadline))
    assert [slot_fingerprint(slot) for slot in streamed] == [
        slot_fingerprint(slot) for slot in expected
    ]


def test_future_signal_and_resets_do_not_change_features_or_masks_of_published_prefix():
    baseline, with_future = production_grid(), production_grid()
    for item in [sample(1, 0), sample(2, 10 * MS, pair=(0, 12))]:
        baseline.ingest(item)
        with_future.ingest(item)
    with_future.ingest(sample(3, 40 * MS, pair=(-128, -128), segment="future"))
    with_future.invalidate((0,), "future_boundary", available_ns=BASE + 60 * MS)
    expected = baseline.commit_through(BASE + 20 * MS)
    prefix = with_future.commit_through(BASE + 20 * MS)
    fingerprint = [slot_fingerprint(slot) for slot in prefix]
    assert fingerprint == [slot_fingerprint(slot) for slot in expected]
    packet = with_future.window_at(BASE + 20 * MS)
    packet_bytes = packet.raw_x.tobytes(), packet.feature_valid.tobytes()
    with_future.commit_through(BASE + 100 * MS)
    assert [slot_fingerprint(slot) for slot in prefix] == fingerprint
    assert (packet.raw_x.tobytes(), packet.feature_valid.tobytes()) == packet_bytes
