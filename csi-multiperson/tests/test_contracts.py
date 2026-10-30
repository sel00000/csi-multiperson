from dataclasses import replace
import json

import numpy as np
import pytest

from csi_mp.data.contracts import (
    elapsed_ns,
    require_training_capacity,
    validate_compatibility,
    validate_coordinate_spec,
    validate_data_spec,
    validate_labels,
    validate_window,
)
from csi_mp.data.joints import (
    BODY18_NAMES,
    coco17_to_body18,
    denormalize_image_coordinates,
    normalize_image_coordinates,
)
from csi_mp.data.schema import (
    DERIVED_SCHEMA_ID,
    FEATURE_SPEC_ID,
    LABEL_SCHEMA_ID,
    MODEL_SCHEMA_ID,
    RAW_SCHEMA_ID,
    CapacityOverflow,
    ContractError,
    CoordinateSpec,
    DataSpec,
    PoseFrame,
    RawChunk,
    SourceSpan,
    Stamp,
    WindowPacket,
    WireFrame,
)


def make_frame(iq=None):
    stamp = Stamp(100, "fixture:rx0", "virtual")
    span = SourceSpan("0" * 64, "rx0", 0, 18, 130)
    return WireFrame(
        spans=(span,),
        wire_bytes=bytes(130),
        rx_id=0,
        tx_id=0,
        seq_u32=7,
        rx_timer_u32_us=100,
        rx_boot_u8=0,
        iq=np.zeros((56, 2), dtype=np.int8) if iq is None else iq,
        rssi_dbm=-41,
        noise_dbm=-96,
        signal_mode=None,
        bridge_complete=stamp,
        infer_available=None,
        ingest_ordinal=0,
    )


def make_labels(n=2, *, complete=True, scene_status=None):
    scene_status = scene_status or ("persons_labeled" if n else "empty_confirmed")
    if not complete:
        scene_status = "incomplete"
    return PoseFrame(
        frame_idx=3,
        time=Stamp(100, "video:fixture", "virtual"),
        camera_id="cam0",
        joints=np.full((n, 18, 2), 0.5, dtype=np.float32),
        joint_valid=np.ones((n, 18), dtype=np.bool_),
        joint_weight=np.ones((n, 18), dtype=np.float32),
        gt_track_id=np.arange(n, dtype=np.int64),
        annotation_complete=complete,
        scene_status=scene_status,
        coordinate_spec=CoordinateSpec(image_width=640, image_height=480),
    )


def make_window():
    values = np.zeros((100, 9, 56, 2), dtype=np.float32)
    valid = np.zeros(values.shape, dtype=np.bool_)
    observed = np.zeros((100, 9), dtype=np.bool_)
    state = np.zeros((100, 9), dtype=np.uint8)
    age = np.full((100, 9), -1, dtype=np.int64)
    values[-1, 0, :, 0] = 1
    valid[-1, 0, :, 0] = True
    observed[-1, 0] = True
    state[-1, 0] = 1
    age[-1, 0] = 0
    return WindowPacket(
        session_id="synthetic",
        target_ns=990_000_000,
        decision_ns=990_000_000,
        grid_ns=np.arange(100, dtype=np.int64) * 10_000_000,
        raw_x=values,
        feature_valid=valid,
        observed_mask=observed,
        link_state=state,
        age_ns=age,
        schema_id=DERIVED_SCHEMA_ID,
        feature_id=FEATURE_SPEC_ID,
        provenance_id="fixture_v1",
        availability_basis="synthetic",
        status="ready",
    )


def test_schema_families_have_independent_identifiers():
    assert len({RAW_SCHEMA_ID, DERIVED_SCHEMA_ID, LABEL_SCHEMA_ID, MODEL_SCHEMA_ID}) == 4


def test_iq_signed_and_immutable_after_publication():
    iq = np.zeros((56, 2), dtype=np.int8)
    iq[0] = [-128, 127]
    frame = make_frame(iq)
    iq[0] = 0
    np.testing.assert_array_equal(frame.iq[0], [-128, 127])
    with pytest.raises(ValueError):
        frame.iq[0] = 1
    with pytest.raises(ValueError):
        frame.iq.flags.writeable = True
    with pytest.raises(ContractError, match="iq.dtype"):
        make_frame(np.zeros((56, 2), dtype=np.uint8))


def test_iq_pair_shape_cannot_silently_flatten():
    with pytest.raises(ContractError, match="iq.shape"):
        make_frame(np.zeros(112, dtype=np.int8))


def test_stamp_large_integer_json_roundtrip_and_clock_kinds():
    ns = 1_798_761_234_987_654_321
    stamp = Stamp(ns, "bridge:legacy", "unix_wall")
    payload = json.loads(json.dumps(stamp.to_dict()))
    assert payload["ns"] == str(ns)
    assert Stamp.from_dict(payload) == stamp
    with pytest.raises(ContractError, match="decimal string"):
        Stamp.from_dict({**payload, "ns": ns})
    with pytest.raises(ContractError, match="clock_domain"):
        elapsed_ns(Stamp(ns, "infer:host", "monotonic"), stamp)
    with pytest.raises(ContractError, match="clock_kind"):
        elapsed_ns(replace(stamp, ns=ns + 100), stamp)
    assert elapsed_ns(Stamp(30, "fixture", "virtual"), Stamp(10, "fixture", "virtual")) == 20


def test_raw_availability_basis_preserves_legacy_and_measured():
    span = SourceSpan("0" * 64, "rx0", 0, 18, 2)
    chunk = RawChunk(span, b"ab", Stamp(9, "legacy", "unix_wall"), None, 0)
    assert chunk.availability_basis == "legacy_bridge_proxy"
    assert (
        replace(chunk, infer_available=Stamp(20, "host:boot", "monotonic")).availability_basis
        == "measured_infer"
    )
    assert (
        replace(chunk, bridge_read=Stamp(9, "fixture", "virtual")).availability_basis == "synthetic"
    )
    with pytest.raises(ContractError, match="byte_count"):
        replace(chunk, payload=b"abc")
    with pytest.raises(ContractError, match="infer_available.kind"):
        replace(chunk, infer_available=Stamp(20, "wall", "unix_wall"))
    assert (
        replace(chunk, infer_available=Stamp(20, "replay", "virtual")).availability_basis
        == "synthetic"
    )


def test_body18_mapping_preserves_multiple_people_and_neck_mask():
    points = np.zeros((2, 17, 2), dtype=np.float32)
    points[0, 5] = [1, 4]
    points[0, 6] = [3, 8]
    points[1, 5] = [10, 10]
    points[1, 6] = [20, 20]
    valid = np.ones((2, 17), dtype=np.bool_)
    valid[1, 5] = False
    result, mask = coco17_to_body18(points, valid)
    assert result.shape == (2, 18, 2)
    np.testing.assert_array_equal(result[0, 1], [2, 6])
    np.testing.assert_array_equal(result[0, 2], points[0, 6])
    np.testing.assert_array_equal(result[0, 5], points[0, 5])
    assert mask[0, 1]
    assert not mask[1, 1]
    assert np.isnan(result[1, 1]).all()


def test_neck_invalid_when_shoulder_is_nan_even_with_mask_true():
    points = np.zeros((17, 2), dtype=np.float32)
    points[6, 0] = np.nan
    _, valid = coco17_to_body18(points, np.ones(17, dtype=np.bool_))
    assert not valid[1]
    assert not valid[2]


def test_pixel_normalization_roundtrip_uses_width_height_independently():
    pixels = np.array([[320, 120], [640, 480]], dtype=np.float32)
    normalized = normalize_image_coordinates(pixels, image_width=640, image_height=480)
    np.testing.assert_array_equal(normalized, [[0.5, 0.25], [1, 1]])
    restored = denormalize_image_coordinates(normalized, image_width=640, image_height=480)
    np.testing.assert_array_equal(restored, pixels)


def test_data_spec_rejects_joint_order_disagreement():
    wrong_order = (BODY18_NAMES[1], BODY18_NAMES[0], *BODY18_NAMES[2:])
    with pytest.raises(ContractError, match="joint_names"):
        validate_compatibility(DataSpec(), DataSpec(joint_names=wrong_order))


@pytest.mark.parametrize(
    "changes,field",
    [
        ({"D": 4}, "D"),
        ({"T": 0}, "T"),
        ({"patch_size": 3}, "patch_size"),
        ({"Q": 0}, "Q"),
        ({"F": 55}, "F"),
    ],
)
def test_invalid_spec_fields_have_distinct_errors(changes, field):
    with pytest.raises(ContractError, match=field):
        validate_data_spec(replace(DataSpec(), **changes))


def test_d3_requires_calibration_and_gt_source():
    d3 = CoordinateSpec(
        dimension=3,
        spec_id="world_metric_v1",
        unit="m",
        axes=("x_right", "y_forward", "z_up"),
        normalization="none",
        coordinate_system="world",
    )
    with pytest.raises(ContractError, match="calibration_id"):
        validate_coordinate_spec(d3)
    with pytest.raises(ContractError, match="gt_source_id"):
        validate_coordinate_spec(replace(d3, calibration_id="calibration_sha"))
    validate_coordinate_spec(
        replace(d3, calibration_id="calibration_sha", gt_source_id="mocap:fixture")
    )


def test_d2_schema_may_be_ingest_only_but_labels_need_image_size():
    validate_coordinate_spec(CoordinateSpec())
    with pytest.raises(ContractError, match="image_width"):
        validate_labels(replace(make_labels(), coordinate_spec=CoordinateSpec()))


def test_compatibility_checks_feature_and_coordinate_versions():
    with pytest.raises(ContractError, match="feature_id"):
        validate_compatibility(DataSpec(), replace(DataSpec(), feature_id="other.v1"))
    with pytest.raises(ContractError, match="coordinate_spec"):
        validate_compatibility(
            DataSpec(), replace(DataSpec(), coordinate_spec=CoordinateSpec(spec_id="other.v1"))
        )


def test_same_unsupported_schema_cannot_bypass_compatibility():
    unsupported = replace(DataSpec(), derived_schema_id="csi_mp.window.v99")
    with pytest.raises(ContractError, match="derived_schema_id"):
        validate_compatibility(unsupported, unsupported)


def test_label_capacity_is_reported_without_losing_people():
    labels = make_labels(7)
    result = validate_labels(labels, capacity=6)
    assert result.capacity_status == "capacity_overflow"
    assert result.person_count == 7
    assert labels.joints.shape[0] == 7
    with pytest.raises(CapacityOverflow, match="N=7 exceeds Q=6") as error:
        require_training_capacity(labels, capacity=6)
    assert not isinstance(error.value, ContractError)
    assert labels.joints.shape[0] == 7


def test_empty_complete_and_incomplete_labels_are_distinct():
    empty = make_labels(0)
    unknown = make_labels(0, complete=False)
    assert validate_labels(empty).annotation_complete
    assert not validate_labels(unknown).annotation_complete
    with pytest.raises(ContractError, match="empty_confirmed"):
        validate_labels(replace(unknown, scene_status="empty_confirmed"))


def test_missing_label_joint_preserves_nan_and_person_row():
    labels = make_labels(2, complete=False)
    points = labels.joints.copy()
    valid = labels.joint_valid.copy()
    points[1, 0] = np.nan
    valid[1, 0] = False
    partial = replace(labels, joints=points, joint_valid=valid)
    assert validate_labels(partial).person_count == 2
    assert np.isnan(partial.joints[1, 0]).all()
    with pytest.raises(ContractError, match="finite coordinates"):
        validate_labels(replace(partial, joint_valid=labels.joint_valid))


def test_valid_window_and_defensive_copy():
    window = make_window()
    validate_window(window)
    assert not window.raw_x.flags.writeable
    with pytest.raises(ValueError):
        window.grid_ns[0] = -1


def test_window_rejects_wrong_grid_and_schema():
    window = make_window()
    with pytest.raises(ContractError, match="grid_ns"):
        validate_window(replace(window, grid_ns=window.grid_ns + 1))
    with pytest.raises(ContractError, match="schema_id"):
        validate_window(replace(window, schema_id=LABEL_SCHEMA_ID))
    with pytest.raises(ContractError, match="target_ns"):
        validate_window(replace(window, target_ns=float(window.target_ns)))


def test_window_masks_do_not_turn_missing_or_held_values_into_observations():
    window = make_window()
    state = window.link_state.copy()
    state[-1, 0] = 2
    with pytest.raises(ContractError, match="observed_mask"):
        validate_window(replace(window, link_state=state))
    observed = window.observed_mask.copy()
    observed[-1, 0] = False
    held = replace(window, link_state=state, observed_mask=observed)
    validate_window(held)
    valid = held.feature_valid.copy()
    valid[-1, 0, :, 1] = True
    with pytest.raises(ContractError, match="held delta"):
        validate_window(replace(held, feature_valid=valid))
    values = window.raw_x.copy()
    values[0, 1, 0, 0] = 9
    with pytest.raises(ContractError, match="invalid features"):
        validate_window(replace(window, raw_x=values))
