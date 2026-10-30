from dataclasses import replace
import json

import pytest

from csi_mp.viz.schema import DASHBOARD_SCHEMA_ID, DashboardFrame, DisplayPerson, to_dashboard_frame


def context():
    return dict(
        schema_id=DASHBOARD_SCHEMA_ID,
        run_id="r",
        session_id="s",
        stream_epoch="seek0",
        event_sequence=1,
        mode="demo",
        coordinate_spec={"dimension": 2, "spec_id": "camera_normalized_v1", "unit": "normalized"},
        source_time_ns="1800000000000000001",
        source_clock_domain="source-wall",
        source_clock_kind="unix_wall",
        emitted_monotonic_ns="234567890123456789",
        emitter_clock_domain="backend-boot1",
        availability_basis="legacy_bridge_proxy",
    )


def person(**overrides):
    data = dict(
        candidate_key="candidate-0",
        track_id=None,
        estimate_kind="observed",
        joints=((0.2, 0.3),) * 18,
        joint_valid=(True,) * 18,
        presence_score=0.8,
        evidence_age_ms=10.0,
    )
    return DisplayPerson(**{**data, **overrides})


def test_unknown_and_observed_empty_survive_json_as_different_states():
    unknown = to_dashboard_frame(None, context())
    empty = to_dashboard_frame(
        dict(status="observed", current_count=0, predicted_count=0, persons=()), context()
    )
    a, b = DashboardFrame.from_json(unknown.to_json()), DashboardFrame.from_json(empty.to_json())
    assert a.status == "unknown" and a.current_count is None and a.predicted_count is None
    assert b.status == "observed" and b.current_count == 0
    assert a.source_time_ns == "1800000000000000001"
    assert a.source_clock_kind == "unix_wall" and a.emitter_clock_domain == "backend-boot1"
    assert isinstance(json.loads(a.to_json())["emitted_monotonic_ns"], str)
    with pytest.raises(ValueError, match="null counts"):
        replace(unknown, current_count=0)


def test_ingest_is_unknown_without_model_output():
    frame = to_dashboard_frame(None, {**context(), "mode": "ingest"})
    assert frame.current_count is None and frame.persons == ()
    with pytest.raises(ValueError, match="ingest mode"):
        replace(frame, status="observed", current_count=0, predicted_count=0)


def test_query_candidates_do_not_become_track_ids():
    candidate = person()
    frame = to_dashboard_frame(
        dict(status="observed", current_count=1, predicted_count=0, persons=(candidate,)), context()
    )
    assert frame.persons[0].track_id is None
    with pytest.raises(ValueError, match="tracking_enabled"):
        replace(frame, persons=(replace(candidate, track_id="0"),))
    tracked = replace(
        frame,
        persons=(replace(candidate, track_id="s:track0:17"),),
        quality={"tracking_enabled": True, "track_epoch": "track0"},
    )
    assert DashboardFrame.from_json(tracked.to_json()).persons[0].track_id == "s:track0:17"


def test_predicted_people_are_not_current_count_and_require_age():
    predicted = person(estimate_kind="predicted", evidence_age_ms=200)
    frame = to_dashboard_frame(
        dict(status="predicted", current_count=None, predicted_count=1, persons=(predicted,)),
        context(),
    )
    assert frame.current_count is None and frame.predicted_count == 1
    with pytest.raises(ValueError, match="null current_count"):
        replace(frame, current_count=1)
    with pytest.raises(ValueError, match="requires evidence_age"):
        person(estimate_kind="predicted", evidence_age_ms=None)


def test_invalid_joint_is_json_null_and_valid_nonfinite_is_rejected():
    p = person(
        joints=((float("nan"), float("nan")),) + ((0.1, 0.2),) * 17,
        joint_valid=(False,) + (True,) * 17,
    )
    assert p.to_dict()["joints"][0] == [None, None]
    assert DisplayPerson.from_dict(p.to_dict()) == p
    with pytest.raises(ValueError, match="finite coordinates"):
        person(joints=((float("nan"), 0.2),) + ((0.1, 0.2),) * 17)


def test_schema_clock_and_d3_calibration_errors_are_explicit():
    frame = to_dashboard_frame(None, context())
    with pytest.raises(ValueError, match="schema_id"):
        replace(frame, schema_id="wrong")
    with pytest.raises(ValueError, match="decimal string"):
        replace(frame, source_time_ns=1800000000000000001)
    with pytest.raises(ValueError, match="calibration_id"):
        replace(
            frame,
            coordinate_spec={
                "dimension": 3,
                "spec_id": "world",
                "unit": "m",
                "axes": ["x", "y", "z"],
                "normalization": "none",
            },
        )
    with pytest.raises(ValueError, match="D2 uses unit"):
        replace(frame, coordinate_spec={"dimension": 2, "unit": "px"})
    with pytest.raises(ValueError, match="D3 uses meters"):
        replace(frame, coordinate_spec={"dimension": 3, "unit": "mm", "axes": ["x", "y", "z"]})
    with pytest.raises(ValueError, match="presence_score"):
        person(presence_score=True)
    with pytest.raises(ValueError, match="source/emitter clock"):
        replace(frame, source_clock_kind="unknown")
