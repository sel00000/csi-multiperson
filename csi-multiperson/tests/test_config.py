from pathlib import Path

import pytest
import yaml

from csi_mp.config import ConfigError, load_config, selected_metrics, validate_config
from csi_mp.data.joints import BODY18_NAMES
from csi_mp.telemetry.registry import INGEST_V1_KEYS


def write_config(path: Path, value: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
    return path


def source(path="capture.rawlog", **changes):
    return {
        "source_id": "rx0",
        "rx_id": 0,
        "path": path,
        "clock_domain": "bridge:legacy",
        "clock_kind": "unix_wall",
        **changes,
    }


def test_default_config_is_ingest_only_and_preserves_spec_types(tmp_path):
    cfg = load_config(write_config(tmp_path / "capture.yaml", {}))
    validate_config(cfg)
    assert cfg.model_spec.D == 2
    assert cfg.model_spec.joint_names == BODY18_NAMES
    assert cfg.model_spec.coordinate_spec.axes == ("x_right", "y_down")
    assert cfg.model_spec.coordinate_spec.image_width is None
    assert selected_metrics(cfg) == INGEST_V1_KEYS
    assert cfg.as_dict()["measurement"]["offline_include"] == []
    assert cfg.as_dict()["acceptance"]["manifest"] is None
    assert cfg.as_dict()["capture"]["sources"] == []
    assert cfg.as_dict()["clock"] == {
        "policy_id": "clock_fixture_v1",
        "window_size": 64,
        "warmup_samples": 3,
        "min_span_us": 10_000,
        "max_drift_ppm": 1000,
        "legacy_delay_ns": 0,
    }


@pytest.mark.parametrize(
    "field,value",
    [
        ("D", 1),
        ("D", True),
        ("T", 0),
        ("T", 100.0),
        ("patch_size", 0),
        ("patch_size", 3),
        ("Q", 0),
        ("Q", -1),
    ],
)
def test_invalid_model_values_report_field(tmp_path, field, value):
    path = write_config(tmp_path / "bad.yaml", {"model": {field: value}})
    with pytest.raises(ConfigError, match=field):
        load_config(path)


@pytest.mark.parametrize(
    "value,match",
    [
        ({"unknown": 1}, "unknown fields"),
        ({"model": {"query_slots": 6}}, "query_slots"),
        ({"model": {"coordinate_spec": {"coordinate_typo": 1}}}, "coordinate_typo"),
        ({"input": {"grid_ms_typo": 1}}, "grid_ms_typo"),
        ({"clock": {"future_fit": True}}, "future_fit"),
    ],
)
def test_unknown_keys_rejected_at_nested_boundaries(tmp_path, value, match):
    with pytest.raises(ConfigError, match=match):
        load_config(write_config(tmp_path / "bad.yaml", value))


def test_duplicate_yaml_keys_do_not_silently_override(tmp_path):
    path = tmp_path / "duplicate.yaml"
    path.write_text("model:\n  Q: 6\n  Q: 8\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="duplicate YAML key: Q"):
        load_config(path)


@pytest.mark.parametrize(
    "field,value",
    [
        ("policy_id", ""),
        ("window_size", 0),
        ("warmup_samples", True),
        ("min_span_us", -1),
        ("max_drift_ppm", -1),
        ("legacy_delay_ns", -1),
    ],
)
def test_invalid_clock_policy_is_a_named_config_error(tmp_path, field, value):
    with pytest.raises(ConfigError, match=field):
        load_config(write_config(tmp_path / "clock.yaml", {"clock": {field: value}}))


def test_clock_policy_overrides_are_recorded_in_config_hash(tmp_path):
    path = write_config(tmp_path / "clock.yaml", {})
    baseline = load_config(path)
    changed = load_config(path, overrides={"clock": {"legacy_delay_ns": 2_000_000}})
    assert changed.as_dict()["clock"]["legacy_delay_ns"] == 2_000_000
    assert changed.config_hash != baseline.config_hash


def test_includes_precedence_then_explicit_override(tmp_path):
    write_config(tmp_path / "base.yaml", {"model": {"Q": 8, "d": 64}})
    write_config(tmp_path / "second.yaml", {"model": {"Q": 10}})
    path = write_config(
        tmp_path / "capture.yaml",
        {
            "includes": ["base.yaml", "second.yaml"],
            "model": {"Q": 7},
        },
    )
    cfg = load_config(path, overrides={"model": {"Q": 9}})
    assert cfg.model_spec.Q == 9
    assert cfg.as_dict()["model"]["d"] == 64


def test_include_cycle_identifies_culprit(tmp_path):
    a = write_config(tmp_path / "a.yaml", {"includes": ["b.yaml"]})
    write_config(tmp_path / "b.yaml", {"includes": ["a.yaml"]})
    with pytest.raises(ConfigError, match="includes: cycle"):
        load_config(a)


def test_include_missing_file_reports_path(tmp_path):
    path = write_config(tmp_path / "capture.yaml", {"includes": ["missing.yaml"]})
    with pytest.raises(ConfigError, match="missing.yaml"):
        load_config(path)


def test_relative_paths_belong_to_declaring_file_not_cwd(tmp_path, monkeypatch):
    included = tmp_path / "defaults" / "capture.yaml"
    write_config(
        included,
        {
            "capture": {"sources": [source("../captures/rx0.rawlog")]},
            "output_dir": "../outputs",
        },
    )
    root = write_config(tmp_path / "app" / "main.yaml", {"includes": ["../defaults/capture.yaml"]})
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    values = load_config(root).as_dict()
    assert values["capture"]["sources"][0]["path"] == str(tmp_path / "captures" / "rx0.rawlog")
    assert values["output_dir"] == str(tmp_path / "outputs")


def test_relative_explicit_override_paths_use_main_config_directory(tmp_path, monkeypatch):
    path = write_config(tmp_path / "app" / "capture.yaml", {})
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    cfg = load_config(
        path,
        overrides={
            "capture": {"sources": [source("../data/rx0.rawlog")]},
            "output_dir": "results",
        },
    )
    assert cfg.as_dict()["output_dir"] == str(path.parent / "results")
    assert cfg.as_dict()["capture"]["sources"][0]["path"] == str(tmp_path / "data" / "rx0.rawlog")


def test_configuration_hash_is_order_independent_but_content_sensitive(tmp_path):
    a = write_config(tmp_path / "a.yaml", {"run_id": "fixture", "model": {"Q": 6, "d": 128}})
    b = write_config(tmp_path / "b.yaml", {"model": {"d": 128, "Q": 6}, "run_id": "fixture"})
    first, second = load_config(a), load_config(b)
    assert first.config_hash == second.config_hash
    assert len(first.config_hash) == 64
    assert load_config(b, overrides={"model": {"Q": 7}}).config_hash != first.config_hash


def test_configuration_access_and_input_mutation_cannot_change_hash(tmp_path):
    overrides = {"model": {"Q": 7}}
    cfg = load_config(write_config(tmp_path / "capture.yaml", {}), overrides=overrides)
    digest = cfg.config_hash
    values = cfg.as_dict()
    values["model"]["Q"] = 100
    values["measurement"]["include"].clear()
    overrides["model"]["Q"] = 10
    assert cfg.model_spec.Q == 7
    assert cfg.config_hash == digest
    assert selected_metrics(cfg) == INGEST_V1_KEYS


@pytest.mark.parametrize(
    "selected,match",
    [
        (["missing.metric"], "unknown metric ID"),
        (["link.rssi_dbm", "link.rssi_dbm"], "duplicate metric"),
        ([{}], "measurement.include"),
        ([7], "measurement.include"),
    ],
)
def test_invalid_metric_selection_is_config_error(tmp_path, selected, match):
    path = write_config(tmp_path / "bad.yaml", {"measurement": {"include": selected}})
    with pytest.raises(ConfigError, match=match):
        load_config(path)


def test_custom_metric_selection_is_exact_and_hidden_panel_has_no_effect(tmp_path):
    cfg = load_config(
        write_config(
            tmp_path / "capture.yaml",
            {
                "measurement": {"include": ["link.rssi_dbm"], "offline_include": ["eval.pck"]},
                "viewer": {"show_candidate_keys": False},
            },
        )
    )
    assert selected_metrics(cfg) == ("link.rssi_dbm", "eval.pck")


def test_d3_config_requires_calibration_metadata(tmp_path):
    coordinate = {
        "dimension": 3,
        "spec_id": "world_metric_v1",
        "unit": "m",
        "axes": ["x_right", "y_forward", "z_up"],
        "normalization": "none",
        "coordinate_system": "world",
        "gt_source_id": "mocap:fixture",
    }
    path = write_config(tmp_path / "d3.yaml", {"model": {"D": 3, "coordinate_spec": coordinate}})
    with pytest.raises(ConfigError, match="calibration_id"):
        load_config(path)
    cfg = load_config(
        path, overrides={"model": {"coordinate_spec": {"calibration_id": "calibration:fixture"}}}
    )
    assert cfg.model_spec.D == 3


def test_slots_require_integer_even_if_float_has_equal_value(tmp_path):
    with pytest.raises(ConfigError, match="window_slots"):
        load_config(write_config(tmp_path / "bad.yaml", {"input": {"window_slots": 100.0}}))


def test_source_mapping_requires_trusted_receiver_and_wall_clock(tmp_path):
    with pytest.raises(ConfigError, match="clock_kind"):
        load_config(
            write_config(
                tmp_path / "clock.yaml",
                {
                    "capture": {"sources": [source(clock_kind="monotonic")]},
                },
            )
        )
    with pytest.raises(ConfigError, match="rx_id"):
        load_config(
            write_config(
                tmp_path / "receiver.yaml",
                {
                    "capture": {"sources": [source(rx_id=3)]},
                },
            )
        )
    with pytest.raises(ConfigError, match="duplicate source mapping"):
        load_config(
            write_config(
                tmp_path / "duplicate.yaml",
                {
                    "capture": {
                        "sources": [source(), source("second.rawlog", source_id="rx0_again")]
                    },
                },
            )
        )


@pytest.mark.parametrize(
    "section,value", [("boards", "unknown"), ("metadata", []), ("environment", 0)]
)
def test_capture_metadata_containers_are_validated(tmp_path, section, value):
    with pytest.raises(ConfigError, match=section):
        load_config(write_config(tmp_path / "bad.yaml", {"capture": {section: value}}))


def test_unknown_radio_measurement_is_not_a_fabricated_value(tmp_path):
    path = write_config(
        tmp_path / "bad.yaml",
        {
            "capture": {
                "radio": {
                    "tx_power_dbm": {"value": 20, "provenance": "unknown", "evidence": "not_read"},
                }
            }
        },
    )
    with pytest.raises(ConfigError, match="unknown values must be null"):
        load_config(path)
