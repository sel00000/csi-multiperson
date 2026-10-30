"""Strict YAML configuration: includes < declaring file < explicit CLI overrides.

Paths are resolved against the file that declares them, never the working
directory. Environment variables do not silently alter a run. The content hash
uses the fully resolved, default-expanded configuration and excludes YAML order.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml


class ConfigError(ValueError):
    pass


class _UniqueLoader(yaml.SafeLoader):
    pass


def _mapping(loader, node, deep=False):
    loader.flatten_mapping(node)
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if not isinstance(key, str):
            raise ConfigError("YAML mapping keys must be strings")
        if key in result:
            raise ConfigError(f"duplicate YAML key: {key}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _mapping)


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def content_hash(value):
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def _merge(base, overlay):
    result = copy.deepcopy(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _resolve(value, directory):
    if not isinstance(value, str) or not value:
        raise ConfigError("path: expected a nonempty string")
    return str((directory / value).resolve())


def _resolve_paths(value, directory):
    value = copy.deepcopy(value)
    capture = value.get("capture", {})
    if isinstance(capture, dict) and isinstance(capture.get("sources", []), list):
        for source in capture.get("sources", []):
            if isinstance(source, dict) and "path" in source:
                source["path"] = _resolve(source["path"], directory)
    if "output_dir" in value:
        value["output_dir"] = _resolve(value["output_dir"], directory)
    acceptance = value.get("acceptance", {})
    if isinstance(acceptance, dict) and acceptance.get("manifest") is not None:
        acceptance["manifest"] = _resolve(acceptance["manifest"], directory)
    return value


def _read(path, stack):
    path = path.resolve()
    if path in stack:
        raise ConfigError(f"includes: cycle at {path}")
    try:
        value = yaml.load(path.read_text(encoding="utf-8"), Loader=_UniqueLoader)
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(f"config {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ConfigError(f"config {path}: expected a mapping")
    includes = value.pop("includes", [])
    if not isinstance(includes, list) or not all(isinstance(p, str) for p in includes):
        raise ConfigError("includes: expected a list of paths")
    result = {}
    for include in includes:
        result = _merge(result, _read(path.parent / include, (*stack, path)))
    return _merge(result, _resolve_paths(value, path.parent))


@dataclass(frozen=True)
class AppConfig:
    path: Path
    _canonical_json: str

    def as_dict(self):
        return json.loads(self._canonical_json)

    @property
    def config_hash(self):
        return hashlib.sha256(self._canonical_json.encode()).hexdigest()

    @property
    def model_spec(self):
        from .data.schema import CoordinateSpec, DataSpec

        model = self.as_dict()["model"]
        model.pop("d")
        model["coordinate_spec"] = CoordinateSpec.from_dict(model["coordinate_spec"])
        return DataSpec(**model)


def _keys(value, allowed, field):
    if not isinstance(value, dict):
        raise ConfigError(f"{field}: expected a mapping")
    unknown = value.keys() - set(allowed)
    if unknown:
        raise ConfigError(f"{field}: unknown fields {sorted(unknown)}")


def _positive(value, field, *, zero=False, integer=False):
    types = (int,) if integer else (int, float)
    try:
        valid = not isinstance(value, bool) and isinstance(value, types) and math.isfinite(value)
    except OverflowError:
        valid = False
    if not valid:
        raise ConfigError(f"{field}: expected {'integer' if integer else 'finite number'}")
    if value < 0 or (not zero and value == 0):
        raise ConfigError(f"{field}: expected {'nonnegative' if zero else 'positive'} value")


def _text(value, field):
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{field}: expected nonempty string")


def _defaults(directory):
    from .data.schema import DataSpec
    from .ingest.clock import ClockPolicy
    from .ingest.epochs import EpochPolicy
    from .telemetry.registry import INGEST_V1_KEYS

    return {
        "run_id": "inspect",
        "output_dir": str(directory / "runs"),
        "model": {**asdict(DataSpec()), "d": 128},
        "input": {
            "grid_step_ms": 10,
            "window_slots": 100,
            "output_period_ms": 50,
            "settle_ms": 0,
            "max_hold_age_ms": 30,
            "seq_reorder_grace_ms": None,
            "verification": "fixture_only",
        },
        "capture": {
            "session_id": "unassigned",
            "sources": [],
            "controls": [],
            "boards": [],
            "radio": {},
            "transport": {},
            "signal": {},
            "environment": {},
            "metadata": {},
        },
        "epoch": asdict(EpochPolicy()),
        "clock": asdict(ClockPolicy()),
        "measurement": {
            "spec_version": 2,
            "profile": "custom",
            "selection_id": "ingest_v1",
            "window_s": 10,
            "publish_period_s": 1,
            "include": list(INGEST_V1_KEYS),
            "exclude": [],
            "offline_include": [],
            "percentiles": [5, 50, 95],
            "quantile_method": "nearest_rank",
            "quantile_min_samples": {"p5": 20, "p50": 1, "p95": 20},
            "warmup_policy": "report_separately",
            "sinks": ["jsonl"],
            "missing_value": None,
            "require_reason_when_missing": True,
        },
        "viewer": {
            "schema_id": "csi_mp.dashboard.v1",
            "history_s": 60,
            "metric_snapshot_period_s": 1,
            "mode": "ingest",
            "show_candidate_keys": True,
            "show_track_ids": False,
            "acceptance_status": "not_evaluated",
        },
        "acceptance": {"manifest": None},
    }


def load_config(path, *, overrides=None) -> AppConfig:
    path = Path(path).resolve()
    raw = _read(path, ())
    if overrides:
        raw = _merge(raw, _resolve_paths(overrides, path.parent))
    value = _merge(_defaults(path.parent), raw)
    try:
        cfg = AppConfig(path, canonical_json(value))
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"config: values must be finite JSON data: {exc}") from exc
    validate_config(cfg)
    return cfg


def selected_metrics(cfg):
    from .telemetry.registry import default_registry, validate_selection

    m = cfg.as_dict()["measurement"]
    registry = default_registry()
    # core/research derive definitions from registry; custom uses exactly include.
    if m["profile"] == "custom":
        base = m["include"]
    else:
        groups = {"core"} if m["profile"] == "core" else {"core", "diagnostic"}
        base = [key for key in registry if registry[key].group in groups]
        base = list(dict.fromkeys([*base, *m["include"]]))
    return validate_selection(
        list(dict.fromkeys([k for k in base if k not in m["exclude"]] + m["offline_include"]))
    )


def validate_config(cfg):
    from .data.contracts import validate_data_spec
    from .ingest.clock import ClockPolicy
    from .ingest.epochs import EpochPolicy
    from .telemetry.registry import validate_selection

    value = cfg.as_dict()
    defaults = _defaults(cfg.path.parent)
    _keys(value, defaults, "config")
    for section in (
        "model",
        "input",
        "capture",
        "epoch",
        "clock",
        "measurement",
        "viewer",
        "acceptance",
    ):
        _keys(value[section], defaults[section], section)
    _text(value["run_id"], "run_id")
    if Path(value["run_id"]).name != value["run_id"] or value["run_id"] in {".", ".."}:
        raise ConfigError("run_id: expected a single path-safe identifier")
    _text(value["output_dir"], "output_dir")
    try:
        validate_data_spec(cfg.model_spec)
        EpochPolicy(**value["epoch"])
        ClockPolicy(**value["clock"])
    except (TypeError, ValueError) as exc:
        raise ConfigError(str(exc)) from exc
    _positive(value["model"]["d"], "model.d", integer=True)
    input_cfg = value["input"]
    for key in ("grid_step_ms", "window_slots", "output_period_ms", "settle_ms", "max_hold_age_ms"):
        _positive(
            input_cfg[key],
            f"input.{key}",
            zero=key in {"settle_ms", "max_hold_age_ms"},
            integer=key == "window_slots",
        )
    if input_cfg["window_slots"] != value["model"]["T"]:
        raise ConfigError("input.window_slots: must equal model.T")
    if input_cfg["grid_step_ms"] * 1_000_000 != value["model"]["grid_step_ns"]:
        raise ConfigError("input.grid_step_ms: must match model.grid_step_ns")
    from .data.grid import GridPolicy

    try:
        GridPolicy.from_config(cfg)
    except (TypeError, ValueError) as exc:
        raise ConfigError(str(exc)) from exc
    if input_cfg["seq_reorder_grace_ms"] is not None:
        _positive(input_cfg["seq_reorder_grace_ms"], "input.seq_reorder_grace_ms", zero=True)
    if input_cfg["verification"] not in {"fixture_only", "configured", "measured"}:
        raise ConfigError("input.verification: expected fixture_only/configured/measured")
    capture = value["capture"]
    _text(capture["session_id"], "capture.session_id")
    for field, expected_type in (("boards", list), ("environment", dict), ("metadata", dict)):
        if not isinstance(capture[field], expected_type):
            raise ConfigError(f"capture.{field}: expected {expected_type.__name__}")
    if not isinstance(capture["sources"], list):
        raise ConfigError("capture.sources: expected a list")
    ids, paths, receivers = set(), set(), set()
    for i, source in enumerate(capture["sources"]):
        prefix = f"capture.sources[{i}]"
        _keys(
            source,
            {
                "source_id",
                "path",
                "rx_id",
                "clock_domain",
                "clock_kind",
                "sha256",
                "provenance",
                "raw_schema_id",
                "wire_schema_id",
                "format",
            },
            prefix,
        )
        for field in ("source_id", "path"):
            _text(source.get(field), f"{prefix}.{field}")
        source_format = source.get("format", "csiraw01")
        if source_format not in {"csiraw01", "legacy_hdf5"}:
            raise ConfigError(f"{prefix}.format: expected csiraw01 or legacy_hdf5")
        if source_format == "csiraw01":
            _text(source.get("clock_domain"), f"{prefix}.clock_domain")
            if type(source.get("rx_id")) is not int or source["rx_id"] not in range(3):
                raise ConfigError(f"{prefix}.rx_id: expected 0, 1 or 2")
            if source.get("clock_kind", "unix_wall") != "unix_wall":
                raise ConfigError(f"{prefix}.clock_kind: CSIRAW01 requires unix_wall")
            if (
                source.get("raw_schema_id", "CSIRAW01") != "CSIRAW01"
                or source.get("wire_schema_id", "csi130.v1") != "csi130.v1"
            ):
                raise ConfigError(f"{prefix}: unsupported raw/wire schema")
            if source["rx_id"] in receivers:
                raise ConfigError(f"{prefix}.rx_id: duplicate source mapping")
            receivers.add(source["rx_id"])
        else:
            if source.get("rx_id") is not None:
                raise ConfigError(f"{prefix}.rx_id: /links identifies receivers per group")
            if source.get("clock_domain") is not None:
                _text(source["clock_domain"], f"{prefix}.clock_domain")
            if source.get("clock_kind", "unix_wall") != "unix_wall":
                raise ConfigError(f"{prefix}.clock_kind: legacy t_ns requires unix_wall")
            if source.get("raw_schema_id", "legacy.links.v1") != "legacy.links.v1":
                raise ConfigError(f"{prefix}.raw_schema_id: expected legacy.links.v1")
            if source.get("wire_schema_id") is not None:
                raise ConfigError(f"{prefix}.wire_schema_id: original wire absent in /links")
        for item, seen, field in (
            (source["source_id"], ids, "source_id"),
            (source["path"], paths, "path"),
        ):
            if item in seen:
                raise ConfigError(f"{prefix}.{field}: duplicate source mapping")
            seen.add(item)
        digest = source.get("sha256")
        if digest is not None and (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(c not in "0123456789abcdef" for c in digest)
        ):
            raise ConfigError(f"{prefix}.sha256: expected lowercase SHA256 or null")
    if not isinstance(capture["controls"], list):
        raise ConfigError("capture.controls: expected a list")
    for i, evidence in enumerate(capture["controls"]):
        prefix = f"capture.controls[{i}]"
        _keys(
            evidence,
            {"kind", "device_id", "available_ns", "clock_domain", "clock_kind", "source_ref"},
            prefix,
        )
        if evidence.get("kind") not in {"tx_start", "tx_reboot", "operator_boundary"}:
            raise ConfigError(
                f"{prefix}.kind: expected TX-scoped tx_start/tx_reboot/operator_boundary"
            )
        if type(evidence.get("device_id")) is not int or evidence["device_id"] not in range(3):
            raise ConfigError(f"{prefix}.device_id: expected 0, 1 or 2")
        stamp = evidence.get("available_ns")
        if not isinstance(stamp, str) or not stamp.isdecimal():
            raise ConfigError(f"{prefix}.available_ns: expected decimal string")
        for field in ("clock_domain", "source_ref"):
            _text(evidence.get(field), f"{prefix}.{field}")
        if evidence.get("clock_kind") != "unix_wall":
            raise ConfigError(f"{prefix}.clock_kind: raw inspect control must use unix_wall")
    for section in ("radio", "transport", "signal"):
        if not isinstance(capture[section], dict):
            raise ConfigError(f"capture.{section}: expected mapping")
        for key, parameter in capture[section].items():
            field = f"capture.{section}.{key}"
            _keys(parameter, {"value", "provenance", "evidence"}, field)
            if "value" not in parameter or parameter.get("provenance") not in {
                "configured",
                "device_reported",
                "measured",
                "unknown",
            }:
                raise ConfigError(
                    f"{field}: requires value and configured/device_reported/measured/unknown provenance"
                )
            if parameter["provenance"] == "unknown" and parameter["value"] is not None:
                raise ConfigError(f"{field}: unknown values must be null")
            _text(parameter.get("evidence"), f"{field}.evidence")
    m = value["measurement"]
    if m["profile"] not in {"core", "research", "custom"}:
        raise ConfigError("measurement.profile: expected core/research/custom")
    _text(m["selection_id"], "measurement.selection_id")
    for field in ("include", "exclude", "offline_include"):
        if not isinstance(m[field], list) or not all(isinstance(key, str) for key in m[field]):
            raise ConfigError(f"measurement.{field}: expected list of metric IDs")
        try:
            validate_selection(m[field])
        except (ValueError, TypeError) as exc:
            raise ConfigError(f"measurement.{field}: {exc}") from exc
    for field in ("window_s", "publish_period_s"):
        _positive(m[field], f"measurement.{field}")
        if m[field] * 1_000_000_000 < 1:
            raise ConfigError(f"measurement.{field}: minimum supported duration is 1 ns")
    for field in (
        "spec_version",
        "percentiles",
        "quantile_method",
        "quantile_min_samples",
        "warmup_policy",
        "missing_value",
        "require_reason_when_missing",
    ):
        if m[field] != defaults["measurement"][field]:
            raise ConfigError(f"measurement.{field}: unsupported by metric spec v2")
    if m["sinks"] not in ([], ["jsonl"]):
        raise ConfigError(
            "measurement.sinks: S00–S04 supports [] or [jsonl]; dashboard transport needs V01"
        )
    viewer = value["viewer"]
    for field in ("history_s", "metric_snapshot_period_s"):
        _positive(viewer[field], f"viewer.{field}")
    for field in ("schema_id", "mode", "acceptance_status"):
        if viewer[field] != defaults["viewer"][field]:
            raise ConfigError(f"viewer.{field}: unsupported for ingest-only V00")
    for field in ("show_candidate_keys", "show_track_ids"):
        if type(viewer[field]) is not bool:
            raise ConfigError(f"viewer.{field}: expected boolean")
    if value["acceptance"]["manifest"] is not None:
        raise ConfigError("acceptance.manifest: S00–S04 does not evaluate acceptance gates")
