"""Versioned metric catalogue; definitions do not imply provider availability."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable

from .schema import MetricSpec, MetricsSnapshot, MetricValue

INGEST_V1_KEYS = (
    "source.bytes_received_count",
    "source.raw_record_count",
    "source.valid_frame_count",
    "source.crc_error_count",
    "source.semantic_reject_count",
    "source.discarded_bytes_count",
    "source.truncated_tail_bytes",
    "link.rx_unique_count",
    "link.duplicate_count",
    "link.reorder_count",
    "link.epoch_uncertain_count",
    "link.epoch_excluded_frame_count",
    "epoch.event_count",
    "link.rssi_dbm",
    "link.noise_dbm",
)
UNITS = frozenset(
    {
        "byte",
        "record",
        "frame",
        "candidate",
        "segment",
        "event",
        "dBm",
        "packet/s",
        "packet",
        "ratio",
        "ms",
        "ns",
        "Hz",
        "tick",
        "person",
        "person/frame",
        "px",
        "mm",
        "s",
        "session",
        "GPU·h",
        "%",
        "score",
        "track",
    }
)


def content_hash(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class MetricRegistry:
    def __init__(self, specs: Iterable[MetricSpec] = ()) -> None:
        self._specs: dict[str, MetricSpec] = {}
        for spec in specs:
            self.register(spec)

    def register(self, spec: MetricSpec) -> None:
        if spec.key in self._specs:
            raise ValueError(f"metric {spec.key!r}: already registered")
        if spec.unit not in UNITS:
            raise ValueError(f"metric {spec.key!r}: unknown unit {spec.unit!r}")
        self._specs[spec.key] = spec

    def __contains__(self, key: str) -> bool:
        return key in self._specs

    def __getitem__(self, key: str) -> MetricSpec:
        try:
            return self._specs[key]
        except KeyError as exc:
            raise ValueError(f"unknown metric ID: {key}") from exc

    def __iter__(self):
        return iter(sorted(self._specs))

    @property
    def spec_hash(self) -> str:
        return content_hash([self._specs[key].to_dict() for key in self])

    def validate_selection(self, keys: Iterable[str]) -> tuple[str, ...]:
        selected = tuple(keys)
        if len(set(selected)) != len(selected):
            raise ValueError("measurement.include: duplicate metric ID")
        for key in selected:
            self[key]
        return selected

    def validate_request(
        self,
        key: str,
        *,
        unit: str | None = None,
        reduction: str | None = None,
        aggregation_scope: str | None = None,
    ) -> None:
        spec = self[key]
        if unit is not None and unit != spec.unit:
            raise ValueError(f"{key}.unit: expected {spec.unit!r}, got {unit!r}")
        if reduction is not None and reduction not in spec.reductions:
            raise ValueError(f"{key}.reduction: unsupported {reduction!r}")
        if aggregation_scope is not None and aggregation_scope not in spec.aggregation_scopes:
            raise ValueError(f"{key}.aggregation_scope: unsupported {aggregation_scope!r}")

    def validate_value(self, value: MetricValue) -> None:
        self.validate_request(
            value.key, reduction=value.reduction, aggregation_scope=value.aggregation_scope
        )
        spec = self[value.key]
        if not set(value.labels) <= set(spec.labels):
            raise ValueError(
                f"{value.key}.labels: unknown labels {set(value.labels) - set(spec.labels)}"
            )
        if value.value is not None and set(value.labels) != set(spec.labels):
            raise ValueError(f"{value.key}.labels: measured value requires {spec.labels}")

    def validate_snapshot(self, snapshot: MetricsSnapshot) -> None:
        if snapshot.spec_hash != self.spec_hash:
            raise ValueError("MetricsSnapshot.spec_hash: registry mismatch")
        for value in snapshot.values:
            self.validate_value(value)


def validate_selection(
    keys: Iterable[str], *, registry: MetricRegistry | None = None
) -> tuple[str, ...]:
    return (registry or default_registry()).validate_selection(keys)


def availability_reason(
    *,
    selected: bool,
    supported: bool = True,
    requires_gt: bool = False,
    has_gt: bool = False,
    provider_connected: bool = False,
    sample_count: int | None = None,
    min_samples: int = 1,
    denominator: float | int | None = None,
) -> str | None:
    """Resolve unavailable states in the order fixed by measurement-spec v0.2."""
    if not selected:
        return "disabled"
    if not supported:
        return "unsupported"
    if requires_gt and not has_gt:
        return "no_gt"
    if not provider_connected:
        return "not_measured"
    if (sample_count is not None and sample_count < min_samples) or denominator == 0:
        return "no_samples"
    return None


def default_registry() -> MetricRegistry:
    specs: list[MetricSpec] = []

    def add(
        key: str,
        unit: str,
        labels: tuple[str, ...] = (),
        *,
        group: str = "core",
        reductions: tuple[str, ...] = ("sum",),
        scopes: tuple[str, ...] = ("cumulative", "window"),
        prerequisites: tuple[str, ...] = (),
        clock: str = "availability_same_domain",
    ) -> None:
        quantiles = set(reductions) & {"p5", "p50", "p95"}
        specs.append(
            MetricSpec(
                key=key,
                version=1,
                unit=unit,
                group=group,
                labels=labels,
                calculation_id=f"{key}:v1",
                clock_requirement=clock,
                prerequisites=prerequisites,
                default_window_s=None if group == "offline" else 10.0,
                default_publish_period_s=None if group == "offline" else 1.0,
                reductions=reductions,
                aggregation_scopes=scopes,
                quantile_method="nearest_rank" if quantiles else None,
                min_samples={q: 1 if q == "p50" else 20 for q in sorted(quantiles)},
            )
        )

    source = ("source_id",)
    link = ("rx_id", "tx_id")
    for name, unit in (
        ("bytes_received_count", "byte"),
        ("raw_record_count", "record"),
        ("valid_frame_count", "frame"),
        ("crc_error_count", "candidate"),
    ):
        add(f"source.{name}", unit, source)
    add("source.semantic_reject_count", "frame", source + ("reason",))
    add("source.discarded_bytes_count", "byte", source + ("reason",))
    add("source.truncated_tail_bytes", "byte", source + ("layer",))
    for name, unit in (
        ("rx_unique_count", "frame"),
        ("duplicate_count", "frame"),
        ("reorder_count", "frame"),
        ("epoch_uncertain_count", "segment"),
    ):
        add(f"link.{name}", unit, link)
    add("link.epoch_excluded_frame_count", "frame", link + ("reason",))
    add("epoch.event_count", "event", ("scope", "device_id", "kind", "certainty"))
    for key in ("link.rssi_dbm", "link.noise_dbm"):
        add(
            key,
            "dBm",
            link,
            group="diagnostic",
            reductions=("p5", "p50", "p95"),
            scopes=("window",),
        )
    add(
        "link.rx_unique_pps",
        "packet/s",
        link,
        reductions=("rate",),
        scopes=("window",),
        prerequisites=("capture_active_exposure", "epoch_provider"),
    )
    add(
        "link.seq_gap_estimate",
        "packet",
        link,
        scopes=("window",),
        prerequisites=("mature_seq_intervals",),
    )
    add(
        "link.seq_gap_ratio_estimate",
        "ratio",
        link,
        reductions=("ratio",),
        scopes=("window",),
        prerequisites=("mature_seq_intervals",),
    )
    for key in (
        "grid.observed_ratio",
        "grid.held_ratio",
        "grid.amp_coverage_ratio",
        "grid.delta_coverage_ratio",
        "link.late_admission_ratio",
    ):
        add(
            key,
            "ratio",
            link,
            reductions=("ratio",),
            scopes=("window",),
            prerequisites=("grid_provider",),
        )
    add("grid.measurement_age_ms", "ms", link, reductions=("p50", "p95"), scopes=("window",))
    add("source.reconnect_count", "event", source, prerequisites=("source_provider",))
    add("source.last_valid_frame_age_ms", "ms", source, reductions=("last",), scopes=("instant",))
    add("runtime.input_queue_depth", "event", reductions=("last", "max"), scopes=("window",))
    add("runtime.input_drop_count", "event", ("reason",))
    add("runtime.skipped_tick_count", "tick", ("reason",))
    for key in (
        "latency.queue_wait_ms",
        "latency.model_ready_ms",
        "latency.target_to_publish_ms",
        "latency.bridge_to_infer_ms",
        "latency.window_build_ms",
        "latency.model_gpu_ms",
        "latency.tracker_ms",
        "latency.render_submit_ms",
        "latency.target_to_render_submit_ms",
    ):
        add(
            key,
            "ms",
            reductions=("p50", "p95"),
            scopes=("window",),
            prerequisites=("latency_provider",),
            clock="monotonic_or_verified_transform",
        )
    add(
        "output.usable_tick_ratio",
        "ratio",
        ("population",),
        reductions=("ratio",),
        scopes=("window",),
    )
    add(
        "output.publish_rate_hz", "Hz", ("estimate_kind",), reductions=("rate",), scopes=("window",)
    )
    add("output.tick_status_count", "tick", ("status",))
    add(
        "scene.estimated_person_count",
        "person",
        ("basis",),
        reductions=("last",),
        scopes=("instant",),
    )
    add("resource.process_rss_bytes", "byte", reductions=("last", "max"), scopes=("window",))
    for key, unit in (
        ("eval.count_mae", "person"),
        ("eval.exact_count_ratio", "ratio"),
        ("eval.person_precision", "ratio"),
        ("eval.person_recall", "ratio"),
        ("eval.false_positive_per_frame", "person/frame"),
        ("eval.pck", "ratio"),
        ("eval.matched_joint_error_px", "px"),
        ("eval.idf1", "ratio"),
        ("eval.mpjpe_world_mm", "mm"),
    ):
        labels = (
            ("alpha", "normalizer", "joint_spec", "matching_version", "population")
            if key == "eval.pck"
            else ()
        )
        add(
            key,
            unit,
            labels,
            group="offline",
            reductions=("ratio",) if unit == "ratio" else ("mean",),
            scopes=("evaluation",),
            prerequisites=("ground_truth", "evaluation_provider"),
            clock="evaluation_protocol",
        )
    for key, unit in (
        ("data.labelled_clip_seconds", "s"),
        ("data.labelled_sessions_count", "session"),
        ("data.unlabelled_pretrain_seconds", "s"),
        ("train.allocated_gpu_hours", "GPU·h"),
    ):
        add(
            key,
            unit,
            group="offline",
            scopes=("evaluation",),
            prerequisites=("run_manifest",),
            clock="manifest",
        )
    return MetricRegistry(specs)
