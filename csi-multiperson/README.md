# CSI Multiperson — S00–S07 / V00

A package for ESP32 CSI with three transmitters and three receivers (TX3×RX3). The current implementation covers configuration and shared contracts, CSIRAW01/130B parsing, RX/TX epoch detection, causal RX clock mapping, session import without modifying source files, a causal grid with 10 ms slots and 100-slot windows, log-amplitude and delta features with validity masks, and an initial set of 15 metrics and display output contracts.

Validation uses synthetic fixtures. Fitting normalization statistics and applying frozen statistics are planned after the S11 data split. Models, live hardware capture, and the browser interface are also planned for later stages.

Validation results: [S00–S03](reports/s00-s03-validation.json), [S04 clock mapping](reports/s04-validation-v2.json), [S05 import and storage](reports/s05-validation.json), [S06 causal grid/window](reports/s06-validation.json), [S07 feature generation](reports/s07-validation.json).

## Architecture

![Architecture for CSI-based multi-person pose estimation](../docs/multiperson-csi-design/figures/multiperson-csi-architecture.png)

The diagram shows the full proposed architecture. Model training, multi-person pose inference, and hardware validation remain future work. The usage instructions and APIs below describe what is currently implemented.

## Getting started

Run the following commands from `csi-multiperson/`. Python 3.11 or later is required. The setup below has been verified with an editable installation in a `.venv` that shares existing NumPy, PyYAML, and pytest installations.

```bash
python -m venv --system-site-packages .venv
.venv/bin/python -m pip install --no-build-isolation --no-deps -e .
.venv/bin/python -m csi_mp --help
.venv/bin/python -m csi_mp validate-config --config configs/capture.yaml
.venv/bin/python -m csi_mp inspect --config configs/capture.yaml
.venv/bin/python -m csi_mp inspect --config configs/capture.yaml --save
```

For a fresh environment without NumPy, PyYAML, or pytest, create a standard virtual environment and install with `.venv/bin/python -m pip install -e '.[dev]'`. Optional dependencies are grouped separately under `dataset`, `teacher`, `train`, and `live`. Help commands and ingestion checks do not load Torch, cameras, or teacher models.

By default, `inspect` reads the source files and writes a JSON summary to stdout. Use `--save` to write reports under the configured `output_dir/run_id`, or `--output PATH` to use a new directory of your choice. Existing destinations are rejected. If the run fails, the new output directory retains `failure.json` and any diagnostics collected so far. The reference directories `examples/` and `ex)robot/` cannot be used as output destinations.

Reports include `manifest.json`, `config.resolved.json`, `frames.jsonl`, `issues.jsonl`, `epochs.jsonl`, `metrics.jsonl`, the final `metrics.json`, and `dashboard.json`. Each frame record preserves its original wire representation, signed IQ values, source byte span, completion time, epoch, usability status, and `timing`.

The `timing` record contains immutable acquisition and availability timestamps, a shared clock domain and reference point, the model ID, fit residuals, warmup status, and exclusion reasons. The manifest also preserves the clock policy, final snapshots, counts by status, and hashes of the sources, configuration, implementation, metric definitions, providers, and selection lists.

Setting `measurement.sinks: []` disables periodic writes to `metrics.jsonl`. The final `metrics.json` and `dashboard.json` are still saved as contract snapshots for reviewing the run. The only streaming sink currently supported is `jsonl`; dashboard streaming is planned for V01 or later. A completion manifest is published only after all writers have closed and the final files have been written successfully.

## Reproducible session import

```bash
.venv/bin/python -m csi_mp import --config configs/capture.yaml --output runs/ingest-v1.csisession
```

`import` creates a new `.csisession` SQLite file containing every source byte and its SHA256 hash, along with observations, epochs, timestamps, diagnostics, and control evidence. Junk bytes and incomplete trailing data are preserved too. The manifest records configuration, implementation, and record hashes; the verification status of I/Q and SC ordering; and clock policies and assumptions.

Importing the same sources with the same configuration reproduces the observations and provenance. In the current validation environment, it also produces identical file bytes. The importer does not yet generate grids, features, or T100 windows.

The raw import path reuses S02–S04. Legacy HDF5 import requires the optional `h5py` dependency and an explicit format in `capture.sources`, as shown below. Paths are relative to the YAML file.

```yaml
capture:
  session_id: legacy-session
  sources:
    - source_id: legacy
      path: ../data/recording.h5
      format: legacy_hdf5
      clock_domain: bridge:recording-host
```

`rx_id` is read from HDF5 `/links/{rx}{tx}`. Omit `clock_domain` if it is unknown; `t_ns` retains its original Unix wall-clock value. Missing fields are stored as null with an explanation. The importer does not invent CRC values, original wire data, confirmed epochs, or acquisition and availability timestamps absent from `/links`. Preserved rows are not marked as usable causal inputs.

Files containing only `/samples/X`, invalid shapes, dtypes, or row counts, and HDF5 files that depend on external files are rejected. Raw data and `/links` data must be imported into separate sessions.

After closing the temporary file, the importer validates the schema, counts, and hashes and checks that the sources are unchanged. It then publishes the final path atomically using a hard link on the same filesystem. It never overwrites an existing destination or another writer's output.

Failed or interrupted imports leave a record in the sibling path `.filename.importing/state.json`, which blocks retries. Before retrying, check whether the import is still running, inspect that record, and move the staging directory aside or remove it. Successful imports clean up their staging directories.

The Python API is `import_session(source, destination, spec)`. Pass the `AppConfig` returned by `load_config(...)` as `spec`; `source=None` selects all sources in the configuration. Stored sessions can be read without access to the original source paths.

```python
from csi_mp.data.store import SessionStore

with SessionStore.open_readonly("runs/ingest-v1.csisession") as session:
    manifest = session.manifest
    for observation in session.observations():
        timing = observation["timing"]
    # session.epochs(), session.issues(), session.records("controls")
    # session.source_bytes("rx0") returns the original bytes in chunks.
```

`open_readonly` checks the completion manifest and the stored data's schema, counts, and hashes, then opens the session read-only. `MigrationReport.accepted_frames` counts preserved observations with a valid format; `usable_frames` separately counts those usable under the epoch and clock rules. The default raw fixture has 13 preserved observations, 10 usable observations, and 2 with unresolved epochs.

## Causal grid/window API

`csi_mp.data.grid.CausalGridBuilder` consumes S04 `TimedObservation` objects and commits each slot exactly once, at 10 ms intervals by default. Use it in this order: `ingest(observation)` → `commit_through(deadline_ns)` → `window_at(target_ns)`.

Construction requires the session, the first grid `Stamp`, provenance, an availability basis, and a `FeatureBuilder`. `from_config(config, ...)` uses the existing model/input configuration.

- The default window shape is `[100,9,56,2]`. Its status remains `warming_up` until 100 actual slots exist.
- Both acquisition and arrival deadlines are checked. Late packets may qualify for later slots, but committed slots never change.
- `link_state` is 1 for observed, 2 for held, and 0 for unavailable. Values may be held for up to 30 ms by default; held deltas are zero and invalid. Missing data has all features set to zero, masks set to false, and age set to −1.
- Epoch boundaries, clock boundaries, and long gaps interrupt both holding and differencing. Missing data on every link currently yields `insufficient_coverage`; it does not indicate an empty room.
- Only the latest 100 slots are retained, and returned objects are immutable. Inject S07's `csi_mp.data.features.FeatureBuilder` to compute log-amplitude and temporal delta features before normalization. The S06 test transform is for fixtures only.

See the [implementation](src/csi_mp/data/grid.py), [tests](tests/test_causal_window.py), and [S06 validation results](reports/s06-validation.json) for the API, late-packet counters, and retention rules. The importer still stores only raw sessions; it does not run grid or feature generation automatically.

## Log-amplitude and delta feature API

`FeatureBuilder` produces channels in the order `amp`, `delta_amp`. The amplitude formula, `amp=log1p(hypot(I,Q))`, uses the natural logarithm with a fixed +1 and no additional epsilon. The delta is the difference in log-amplitude between the current and previous actual observations, divided by the elapsed acquisition time in seconds.

To preserve precision for large absolute timestamps, integer nanosecond values are subtracted first. Calculations then use float64, and results are stored as float32.

Zero I/Q is a valid amplitude of zero. Deltas are zero with a false mask for the first observation, held values, resets, long gaps, and equal or out-of-order timestamps. Equal and out-of-order observations do not update the baseline used for subsequent deltas.

A delta is valid only within the same link, RX/TX epochs, link and clock segments, and clock domain/kind, with `0 < dt_ns <= max_delta_gap_ns`. The default maximum gap is 30 ms; `FeatureBuilder.from_config(config)` uses the grid's `max_hold_age_ms` setting.

```python
from csi_mp.config import load_config
from csi_mp.data.features import FeatureBuilder
from csi_mp.data.grid import CausalGridBuilder
from csi_mp.data.schema import Stamp

config = load_config("configs/capture.yaml")
features = FeatureBuilder.from_config(config)
grid = CausalGridBuilder.from_config(
    config,
    start=Stamp(0, "example-replay", "virtual"),
    feature_builder=features,
    provenance_id="example-replay",
    availability_basis="synthetic",
)
feature_manifest = features.spec.to_dict()
feature_spec_hash = features.spec.spec_hash
# Pass TimedObservation objects with the same clock and availability basis to grid.ingest(...).
# After grid.commit_through(...), grid.window_at(...) exposes unnormalized features in raw_x.
```

`FeatureSpec` is immutable. Its manifest and hash identify the `logamp_delta.v1` formula, channels, masks, and gap policy. Derived artifacts record that manifest and hash alongside the grid policy in their provenance.

The provider retains only a snapshot of the latest processed observation for each link. Query it immediately after `on_observation`, as the grid does; requesting an older ID raises an error. Returned `SlotFeatures` remain unchanged by later observations or resets.

`fit_stats` and `FrozenFeatureNormalizer` are not yet available. They are planned after the S11 training manifest is finalized.

## Fixtures and validation

```bash
.venv/bin/python scripts/generate_fixtures.py
.venv/bin/python -m pytest -q
.venv/bin/python -m ruff check src tests scripts
```

[fixtures/ingest_v1/expected.json](fixtures/ingest_v1/expected.json) records the seed, SHA256 hashes of both input files, the scenario for each record, and expected counts constructed independently of the parser. The generator reproduces identical bytes.

| Item | Expected value |
|---|---:|
| Complete raw records / bytes fed to the parser | 16 / 1,990B |
| Frames passing CRC and semantic checks | 13 |
| Unique usable frames / duplicates / frames excluded for uncertain epochs | 10 / 1 / 2 |
| Unique frames received out of order | 1 |
| Candidates failing CRC / semantic rejections | 1 / 1 |
| Junk bytes / CRC resynchronization bytes | 23B / 130B |
| Incomplete trailing raw record / wire frame | 15B / 17B |
| Inferred TX restarts / confirmed RX restarts | 1 / 1 |

CRC candidate counts and discarded byte counts have different units, so they are not combined into a loss rate. The reader does not yield the incomplete raw tail, so those bytes are not included in the parser's input count. When RX0 and RX1 receive the same TX sequence number, each reception is a separate unique observation.

## Configuration and interpretation

- YAML settings are applied in this order: `includes` entries in list order, the current file, then `load_config(..., overrides=...)`. Nested mappings are merged; lists are replaced. Relative paths are resolved against the YAML file that declares them, while explicit overrides are resolved against the main configuration file. CLI `--output` paths are relative to the working directory. Environment variables do not change the configuration.
- Unknown configuration fields and metric IDs, invalid D/T/patch/Q values, and unsupported schema or coordinate contracts are rejected. The configuration hash covers the resolved content, including defaults and paths. Model Q does not limit the number of people stored in labels.
- The initial selection is the 15 metrics in `custom/ingest_v1`. Status is serialized as `disabled` for unselected metrics, `not_measured` for unconnected metrics, `no_gt` when ground truth is absent, `no_samples` when there are insufficient samples, and `unsupported` for format or clock limitations. Cumulative values, recent-window values, and p5/p50/p95 use separate series.
- Quantiles use the nearest-rank method. p50 requires at least 1 unique observation; p5 and p95 require at least 20. Missing values are not filled with zero. Display output reports ingestion status as `unknown`, the person count as `null`, and persons as `[]`.
- RX boot epochs and TX sequence epochs are independent. A single low sequence number is excluded; the default policy requires 3 observations showing sequence progression to infer a reset with status `inferred`. Control evidence from `tx_start`, `tx_reboot`, or a TX-scoped `operator_boundary` yields `confirmed`. The current default thresholds, `epochs_fixture_v1`, have not been validated on hardware in the field.
- Legacy timestamps use `unix_wall`. Synthetic fixtures are identified separately through provenance. Inference arrival times are not fabricated, and timestamps from different clocks are not subtracted directly. Multiple raw files can be inspected together only when they explicitly share the same clock domain. File order and diagnostics for backward timestamp jumps are preserved.
- S04's `OnlineClockMapper` updates a bounded affine fit for each RX epoch in arrival order. Previously emitted timestamps and models are immutable; reordered observations use the existing fit. Different host clocks require an explicit `ClockTransform` through the API. The legacy CLI does not correct them automatically.
- `clock_fixture_v1` is a development policy using the latest 64 observations, a warmup of 3 observations, a minimum span of 10 ms, and a maximum drift of 1000 ppm. Provisional estimates are emitted even during `warming_up`, so consumers must check `clock_status`. `clock_error_ns` represents the bridge residual and known transformation error, not an upper bound on actual acquisition error. Constant communication delay cannot be identified.
- When actual inference arrival timestamps are available, the availability basis is `measured_infer`. Otherwise, it is `legacy_bridge_proxy`, using the bridge timestamp plus `clock.legacy_delay_ns` (0 by default). `synthetic` denotes a separate synthetic basis. A backward host-clock jump excludes the affected frame and starts a new clock segment with the next frame; timestamps are not adjusted by +1 ns.
- Metric snapshots are generated at fixed periodic boundaries in source time. Intervals without input are coalesced into the latest boundary, and the manifest records the number of skipped boundaries. An EOF summary is also added. This file-replay cadence does not represent a live output rate in Hz.

## Next steps and limitations

S08, the multi-person teacher, is next. Fitting normalization statistics and applying frozen statistics will follow the S11 data split. The grid metric provider and measurements of actual exposure time and mature sequence ranges still need to be connected. pps and sequence-gap metrics have definitions but are disabled in the initial selection.

RAW dumps (0xC51F), live arrival capture, hardware threshold validation, model training, accuracy and latency measurements, and the V01 browser interface are not yet supported.

The legacy wire format has no TX boot identifier and only an 8-bit RX boot identifier, so the epoch of every delayed packet cannot be determined conclusively. Retransmissions of previously observed old wire frames are isolated, but old packets that were never observed remain a limitation. Forward clock jumps cannot be distinguished from genuine long capture gaps without external evidence. Acceptance criteria remain `not_evaluated`.
