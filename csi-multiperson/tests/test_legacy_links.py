"""Legacy data loss stays explicit when HDF5 rows become session observations."""

from collections import defaultdict
import hashlib
import json

import numpy as np
import pytest

from csi_mp.ingest.legacy_links import import_links


h5py = pytest.importorskip("h5py")
BASE_NS = 1_700_000_000_000_000_123


def write_links(path, *, name="00", count=2, optional=True):
    with h5py.File(path, "w") as h5:
        group = h5.create_group(f"links/{name}")
        group.create_dataset(
            "iq", data=np.arange(count * 112, dtype=np.int64).astype(np.int8).reshape(count, 56, 2)
        )
        if optional:
            group.create_dataset("t_ns", data=np.array([BASE_NS - i for i in range(count)], "u8"))
            group.create_dataset("esp_us", data=np.arange(count, dtype="u8") + 2**32)
            group.create_dataset("seq", data=np.arange(count, dtype="u4"))
            group.create_dataset("rssi", data=np.full(count, -60, dtype="i1"))
            group.create_dataset("noise", data=np.full(count, -90, dtype="i1"))
            group.create_dataset("boot_id", data=np.full(count, 255, dtype="u1"))


def source(path, *, source_id="legacy", domain="test:bridge"):
    return {
        "original_path": str(path),
        "path": str(path),
        "source_id": source_id,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "clock_domain": domain,
        "provenance": "fixture",
    }


def collect(sources):
    rows = defaultdict(list)
    result = import_links(sources, emit=lambda kind, row: rows[kind].append(row))
    json.dumps({"result": result, "rows": rows}, allow_nan=False)
    return result, rows


def test_full_legacy_rows_keep_values_and_unknown_epoch_and_availability(tmp_path):
    path = tmp_path / "legacy.h5"
    write_links(path)
    before = path.read_bytes()
    description = source(path)
    result, rows = collect([description])
    again, repeated = collect([description])
    assert (result, rows) == (again, repeated)
    assert path.read_bytes() == before
    assert result["counts"] == {"stored_frames": 2, "accepted_frames": 2, "usable_frames": 0}
    assert result["unresolved_epochs"] == 2
    assert result["rejected_by_reason"] == {}
    assert not rows["epochs"]
    for index, observation in enumerate(rows["observations"]):
        assert observation["source_ref"] == {
            "source_id": "legacy",
            "source_sha256": description["sha256"],
            "dataset_path": "/links/00",
            "row_index": index,
        }
        assert observation["t_ns"] == str(BASE_NS - index)
        assert observation["esp_us"] == str(2**32 + index)
        assert observation["seq"] == index
        assert observation["boot_id"] == 255
        assert observation["rssi"] == -60 and observation["noise"] == -90
        assert np.asarray(observation["iq"]).shape == (56, 2)
        assert observation["bridge_complete"] == {
            "ns": str(BASE_NS - index),
            "domain": "test:bridge",
            "kind": "unix_wall",
        }
        assert observation["wire_hex"] is None and observation["spans"] == []
        assert observation["infer_available"] is None
        assert observation["epochs"]["rx_epoch"] is None
        assert observation["epochs"]["tx_epoch"] is None
        assert observation["epochs"]["timer_unwrapped_us"] is None
        assert observation["timing"]["acquisition_ns"] is None
        assert observation["timing"]["availability_ns"] is None
        assert not observation["timing"]["usable"]


def test_missing_fields_and_clock_are_null_not_fabricated(tmp_path):
    path = tmp_path / "minimal.h5"
    write_links(path, optional=False)
    result, rows = collect([source(path, domain=None)])
    assert result["counts"]["accepted_frames"] == 2
    assert len(rows["issues"]) == 1
    observation = rows["observations"][0]
    for field in ("t_ns", "esp_us", "seq", "rssi", "noise", "boot_id", "bridge_complete"):
        assert observation[field] is None
        assert field in observation["unknown_fields"]
        assert observation["unknown_reasons"][field]
    assert observation["timing"]["clock_domain"] is None
    assert observation["timing"]["clock_kind"] is None


def test_known_timestamp_without_domain_stays_a_raw_value(tmp_path):
    path = tmp_path / "unknown-clock.h5"
    write_links(path)
    _, rows = collect([source(path, domain=None)])
    observation = rows["observations"][0]
    assert observation["t_ns"] == str(BASE_NS)
    assert observation["bridge_complete"] is None
    assert "clock_domain" in observation["unknown_fields"]


def test_archive_order_is_canonical_but_does_not_sort_timestamps(tmp_path):
    paths = [tmp_path / f"{index}.h5" for index in range(2)]
    for path in paths:
        write_links(path, name="11")
        with h5py.File(path, "r+") as h5:
            h5.copy("links/11", "links/00")
    descriptions = [source(path, source_id=name) for path, name in zip(paths, ("z", "a"))]
    first, rows = collect(descriptions)
    second, again = collect(descriptions[::-1])
    assert (first, rows) == (second, again)
    observations = rows["observations"]
    assert [row["source_ref"]["source_id"] for row in observations] == ["a"] * 4 + ["z"] * 4
    assert [row["source_ref"]["dataset_path"] for row in observations[:4]] == [
        "/links/00",
        "/links/00",
        "/links/11",
        "/links/11",
    ]
    assert [row["t_ns"] for row in observations[:2]] == [str(BASE_NS), str(BASE_NS - 1)]
    assert [row["ingest_ordinal"] for row in observations] == list(range(8))
    assert "not_replay_arrival" in first["observation_order"]


def test_metadata_keeps_bytes_array_dtype_and_literal_strings(tmp_path):
    path = tmp_path / "metadata.h5"
    write_links(path)
    with h5py.File(path, "r+") as h5:
        meta = h5.create_group("meta")
        meta.attrs["created_ns"] = np.uint64(BASE_NS)
        meta.attrs["sources"] = '{"0": ["original.csiraw"]}'
        meta.attrs["opaque"] = np.bytes_(b"\xff\xfe")
        meta.attrs["counts"] = np.array([1, 2, 3], dtype="u4")
        h5.attrs["version"] = "legacy"
    result, _ = collect([source(path)])
    metadata = result["source_metadata"]["legacy"]
    assert metadata["root_attributes"] == {"version": "legacy"}
    attrs = metadata["meta_attributes"]
    assert attrs["created_ns"] == BASE_NS
    assert attrs["sources"] == '{"0": ["original.csiraw"]}'
    assert attrs["opaque"] == {"type": "bytes", "hex": "fffe"}
    assert attrs["counts"] == {"type": "ndarray", "dtype": "<u4", "shape": [3], "values": [1, 2, 3]}


@pytest.mark.parametrize(
    "case,message",
    [
        ("samples_only", "/samples alone"),
        ("missing_links", "missing /links"),
        ("empty_links", "nonempty group"),
        ("zero_rows", "no observations"),
        ("bad_link", "two-digit"),
        ("missing_iq", "required iq"),
        ("bad_iq_shape", "shape"),
        ("unsigned_iq", "dtype int8"),
        ("float_time", "dtype uint64"),
        ("short_seq", "incomplete"),
        ("extra_field", "unsupported legacy link fields"),
        ("nonfinite_meta", "nonfinite HDF5 metadata"),
    ],
)
def test_malformed_or_derived_only_sources_fail_without_rows(tmp_path, case, message):
    path = tmp_path / "bad.h5"
    if case in {"samples_only", "missing_links", "empty_links"}:
        with h5py.File(path, "w") as h5:
            if case == "samples_only":
                h5.create_dataset("samples/X", data=np.zeros((2, 280, 3, 3), dtype="f2"))
            elif case == "empty_links":
                h5.create_group("links")
    else:
        write_links(
            path, name="30" if case == "bad_link" else "00", count=0 if case == "zero_rows" else 2
        )
        with h5py.File(path, "r+") as h5:
            if case in {"missing_iq", "bad_iq_shape", "unsigned_iq"}:
                del h5["links/00/iq"]
                if case != "missing_iq":
                    shape = (2, 112) if case == "bad_iq_shape" else (2, 56, 2)
                    dtype = "u1" if case == "unsigned_iq" else "i1"
                    h5.create_dataset("links/00/iq", data=np.zeros(shape, dtype=dtype))
            elif case == "float_time":
                del h5["links/00/t_ns"]
                h5.create_dataset("links/00/t_ns", data=np.zeros(2, dtype="f8"))
            elif case == "short_seq":
                del h5["links/00/seq"]
                h5.create_dataset("links/00/seq", data=np.zeros(1, dtype="u4"))
            elif case == "extra_field":
                h5.create_dataset("links/00/unknown", data=np.zeros(2, dtype="u4"))
            elif case == "nonfinite_meta":
                h5.create_group("meta").attrs["bad"] = float("nan")
    emitted = []
    with pytest.raises(ValueError, match=message):
        import_links([source(path)], emit=lambda *args: emitted.append(args))
    assert emitted == []


@pytest.mark.parametrize("dependency", ["soft_link", "external_link", "virtual", "external"])
def test_hdf5_dependencies_outside_hashed_file_are_rejected(tmp_path, dependency):
    path = tmp_path / "dependent.h5"
    write_links(path)
    with h5py.File(path, "r+") as h5:
        if dependency == "soft_link":
            h5["ignored"] = h5py.SoftLink("/links/00/iq")
        elif dependency == "external_link":
            h5["ignored"] = h5py.ExternalLink("absent.h5", "/data")
        elif dependency == "virtual":
            layout = h5py.VirtualLayout(shape=(1,), dtype="u8")
            layout[:] = h5py.VirtualSource("absent.h5", "/data", shape=(1,))
            h5.create_virtual_dataset("ignored", layout)
        else:
            h5.create_dataset("ignored", shape=(1,), dtype="u8", external=[("absent.bin", 0, 8)])
    with pytest.raises(ValueError, match="unsupported"):
        collect([source(path)])


def test_hard_link_cycles_do_not_recurse_forever(tmp_path):
    path = tmp_path / "cycle.h5"
    write_links(path)
    with h5py.File(path, "r+") as h5:
        group = h5.create_group("metadata_graph")
        group["back"] = h5["/"]
    result, _ = collect([source(path)])
    assert result["counts"]["stored_frames"] == 2


def test_duplicate_source_id_is_rejected_before_emission(tmp_path):
    path = tmp_path / "duplicate.h5"
    write_links(path)
    with pytest.raises(ValueError, match="unique"):
        collect([source(path), source(path)])
