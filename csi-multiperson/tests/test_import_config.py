"""Source format/clock declarations must not silently change import meaning."""

from pathlib import Path

import pytest

from csi_mp.cli import main
from csi_mp.config import ConfigError, load_config
from csi_mp.ingest.import_legacy import import_session


CAPTURE = Path(__file__).resolve().parents[1] / "configs/capture.yaml"


@pytest.mark.parametrize(
    "change,reason",
    [
        ({"rx_id": 0}, "identifies receivers per group"),
        ({"clock_kind": "monotonic"}, "requires unix_wall"),
        ({"clock_domain": ""}, "clock_domain"),
        ({"raw_schema_id": "CSIRAW01"}, "legacy.links.v1"),
        ({"wire_schema_id": "csi130.v1"}, "original wire absent"),
    ],
)
def test_legacy_mapping_cannot_claim_wire_or_monotonic_semantics(tmp_path, change, reason):
    source = {
        "source_id": "legacy",
        "path": str(tmp_path / "archive.h5"),
        "format": "legacy_hdf5",
        **change,
    }
    with pytest.raises(ConfigError, match=reason):
        load_config(CAPTURE, overrides={"capture": {"sources": [source]}})


def test_inspect_directs_legacy_sources_to_import_before_opening_files(tmp_path, capsys):
    config = tmp_path / "legacy.yaml"
    config.write_text(
        "capture:\n  sources:\n    - source_id: legacy\n"
        "      format: legacy_hdf5\n      path: does-not-exist.h5\n"
    )
    assert main(["inspect", "--config", str(config)]) == 2
    result = capsys.readouterr()
    assert result.out == ""
    assert "use import for legacy HDF5" in result.err


def test_import_cannot_merge_archive_rows_with_raw_arrival_stream(tmp_path):
    sources = load_config(CAPTURE).as_dict()["capture"]["sources"]
    sources.append(
        {"source_id": "legacy", "path": str(tmp_path / "missing.h5"), "format": "legacy_hdf5"}
    )
    cfg = load_config(CAPTURE, overrides={"capture": {"sources": sources}})
    output = tmp_path / "mixed.csisession"
    with pytest.raises(ValueError, match="cannot mix"):
        import_session(None, output, cfg)
    assert not output.exists()
    assert not (tmp_path / f".{output.name}.importing").exists()


def test_explicit_source_cannot_bypass_trusted_mapping(tmp_path):
    cfg = load_config(CAPTURE)
    source = cfg.as_dict()["capture"]["sources"][0]["path"]
    output = tmp_path / "subset.csisession"
    with pytest.raises(ValueError, match="exactly match"):
        import_session(source, output, cfg)
    assert not output.exists()
