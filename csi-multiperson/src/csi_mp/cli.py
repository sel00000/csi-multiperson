"""Lightweight entrypoints; no teacher, torch, CUDA, or camera imports."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
import sys
import tempfile


def _write(path, value):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
        # Publish a fully written file exclusively. A racing writer cannot be
        # overwritten, and a partial JSON file never becomes the completion marker.
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _output_directory(path, cfg):
    destination = Path(path).resolve()
    workspaces = {
        parent
        for location in (Path(__file__).resolve(), cfg.path.resolve())
        for parent in location.parents
        if (parent / "docs/multiperson-csi-design").is_dir() and (parent / "examples").is_dir()
    }
    protected = tuple(root / name for root in workspaces for name in ("examples", "ex)robot"))
    if any(destination == root or root in destination.parents for root in protected):
        raise ValueError("output: reference examples and papers are read-only")
    for source in cfg.as_dict()["capture"]["sources"]:
        if destination == Path(source["path"]):
            raise ValueError("output: cannot replace a raw source")
    destination.mkdir(parents=True, exist_ok=False)
    return destination


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="csi-mp", description="CSI S00–S05 inspection and reproducible session import"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (
        ("validate-config", "validate config and report its canonical SHA256"),
        ("inspect", "read CSIRAW01 sources, classify and timestamp frames without a model"),
        ("import", "preserve raw or legacy /links sources in a new verified session file"),
    ):
        sub = commands.add_parser(name, help=help_text)
        sub.add_argument("--config", required=True, type=Path)
        if name == "import":
            sub.add_argument(
                "--output",
                required=True,
                type=Path,
                help="new .csisession file; existing or interrupted imports are refused",
            )
        if name == "inspect":
            outputs = sub.add_mutually_exclusive_group()
            outputs.add_argument(
                "--output", type=Path, help="new report directory; existing paths are refused"
            )
            outputs.add_argument(
                "--save", action="store_true", help="write output_dir/run_id from config"
            )
    args = parser.parse_args(argv)
    destination = None
    try:
        from .config import load_config, selected_metrics

        cfg = load_config(args.config)
        if args.command == "validate-config":
            print(
                json.dumps(
                    {
                        "valid": True,
                        "config_hash": cfg.config_hash,
                        "selected_metrics": selected_metrics(cfg),
                    },
                    sort_keys=True,
                )
            )
            return 0
        if args.command == "import":
            from .ingest.import_legacy import import_session

            report = import_session(None, args.output, cfg)
            print(
                json.dumps(
                    {
                        "status": "completed",
                        "output": str(args.output.absolute()),
                        **report.to_dict(),
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
            return 0
        from .ingest.inspect import inspect_capture

        with ExitStack() as stack:
            writers = {}
            output = args.output
            if args.save:
                resolved = cfg.as_dict()
                output = Path(resolved["output_dir"]) / resolved["run_id"]
            if output is not None:
                destination = _output_directory(output, cfg)
                _write(destination / "config.resolved.json", cfg.as_dict())
                kinds = ["frames", "issues", "epochs"]
                if "jsonl" in cfg.as_dict()["measurement"]["sinks"]:
                    kinds.append("metrics")
                for kind in kinds:
                    writers[kind] = stack.enter_context(
                        (destination / f"{kind}.jsonl").open("x", encoding="utf-8")
                    )

            def emit(kind, value):
                if kind in writers:
                    writers[kind].write(
                        json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)
                        + "\n"
                    )

            report = inspect_capture(cfg, emit=emit)
        # Closing JSONL writers can fail while flushing. Complete all writes
        # before publishing the successful manifest and stdout status.
        if destination is not None:
            for kind in ("metrics", "dashboard", "manifest"):
                _write(destination / f"{kind}.json", report[kind])
        print(
            json.dumps(
                {**report["manifest"], "output": str(destination) if destination else None},
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0
    except (OSError, ValueError, TypeError) as exc:
        if destination is not None:
            try:
                _write(destination / "failure.json", {"status": "failed", "error": str(exc)})
            except OSError as report_error:
                print(f"error: could not write failure report: {report_error}", file=sys.stderr)
        print(f"error: {exc}", file=sys.stderr)
        return 2
