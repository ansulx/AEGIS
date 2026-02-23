#!/usr/bin/env python3
"""
CLI entrypoint for strict MICCAI 2026 pipeline runs.

Example:
    export PYTHONPATH=src
    python src/main.py full_run \
      --baseline-config configs/default.yaml \
      --main-config configs/phase2_swinunetr.yaml \
      --run-name miccai2026_full
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import asdict
import json
from pathlib import Path
import re
from typing import Any

import torch
import yaml

from aegis.data_loading import DatasetLoadError
from aegis.research_pipeline import PipelineConfig, run_research_pipeline

_SAFE_RUN_NAME = re.compile(r"^[A-Za-z0-9._-]+$")
_PROTECTED_TOP_LEVEL_SECTIONS = {"metrics"}
_PIPELINE_KEYS = set(PipelineConfig.__dataclass_fields__.keys())


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_yaml_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: '{path}'.")
    if not path.is_file():
        raise NotADirectoryError(f"Config path is not a file: '{path}'.")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping/object: '{path}'.")
    return data


def _validate_run_name(run_name: str) -> str:
    if not run_name:
        raise ValueError("run-name cannot be empty.")
    if not _SAFE_RUN_NAME.match(run_name):
        raise ValueError(
            "run-name contains invalid characters. "
            "Use letters, numbers, dot, underscore, and hyphen only."
        )
    return run_name


def _ensure_mapping(value: Any, label: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a mapping/object.")
    return value


def _resolve_pipeline_config(
    baseline_data: dict[str, Any],
    main_data: dict[str, Any],
    allow_cpu_override: bool,
) -> tuple[PipelineConfig, dict[str, Any]]:
    baseline_pipeline = _ensure_mapping(baseline_data.get("pipeline"), "baseline.pipeline")
    main_pipeline = _ensure_mapping(main_data.get("pipeline"), "main.pipeline")

    locked_keys = baseline_data.get("safety", {}).get("locked_pipeline_keys", [])
    if not isinstance(locked_keys, list) or any(not isinstance(key, str) for key in locked_keys):
        raise ValueError("safety.locked_pipeline_keys must be a list of strings.")
    locked = set(locked_keys)

    unknown_baseline = set(baseline_pipeline.keys()) - _PIPELINE_KEYS
    unknown_main = set(main_pipeline.keys()) - _PIPELINE_KEYS
    if unknown_baseline:
        raise ValueError(f"Unknown baseline.pipeline keys: {sorted(unknown_baseline)}.")
    if unknown_main:
        raise ValueError(f"Unknown main.pipeline keys: {sorted(unknown_main)}.")

    for section in _PROTECTED_TOP_LEVEL_SECTIONS:
        if section in main_data:
            raise ValueError(
                f"'{section}' section in main config is protected and cannot be overridden."
            )

    for key in main_pipeline:
        if key in locked:
            raise ValueError(f"main.pipeline attempts to override locked key '{key}'.")

    merged_pipeline = dict(baseline_pipeline)
    merged_pipeline.update(main_pipeline)
    if allow_cpu_override:
        merged_pipeline["require_gpu"] = False

    config = PipelineConfig(**merged_pipeline)

    resolved = {
        "baseline": baseline_data,
        "main": main_data,
        "effective_pipeline": asdict(config),
    }
    return config, resolved


def _resolve_output_dir(
    baseline_data: dict[str, Any], run_name: str, repo_root: Path, force: bool
) -> Path:
    execution = _ensure_mapping(baseline_data.get("execution"), "baseline.execution")
    output_root_value = execution.get("output_root", "outputs")
    if not isinstance(output_root_value, str):
        raise ValueError("execution.output_root must be a string.")
    output_root = (repo_root / output_root_value).resolve()
    output_dir = output_root / run_name

    fail_if_exists = bool(execution.get("fail_if_output_exists", True))
    if output_dir.exists() and any(output_dir.iterdir()) and fail_if_exists and not force:
        raise FileExistsError(
            f"Output directory already exists and is not empty: '{output_dir}'. "
            "Use --force to proceed."
        )
    return output_dir


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AEGIS MICCAI 2026 strict runner.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    full_run = subparsers.add_parser(
        "full_run", help="Run strict ADAM-train + Indian-inference pipeline."
    )
    full_run.add_argument("--baseline-config", type=Path, required=True)
    full_run.add_argument("--main-config", type=Path, required=True)
    full_run.add_argument("--run-name", type=str, required=True)
    full_run.add_argument(
        "--allow-cpu", action="store_true",
        help="Allow CPU fallback. Default is strict GPU mode.",
    )
    full_run.add_argument(
        "--force", action="store_true",
        help="Allow writing into an existing non-empty run directory.",
    )
    full_run.add_argument(
        "--resume-from", type=Path, default=None,
        help="Path to checkpoint to resume training from.",
    )
    full_run.add_argument(
        "--skip-training", action="store_true",
        help="Skip training and go straight to inference using --resume-from checkpoint.",
    )
    return parser


def _handle_full_run(args: argparse.Namespace) -> int:
    repo_root = _repo_root()
    run_name = _validate_run_name(args.run_name)

    baseline_path = (repo_root / args.baseline_config).resolve()
    main_path = (repo_root / args.main_config).resolve()
    baseline_data = _load_yaml_file(baseline_path)
    main_data = _load_yaml_file(main_path)
    experiment_data = _ensure_mapping(main_data.get("experiment"), "main.experiment")

    config, resolved_config = _resolve_pipeline_config(
        baseline_data=baseline_data,
        main_data=main_data,
        allow_cpu_override=bool(args.allow_cpu),
    )
    output_dir = _resolve_output_dir(
        baseline_data=baseline_data,
        run_name=run_name,
        repo_root=repo_root,
        force=bool(args.force),
    )

    logging.info("=" * 60)
    logging.info("AEGIS MICCAI 2026 – Strict Pipeline")
    logging.info("Run name: %s", run_name)
    logging.info("Output dir: %s", output_dir)
    if torch.cuda.is_available():
        logging.info("GPU: %s", torch.cuda.get_device_name(0))
        logging.info("CUDA version: %s", torch.version.cuda)
    logging.info("=" * 60)

    skip_training = bool(getattr(args, "skip_training", False))
    resume_path = args.resume_from
    if skip_training and resume_path is None:
        logging.error("--skip-training requires --resume-from <checkpoint>")
        return 1

    summary = run_research_pipeline(
        output_dir=output_dir,
        base_dir=repo_root,
        config=config,
        resume_from=resume_path,
        skip_training=skip_training,
    )

    reports_dir = output_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    resolved_path = reports_dir / "resolved_config.json"
    with resolved_path.open("w", encoding="utf-8") as handle:
        json.dump(resolved_config, handle, indent=2)

    print("\n" + "=" * 60)
    print("[OK] Full run completed.")
    print(f"Run directory: {output_dir}")
    print(f"Summary report: {reports_dir / 'summary.json'}")
    print(f"Per-case metrics: {reports_dir / 'per_case_metrics.csv'}")
    print(f"Resolved config: {resolved_path}")
    print(f"Model type: {summary['model']['type']}")
    print(f"Device used: {summary['model']['device']}")
    configured_model_name = experiment_data.get("model_name")
    if isinstance(configured_model_name, str):
        print(f"Experiment tag: {configured_model_name}")
    if "failure_analysis" in summary:
        fa = summary["failure_analysis"]
        print(f"Failure detection AUROC: {fa['auroc_trust_vs_failure']:.4f}")
        print(f"Workload reduction: {fa['workload_reduction_fraction']:.1%}")
    print("=" * 60)
    return 0


def main() -> int:
    _setup_logging()
    parser = _build_parser()
    args = parser.parse_args()

    try:
        if args.command == "full_run":
            return _handle_full_run(args)
        raise ValueError(f"Unknown command: {args.command}")
    except (
        RuntimeError, ValueError, FileNotFoundError, NotADirectoryError,
        FileExistsError, DatasetLoadError,
    ) as exc:
        print(f"\n[ERROR] {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
