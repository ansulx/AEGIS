#!/usr/bin/env python3
"""
Run strict AEGIS research pipeline:
- Train on ADAM only
- Inference on ADAM + Indian cohorts
- Monte Carlo uncertainty outputs
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from aegis.data_loading import DatasetLoadError
from aegis.research_pipeline import PipelineConfig, run_research_pipeline


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Strict ADAM-train / Indian-inference-only AEGIS pipeline."
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=REPO_ROOT,
        help="Repository base directory containing dataset folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "research_pipeline",
        help="Directory to store reports, predictions, and uncertainty maps.",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--mc-samples", type=int, default=20)
    parser.add_argument("--mc-threshold-jitter", type=float, default=0.15)
    parser.add_argument("--mc-input-noise-std", type=float, default=0.05)
    parser.add_argument("--adam-train-fraction", type=float, default=0.8)
    parser.add_argument("--min-adam-train-cases", type=int, default=5)
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Allow CPU fallback. By default, this script strictly requires CUDA.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        config = PipelineConfig(
            adam_train_fraction=args.adam_train_fraction,
            min_adam_train_cases=args.min_adam_train_cases,
            seed=args.seed,
            mc_samples=args.mc_samples,
            mc_threshold_jitter=args.mc_threshold_jitter,
            mc_input_noise_std=args.mc_input_noise_std,
            require_gpu=not args.allow_cpu,
        )
        summary = run_research_pipeline(
            output_dir=args.output_dir,
            base_dir=args.base_dir,
            config=config,
        )
    except (RuntimeError, ValueError, FileNotFoundError, NotADirectoryError, DatasetLoadError) as exc:
        print(f"[ERROR] Pipeline failed: {exc}")
        return 1

    print("[OK] Strict pipeline completed successfully.")
    print(f"Summary report: {args.output_dir / 'reports' / 'summary.json'}")
    print(f"Per-case metrics: {args.output_dir / 'reports' / 'per_case_metrics.csv'}")
    print(
        "Policy enforced: ADAM training enabled, Indian training disabled, "
        "Indian inference-only enabled."
    )
    print(f"Device used: {summary['model']['device']}")
    print(f"Selected threshold: {summary['model']['threshold']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
