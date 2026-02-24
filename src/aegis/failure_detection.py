"""
Failure-detection analysis for the AEGIS clinical AI pipeline.

Evaluates whether Monte Carlo uncertainty (trust_score) can reliably
identify segmentation failures, enabling a fail-safe workflow where
uncertain cases are routed to manual expert review.
"""

from __future__ import annotations

import csv
import json
import logging
import math
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve

from .data_loading import DatasetLoadError

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FailureDetectionResult:
    auroc_trust_vs_failure: float
    auroc_vfstd_vs_failure: float
    auroc_reliability_lr: float | None
    auroc_reliability_mlp: float | None
    auroc_reliability_rf: float | None
    optimal_trust_threshold: float
    optimal_trust_sensitivity: float
    optimal_trust_specificity: float
    failure_prevalence: float
    cases_flagged_at_optimal: int
    cases_total: int
    workload_reduction_fraction: float
    missed_failure_rate: float
    roc_fpr: list[float]
    roc_tpr: list[float]
    roc_fpr_reliability: list[float] | None
    roc_tpr_reliability: list[float] | None
    per_cohort_results: dict[str, dict[str, Any]] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _safe_float(value: str | None) -> float | None:
    """Parse a CSV cell to float, returning None for missing / unparsable."""
    if value is None:
        return None
    value = value.strip()
    if value == "" or value.lower() in ("none", "nan", "na", "null"):
        return None
    try:
        f = float(value)
        return None if math.isnan(f) else f
    except (ValueError, TypeError):
        return None


def _safe_bool(value: str | None) -> bool | None:
    if value is None:
        return None
    value = value.strip().lower()
    if value in ("true", "1", "yes"):
        return True
    if value in ("false", "0", "no"):
        return False
    return None


def _load_cases(per_case_csv_path: Path) -> list[dict[str, Any]]:
    """Load the per-case CSV, keeping only rows with ground truth and valid metrics."""
    if not per_case_csv_path.exists():
        raise FileNotFoundError(
            f"Per-case metrics CSV not found: '{per_case_csv_path.resolve()}'."
        )

    rows: list[dict[str, Any]] = []
    with open(per_case_csv_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for raw in reader:
            has_gt = _safe_bool(raw.get("has_ground_truth"))
            if has_gt is not True:
                continue

            dice = _safe_float(raw.get("dice"))
            trust = _safe_float(raw.get("trust_score"))
            vfstd = _safe_float(raw.get("volume_fraction_std"))

            if dice is None or trust is None:
                continue

            row_dict: dict[str, Any] = {
                "cohort": (raw.get("cohort") or "unknown").strip(),
                "session_id": (raw.get("session_id") or "").strip(),
                "dice": dice,
                "trust_score": trust,
                "volume_fraction_std": vfstd if vfstd is not None else 0.0,
            }
            rel_lr = _safe_float(raw.get("reliability_score_lr"))
            rel_mlp = _safe_float(raw.get("reliability_score_mlp"))
            rel_rf = _safe_float(raw.get("reliability_score_rf"))
            if rel_lr is not None:
                row_dict["reliability_score_lr"] = rel_lr
            if rel_mlp is not None:
                row_dict["reliability_score_mlp"] = rel_mlp
            if rel_rf is not None:
                row_dict["reliability_score_rf"] = rel_rf
            rows.append(row_dict)

    return rows


def _compute_auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    """AUROC with single-class guard."""
    unique = np.unique(labels)
    if len(unique) < 2:
        warnings.warn(
            "Only one class present in labels — AUROC is undefined; returning NaN.",
            stacklevel=3,
        )
        return float("nan")
    return float(roc_auc_score(labels, scores))


def _optimal_threshold_youden(
    fpr: np.ndarray, tpr: np.ndarray, thresholds: np.ndarray
) -> tuple[float, float, float, int]:
    """Find the threshold maximising Youden's J (TPR - FPR).

    Returns (threshold, sensitivity, specificity, index).
    """
    j_scores = tpr - fpr
    idx = int(np.argmax(j_scores))
    return float(thresholds[idx]), float(tpr[idx]), float(1.0 - fpr[idx]), idx


def _cohort_metrics(
    cases: list[dict[str, Any]],
    failure_dice_threshold: float,
) -> dict[str, Any]:
    """Compute failure-detection metrics for a single cohort subset."""
    n = len(cases)
    if n == 0:
        return {"n": 0, "note": "no cases"}

    labels = np.array([1 if c["dice"] < failure_dice_threshold else 0 for c in cases])
    trust = np.array([c["trust_score"] for c in cases])

    n_failures = int(labels.sum())
    prevalence = n_failures / n

    # Trust score: low trust → high failure probability, so predict failure
    # with (1 - trust_score) as the "failure score".
    failure_scores = 1.0 - trust
    auroc = _compute_auroc(labels, failure_scores)

    result: dict[str, Any] = {
        "n": n,
        "n_failures": n_failures,
        "failure_prevalence": round(prevalence, 4),
        "auroc_trust_vs_failure": round(auroc, 4) if not math.isnan(auroc) else None,
    }

    if n_failures == 0:
        result["note"] = "no failures in cohort"
    elif n_failures == n:
        result["note"] = "all cases are failures"

    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_failure_detection_analysis(
    per_case_csv_path: Path,
    failure_dice_threshold: float = 0.5,
) -> FailureDetectionResult:
    """Analyse whether uncertainty metrics predict segmentation failures.

    Parameters
    ----------
    per_case_csv_path:
        Path to ``reports/per_case_metrics.csv``.
    failure_dice_threshold:
        Dice scores below this value are classified as failures.

    Returns
    -------
    FailureDetectionResult
    """
    cases = _load_cases(per_case_csv_path)
    if len(cases) < 5:
        raise DatasetLoadError(
            f"Need at least 5 cases with ground truth for failure detection "
            f"analysis; found {len(cases)} in '{per_case_csv_path}'."
        )

    labels = np.array(
        [1 if c["dice"] < failure_dice_threshold else 0 for c in cases]
    )
    trust = np.array([c["trust_score"] for c in cases])
    vfstd = np.array([c["volume_fraction_std"] for c in cases])
    n = len(cases)
    n_failures = int(labels.sum())
    prevalence = n_failures / n

    # --- Edge: no failures ------------------------------------------------
    if n_failures == 0:
        return FailureDetectionResult(
            auroc_trust_vs_failure=1.0,
            auroc_vfstd_vs_failure=1.0,
            auroc_reliability_lr=None,
            auroc_reliability_mlp=None,
            auroc_reliability_rf=None,
            optimal_trust_threshold=0.0,
            optimal_trust_sensitivity=1.0,
            optimal_trust_specificity=1.0,
            failure_prevalence=0.0,
            cases_flagged_at_optimal=0,
            cases_total=n,
            workload_reduction_fraction=1.0,
            missed_failure_rate=0.0,
            roc_fpr=[0.0, 1.0],
            roc_tpr=[0.0, 1.0],
            roc_fpr_reliability=None,
            roc_tpr_reliability=None,
            per_cohort_results={},
        )

    # --- Edge: all failures -----------------------------------------------
    if n_failures == n:
        return FailureDetectionResult(
            auroc_trust_vs_failure=0.5,
            auroc_vfstd_vs_failure=0.5,
            auroc_reliability_lr=None,
            auroc_reliability_mlp=None,
            auroc_reliability_rf=None,
            optimal_trust_threshold=1.0,
            optimal_trust_sensitivity=1.0,
            optimal_trust_specificity=0.0,
            failure_prevalence=1.0,
            cases_flagged_at_optimal=n,
            cases_total=n,
            workload_reduction_fraction=0.0,
            missed_failure_rate=0.0,
            roc_fpr=[0.0, 1.0],
            roc_tpr=[0.0, 1.0],
            roc_fpr_reliability=None,
            roc_tpr_reliability=None,
            per_cohort_results={},
        )

    # --- Normal path: both classes present --------------------------------
    failure_scores_trust = 1.0 - trust
    auroc_trust = _compute_auroc(labels, failure_scores_trust)
    auroc_vfstd = _compute_auroc(labels, vfstd)

    fpr, tpr, thresholds = roc_curve(labels, failure_scores_trust)

    opt_thresh_inv, sensitivity, specificity, _ = _optimal_threshold_youden(
        fpr, tpr, thresholds
    )
    # Convert back from inverted score to trust-score threshold
    optimal_trust_threshold = 1.0 - opt_thresh_inv

    # Cases flagged = trust_score < optimal_trust_threshold (uncertain cases)
    flagged_mask = trust < optimal_trust_threshold
    cases_flagged = int(flagged_mask.sum())
    not_flagged_mask = ~flagged_mask

    workload_reduction = float(not_flagged_mask.sum()) / n

    n_not_flagged = int(not_flagged_mask.sum())
    if n_not_flagged == 0:
        missed_failure_rate = 0.0
    else:
        missed_failures = int(labels[not_flagged_mask].sum())
        missed_failure_rate = missed_failures / n_not_flagged

    # Per-cohort breakdown
    cohorts: dict[str, list[dict[str, Any]]] = {}
    for c in cases:
        cohorts.setdefault(c["cohort"], []).append(c)

    per_cohort: dict[str, dict[str, Any]] = {}
    for cohort_name, cohort_cases in sorted(cohorts.items()):
        per_cohort[cohort_name] = _cohort_metrics(cohort_cases, failure_dice_threshold)

    # --- Reliability score AUROC (if column present) -----------------------
    auroc_rel_lr: float | None = None
    auroc_rel_mlp: float | None = None
    auroc_rel_rf: float | None = None
    roc_fpr_rel: list[float] | None = None
    roc_tpr_rel: list[float] | None = None

    has_reliability = all(
        c.get("reliability_score_lr") is not None for c in cases
    )
    if has_reliability:
        rel_lr = np.array([float(c["reliability_score_lr"]) for c in cases])
        rel_mlp = np.array([float(c.get("reliability_score_mlp", 0.5)) for c in cases])
        rel_rf = np.array([float(c.get("reliability_score_rf", 0.5)) for c in cases])
        auroc_rel_lr = round(_compute_auroc(labels, 1.0 - rel_lr), 4)
        auroc_rel_mlp = round(_compute_auroc(labels, 1.0 - rel_mlp), 4)
        auroc_rel_rf = round(_compute_auroc(labels, 1.0 - rel_rf), 4)
        best_auroc = max(
            auroc_rel_lr if not math.isnan(auroc_rel_lr) else 0.0,
            auroc_rel_rf if auroc_rel_rf is not None and not math.isnan(auroc_rel_rf) else 0.0,
        )
        if best_auroc == auroc_rel_lr:
            best_rel_scores = rel_lr
            best_rel_name = "LR"
        else:
            best_rel_scores = rel_rf
            best_rel_name = "RF"
        fpr_rel, tpr_rel, _ = roc_curve(labels, 1.0 - best_rel_scores)
        roc_fpr_rel = [round(float(x), 6) for x in fpr_rel]
        roc_tpr_rel = [round(float(x), 6) for x in tpr_rel]
        logger.info(
            "Failure detection AUROC — trust: %.4f, rel_lr: %.4f, rel_mlp: %.4f, rel_rf: %.4f (best curve: %s)",
            auroc_trust, auroc_rel_lr, auroc_rel_mlp, auroc_rel_rf, best_rel_name,
        )

    return FailureDetectionResult(
        auroc_trust_vs_failure=round(auroc_trust, 4),
        auroc_vfstd_vs_failure=round(auroc_vfstd, 4),
        auroc_reliability_lr=auroc_rel_lr,
        auroc_reliability_mlp=auroc_rel_mlp,
        auroc_reliability_rf=auroc_rel_rf,
        optimal_trust_threshold=round(optimal_trust_threshold, 4),
        optimal_trust_sensitivity=round(sensitivity, 4),
        optimal_trust_specificity=round(specificity, 4),
        failure_prevalence=round(prevalence, 4),
        cases_flagged_at_optimal=cases_flagged,
        cases_total=n,
        workload_reduction_fraction=round(workload_reduction, 4),
        missed_failure_rate=round(missed_failure_rate, 4),
        roc_fpr=[round(float(x), 6) for x in fpr],
        roc_tpr=[round(float(x), 6) for x in tpr],
        roc_fpr_reliability=roc_fpr_rel,
        roc_tpr_reliability=roc_tpr_rel,
        per_cohort_results=per_cohort,
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def generate_failure_detection_report(
    result: FailureDetectionResult,
    out_path: Path,
) -> None:
    """Serialise the analysis result to a JSON report with clinical interpretation."""
    auto_accepted_pct = round(result.workload_reduction_fraction * 100, 1)
    missed_pct = round(result.missed_failure_rate * 100, 1)

    interpretation = (
        f"At the optimal trust threshold of {result.optimal_trust_threshold:.2f}, "
        f"{auto_accepted_pct}% of cases would be auto-accepted, "
        f"missing {missed_pct}% of failures among those auto-accepted cases. "
        f"This enables a clinical workflow where {result.cases_flagged_at_optimal} "
        f"of {result.cases_total} cases are routed for expert review, reducing "
        f"radiologist workload by {auto_accepted_pct}% while maintaining a "
        f"sensitivity of {result.optimal_trust_sensitivity:.2f} for failure detection."
    )

    payload: dict[str, Any] = asdict(result)
    payload["interpretation"] = interpretation
    payload["clinical_workflow"] = {
        "auto_accept_above_trust": result.optimal_trust_threshold,
        "flag_for_review_below_trust": result.optimal_trust_threshold,
        "expected_workload_reduction_pct": auto_accepted_pct,
        "expected_missed_failure_pct": missed_pct,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=str)

    logger.info("Failure-detection report written to %s", out_path)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def generate_roc_figure(
    result: FailureDetectionResult,
    out_path: Path,
) -> None:
    """Publication-quality ROC curve with AUROC annotation and optimal operating point."""
    fig, ax = plt.subplots(figsize=(6, 6), dpi=200)

    ax.plot(
        result.roc_fpr,
        result.roc_tpr,
        color="#2563eb",
        linewidth=2,
        label=f"Trust score (AUROC = {result.auroc_trust_vs_failure:.3f})",
    )

    best_rel_auroc = max(
        result.auroc_reliability_lr or 0.0,
        result.auroc_reliability_rf or 0.0,
    )
    best_rel_label = "LR" if (result.auroc_reliability_lr or 0.0) >= (result.auroc_reliability_rf or 0.0) else "RF"
    if (
        result.roc_fpr_reliability is not None
        and result.roc_tpr_reliability is not None
        and best_rel_auroc > 0
    ):
        ax.plot(
            result.roc_fpr_reliability,
            result.roc_tpr_reliability,
            color="#16a34a",
            linewidth=2,
            linestyle="-.",
            label=f"Reliability {best_rel_label} (AUROC = {best_rel_auroc:.3f})",
        )

    ax.plot([0, 1], [0, 1], linestyle="--", color="#9ca3af", linewidth=1, label="Chance")

    opt_fpr = 1.0 - result.optimal_trust_specificity
    opt_tpr = result.optimal_trust_sensitivity
    ax.scatter(
        [opt_fpr],
        [opt_tpr],
        color="#dc2626",
        s=80,
        zorder=5,
        label=(
            f"Optimal (J): threshold={result.optimal_trust_threshold:.2f}, "
            f"sens={opt_tpr:.2f}, spec={result.optimal_trust_specificity:.2f}"
        ),
    )

    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=12)
    ax.set_title("Failure Detection: Trust Score ROC", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("ROC figure saved to %s", out_path)


def generate_calibration_figure(
    per_case_csv_path: Path,
    out_path: Path,
    n_bins: int = 10,
    failure_dice_threshold: float = 0.5,
) -> None:
    """Reliability diagram: trust-score calibration against actual success rate."""
    cases = _load_cases(per_case_csv_path)
    if len(cases) < 5:
        raise DatasetLoadError(
            f"Need at least 5 cases for calibration figure; found {len(cases)}."
        )

    trust = np.array([c["trust_score"] for c in cases])
    success = np.array([1 if c["dice"] >= failure_dice_threshold else 0 for c in cases])

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_means: list[float] = []
    bin_success_fracs: list[float] = []
    bin_counts: list[int] = []

    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        if lo == bin_edges[0]:
            mask = (trust >= lo) & (trust <= hi)
        else:
            mask = (trust > lo) & (trust <= hi)
        count = int(mask.sum())
        bin_counts.append(count)
        if count == 0:
            bin_means.append(float((lo + hi) / 2))
            bin_success_fracs.append(float("nan"))
        else:
            bin_means.append(float(trust[mask].mean()))
            bin_success_fracs.append(float(success[mask].mean()))

    fig, ax1 = plt.subplots(figsize=(6, 6), dpi=200)

    valid = [i for i, f in enumerate(bin_success_fracs) if not math.isnan(f)]
    ax1.plot(
        [bin_means[i] for i in valid],
        [bin_success_fracs[i] for i in valid],
        marker="o",
        color="#2563eb",
        linewidth=2,
        markersize=6,
        label="Observed success fraction",
    )
    ax1.plot([0, 1], [0, 1], linestyle="--", color="#9ca3af", linewidth=1, label="Perfect calibration")
    ax1.set_xlabel("Mean Trust Score (binned)", fontsize=12)
    ax1.set_ylabel("Fraction of Successes (Dice >= threshold)", fontsize=12)
    ax1.set_xlim(-0.02, 1.02)
    ax1.set_ylim(-0.02, 1.02)
    ax1.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax1.set_title("Trust Score Calibration", fontsize=13, fontweight="bold")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    bar_width = 1.0 / n_bins * 0.8
    ax2.bar(
        bin_means,
        bin_counts,
        width=bar_width,
        alpha=0.25,
        color="#6366f1",
        label="Case count",
    )
    ax2.set_ylabel("Number of Cases", fontsize=11, color="#6366f1")
    ax2.tick_params(axis="y", labelcolor="#6366f1")
    ax2.legend(loc="upper right", fontsize=9, framealpha=0.9)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Calibration figure saved to %s", out_path)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def run_full_failure_analysis(
    per_case_csv_path: Path,
    output_dir: Path,
    failure_dice_threshold: float = 0.5,
) -> FailureDetectionResult:
    """End-to-end failure-detection analysis with reports and figures.

    Outputs are written to ``output_dir/failure_analysis/``.
    """
    per_case_csv_path = Path(per_case_csv_path)
    if not per_case_csv_path.exists():
        raise FileNotFoundError(
            f"Per-case metrics CSV not found: '{per_case_csv_path.resolve()}'."
        )

    analysis_dir = Path(output_dir) / "failure_analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    result = run_failure_detection_analysis(
        per_case_csv_path,
        failure_dice_threshold=failure_dice_threshold,
    )

    generate_failure_detection_report(result, analysis_dir / "failure_detection_report.json")
    generate_roc_figure(result, analysis_dir / "roc_curve.png")
    generate_calibration_figure(
        per_case_csv_path,
        analysis_dir / "calibration.png",
        failure_dice_threshold=failure_dice_threshold,
    )

    logger.info(
        "Full failure analysis complete — outputs in %s",
        analysis_dir.resolve(),
    )
    return result
