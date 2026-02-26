"""Publication-quality figures for the AEGIS cerebrovascular segmentation pipeline."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

logger = logging.getLogger(__name__)

_COHORT_PALETTE: dict[str, str] = {
    "adam_train": "#1f77b4",
    "adam_holdout": "#ff7f0e",
    "indian_car_inference_only": "#2ca02c",
    "indian_ncar_inference_only": "#d62728",
}


def _cohort_color(cohort: str) -> str:
    return _COHORT_PALETTE.get(cohort, "#7f7f7f")


def _find_max_uncertainty_slice(uncertainty: np.ndarray) -> int:
    per_slice = uncertainty.sum(axis=(0, 1))
    if per_slice.max() == 0:
        return uncertainty.shape[2] // 2
    return int(np.argmax(per_slice))


def _mask_contour(mask_2d: np.ndarray) -> np.ndarray:
    """Return a binary edge map from a 2-D binary mask using simple erosion."""
    from scipy.ndimage import binary_erosion

    if mask_2d.max() == 0:
        return np.zeros_like(mask_2d, dtype=bool)
    interior = binary_erosion(mask_2d.astype(bool), iterations=1)
    return np.logical_and(mask_2d.astype(bool), ~interior)


# ---------------------------------------------------------------------------
# 1. Per-case panel
# ---------------------------------------------------------------------------

def generate_case_panel(
    image: np.ndarray,
    pred_mask: np.ndarray,
    uncertainty: np.ndarray,
    out_path: Path | str,
    gt_mask: Optional[np.ndarray] = None,
    metrics: Optional[dict[str, float]] = None,
    session_id: Optional[str] = None,
) -> Path:
    """Create a publication-quality 1-row panel for a single case.

    Returns the path of the saved PNG.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    slice_idx = _find_max_uncertainty_slice(uncertainty)
    img_slice = image[:, :, slice_idx]
    pred_slice = pred_mask[:, :, slice_idx]
    unc_slice = uncertainty[:, :, slice_idx]
    gt_slice = gt_mask[:, :, slice_idx] if gt_mask is not None else None

    ncols = 4 if gt_slice is not None else 3
    fig, axes = plt.subplots(1, ncols, figsize=(16, 4))

    col = 0

    # --- Input with pred contour (red) and GT contour (green) ---
    ax = axes[col]
    ax.imshow(img_slice.T, cmap="Greys_r", origin="lower")
    pred_edge = _mask_contour(pred_slice)
    if pred_edge.any():
        overlay = np.zeros((*pred_edge.shape, 4), dtype=np.float32)
        overlay[pred_edge, :] = [1.0, 0.0, 0.0, 0.9]
        ax.imshow(overlay.transpose(1, 0, 2), origin="lower")
    if gt_slice is not None:
        gt_edge = _mask_contour(gt_slice)
        if gt_edge.any():
            overlay_gt = np.zeros((*gt_edge.shape, 4), dtype=np.float32)
            overlay_gt[gt_edge, :] = [0.0, 1.0, 0.0, 0.9]
            ax.imshow(overlay_gt.transpose(1, 0, 2), origin="lower")
    ax.set_title("Input", fontsize=10)
    ax.axis("off")
    col += 1

    # --- Ground truth (if available) ---
    if gt_slice is not None:
        ax = axes[col]
        ax.imshow(gt_slice.T, cmap="Greys_r", origin="lower", vmin=0, vmax=1)
        ax.set_title("Ground Truth", fontsize=10)
        ax.axis("off")
        col += 1

    # --- Prediction ---
    ax = axes[col]
    ax.imshow(pred_slice.T, cmap="Greys_r", origin="lower", vmin=0, vmax=1)
    ax.set_title("Prediction", fontsize=10)
    ax.axis("off")
    col += 1

    # --- Uncertainty ---
    ax = axes[col]
    unc_max = float(unc_slice.max()) if unc_slice.max() > 0 else 1.0
    ax.imshow(unc_slice.T, cmap="hot", origin="lower", norm=Normalize(vmin=0, vmax=unc_max))
    ax.set_title("Uncertainty", fontsize=10)
    ax.axis("off")

    # --- Annotations ---
    parts: list[str] = []
    if session_id:
        parts.append(session_id)
    if metrics:
        if metrics.get("dice") is not None:
            parts.append(f"Dice={metrics['dice']:.3f}")
        if metrics.get("trust_score") is not None:
            parts.append(f"Trust={metrics['trust_score']:.3f}")
    if parts:
        fig.suptitle("  |  ".join(parts), fontsize=11, y=0.98)

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# 2. Cohort summary figure
# ---------------------------------------------------------------------------

def generate_cohort_summary_figure(
    per_case_csv_path: Path | str,
    out_path: Path | str,
) -> Path:
    """2x2 summary of Dice / trust / scatter / bar across cohorts."""
    per_case_csv_path = Path(per_case_csv_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(per_case_csv_path)
    df["has_ground_truth"] = df["has_ground_truth"].astype(str).str.lower().isin(["true", "1"])

    gt_df = df[df["has_ground_truth"]].copy()
    cohorts_with_gt = sorted(gt_df["cohort"].unique())
    all_cohorts = sorted(df["cohort"].unique())

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # --- Top-left: Dice box plot (cohorts with GT only) ---
    ax = axes[0, 0]
    if cohorts_with_gt:
        data = [gt_df.loc[gt_df["cohort"] == c, "dice"].dropna().values for c in cohorts_with_gt]
        bp = ax.boxplot(data, labels=cohorts_with_gt, patch_artist=True)
        for patch, cohort in zip(bp["boxes"], cohorts_with_gt):
            patch.set_facecolor(_cohort_color(cohort))
            patch.set_alpha(0.7)
    ax.set_title("Dice Score by Cohort")
    ax.set_ylabel("Dice")
    ax.tick_params(axis="x", rotation=25)

    # --- Top-right: Trust score box plot (all cohorts) ---
    ax = axes[0, 1]
    if all_cohorts:
        data = [df.loc[df["cohort"] == c, "trust_score"].dropna().values for c in all_cohorts]
        bp = ax.boxplot(data, labels=all_cohorts, patch_artist=True)
        for patch, cohort in zip(bp["boxes"], all_cohorts):
            patch.set_facecolor(_cohort_color(cohort))
            patch.set_alpha(0.7)
    ax.set_title("Trust Score by Cohort")
    ax.set_ylabel("Trust Score")
    ax.tick_params(axis="x", rotation=25)

    # --- Bottom-left: Dice vs Trust scatter ---
    ax = axes[1, 0]
    for cohort in cohorts_with_gt:
        sub = gt_df[gt_df["cohort"] == cohort]
        ax.scatter(
            sub["trust_score"], sub["dice"],
            c=_cohort_color(cohort), label=cohort, alpha=0.7, edgecolors="k", linewidths=0.3, s=40,
        )
    ax.set_xlabel("Trust Score")
    ax.set_ylabel("Dice")
    ax.set_title("Dice vs Trust Score")
    if cohorts_with_gt:
        ax.legend(fontsize=7, loc="lower right")

    # --- Bottom-right: Mean metrics bar chart ---
    ax = axes[1, 1]
    metric_names = ["dice", "iou", "sensitivity", "specificity"]
    if cohorts_with_gt:
        x = np.arange(len(metric_names))
        width = 0.8 / max(len(cohorts_with_gt), 1)
        for i, cohort in enumerate(cohorts_with_gt):
            means = [gt_df.loc[gt_df["cohort"] == cohort, m].mean() for m in metric_names]
            ax.bar(x + i * width, means, width, label=cohort, color=_cohort_color(cohort), alpha=0.8)
        ax.set_xticks(x + width * (len(cohorts_with_gt) - 1) / 2)
        ax.set_xticklabels(metric_names)
        ax.legend(fontsize=7)
    ax.set_title("Mean Metrics by Cohort")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# 3. Uncertainty-error correlation figure
# ---------------------------------------------------------------------------

def generate_uncertainty_correlation_figure(
    per_case_csv_path: Path | str,
    out_path: Path | str,
) -> Path:
    """Scatter of (1 - trust_score) vs (1 - dice) with regression line."""
    per_case_csv_path = Path(per_case_csv_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(per_case_csv_path)
    df["has_ground_truth"] = df["has_ground_truth"].astype(str).str.lower().isin(["true", "1"])
    gt_df = df[df["has_ground_truth"]].dropna(subset=["dice", "trust_score"]).copy()

    fig, ax = plt.subplots(figsize=(8, 6))

    if gt_df.empty:
        ax.text(0.5, 0.5, "No cases with ground truth", transform=ax.transAxes, ha="center")
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return out_path

    gt_df["uncertainty"] = 1.0 - gt_df["trust_score"]
    gt_df["error"] = 1.0 - gt_df["dice"]

    for cohort in sorted(gt_df["cohort"].unique()):
        sub = gt_df[gt_df["cohort"] == cohort]
        ax.scatter(
            sub["uncertainty"], sub["error"],
            c=_cohort_color(cohort), label=cohort, alpha=0.7, edgecolors="k", linewidths=0.3, s=45,
        )

    x_all = gt_df["uncertainty"].values
    y_all = gt_df["error"].values

    if len(x_all) >= 3:
        slope, intercept, r_value, _, _ = stats.linregress(x_all, y_all)
        x_line = np.linspace(float(x_all.min()), float(x_all.max()), 100)
        ax.plot(x_line, slope * x_line + intercept, "k--", linewidth=1.2, alpha=0.8)

        pearson_r, pearson_p = stats.pearsonr(x_all, y_all)
        ax.annotate(
            f"r = {pearson_r:.3f}  (p = {pearson_p:.2e})",
            xy=(0.05, 0.95), xycoords="axes fraction",
            fontsize=10, verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.5),
        )

    ax.set_xlabel("Uncertainty  (1 − Trust Score)")
    ax.set_ylabel("Segmentation Error  (1 − Dice)")
    ax.set_title("Uncertainty–Error Correlation")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# 4. Best / worst gallery
# ---------------------------------------------------------------------------

def generate_best_worst_gallery(
    per_case_csv_path: Path | str,
    qualitative_dir: Path | str,
    out_path: Path | str,
    n_cases: int = 3,
) -> Path:
    """Gallery of top-N best and worst Dice cases."""
    per_case_csv_path = Path(per_case_csv_path)
    qualitative_dir = Path(qualitative_dir)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(per_case_csv_path)
    df["has_ground_truth"] = df["has_ground_truth"].astype(str).str.lower().isin(["true", "1"])
    gt_df = df[df["has_ground_truth"]].dropna(subset=["dice"]).copy()

    if gt_df.empty:
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.text(0.5, 0.5, "No cases with ground truth", transform=ax.transAxes, ha="center")
        ax.axis("off")
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return out_path

    gt_df = gt_df.sort_values("dice", ascending=False)
    best = gt_df.head(n_cases)
    worst = gt_df.tail(n_cases).iloc[::-1]
    best_ids = set(best["session_id"].tolist())
    selected = pd.concat([best, worst], ignore_index=True)

    loaded_rows: list[dict[str, Any]] = []
    for _, row in selected.iterrows():
        sid = row["session_id"]
        cohort = row["cohort"]
        npz_path = qualitative_dir / cohort / f"{sid}_qualitative.npz"
        if not npz_path.exists():
            npz_candidates = list(qualitative_dir.rglob(f"{sid}_qualitative.npz"))
            npz_path = npz_candidates[0] if npz_candidates else None
        if npz_path is None or not npz_path.exists():
            logger.warning("Missing NPZ for session %s – skipping.", sid)
            continue
        data = np.load(npz_path)
        loaded_rows.append({
            "session_id": sid,
            "dice": float(row["dice"]),
            "image": data["image_slice"],
            "pred": data["pred_slice"],
            "uncertainty": data["uncertainty_slice"],
            "gt": data["gt_slice"] if "gt_slice" in data else None,
            "is_best": sid in best_ids,
        })

    if not loaded_rows:
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.text(0.5, 0.5, "No qualitative data found", transform=ax.transAxes, ha="center")
        ax.axis("off")
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return out_path

    nrows = len(loaded_rows)
    fig, axes = plt.subplots(nrows, 3, figsize=(12, 3.5 * nrows))
    if nrows == 1:
        axes = axes[np.newaxis, :]

    for i, entry in enumerate(loaded_rows):
        img = entry["image"]
        pred = entry["pred"]
        unc = entry["uncertainty"]
        gt = entry["gt"]

        # Image with contours
        ax = axes[i, 0]
        ax.imshow(img.T, cmap="Greys_r", origin="lower")
        pred_edge = _mask_contour(pred)
        if pred_edge.any():
            ov = np.zeros((*pred_edge.shape, 4), dtype=np.float32)
            ov[pred_edge, :] = [1.0, 0.0, 0.0, 0.9]
            ax.imshow(ov.transpose(1, 0, 2), origin="lower")
        if gt is not None:
            gt_edge = _mask_contour(gt)
            if gt_edge.any():
                ov_gt = np.zeros((*gt_edge.shape, 4), dtype=np.float32)
                ov_gt[gt_edge, :] = [0.0, 1.0, 0.0, 0.9]
                ax.imshow(ov_gt.transpose(1, 0, 2), origin="lower")
        ax.axis("off")

        # Pred + GT overlay
        ax = axes[i, 1]
        ax.imshow(pred.T, cmap="Greys_r", origin="lower", vmin=0, vmax=1)
        if gt is not None:
            gt_edge = _mask_contour(gt)
            if gt_edge.any():
                ov_gt = np.zeros((*gt_edge.shape, 4), dtype=np.float32)
                ov_gt[gt_edge, :] = [0.0, 1.0, 0.0, 0.9]
                ax.imshow(ov_gt.transpose(1, 0, 2), origin="lower")
        ax.axis("off")

        # Uncertainty
        ax = axes[i, 2]
        unc_max = float(unc.max()) if unc.max() > 0 else 1.0
        ax.imshow(unc.T, cmap="hot", origin="lower", norm=Normalize(vmin=0, vmax=unc_max))
        ax.axis("off")

        tag = "BEST" if entry["is_best"] else "WORST"
        axes[i, 0].set_ylabel(
            f"[{tag}] {entry['session_id']}\nDice={entry['dice']:.3f}",
            fontsize=8, rotation=0, labelpad=80, va="center",
        )

    col_titles = ["Image + Contours", "Pred + GT Overlay", "Uncertainty"]
    for j, title in enumerate(col_titles):
        axes[0, j].set_title(title, fontsize=10)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# 5. ROC comparison: trust score vs learned reliability
# ---------------------------------------------------------------------------

def generate_roc_comparison_figure(
    per_case_csv_path: Path | str,
    out_path: Path | str,
    failure_dice_threshold: float = 0.1,
) -> Path:
    """Overlay ROC curves for trust_score, reliability_lr, and reliability_mlp."""
    from sklearn.metrics import roc_auc_score, roc_curve as sk_roc_curve

    per_case_csv_path = Path(per_case_csv_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(per_case_csv_path)
    df["has_ground_truth"] = df["has_ground_truth"].astype(str).str.lower().isin(["true", "1"])
    gt_df = df[df["has_ground_truth"]].dropna(subset=["dice", "trust_score"]).copy()

    fig, ax = plt.subplots(figsize=(7, 7))

    if gt_df.empty or gt_df["dice"].nunique() < 2:
        ax.text(0.5, 0.5, "Insufficient data for ROC comparison",
                transform=ax.transAxes, ha="center", fontsize=12)
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return out_path

    labels = (gt_df["dice"] < failure_dice_threshold).astype(int).values

    if len(np.unique(labels)) < 2:
        ax.text(0.5, 0.5, "Only one class present — ROC undefined",
                transform=ax.transAxes, ha="center", fontsize=12)
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return out_path

    trust_scores = 1.0 - gt_df["trust_score"].values
    fpr_t, tpr_t, _ = sk_roc_curve(labels, trust_scores)
    auc_t = roc_auc_score(labels, trust_scores)
    ax.plot(fpr_t, tpr_t, color="#1f77b4", lw=2.0,
            label=f"Trust Score (AUROC = {auc_t:.3f})")

    has_lr = "reliability_score_lr" in gt_df.columns and gt_df["reliability_score_lr"].notna().all()
    has_mlp = "reliability_score_mlp" in gt_df.columns and gt_df["reliability_score_mlp"].notna().all()
    has_rf = "reliability_score_rf" in gt_df.columns and gt_df["reliability_score_rf"].notna().all()

    if has_lr:
        rel_lr = 1.0 - gt_df["reliability_score_lr"].values
        fpr_lr, tpr_lr, _ = sk_roc_curve(labels, rel_lr)
        auc_lr = roc_auc_score(labels, rel_lr)
        ax.plot(fpr_lr, tpr_lr, color="#ff7f0e", lw=2.0, linestyle="--",
                label=f"Reliability LR (AUROC = {auc_lr:.3f})")

    if has_mlp:
        rel_mlp = 1.0 - gt_df["reliability_score_mlp"].values
        fpr_mlp, tpr_mlp, _ = sk_roc_curve(labels, rel_mlp)
        auc_mlp = roc_auc_score(labels, rel_mlp)
        ax.plot(fpr_mlp, tpr_mlp, color="#2ca02c", lw=2.0, linestyle="-.",
                label=f"Reliability MLP (AUROC = {auc_mlp:.3f})")

    if has_rf:
        rel_rf = 1.0 - gt_df["reliability_score_rf"].values
        fpr_rf, tpr_rf, _ = sk_roc_curve(labels, rel_rf)
        auc_rf = roc_auc_score(labels, rel_rf)
        ax.plot(fpr_rf, tpr_rf, color="#d62728", lw=2.0, linestyle="-",
                label=f"Reliability RF (AUROC = {auc_rf:.3f})")

    ax.plot([0, 1], [0, 1], "k:", lw=0.8, alpha=0.5, label="Random")
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title("Failure Detection: ROC Comparison", fontsize=13)
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# 6. Feature importance bar chart
# ---------------------------------------------------------------------------

def generate_feature_importance_figure(
    feature_importances: dict[str, float],
    out_path: Path | str,
    rf_feature_importances: dict[str, float] | None = None,
) -> Path:
    """Horizontal bar chart of feature importances for the reliability module."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    has_rf = rf_feature_importances is not None and len(rf_feature_importances) > 0
    ncols = 2 if has_rf else 1
    fig, axes = plt.subplots(1, ncols, figsize=(8 * ncols, max(4, len(feature_importances) * 0.35)))
    if ncols == 1:
        axes = [axes]

    for ax, imp, title, cmap_name in [
        (axes[0], feature_importances, "Feature Importances (LR)", "viridis"),
    ] + ([
        (axes[1], rf_feature_importances, "Feature Importances (RF)", "plasma"),
    ] if has_rf else []):
        sorted_feat = sorted(imp.items(), key=lambda x: x[1], reverse=True)
        names = [f[0] for f in sorted_feat]
        values = [f[1] for f in sorted_feat]
        colors = getattr(plt.cm, cmap_name)(np.linspace(0.3, 0.9, len(names)))
        bars = ax.barh(range(len(names)), values, color=colors, edgecolor="k", linewidth=0.3)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("Relative Importance", fontsize=10)
        ax.set_title(title, fontsize=12)
        for bar, val in zip(bars, values):
            if val > 0.01:
                ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                        f"{val:.3f}", va="center", fontsize=7)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# 7. Orchestrator
# ---------------------------------------------------------------------------

def generate_all_figures(
    output_dir: Path | str,
    feature_importances: dict[str, float] | None = None,
    rf_feature_importances: dict[str, float] | None = None,
    failure_dice_threshold: float = 0.1,
) -> list[Path]:
    """Generate all publication figures from pipeline outputs.

    Expected layout under *output_dir*:
        reports/per_case_metrics.csv
        qualitative/{cohort}/*.npz

    Saves figures to output_dir/figures/.
    """
    output_dir = Path(output_dir)
    csv_path = output_dir / "reports" / "per_case_metrics.csv"
    qualitative_dir = output_dir / "qualitative"
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    generated: list[Path] = []

    # --- Per-case panels from NPZ files ---
    if csv_path.exists():
        try:
            df = pd.read_csv(csv_path)
            df["has_ground_truth"] = df["has_ground_truth"].astype(str).str.lower().isin(
                ["true", "1"]
            )
        except Exception:
            logger.warning("Could not read per-case CSV at %s", csv_path, exc_info=True)
            df = pd.DataFrame()

        if not df.empty and qualitative_dir.exists():
            panels_dir = figures_dir / "panels"
            panels_dir.mkdir(parents=True, exist_ok=True)
            for _, row in df.iterrows():
                sid = row["session_id"]
                cohort = row["cohort"]
                npz_path = qualitative_dir / cohort / f"{sid}_qualitative.npz"
                if not npz_path.exists():
                    continue
                try:
                    data = np.load(npz_path)
                    img = data["image_slice"]
                    pred = data["pred_slice"]
                    unc = data["uncertainty_slice"]
                    gt = data["gt_slice"] if "gt_slice" in data else None

                    h, w = img.shape
                    image_3d = img[:, :, np.newaxis]
                    pred_3d = pred[:, :, np.newaxis]
                    unc_3d = unc[:, :, np.newaxis]
                    gt_3d = gt[:, :, np.newaxis] if gt is not None else None

                    metrics_dict: dict[str, float | None] = {
                        "dice": row.get("dice"),
                        "trust_score": row.get("trust_score"),
                    }
                    panel_path = generate_case_panel(
                        image=image_3d,
                        pred_mask=pred_3d,
                        uncertainty=unc_3d,
                        out_path=panels_dir / f"{sid}_panel.png",
                        gt_mask=gt_3d,
                        metrics=metrics_dict,
                        session_id=str(sid),
                    )
                    generated.append(panel_path)
                except Exception:
                    logger.warning("Failed to generate panel for %s", sid, exc_info=True)

        # --- Cohort summary ---
        if not df.empty:
            try:
                path = generate_cohort_summary_figure(csv_path, figures_dir / "cohort_summary.png")
                generated.append(path)
            except Exception:
                logger.warning("Failed to generate cohort summary figure.", exc_info=True)

            # --- Uncertainty correlation ---
            try:
                path = generate_uncertainty_correlation_figure(
                    csv_path, figures_dir / "uncertainty_correlation.png"
                )
                generated.append(path)
            except Exception:
                logger.warning("Failed to generate uncertainty correlation figure.", exc_info=True)

            # --- Best / worst gallery ---
            if qualitative_dir.exists():
                try:
                    path = generate_best_worst_gallery(
                        csv_path, qualitative_dir, figures_dir / "best_worst_gallery.png"
                    )
                    generated.append(path)
                except Exception:
                    logger.warning("Failed to generate best/worst gallery.", exc_info=True)

            # --- ROC comparison (trust vs learned reliability) ---
            try:
                path = generate_roc_comparison_figure(
                    csv_path, figures_dir / "roc_comparison.png",
                    failure_dice_threshold=failure_dice_threshold,
                )
                generated.append(path)
            except Exception:
                logger.warning("Failed to generate ROC comparison figure.", exc_info=True)

            # --- Feature importance ---
            if feature_importances:
                try:
                    path = generate_feature_importance_figure(
                        feature_importances, figures_dir / "feature_importances.png",
                        rf_feature_importances=rf_feature_importances,
                    )
                    generated.append(path)
                except Exception:
                    logger.warning("Failed to generate feature importance figure.", exc_info=True)
    else:
        logger.warning("Per-case CSV not found at %s – skipping all summary figures.", csv_path)

    logger.info("Generated %d figures in %s", len(generated), figures_dir)
    return generated
