"""
Learned reliability prediction module for segmentation quality assessment.

Jointly analyzes image, predicted mask, and uncertainty distribution to classify
whether a segmentation is trustworthy or requires manual inspection. Reliability
labels are automatically derived from agreement with reference annotations
(dice >= threshold → reliable), requiring no additional expert labeling.

Two classifiers are trained and reported:
  - Logistic Regression (interpretable baseline)
  - MLP (primary, higher capacity)
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import ndimage, stats
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneOut, StratifiedKFold, cross_val_predict
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

FEATURE_NAMES: list[str] = [
    "entropy_mean",
    "entropy_std",
    "entropy_max",
    "entropy_skew",
    "entropy_frac_above_0.3",
    "entropy_frac_above_0.5",
    "entropy_frac_above_0.7",
    "pred_volume_fraction",
    "pred_n_components",
    "pred_largest_component_frac",
    "pred_surface_voxels_frac",
    "img_mean_inside_pred",
    "img_mean_outside_pred",
    "img_intensity_contrast",
    "trust_score",
    "volume_fraction_std",
    "entropy_weighted_volume",
    "uncertainty_concentration",
]


@dataclass
class ReliabilityFeatures:
    """Extracted features for a single case."""

    session_id: str
    cohort: str
    feature_vector: np.ndarray
    feature_names: list[str] = field(default_factory=lambda: list(FEATURE_NAMES))


@dataclass
class ReliabilityPrediction:
    """Reliability prediction for a single case."""

    session_id: str
    cohort: str
    reliability_score_lr: float
    reliability_score_mlp: float


@dataclass
class ReliabilityModuleResult:
    """Aggregate result from training and evaluating the reliability module."""

    auroc_lr_cv: float
    auroc_mlp_cv: float
    auroc_lr_indian: float | None
    auroc_mlp_indian: float | None
    n_train_cases: int
    n_positive_train: int
    n_indian_cases: int
    feature_importances: dict[str, float]
    predictions: list[ReliabilityPrediction]


def extract_reliability_features(
    image: np.ndarray,
    pred_mask: np.ndarray,
    entropy: np.ndarray,
    trust_score: float,
    volume_fraction_std: float,
    session_id: str = "",
    cohort: str = "",
) -> ReliabilityFeatures:
    """Extract a feature vector from image, predicted mask, and uncertainty map.

    All inputs are 3D numpy arrays of the same spatial shape.
    """
    eps = 1e-10
    n_voxels = max(float(entropy.size), 1.0)

    entropy_flat = entropy.ravel().astype(np.float64)
    entropy_mean = float(np.mean(entropy_flat))
    entropy_std = float(np.std(entropy_flat))
    entropy_max = float(np.max(entropy_flat))
    entropy_skew = float(stats.skew(entropy_flat))
    entropy_frac_03 = float(np.sum(entropy_flat > 0.3)) / n_voxels
    entropy_frac_05 = float(np.sum(entropy_flat > 0.5)) / n_voxels
    entropy_frac_07 = float(np.sum(entropy_flat > 0.7)) / n_voxels

    pred_bool = pred_mask.astype(bool)
    pred_count = float(pred_bool.sum())
    pred_volume_fraction = pred_count / n_voxels

    if pred_count > 0:
        labeled, n_components = ndimage.label(pred_bool)
        component_sizes = ndimage.sum(pred_bool, labeled, range(1, n_components + 1))
        largest_frac = float(np.max(component_sizes)) / pred_count if n_components > 0 else 0.0
        eroded = ndimage.binary_erosion(pred_bool)
        surface = pred_bool.astype(int) - eroded.astype(int)
        surface_frac = float(surface.sum()) / pred_count
    else:
        n_components = 0
        largest_frac = 0.0
        surface_frac = 0.0

    image_float = image.astype(np.float64)
    if pred_count > 0:
        img_inside = float(np.mean(image_float[pred_bool]))
    else:
        img_inside = 0.0
    bg_mask = ~pred_bool
    if bg_mask.sum() > 0:
        img_outside = float(np.mean(image_float[bg_mask]))
    else:
        img_outside = 0.0
    intensity_contrast = (img_inside - img_outside) / (abs(img_outside) + eps)

    entropy_weighted_vol = float(np.sum(entropy_flat * pred_mask.ravel().astype(np.float64))) / n_voxels

    if entropy_mean > eps:
        high_unc_mask = entropy > (entropy_mean + entropy_std)
        high_unc_count = float(high_unc_mask.sum())
        if high_unc_count > 1:
            coords = np.argwhere(high_unc_mask).astype(np.float64)
            centroid = coords.mean(axis=0)
            dists = np.linalg.norm(coords - centroid, axis=1)
            spatial_extent = np.array(entropy.shape, dtype=np.float64)
            max_dist = float(np.linalg.norm(spatial_extent))
            uncertainty_concentration = 1.0 - (float(np.mean(dists)) / (max_dist + eps))
        else:
            uncertainty_concentration = 1.0
    else:
        uncertainty_concentration = 1.0

    vec = np.array([
        entropy_mean,
        entropy_std,
        entropy_max,
        entropy_skew,
        entropy_frac_03,
        entropy_frac_05,
        entropy_frac_07,
        pred_volume_fraction,
        float(n_components),
        largest_frac,
        surface_frac,
        img_inside,
        img_outside,
        intensity_contrast,
        trust_score,
        volume_fraction_std,
        entropy_weighted_vol,
        uncertainty_concentration,
    ], dtype=np.float64)

    vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)

    return ReliabilityFeatures(
        session_id=session_id,
        cohort=cohort,
        feature_vector=vec,
    )


def _safe_auroc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    if len(np.unique(labels)) < 2:
        return None
    return float(roc_auc_score(labels, scores))


def train_and_evaluate_reliability(
    features_adam: list[ReliabilityFeatures],
    dice_scores_adam: list[float],
    features_indian: list[ReliabilityFeatures] | None = None,
    dice_scores_indian: list[float] | None = None,
    features_inference_only: list[ReliabilityFeatures] | None = None,
    failure_dice_threshold: float = 0.1,
) -> ReliabilityModuleResult:
    """Train reliability classifiers on ADAM data and evaluate.

    Uses cross-validated predictions on ADAM for honest AUROC,
    then optionally evaluates on Indian CAR data (cross-domain with GT),
    and predicts on inference-only cases (no GT, e.g. Indian NCAR).
    """
    X = np.array([f.feature_vector for f in features_adam])
    labels = np.array([0 if d < failure_dice_threshold else 1 for d in dice_scores_adam])
    n_pos = int(labels.sum())
    n_neg = len(labels) - n_pos

    logger.info(
        "Reliability module: %d ADAM cases (%d reliable, %d failures)",
        len(labels), n_pos, n_neg,
    )

    if n_pos < 2 or n_neg < 2:
        logger.warning(
            "Insufficient class balance for reliability training "
            "(pos=%d, neg=%d). Returning trivial result.", n_pos, n_neg,
        )
        predictions = [
            ReliabilityPrediction(
                session_id=f.session_id, cohort=f.cohort,
                reliability_score_lr=0.5, reliability_score_mlp=0.5,
            )
            for f in features_adam
        ]
        if features_indian:
            predictions.extend([
                ReliabilityPrediction(
                    session_id=f.session_id, cohort=f.cohort,
                    reliability_score_lr=0.5, reliability_score_mlp=0.5,
                )
                for f in features_indian
            ])
        if features_inference_only:
            predictions.extend([
                ReliabilityPrediction(
                    session_id=f.session_id, cohort=f.cohort,
                    reliability_score_lr=0.5, reliability_score_mlp=0.5,
                )
                for f in features_inference_only
            ])
        return ReliabilityModuleResult(
            auroc_lr_cv=0.5,
            auroc_mlp_cv=0.5,
            auroc_lr_indian=None,
            auroc_mlp_indian=None,
            n_train_cases=len(labels),
            n_positive_train=n_pos,
            n_indian_cases=len(features_indian) if features_indian else 0,
            feature_importances={},
            predictions=predictions,
        )

    n_splits = min(5, n_pos, n_neg)
    if n_splits < 2:
        cv = LeaveOneOut()
    else:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=2026)

    lr_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            C=1.0, class_weight="balanced", max_iter=2000, random_state=2026,
        )),
    ])

    mlp_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            alpha=1e-3,
            max_iter=2000,
            early_stopping=True,
            validation_fraction=0.15,
            random_state=2026,
        )),
    ])

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)

        lr_probs_cv = cross_val_predict(lr_pipe, X, labels, cv=cv, method="predict_proba")[:, 1]
        mlp_probs_cv = cross_val_predict(mlp_pipe, X, labels, cv=cv, method="predict_proba")[:, 1]

    auroc_lr_cv = _safe_auroc(labels, lr_probs_cv) or 0.5
    auroc_mlp_cv = _safe_auroc(labels, mlp_probs_cv) or 0.5

    logger.info("Reliability CV AUROC — LR: %.4f, MLP: %.4f", auroc_lr_cv, auroc_mlp_cv)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        lr_pipe.fit(X, labels)
        mlp_pipe.fit(X, labels)

    lr_coefs = lr_pipe.named_steps["clf"].coef_.ravel()
    scaler = lr_pipe.named_steps["scaler"]
    scaled_coefs = lr_coefs * scaler.scale_
    abs_coefs = np.abs(scaled_coefs)
    if abs_coefs.sum() > 0:
        importance = abs_coefs / abs_coefs.sum()
    else:
        importance = np.ones_like(abs_coefs) / len(abs_coefs)
    feature_importances = {
        name: round(float(imp), 4)
        for name, imp in zip(FEATURE_NAMES, importance)
    }

    predictions: list[ReliabilityPrediction] = []
    for i, feat in enumerate(features_adam):
        predictions.append(ReliabilityPrediction(
            session_id=feat.session_id,
            cohort=feat.cohort,
            reliability_score_lr=float(lr_probs_cv[i]),
            reliability_score_mlp=float(mlp_probs_cv[i]),
        ))

    auroc_lr_indian = None
    auroc_mlp_indian = None

    if features_indian and dice_scores_indian:
        X_indian = np.array([f.feature_vector for f in features_indian])
        labels_indian = np.array([
            0 if d < failure_dice_threshold else 1 for d in dice_scores_indian
        ])

        lr_probs_indian = lr_pipe.predict_proba(X_indian)[:, 1]
        mlp_probs_indian = mlp_pipe.predict_proba(X_indian)[:, 1]

        auroc_lr_indian = _safe_auroc(labels_indian, lr_probs_indian)
        auroc_mlp_indian = _safe_auroc(labels_indian, mlp_probs_indian)

        if auroc_lr_indian is not None:
            logger.info(
                "Reliability Indian AUROC — LR: %.4f, MLP: %.4f",
                auroc_lr_indian, auroc_mlp_indian or 0.0,
            )

        for i, feat in enumerate(features_indian):
            predictions.append(ReliabilityPrediction(
                session_id=feat.session_id,
                cohort=feat.cohort,
                reliability_score_lr=float(lr_probs_indian[i]),
                reliability_score_mlp=float(mlp_probs_indian[i]),
            ))

    if features_inference_only:
        X_infer = np.array([f.feature_vector for f in features_inference_only])
        lr_probs_infer = lr_pipe.predict_proba(X_infer)[:, 1]
        mlp_probs_infer = mlp_pipe.predict_proba(X_infer)[:, 1]
        for i, feat in enumerate(features_inference_only):
            predictions.append(ReliabilityPrediction(
                session_id=feat.session_id,
                cohort=feat.cohort,
                reliability_score_lr=float(lr_probs_infer[i]),
                reliability_score_mlp=float(mlp_probs_infer[i]),
            ))

    return ReliabilityModuleResult(
        auroc_lr_cv=round(auroc_lr_cv, 4),
        auroc_mlp_cv=round(auroc_mlp_cv, 4),
        auroc_lr_indian=round(auroc_lr_indian, 4) if auroc_lr_indian is not None else None,
        auroc_mlp_indian=round(auroc_mlp_indian, 4) if auroc_mlp_indian is not None else None,
        n_train_cases=len(labels),
        n_positive_train=n_pos,
        n_indian_cases=len(features_indian) if features_indian else 0,
        feature_importances=feature_importances,
        predictions=predictions,
    )
