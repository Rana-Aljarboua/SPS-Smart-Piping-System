from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline

from train_baseline import LABEL_COLUMN, load_experiment_data


PROJECT_DIR = Path(__file__).resolve().parents[1]
REPORT_FILE = PROJECT_DIR / "reports" / "sps_threshold_calibration.txt"
RANDOM_STATE = 42
DEFAULT_THRESHOLD = 0.50


def build_feature_matrix(
    rows: list[dict[str, float | None]], feature_names: list[str]
) -> np.ndarray:
    return np.asarray(
        [
            [
                row[feature] if row[feature] is not None else np.nan
                for feature in feature_names
            ]
            for row in rows
        ],
        dtype=float,
    )


def create_model() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("variance", VarianceThreshold()),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=300,
                    class_weight="balanced_subsample",
                    random_state=RANDOM_STATE,
                    n_jobs=1,
                ),
            ),
        ]
    )


def calculate_metrics(
    true_labels: np.ndarray, predicted_labels: np.ndarray
) -> dict[str, object]:
    matrix = confusion_matrix(true_labels, predicted_labels, labels=[0, 1])
    return {
        "precision": float(
            precision_score(true_labels, predicted_labels, pos_label=1, zero_division=0)
        ),
        "recall": float(
            recall_score(true_labels, predicted_labels, pos_label=1, zero_division=0)
        ),
        "f1": float(
            f1_score(true_labels, predicted_labels, pos_label=1, zero_division=0)
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(true_labels, predicted_labels)
        ),
        "confusion_matrix": matrix.tolist(),
    }


def choose_threshold(
    validation_labels: np.ndarray, validation_scores: np.ndarray
) -> tuple[float, float]:
    if set(validation_labels.tolist()) != {0, 1}:
        raise ValueError("Validation workbook must contain both Labels classes")
    if validation_scores.size != validation_labels.size:
        raise ValueError("Validation score and label counts do not match")
    if not np.isfinite(validation_scores).all():
        raise ValueError("Validation probabilities contain non-finite values")

    candidates = np.unique(
        np.concatenate(
            (
                np.asarray([0.0, DEFAULT_THRESHOLD]),
                validation_scores,
                np.asarray([np.nextafter(validation_scores.max(), np.inf)]),
            )
        )
    )
    scored_candidates: list[tuple[float, float]] = []
    for threshold in candidates:
        predictions = (validation_scores >= threshold).astype(int)
        score = float(
            f1_score(
                validation_labels,
                predictions,
                pos_label=1,
                zero_division=0,
            )
        )
        scored_candidates.append((score, float(threshold)))

    best_f1 = max(score for score, _ in scored_candidates)
    best_threshold = max(
        threshold for score, threshold in scored_candidates if score == best_f1
    )
    return best_threshold, best_f1


def format_metrics(
    title: str, threshold: float | None, metrics: dict[str, object]
) -> list[str]:
    matrix = metrics["confusion_matrix"]
    if not isinstance(matrix, list):
        raise TypeError("Confusion matrix has an unexpected type")
    threshold_description = (
        f"threshold={threshold:.12g}" if threshold is not None else "fold-specific thresholds"
    )
    return [
        f"  {title} ({threshold_description}):",
        f"    Precision (Labels=1): {metrics['precision']:.4f}",
        f"    Recall (Labels=1): {metrics['recall']:.4f}",
        f"    F1 (Labels=1): {metrics['f1']:.4f}",
        f"    Balanced accuracy: {metrics['balanced_accuracy']:.4f}",
        "    Confusion matrix (rows=true, columns=predicted; order [0, 1]):",
        f"      [[{matrix[0][0]}, {matrix[0][1]}],",
        f"       [{matrix[1][0]}, {matrix[1][1]}]]",
    ]


def make_report() -> str:
    feature_names, records = load_experiment_data()
    if LABEL_COLUMN in feature_names:
        raise ValueError(f"{LABEL_COLUMN} must not appear in model features")

    matrices: dict[str, np.ndarray] = {}
    targets: dict[str, np.ndarray] = {}
    class_counts: dict[str, Counter[int]] = {}
    clean_names: list[str] = []
    leakage_names: list[str] = []
    for record in records:
        name = f"{record['group']}/{record['name']}"
        rows = record["rows"]
        labels = record["labels"]
        if not isinstance(rows, list) or not isinstance(labels, list):
            raise ValueError(f"Malformed workbook data for {name}")
        if len(rows) != len(labels):
            raise ValueError(f"Feature/label length mismatch for {name}")
        counts = Counter(labels)
        if set(counts) - {0, 1}:
            raise ValueError(f"{name} has non-binary target labels: {dict(counts)}")
        matrices[name] = build_feature_matrix(rows, feature_names)
        targets[name] = np.asarray(labels, dtype=int)
        class_counts[name] = counts
        if record["group"] == "Clean":
            clean_names.append(name)
        elif record["group"] == "Leakage":
            leakage_names.append(name)

    if len(clean_names) != 1 or len(leakage_names) != 3:
        raise ValueError(
            "Expected one Clean and three Leakage workbooks; found "
            f"{len(clean_names)} and {len(leakage_names)}"
        )
    clean_name = clean_names[0]
    if set(class_counts[clean_name]) != {0}:
        raise ValueError(
            f"Clean workbook must contain only class 0, found {dict(class_counts[clean_name])}"
        )
    for name in leakage_names:
        if set(class_counts[name]) != {0, 1}:
            raise ValueError(
                f"{name} must contain both target classes, found {dict(class_counts[name])}"
            )

    lines = [
        "SPS grouped-workbook threshold calibration",
        "==========================================",
        "Target: Labels (0=normal sample, 1=anomalous event sample); not leak-only.",
        "Model features: sensor measurements only; Labels and workbook identifiers excluded.",
        "Protocol: each Leakage workbook is test once; a different Leakage workbook is validation.",
        "Training for each rotation: Clean workbook + the remaining Leakage workbook.",
        "No row from validation or test is used to fit the model or preprocessing.",
        "Model: same RandomForest pipeline as grouped baseline; 300 trees,",
        "class_weight=balanced_subsample, random_state=42, n_jobs=1.",
        "Threshold: maximize validation F1 over observed validation probabilities and",
        "boundary candidates; ties choose the highest threshold. Test data do not select it.",
        "Default predictions use the classifier's standard 0.50 decision rule.",
        "",
        "Workbook class counts:",
    ]
    for name in [clean_name, *leakage_names]:
        counts = class_counts[name]
        lines.append(
            f"- {name}: rows={len(targets[name])}, "
            f"Labels=0:{counts[0]}, Labels=1:{counts[1]}"
        )

    lines.extend(["", "Fold results:"])
    pooled_default_true: list[int] = []
    pooled_default_pred: list[int] = []
    pooled_calibrated_true: list[int] = []
    pooled_calibrated_pred: list[int] = []
    for fold_index, test_name in enumerate(leakage_names):
        validation_name = leakage_names[(fold_index + 1) % len(leakage_names)]
        training_leakage_name = next(
            name for name in leakage_names if name not in {test_name, validation_name}
        )
        x_train = np.concatenate(
            [matrices[clean_name], matrices[training_leakage_name]], axis=0
        )
        y_train = np.concatenate(
            [targets[clean_name], targets[training_leakage_name]], axis=0
        )
        x_validation = matrices[validation_name]
        y_validation = targets[validation_name]
        x_test = matrices[test_name]
        y_test = targets[test_name]

        if set(y_train.tolist()) != {0, 1}:
            raise ValueError(f"Training split for {test_name} lacks a target class")
        if set(y_validation.tolist()) != {0, 1}:
            raise ValueError(f"Validation split {validation_name} lacks a target class")
        if set(y_test.tolist()) != {0, 1}:
            raise ValueError(f"Test split {test_name} lacks a target class")

        model = create_model()
        model.fit(x_train, y_train)
        classifier = model.named_steps["classifier"]
        positive_class_index = int(np.flatnonzero(classifier.classes_ == 1)[0])
        validation_scores = model.predict_proba(x_validation)[:, positive_class_index]
        threshold, validation_f1 = choose_threshold(y_validation, validation_scores)

        default_predictions = model.predict(x_test)
        test_scores = model.predict_proba(x_test)[:, positive_class_index]
        calibrated_predictions = (test_scores >= threshold).astype(int)
        default_metrics = calculate_metrics(y_test, default_predictions)
        calibrated_metrics = calculate_metrics(y_test, calibrated_predictions)
        removed_features = [
            feature
            for feature, keep in zip(
                feature_names, model.named_steps["variance"].get_support()
            )
            if not keep
        ]

        train_counts = Counter(y_train.tolist())
        validation_counts = class_counts[validation_name]
        test_counts = class_counts[test_name]
        lines.extend(
            [
                "",
                f"[Test: {test_name}]",
                f"  Training workbooks: {clean_name}, {training_leakage_name}",
                f"  Validation workbook: {validation_name} "
                f"(rows={len(y_validation)}, Labels=0:{validation_counts[0]}, "
                f"Labels=1:{validation_counts[1]})",
                f"  Test rows={len(y_test)} (Labels=0:{test_counts[0]}, "
                f"Labels=1:{test_counts[1]})",
                f"  Training rows={len(y_train)} (Labels=0:{train_counts[0]}, "
                f"Labels=1:{train_counts[1]})",
                f"  Training-only constant features removed: "
                f"{', '.join(removed_features) if removed_features else 'none'}",
                f"  Selected threshold={threshold:.12g}; validation F1={validation_f1:.4f}",
            ]
        )
        lines.extend(format_metrics("Default", DEFAULT_THRESHOLD, default_metrics))
        lines.extend(format_metrics("Calibrated", threshold, calibrated_metrics))

        pooled_default_true.extend(y_test.tolist())
        pooled_default_pred.extend(default_predictions.tolist())
        pooled_calibrated_true.extend(y_test.tolist())
        pooled_calibrated_pred.extend(calibrated_predictions.tolist())

    pooled_default_metrics = calculate_metrics(
        np.asarray(pooled_default_true, dtype=int),
        np.asarray(pooled_default_pred, dtype=int),
    )
    pooled_calibrated_metrics = calculate_metrics(
        np.asarray(pooled_calibrated_true, dtype=int),
        np.asarray(pooled_calibrated_pred, dtype=int),
    )
    lines.extend(
        [
            "",
            "Pooled held-out Leakage rows (each workbook appears as test once):",
        ]
    )
    lines.extend(
        format_metrics("Default", DEFAULT_THRESHOLD, pooled_default_metrics)
    )
    lines.extend(
        format_metrics("Calibrated", None, pooled_calibrated_metrics)
    )
    lines.extend(
        [
            "",
            "Interpretation and limitations:",
            "- Calibrated thresholds are selected independently per fold from that fold's",
            "  validation workbook. Pooled calibrated metrics aggregate test predictions made",
            "  with three fold-specific thresholds; there is no single pooled threshold.",
            "- The model is freshly fitted for each rotation, but the saved baseline artifact",
            "  is not loaded, overwritten, or modified.",
            "- This tests a threshold effect only. It cannot establish that threshold choice",
            "  caused or resolves the cross-workbook distribution shift.",
            "- Per-row timestamps are unavailable; rows may be temporally dependent.",
            "- Labels=1 denotes an anomalous event sample per the project-provided paper",
            "  excerpt, not necessarily a water leak. These are not leak-detection metrics.",
            "- The available workbook runs are few and this evaluation does not establish",
            "  performance on residential SPS sensors.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    try:
        report = make_report()
        REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
        with REPORT_FILE.open("x", encoding="utf-8", newline="\n") as output:
            output.write(report)
    except (
        OSError,
        KeyError,
        ValueError,
        TypeError,
        zipfile.BadZipFile,
        ET.ParseError,
    ) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    print(report, end="")
    print(f"Report saved to: {REPORT_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
