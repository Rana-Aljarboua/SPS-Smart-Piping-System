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
REPORT_FILE = PROJECT_DIR / "reports" / "sps_grouped_evaluation.txt"
RANDOM_STATE = 42


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


def format_metrics(
    true_labels: np.ndarray, predicted_labels: np.ndarray
) -> tuple[list[str], np.ndarray]:
    matrix = confusion_matrix(true_labels, predicted_labels, labels=[0, 1])
    lines = [
        f"  Precision (anomalous=1): "
        f"{precision_score(true_labels, predicted_labels, pos_label=1, zero_division=0):.4f}",
        f"  Recall (anomalous=1): "
        f"{recall_score(true_labels, predicted_labels, pos_label=1, zero_division=0):.4f}",
        f"  F1 (anomalous=1): "
        f"{f1_score(true_labels, predicted_labels, pos_label=1, zero_division=0):.4f}",
        f"  Balanced accuracy: {balanced_accuracy_score(true_labels, predicted_labels):.4f}",
        "  Confusion matrix (rows=true, columns=predicted; order [0, 1]):",
        f"    [[{matrix[0, 0]}, {matrix[0, 1]}],",
        f"     [{matrix[1, 0]}, {matrix[1, 1]}]]",
    ]
    return lines, matrix


def make_report() -> str:
    feature_names, records = load_experiment_data()
    if LABEL_COLUMN in feature_names:
        raise ValueError(f"{LABEL_COLUMN} must not appear in model features")

    clean_records = [record for record in records if record["group"] == "Clean"]
    leakage_records = [record for record in records if record["group"] == "Leakage"]
    if len(clean_records) != 1:
        raise ValueError(f"Expected exactly one Clean workbook, found {len(clean_records)}")
    if len(leakage_records) != 3:
        raise ValueError(f"Expected exactly three Leakage workbooks, found {len(leakage_records)}")

    all_records = clean_records + leakage_records
    matrices: dict[str, np.ndarray] = {}
    targets: dict[str, np.ndarray] = {}
    class_counts: dict[str, Counter[int]] = {}
    for record in all_records:
        name = f"{record['group']}/{record['name']}"
        rows = record["rows"]
        labels = record["labels"]
        if not isinstance(rows, list) or not isinstance(labels, list):
            raise ValueError(f"Malformed workbook data for {name}")
        if len(rows) != len(labels):
            raise ValueError(f"Feature/label length mismatch for {name}")
        counts = Counter(labels)
        if not counts or any(label not in (0, 1) for label in counts):
            raise ValueError(f"{name} must contain binary labels 0 and 1 only")
        matrices[name] = build_feature_matrix(rows, feature_names)
        targets[name] = np.asarray(labels, dtype=int)
        class_counts[name] = counts

    clean_name = next(name for name in matrices if name.startswith("Clean/"))
    clean_classes = set(class_counts[clean_name])
    if clean_classes != {0}:
        raise ValueError(
            "Expected the Clean workbook to contain only class 0 for this grouped strategy; "
            f"found {dict(class_counts[clean_name])}"
        )
    for name in matrices:
        if name.startswith("Leakage/") and set(class_counts[name]) != {0, 1}:
            raise ValueError(
                f"Grouped holdout requires both classes in each Leakage workbook; "
                f"{name} has {dict(class_counts[name])}"
            )

    lines = [
        "SPS grouped workbook evaluation",
        "=" * 34,
        "Target: Labels (0=normal sample, 1=anomalous event sample).",
        "Features: sensor measurements only; Labels, workbook/file names, paths, and folder names excluded.",
        "Strategy: leave one entire Leakage workbook out per fold; Clean workbook remains in training.",
        f"RandomForest: n_estimators=300, class_weight=balanced_subsample, random_state={RANDOM_STATE}.",
        "",
        "Class counts by workbook:",
    ]
    for name in matrices:
        counts = class_counts[name]
        lines.append(
            f"- {name}: rows={len(targets[name])}, class 0={counts[0]}, class 1={counts[1]}"
        )

    lines.extend(["", "Grouped fold results:"])
    out_of_fold_true: list[int] = []
    out_of_fold_predicted: list[int] = []
    for held_out_name in matrices:
        if not held_out_name.startswith("Leakage/"):
            continue

        training_names = [
            name for name in matrices if name != held_out_name
        ]
        x_train = np.concatenate([matrices[name] for name in training_names], axis=0)
        y_train = np.concatenate([targets[name] for name in training_names], axis=0)
        x_test = matrices[held_out_name]
        y_test = targets[held_out_name]
        train_classes = set(y_train.tolist())
        test_classes = set(y_test.tolist())
        if train_classes != {0, 1} or test_classes != {0, 1}:
            raise ValueError(
                f"Invalid grouped split for {held_out_name}: "
                f"train classes={sorted(train_classes)}, test classes={sorted(test_classes)}"
            )

        model = create_model()
        model.fit(x_train, y_train)
        predicted = model.predict(x_test)
        retained = model.named_steps["variance"].get_support()
        removed_features = [
            feature for feature, keep in zip(feature_names, retained) if not keep
        ]

        lines.extend(
            [
                f"- Held-out workbook: {held_out_name}",
                f"  Training workbooks: {', '.join(training_names)}",
                f"  Train rows={len(y_train)} (class 0={int(np.count_nonzero(y_train == 0))}, "
                f"class 1={int(np.count_nonzero(y_train == 1))})",
                f"  Test rows={len(y_test)} (class 0={int(np.count_nonzero(y_test == 0))}, "
                f"class 1={int(np.count_nonzero(y_test == 1))})",
                f"  Constant features removed using training data only: "
                f"{', '.join(removed_features) if removed_features else 'none'}",
            ]
        )
        metric_lines, _ = format_metrics(y_test, predicted)
        lines.extend(metric_lines)
        out_of_fold_true.extend(y_test.tolist())
        out_of_fold_predicted.extend(predicted.tolist())

    combined_true = np.asarray(out_of_fold_true, dtype=int)
    combined_predicted = np.asarray(out_of_fold_predicted, dtype=int)
    if combined_true.size == 0 or set(combined_true.tolist()) != {0, 1}:
        raise ValueError("Grouped held-out workbooks do not provide both target classes")

    lines.append("")
    lines.append("Combined out-of-fold results (each Leakage workbook predicted only when held out):")
    combined_lines, _ = format_metrics(combined_true, combined_predicted)
    lines.extend(combined_lines)
    lines.extend(
        [
            "",
            "Limitations:",
            "- Every row from a workbook stays together. The Clean workbook is always in training "
            "because it is the only Clean workbook; it is never independently tested.",
            "- Test results cover the three held-out Leakage scenario workbooks only, not a held-out "
            "Clean run or a general population of runs.",
            "- Per-row timestamps are unavailable. Within-workbook rows may be temporally dependent, "
            "so workbook grouping does not prove independence between samples or leakage-free chronology.",
            "- Labels=1 means anomalous event sample, not necessarily a water leak; these metrics are "
            "not leak-detection accuracy.",
            "- This evaluates generalization across the available workbook scenarios only; the dataset is "
            "small and does not establish performance on SPS residential sensors.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    try:
        report = make_report()
        REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
        with REPORT_FILE.open("x", encoding="utf-8") as output:
            output.write(report)
    except (OSError, KeyError, ValueError, TypeError, zipfile.BadZipFile, ET.ParseError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    print(report, end="")
    print(f"Report saved to: {REPORT_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
