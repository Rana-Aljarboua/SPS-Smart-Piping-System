from __future__ import annotations

import re
import sys
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel, VarianceThreshold
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
GROUPED_REPORT = PROJECT_DIR / "reports" / "sps_grouped_evaluation.txt"
SHIFT_REPORT = PROJECT_DIR / "reports" / "sps_group_shift_diagnosis.txt"
REPORT_FILE = PROJECT_DIR / "reports" / "sps_feature_selection_results.txt"
RANDOM_STATE = 42
N_ESTIMATORS = 300


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


def create_pipeline(select_features: bool) -> Pipeline:
    steps: list[tuple[str, object]] = [
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("variance", VarianceThreshold()),
    ]
    if select_features:
        steps.append(
            (
                "selector",
                SelectFromModel(
                    estimator=RandomForestClassifier(
                        n_estimators=N_ESTIMATORS,
                        class_weight="balanced_subsample",
                        random_state=RANDOM_STATE,
                        n_jobs=1,
                    ),
                    threshold="median",
                ),
            )
        )
    steps.append(
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=N_ESTIMATORS,
                class_weight="balanced_subsample",
                random_state=RANDOM_STATE,
                n_jobs=1,
            ),
        )
    )
    return Pipeline(steps=steps)


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
        "f1": float(f1_score(true_labels, predicted_labels, pos_label=1, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(true_labels, predicted_labels)),
        "confusion_matrix": matrix.tolist(),
    }


def format_metrics(metrics: dict[str, object]) -> list[str]:
    matrix = metrics["confusion_matrix"]
    assert isinstance(matrix, list)
    return [
        f"  Precision (anomalous=1): {metrics['precision']:.4f}",
        f"  Recall (anomalous=1): {metrics['recall']:.4f}",
        f"  F1 (anomalous=1): {metrics['f1']:.4f}",
        f"  Balanced accuracy: {metrics['balanced_accuracy']:.4f}",
        "  Confusion matrix (rows=true, columns=predicted; order [0, 1]):",
        f"    [[{matrix[0][0]}, {matrix[0][1]}],",
        f"     [{matrix[1][0]}, {matrix[1][1]}]]",
    ]


def parse_grouped_baseline(path: Path) -> dict[str, dict[str, object]]:
    text = path.read_text(encoding="utf-8")
    results: dict[str, dict[str, object]] = {}
    current: str | None = None
    matrix_lines: list[str] = []
    for line in text.splitlines():
        match = re.match(r"- Held-out workbook: (.+)", line)
        if match:
            current = match.group(1).strip()
            results[current] = {}
            matrix_lines = []
            continue
        if current is None:
            continue
        metric_match = re.match(
            r"\s+(Precision|Recall|F1|Balanced accuracy)(?: \(anomalous=1\))?\s*:\s*([0-9.]+)",
            line,
        )
        if metric_match:
            metric_name = metric_match.group(1)
            key = {
                "Precision": "precision",
                "Recall": "recall",
                "F1": "f1",
                "Balanced accuracy": "balanced_accuracy",
            }[metric_name]
            results[current][key] = float(metric_match.group(2))
        if line.strip().startswith("[["):
            matrix_lines.append(line.strip())
        elif matrix_lines and line.strip().startswith("["):
            matrix_lines.append(line.strip())
            numbers = [int(value) for value in re.findall(r"\d+", " ".join(matrix_lines))]
            if len(numbers) == 4:
                results[current]["confusion_matrix"] = [numbers[:2], numbers[2:]]
                current = None
                matrix_lines = []
    if not results or any(
        not {"precision", "recall", "f1", "balanced_accuracy", "confusion_matrix"}
        <= result.keys()
        for result in results.values()
    ):
        raise ValueError(f"Could not parse grouped baseline fold metrics from {path}")
    return results


def parse_shift_summary(path: Path) -> dict[str, list[str]]:
    text = path.read_text(encoding="utf-8")
    summaries: dict[str, list[str]] = {}
    current: str | None = None
    for line in text.splitlines():
        match = re.match(r"\[(Leakage/[^]]+)\]$", line)
        if match:
            current = match.group(1)
            summaries.setdefault(current, [])
            continue
        if current is not None and line.startswith("      "):
            summaries[current].append(line.strip())
    return summaries


def make_report() -> str:
    if not GROUPED_REPORT.is_file():
        raise FileNotFoundError(f"Grouped baseline report not found: {GROUPED_REPORT}")
    if not SHIFT_REPORT.is_file():
        raise FileNotFoundError(f"Group-shift report not found: {SHIFT_REPORT}")

    baseline_results = parse_grouped_baseline(GROUPED_REPORT)
    shift_summary = parse_shift_summary(SHIFT_REPORT)
    feature_names, records = load_experiment_data()
    if LABEL_COLUMN in feature_names:
        raise ValueError(f"{LABEL_COLUMN} must not be a sensor feature")

    matrices: dict[str, np.ndarray] = {}
    targets: dict[str, np.ndarray] = {}
    class_counts: dict[str, Counter[int]] = {}
    for record in records:
        name = f"{record['group']}/{record['name']}"
        rows = record["rows"]
        labels = record["labels"]
        if not isinstance(rows, list) or not isinstance(labels, list):
            raise ValueError(f"Malformed workbook data for {name}")
        if len(rows) != len(labels):
            raise ValueError(f"Feature/label length mismatch for {name}")
        matrices[name] = build_feature_matrix(rows, feature_names)
        targets[name] = np.asarray(labels, dtype=int)
        class_counts[name] = Counter(labels)

    clean_names = [name for name in matrices if name.startswith("Clean/")]
    leakage_names = [name for name in matrices if name.startswith("Leakage/")]
    if len(clean_names) != 1 or len(leakage_names) != 3:
        raise ValueError(
            f"Expected one Clean and three Leakage workbooks; found "
            f"{len(clean_names)} and {len(leakage_names)}"
        )
    if set(leakage_names) != set(baseline_results):
        raise ValueError(
            "Workbook names in grouped baseline report do not match current dataset"
        )

    results: dict[str, dict[str, dict[str, object]]] = {
        "all_features": {},
        "selected_features": {},
    }
    selected_feature_names: dict[str, list[str]] = {}

    for held_out in leakage_names:
        training_names = [name for name in matrices if name != held_out]
        x_train = np.concatenate([matrices[name] for name in training_names], axis=0)
        y_train = np.concatenate([targets[name] for name in training_names], axis=0)
        x_test = matrices[held_out]
        y_test = targets[held_out]
        if set(y_train.tolist()) != {0, 1} or set(y_test.tolist()) != {0, 1}:
            raise ValueError(f"Both classes required in training and held-out {held_out}")

        baseline_model = create_pipeline(select_features=False)
        baseline_model.fit(x_train, y_train)
        baseline_predictions = baseline_model.predict(x_test)
        baseline_metrics = calculate_metrics(y_test, baseline_predictions)
        baseline_metrics["feature_count"] = int(
            np.count_nonzero(baseline_model.named_steps["variance"].get_support())
        )
        results["all_features"][held_out] = baseline_metrics

        selected_model = create_pipeline(select_features=True)
        selected_model.fit(x_train, y_train)
        selected_predictions = selected_model.predict(x_test)
        selected_metrics = calculate_metrics(y_test, selected_predictions)
        selector = selected_model.named_steps["selector"]
        retained_after_variance = selected_model.named_steps["variance"].get_support()
        selected_after_variance = selector.get_support()
        names_after_variance = [
            name for name, retained in zip(feature_names, retained_after_variance) if retained
        ]
        if len(names_after_variance) != len(selected_after_variance):
            raise ValueError(f"Feature selector dimension mismatch in fold {held_out}")
        chosen = [
            name
            for name, keep in zip(names_after_variance, selected_after_variance)
            if keep
        ]
        if not chosen:
            raise ValueError(f"Feature selection retained no features in fold {held_out}")
        selected_feature_names[held_out] = chosen
        selected_metrics["feature_count"] = len(chosen)
        results["selected_features"][held_out] = selected_metrics

    lines = [
        "SPS grouped feature-selection experiment",
        "=" * 44,
        "Protocol: leave one entire Leakage workbook out; Clean remains in training.",
        "Target: Labels (0=normal sample, 1=anomalous event sample).",
        "X contains sensor readings only; Labels, file names, paths, and folder names are excluded.",
        "Feature selection: training-fold RandomForest importances with median threshold, inside Pipeline.",
        "Selection, imputation, constant filtering, and classifier fit use training workbooks only.",
        "RandomForest: 300 trees, class_weight=balanced_subsample, random_state=42.",
        "",
        "Fold results: all features vs training-selected subset",
    ]
    for held_out in leakage_names:
        counts = class_counts[held_out]
        baseline = results["all_features"][held_out]
        selected = results["selected_features"][held_out]
        baseline_report = baseline_results[held_out]
        baseline_matches = all(
            round(float(baseline[key]), 4) == round(float(baseline_report[key]), 4)
            for key in ("precision", "recall", "f1", "balanced_accuracy")
        ) and baseline["confusion_matrix"] == baseline_report["confusion_matrix"]
        lines.extend(
            [
                f"\n[{held_out}] held-out rows={len(targets[held_out])}, "
                f"class 0={counts[0]}, class 1={counts[1]}",
                f"  Same-fold all-feature rerun matches existing report: "
                f"{'yes' if baseline_matches else 'NO'}",
                f"  Baseline features ({baseline['feature_count']} after training-only "
                "constant filtering):",
            ]
        )
        lines.extend(format_metrics(baseline))
        lines.append(
            f"  Training-selected features ({selected['feature_count']}): "
            f"{', '.join(selected_feature_names[held_out])}"
        )
        lines.extend(format_metrics(selected))
        for key in ("precision", "recall", "f1", "balanced_accuracy"):
            delta = float(selected[key]) - float(baseline[key])
            lines.append(f"  Delta selected - all ({key}): {delta:+.4f}")
        shifts = shift_summary.get(held_out, [])
        if shifts:
            lines.append(
                "  Largest previously reported same-label shifts for this held-out file:"
            )
            lines.extend(f"    {item}" for item in shifts[:3])

    aggregate: dict[str, dict[str, object]] = {}
    for variant in ("all_features", "selected_features"):
        total_matrix = np.zeros((2, 2), dtype=int)
        for name in leakage_names:
            matrix = np.asarray(results[variant][name]["confusion_matrix"], dtype=int)
            total_matrix += matrix
        tn, fp = int(total_matrix[0, 0]), int(total_matrix[0, 1])
        fn, tp = int(total_matrix[1, 0]), int(total_matrix[1, 1])
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        recall_0 = tn / (tn + fp) if tn + fp else 0.0
        aggregate[variant] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "balanced_accuracy": (recall + recall_0) / 2,
            "confusion_matrix": total_matrix.tolist(),
        }

    lines.extend(["", "Combined out-of-fold results:"])
    for variant, title in (
        ("all_features", "All features"),
        ("selected_features", "Training-selected features"),
    ):
        lines.append(f"- {title}:")
        lines.extend(format_metrics(aggregate[variant]))
    lines.append(
        f"- Feature counts: all-input={len(feature_names)}; baseline retained by fold="
        + ", ".join(
            f"{name.rsplit('/', 1)[-1]}:{results['all_features'][name]['feature_count']}"
            for name in leakage_names
        )
        + "; selected by fold="
        + ", ".join(
            f"{name.rsplit('/', 1)[-1]}:{len(selected_feature_names[name])}"
            for name in leakage_names
        )
    )

    all_f1 = float(aggregate["all_features"]["f1"])
    selected_f1 = float(aggregate["selected_features"]["f1"])
    all_recall = float(aggregate["all_features"]["recall"])
    selected_recall = float(aggregate["selected_features"]["recall"])
    if selected_f1 > all_f1 and selected_recall >= all_recall:
        conclusion = (
            "Feature selection improved combined F1 without reducing recall in this grouped "
            "experiment; this is exploratory evidence only."
        )
    elif selected_f1 < all_f1 or selected_recall < all_recall:
        conclusion = (
            "Feature selection did not improve the combined grouped result; it reduced F1 "
            "or recall relative to the all-feature rerun."
        )
    else:
        conclusion = (
            "Feature selection did not materially improve combined F1 or recall in this "
            "grouped experiment."
        )
    lines.extend(
        [
            "",
            "Comparison with prior grouped evaluation:",
            "- The all-feature rerun uses the same fold definitions, model type, preprocessing, and seed "
            "as the existing grouped evaluation; each fold is checked for metric equality above.",
            "- This experiment is a controlled comparison to training-selected features, not a search "
            "over held-out metrics or features.",
            f"- {conclusion}",
            "",
            "Limitations:",
            "- Only three Leakage workbooks are available for grouped holdout; Clean is always included "
            "in training and is not independently tested.",
            "- Within-workbook rows may be temporally dependent; per-row timestamps are unavailable.",
            "- Same-label feature differences reported alongside each fold are descriptive and do not "
            "establish why a model succeeds or fails.",
            "- Labels=1 means anomalous event, not leak-only. No real-world water-leak detection claim "
            "is supported by this experiment.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    try:
        report = make_report()
        REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
        REPORT_FILE.write_text(report, encoding="utf-8")
    except (OSError, KeyError, ValueError, TypeError, zipfile.BadZipFile, ET.ParseError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    print(report, end="")
    print(f"Report saved to: {REPORT_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
