from __future__ import annotations

import math
import sys
import warnings
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

if __package__:
    from scripts.prepare_data import EXPECTED_COLUMNS, read_workbook
else:
    from prepare_data import EXPECTED_COLUMNS, read_workbook


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data" / "processed" / "WDSEventDB"
REPORT_FILE = PROJECT_DIR / "reports" / "sps_wdseventdb_model_comparison.txt"
LABEL_COLUMN = "Labels"
RANDOM_STATE = 42
EXPECTED_WORKBOOKS = (
    Path("Clean") / "CleanData.xlsx",
    Path("Leakage") / "LeakEvent1.xlsx",
    Path("Leakage") / "LeakEvent2.xlsx",
    Path("Leakage") / "LeakEvent3.xlsx",
)
MODEL_NAMES = ("Random Forest", "Logistic Regression")


@dataclass(frozen=True)
class WorkbookData:
    name: str
    category: str
    features: np.ndarray
    labels: np.ndarray
    duplicate_rows: int


@dataclass(frozen=True)
class FoldResult:
    workbook: WorkbookData
    training_workbooks: tuple[str, ...]
    metrics: dict[str, float | int | None]
    y_true: np.ndarray
    y_pred: np.ndarray
    training_rows: int
    removed_overlap_rows: int
    shared_feature_vectors: int
    fit_warnings: tuple[str, ...]


def load_wdseventdb(data_dir: Path = DATA_DIR) -> tuple[list[str], list[WorkbookData]]:
    data_dir = data_dir.resolve()
    try:
        data_dir.relative_to(PROJECT_DIR.resolve())
    except ValueError as error:
        raise ValueError(f"Data directory must be inside the SPS workspace: {data_dir}") from error
    if not data_dir.is_dir():
        raise FileNotFoundError(f"WDSEventDB directory not found: {data_dir}")

    feature_names = [name for name in EXPECTED_COLUMNS if name != LABEL_COLUMN]
    if len(feature_names) != 15 or LABEL_COLUMN not in EXPECTED_COLUMNS:
        raise ValueError("Expected WDSEventDB schema with 15 sensors and Labels")

    paths = [data_dir / relative for relative in EXPECTED_WORKBOOKS]
    missing_paths = [path for path in paths if not path.is_file()]
    if missing_paths:
        raise FileNotFoundError(
            "Required WDSEventDB workbook(s) missing: "
            + ", ".join(str(path) for path in missing_paths)
        )

    workbooks: list[WorkbookData] = []
    for path in paths:
        _sheet_name, headers, raw_rows = read_workbook(path)
        if tuple(headers) != EXPECTED_COLUMNS:
            raise ValueError(
                f"{path}: unexpected columns; got {headers!r}, "
                f"expected {list(EXPECTED_COLUMNS)!r}"
            )

        matrix: list[list[float]] = []
        labels: list[int] = []
        exact_rows: Counter[tuple[object, ...]] = Counter()
        label_index = headers.index(LABEL_COLUMN)
        feature_indices = [headers.index(name) for name in feature_names]

        for row_number, raw_row in enumerate(raw_rows, start=2):
            if len(raw_row) > len(headers):
                raise ValueError(f"{path}, row {row_number}: more cells than header columns")
            row = raw_row + [None] * (len(headers) - len(raw_row))
            label = row[label_index]
            if isinstance(label, bool) or label not in (0, 1):
                raise ValueError(
                    f"{path}, row {row_number}: expected binary Labels 0 or 1, found {label!r}"
                )

            feature_values: list[float] = []
            for feature_name, feature_index in zip(feature_names, feature_indices):
                raw_value = row[feature_index]
                if raw_value is None or raw_value == "":
                    feature_values.append(float("nan"))
                    continue
                try:
                    value = float(raw_value)
                except (TypeError, ValueError) as error:
                    raise ValueError(
                        f"{path}, row {row_number}, feature {feature_name!r}: "
                        f"not numeric ({raw_value!r})"
                    ) from error
                if not math.isfinite(value):
                    raise ValueError(
                        f"{path}, row {row_number}, feature {feature_name!r}: "
                        f"not finite ({raw_value!r})"
                    )
                feature_values.append(value)

            matrix.append(feature_values)
            labels.append(int(label))
            exact_rows[tuple(row)] += 1

        if not matrix:
            raise ValueError(f"{path}: workbook contains no data rows")
        workbooks.append(
            WorkbookData(
                name=path.relative_to(data_dir).as_posix(),
                category=path.parent.name,
                features=np.asarray(matrix, dtype=float),
                labels=np.asarray(labels, dtype=int),
                duplicate_rows=sum(count - 1 for count in exact_rows.values()),
            )
        )
    return feature_names, workbooks


def iter_grouped_folds(
    workbooks: list[WorkbookData],
) -> Iterator[tuple[WorkbookData, list[WorkbookData]]]:
    if len(workbooks) < 2:
        raise ValueError("Grouped evaluation requires at least two workbooks")
    for test_workbook in workbooks:
        training_workbooks = [item for item in workbooks if item is not test_workbook]
        if len(training_workbooks) != len(workbooks) - 1:
            raise ValueError("A grouped fold must hold exactly one complete workbook out")
        yield test_workbook, training_workbooks


def _feature_signatures(matrix: np.ndarray) -> list[tuple[object, ...]]:
    return [
        tuple(None if math.isnan(float(value)) else float(value) for value in row)
        for row in matrix
    ]


def remove_train_test_feature_overlaps(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    train_signatures = _feature_signatures(x_train)
    test_signatures = set(_feature_signatures(x_test))
    shared_signatures = set(train_signatures) & test_signatures
    keep_mask = np.asarray(
        [signature not in test_signatures for signature in train_signatures],
        dtype=bool,
    )
    filtered_x = x_train[keep_mask]
    filtered_y = y_train[keep_mask]
    removed_rows = int(np.count_nonzero(~keep_mask))
    if filtered_y.size == 0:
        raise ValueError("No training rows remain after removing test-vector overlaps")
    if set(filtered_y.tolist()) != {0, 1}:
        raise ValueError("Removing test-vector overlaps left training without both Labels classes")
    return filtered_x, filtered_y, removed_rows, len(shared_signatures)


def create_model(model_name: str) -> Pipeline:
    steps = [
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("variance", VarianceThreshold()),
    ]
    if model_name == "Random Forest":
        steps.append(
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=300,
                    class_weight="balanced_subsample",
                    random_state=RANDOM_STATE,
                    n_jobs=1,
                ),
            )
        )
    elif model_name == "Logistic Regression":
        steps.extend(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=2000,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        )
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    return Pipeline(steps=steps)


def calculate_metrics(
    true_labels: np.ndarray, predicted_labels: np.ndarray
) -> dict[str, float | int | None]:
    if true_labels.shape != predicted_labels.shape:
        raise ValueError("True and predicted labels must have the same shape")
    if not np.isin(true_labels, [0, 1]).all() or not np.isin(predicted_labels, [0, 1]).all():
        raise ValueError("Metrics require binary labels 0 and 1")
    tn = int(np.count_nonzero((true_labels == 0) & (predicted_labels == 0)))
    fp = int(np.count_nonzero((true_labels == 0) & (predicted_labels == 1)))
    fn = int(np.count_nonzero((true_labels == 1) & (predicted_labels == 0)))
    tp = int(np.count_nonzero((true_labels == 1) & (predicted_labels == 1)))
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1_denominator = 2 * tp + fp + fn
    f1 = (
        2 * tp / f1_denominator
        if f1_denominator and recall is not None
        else None
    )
    false_positive_rate = fp / (fp + tn) if fp + tn else None
    specificity = tn / (tn + fp) if tn + fp else None
    balanced_accuracy = (
        (recall + specificity) / 2
        if recall is not None and specificity is not None
        else None
    )
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "balanced_accuracy": balanced_accuracy,
        "false_positive_rate": false_positive_rate,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def evaluate_models(
    workbooks: list[WorkbookData],
) -> dict[str, list[FoldResult]]:
    results: dict[str, list[FoldResult]] = {name: [] for name in MODEL_NAMES}
    for test_workbook, training_workbooks in iter_grouped_folds(workbooks):
        x_train = np.concatenate([item.features for item in training_workbooks], axis=0)
        y_train = np.concatenate([item.labels for item in training_workbooks], axis=0)
        x_train, y_train, removed_rows, shared_vectors = (
            remove_train_test_feature_overlaps(
                x_train,
                y_train,
                test_workbook.features,
            )
        )
        if set(y_train.tolist()) != {0, 1}:
            raise ValueError(
                f"Training fold for {test_workbook.name} does not contain both Labels classes"
            )

        for model_name in MODEL_NAMES:
            model = create_model(model_name)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                model.fit(x_train, y_train)
            warning_messages = tuple(
                f"{warning.category.__name__}: {warning.message}" for warning in caught
            )
            predictions = model.predict(test_workbook.features)
            results[model_name].append(
                FoldResult(
                    workbook=test_workbook,
                    training_workbooks=tuple(item.name for item in training_workbooks),
                    metrics=calculate_metrics(test_workbook.labels, predictions),
                    y_true=test_workbook.labels.copy(),
                    y_pred=predictions,
                    training_rows=len(y_train),
                    removed_overlap_rows=removed_rows,
                    shared_feature_vectors=shared_vectors,
                    fit_warnings=warning_messages,
                )
            )
    return results


def _format_metric(value: float | int | None) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return f"{value:.4f}"


def _macro_metrics(
    results: list[FoldResult], field: str, include_clean: bool = False
) -> float | None:
    eligible = [
        result
        for result in results
        if (include_clean or result.workbook.category == "Leakage")
        and result.metrics[field] is not None
    ]
    if not eligible:
        return None
    return float(np.mean([float(result.metrics[field]) for result in eligible]))


def _weighted_metrics(
    results: list[FoldResult], field: str, include_clean: bool = False
) -> float | None:
    eligible = [
        result
        for result in results
        if (include_clean or result.workbook.category == "Leakage")
        and result.metrics[field] is not None
    ]
    if not eligible:
        return None
    total_weight = sum(len(result.y_true) for result in eligible)
    return float(
        sum(float(result.metrics[field]) * len(result.y_true) for result in eligible)
        / total_weight
    )


def _pooled_metrics(results: list[FoldResult]) -> dict[str, float | int | None]:
    y_true = np.concatenate([result.y_true for result in results])
    y_pred = np.concatenate([result.y_pred for result in results])
    return calculate_metrics(y_true, y_pred)


def make_report() -> str:
    feature_names, workbooks = load_wdseventdb()
    model_results = evaluate_models(workbooks)
    total_rows = sum(len(workbook.labels) for workbook in workbooks)
    total_labels = Counter(
        int(label) for workbook in workbooks for label in workbook.labels
    )
    duplicate_total = sum(workbook.duplicate_rows for workbook in workbooks)
    lines = [
        "SPS WDSEventDB grouped model comparison",
        "=======================================",
        "Decision target: source Labels=0 vs Labels=1 (positive/anomalous label); not confirmed leak ground truth.",
        "Data scope: WDSEventDB Clean/CleanData.xlsx and the three Leakage workbooks only.",
        "Excluded: Cyberattack, Sensor Failure, and data/raw/water_network.csv.",
        "Features: the 15 sensor columns only; Labels, category, workbook name, and path are never input features.",
        "Protocol: fixed leave-one-workbook-out over the four included workbooks (four folds); one entire workbook is test and the other three are training in each fold.",
        "This protocol is identical for both models. No rows from a held-out workbook enter training, preprocessing fitting, or model fitting.",
        "Exact matching 15-sensor vectors in training are excluded if they also appear in the held-out workbook; source rows are not changed. Repeated rows inside a held-out file remain test observations.",
        "Preprocessing is fitted within each training-fold Pipeline: median imputation and constant-feature filtering for both models; StandardScaler is added only for Logistic Regression.",
        "Random Forest matches the existing baseline configuration: 300 trees, class_weight='balanced_subsample', random_state=42, n_jobs=1.",
        "Logistic Regression: class_weight='balanced', lbfgs/default solver, max_iter=2000, random_state=42.",
        "No third classifier was added: the two requested models permit a direct comparison, while only three Leakage workbooks are available for grouped validation.",
        "Both use the estimator's default 0.5 class decision rule; no threshold tuning, feature selection, or hyperparameter search was performed.",
        "No model artifacts are written or replaced. Scikit-learn version: " + sklearn.__version__ + ".",
        "",
        f"Dataset: {total_rows} rows; Labels=0:{total_labels[0]}, Labels=1:{total_labels[1]}; exact duplicate complete rows within files={duplicate_total} (retained).",
        f"Features ({len(feature_names)}): {', '.join(feature_names)}",
        "",
        "Per-workbook held-out results",
        "-----------------------------",
        "Confusion matrix is shown as [[TN, FP], [FN, TP]]. False-positive rate = FP/(FP+TN).",
    ]
    for model_name in MODEL_NAMES:
        lines.extend(["", f"## {model_name}"])
        for result in model_results[model_name]:
            labels = Counter(int(label) for label in result.workbook.labels)
            metrics = result.metrics
            matrix = (
                f"[[{metrics['tn']}, {metrics['fp']}], "
                f"[{metrics['fn']}, {metrics['tp']}]]"
            )
            lines.extend(
                [
                    f"- Test workbook: {result.workbook.name}",
                    f"  Training workbooks: {', '.join(result.training_workbooks)}",
                    f"  Samples={len(result.y_true)}; Labels=0:{labels[0]}; Labels=1:{labels[1]}",
                    f"  Training rows after overlap exclusion={result.training_rows}; "
                    f"removed overlapping rows={result.removed_overlap_rows}; "
                    f"shared feature vectors={result.shared_feature_vectors}; "
                    f"exact duplicate rows within held-out workbook={result.workbook.duplicate_rows}",
                    f"  Confusion Matrix: {matrix}",
                    f"  Precision={_format_metric(metrics['precision'])}; "
                    f"Recall={_format_metric(metrics['recall'])}; "
                    f"F1={_format_metric(metrics['f1'])}; "
                    f"Balanced Accuracy={_format_metric(metrics['balanced_accuracy'])}; "
                    f"False-positive rate={_format_metric(metrics['false_positive_rate'])}",
                ]
            )
            if result.fit_warnings:
                lines.append(
                    "  Fit warnings: " + " | ".join(result.fit_warnings)
                )

        leakage_folds = [
            result for result in model_results[model_name]
            if result.workbook.category == "Leakage"
        ]
        leakage_pooled = _pooled_metrics(leakage_folds)
        pooled = _pooled_metrics(model_results[model_name])
        lines.extend(
            [
                "",
                "Leakage-only summaries (the three held-out Leakage workbooks; Clean excluded):",
                "  Macro: "
                + "; ".join(
                    f"{field}={_format_metric(_macro_metrics(leakage_folds, field))}"
                    for field in (
                        "precision",
                        "recall",
                        "f1",
                        "balanced_accuracy",
                        "false_positive_rate",
                    )
                ),
                "  Sample-weighted: "
                + "; ".join(
                    f"{field}={_format_metric(_weighted_metrics(leakage_folds, field))}"
                    for field in (
                        "precision",
                        "recall",
                        "f1",
                        "balanced_accuracy",
                        "false_positive_rate",
                    )
                ),
                "Leakage-only pooled out-of-fold:",
                "  "
                + "; ".join(
                    f"{field}={_format_metric(leakage_pooled[field])}"
                    for field in (
                        "precision",
                        "recall",
                        "f1",
                        "balanced_accuracy",
                        "false_positive_rate",
                    )
                ),
                "  "
                + f"TP={leakage_pooled['tp']}; FP={leakage_pooled['fp']}; "
                + f"TN={leakage_pooled['tn']}; FN={leakage_pooled['fn']}",
                "All four held-out files pooled out-of-fold:",
                "  "
                + "; ".join(
                    f"{field}={_format_metric(pooled[field])}"
                    for field in (
                        "precision",
                        "recall",
                        "f1",
                        "balanced_accuracy",
                        "false_positive_rate",
                    )
                ),
                "  "
                + f"TP={pooled['tp']}; FP={pooled['fp']}; TN={pooled['tn']}; FN={pooled['fn']}",
            ]
        )

    lines.extend(
        [
            "",
            "Comparison with previous project results",
            "-----------------------------------------",
            "The original random-row-split baseline reported F1=0.9963, but rows from the same workbooks appeared in both sides and 3 exact sensor-vector signatures crossed the split.",
            "The earlier Leakage-only leave-one-workbook-out evaluation reported pooled F1=0.0019. Its split set and training scope differ from this four-fold comparison.",
            "Those values are context, not direct apples-to-apples comparators; the present results compare models under the same fixed four-workbook protocol.",
            "",
            "Observed evidence and limitations",
            "----------------------------------",
            "Labels is the source-provided target. Project documentation describes a general normal/anomalous event distinction but does not independently establish Labels=1 as confirmed leakage for every row.",
            "The Clean holdout contains only Labels=0, so positive-class recall, F1, and balanced accuracy are N/A when mathematically undefined; Precision is N/A if the model predicts no positive samples. The false-positive rate remains defined because this test file contains negatives.",
            "Leakage-only macro and sample-weighted averages omit undefined metric values (such as Precision when a fold has no positive predictions); pooled metrics are recomputed from summed out-of-fold confusion counts.",
            "There are only three Leakage workbooks for the main anomaly-class comparison. Workbook-held-out testing measures transfer to these particular files, not proven independent-run or future-time generalization.",
            "No per-row timestamps, documented sampling cadence, run IDs, or pipe-segment labels are used or assumed. Pipe-segment localization is not evaluated.",
            "Per-sensor ranking/separability is not used to select a feature or to claim leak detection.",
            "Within-file duplicate rows are reported and retained. Any exact train/test feature-vector overlap is excluded from training in that fold and reported above.",
            "",
            "Conclusion and single recommendation",
            "------------------------------------",
        ]
    )
    lines.append(
        "- Do not select or replace the production model from this comparison: Logistic Regression has higher Leakage macro F1 (0.5309 vs 0.0022 for Random Forest), but its per-file F1 ranges from 0.0000 to 0.9785 and its Leakage macro false-positive rate is 0.3494. With only three Leakage workbooks and unverified Labels semantics, neither result establishes reliable leak detection or independent-run generalization. Keep the saved Random Forest unchanged until verified leak labels and additional run-level holdouts are available."
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    try:
        report = make_report()
        with REPORT_FILE.open("x", encoding="utf-8", newline="\n") as output:
            output.write(report)
    except (
        OSError,
        KeyError,
        TypeError,
        ValueError,
        zipfile.BadZipFile,
        ET.ParseError,
    ) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(f"Report saved to: {REPORT_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
