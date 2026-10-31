from __future__ import annotations

import math
import sys
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline

if __package__:
    from scripts.prepare_data import EXPECTED_COLUMNS, read_workbook
else:
    from prepare_data import EXPECTED_COLUMNS, read_workbook


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data" / "processed" / "WDSEventDB"
REPORT_FILE = PROJECT_DIR / "reports" / "sps_wdseventdb_grouped_evaluation.txt"
LABEL_COLUMN = "Labels"
RANDOM_STATE = 42
EXPECTED_WORKBOOK_COUNTS = {
    "Clean": 1,
    "Leakage": 3,
    "Cyberattack": 1,
    "Sensor Failure": 4,
}


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
    metrics: dict[str, float | int | None]
    training_rows: int
    removed_overlap_rows: int
    shared_feature_vectors: int


def load_wdseventdb(data_dir: Path = DATA_DIR) -> tuple[list[str], list[WorkbookData]]:
    files = sorted(data_dir.glob("*/*.xlsx"))
    if not files:
        raise FileNotFoundError(f"No WDSEventDB workbooks found under {data_dir}")

    category_counts = Counter(path.parent.name for path in files)
    if dict(category_counts) != EXPECTED_WORKBOOK_COUNTS:
        raise ValueError(
            "Expected WDSEventDB workbook counts "
            f"{EXPECTED_WORKBOOK_COUNTS}, found {dict(category_counts)}"
        )

    feature_names = [name for name in EXPECTED_COLUMNS if name != LABEL_COLUMN]
    if LABEL_COLUMN not in EXPECTED_COLUMNS or len(feature_names) != 15:
        raise ValueError("WDSEventDB expected schema must have 15 features and Labels")

    workbooks: list[WorkbookData] = []
    for path in files:
        _sheet_name, headers, raw_rows = read_workbook(path)
        if tuple(headers) != EXPECTED_COLUMNS:
            raise ValueError(
                f"{path}: unexpected columns; got {headers!r}, "
                f"expected {list(EXPECTED_COLUMNS)!r}"
            )

        matrix: list[list[float]] = []
        labels: list[int] = []
        row_signatures: Counter[tuple[object, ...]] = Counter()
        for row_number, raw_row in enumerate(raw_rows, start=2):
            if len(raw_row) > len(headers):
                raise ValueError(f"{path}: row {row_number} has more cells than the header")
            row = raw_row + [None] * (len(headers) - len(raw_row))

            feature_values: list[float] = []
            for feature_name, raw_value in zip(feature_names, row[:-1]):
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

            raw_label = row[headers.index(LABEL_COLUMN)]
            if isinstance(raw_label, bool) or raw_label not in (0, 1):
                raise ValueError(
                    f"{path}, row {row_number}: expected binary Labels 0 or 1, "
                    f"found {raw_label!r}"
                )
            label = int(raw_label)
            matrix.append(feature_values)
            labels.append(label)
            row_signatures[
                tuple(None if math.isnan(value) else value for value in feature_values)
                + (label,)
            ] += 1

        if not matrix:
            raise ValueError(f"{path}: workbook contains no data rows")
        workbooks.append(
            WorkbookData(
                name=path.relative_to(data_dir).as_posix(),
                category=path.parent.name,
                features=np.asarray(matrix, dtype=float),
                labels=np.asarray(labels, dtype=int),
                duplicate_rows=sum(count - 1 for count in row_signatures.values()),
            )
        )

    return feature_names, workbooks


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
    """Exclude training rows whose complete sensor vector occurs in the test fold."""
    train_signatures = _feature_signatures(x_train)
    test_signatures = set(_feature_signatures(x_test))
    shared_signatures = set(train_signatures) & test_signatures
    keep_mask = np.asarray(
        [signature not in test_signatures for signature in train_signatures],
        dtype=bool,
    )
    removed_rows = int(np.count_nonzero(~keep_mask))
    filtered_x = x_train[keep_mask]
    filtered_y = y_train[keep_mask]
    if filtered_y.size == 0:
        raise ValueError("No training rows remain after removing test-vector overlaps")
    if set(filtered_y.tolist()) != {0, 1}:
        raise ValueError(
            "Removing test-vector overlaps left training without both Labels classes"
        )
    return filtered_x, filtered_y, removed_rows, len(shared_signatures)


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
) -> dict[str, float | int | None]:
    matrix = confusion_matrix(true_labels, predicted_labels, labels=[0, 1])
    tn, fp, fn, tp = (int(value) for value in matrix.ravel())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if recall is not None and precision + recall
        else 0.0 if recall is not None else None
    )
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
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def evaluate_grouped(workbooks: list[WorkbookData]) -> list[FoldResult]:
    if len(workbooks) < 2:
        raise ValueError("Grouped evaluation requires at least two workbooks")

    results: list[FoldResult] = []
    for test_workbook in workbooks:
        training_workbooks = [item for item in workbooks if item is not test_workbook]
        x_train = np.concatenate([item.features for item in training_workbooks], axis=0)
        y_train = np.concatenate([item.labels for item in training_workbooks], axis=0)
        x_train, y_train, removed_rows, shared_vectors = (
            remove_train_test_feature_overlaps(
                x_train,
                y_train,
                test_workbook.features,
            )
        )

        model = create_model()
        model.fit(x_train, y_train)
        predictions = model.predict(test_workbook.features)
        results.append(
            FoldResult(
                workbook=test_workbook,
                metrics=calculate_metrics(test_workbook.labels, predictions),
                training_rows=len(y_train),
                removed_overlap_rows=removed_rows,
                shared_feature_vectors=shared_vectors,
            )
        )
    return results


def _format_metric(value: float | int | None) -> str:
    return "N/A" if value is None else f"{value:.4f}" if isinstance(value, float) else str(value)


def _aggregate(
    results: list[FoldResult], metric_name: str, weighted: bool
) -> float:
    values = [float(result.metrics[metric_name]) for result in results]
    if not weighted:
        return sum(values) / len(values)
    weights = [len(result.workbook.labels) for result in results]
    return sum(value * weight for value, weight in zip(values, weights)) / sum(weights)


def make_report() -> str:
    feature_names, workbooks = load_wdseventdb()
    folds = evaluate_grouped(workbooks)
    total_confusion = np.zeros((2, 2), dtype=int)
    for result in folds:
        metrics = result.metrics
        total_confusion += np.asarray(
            [
                [int(metrics["tn"]), int(metrics["fp"])],
                [int(metrics["fn"]), int(metrics["tp"])],
            ],
            dtype=int,
        )
    total_tn, total_fp, total_fn, total_tp = (
        int(value) for value in total_confusion.ravel()
    )
    total_predicted = np.asarray(
        [0] * total_tn
        + [1] * total_fp
        + [0] * total_fn
        + [1] * total_tp,
        dtype=int,
    )
    total_true = np.asarray(
        [0] * total_tn
        + [0] * total_fp
        + [1] * total_fn
        + [1] * total_tp,
        dtype=int,
    )
    pooled = calculate_metrics(total_true, total_predicted)

    eligible_folds = [
        result
        for result in folds
        if set(result.workbook.labels.tolist()) == {0, 1}
    ]
    if not eligible_folds:
        raise ValueError("No held-out workbooks contain both Labels classes")
    label_totals = Counter(
        int(label) for workbook in workbooks for label in workbook.labels.tolist()
    )
    internal_duplicates = sum(workbook.duplicate_rows for workbook in workbooks)
    overlap_rows_removed = sum(result.removed_overlap_rows for result in folds)

    lines = [
        "WDSEventDB grouped workbook evaluation",
        "======================================",
        "Target: Labels=0 vs Labels=1 (positive/anomalous label); not confirmed leak detection.",
        "Scope: WDSEventDB Clean, Leakage, Cyberattack, and Sensor Failure workbooks only.",
        "Excluded: data/raw/water_network.csv.",
        "Protocol: leave one complete workbook out; all remaining workbooks form that fold's training set.",
        "Inputs: the 15 WDSEventDB sensor columns only. Labels, category, workbook name, and path are metadata/target, not features.",
        "Preprocessing/model: training-fold median imputation, variance filtering, and RandomForestClassifier.",
        "Model settings match the existing baseline (300 trees, balanced_subsample, random_state=42); no tuning or feature selection.",
        "Each fold model is fitted in memory only; no model artifact is written or replaced.",
        "Predictions use the classifier's default decision rule; no threshold tuning.",
        "Exact matching sensor vectors in the test workbook are removed from that fold's training rows; source workbooks remain unchanged.",
        "Metrics use positive label=1. Precision uses zero_division=0. Recall/F1/balanced accuracy are N/A when undefined for a single-class test workbook.",
        "",
        f"Schema: {len(feature_names)} sensor features; {len(workbooks)} workbooks.",
        f"Overall Labels counts: 0={label_totals[0]}, 1={label_totals[1]}.",
        f"Exact duplicate rows within source workbooks (reported, not deleted): {internal_duplicates}.",
        f"Training rows removed across folds due to test sensor-vector overlap: {overlap_rows_removed}.",
        "",
        "Per-workbook held-out results:",
    ]

    for result in folds:
        workbook = result.workbook
        counts = Counter(int(label) for label in workbook.labels.tolist())
        metrics = result.metrics
        lines.extend(
            [
                f"- Test workbook: {workbook.name} [{workbook.category}]",
                f"  Samples={len(workbook.labels)}; Labels=0:{counts[0]}; Labels=1:{counts[1]}",
                f"  Precision={_format_metric(metrics['precision'])}; Recall={_format_metric(metrics['recall'])}; "
                f"F1={_format_metric(metrics['f1'])}; Balanced Accuracy={_format_metric(metrics['balanced_accuracy'])}",
                f"  TP={metrics['tp']}; FP={metrics['fp']}; TN={metrics['tn']}; FN={metrics['fn']}",
                f"  Training samples after overlap exclusion={result.training_rows}; "
                f"removed overlapping rows={result.removed_overlap_rows}; "
                f"shared unique feature vectors={result.shared_feature_vectors}; "
                f"duplicate rows within test workbook={workbook.duplicate_rows}",
            ]
        )

    lines.extend(["", "Aggregate results:"])
    lines.append(
        "Macro averages across workbooks containing both labels "
        f"(n={len(eligible_folds)}; single-class Clean workbook excluded):"
    )
    for metric_name in ("precision", "recall", "f1", "balanced_accuracy"):
        lines.append(
            f"  {metric_name.replace('_', ' ').title()}="
            f"{_aggregate(eligible_folds, metric_name, weighted=False):.4f}"
        )
    lines.append(
        "Sample-weighted averages across those same multi-class workbooks:"
    )
    for metric_name in ("precision", "recall", "f1", "balanced_accuracy"):
        lines.append(
            f"  {metric_name.replace('_', ' ').title()}="
            f"{_aggregate(eligible_folds, metric_name, weighted=True):.4f}"
        )
    lines.extend(
        [
            "Pooled out-of-fold metrics across all workbooks:",
            f"  Precision={_format_metric(pooled['precision'])}; Recall={_format_metric(pooled['recall'])}; "
            f"F1={_format_metric(pooled['f1'])}; Balanced Accuracy={_format_metric(pooled['balanced_accuracy'])}",
            f"  TP={pooled['tp']}; FP={pooled['fp']}; TN={pooled['tn']}; FN={pooled['fn']}",
            "",
            "Comparison with existing evaluations:",
            "- Existing random row-split result: F1=0.9963, recall=0.9963, balanced accuracy=0.9978 "
            "(Clean plus Leakage only; rows from each workbook may occur in both splits).",
            "- Existing leave-one-Leakage-workbook-out result: F1=0.0019, recall=0.0009, balanced accuracy=0.5005 "
            "(Clean plus the other two Leakage workbooks in training; only 2/2,138 positive rows detected).",
            "- The current evaluation expands training and test domains to all four WDSEventDB categories, "
            "and also holds out Clean. Its aggregate is therefore not a like-for-like replacement for either prior score.",
            "- Random row splitting can place highly similar or repeated rows from one workbook in both train and test, "
            "letting workbook-specific operating conditions appear on both sides and making results optimistic. "
            "The existing baseline report recorded 3 exact sensor-vector signatures across its random split.",
            "",
            "Observed evidence:",
            "- The prior group-shift report documents same-label sensor-distribution differences across Clean/Leakage workbooks, including VFD, pressure, and flow features.",
            "- The grouped reports show near-zero positive recall on held-out Leakage files. This is observed performance, not proof of a specific cause.",
            "- WDSEventDB workbooks have no per-row timestamp or documented sampling interval; workbook-held-out results do not prove chronological or independent-run validation.",
            "- The dataset preflight found 20 exact duplicate rows within Leakage/LeakEvent2.xlsx; this evaluator reports them and removes any train row matching a test sensor vector without changing source data.",
            "",
            "Possible explanations (not established causes):",
            "- Between-workbook distribution shifts may impair transfer to a held-out workbook; the summaries support the existence of shifts but cannot identify their cause.",
            "- Combining scenario workbooks into a generic binary Labels target may mix different event types. Row-level label semantics by event type are not established in available project documentation.",
            "- Available sensor features may not separate anomalous and normal rows robustly across operating scenarios. This evaluation alone does not establish feature insufficiency.",
            "",
            "Conclusion and Decision",
            "-----------------------",
            "- WDSEventDB supports a limited grouped evaluation of binary Labels behavior, but not a strong reliability claim: label semantics, time order, and run independence are not established.",
            "- Workbook generalization is assessed by the per-workbook results above; compare them with the prior Leakage-only folds cautiously because this evaluation trains on additional event categories.",
            "- There is no sufficient evidence to claim confirmed leak detection: Labels is treated as positive/anomalous, not a verified leak-only target.",
            "- No WDSEventDB target or metadata currently supports pipe-segment localization.",
            "- The evidence shows poor prior grouped performance and dataset/workbook shifts; it does not isolate a single root cause as model-only or data-only.",
            "- Recommended next step: verify row-level label/event definitions and obtain run/time metadata or independent workbooks before choosing a leak-specific target or changing model behavior.",
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
        TypeError,
        ValueError,
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
