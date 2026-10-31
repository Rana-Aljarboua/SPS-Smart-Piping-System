from __future__ import annotations

import argparse
import math
import sys
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

if __package__:
    from scripts.prepare_data import EXPECTED_COLUMNS, read_workbook
else:
    from prepare_data import EXPECTED_COLUMNS, read_workbook


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data" / "processed" / "WDSEventDB"
REPORT_FILE = PROJECT_DIR / "reports" / "sps_wdseventdb_leak_separability.txt"
LABEL_COLUMN = "Labels"
EXPECTED_FILES = (
    Path("Clean") / "CleanData.xlsx",
    Path("Leakage") / "LeakEvent1.xlsx",
    Path("Leakage") / "LeakEvent2.xlsx",
    Path("Leakage") / "LeakEvent3.xlsx",
)


@dataclass
class WorkbookRows:
    relative_path: str
    category: str
    features: dict[str, list[float | None]]
    labels: list[int | None]
    duplicate_rows: int
    repeated_feature_vector_rows: int
    conflicting_feature_vectors: int
    missing_by_label: Counter[int]
    invalid_features_by_label: Counter[int]
    invalid_labels: int


def _signature(values: tuple[float | None, ...]) -> tuple[float | None, ...]:
    return tuple(None if value is None else float(value) for value in values)


def load_workbook(path: Path, data_dir: Path, feature_names: list[str]) -> WorkbookRows:
    _sheet_name, headers, raw_rows = read_workbook(path)
    if tuple(headers) != EXPECTED_COLUMNS:
        raise ValueError(
            f"{path}: unexpected columns; got {headers!r}, "
            f"expected {list(EXPECTED_COLUMNS)!r}"
        )

    features = {name: [] for name in feature_names}
    labels: list[int | None] = []
    missing_by_label: Counter[int] = Counter()
    invalid_features_by_label: Counter[int] = Counter()
    exact_rows: Counter[tuple[object, ...]] = Counter()
    vector_labels: dict[tuple[float | None, ...], set[int | None]] = defaultdict(set)
    feature_value_counts: Counter[tuple[float | None, ...]] = Counter()
    invalid_labels = 0

    for row_number, raw_row in enumerate(raw_rows, start=2):
        if len(raw_row) > len(headers):
            raise ValueError(f"{path}, row {row_number}: more cells than header columns")
        row = raw_row + [None] * (len(headers) - len(raw_row))

        raw_label = row[headers.index(LABEL_COLUMN)]
        if isinstance(raw_label, bool) or raw_label not in (0, 1):
            label = None
            invalid_labels += 1
        else:
            label = int(raw_label)
        labels.append(label)

        numeric_vector: list[float | None] = []
        for feature_name in feature_names:
            raw_value = row[headers.index(feature_name)]
            if raw_value is None or raw_value == "":
                value = None
                if label is not None:
                    missing_by_label[label] += 1
            else:
                try:
                    value = float(raw_value)
                except (TypeError, ValueError):
                    value = None
                    if label is not None:
                        invalid_features_by_label[label] += 1
                else:
                    if not math.isfinite(value):
                        value = None
                        if label is not None:
                            invalid_features_by_label[label] += 1
            features[feature_name].append(value)
            numeric_vector.append(value)

        vector = _signature(tuple(numeric_vector))
        vector_labels[vector].add(label)
        feature_value_counts[vector] += 1
        exact_rows[tuple(row)] += 1

    return WorkbookRows(
        relative_path=path.relative_to(data_dir).as_posix(),
        category=path.parent.name,
        features=features,
        labels=labels,
        duplicate_rows=sum(count - 1 for count in exact_rows.values()),
        repeated_feature_vector_rows=sum(
            count - 1 for count in feature_value_counts.values() if count > 1
        ),
        conflicting_feature_vectors=sum(
            len(values) > 1 for values in vector_labels.values()
        ),
        missing_by_label=missing_by_label,
        invalid_features_by_label=invalid_features_by_label,
        invalid_labels=invalid_labels,
    )


def summarize(values: list[float | None]) -> dict[str, float | int | None]:
    finite = np.asarray([value for value in values if value is not None], dtype=float)
    if finite.size == 0:
        return {
            "n": 0,
            "median": None,
            "q1": None,
            "q3": None,
            "iqr": None,
            "mean": None,
            "sd": None,
            "minimum": None,
            "maximum": None,
            "q05": None,
            "q95": None,
        }
    q05, q1, median, q3, q95 = np.quantile(
        finite, [0.05, 0.25, 0.5, 0.75, 0.95]
    )
    return {
        "n": int(finite.size),
        "median": float(median),
        "q1": float(q1),
        "q3": float(q3),
        "iqr": float(q3 - q1),
        "mean": float(np.mean(finite)),
        "sd": float(np.std(finite, ddof=1)) if finite.size > 1 else None,
        "minimum": float(np.min(finite)),
        "maximum": float(np.max(finite)),
        "q05": float(q05),
        "q95": float(q95),
    }


def standardized_mean_difference(
    class_zero: list[float | None], class_one: list[float | None]
) -> float | None:
    zero = np.asarray([value for value in class_zero if value is not None], dtype=float)
    one = np.asarray([value for value in class_one if value is not None], dtype=float)
    if zero.size < 2 or one.size < 2:
        return None
    pooled_variance = (
        (zero.size - 1) * np.var(zero, ddof=1)
        + (one.size - 1) * np.var(one, ddof=1)
    ) / (zero.size + one.size - 2)
    if pooled_variance <= 0:
        return None
    return float((np.mean(one) - np.mean(zero)) / math.sqrt(pooled_variance))


def rank_auc(
    class_zero: list[float | None], class_one: list[float | None]
) -> float | None:
    zero = np.asarray([value for value in class_zero if value is not None], dtype=float)
    one = np.asarray([value for value in class_one if value is not None], dtype=float)
    if zero.size == 0 or one.size == 0:
        return None
    values = np.concatenate((zero, one))
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=float)
    start = 0
    while start < order.size:
        end = start + 1
        while end < order.size and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2
        start = end
    positive_rank_sum = float(np.sum(ranks[zero.size :]))
    return (positive_rank_sum - one.size * (one.size + 1) / 2) / (
        zero.size * one.size
    )


def histogram_overlap(
    left_values: list[float | None], right_values: list[float | None]
) -> float | None:
    left = np.asarray([value for value in left_values if value is not None], dtype=float)
    right = np.asarray([value for value in right_values if value is not None], dtype=float)
    if left.size == 0 or right.size == 0:
        return None
    pooled = np.concatenate((left, right))
    spread = float(np.quantile(pooled, 0.75) - np.quantile(pooled, 0.25))
    width = 2 * spread / (pooled.size ** (1 / 3))
    value_range = float(np.max(pooled) - np.min(pooled))
    if value_range == 0:
        return 1.0
    if width <= 0:
        bins = max(1, int(math.ceil(math.sqrt(pooled.size))))
    else:
        bins = int(math.ceil(value_range / width))
    bins = min(max(bins, 10), 80)
    _, edges = np.histogram(
        pooled,
        bins=bins,
        range=(float(np.min(pooled)), float(np.max(pooled))),
    )
    right_hist, _ = np.histogram(right, bins=edges)
    left_hist, _ = np.histogram(left, bins=edges)
    return float(np.sum(np.minimum(left_hist / left.size, right_hist / right.size)))


def interval_overlap_iou(
    left_stats: dict[str, float | int | None],
    right_stats: dict[str, float | int | None],
) -> float | None:
    left_low, left_high = left_stats["q05"], left_stats["q95"]
    right_low, right_high = right_stats["q05"], right_stats["q95"]
    if None in (left_low, left_high, right_low, right_high):
        return None
    intersection = max(
        0.0,
        min(float(left_high), float(right_high))
        - max(float(left_low), float(right_low)),
    )
    union = max(float(left_high), float(right_high)) - min(float(left_low), float(right_low))
    if union == 0:
        return 1.0
    return intersection / union


def _fmt(value: float | int | None, digits: int = 4) -> str:
    if value is None:
        return "—"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return f"{value:.{digits}f}"


def _class_values(
    workbook: WorkbookRows, feature_name: str, label: int
) -> list[float | None]:
    return [
        value
        for value, row_label in zip(workbook.features[feature_name], workbook.labels)
        if row_label == label
    ]


def _feature_vector_maps(
    workbook: WorkbookRows,
) -> tuple[
    Counter[tuple[float | None, ...]],
    dict[tuple[float | None, ...], set[int | None]],
]:
    vectors: Counter[tuple[float | None, ...]] = Counter()
    vector_labels: dict[tuple[float | None, ...], set[int | None]] = defaultdict(set)
    row_count = len(workbook.labels)
    feature_names = tuple(workbook.features)
    for index in range(row_count):
        vector = _signature(
            tuple(workbook.features[name][index] for name in feature_names)
        )
        vectors[vector] += 1
        vector_labels[vector].add(workbook.labels[index])
    return vectors, vector_labels


def build_report(data_dir: Path = DATA_DIR) -> str:
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

    paths = [data_dir / relative for relative in EXPECTED_FILES]
    missing_paths = [path for path in paths if not path.is_file()]
    if missing_paths:
        raise FileNotFoundError(
            "Required WDSEventDB workbook(s) missing: "
            + ", ".join(str(path) for path in missing_paths)
        )
    workbooks = [load_workbook(path, data_dir, feature_names) for path in paths]
    if any(not workbook.labels for workbook in workbooks):
        raise ValueError("One or more required workbooks contain no data rows")

    total_label_counts = Counter(
        label
        for workbook in workbooks
        for label in workbook.labels
        if label is not None
    )
    total_rows = sum(len(workbook.labels) for workbook in workbooks)
    total_missing = sum(
        sum(workbook.missing_by_label.values()) for workbook in workbooks
    )
    total_invalid_features = sum(
        sum(workbook.invalid_features_by_label.values()) for workbook in workbooks
    )
    total_invalid_labels = sum(workbook.invalid_labels for workbook in workbooks)
    total_duplicate_rows = sum(workbook.duplicate_rows for workbook in workbooks)
    total_conflicting_vectors = sum(
        workbook.conflicting_feature_vectors for workbook in workbooks
    )

    lines = [
        "WDSEventDB Clean/Leakage sensor separability analysis",
        "=====================================================",
        "",
        "Executive summary",
        "-----------------",
        "This is a descriptive, univariate analysis of the 15 WDSEventDB sensor columns against the source Labels values.",
        f"Scope: {total_rows} rows across Clean and the three Leakage workbooks; Labels=0:{total_label_counts[0]}, Labels=1:{total_label_counts[1]}.",
        f"Data quality: missing sensor cells={total_missing}, invalid sensor cells={total_invalid_features}, invalid Labels rows={total_invalid_labels}.",
        f"Duplicate handling: {total_duplicate_rows} exact duplicate row copies were retained; conflicting sensor vectors within a workbook={total_conflicting_vectors}.",
        "Labels=1 is called the positive/anomalous label; this analysis does not treat it as confirmed water leakage.",
        "The per-workbook tables show sensor-level differences for the source label, not evidence that these readings identify confirmed leaks or generalize to new experiments.",
        "No model was trained, no feature was selected for a predictive model, and no threshold or hyperparameter was tuned.",
        "All statistics are computed per workbook and per source label. Sensor distributions are not pooled across workbook identities for the within-file class comparisons.",
        "",
        "Data scope and schema",
        "---------------------",
        f"WDSEventDB path: {data_dir}",
        "Included workbooks:",
    ]
    for workbook in workbooks:
        counts = Counter(label for label in workbook.labels if label is not None)
        lines.append(
            f"- {workbook.relative_path}: {len(workbook.labels)} rows; "
            f"Labels=0:{counts[0]}, Labels=1:{counts[1]}, invalid Labels:{workbook.invalid_labels}"
        )
    lines.extend(
        [
            f"Expected columns ({len(EXPECTED_COLUMNS)}): {', '.join(EXPECTED_COLUMNS)}",
            f"Analyzed sensor columns ({len(feature_names)}): {', '.join(feature_names)}",
            "Column headers must exactly match the WDSEventDB schema in scripts/prepare_data.py.",
            "No workbook/path/category identifiers are used as measurements or predictors.",
            "data/raw/water_network.csv was not read or used.",
            "",
            "Source-label evidence and interpretation",
            "----------------------------------------",
            "The local scripts validate Labels as binary 0/1, but do not establish its event semantics.",
            "README.md describes the supplied paper excerpt as normal (0) versus anomalous event (1), and notes that the source paper page could not be independently accessed.",
            "Therefore Labels=1 is not treated as proof of leak presence. Workbook names suggest scenario context, not verified per-row leak ground truth.",
            "",
            "Data quality by workbook and source label",
            "------------------------------------------",
            "| Workbook | Labels | Rows | Missing sensor cells | Invalid sensor cells | Invalid-label rows |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for workbook in workbooks:
        for label in (0, 1):
            row_count = sum(value == label for value in workbook.labels)
            lines.append(
                f"| {workbook.relative_path} | {label} | {row_count} | "
                f"{workbook.missing_by_label[label]} | "
                f"{workbook.invalid_features_by_label[label]} | — |"
            )
        lines.append(
            f"| {workbook.relative_path} | invalid | {workbook.invalid_labels} | — | — | "
            f"{workbook.invalid_labels} |"
        )
    lines.extend(
        [
            "",
            "Per-workbook, per-sensor statistics and within-file class comparison",
            "---------------------------------------------------------------------",
            "IQR is Q3−Q1. Effect is Cohen's standardized mean difference (Labels=1 minus Labels=0; pooled SD).",
            "AUC is the rank-based probability that a random Labels=1 value exceeds a random Labels=0 value (ties count half); balanced AUC=max(AUC, 1−AUC).",
            "Histogram overlap is the approximate overlap coefficient (0=no shared histogram mass, 1=identical binned distributions), using at least 10 pooled Freedman–Diaconis-derived bins and a cap of 80.",
            "Q05–Q95 overlap is interval intersection/union, a robust range-overlap indicator, not an estimate of shared probability mass.",
            "AUC and overlap are univariate descriptive statistics on the same samples; they are not held-out predictive performance.",
        ]
    )

    for workbook in workbooks:
        lines.extend(
            [
                "",
                f"### {workbook.relative_path}",
                "| Sensor | L0 n | L0 median | L0 IQR | L0 mean | L0 SD | L0 min–max | L1 n | L1 median | L1 IQR | L1 mean | L1 SD | L1 min–max | SMD | AUC | Balanced AUC | Hist. overlap | Q05–Q95 IoU |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|",
            ]
        )
        for feature_name in feature_names:
            zero = _class_values(workbook, feature_name, 0)
            one = _class_values(workbook, feature_name, 1)
            stats_zero = summarize(zero)
            stats_one = summarize(one)
            auc = rank_auc(zero, one)
            balanced_auc = None if auc is None else max(auc, 1 - auc)
            overlap = histogram_overlap(zero, one)
            interval_iou = interval_overlap_iou(stats_zero, stats_one)
            lines.append(
                f"| {feature_name} | {stats_zero['n']} | {_fmt(stats_zero['median'])} | "
                f"{_fmt(stats_zero['iqr'])} | {_fmt(stats_zero['mean'])} | {_fmt(stats_zero['sd'])} | "
                f"{_fmt(stats_zero['minimum'])}–{_fmt(stats_zero['maximum'])} | "
                f"{stats_one['n']} | {_fmt(stats_one['median'])} | {_fmt(stats_one['iqr'])} | "
                f"{_fmt(stats_one['mean'])} | {_fmt(stats_one['sd'])} | "
                f"{_fmt(stats_one['minimum'])}–{_fmt(stats_one['maximum'])} | "
                f"{_fmt(standardized_mean_difference(zero, one))} | {_fmt(auc)} | "
                f"{_fmt(balanced_auc)} | {_fmt(overlap)} | {_fmt(interval_iou)} |"
            )

    lines.extend(
        [
            "",
            "Cross-workbook comparison within Leakage",
            "-----------------------------------------",
            "For each sensor, median ranges compare the three Leakage workbooks separately for each Labels class.",
            "Clean-vs-Leakage OVL compares Clean Labels=0 with each Leakage workbook's Labels=0 using the same histogram-overlap definition.",
            "| Sensor | L0 median range (3 Leakage files) | L1 median range (3 Leakage files) | Median range L0 | Median range L1 | Clean-vs-Leakage L0 OVL (min/median/max) |",
            "|---|---|---|---:|---:|---|",
        ]
    )
    clean = next(workbook for workbook in workbooks if workbook.category == "Clean")
    leakage = [workbook for workbook in workbooks if workbook.category == "Leakage"]
    for feature_name in feature_names:
        medians_zero = [
            summarize(_class_values(workbook, feature_name, 0))["median"]
            for workbook in leakage
        ]
        medians_one = [
            summarize(_class_values(workbook, feature_name, 1))["median"]
            for workbook in leakage
        ]
        overlap_values = [
            histogram_overlap(
                _class_values(clean, feature_name, 0),
                _class_values(workbook, feature_name, 0),
            )
            for workbook in leakage
        ]
        valid_zero = [float(value) for value in medians_zero if value is not None]
        valid_one = [float(value) for value in medians_one if value is not None]
        valid_overlap = [float(value) for value in overlap_values if value is not None]
        lines.append(
            f"| {feature_name} | {_fmt(min(valid_zero))}–{_fmt(max(valid_zero))} | "
            f"{_fmt(min(valid_one))}–{_fmt(max(valid_one))} | "
            f"{_fmt(max(valid_zero) - min(valid_zero))} | "
            f"{_fmt(max(valid_one) - min(valid_one))} | "
            f"{_fmt(min(valid_overlap))}/{_fmt(float(np.median(valid_overlap)))}/{_fmt(max(valid_overlap))} |"
        )

    lines.extend(
        [
            "",
            "Effect-direction consistency across Leakage workbooks",
            "------------------------------------------------------",
            "Direction is based on mean difference (SMD sign) and on whether AUC is above or below 0.5. Direction consistency is descriptive and does not establish a leak-specific sensor signature.",
            "| Sensor | SMD positive in | SMD negative in | AUC > 0.5 in | AUC < 0.5 in | Balanced AUC range |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for feature_name in feature_names:
        smds: list[float] = []
        aucs: list[float] = []
        for workbook in leakage:
            smd = standardized_mean_difference(
                _class_values(workbook, feature_name, 0),
                _class_values(workbook, feature_name, 1),
            )
            auc = rank_auc(
                _class_values(workbook, feature_name, 0),
                _class_values(workbook, feature_name, 1),
            )
            if smd is not None:
                smds.append(smd)
            if auc is not None:
                aucs.append(auc)
        balanced_aucs = [max(value, 1 - value) for value in aucs]
        lines.append(
            f"| {feature_name} | {sum(value > 0 for value in smds)} | "
            f"{sum(value < 0 for value in smds)} | {sum(value > 0.5 for value in aucs)} | "
            f"{sum(value < 0.5 for value in aucs)} | "
            f"{_fmt(min(balanced_aucs) if balanced_aucs else None)}–"
            f"{_fmt(max(balanced_aucs) if balanced_aucs else None)} |"
        )

    leakage_feature_summary: list[tuple[str, float, float, float]] = []
    for feature_name in feature_names:
        balanced_aucs = []
        histogram_overlaps = []
        for workbook in leakage:
            auc = rank_auc(
                _class_values(workbook, feature_name, 0),
                _class_values(workbook, feature_name, 1),
            )
            overlap = histogram_overlap(
                _class_values(workbook, feature_name, 0),
                _class_values(workbook, feature_name, 1),
            )
            if auc is not None:
                balanced_aucs.append(max(auc, 1 - auc))
            if overlap is not None:
                histogram_overlaps.append(overlap)
        if balanced_aucs and histogram_overlaps:
            leakage_feature_summary.append(
                (
                    feature_name,
                    float(np.median(balanced_aucs)),
                    float(np.median(histogram_overlaps)),
                    max(balanced_aucs) - min(balanced_aucs),
                )
            )

    highest_rank_separation = sorted(
        leakage_feature_summary, key=lambda row: row[1], reverse=True
    )[:5]
    lowest_rank_separation = sorted(
        leakage_feature_summary, key=lambda row: row[1]
    )[:5]
    highest_overlap = sorted(
        leakage_feature_summary, key=lambda row: row[2], reverse=True
    )[:5]
    lines.extend(
        [
            "",
            "Observed within-file separability highlights",
            "--------------------------------------------",
            "The following rankings use the median of the three per-Leakage-workbook values; they summarize univariate separation for the source label, not leak detection or held-out prediction.",
            "Highest median balanced AUC:",
        ]
    )
    lines.extend(
        f"- {name}: median balanced AUC={_fmt(auc)}, median histogram overlap={_fmt(overlap)}, balanced-AUC range={_fmt(auc_range)}"
        for name, auc, overlap, auc_range in highest_rank_separation
    )
    lines.append("Lowest median balanced AUC:")
    lines.extend(
        f"- {name}: median balanced AUC={_fmt(auc)}, median histogram overlap={_fmt(overlap)}, balanced-AUC range={_fmt(auc_range)}"
        for name, auc, overlap, auc_range in lowest_rank_separation
    )
    lines.append("Highest median histogram overlap (most shared binned mass):")
    lines.extend(
        f"- {name}: median histogram overlap={_fmt(overlap)}, median balanced AUC={_fmt(auc)}, balanced-AUC range={_fmt(auc_range)}"
        for name, auc, overlap, auc_range in highest_overlap
    )

    within_duplicates = sum(workbook.duplicate_rows for workbook in workbooks)
    repeated_vector_rows = sum(workbook.repeated_feature_vector_rows for workbook in workbooks)
    conflicting_vectors = sum(workbook.conflicting_feature_vectors for workbook in workbooks)
    all_vector_labels: dict[tuple[float | None, ...], set[int | None]] = defaultdict(set)
    all_vector_workbooks: dict[tuple[float | None, ...], set[str]] = defaultdict(set)
    for workbook in workbooks:
        vectors, vector_labels = _feature_vector_maps(workbook)
        for vector in vectors:
            all_vector_workbooks[vector].add(workbook.relative_path)
            all_vector_labels[vector].update(vector_labels[vector])
    shared_cross_workbook_vectors = sum(
        len(paths) > 1 for paths in all_vector_workbooks.values()
    )
    cross_workbook_conflicts = sum(
        len(paths) > 1 and len(labels) > 1
        for vector, paths in all_vector_workbooks.items()
        for labels in [all_vector_labels[vector]]
    )

    lines.extend(
        [
            "",
            "Duplicates and conflicting labels",
            "----------------------------------",
            "| Workbook | Exact duplicate complete rows (extra copies) | Repeated sensor-vector rows (extra copies) | Sensor vectors with both Labels values |",
            "|---|---:|---:|---:|",
        ]
    )
    for workbook in workbooks:
        lines.append(
            f"| {workbook.relative_path} | {workbook.duplicate_rows} | "
            f"{workbook.repeated_feature_vector_rows} | "
            f"{workbook.conflicting_feature_vectors} |"
        )
    lines.extend(
        [
            "",
            f"Exact duplicate complete rows within the four analyzed workbooks (extra copies): {within_duplicates}.",
            f"Repeated sensor-vector rows within workbooks (extra copies, regardless of label): {repeated_vector_rows}.",
            f"Sensor vectors associated with both Labels values within a workbook: {conflicting_vectors}.",
            f"Unique complete sensor vectors appearing in more than one workbook: {shared_cross_workbook_vectors}.",
            f"Cross-workbook shared vectors associated with differing Labels values: {cross_workbook_conflicts}.",
            "Duplicate source rows were retained. No rows were removed and no duplicate-based train/test experiment was performed.",
            "",
            "Observed evidence",
            "-----------------",
            "The tables report measurements from the four source workbooks, per sensor and per Labels value; they do not establish causality or independent sampling.",
            "The same sensor may show different effect directions, AUCs, or distribution overlap across Leakage workbooks; inspect the per-workbook tables rather than interpreting a pooled summary as a universal signature.",
            "The files contain no verified timestamp, sampling interval, run identifier, or per-row event identity in their expected schema. Row order is not treated as time.",
            "Median/IQR and the 5th–95th percentile intervals describe robust location/spread; mean/SD and SMD can be strongly affected by skew, multimodality, outliers, and mixtures of operating states.",
            "AUC gives univariate rank separation for the existing binary label only. Even high AUC in one workbook is not proof of leak detection or of multivariate generalization.",
            "",
            "Possible explanations for grouped generalization failure (hypotheses)",
            "-----------------------------------------------------------------------",
            "Observed between-workbook shifts in same-label sensor distributions can make a learned decision boundary transfer poorly. The descriptive shift does not reveal whether operating conditions, event realization, sensor calibration, or other factors caused it.",
            "A generic binary Labels target across scenario folders may combine event classes with different sensor responses. Per-row label-to-event semantics are not sufficiently verified in local project documentation.",
            "Adjacent rows may be correlated if they are sequential measurements, but no timestamp or sampling metadata establishes that relationship; the analysis cannot estimate the effective independent sample size.",
            "Univariate overlap does not rule out multivariate separability; conversely, univariate effect sizes or AUCs do not establish generalization of the full sensor vector.",
            "",
            "Conclusion and prioritized next steps",
            "-------------------------------------",
            "1. Obtain authoritative row-level documentation for Labels and confirm which rows correspond to leak events versus other anomalies and normal operation.",
            "2. Obtain experiment/run identifiers, timestamps or sampling cadence, and event start/end annotations so independent-run or time-block validation can be designed without inferring order from row position.",
            "3. Obtain sensor metadata (units, locations, calibration, topology and mapping of sensors to pipe segments) and operating-condition metadata (valve/pump settings, demand, pressure regime).",
            "4. After the target and grouping metadata are verified, define a leak-specific grouped evaluation protocol before changing model or inference behavior.",
            "No confirmed leak-detection performance or pipe-segment localization capability is established by this analysis.",
            "",
            "Method details",
            "--------------",
            "Cohen SMD uses the pooled within-class sample standard deviation; it is undefined for insufficient samples or zero pooled variance.",
            "Histogram overlap is an approximation whose value depends on pooled Freedman–Diaconis binning; it is descriptive, not a significance test.",
            "Q05–Q95 interval overlap can be high despite differing shapes, and low despite tail overlap; it is not a full distribution-distance measure.",
            "No p-value or causal claim is used. No data are combined with water_network.csv.",
        ]
    )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Describe within-workbook WDSEventDB sensor separability by Labels."
    )
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output", type=Path, default=REPORT_FILE)
    args = parser.parse_args(argv)
    try:
        output_path = args.output.resolve()
        try:
            output_path.relative_to(PROJECT_DIR.resolve())
        except ValueError as error:
            raise ValueError(
                f"Output file must be inside the SPS workspace: {output_path}"
            ) from error
        report = build_report(args.data_dir)
        with output_path.open("x", encoding="utf-8", newline="\n") as output:
            output.write(report)
    except (
        OSError,
        KeyError,
        IndexError,
        TypeError,
        ValueError,
        zipfile.BadZipFile,
        ET.ParseError,
    ) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    print(f"Report saved to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
