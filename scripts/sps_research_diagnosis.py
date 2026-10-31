from __future__ import annotations

import re
import sys
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter
from pathlib import Path

from train_baseline import LABEL_COLUMN, load_experiment_data, read_workbook


PROJECT_DIR = Path(__file__).resolve().parents[1]
REPORT_DIR = PROJECT_DIR / "reports"
DATA_DIR = PROJECT_DIR / "data" / "processed" / "WDSEventDB"
GROUPED_REPORT = REPORT_DIR / "sps_grouped_evaluation.txt"
FEATURE_SELECTION_REPORT = REPORT_DIR / "sps_feature_selection_results.txt"
SHIFT_REPORT = REPORT_DIR / "sps_group_shift_diagnosis.txt"
BASELINE_REPORT = REPORT_DIR / "sps_supervised_baseline_results.txt"
REPORT_FILE = REPORT_DIR / "sps_research_diagnosis.txt"


def parse_grouped_results(text: str) -> tuple[dict[str, dict[str, object]], dict[str, object]]:
    folds: dict[str, dict[str, object]] = {}
    current: str | None = None
    combined: dict[str, object] = {}
    reading_combined = False
    matrix_rows: list[list[int]] = []

    for line in text.splitlines():
        fold_match = re.match(r"- Held-out workbook: (.+)", line)
        if fold_match:
            current = fold_match.group(1).strip()
            folds[current] = {}
            reading_combined = False
            matrix_rows = []
            continue
        if "Combined out-of-fold results" in line:
            current = None
            reading_combined = True
            matrix_rows = []
            continue

        target = combined if reading_combined else folds.get(current, {})
        metric_match = re.match(
            r"\s+(Precision|Recall|F1|Balanced accuracy)(?: \(anomalous=1\))?:\s+([0-9.]+)",
            line,
        )
        if metric_match:
            key = metric_match.group(1).lower().replace(" ", "_")
            target[key] = float(metric_match.group(2))
        matrix_match = re.match(r"\s*\[+\s*([0-9]+),\s*([0-9]+)\]", line)
        if matrix_match:
            matrix_rows.append([int(matrix_match.group(1)), int(matrix_match.group(2))])
            if len(matrix_rows) == 2:
                target["confusion_matrix"] = matrix_rows
                matrix_rows = []

    if not folds or not {"precision", "recall", "f1", "balanced_accuracy", "confusion_matrix"} <= combined.keys():
        raise ValueError(f"Could not parse grouped metrics from {GROUPED_REPORT}")
    return folds, combined


def parse_feature_selection_f1(text: str) -> float:
    marker = "- Training-selected features:"
    start = text.find(marker, text.find("Combined out-of-fold results:"))
    if start < 0:
        raise ValueError(f"Could not find selected-feature summary in {FEATURE_SELECTION_REPORT}")
    section = text[start:]
    match = re.search(r"\n\s+F1 \(anomalous=1\):\s+([0-9.]+)", section)
    if match is None:
        raise ValueError(f"Could not parse selected-feature F1 from {FEATURE_SELECTION_REPORT}")
    return float(match.group(1))


def extract_fold_shift_features(text: str) -> dict[str, list[str]]:
    results: dict[str, list[str]] = {}
    current: str | None = None
    in_shift_section = False
    for line in text.splitlines():
        fold_match = re.match(r"\[(Leakage/[^]]+)\] held-out rows=", line)
        if fold_match:
            current = fold_match.group(1)
            results[current] = []
            in_shift_section = False
            continue
        if "Largest previously reported same-label shifts" in line:
            in_shift_section = True
            continue
        if current is not None and in_shift_section:
            feature_match = re.match(r"\s+([^|]+?)\s+\|", line)
            if feature_match:
                results[current].append(feature_match.group(1).strip())
            elif line.strip():
                in_shift_section = False
    return results


def inspect_all_workbooks() -> tuple[
    list[str], Counter[int], list[str], list[str], list[str]
]:
    paths = sorted(DATA_DIR.rglob("*.xlsx"))
    if len(paths) != 9:
        raise ValueError(f"Expected 9 extracted WDSEventDB workbooks, found {len(paths)}")

    reference_features: list[str] | None = None
    category_counts: Counter[int] = Counter()
    workbook_lines: list[str] = []
    timestamp_headers: set[str] = set()
    order_lines: list[str] = []
    for path in paths:
        headers, rows = read_workbook(path)
        if LABEL_COLUMN not in headers:
            raise ValueError(f"{LABEL_COLUMN!r} missing from {path}")
        features = [header for header in headers if header != LABEL_COLUMN]
        if reference_features is None:
            reference_features = features
        elif features != reference_features:
            raise ValueError(f"Sensor columns differ in {path}")
        timestamp_headers.update(
            header
            for header in headers
            if re.search(r"(timestamp|datetime|date|^time$)", header, flags=re.IGNORECASE)
        )

        label_index = headers.index(LABEL_COLUMN)
        labels: list[int] = []
        for row_number, row in enumerate(rows, start=2):
            label = row[label_index] if label_index < len(row) else None
            if label not in (0, 1):
                raise ValueError(f"{path}, row {row_number}: invalid Labels value {label!r}")
            labels.append(int(label))
        counts = Counter(labels)
        category_counts.update(counts)
        category = path.parent.name
        workbook_lines.append(
            f"- {category}/{path.name}: rows={len(labels)}, "
            f"Labels=0:{counts[0]}, Labels=1:{counts[1]}"
        )
        transitions = sum(left != right for left, right in zip(labels, labels[1:]))
        order_lines.append(
            f"- {category}/{path.name}: {transitions} label transitions in stored row order "
            f"(not evidence of chronological order)"
        )

    if reference_features is None:
        raise ValueError(f"No extracted dataset workbooks found under {DATA_DIR}")
    return (
        reference_features,
        category_counts,
        workbook_lines,
        sorted(timestamp_headers),
        order_lines,
    )


def make_report() -> str:
    for path in (GROUPED_REPORT, FEATURE_SELECTION_REPORT, SHIFT_REPORT, BASELINE_REPORT):
        if not path.is_file():
            raise FileNotFoundError(f"Required existing report not found: {path}")

    all_features, all_class_counts, all_workbook_lines, timestamp_headers, order_lines = (
        inspect_all_workbooks()
    )
    feature_names, workbooks = load_experiment_data()
    if feature_names != all_features:
        raise ValueError("Training workbooks do not match the complete dataset sensor schema")
    if LABEL_COLUMN in feature_names:
        raise ValueError("Labels must be excluded from sensor features")
    if len(feature_names) != 15:
        raise ValueError(f"Expected 15 sensor features in this dataset version, found {len(feature_names)}")

    total_counts: Counter[int] = Counter()
    for record in workbooks:
        labels = record["labels"]
        rows = record["rows"]
        if not isinstance(labels, list) or not isinstance(rows, list) or len(labels) != len(rows):
            raise ValueError(f"Malformed local workbook record: {record.get('name')}")
        counts = Counter(labels)
        total_counts.update(counts)
    grouped_text = GROUPED_REPORT.read_text(encoding="utf-8")
    grouped_folds, grouped_combined = parse_grouped_results(grouped_text)
    feature_text = FEATURE_SELECTION_REPORT.read_text(encoding="utf-8")
    selected_f1 = parse_feature_selection_f1(feature_text)
    shifts = extract_fold_shift_features(feature_text)
    baseline_text = BASELINE_REPORT.read_text(encoding="utf-8")
    baseline_f1_match = re.search(r"- F1:\s*([0-9.]+)", baseline_text)
    if baseline_f1_match is None:
        raise ValueError(f"Could not parse random-row F1 from {BASELINE_REPORT}")
    random_row_f1 = float(baseline_f1_match.group(1))
    shift_text = SHIFT_REPORT.read_text(encoding="utf-8")

    fold_lines: list[str] = []
    for workbook, metrics in grouped_folds.items():
        matrix = metrics.get("confusion_matrix")
        if not isinstance(matrix, list):
            raise ValueError(f"Missing confusion matrix for {workbook}")
        fn_count = matrix[1][0]
        shift_features = shifts.get(workbook, [])[:3]
        fold_lines.append(
            f"- {workbook}: recall={metrics.get('recall')}, FN={fn_count}; "
            f"largest reported same-label shifts: {', '.join(shift_features) or 'not parsed'}"
        )

    if "timestamp" in shift_text.lower() and "unavailable" not in shift_text.lower():
        timestamp_note = "The existing group-shift report does not establish row timestamps."
    else:
        timestamp_note = "The existing group-shift report states per-row timestamps are unavailable."

    lines = [
        "SPS research and grouped-generalization diagnosis",
        "==================================================",
        "",
        "1. Sources inspected and verified",
        "- Official GitHub README: https://github.com/Moazeni-s-Concise-Lab/WDSEventDB",
        '  It calls WDSEventDB a "sensor-instrumented water distribution testbed" and says the',
        '  Leakage folder contains "clean and leak events"; Cyberattack and Sensor failure',
        "  folders likewise contain clean rows and their scenario events.",
        "- Official repository tree: LICENSE, README.md, WDSEventDB.zip only; no training",
        "  scripts, baseline implementations, notebooks, or model weights were present.",
        "- Repository LICENSE identifies CC0 1.0 Universal. Zenodo record",
        "  https://zenodo.org/records/17547955 declares the dataset CC BY 4.0. These are",
        "  distinct source declarations; verify intended reuse terms with the authors if needed.",
        "- Zenodo metadata cites Raza & Moazeni, 2025, DOI",
        "  https://doi.org/10.1061/9780784486184.082, titled “Hardware-in-the-Loop Based",
        "  Investigations of a Multi-Modal Detection Framework for Cybersecurity and Operational",
        "  Failures in Smart Water Systems: A Real-Time Validation Framework.” Crossref verifies",
        "  bibliographic metadata only, not the paper's methods.",
        "- OpenAlex metadata (https://api.openalex.org/works/https://doi.org/10.1061/9780784486184.082)",
        "  says the related HIL framework uses",
        "  autoencoders and reports F1 >0.94 across five scenarios. This does not identify the",
        "  four claimed WDSEventDB baseline models or establish a compatible SPS protocol.",
        "- The WDSEventDB paper page supplied for this task returned HTTP 403; the ASCE paper",
        "  full text was also not accessible in this inspection. The exact names of the four",
        "  claimed baseline models, their splits, and their metrics therefore remain UNVERIFIED.",
        "- The project-provided paper excerpt says Labels=0 means normal samples and Labels=1",
        "  anomalous event samples. The cited page could not be opened to verify that quotation.",
        "  The official repository contains no code that can be used to reproduce them.",
        "",
        "2. Local dataset and preparation/evaluation audit",
        f"- Parsed all {len(all_workbook_lines)} extracted workbooks with {len(all_features)}",
        "  sensor columns. Full-dataset totals:",
        f"  rows={sum(all_class_counts.values())}, Labels=0:{all_class_counts[0]},",
        f"  Labels=1:{all_class_counts[1]}.",
        "- Per-workbook verified counts across all four scenario folders:",
        *all_workbook_lines,
        "- The training/evaluation scripts use only the Clean workbook plus the three Leakage",
        "  workbooks. Within that selected subset:",
        f"  workbooks={len(workbooks)}, rows={sum(len(record['labels']) for record in workbooks)},",
        f"  Labels=0:{total_counts[0]}, Labels=1:{total_counts[1]}. Labels is target only.",
        "- Labels=0 rows embedded in the three Leakage workbooks total 1,970; the separate",
        "  Clean workbook has 9,743 Labels=0 rows. The three Leakage workbooks contain 2,138",
        "  Labels=1 rows. This matches the counts previously supplied from the paper excerpt,",
        "  but the excerpt's full text could not be independently retrieved here.",
        f"- Timestamp-like columns found in any workbook: {', '.join(timestamp_headers) if timestamp_headers else 'none'}.",
        f"- {timestamp_note} No per-row timestamp, sampling interval, or run identifier is",
        "  available in the parsed workbooks. A label transition count in worksheet order",
        "  cannot validate chronology; shared or separate experimental runs across files",
        "  cannot be determined from the available metadata.",
        *order_lines,
        "- The README describes folder-level scenarios as Clean/normal, Leakage/clean plus",
        "  leak events, Cyberattack/clean plus attack events, and Sensor failure/clean plus",
        "  failure events. This verifies scenario organization, not a row-by-row label mapping.",
        "  The supplied excerpt's general mapping, if accurate, means Label=1 is an event",
        "  sample whose type depends on scenario; it is not leak-only. Exact row-level mapping",
        "  in each category remains unverified, and the label does not localize an event.",
        "- In the local supervised scripts, Labels is y and is excluded from X, as are",
        "  workbook names, paths, and folder names. No label-as-feature mismatch was found.",
        "- The grouped protocol holds each Leakage workbook out intact and keeps the only Clean",
        "  workbook in training. Each held-out Leakage file contains both labels, so its test",
        "  includes that run's normal rows. The protocol is a legitimate workbook-held-out",
        "  stress test, but it does not test a new Clean run and cannot establish independence",
        "  of rows or chronological generalization.",
        "- There is no confirmed label/preprocessing leakage in the grouped scripts reviewed:",
        "  imputation, constant-feature filtering, feature selection, and fitting are within",
        "  each training fold. The initial random-row split is a confirmed different evaluation",
        "  unit: rows from each workbook can occur in both train and test.",
        "",
        "3. Official methods and comparability",
        "- The accessible official README and Zenodo metadata describe event/anomaly diagnosis",
        "  on a laboratory testbed, not a residential sensor deployment. They do not expose the",
        "  exact four baseline algorithms or the paper's complete split protocol.",
        "- The related HIL paper's accessible abstract supports only the broad method family",
        "  (autoencoders) and its abstract-level result; it is insufficient to reproduce the",
        "  specific WDSEventDB baselines or compare metrics without the full methods and splits.",
        "- Official baseline source code and pretrained weights were not present in the",
        "  official GitHub repository tree. Consequently, SPS cannot claim to reproduce an",
        "  official baseline.",
        f"- SPS random-row result: F1={random_row_f1:.4f}; grouped workbook result:",
        f"  F1={grouped_combined['f1']:.4f}, recall={grouped_combined['recall']:.4f};",
        f"  training-only feature selection grouped F1={selected_f1:.4f}.",
        "- These SPS results use the same local Labels target and sensor-only inputs but",
        "  different evaluation units (random rows versus held-out Leakage workbooks); they",
        "  are not directly comparable to each other as estimates of independent-run accuracy,",
        "  nor to the inaccessible official paper metrics.",
        "",
        "4. Evidence-supported explanation for poor grouped generalization",
        "- Verified: grouped recall is near zero, with false negatives in all held-out Leakage",
        "  workbooks; the model predicts almost exclusively class 0. Training-only feature",
        "  selection increased the combined count of detected positives only from 2 to 4.",
        "- Verified descriptive evidence: the shift diagnosis reports substantial same-label",
        "  sensor-distribution differences across workbooks (including VFD and flow/pressure",
        "  features). This supports workbook/domain shift as the strongest current explanation.",
        "- Uncertainty: the data do not establish why those distributions differ; operating",
        "  conditions, experiment/run differences, row ordering, or sensor behavior could",
        "  contribute. Class imbalance is also present, but class weighting was used and no",
        "  causal attribution is possible from aggregate metrics and descriptive summaries.",
        "  Per-row predictions/time metadata were not available to test error causes.",
        "",
        "5. One recommended next experiment (not run)",
        "- Keep the current Random Forest and sensor features fixed; calibrate only its",
        "  probability threshold using workbook-separated validation, then evaluate on a",
        "  different untouched Leakage workbook. Rotate the validation/test workbooks across",
        "  the three available Leakage files; for each fold use Clean plus the remaining",
        "  Leakage workbook for fitting, a second Leakage workbook for threshold selection,",
        "  and the final Leakage workbook only for evaluation. Fit imputing/filtering/model",
        "  on training workbooks only; choose the threshold by validation F1 (tie: higher",
        "  threshold). Report each fold and pooled held-out results. Do not tune against test.",
        "- This is justified because the current classifier's default decision rule almost",
        "  never predicts anomalies; it tests whether the decision threshold contributes to",
        "  the miss rate without changing algorithms. It cannot remove domain shift and may",
        "  overfit the small validation workbook. It remains an anomaly-event experiment,",
        "  not evidence of residential leak-detection readiness.",
        "- After implementing the separate evaluator (not created or executed in this task),",
        "  the intended command is:",
        r"  python -B scripts\sps_threshold_calibration.py --protocol nested-workbook --clean data\processed\WDSEventDB\Clean\CleanData.xlsx --leakage-dir data\processed\WDSEventDB\Leakage --threshold-objective validation-f1 --seed 42 --output reports\sps_threshold_calibration_results.txt",
        "",
        "6. Files",
        "- Created by this task: scripts/sps_research_diagnosis.py and",
        "  reports/sps_research_diagnosis.txt.",
        "- Existing data, model, preparation/training/evaluation scripts, and reports are read",
        "  only and are not modified by this diagnostic.",
        "",
        "Local references:",
        "- scripts/train_baseline.py",
        "- scripts/evaluate_grouped.py",
        "- scripts/evaluate_feature_selection.py",
        "- reports/sps_supervised_baseline_results.txt",
        "- reports/sps_grouped_evaluation.txt",
        "- reports/sps_group_shift_diagnosis.txt",
        "- reports/sps_feature_selection_results.txt",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    try:
        report = make_report()
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        REPORT_FILE.write_text(report, encoding="utf-8", newline="\n")
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
