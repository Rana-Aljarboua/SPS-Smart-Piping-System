from __future__ import annotations

import math
import posixpath
import re
import sys
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import joblib


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data" / "processed" / "WDSEventDB"
CLEAN_FILE = DATA_DIR / "Clean" / "CleanData.xlsx"
LEAKAGE_DIR = DATA_DIR / "Leakage"
REPORT_FILE = PROJECT_DIR / "reports" / "sps_supervised_baseline_results.txt"
MODEL_FILE = PROJECT_DIR / "models" / "sps_event_baseline.joblib"
LABEL_COLUMN = "Labels"
RANDOM_STATE = 42
TEST_SIZE = 0.25

NAMESPACE = {
    "main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
    "office_rel": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "package_rel": "http://schemas.openxmlformats.org/package/2006/relationships",
}


def column_index(cell_reference: str) -> int:
    match = re.match(r"[A-Z]+", cell_reference.upper())
    if match is None:
        raise ValueError(f"Invalid Excel cell reference: {cell_reference}")

    index = 0
    for letter in match.group():
        index = index * 26 + ord(letter) - ord("A") + 1
    return index - 1


def read_cell(cell: ET.Element, shared_strings: list[str]) -> object:
    cell_type = cell.attrib.get("t")
    if cell_type == "inlineStr":
        return "".join(text.text or "" for text in cell.findall(".//main:t", NAMESPACE))

    value = cell.find("main:v", NAMESPACE)
    if value is None or value.text is None:
        return None

    raw_value = value.text
    if cell_type == "s":
        return shared_strings[int(raw_value)]
    if cell_type == "str":
        return raw_value
    if cell_type == "b":
        return raw_value == "1"

    try:
        number = float(raw_value)
    except ValueError:
        return raw_value
    return int(number) if number.is_integer() else number


def read_workbook(path: Path) -> tuple[list[str], list[list[object]]]:
    with zipfile.ZipFile(path) as workbook:
        names = workbook.namelist()
        shared_strings: list[str] = []
        if "xl/sharedStrings.xml" in names:
            string_root = ET.fromstring(workbook.read("xl/sharedStrings.xml"))
            shared_strings = [
                "".join(text.text or "" for text in item.findall(".//main:t", NAMESPACE))
                for item in string_root.findall("main:si", NAMESPACE)
            ]

        workbook_root = ET.fromstring(workbook.read("xl/workbook.xml"))
        relationships_root = ET.fromstring(workbook.read("xl/_rels/workbook.xml.rels"))
        relationships = {
            relationship.attrib["Id"]: relationship.attrib["Target"]
            for relationship in relationships_root.findall(
                "package_rel:Relationship", NAMESPACE
            )
        }
        sheet = workbook_root.find("main:sheets/main:sheet", NAMESPACE)
        if sheet is None:
            raise ValueError(f"No worksheet found in {path}")

        relationship_id = sheet.attrib[f"{{{NAMESPACE['office_rel']}}}id"]
        target = relationships[relationship_id]
        worksheet_path = (
            target.lstrip("/")
            if target.startswith("/")
            else posixpath.normpath(posixpath.join("xl", target))
        )
        worksheet_root = ET.fromstring(workbook.read(worksheet_path))
        sheet_rows = worksheet_root.findall("main:sheetData/main:row", NAMESPACE)

        rows: list[list[object]] = []
        for row in sheet_rows:
            cells = {
                column_index(cell.attrib["r"]): read_cell(cell, shared_strings)
                for cell in row.findall("main:c", NAMESPACE)
            }
            if cells:
                rows.append([cells.get(index) for index in range(max(cells) + 1)])

    if not rows:
        raise ValueError(f"No rows found in {path}")
    headers = [str(value) if value is not None else "" for value in rows[0]]
    return headers, rows[1:]


def load_experiment_data() -> tuple[list[str], list[dict[str, object]]]:
    if not CLEAN_FILE.is_file():
        raise FileNotFoundError(f"Clean workbook not found: {CLEAN_FILE}")
    leakage_files = sorted(LEAKAGE_DIR.glob("*.xlsx"))
    if len(leakage_files) != 3:
        raise ValueError(f"Expected 3 Leakage workbooks, found {len(leakage_files)}")

    paths = [CLEAN_FILE, *leakage_files]
    feature_names: list[str] | None = None
    workbook_records: list[dict[str, object]] = []

    for path in paths:
        headers, rows = read_workbook(path)
        if LABEL_COLUMN not in headers:
            raise ValueError(f"Required column {LABEL_COLUMN!r} missing from {path}")

        current_features = [name for name in headers if name != LABEL_COLUMN]
        if feature_names is None:
            feature_names = current_features
        elif current_features != feature_names:
            raise ValueError(f"Sensor feature columns differ in {path}")

        sensor_rows: list[dict[str, float | None]] = []
        labels: list[int] = []
        missing: Counter[str] = Counter()
        for row_number, raw_row in enumerate(rows, start=2):
            sensor_row: dict[str, float | None] = {}
            for feature in feature_names:
                index = headers.index(feature)
                raw_value = raw_row[index] if index < len(raw_row) else None
                if raw_value is None or raw_value == "":
                    sensor_row[feature] = None
                    missing[feature] += 1
                    continue
                try:
                    value = float(raw_value)
                except (TypeError, ValueError) as error:
                    raise ValueError(
                        f"{path}, row {row_number}, feature {feature!r}: "
                        f"not numeric ({raw_value!r})"
                    ) from error
                if not math.isfinite(value):
                    raise ValueError(
                        f"{path}, row {row_number}, feature {feature!r}: "
                        f"not finite ({raw_value!r})"
                    )
                sensor_row[feature] = value
            label_position = headers.index(LABEL_COLUMN)
            raw_label = raw_row[label_position] if label_position < len(raw_row) else None
            if raw_label not in (0, 1):
                raise ValueError(
                    f"{path}, row {row_number}: expected binary Labels 0 or 1, "
                    f"found {raw_label!r}"
                )
            labels.append(int(raw_label))
            sensor_rows.append(sensor_row)

        # Workbook identity and group are metadata only, never model features.
        group = "Clean" if path == CLEAN_FILE else "Leakage"
        workbook_records.append(
            {
                "group": group,
                "name": path.name,
                "rows": sensor_rows,
                "labels": labels,
                "row_count": len(sensor_rows),
                "missing": missing,
            }
        )

    if feature_names is None:
        raise ValueError("No sensor features found")
    return feature_names, workbook_records


def make_report() -> str:
    feature_names, workbooks = load_experiment_data()
    if LABEL_COLUMN in feature_names:
        raise ValueError("Label leakage: Labels must not be included in sensor features")
    if not workbooks:
        raise ValueError("No workbooks were loaded")

    all_rows: list[dict[str, float | None]] = []
    all_labels: list[int] = []
    per_file_counts: list[tuple[str, Counter[int], int]] = []
    missing_total = 0
    for record in workbooks:
        rows = record["rows"]
        labels = record["labels"]
        missing = record["missing"]
        assert isinstance(rows, list)
        assert isinstance(labels, list)
        assert isinstance(missing, Counter)
        if len(rows) != len(labels):
            raise ValueError(f"Row/label count mismatch in {record['name']}")
        file_counts: Counter[int] = Counter(labels)
        per_file_counts.append(
            (f"{record['group']}/{record['name']}", file_counts, len(rows))
        )
        all_rows.extend(rows)
        all_labels.extend(labels)
        missing_total += sum(missing.values())

    class_counts = Counter(all_labels)
    if set(class_counts) != {0, 1}:
        raise ValueError(
            f"Both label classes 0 and 1 are required; found {dict(class_counts)}"
        )
    if len(all_rows) != len(all_labels):
        raise ValueError("Feature and target row counts do not match")

    feature_matrix = np.asarray(
        [
            [
                row[feature] if row[feature] is not None else np.nan
                for feature in feature_names
            ]
            for row in all_rows
        ],
        dtype=float,
    )
    target = np.asarray(all_labels, dtype=int)
    if not np.isfinite(feature_matrix[~np.isnan(feature_matrix)]).all():
        raise ValueError("Sensor features contain non-finite numeric values")

    row_indices = np.arange(len(target))
    train_indices, test_indices = train_test_split(
        row_indices,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=target,
    )
    x_train = feature_matrix[train_indices]
    x_test = feature_matrix[test_indices]
    y_train = target[train_indices]
    y_test = target[test_indices]

    pipeline = Pipeline(
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
    pipeline.fit(x_train, y_train)
    predictions = pipeline.predict(x_test)
    retained_mask = pipeline.named_steps["variance"].get_support()
    retained_features = [
        feature for feature, retained in zip(feature_names, retained_mask) if retained
    ]
    if not retained_features:
        raise ValueError("No non-constant sensor features remain for classification")

    constant_features = [
        feature for feature, retained in zip(feature_names, retained_mask) if not retained
    ]
    train_counts = Counter(y_train.tolist())
    test_counts = Counter(y_test.tolist())
    matrix = confusion_matrix(y_test, predictions, labels=[0, 1])
    class_report = classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["Normal (0)", "Anomalous event (1)"],
        digits=4,
        zero_division=0,
    )

    train_keys = {
        tuple(None if np.isnan(value) else float(value) for value in feature_matrix[index])
        for index in train_indices
    }
    test_keys = {
        tuple(None if np.isnan(value) else float(value) for value in feature_matrix[index])
        for index in test_indices
    }
    cross_split_duplicate_vectors = len(train_keys & test_keys)

    lines = [
        "SPS supervised event baseline (initial row-level experiment)",
        "=" * 61,
        'Target: Labels; the cited paper defines "normal samples (0)" and '
        '"anomalous event samples (1)".',
        "This is event/anomaly classification on the selected dataset, not a confirmed water-leak detector.",
        "Label interpretation source: WDSEventDB paper supplied by the project owner: "
        "https://www.researchgate.net/publication/404116448_WDSEventDB_A_Real-Time_Benchmark_Dataset_for_Event_Diagnosis_in_Water_Distribution_Networks",
        "The source page returned HTTP 403 when checked during this run; the supplied paper excerpt is "
        "cross-checked against the workbook label counts below.",
        f"Random seed: {RANDOM_STATE}",
        f"Split: stratified random rows; test_size={TEST_SIZE:.0%}.",
        "Model: RandomForestClassifier(n_estimators=300, class_weight='balanced_subsample').",
        f"scikit-learn version: {sklearn.__version__}",
        "",
        "Dataset and target counts:",
        f"- Total rows={len(target)}; sensor features={len(feature_names)}; "
        f"sensor missing values={missing_total}.",
        f"- Overall classes: normal (0)={class_counts[0]}, anomalous event (1)={class_counts[1]}.",
        "- Workbook-level target counts:",
    ]
    for source, counts, rows_count in per_file_counts:
        lines.append(
        f"  {source}: rows={rows_count}, normal (0)={counts[0]}, "
        f"anomalous event (1)={counts[1]}"
        )
    lines.extend(
        [
        "",
        "Feature and leakage checks:",
        "- X contains sensor measurements only; Labels is y and is explicitly excluded from X.",
        "- File names, paths, and folder/category names are not model inputs.",
        f"- Constant features in training split (removed by VarianceThreshold): "
        f"{', '.join(constant_features) if constant_features else 'none'}.",
        f"- Features retained: {', '.join(retained_features)}.",
        "- Median imputation, constant-feature filtering, and classifier are in one Pipeline; "
        "all fitting is performed on training rows only.",
        f"- Exact sensor-vector signatures appearing in both train and test: "
        f"{cross_split_duplicate_vectors}; repeated measurements may therefore contribute to optimistic "
        "row-level metrics.",
        "",
        "Train/test class counts:",
        f"- Train: total={len(y_train)}, normal (0)={train_counts[0]}, "
        f"anomalous event (1)={train_counts[1]}.",
        f"- Test: total={len(y_test)}, normal (0)={test_counts[0]}, "
        f"anomalous event (1)={test_counts[1]}.",
        "",
        "Held-out row-level metrics (positive class: anomalous event=1):",
        f"- Precision: {precision_score(y_test, predictions, pos_label=1, zero_division=0):.4f}",
        f"- Recall: {recall_score(y_test, predictions, pos_label=1, zero_division=0):.4f}",
        f"- F1: {f1_score(y_test, predictions, pos_label=1, zero_division=0):.4f}",
        f"- Balanced accuracy: {balanced_accuracy_score(y_test, predictions):.4f}",
        "Confusion matrix (rows=true, columns=predicted; order [0, 1]):",
        f"  [[{matrix[0, 0]}, {matrix[0, 1]}],",
        f"   [{matrix[1, 0]}, {matrix[1, 1]}]]",
        "Classification report:",
        class_report.rstrip(),
        "",
        "Limitations:",
        "- The stratified split is by row, not by workbook or event. Rows can be temporally dependent "
        "because per-row timestamps are unavailable; this is not independent event-level validation.",
        "- A workbook may contribute rows to both train and test. Workbook-level conditions and repeated "
        "sensor patterns may make these metrics optimistic.",
        "- Labels distinguishes normal from anomalous event samples per the cited paper; the positive class "
        "includes anomaly events and must not be interpreted as leak-only unless event type is independently "
        "established for each row.",
        "- Results apply only to this dataset and do not establish performance on SPS residential sensors.",
        ]
    )

    model_payload = {
        "model": pipeline,
        "feature_names": feature_names,
        "retained_feature_names": retained_features,
        "target_name": LABEL_COLUMN,
        "target_mapping": {0: "normal sample", 1: "anomalous event sample"},
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "model_name": "RandomForestClassifier",
        "dataset_scope": "WDSEventDB Clean plus Leakage workbooks",
    }
    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_payload, MODEL_FILE)
    lines.extend(
        [
        "",
        f"Model artifact: {MODEL_FILE.relative_to(PROJECT_DIR)}",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    try:
        report = make_report()
        REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
        REPORT_FILE.write_text(report, encoding="utf-8")
    except (OSError, KeyError, ValueError, zipfile.BadZipFile, ET.ParseError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    print(report, end="")
    print(f"Report saved to: {REPORT_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
