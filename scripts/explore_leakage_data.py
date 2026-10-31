from __future__ import annotations

import math
import posixpath
import re
import statistics
import sys
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data" / "processed" / "WDSEventDB"
CLEAN_FILE = DATA_DIR / "Clean" / "CleanData.xlsx"
LEAKAGE_DIR = DATA_DIR / "Leakage"
REPORT_FILE = PROJECT_DIR / "reports" / "leakage_exploration.txt"
LABEL_COLUMN = "Labels"

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
        workbook_names = workbook.namelist()
        shared_strings: list[str] = []
        if "xl/sharedStrings.xml" in workbook_names:
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
        worksheet_rows = worksheet_root.findall("main:sheetData/main:row", NAMESPACE)

        rows: list[list[object]] = []
        for row in worksheet_rows:
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


def parse_feature_value(value: object) -> tuple[float | None, bool]:
    if value is None or value == "":
        return None, False
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None, True
    if not math.isfinite(number):
        return None, True
    return number, False


def load_group(paths: list[Path], expected_features: list[str]) -> dict[str, list[float]]:
    values = {feature: [] for feature in expected_features}
    for path in paths:
        headers, rows = read_workbook(path)
        features = [header for header in headers if header != LABEL_COLUMN]
        if features != expected_features:
            raise ValueError(
                f"{path}: feature columns differ from CleanData.xlsx; "
                f"expected {expected_features!r}, found {features!r}"
            )

        feature_positions = {feature: headers.index(feature) for feature in expected_features}
        for row_number, row in enumerate(rows, start=2):
            for feature, position in feature_positions.items():
                value = row[position] if position < len(row) else None
                number, invalid = parse_feature_value(value)
                if invalid:
                    raise ValueError(
                        f"{path}, row {row_number}, feature {feature!r}: "
                        f"not a finite number ({value!r})"
                    )
                if number is not None:
                    values[feature].append(number)
    return values


def count_rows(path: Path) -> tuple[int, Counter[str]]:
    headers, rows = read_workbook(path)
    missing_by_feature: Counter[str] = Counter()
    feature_columns = [header for header in headers if header != LABEL_COLUMN]
    for row in rows:
        for feature in feature_columns:
            position = headers.index(feature)
            value = row[position] if position < len(row) else None
            if value is None or value == "":
                missing_by_feature[feature] += 1
    return len(rows), missing_by_feature


def describe(values: list[float]) -> tuple[float, float, float, float, float]:
    return (
        statistics.mean(values),
        statistics.median(values),
        statistics.stdev(values) if len(values) > 1 else 0.0,
        min(values),
        max(values),
    )


def cohen_d(clean: list[float], leakage: list[float]) -> float | None:
    if len(clean) < 2 or len(leakage) < 2:
        return None
    pooled_variance = (
        (len(clean) - 1) * statistics.variance(clean)
        + (len(leakage) - 1) * statistics.variance(leakage)
    ) / (len(clean) + len(leakage) - 2)
    if pooled_variance == 0:
        return 0.0
    return (statistics.mean(leakage) - statistics.mean(clean)) / math.sqrt(
        pooled_variance
    )


def build_report() -> str:
    if not CLEAN_FILE.is_file():
        raise FileNotFoundError(f"Clean workbook not found: {CLEAN_FILE}")
    leakage_files = sorted(LEAKAGE_DIR.glob("*.xlsx"))
    if len(leakage_files) != 3:
        raise ValueError(
            f"Expected exactly 3 Leakage workbooks, found {len(leakage_files)}"
        )

    clean_headers, clean_rows = read_workbook(CLEAN_FILE)
    if LABEL_COLUMN not in clean_headers:
        raise ValueError(f"Required label column {LABEL_COLUMN!r} missing from Clean file")
    features = [header for header in clean_headers if header != LABEL_COLUMN]
    if not features:
        raise ValueError("No sensor features found after excluding Labels")

    clean_values = load_group([CLEAN_FILE], features)
    leakage_values = load_group(leakage_files, features)

    lines = [
        "WDSEventDB Leakage vs Clean exploration",
        "=" * 40,
        "Purpose: descriptive comparison only; no model training or inference.",
        f"Clean workbook: {CLEAN_FILE.relative_to(PROJECT_DIR)} ({len(clean_rows)} rows)",
        f"Leakage workbooks: {len(leakage_files)}",
        "Sensor features compared (Labels excluded): " + ", ".join(features),
        "",
        "Rows by source file (missing values count sensor features only):",
    ]

    group_missing: dict[str, Counter[str]] = {
        "Clean": Counter(),
        "Leakage": Counter(),
    }
    clean_count, clean_missing = count_rows(CLEAN_FILE)
    group_missing["Clean"].update(clean_missing)
    lines.append(f"  Clean/CleanData.xlsx: rows={clean_count}")
    if clean_missing:
        lines.append(f"    missing_by_feature={dict(clean_missing)}")

    leakage_count = 0
    for path in leakage_files:
        row_count, missing = count_rows(path)
        leakage_count += row_count
        group_missing["Leakage"].update(missing)
        lines.append(f"  Leakage/{path.name}: rows={row_count}")
        if missing:
            lines.append(f"    missing_by_feature={dict(missing)}")
    lines.extend(
        [
            f"  Combined groups: Clean={clean_count} rows; Leakage={leakage_count} rows",
            "",
            "Feature statistics (computed from available numeric values):",
            "  Feature | Clean n/missing mean median std min max | "
            "Leakage n/missing mean median std min max | "
            "Leakage-Clean mean delta | Cohen d",
        ]
    )

    effect_sizes: list[tuple[float, str, float]] = []
    for feature in features:
        clean_feature = clean_values[feature]
        leakage_feature = leakage_values[feature]
        clean_stats = describe(clean_feature)
        leakage_stats = describe(leakage_feature)
        delta = leakage_stats[0] - clean_stats[0]
        effect = cohen_d(clean_feature, leakage_feature)
        effect_text = "n/a" if effect is None else f"{effect:.3f}"
        if effect is not None:
            effect_sizes.append((abs(effect), feature, effect))
        lines.append(
            f"  {feature} | "
            f"{len(clean_feature)}/{group_missing['Clean'][feature]} "
            f"{clean_stats[0]:.6g} {clean_stats[1]:.6g} {clean_stats[2]:.6g} "
            f"{clean_stats[3]:.6g} {clean_stats[4]:.6g} | "
            f"{len(leakage_feature)}/{group_missing['Leakage'][feature]} "
            f"{leakage_stats[0]:.6g} {leakage_stats[1]:.6g} {leakage_stats[2]:.6g} "
            f"{leakage_stats[3]:.6g} {leakage_stats[4]:.6g} | "
            f"{delta:.6g} | {effect_text}"
        )

    lines.extend(
        [
            "",
            "Largest descriptive standardized differences (absolute Cohen d):",
        ]
    )
    for magnitude, feature, effect in sorted(effect_sizes, reverse=True)[:5]:
        lines.append(f"  {feature}: Cohen d={effect:.3f} (absolute={magnitude:.3f})")

    lines.extend(
        [
            "",
            "Interpretation limits:",
            "- Every row from CleanData.xlsx is compared with every row from the three Leakage workbooks.",
            "- Labels was excluded from feature statistics and was not used to filter rows.",
            "- The reported differences are descriptive, not evidence of causal leak signatures.",
            "- No per-row timestamp is available here, so rows are not treated as an ordered time series.",
            "- The dataset documentation does not establish that Labels=1 means leak in every workbook; "
            "the Leakage-folder comparison is therefore not a confirmed binary leak-vs-normal target.",
            "- These sensor readings are from a research testbed and are not validation of SPS residential hardware.",
            "- Cohen d is a standardized mean difference, not a significance test or model score.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    try:
        report = build_report()
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
