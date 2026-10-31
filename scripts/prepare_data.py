from __future__ import annotations

import posixpath
import re
import sys
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter, defaultdict
from pathlib import Path


DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "processed" / "WDSEventDB"
NAMESPACE = {
    "main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
    "office_rel": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "package_rel": "http://schemas.openxmlformats.org/package/2006/relationships",
}
EXPECTED_COLUMNS = (
    "Pressure 1 Out",
    "Pressure 2 Out",
    "Pressure 3 In",
    "Pressure 4 In",
    "Water Flow 1",
    "Water Flow 2",
    "Water Flow 3",
    "Water Flow 4",
    "VFD 1",
    "VFD 2",
    "VFD 3",
    "VFD 4-1",
    "VFD 4-2",
    "Analog Valve 1",
    "Analog Valve 2",
    "Labels",
)


def column_index(cell_reference: str) -> int:
    letters = re.match(r"[A-Z]+", cell_reference.upper())
    if letters is None:
        raise ValueError(f"Invalid Excel cell reference: {cell_reference}")
    index = 0
    for letter in letters.group():
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


def read_workbook(path: Path) -> tuple[str, list[str], list[list[object]]]:
    with zipfile.ZipFile(path) as workbook:
        shared_strings: list[str] = []
        if "xl/sharedStrings.xml" in workbook.namelist():
            string_root = ET.fromstring(workbook.read("xl/sharedStrings.xml"))
            shared_strings = [
                "".join(text.text or "" for text in item.findall(".//main:t", NAMESPACE))
                for item in string_root.findall("main:si", NAMESPACE)
            ]

        workbook_root = ET.fromstring(workbook.read("xl/workbook.xml"))
        relationships_root = ET.fromstring(workbook.read("xl/_rels/workbook.xml.rels"))
        relationships = {
            relationship.attrib["Id"]: relationship.attrib["Target"]
            for relationship in relationships_root.findall("package_rel:Relationship", NAMESPACE)
        }
        sheet = workbook_root.find("main:sheets/main:sheet", NAMESPACE)
        if sheet is None:
            raise ValueError(f"No worksheet found in {path.name}")

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
        raise ValueError(f"No rows found in {path.name}")
    headers = [str(value) if value is not None else "" for value in rows[0]]
    return sheet.attrib["name"], headers, rows[1:]


def summarize_file(path: Path) -> dict[str, object]:
    category = path.parent.name
    sheet_name, headers, rows = read_workbook(path)
    if tuple(headers) != EXPECTED_COLUMNS:
        raise ValueError(
            f"{path}: unexpected columns; got {headers!r}, expected {list(EXPECTED_COLUMNS)!r}"
        )

    label_counts: Counter[int] = Counter()
    missing_by_column: Counter[str] = Counter()
    seen_rows: set[tuple[object, ...]] = set()
    duplicate_rows = 0

    for row_number, row in enumerate(rows, start=2):
        if len(row) < len(headers):
            row.extend([None] * (len(headers) - len(row)))
        if len(row) > len(headers):
            raise ValueError(f"{path}: row {row_number} has more cells than the header")

        for header, value in zip(headers, row):
            if value is None or value == "":
                missing_by_column[header] += 1

        label = row[headers.index("Labels")]
        if label not in (0, 1):
            raise ValueError(f"{path}: invalid Labels value {label!r} in row {row_number}")
        label_counts[label] += 1

        row_key = tuple(row)
        if row_key in seen_rows:
            duplicate_rows += 1
        else:
            seen_rows.add(row_key)

    timestamp_columns = [
        header
        for header in headers
        if any(term in header.lower() for term in ("timestamp", "datetime", "date", "time"))
    ]
    return {
        "category": category,
        "file": path.name,
        "sheet": sheet_name,
        "rows": len(rows),
        "labels": label_counts,
        "missing": missing_by_column,
        "duplicates": duplicate_rows,
        "timestamp_columns": timestamp_columns,
    }


def main() -> int:
    if not DATA_DIR.is_dir():
        print(f"ERROR: data directory not found: {DATA_DIR}", file=sys.stderr)
        return 1

    files = sorted(DATA_DIR.glob("*/*.xlsx"))
    if len(files) != 9:
        print(f"ERROR: expected 9 XLSX files, found {len(files)}", file=sys.stderr)
        return 1

    try:
        summaries = [summarize_file(path) for path in files]
    except (OSError, KeyError, ValueError, zipfile.BadZipFile, ET.ParseError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    category_rows: Counter[str] = Counter()
    category_labels: dict[str, Counter[int]] = defaultdict(Counter)
    total_labels: Counter[int] = Counter()
    total_missing = 0
    total_duplicates = 0

    print(f"Data directory: {DATA_DIR}")
    print(f"Workbook files: {len(summaries)}")
    print(f"Expected columns ({len(EXPECTED_COLUMNS)}): {', '.join(EXPECTED_COLUMNS)}")
    print("Per-file results:")
    for summary in summaries:
        labels = summary["labels"]
        missing = summary["missing"]
        missing_total = sum(missing.values())
        category = str(summary["category"])
        rows = int(summary["rows"])
        category_rows[category] += rows
        category_labels[category].update(labels)
        total_labels.update(labels)
        total_missing += missing_total
        total_duplicates += int(summary["duplicates"])
        print(
            f"  {category}/{summary['file']}: rows={rows}, "
            f"Labels=0:{labels[0]} 1:{labels[1]}, missing={missing_total}, "
            f"duplicate_rows={summary['duplicates']}, sheet={summary['sheet']!r}"
        )
        if missing:
            print(f"    missing_by_column={dict(missing)}")

    print("Category totals:")
    for category in sorted(category_rows):
        labels = category_labels[category]
        print(
            f"  {category}: rows={category_rows[category]}, "
            f"Labels=0:{labels[0]} 1:{labels[1]}"
        )
    print(
        f"Overall: rows={sum(category_rows.values())}, "
        f"Labels=0:{total_labels[0]} 1:{total_labels[1]}, "
        f"missing={total_missing}, duplicate_rows={total_duplicates}"
    )

    timestamp_columns = sorted(
        {column for summary in summaries for column in summary["timestamp_columns"]}
    )
    print(f"Timestamp columns: {timestamp_columns if timestamp_columns else 'none'}")
    print("Row order: retained as stored; no per-row time field or documented sampling interval was found.")
    print("Label meanings: binary values validated; numeric meanings are not asserted by this script.")
    print("Validation: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
