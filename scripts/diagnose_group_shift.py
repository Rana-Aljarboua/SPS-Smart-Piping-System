from __future__ import annotations

import re
import sys
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np

from train_baseline import LABEL_COLUMN, load_experiment_data


PROJECT_DIR = Path(__file__).resolve().parents[1]
GROUPED_REPORT = PROJECT_DIR / "reports" / "sps_grouped_evaluation.txt"
REPORT_FILE = PROJECT_DIR / "reports" / "sps_group_shift_diagnosis.txt"
LABELS = (0, 1)


def feature_matrix(
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


def describe(values: np.ndarray) -> tuple[int, float, float, float, float, float]:
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        return 0, *(float("nan"),) * 5
    return (
        int(finite_values.size),
        float(np.mean(finite_values)),
        float(np.median(finite_values)),
        float(np.std(finite_values, ddof=1)) if finite_values.size > 1 else 0.0,
        float(np.min(finite_values)),
        float(np.max(finite_values)),
    )


def standardized_mean_difference(
    train_values: np.ndarray, held_out_values: np.ndarray
) -> float | None:
    train = train_values[np.isfinite(train_values)]
    held_out = held_out_values[np.isfinite(held_out_values)]
    if train.size < 2 or held_out.size < 2:
        return None

    pooled_variance = (
        (train.size - 1) * np.var(train, ddof=1)
        + (held_out.size - 1) * np.var(held_out, ddof=1)
    ) / (train.size + held_out.size - 2)
    if pooled_variance <= 0:
        return None
    return float((np.mean(held_out) - np.mean(train)) / np.sqrt(pooled_variance))


def parse_grouped_report(path: Path) -> dict[str, dict[str, object]]:
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
        if line.strip().startswith("[["):
            matrix_lines.append(line.strip())
        elif matrix_lines and line.strip().startswith("["):
            matrix_lines.append(line.strip())
            joined = " ".join(matrix_lines)
            numbers = [int(value) for value in re.findall(r"\d+", joined)]
            if len(numbers) == 4:
                results[current]["confusion_matrix"] = [
                    numbers[:2],
                    numbers[2:],
                ]
                current = None
                matrix_lines = []
    if not results or any("confusion_matrix" not in result for result in results.values()):
        raise ValueError(f"Could not parse per-workbook confusion matrices from {path}")
    return results


def make_report() -> str:
    if not GROUPED_REPORT.is_file():
        raise FileNotFoundError(f"Grouped evaluation report not found: {GROUPED_REPORT}")

    feature_names, records = load_experiment_data()
    if LABEL_COLUMN in feature_names:
        raise ValueError("Labels must not be included among sensor features")
    grouped_results = parse_grouped_report(GROUPED_REPORT)

    matrices: dict[str, np.ndarray] = {}
    targets: dict[str, np.ndarray] = {}
    groups: dict[str, str] = {}
    counts_by_workbook: dict[str, Counter[int]] = {}
    for record in records:
        name = f"{record['group']}/{record['name']}"
        rows = record["rows"]
        labels = record["labels"]
        if not isinstance(rows, list) or not isinstance(labels, list):
            raise ValueError(f"Malformed workbook contents: {name}")
        if len(rows) != len(labels):
            raise ValueError(f"Feature/label length mismatch: {name}")
        matrices[name] = feature_matrix(rows, feature_names)
        targets[name] = np.asarray(labels, dtype=int)
        groups[name] = str(record["group"])
        counts_by_workbook[name] = Counter(labels)

    leakage_names = [name for name in matrices if groups[name] == "Leakage"]
    if set(leakage_names) != set(grouped_results):
        raise ValueError(
            "The Leakage workbook names in the grouped evaluation report do not match "
            "the current dataset files"
        )

    lines = [
        "SPS grouped generalization: descriptive shift diagnosis",
        "=" * 58,
        f"Grouped evaluation source: {GROUPED_REPORT.relative_to(PROJECT_DIR)}",
        f"Workbooks analyzed: {len(matrices)}; sensor features: {len(feature_names)}.",
        "Labels is used only to form descriptive subsets, never as a sensor feature.",
        "No model was fitted, retrained, or modified in this diagnosis.",
        "",
        "Grouped-evaluation errors by held-out workbook:",
        "Confusion matrices use rows=true, columns=predicted; label order [0, 1].",
    ]
    for name in leakage_names:
        matrix = grouped_results[name]["confusion_matrix"]
        assert isinstance(matrix, list)
        false_positives = matrix[0][1]
        false_negatives = matrix[1][0]
        lines.append(
            f"- {name}: FP={false_positives}, FN={false_negatives}, "
            f"total_errors={false_positives + false_negatives}, "
            f"class-0 rows={counts_by_workbook[name][0]}, "
            f"class-1 rows={counts_by_workbook[name][1]}"
        )

    lines.extend(
        [
            "",
            "Sensor-feature summaries by workbook and Labels class:",
            "Each feature/class row reports count, mean, median, sample SD, minimum, maximum.",
            "A class count of zero means that workbook has no rows of that class.",
        ]
    )
    for name in matrices:
        matrix = matrices[name]
        labels = targets[name]
        counts = counts_by_workbook[name]
        lines.append(
            f"\n[{name}] rows={len(labels)}; Labels=0:{counts[0]}, Labels=1:{counts[1]}"
        )
        for label in LABELS:
            subset = matrix[labels == label]
            lines.append(f"  Labels={label}:")
            for index, feature in enumerate(feature_names):
                stats = describe(subset[:, index]) if subset.size else (
                    0,
                    float("nan"),
                    float("nan"),
                    float("nan"),
                    float("nan"),
                    float("nan"),
                )
                lines.append(
                    f"    {feature}: n={stats[0]}, mean={stats[1]:.6g}, "
                    f"median={stats[2]:.6g}, sd={stats[3]:.6g}, "
                    f"min={stats[4]:.6g}, max={stats[5]:.6g}"
                )

    lines.extend(
        [
            "",
            "Largest same-label distribution shifts: held-out workbook vs its training workbooks",
            "Ranked by absolute standardized mean difference (Cohen d); positive means held-out mean is higher.",
            "Comparison uses the same Labels class in both sides to reduce class-mixture confounding.",
        ]
    )
    for held_out in leakage_names:
        training_names = [name for name in matrices if name != held_out]
        lines.append(f"\n[{held_out}]")
        for label in LABELS:
            train_class_parts = [
                matrices[name][targets[name] == label]
                for name in training_names
                if np.any(targets[name] == label)
            ]
            held_class = matrices[held_out][targets[held_out] == label]
            if not train_class_parts or held_class.shape[0] == 0:
                lines.append(f"  Labels={label}: comparison unavailable (class absent)")
                continue
            train_class = np.concatenate(train_class_parts, axis=0)
            shifts: list[tuple[float, str, float, float, float, float]] = []
            undefined: list[str] = []
            for index, feature in enumerate(feature_names):
                train_feature = train_class[:, index]
                held_feature = held_class[:, index]
                train_stats = describe(train_feature)
                held_stats = describe(held_feature)
                effect = standardized_mean_difference(train_feature, held_feature)
                mean_delta = held_stats[1] - train_stats[1]
                median_delta = held_stats[2] - train_stats[2]
                if effect is None:
                    undefined.append(f"{feature} (mean delta={mean_delta:.6g})")
                else:
                    shifts.append(
                        (abs(effect), feature, effect, mean_delta, median_delta, train_stats[3])
                    )

            lines.append(f"  Labels={label}: training n={train_class.shape[0]}, "
                         f"held-out n={held_class.shape[0]}")
            if shifts:
                lines.append("    Largest standardized shifts: feature | Cohen d | mean delta | median delta")
                for _, feature, effect, mean_delta, median_delta, _ in sorted(
                    shifts, reverse=True
                )[:5]:
                    lines.append(
                        f"      {feature} | {effect:.3f} | {mean_delta:.6g} | {median_delta:.6g}"
                    )
            if undefined:
                lines.append(
                    "    Standardized effect undefined (insufficient values or zero pooled SD): "
                    + "; ".join(undefined)
                )

    lines.extend(
        [
            "",
            "Constant-feature check by grouped fold:",
            "Features constant across the training workbooks but variable in the held-out workbook "
            "are listed as a possible shift indicator, not as a cause.",
        ]
    )
    for held_out in leakage_names:
        training_names = [name for name in matrices if name != held_out]
        constant_in_train: list[str] = []
        changed_in_holdout: list[str] = []
        for index, feature in enumerate(feature_names):
            train_values = np.concatenate(
                [matrices[name][:, index] for name in training_names]
            )
            train_values = train_values[np.isfinite(train_values)]
            held_values = matrices[held_out][:, index]
            held_values = held_values[np.isfinite(held_values)]
            if train_values.size and np.min(train_values) == np.max(train_values):
                constant_in_train.append(feature)
                if held_values.size and np.min(held_values) != np.max(held_values):
                    changed_in_holdout.append(feature)
        lines.append(
            f"- {held_out}: constant in training={', '.join(constant_in_train) or 'none'}; "
            f"variable in held-out={', '.join(changed_in_holdout) or 'none'}"
        )

    lines.extend(
        [
            "",
            "Interpretation:",
            "- The grouped report shows errors concentrated in anomalous class 1: the held-out Leakage "
            "workbooks had many false negatives and almost no false positives.",
            "- Same-class distribution differences and between-workbook shifts can be associated with these "
            "failures descriptively, but this analysis cannot establish causation.",
            "- The grouped report contains aggregate confusion matrices, not row-level predictions. It is "
            "therefore impossible to test whether individual false negatives correlate with a particular "
            "feature value or constant-feature transition.",
            "- Constant features can encode workbook-specific operating states or reduce usable signals, but "
            "this report does not prove they caused the classifier behavior.",
            "- Labels=1 denotes anomalous event samples, not confirmed leak-only samples. No leak-detection "
            "claim is made.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    if REPORT_FILE.exists():
        print(f"ERROR: refusing to overwrite existing report: {REPORT_FILE}", file=sys.stderr)
        return 1
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
