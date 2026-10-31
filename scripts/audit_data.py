#!/usr/bin/env python3
"""Read-only audit for the SPS water-network CSV."""

import argparse
import csv
import math
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path


MISSING_TOKENS = {"", "na", "n/a", "nan", "null", "none"}
EXPECTED_SCORE_RANGE = (0.0, 100.0)
EXPECTED_INTERVAL = timedelta(hours=1)
LEAK_WINDOW = timedelta(days=7)


def parse_timestamp(value):
    try:
        return datetime.strptime(value.strip(), "%d-%m-%Y %H:%M")
    except (AttributeError, ValueError):
        return None


def parse_number(value):
    if value is None or value.strip().lower() in MISSING_TOKENS:
        return None
    try:
        result = float(value)
    except ValueError:
        return None
    return result if math.isfinite(result) else None


def pearson_correlation(left, right):
    pairs = [(x, y) for x, y in zip(left, right) if x is not None and y is not None]
    if len(pairs) < 2:
        return len(pairs), None

    left_values, right_values = zip(*pairs)
    left_mean = sum(left_values) / len(pairs)
    right_mean = sum(right_values) / len(pairs)
    covariance = sum(
        (x - left_mean) * (y - right_mean) for x, y in pairs
    )
    left_sum_squares = sum((x - left_mean) ** 2 for x in left_values)
    right_sum_squares = sum((y - right_mean) ** 2 for y in right_values)
    denominator = math.sqrt(left_sum_squares * right_sum_squares)
    if denominator == 0:
        return len(pairs), None
    return len(pairs), covariance / denominator


def read_dataset(path):
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        headers = reader.fieldnames
        if not headers:
            raise ValueError("CSV has no header row")
        rows = list(reader)
    return headers, rows


def audit(path):
    headers, rows = read_dataset(path)
    duplicate_columns = [
        name for name, count in Counter(headers).items() if count > 1
    ]
    malformed_rows = sum(None in row or None in row.values() for row in rows)

    feature_columns = [
        column for column in headers if column not in {"timestamp", "fault_d7"}
    ]
    missing_counts = Counter()
    invalid_numeric_counts = Counter()
    out_of_range_counts = Counter()
    parsed_features = {column: [] for column in feature_columns}
    timestamps = []
    labels = []
    observations = []
    exact_rows = Counter()
    monthly_labels = defaultdict(Counter)
    invalid_timestamps = 0
    invalid_labels = 0
    labels_outside_binary = 0
    score_min, score_max = EXPECTED_SCORE_RANGE

    for row in rows:
        exact_rows[tuple(row.get(header) for header in headers)] += 1

        for column in headers:
            value = row.get(column)
            if value is None or value.strip().lower() in MISSING_TOKENS:
                missing_counts[column] += 1

        timestamp = parse_timestamp(row.get("timestamp"))
        if timestamp is None:
            invalid_timestamps += 1
        else:
            timestamps.append(timestamp)

        label_value = row.get("fault_d7")
        try:
            label = int(label_value)
            if str(label) != (label_value or "").strip() or label not in (0, 1):
                raise ValueError
        except (TypeError, ValueError):
            label = None
            invalid_labels += 1
            labels_outside_binary += 1
        labels.append(label)
        if timestamp is not None and label is not None:
            monthly_labels[timestamp.strftime("%Y-%m")][label] += 1
            observations.append((timestamp, label))

        for column in feature_columns:
            value = row.get(column)
            number = parse_number(value)
            if value is None or value.strip().lower() in MISSING_TOKENS:
                parsed_features[column].append(None)
                continue
            if number is None:
                invalid_numeric_counts[column] += 1
                parsed_features[column].append(None)
                continue
            parsed_features[column].append(number)
            if not score_min <= number <= score_max:
                out_of_range_counts[column] += 1

    timestamp_counts = Counter(timestamps)
    duplicate_timestamp_rows = sum(count - 1 for count in timestamp_counts.values())
    source_order_reversals = 0
    for previous, current in zip(timestamps, timestamps[1:]):
        if current < previous:
            source_order_reversals += 1

    chronological_timestamps = sorted(timestamp_counts)
    unusual_intervals = 0
    forward_gaps = 0
    subhour_intervals = 0
    missing_hour_slots = 0

    for previous, current in zip(
        chronological_timestamps, chronological_timestamps[1:]
    ):
        difference = current - previous
        if difference == EXPECTED_INTERVAL:
            continue
        unusual_intervals += 1
        if difference < EXPECTED_INTERVAL:
            subhour_intervals += 1
        else:
            forward_gaps += 1
            if difference.total_seconds() % EXPECTED_INTERVAL.total_seconds() == 0:
                missing_hour_slots += (
                    int(difference.total_seconds() / EXPECTED_INTERVAL.total_seconds())
                    - 1
                )

    print(f"Input: {path}")
    print(f"Rows: {len(rows)}")
    print(f"Columns: {len(headers)}")
    print(f"Duplicate column names: {duplicate_columns or 'none'}")
    print(f"Rows with unexpected CSV field count: {malformed_rows}")
    print(f"Exact duplicate rows (extra copies): {sum(n - 1 for n in exact_rows.values())}")
    print(f"Invalid timestamps: {invalid_timestamps}")
    print(f"Duplicate timestamps (extra rows): {duplicate_timestamp_rows}")

    if timestamps:
        earliest = min(timestamps)
        latest = max(timestamps)
        print(f"Earliest timestamp: {earliest}")
        print(f"Latest timestamp: {latest}")
        print(f"Out-of-order adjacent timestamp pairs in source rows: {source_order_reversals}")
        print(
            "Intervals not exactly one hour after chronological sorting "
            "and timestamp deduplication: "
            f"{unusual_intervals}"
        )
        print(
            "  Forward gaps: "
            f"{forward_gaps}; missing whole-hour slots within aligned gaps: "
            f"{missing_hour_slots}"
        )
        print(f"  Sub-hour intervals: {subhour_intervals}")
    else:
        earliest = latest = None
        print("No valid timestamps available for interval or split analysis.")

    print(f"Invalid fault_d7 values: {invalid_labels}")
    print(f"fault_d7 values outside binary labels 0/1: {labels_outside_binary}")
    print("Monthly fault_d7 counts (rows with valid timestamp and label):")
    for month in sorted(monthly_labels):
        counts = monthly_labels[month]
        print(f"  {month}: 0={counts[0]}, 1={counts[1]}")

    print("Missing values by column:")
    for column in headers:
        print(f"  {column}: {missing_counts[column]}")

    if invalid_numeric_counts:
        print("Non-numeric/non-finite values in non-missing feature cells:")
        for column, count in sorted(invalid_numeric_counts.items()):
            print(f"  {column}: {count}")
    else:
        print("Non-numeric/non-finite values in non-missing feature cells: none")

    if out_of_range_counts:
        print(
            "Feature values outside documented anomaly-score range [0, 100]:"
        )
        for column, count in sorted(out_of_range_counts.items()):
            print(f"  {column}: {count}")
    else:
        print(
            "Feature values outside documented anomaly-score range [0, 100]: none"
        )

    numeric_label = [float(label) if label is not None else None for label in labels]
    correlations = []
    for column in feature_columns:
        count, correlation = pearson_correlation(
            parsed_features[column], numeric_label
        )
        if correlation is not None:
            correlations.append((abs(correlation), column, correlation, count))
    correlations.sort(reverse=True)
    print("Largest absolute Pearson correlations with fault_d7 (pairwise complete):")
    if correlations:
        for _, column, correlation, count in correlations[:10]:
            print(f"  {column}: r={correlation:.4f}, n={count}")
    else:
        print("  Not computable.")

    print(
        "Correlation caveat: association is descriptive, not proof of leakage. "
        "Source documentation says feature values are precomputed anomaly scores; "
        "their fit window and whether future data/events influenced them are not "
        "established by this CSV."
    )
    print(
        "Range caveat: [0, 100] is the documented anomaly-score scale. "
        "Out-of-range values are flagged for review, not changed."
    )
    print(
        "Data caveat: fault_d7 labels proximity within +/-7 days of recorded "
        "leak events; it does not mean a leak was present at that exact hour."
    )

    if earliest is not None and len(observations) >= 3:
        ordered = sorted(set(timestamps))
        first_cut = ordered[int(len(ordered) * 0.70)]
        second_cut = ordered[int(len(ordered) * 0.85)]
        print("Suggested chronological split (not applied):")
        print(f"  Candidate boundaries: {first_cut} and {second_cut}")
        print(
            "  Purge rows within +/-7 days of either boundary; this removes "
            "records whose event-proximity label windows may cross partitions."
        )
        partitions = {
            "train": [],
            "validation": [],
            "test": [],
        }
        purged = 0
        for timestamp, label in observations:
            if (
                abs(timestamp - first_cut) <= LEAK_WINDOW
                or abs(timestamp - second_cut) <= LEAK_WINDOW
            ):
                purged += 1
            elif timestamp < first_cut:
                partitions["train"].append((timestamp, label))
            elif timestamp < second_cut:
                partitions["validation"].append((timestamp, label))
            else:
                partitions["test"].append((timestamp, label))
        print(f"  Rows purged around boundaries: {purged}")
        for name, partition in partitions.items():
            counts = Counter(label for _, label in partition)
            if partition:
                period = (
                    f"{min(timestamp for timestamp, _ in partition)} .. "
                    f"{max(timestamp for timestamp, _ in partition)}"
                )
            else:
                period = "empty"
            print(
                f"  {name}: n={len(partition)}, 0={counts[0]}, 1={counts[1]}, "
                f"range={period}"
            )
        print(
            "  Also verify/recompute anomaly-score features using training-only "
            "history and strictly causal windows before claiming a leakage-safe "
            "evaluation."
        )


def main():
    project_root = Path(__file__).resolve().parents[1]
    default_csv = project_root / "data" / "raw" / "water_network.csv"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "csv_path",
        nargs="?",
        type=Path,
        default=default_csv,
        help=f"CSV to inspect (default: {default_csv})",
    )
    args = parser.parse_args()
    if not args.csv_path.is_file():
        parser.error(f"CSV file not found: {args.csv_path}")
    audit(args.csv_path)


if __name__ == "__main__":
    main()
