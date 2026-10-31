from __future__ import annotations

import unittest

import numpy as np

from scripts.evaluate_wdseventdb_grouped import (
    calculate_metrics,
    remove_train_test_feature_overlaps,
)


class GroupedEvaluationTests(unittest.TestCase):
    def test_metrics_include_confusion_counts_and_binary_scores(self) -> None:
        metrics = calculate_metrics(
            np.asarray([0, 0, 1, 1]),
            np.asarray([0, 1, 1, 0]),
        )

        self.assertEqual(
            {name: metrics[name] for name in ("tp", "fp", "tn", "fn")},
            {"tp": 1, "fp": 1, "tn": 1, "fn": 1},
        )
        self.assertEqual(metrics["precision"], 0.5)
        self.assertEqual(metrics["recall"], 0.5)
        self.assertEqual(metrics["f1"], 0.5)
        self.assertEqual(metrics["balanced_accuracy"], 0.5)

    def test_single_class_test_metrics_mark_undefined_scores_as_missing(self) -> None:
        metrics = calculate_metrics(np.asarray([0, 0]), np.asarray([0, 1]))

        self.assertEqual(metrics["tp"], 0)
        self.assertEqual(metrics["fp"], 1)
        self.assertEqual(metrics["tn"], 1)
        self.assertEqual(metrics["fn"], 0)
        self.assertEqual(metrics["precision"], 0.0)
        self.assertIsNone(metrics["recall"])
        self.assertIsNone(metrics["f1"])
        self.assertIsNone(metrics["balanced_accuracy"])

    def test_train_rows_matching_test_sensor_vectors_are_excluded(self) -> None:
        x_train = np.asarray([[1.0, 2.0], [1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        y_train = np.asarray([0, 1, 0, 1])
        x_test = np.asarray([[1.0, 2.0], [7.0, 8.0]])

        filtered_x, filtered_y, removed_rows, shared_vectors = (
            remove_train_test_feature_overlaps(x_train, y_train, x_test)
        )

        self.assertEqual(removed_rows, 2)
        self.assertEqual(shared_vectors, 1)
        self.assertEqual(filtered_x.tolist(), [[3.0, 4.0], [5.0, 6.0]])
        self.assertEqual(filtered_y.tolist(), [0, 1])
        self.assertEqual(x_train.tolist(), [[1.0, 2.0], [1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])


if __name__ == "__main__":
    unittest.main()
