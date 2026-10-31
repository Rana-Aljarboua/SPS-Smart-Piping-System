from __future__ import annotations

import unittest

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from scripts.compare_wdseventdb_models import (
    WorkbookData,
    calculate_metrics,
    create_model,
    iter_grouped_folds,
    remove_train_test_feature_overlaps,
)


class ModelComparisonTests(unittest.TestCase):
    def test_grouped_folds_hold_out_exactly_one_complete_workbook(self) -> None:
        workbooks = [
            WorkbookData(
                name=f"Leakage/file{index}.xlsx",
                category="Leakage",
                features=np.asarray([[float(index), 0.0]]),
                labels=np.asarray([index % 2]),
                duplicate_rows=0,
            )
            for index in range(4)
        ]

        folds = list(iter_grouped_folds(workbooks))

        self.assertEqual(len(folds), 4)
        for held_out, training in folds:
            self.assertEqual(len(training), 3)
            self.assertNotIn(held_out, training)
            self.assertEqual(
                {item.name for item in training} | {held_out.name},
                {item.name for item in workbooks},
            )

    def test_train_rows_with_test_feature_vectors_are_removed(self) -> None:
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
        self.assertEqual(x_train.shape[0], 4)

    def test_precision_is_undefined_when_no_positive_prediction_exists(self) -> None:
        metrics = calculate_metrics(
            np.asarray([0, 0, 1, 1]),
            np.asarray([0, 0, 0, 0]),
        )

        self.assertIsNone(metrics["precision"])
        self.assertEqual(metrics["recall"], 0.0)
        self.assertEqual(metrics["f1"], 0.0)
        self.assertEqual(metrics["false_positive_rate"], 0.0)
        self.assertEqual(
            {name: metrics[name] for name in ("tp", "fp", "tn", "fn")},
            {"tp": 0, "fp": 0, "tn": 2, "fn": 2},
        )

    def test_single_class_holdout_marks_undefined_metrics_as_missing(self) -> None:
        metrics = calculate_metrics(
            np.asarray([0, 0]),
            np.asarray([0, 1]),
        )

        self.assertEqual(metrics["precision"], 0.0)
        self.assertIsNone(metrics["recall"])
        self.assertIsNone(metrics["f1"])
        self.assertIsNone(metrics["balanced_accuracy"])
        self.assertEqual(metrics["false_positive_rate"], 0.5)

    def test_logistic_regression_scaling_is_inside_pipeline(self) -> None:
        pipeline = create_model("Logistic Regression")

        self.assertIsInstance(pipeline.named_steps["scaler"], StandardScaler)
        self.assertIsInstance(pipeline.named_steps["classifier"], LogisticRegression)
        self.assertEqual(
            list(pipeline.named_steps),
            ["imputer", "variance", "scaler", "classifier"],
        )

    def test_random_forest_matches_existing_baseline_configuration(self) -> None:
        classifier = create_model("Random Forest").named_steps["classifier"]

        self.assertEqual(classifier.n_estimators, 300)
        self.assertEqual(classifier.class_weight, "balanced_subsample")
        self.assertEqual(classifier.random_state, 42)


if __name__ == "__main__":
    unittest.main()
