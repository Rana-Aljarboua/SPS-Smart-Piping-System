from __future__ import annotations

import unittest

from scripts.prepare_data import EXPECTED_COLUMNS
from sps.inference import SPSInference


class SPSInferenceContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.inference = SPSInference()

    def make_sample(self) -> dict[str, float]:
        return {feature: 0.25 for feature in self.inference.feature_names}

    def test_saved_model_schema_matches_wdseventdb_sensor_order(self) -> None:
        self.assertEqual(self.inference.feature_names, EXPECTED_COLUMNS[:-1])
        self.assertEqual(self.inference.model.n_features_in_, len(EXPECTED_COLUMNS) - 1)
        self.assertEqual(self.inference.classes, (0, 1))
        self.assertEqual(
            self.inference.target_mapping,
            {0: "normal sample", 1: "anomalous event sample"},
        )

    def test_non_mapping_sample_is_rejected(self) -> None:
        with self.assertRaisesRegex(TypeError, "must be a mapping"):
            self.inference.predict_batch([None])  # type: ignore[list-item]

    def test_boolean_sensor_value_is_rejected(self) -> None:
        sample = self.make_sample()
        sample[self.inference.feature_names[0]] = True  # type: ignore[assignment]

        with self.assertRaisesRegex(TypeError, "must be numeric"):
            self.inference.predict(sample)

    def test_empty_batch_returns_no_predictions(self) -> None:
        self.assertEqual(self.inference.predict_batch([]), [])

    def test_valid_batch_prediction_matches_saved_class_contract(self) -> None:
        predictions = self.inference.predict_batch([self.make_sample()])

        self.assertEqual(len(predictions), 1)
        prediction = predictions[0]
        self.assertIn(prediction.predicted_class, (0, 1))
        self.assertIn(
            prediction.predicted_label,
            ("normal sample", "anomalous event sample"),
        )
        self.assertEqual(
            set(prediction.class_probabilities or {}),
            set(self.inference.classes),
        )


if __name__ == "__main__":
    unittest.main()
