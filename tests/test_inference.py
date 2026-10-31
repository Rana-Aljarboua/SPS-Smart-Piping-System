from __future__ import annotations

import unittest
from collections import OrderedDict
from pathlib import Path

from sps.inference import DEFAULT_MODEL_PATH, SPSInference


class SPSInferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not DEFAULT_MODEL_PATH.is_file():
            raise FileNotFoundError(f"Expected local model artifact: {DEFAULT_MODEL_PATH}")
        cls.inference = SPSInference()

    def make_sample(self) -> dict[str, float]:
        return {feature: 0.25 for feature in self.inference.feature_names}

    def test_single_sample_returns_class_and_estimator_probabilities(self) -> None:
        result = self.inference.predict(self.make_sample())

        self.assertIn(result.predicted_class, self.inference.classes)
        self.assertEqual(
            result.predicted_label,
            self.inference.target_mapping[result.predicted_class],
        )
        self.assertIsNotNone(result.class_probabilities)
        self.assertEqual(set(result.class_probabilities or {}), set(self.inference.classes))
        self.assertAlmostEqual(sum((result.class_probabilities or {}).values()), 1.0)

    def test_batch_returns_one_result_per_sample(self) -> None:
        samples = [self.make_sample(), self.make_sample()]

        results = self.inference.predict_batch(samples)

        self.assertEqual(len(results), len(samples))
        self.assertEqual(results[0], results[1])

    def test_reordered_mapping_uses_saved_feature_order(self) -> None:
        values = {
            feature: float(index + 1)
            for index, feature in enumerate(self.inference.feature_names)
        }
        reversed_values = OrderedDict(reversed(list(values.items())))

        matrix = self.inference._build_matrix([reversed_values])

        self.assertEqual(
            matrix.tolist(),
            [[values[feature] for feature in self.inference.feature_names]],
        )

    def test_missing_column_is_rejected(self) -> None:
        sample = self.make_sample()
        sample.pop(self.inference.feature_names[0])

        with self.assertRaisesRegex(ValueError, "missing columns"):
            self.inference.predict(sample)

    def test_extra_column_is_rejected(self) -> None:
        sample = self.make_sample()
        sample["unexpected"] = 1.0

        with self.assertRaisesRegex(ValueError, "unexpected columns"):
            self.inference.predict(sample)

    def test_non_numeric_value_is_rejected(self) -> None:
        sample = self.make_sample()
        sample[self.inference.feature_names[0]] = "not-a-number"  # type: ignore[assignment]

        with self.assertRaisesRegex(TypeError, "must be numeric"):
            self.inference.predict(sample)

    def test_nan_and_infinity_are_rejected(self) -> None:
        for invalid_value in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(value=invalid_value):
                sample = self.make_sample()
                sample[self.inference.feature_names[0]] = invalid_value
                with self.assertRaisesRegex(ValueError, "must be finite"):
                    self.inference.predict(sample)

    def test_custom_missing_model_path_is_reported(self) -> None:
        missing_path = Path(__file__).resolve().parent / "missing-model.joblib"
        with self.assertRaisesRegex(FileNotFoundError, "Model artifact not found"):
            SPSInference(missing_path)


if __name__ == "__main__":
    unittest.main()
