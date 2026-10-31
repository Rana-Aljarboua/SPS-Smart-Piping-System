from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from numbers import Real
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.pipeline import Pipeline


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_DIR / "models" / "sps_event_baseline.joblib"


@dataclass(frozen=True)
class Prediction:
    """One prediction; probabilities are estimator outputs, not calibrated confidence."""

    predicted_class: int
    predicted_label: str
    class_probabilities: dict[int, float] | None


class SPSInference:
    """Validate sensor inputs and run the existing saved SPS pipeline."""

    def __init__(self, artifact_path: str | Path = DEFAULT_MODEL_PATH) -> None:
        self.artifact_path = Path(artifact_path)
        if not self.artifact_path.is_file():
            raise FileNotFoundError(f"Model artifact not found: {self.artifact_path}")

        payload: Any = joblib.load(self.artifact_path)
        if not isinstance(payload, dict):
            raise ValueError("Model artifact must contain a dictionary payload")

        feature_names = payload.get("feature_names")
        if (
            not isinstance(feature_names, list)
            or not feature_names
            or any(not isinstance(name, str) or not name for name in feature_names)
            or len(set(feature_names)) != len(feature_names)
        ):
            raise ValueError("Model artifact has an invalid feature_names schema")

        model = payload.get("model")
        if not isinstance(model, Pipeline) or not callable(
            getattr(model, "predict", None)
        ):
            raise ValueError("Model artifact does not contain a prediction Pipeline")
        if getattr(model, "n_features_in_", None) != len(feature_names):
            raise ValueError(
                "Model input feature count does not match the saved feature schema"
            )

        target_mapping = payload.get("target_mapping")
        if not isinstance(target_mapping, dict):
            raise ValueError("Model artifact has no valid target_mapping")

        self.feature_names: tuple[str, ...] = tuple(feature_names)
        self.model = model
        self.target_mapping: dict[int, str] = {}
        for raw_class, raw_name in target_mapping.items():
            try:
                class_id = int(raw_class)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Model target mapping has invalid class key: {raw_class!r}"
                ) from error
            if not isinstance(raw_name, str):
                raise ValueError(
                    f"Model target mapping for class {class_id} must be a string"
                )
            self.target_mapping[class_id] = raw_name

        model_classes = getattr(model, "classes_", None)
        if model_classes is None:
            raise ValueError("Saved prediction Pipeline has no fitted class labels")
        self.classes = tuple(int(class_id) for class_id in model_classes)
        if not self.classes:
            raise ValueError("Saved prediction Pipeline has no classes")

    def _build_matrix(
        self, samples: Sequence[Mapping[str, object]]
    ) -> np.ndarray:
        matrix: list[list[float]] = []
        expected = set(self.feature_names)

        for row_index, sample in enumerate(samples):
            if not isinstance(sample, Mapping):
                raise TypeError(f"Sample {row_index} must be a mapping of sensor values")
            if any(not isinstance(name, str) for name in sample):
                raise TypeError(f"Sample {row_index} feature names must be strings")

            actual = set(sample.keys())
            missing = sorted(expected - actual)
            extra = sorted(actual - expected)
            if missing or extra:
                problems: list[str] = []
                if missing:
                    problems.append(f"missing columns: {', '.join(missing)}")
                if extra:
                    problems.append(f"unexpected columns: {', '.join(extra)}")
                raise ValueError(f"Sample {row_index} has " + "; ".join(problems))

            values: list[float] = []
            for feature in self.feature_names:
                value = sample[feature]
                if isinstance(value, bool) or not isinstance(value, (Real, Decimal)):
                    raise TypeError(
                        f"Sample {row_index}, feature {feature!r} must be numeric"
                    )
                numeric_value = float(value)
                if not np.isfinite(numeric_value):
                    raise ValueError(
                        f"Sample {row_index}, feature {feature!r} must be finite"
                    )
                values.append(numeric_value)
            matrix.append(values)

        return np.asarray(matrix, dtype=float).reshape(len(samples), len(self.feature_names))

    def predict(self, sample: Mapping[str, object]) -> Prediction:
        """Predict one sample from sensor values keyed by the saved feature names."""
        return self.predict_batch([sample])[0]

    def predict_batch(
        self, samples: Sequence[Mapping[str, object]]
    ) -> list[Prediction]:
        """Predict a batch; each sample must provide exactly the saved sensor schema."""
        if isinstance(samples, (str, bytes)) or not isinstance(samples, Sequence):
            raise TypeError("samples must be a sequence of sensor-value mappings")
        if not samples:
            return []

        matrix = self._build_matrix(samples)
        predictions = np.asarray(self.model.predict(matrix)).reshape(-1)
        if predictions.size != len(samples):
            raise ValueError("Model returned a different number of predictions than inputs")

        probability_rows: np.ndarray | None = None
        predict_proba = getattr(self.model, "predict_proba", None)
        if callable(predict_proba):
            probability_rows = np.asarray(predict_proba(matrix), dtype=float)
            if probability_rows.shape != (len(samples), len(self.classes)):
                raise ValueError("Model returned an invalid probability matrix shape")
            if not np.isfinite(probability_rows).all():
                raise ValueError("Model returned non-finite class probabilities")

        results: list[Prediction] = []
        for index, raw_prediction in enumerate(predictions):
            predicted_class = int(raw_prediction)
            if predicted_class not in self.classes:
                raise ValueError(
                    f"Model predicted class {predicted_class}, which is not in its fitted classes"
                )

            probabilities = None
            if probability_rows is not None:
                probabilities = {
                    class_id: float(probability_rows[index, class_index])
                    for class_index, class_id in enumerate(self.classes)
                }

            results.append(
                Prediction(
                    predicted_class=predicted_class,
                    predicted_label=self.target_mapping.get(
                        predicted_class, f"Class {predicted_class}"
                    ),
                    class_probabilities=probabilities,
                )
            )
        return results
