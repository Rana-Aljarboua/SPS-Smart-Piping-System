from __future__ import annotations

import math
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from types import MappingProxyType
from typing import Protocol

from scripts.train_baseline import LABEL_COLUMN, load_experiment_data

from sps.inference import Prediction, SPSInference


class InferenceEngine(Protocol):
    feature_names: Sequence[str]

    def predict_batch(
        self, samples: Sequence[Mapping[str, object]]
    ) -> list[Prediction]: ...


class StreamState(str, Enum):
    READY = "ready"
    RUNNING = "running"
    STOPPED = "stopped"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(frozen=True)
class StreamSourceRecord:
    """One real dataset row; row_index is zero-based among data rows, excluding header."""

    source_file: str
    source_row_index: int
    sensor_readings: Mapping[str, object]
    original_label: int | None = None
    source_timestamp: datetime | None = None

    def __post_init__(self) -> None:
        if not self.source_file or not isinstance(self.source_file, str):
            raise ValueError("source_file must be a non-empty relative source name")
        if (
            isinstance(self.source_row_index, bool)
            or not isinstance(self.source_row_index, int)
            or self.source_row_index < 0
        ):
            raise ValueError("source_row_index must be a non-negative integer")
        if not isinstance(self.sensor_readings, Mapping) or not self.sensor_readings:
            raise ValueError("sensor_readings must be a non-empty mapping")
        if LABEL_COLUMN in self.sensor_readings:
            raise ValueError(f"{LABEL_COLUMN} must be metadata, not a sensor feature")
        if self.original_label is not None and (
            isinstance(self.original_label, bool)
            or not isinstance(self.original_label, int)
            or self.original_label not in (0, 1)
        ):
            raise ValueError("original_label must be 0, 1, or None")
        object.__setattr__(
            self, "sensor_readings", MappingProxyType(dict(self.sensor_readings))
        )


@dataclass(frozen=True)
class StreamReading:
    source_file: str
    source_row_index: int
    original_label: int | None
    source_timestamp: datetime | None
    simulated_emission_time: datetime
    sensor_readings: Mapping[str, object]
    prediction: Prediction
    data_origin: str = "simulated_dataset"

    @property
    def prediction_status(self) -> str:
        if self.prediction.predicted_class == 1:
            return "anomalous event prediction"
        return "normal sample prediction"


@dataclass(frozen=True)
class StreamBatch:
    batch_index: int
    readings: tuple[StreamReading, ...]
    data_origin: str = "simulated_dataset"


def load_wdseventdb_records(
    inference: InferenceEngine,
) -> tuple[StreamSourceRecord, ...]:
    """Load the Clean and Leakage records supported by the saved model's training set."""
    feature_names, workbooks = load_experiment_data()
    if tuple(feature_names) != tuple(inference.feature_names):
        raise ValueError(
            "WDSEventDB sensor columns do not match the saved inference feature schema"
        )
    if LABEL_COLUMN in feature_names:
        raise ValueError(f"{LABEL_COLUMN} must not be used as an inference feature")

    records: list[StreamSourceRecord] = []
    for workbook in workbooks:
        group = workbook["group"]
        name = workbook["name"]
        rows = workbook["rows"]
        labels = workbook["labels"]
        if not isinstance(group, str) or not isinstance(name, str):
            raise ValueError("Dataset loader returned invalid workbook metadata")
        if not isinstance(rows, list) or not isinstance(labels, list):
            raise ValueError(f"Dataset loader returned invalid rows for {group}/{name}")
        if len(rows) != len(labels):
            raise ValueError(f"Row and label counts differ for {group}/{name}")

        source_file = f"{group}/{name}"
        for row_index, (row, label) in enumerate(zip(rows, labels)):
            if not isinstance(row, Mapping):
                raise ValueError(
                    f"Dataset loader returned an invalid row for {source_file} "
                    f"at data-row index {row_index}"
                )
            if LABEL_COLUMN in row:
                raise ValueError(
                    f"{LABEL_COLUMN} unexpectedly appears among sensor readings in {source_file}"
                )
            if not set(row) == set(feature_names):
                raise ValueError(
                    f"Sensor columns do not match the saved schema in {source_file} "
                    f"at data-row index {row_index}"
                )
            if label not in (0, 1):
                raise ValueError(
                    f"Invalid original label {label!r} in {source_file} "
                    f"at data-row index {row_index}"
                )
            records.append(
                StreamSourceRecord(
                    source_file=source_file,
                    source_row_index=row_index,
                    sensor_readings=row,
                    original_label=int(label),
                    source_timestamp=None,
                )
            )

    if not records:
        raise ValueError("WDSEventDB loader returned no Clean/Leakage data rows")
    return tuple(records)


class SimulatedSensorStream:
    """Emit dataset-backed prediction batches on a background worker thread.

    Callbacks run on the worker thread. Call ``wait`` to observe completion or
    worker errors, and ``stop`` to interrupt the interval wait and stop safely.
    A stream instance is single-use; create a new instance to replay from row 0.
    """

    def __init__(
        self,
        records: Sequence[StreamSourceRecord],
        inference: InferenceEngine,
        *,
        batch_size: int = 10,
        interval_seconds: float = 1.0,
    ) -> None:
        if isinstance(records, (str, bytes)) or not isinstance(records, Sequence):
            raise TypeError("records must be a sequence of StreamSourceRecord values")
        if not records:
            raise ValueError("records must not be empty")
        if any(not isinstance(record, StreamSourceRecord) for record in records):
            raise TypeError("records must contain only StreamSourceRecord values")
        if (
            isinstance(batch_size, bool)
            or not isinstance(batch_size, int)
            or batch_size < 1
        ):
            raise ValueError("batch_size must be a positive integer")
        if isinstance(interval_seconds, bool) or not isinstance(
            interval_seconds, (int, float)
        ):
            raise TypeError("interval_seconds must be numeric")
        if not math.isfinite(interval_seconds) or interval_seconds < 0:
            raise ValueError("interval_seconds must be finite and non-negative")
        if not callable(getattr(inference, "predict_batch", None)):
            raise TypeError("inference must provide predict_batch")

        expected_features = set(inference.feature_names)
        for record in records:
            if set(record.sensor_readings) != expected_features:
                raise ValueError(
                    f"Sensor schema mismatch in {record.source_file} "
                    f"at data-row index {record.source_row_index}"
                )

        self._records = tuple(records)
        self._inference = inference
        self._batch_size = batch_size
        self._interval_seconds = float(interval_seconds)
        self._state = StreamState.READY
        self._state_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._error: Exception | None = None
        self._batches_emitted = 0

    @classmethod
    def from_wdseventdb(
        cls,
        inference: SPSInference | None = None,
        *,
        batch_size: int = 10,
        interval_seconds: float = 1.0,
    ) -> SimulatedSensorStream:
        engine = inference if inference is not None else SPSInference()
        records = load_wdseventdb_records(engine)
        return cls(
            records,
            engine,
            batch_size=batch_size,
            interval_seconds=interval_seconds,
        )

    @property
    def state(self) -> StreamState:
        with self._state_lock:
            return self._state

    @property
    def batches_emitted(self) -> int:
        with self._state_lock:
            return self._batches_emitted

    def start(self, on_batch: Callable[[StreamBatch], None]) -> None:
        if not callable(on_batch):
            raise TypeError("on_batch must be callable")
        with self._state_lock:
            if self._state is not StreamState.READY:
                raise RuntimeError(
                    f"Stream cannot start from state {self._state.value}; "
                    "create a new instance to replay"
                )
            self._state = StreamState.RUNNING
            self._thread = threading.Thread(
                target=self._run,
                args=(on_batch,),
                name="sps-simulated-sensor-stream",
                daemon=True,
            )
            try:
                self._thread.start()
            except Exception as error:
                self._error = error
                self._state = StreamState.FAILED
                raise

    def stop(self, timeout: float | None = None) -> None:
        with self._state_lock:
            if self._state is StreamState.READY:
                self._state = StreamState.STOPPED
            thread = self._thread
            if self._state is StreamState.RUNNING:
                self._stop_event.set()

        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout)
            if thread.is_alive():
                raise TimeoutError("Timed out while stopping the simulated stream")
        self._raise_worker_error()

    def wait(self, timeout: float | None = None) -> StreamState:
        with self._state_lock:
            thread = self._thread
            state = self._state
        if thread is None:
            if state is StreamState.STOPPED:
                return state
            raise RuntimeError("Stream has not been started")
        if thread is not threading.current_thread():
            thread.join(timeout)
            if thread.is_alive():
                raise TimeoutError("Timed out while waiting for the simulated stream")
        self._raise_worker_error()
        return self.state

    def _run(self, on_batch: Callable[[StreamBatch], None]) -> None:
        try:
            for start in range(0, len(self._records), self._batch_size):
                if self._stop_event.is_set():
                    self._set_state(StreamState.STOPPED)
                    return

                source_batch = self._records[start : start + self._batch_size]
                model_inputs = [record.sensor_readings for record in source_batch]
                predictions = self._inference.predict_batch(model_inputs)
                if len(predictions) != len(source_batch):
                    raise ValueError(
                        "Inference returned a different number of predictions "
                        "than source records"
                    )

                emitted_at = datetime.now(timezone.utc)
                readings = tuple(
                    StreamReading(
                        source_file=record.source_file,
                        source_row_index=record.source_row_index,
                        original_label=record.original_label,
                        source_timestamp=record.source_timestamp,
                        simulated_emission_time=emitted_at,
                        sensor_readings=record.sensor_readings,
                        prediction=prediction,
                    )
                    for record, prediction in zip(source_batch, predictions)
                )
                batch = StreamBatch(
                    batch_index=self.batches_emitted + 1,
                    readings=readings,
                )
                on_batch(batch)
                with self._state_lock:
                    self._batches_emitted += 1

                if self._stop_event.is_set():
                    self._set_state(StreamState.STOPPED)
                    return
                if start + self._batch_size >= len(self._records):
                    self._set_state(StreamState.COMPLETED)
                    return
                if self._stop_event.wait(self._interval_seconds):
                    self._set_state(StreamState.STOPPED)
                    return
            self._set_state(StreamState.COMPLETED)
        except Exception as error:
            with self._state_lock:
                self._error = error
                self._state = StreamState.FAILED

    def _set_state(self, state: StreamState) -> None:
        with self._state_lock:
            self._state = state

    def _raise_worker_error(self) -> None:
        with self._state_lock:
            error = self._error
        if error is not None:
            raise RuntimeError("Simulated stream worker failed") from error
