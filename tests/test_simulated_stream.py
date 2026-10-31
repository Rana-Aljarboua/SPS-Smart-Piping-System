from __future__ import annotations

import threading
import time
import unittest

from sps.inference import Prediction, SPSInference
from sps.simulated_stream import (
    SimulatedSensorStream,
    StreamSourceRecord,
    StreamState,
    load_wdseventdb_records,
)


class FakeInference:
    feature_names = ("sensor",)

    def __init__(self) -> None:
        self.received_samples: list[list[dict[str, object]]] = []

    def predict_batch(self, samples):
        self.received_samples.append([dict(sample) for sample in samples])
        return [
            Prediction(
                predicted_class=int(sample["sensor"] > 0),
                predicted_label=(
                    "anomalous event sample"
                    if sample["sensor"] > 0
                    else "normal sample"
                ),
                class_probabilities=None,
            )
            for sample in samples
        ]


def make_records(count: int) -> tuple[StreamSourceRecord, ...]:
    return tuple(
        StreamSourceRecord(
            source_file="Leakage/LeakEvent1.xlsx",
            source_row_index=index,
            sensor_readings={"sensor": float(index)},
            original_label=index % 2,
        )
        for index in range(count)
    )


class SimulatedStreamTests(unittest.TestCase):
    def test_emits_batches_progressively_with_predictions_and_metadata(self) -> None:
        inference = FakeInference()
        stream = SimulatedSensorStream(
            make_records(5),
            inference,
            batch_size=2,
            interval_seconds=0.1,
        )
        batches = []
        arrival_times = []
        first_batch_received = threading.Event()

        def collect(batch) -> None:
            batches.append(batch)
            arrival_times.append(time.monotonic())
            if len(batches) == 1:
                first_batch_received.set()

        stream.start(collect)
        self.assertTrue(first_batch_received.wait(timeout=2))
        self.assertEqual(stream.wait(timeout=3), StreamState.COMPLETED)
        self.assertEqual([len(batch.readings) for batch in batches], [2, 2, 1])
        self.assertEqual([batch.batch_index for batch in batches], [1, 2, 3])
        self.assertGreaterEqual(arrival_times[1] - arrival_times[0], 0.08)
        self.assertTrue(all(batch.data_origin == "simulated_dataset" for batch in batches))

        first = batches[0].readings[0]
        self.assertEqual(first.source_file, "Leakage/LeakEvent1.xlsx")
        self.assertEqual(first.source_row_index, 0)
        self.assertEqual(first.original_label, 0)
        self.assertIsNone(first.source_timestamp)
        self.assertEqual(first.data_origin, "simulated_dataset")
        self.assertEqual(first.prediction.predicted_class, 0)
        self.assertEqual(first.prediction_status, "normal sample prediction")
        anomalous = batches[0].readings[1]
        self.assertEqual(anomalous.prediction.predicted_class, 1)
        self.assertEqual(anomalous.prediction_status, "anomalous event prediction")
        self.assertEqual(len(inference.received_samples), 3)
        self.assertTrue(
            all("Labels" not in sample for batch in inference.received_samples for sample in batch)
        )

    def test_stop_interrupts_interval_and_prevents_later_batches(self) -> None:
        inference = FakeInference()
        stream = SimulatedSensorStream(
            make_records(5),
            inference,
            batch_size=1,
            interval_seconds=10,
        )
        first_batch_received = threading.Event()
        batches = []

        def collect(batch) -> None:
            batches.append(batch)
            first_batch_received.set()

        stream.start(collect)
        self.assertTrue(first_batch_received.wait(timeout=2))
        stream.stop(timeout=2)

        self.assertEqual(stream.state, StreamState.STOPPED)
        self.assertEqual(len(batches), 1)
        self.assertEqual(stream.batches_emitted, 1)

    def test_end_of_data_is_reported_and_stream_instance_is_single_use(self) -> None:
        stream = SimulatedSensorStream(
            make_records(3),
            FakeInference(),
            batch_size=2,
            interval_seconds=0,
        )
        batches = []

        stream.start(batches.append)
        self.assertEqual(stream.wait(timeout=2), StreamState.COMPLETED)
        self.assertEqual([len(batch.readings) for batch in batches], [2, 1])
        with self.assertRaisesRegex(RuntimeError, "create a new instance to replay"):
            stream.start(batches.append)

    def test_worker_errors_are_raised_from_wait(self) -> None:
        stream = SimulatedSensorStream(
            make_records(1),
            FakeInference(),
            interval_seconds=0,
        )

        def fail_callback(_batch) -> None:
            raise ValueError("callback failed")

        stream.start(fail_callback)
        with self.assertRaisesRegex(RuntimeError, "worker failed") as raised:
            stream.wait(timeout=2)
        self.assertIsInstance(raised.exception.__cause__, ValueError)
        self.assertEqual(stream.state, StreamState.FAILED)

    def test_rejects_empty_or_invalid_dataset_records(self) -> None:
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            SimulatedSensorStream((), FakeInference())
        with self.assertRaisesRegex(TypeError, "only StreamSourceRecord"):
            SimulatedSensorStream(({"sensor": 1.0},), FakeInference())
        with self.assertRaisesRegex(ValueError, "must be metadata"):
            StreamSourceRecord(
                source_file="Clean/CleanData.xlsx",
                source_row_index=0,
                sensor_readings={"sensor": 1.0, "Labels": 0},
            )

    def test_wdseventdb_records_run_through_saved_model(self) -> None:
        inference = SPSInference()
        records = load_wdseventdb_records(inference)

        self.assertEqual(len(records), 13851)
        self.assertEqual(
            {record.source_file for record in records},
            {
                "Clean/CleanData.xlsx",
                "Leakage/LeakEvent1.xlsx",
                "Leakage/LeakEvent2.xlsx",
                "Leakage/LeakEvent3.xlsx",
            },
        )
        first_record = records[0]
        self.assertEqual(first_record.source_file, "Clean/CleanData.xlsx")
        self.assertEqual(first_record.source_row_index, 0)
        self.assertEqual(first_record.original_label, 0)
        self.assertNotIn("Labels", first_record.sensor_readings)
        self.assertIsNone(first_record.source_timestamp)
        stream = SimulatedSensorStream(
            records[:1],
            inference,
            batch_size=1,
            interval_seconds=0,
        )
        batches = []

        stream.start(batches.append)
        self.assertEqual(stream.wait(timeout=5), StreamState.COMPLETED)
        self.assertEqual(len(batches), 1)
        reading = batches[0].readings[0]
        self.assertEqual(reading.source_file, first_record.source_file)
        self.assertEqual(reading.source_row_index, first_record.source_row_index)
        self.assertEqual(reading.original_label, first_record.original_label)
        prediction = batches[0].readings[0].prediction
        self.assertIn(prediction.predicted_class, inference.classes)
        self.assertEqual(
            prediction.predicted_label,
            inference.target_mapping[prediction.predicted_class],
        )


if __name__ == "__main__":
    unittest.main()
