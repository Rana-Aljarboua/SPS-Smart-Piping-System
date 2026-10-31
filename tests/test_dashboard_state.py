from __future__ import annotations

import unittest
from datetime import datetime, timezone
from queue import Queue

from sps.dashboard_state import DashboardHistory, ingest_batches
from sps.inference import Prediction
from sps.simulated_stream import StreamBatch, StreamReading


def reading(index: int, predicted_class: int) -> StreamReading:
    label = (
        "anomalous event sample"
        if predicted_class == 1
        else "normal sample"
    )
    return StreamReading(
        source_file="Leakage/LeakEvent1.xlsx",
        source_row_index=index,
        original_label=predicted_class,
        source_timestamp=None,
        simulated_emission_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
        sensor_readings={"Water Flow 1": float(index)},
        prediction=Prediction(predicted_class, label, None),
    )


def batch(index: int, *readings: StreamReading) -> StreamBatch:
    return StreamBatch(batch_index=index, readings=tuple(readings))


class DashboardHistoryTests(unittest.TestCase):
    def test_kpis_count_actual_predictions_and_batches(self) -> None:
        history = DashboardHistory()

        count = ingest_batches(
            history,
            [
                batch(1, reading(0, 0), reading(1, 1)),
                batch(2, reading(2, 0)),
            ],
        )

        self.assertEqual(count, 2)
        self.assertEqual(history.total_readings, 3)
        self.assertEqual(history.normal_predictions, 2)
        self.assertEqual(history.anomalous_predictions, 1)
        self.assertEqual(history.processed_batches, 2)

    def test_history_is_bounded_but_cumulative_kpis_are_not(self) -> None:
        history = DashboardHistory(max_history=2)
        history.add_batch(
            batch(1, reading(0, 0), reading(1, 1), reading(2, 0), reading(3, 1))
        )

        self.assertEqual(len(history.readings), 2)
        self.assertEqual([item.source_row_index for item in history.readings], [2, 3])
        self.assertEqual(history.total_readings, 4)
        self.assertEqual(history.normal_predictions, 2)
        self.assertEqual(history.anomalous_predictions, 2)
        self.assertEqual(len(history.alerts), 2)
        self.assertEqual([item.source_row_index for item in history.alerts], [3, 1])

    def test_alerts_are_created_only_from_anomalous_predictions(self) -> None:
        history = DashboardHistory()
        history.add_batch(batch(1, reading(0, 0), reading(1, 1)))

        self.assertEqual(
            [alert.prediction.predicted_class for alert in history.alerts],
            [1],
        )

    def test_queue_drain_ingests_batches_without_worker_ui_calls(self) -> None:
        history = DashboardHistory()
        batches = Queue()
        batches.put(batch(1, reading(0, 1)))
        batches.put(batch(2, reading(1, 0)))

        self.assertEqual(history.drain(batches), 2)
        self.assertEqual(history.total_readings, 2)
        self.assertTrue(batches.empty())

    def test_non_simulated_data_is_rejected(self) -> None:
        history = DashboardHistory()
        invalid = StreamBatch(batch_index=1, readings=(), data_origin="hardware")

        with self.assertRaisesRegex(ValueError, "only explicitly simulated"):
            history.add_batch(invalid)

    def test_rejects_invalid_history_size(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be positive"):
            DashboardHistory(max_history=0)


if __name__ == "__main__":
    unittest.main()
