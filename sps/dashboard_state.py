from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from queue import Empty, Queue

from sps.simulated_stream import StreamBatch, StreamReading


class DashboardHistory:
    """Keep bounded display history and exact counters for readings received."""

    def __init__(self, max_history: int = 500) -> None:
        if isinstance(max_history, bool) or not isinstance(max_history, int):
            raise TypeError("max_history must be an integer")
        if max_history < 1:
            raise ValueError("max_history must be positive")
        self.max_history = max_history
        self.readings: deque[StreamReading] = deque(maxlen=max_history)
        self.alerts: deque[StreamReading] = deque(maxlen=max_history)
        self.total_readings = 0
        self.normal_predictions = 0
        self.anomalous_predictions = 0
        self.processed_batches = 0

    def add_batch(self, batch: StreamBatch) -> None:
        if not isinstance(batch, StreamBatch):
            raise TypeError("batch must be a StreamBatch")
        if batch.data_origin != "simulated_dataset":
            raise ValueError("Dashboard accepts only explicitly simulated dataset batches")

        self.processed_batches += 1
        for reading in batch.readings:
            if reading.data_origin != "simulated_dataset":
                raise ValueError(
                    "Dashboard accepts only explicitly simulated dataset readings"
                )
            self.total_readings += 1
            if reading.prediction.predicted_class == 0:
                self.normal_predictions += 1
            elif reading.prediction.predicted_class == 1:
                self.anomalous_predictions += 1
                self.alerts.appendleft(reading)
            self.readings.append(reading)

    def drain(self, batches: Queue[StreamBatch]) -> int:
        """Ingest all batches currently queued by the worker without blocking."""
        drained = 0
        while True:
            try:
                batch = batches.get_nowait()
            except Empty:
                return drained
            self.add_batch(batch)
            drained += 1


def ingest_batches(
    history: DashboardHistory, batches: Iterable[StreamBatch]
) -> int:
    """Apply a batch sequence to dashboard state and return its batch count."""
    count = 0
    for batch in batches:
        history.add_batch(batch)
        count += 1
    return count
