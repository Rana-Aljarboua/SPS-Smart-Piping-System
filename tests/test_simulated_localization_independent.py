from __future__ import annotations

import unittest
from collections import deque

import numpy as np

from sps.simulated_localization import DEFAULT_NETWORK, estimate_affected_segment


def _shortest_node_distances(start: str) -> dict[str, int]:
    adjacency = {node: set() for node in DEFAULT_NETWORK.nodes}
    for segment in DEFAULT_NETWORK.segments:
        adjacency[segment.start_node].add(segment.end_node)
        adjacency[segment.end_node].add(segment.start_node)

    distances = {start: 0}
    pending = deque([start])
    while pending:
        node = pending.popleft()
        for neighbor in adjacency[node]:
            if neighbor not in distances:
                distances[neighbor] = distances[node] + 1
                pending.append(neighbor)
    return distances


def _independent_topology_observation(
    leak_segment_id: str,
    *,
    strength: float = 1.0,
    seed: int | None = None,
) -> tuple[dict[str, float], dict[str, float]]:
    """Build held-out measurements from graph distance and sensor placement."""
    segment_by_id = {
        segment.segment_id: segment for segment in DEFAULT_NETWORK.segments
    }
    node_distances = {
        node: _shortest_node_distances(node) for node in DEFAULT_NETWORK.nodes
    }
    segment_adjacency = {
        segment_id: set() for segment_id in DEFAULT_NETWORK.segment_ids
    }
    for first in DEFAULT_NETWORK.segments:
        for second in DEFAULT_NETWORK.segments:
            if first.segment_id != second.segment_id and {
                first.start_node,
                first.end_node,
            } & {second.start_node, second.end_node}:
                segment_adjacency[first.segment_id].add(second.segment_id)

    segment_distances: dict[str, dict[str, int]] = {}
    for start in DEFAULT_NETWORK.segment_ids:
        distances = {start: 0}
        pending = deque([start])
        while pending:
            segment_id = pending.popleft()
            for neighbor in segment_adjacency[segment_id]:
                if neighbor not in distances:
                    distances[neighbor] = distances[segment_id] + 1
                    pending.append(neighbor)
        segment_distances[start] = distances

    affected_segment = segment_by_id[leak_segment_id]
    reference: dict[str, float] = {}
    readings: dict[str, float] = {}
    rng = np.random.default_rng(seed)
    for sensor in DEFAULT_NETWORK.sensors:
        reference[sensor.sensor_id] = sensor.baseline
        if sensor.kind == "pressure":
            assert sensor.node_id is not None
            distance = min(
                node_distances[sensor.node_id][affected_segment.start_node],
                node_distances[sensor.node_id][affected_segment.end_node],
            )
            response = -(3.5 - 0.8 * distance)
        else:
            assert sensor.segment_id is not None
            distance = segment_distances[sensor.segment_id][leak_segment_id]
            response = max(0.0, 3.8 - 1.5 * distance)

        gain = float(rng.uniform(0.8, 1.2)) if seed is not None else 1.0
        noise = float(rng.normal(0.0, 0.15)) if seed is not None else 0.0
        readings[sensor.sensor_id] = sensor.baseline + sensor.noise_scale * (
            strength * gain * response + noise
        )
    return readings, reference


class IndependentLocalizationEvaluationTests(unittest.TestCase):
    def test_held_out_strengths_and_measurement_variations_are_accounted_per_segment(self) -> None:
        per_segment = {
            segment_id: {
                "correct": 0,
                "incorrect": 0,
                "ambiguous": 0,
                "insufficient": 0,
            }
            for segment_id in DEFAULT_NETWORK.segment_ids
        }
        strengths = (0.5, 1.0, 2.0, 4.0)
        seeds = range(10)

        for segment_id in DEFAULT_NETWORK.segment_ids:
            for strength in strengths:
                for seed in seeds:
                    with self.subTest(
                        segment=segment_id,
                        strength=strength,
                        seed=seed,
                    ):
                        readings, reference = _independent_topology_observation(
                            segment_id,
                            strength=strength,
                            seed=seed,
                        )
                        estimate = estimate_affected_segment(readings, reference)
                        if estimate.status == "localized":
                            result = (
                                "correct"
                                if estimate.estimated_segment_id == segment_id
                                else "incorrect"
                            )
                        elif estimate.status == "ambiguous":
                            result = "ambiguous"
                        else:
                            result = "insufficient"
                        per_segment[segment_id][result] += 1

        self.assertEqual(
            per_segment,
            {
                "P-01": {
                    "correct": 40,
                    "incorrect": 0,
                    "ambiguous": 0,
                    "insufficient": 0,
                },
                "P-02": {
                    "correct": 40,
                    "incorrect": 0,
                    "ambiguous": 0,
                    "insufficient": 0,
                },
                "P-03": {
                    "correct": 40,
                    "incorrect": 0,
                    "ambiguous": 0,
                    "insufficient": 0,
                },
                "P-04": {
                    "correct": 36,
                    "incorrect": 0,
                    "ambiguous": 4,
                    "insufficient": 0,
                },
                "P-05": {
                    "correct": 40,
                    "incorrect": 0,
                    "ambiguous": 0,
                    "insufficient": 0,
                },
            },
        )

    def test_no_event_has_no_leak_pattern(self) -> None:
        reference = {
            sensor.sensor_id: sensor.baseline
            for sensor in DEFAULT_NETWORK.sensors
        }

        estimate = estimate_affected_segment(reference, reference)

        self.assertEqual(estimate.status, "no_leak_pattern")
        self.assertIsNone(estimate.estimated_segment_id)

    def test_weak_event_does_not_produce_a_localization(self) -> None:
        readings, reference = _independent_topology_observation(
            "P-01",
            strength=0.1,
        )

        estimate = estimate_affected_segment(readings, reference)

        self.assertEqual(estimate.status, "no_leak_pattern")
        self.assertIsNone(estimate.estimated_segment_id)

    def test_pressure_only_event_has_insufficient_localization_evidence(self) -> None:
        reference = {
            sensor.sensor_id: sensor.baseline
            for sensor in DEFAULT_NETWORK.sensors
        }
        readings = dict(reference)
        for sensor in DEFAULT_NETWORK.sensors:
            if sensor.kind == "pressure":
                readings[sensor.sensor_id] -= 8 * sensor.noise_scale

        estimate = estimate_affected_segment(readings, reference)

        self.assertEqual(estimate.status, "insufficient_evidence")
        self.assertIsNone(estimate.estimated_segment_id)

    def test_common_mode_flow_event_is_ambiguous(self) -> None:
        reference = {
            sensor.sensor_id: sensor.baseline
            for sensor in DEFAULT_NETWORK.sensors
        }
        readings = {
            sensor.sensor_id: reference[sensor.sensor_id]
            + (2 * sensor.noise_scale if sensor.kind == "flow" else 0.0)
            for sensor in DEFAULT_NETWORK.sensors
        }

        estimate = estimate_affected_segment(readings, reference)

        self.assertEqual(estimate.status, "ambiguous")
        self.assertIsNone(estimate.estimated_segment_id)


if __name__ == "__main__":
    unittest.main()
