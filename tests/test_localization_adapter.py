from __future__ import annotations

import unittest
from dataclasses import replace
from unittest.mock import patch

from sps.localization_adapter import iter_localization_results, localize_frame
from sps.simulated_localization import (
    DEFAULT_NETWORK,
    LocalizationEstimate,
    SimulatedScenario,
    generate_scenario,
)


class LocalizationAdapterTests(unittest.TestCase):
    def test_result_exposes_readings_scores_evidence_and_separate_truth(self) -> None:
        frame = generate_scenario(
            SimulatedScenario("adapter-leak", "P-03", onset_step=1, seed=10),
            frame_count=4,
        )[-1]

        result = localize_frame(frame)

        self.assertEqual(result.step, frame.step)
        self.assertEqual(result.time_seconds, frame.time_seconds)
        self.assertEqual(result.scenario_id, "adapter-leak")
        self.assertEqual(result.sensor_ids, DEFAULT_NETWORK.sensor_ids)
        self.assertEqual(result.segment_ids, DEFAULT_NETWORK.segment_ids)
        self.assertEqual(result.localization_status, "localized")
        self.assertEqual(result.estimated_segment_id, "P-03")
        self.assertEqual(result.ground_truth.leak_segment_id, "P-03")
        self.assertEqual(result.ground_truth.onset_step, 1)
        self.assertTrue(result.evidence)
        self.assertIn("similarity_score", result.diagnostic_scores)
        self.assertIn("score_margin", result.diagnostic_scores)
        self.assertEqual(
            set(result.segment_diagnostic_scores),
            set(DEFAULT_NETWORK.segment_ids),
        )
        with self.assertRaises(TypeError):
            result.sensor_readings["FLOW_MAIN"] = 0.0

    def test_ground_truth_is_not_passed_into_estimator(self) -> None:
        frame = generate_scenario(
            SimulatedScenario("metadata-separation", "P-01", onset_step=0),
            frame_count=1,
        )[0]
        estimate = LocalizationEstimate(
            "ambiguous",
            None,
            0.8,
            0.01,
            (),
        )

        with patch(
            "sps.localization_adapter.estimate_affected_segment",
            return_value=estimate,
        ) as estimator:
            result = localize_frame(frame)

        self.assertEqual(estimator.call_args.args, (
            frame.sensor_readings,
            frame.normal_reference,
        ))
        self.assertEqual(set(estimator.call_args.kwargs), {"network"})
        self.assertEqual(result.localization_status, "ambiguous")
        self.assertEqual(result.ground_truth.leak_segment_id, "P-01")

    def test_statuses_propagate_without_forcing_a_segment(self) -> None:
        frame = generate_scenario(SimulatedScenario("normal"), frame_count=1)[0]
        statuses = (
            "localized",
            "ambiguous",
            "insufficient_evidence",
            "no_leak_pattern",
        )
        for status in statuses:
            with self.subTest(status=status):
                estimate = LocalizationEstimate(
                    status,
                    "P-02" if status == "localized" else None,
                    0.7 if status != "no_leak_pattern" else None,
                    0.1 if status != "no_leak_pattern" else None,
                    (),
                )
                with patch(
                    "sps.localization_adapter.estimate_affected_segment",
                    return_value=estimate,
                ):
                    result = localize_frame(frame)
                self.assertEqual(result.localization_status, status)
                self.assertEqual(
                    result.estimated_segment_id,
                    "P-02" if status == "localized" else None,
                )

    def test_missing_and_invalid_readings_are_rejected(self) -> None:
        frame = generate_scenario(SimulatedScenario("invalid-input"), frame_count=1)[0]
        missing_readings = dict(frame.sensor_readings)
        missing_readings.pop(DEFAULT_NETWORK.sensor_ids[0])
        missing_frame = replace(frame, sensor_readings=missing_readings)

        with self.assertRaisesRegex(ValueError, "missing sensors"):
            localize_frame(missing_frame)

        invalid_readings = dict(frame.sensor_readings)
        invalid_readings[DEFAULT_NETWORK.sensor_ids[0]] = float("nan")
        invalid_frame = replace(frame, sensor_readings=invalid_readings)

        with self.assertRaisesRegex(ValueError, "must be finite"):
            localize_frame(invalid_frame)

    def test_iterator_is_sequential_and_a_new_iteration_resets_scenario(self) -> None:
        scenario = SimulatedScenario("repeatable-adapter", "P-05", 2, seed=12)

        first = tuple(iter_localization_results(scenario, frame_count=5))
        second = tuple(iter_localization_results(scenario, frame_count=5))

        self.assertEqual([result.step for result in first], list(range(5)))
        self.assertEqual([result.step for result in second], list(range(5)))
        self.assertEqual(first, second)
        self.assertEqual(first[2].ground_truth.onset_step, 2)


if __name__ == "__main__":
    unittest.main()
