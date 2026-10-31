from __future__ import annotations

import unittest

from sps.simulated_localization import (
    DEFAULT_NETWORK,
    SimulatedScenario,
    estimate_affected_segment,
    generate_scenario,
    iter_scenario_frames,
)


class SimulatedLocalizationTests(unittest.TestCase):
    def test_normal_scenario_remains_unlocalized(self) -> None:
        frames = generate_scenario(
            SimulatedScenario("normal", seed=17),
            frame_count=12,
        )

        self.assertTrue(all(not frame.ground_truth.leak_active for frame in frames))
        for frame in frames:
            estimate = estimate_affected_segment(
                frame.sensor_readings,
                frame.normal_reference,
            )
            self.assertEqual(estimate.status, "no_leak_pattern")
            self.assertIsNone(estimate.estimated_segment_id)

    def test_leaks_start_at_configured_time_and_localize_each_pipe_segment(self) -> None:
        for segment_id in DEFAULT_NETWORK.segment_ids:
            with self.subTest(segment=segment_id):
                scenario = SimulatedScenario(
                    scenario_id=f"leak-{segment_id}",
                    leak_segment_id=segment_id,
                    onset_step=4,
                    seed=17,
                )
                frames = generate_scenario(scenario, frame_count=12)

                self.assertTrue(all(not frame.ground_truth.leak_active for frame in frames[:4]))
                self.assertTrue(all(frame.ground_truth.leak_active for frame in frames[4:]))
                self.assertEqual(frames[4].ground_truth.leak_segment_id, segment_id)
                self.assertEqual(frames[4].time_seconds, 4.0)
                estimate = estimate_affected_segment(
                    frames[-1].sensor_readings,
                    frames[-1].normal_reference,
                )

                self.assertEqual(estimate.status, "localized")
                self.assertEqual(estimate.estimated_segment_id, segment_id)
                self.assertIsNotNone(estimate.similarity_score)
                self.assertGreater(estimate.score_margin or 0.0, 0.04)
                self.assertTrue(estimate.evidence)

    def test_scenario_repeats_exactly_for_same_seed(self) -> None:
        scenario = SimulatedScenario(
            "repeatable",
            leak_segment_id="P-04",
            onset_step=3,
            seed=123,
        )

        first = generate_scenario(scenario, frame_count=10)
        second = generate_scenario(scenario, frame_count=10)

        self.assertEqual(first, second)
        self.assertEqual(dict(first[-1].sensor_readings), dict(second[-1].sensor_readings))

    def test_scenario_frames_can_be_consumed_as_a_time_evolving_stream(self) -> None:
        scenario = SimulatedScenario(
            "stream",
            leak_segment_id="P-03",
            onset_step=2,
            seed=2,
        )
        stream = iter_scenario_frames(
            scenario,
            frame_count=5,
            sampling_interval_seconds=0.25,
        )

        first = next(stream)
        second = next(stream)
        third = next(stream)

        self.assertEqual((first.step, second.step, third.step), (0, 1, 2))
        self.assertEqual((first.time_seconds, second.time_seconds, third.time_seconds), (0.0, 0.25, 0.5))
        self.assertFalse(first.ground_truth.leak_active)
        self.assertFalse(second.ground_truth.leak_active)
        self.assertTrue(third.ground_truth.leak_active)
        self.assertNotEqual(dict(second.sensor_readings), dict(third.sensor_readings))

    def test_ground_truth_is_separate_and_not_required_by_estimator(self) -> None:
        frame = generate_scenario(
            SimulatedScenario("known-truth", "P-02", 2, seed=9),
            frame_count=8,
        )[-1]

        self.assertEqual(
            set(frame.sensor_readings),
            set(DEFAULT_NETWORK.sensor_ids),
        )
        self.assertNotIn("ground_truth", frame.sensor_readings)
        estimate = estimate_affected_segment(
            frame.sensor_readings,
            frame.normal_reference,
        )
        self.assertEqual(estimate.estimated_segment_id, "P-02")

    def test_invalid_scenario_parameters_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "must define its onset"):
            SimulatedScenario("missing-onset", leak_segment_id="P-01")
        with self.assertRaisesRegex(ValueError, "Unknown pipe segment"):
            generate_scenario(
                SimulatedScenario("unknown-segment", "P-99", 0),
                frame_count=2,
            )
        with self.assertRaisesRegex(ValueError, "within the generated frame range"):
            generate_scenario(
                SimulatedScenario("late-onset", "P-01", 4),
                frame_count=4,
            )

    def test_estimator_rejects_missing_or_non_finite_sensor_values(self) -> None:
        frame = generate_scenario(SimulatedScenario("normal"), frame_count=1)[0]
        missing = dict(frame.sensor_readings)
        missing.pop(DEFAULT_NETWORK.sensor_ids[0])
        with self.assertRaisesRegex(ValueError, "missing sensors"):
            estimate_affected_segment(missing, frame.normal_reference)

        non_finite = dict(frame.sensor_readings)
        non_finite[DEFAULT_NETWORK.sensor_ids[0]] = float("nan")
        with self.assertRaisesRegex(ValueError, "must be finite"):
            estimate_affected_segment(non_finite, frame.normal_reference)

    def test_strong_nonmatching_pattern_is_reported_as_ambiguous(self) -> None:
        frame = generate_scenario(SimulatedScenario("normal"), frame_count=1)[0]
        incompatible = {
            sensor.sensor_id: frame.normal_reference[sensor.sensor_id]
            + sensor.noise_scale
            for sensor in DEFAULT_NETWORK.sensors
        }

        estimate = estimate_affected_segment(incompatible, frame.normal_reference)

        self.assertEqual(estimate.status, "ambiguous")
        self.assertIsNone(estimate.estimated_segment_id)
        self.assertIsNotNone(estimate.similarity_score)


if __name__ == "__main__":
    unittest.main()
