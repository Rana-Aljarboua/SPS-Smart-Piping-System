from __future__ import annotations

import math
import unittest

import numpy as np

from sps import simulated_localization
from sps.localization_adapter import iter_localization_results, localize_frame
from sps.simulated_localization import (
    DEFAULT_NETWORK,
    SimulatedScenario,
    estimate_affected_segment,
    generate_scenario,
)
from sps.water_level import (
    DEFAULT_WATER_LEVEL_CONFIG,
    WATER_LEVEL_SENSOR_ID,
    WATER_LEVEL_STATUS_TEXT,
    WaterLevelAssessment,
    WaterLevelCondition,
    WaterLevelConfig,
    WaterLevelMonitor,
    simulate_water_level_reading,
    summarize_water_level_events,
    water_level_rng,
)


def _run(monitor: WaterLevelMonitor, readings) -> list[WaterLevelAssessment]:
    return [monitor.assess(value) for value in readings]


class ThresholdRuleTests(unittest.TestCase):
    def test_above_threshold_is_normal(self) -> None:
        result = WaterLevelMonitor().assess(200.01)
        self.assertIs(result.condition, WaterLevelCondition.NORMAL)
        self.assertFalse(result.alert_triggered)
        self.assertFalse(result.event_active)

    def test_below_threshold_is_potential_leak_indication(self) -> None:
        result = WaterLevelMonitor().assess(199.99)
        self.assertIs(result.condition, WaterLevelCondition.POTENTIAL_LEAK)
        self.assertTrue(result.alert_triggered)
        self.assertTrue(result.event_active)
        self.assertEqual(result.event_number, 1)

    def test_exactly_threshold_is_explicit_boundary(self) -> None:
        for value in (200, 200.0):
            result = WaterLevelMonitor().assess(value)
            self.assertIs(result.condition, WaterLevelCondition.BOUNDARY)
            self.assertFalse(result.alert_triggered)
            self.assertFalse(result.event_active)
            self.assertEqual(result.reading, 200.0)

    def test_threshold_comes_from_configuration(self) -> None:
        self.assertEqual(DEFAULT_WATER_LEVEL_CONFIG.threshold, 200.0)
        monitor = WaterLevelMonitor(WaterLevelConfig(threshold=50.0, baseline=80.0))
        self.assertIs(monitor.assess(60).condition, WaterLevelCondition.NORMAL)
        self.assertIs(monitor.assess(50).condition, WaterLevelCondition.BOUNDARY)
        self.assertIs(monitor.assess(49).condition, WaterLevelCondition.POTENTIAL_LEAK)

    def test_unusable_readings_are_classified_without_alerts(self) -> None:
        monitor = WaterLevelMonitor(WaterLevelConfig(maximum=1000.0))
        cases = {
            None: WaterLevelCondition.UNAVAILABLE,
            True: WaterLevelCondition.INVALID,
            "180": WaterLevelCondition.INVALID,
            math.nan: WaterLevelCondition.INVALID,
            math.inf: WaterLevelCondition.INVALID,
            -5.0: WaterLevelCondition.OUT_OF_RANGE,
            1500.0: WaterLevelCondition.OUT_OF_RANGE,
        }
        for value, expected in cases.items():
            with self.subTest(value=value):
                result = monitor.assess(value)
                self.assertIs(result.condition, expected)
                self.assertFalse(result.alert_triggered)
        self.assertEqual(monitor.event_count, 0)

    def test_every_condition_has_operator_text(self) -> None:
        self.assertEqual(set(WATER_LEVEL_STATUS_TEXT), set(WaterLevelCondition))


class DuplicateAlertTests(unittest.TestCase):
    def test_one_alert_per_continuous_event(self) -> None:
        results = _run(WaterLevelMonitor(), [240, 230, 195, 190, 185, 180, 182])
        self.assertEqual(sum(item.alert_triggered for item in results), 1)
        self.assertTrue(results[2].alert_triggered)
        self.assertTrue(all(item.event_number == 1 for item in results[2:]))

    def test_boundary_reading_does_not_split_an_event(self) -> None:
        results = _run(WaterLevelMonitor(), [199, 200, 199, 200, 198])
        self.assertEqual(sum(item.alert_triggered for item in results), 1)
        self.assertTrue(all(item.event_active for item in results))

    def test_boundary_does_not_open_an_event(self) -> None:
        results = _run(WaterLevelMonitor(), [210, 200, 200, 205])
        self.assertFalse(any(item.alert_triggered for item in results))
        self.assertFalse(any(item.event_active for item in results))

    def test_noise_around_threshold_does_not_repeat_alerts(self) -> None:
        results = _run(WaterLevelMonitor(), [199, 201, 199, 201, 199.5, 200.5, 198])
        self.assertEqual(sum(item.alert_triggered for item in results), 1)

    def test_event_clears_after_consecutive_normal_readings_then_realerts(self) -> None:
        monitor = WaterLevelMonitor(WaterLevelConfig(clear_after_normal_readings=3))
        results = _run(monitor, [190, 210, 211, 212, 213, 190])
        self.assertEqual([item.event_active for item in results], [True, True, True, False, False, True])
        self.assertEqual([item.alert_triggered for item in results], [True, False, False, False, False, True])
        self.assertEqual(results[-1].event_number, 2)
        self.assertEqual(monitor.event_count, 2)

    def test_boundary_resets_the_clearing_streak(self) -> None:
        monitor = WaterLevelMonitor(WaterLevelConfig(clear_after_normal_readings=2))
        results = _run(monitor, [190, 210, 200, 210, 210])
        self.assertEqual([item.event_active for item in results], [True, True, True, True, False])

    def test_missing_readings_do_not_close_or_reopen_an_event(self) -> None:
        results = _run(WaterLevelMonitor(), [190, None, math.nan, 190])
        self.assertEqual(sum(item.alert_triggered for item in results), 1)
        self.assertTrue(all(item.event_active for item in results))

    def test_reset_forgets_open_event(self) -> None:
        monitor = WaterLevelMonitor()
        monitor.assess(150)
        monitor.reset()
        self.assertFalse(monitor.event_active)
        self.assertTrue(monitor.assess(150).alert_triggered)

    def test_configuration_validation(self) -> None:
        with self.assertRaises(ValueError):
            WaterLevelConfig(clear_after_normal_readings=0)
        with self.assertRaises(ValueError):
            WaterLevelConfig(clear_after_normal_readings=True)
        with self.assertRaises(ValueError):
            WaterLevelConfig(baseline=200.0)
        with self.assertRaises(TypeError):
            WaterLevelMonitor(config="200")


class EventSummaryTests(unittest.TestCase):
    def test_summary_has_one_record_per_event(self) -> None:
        monitor = WaterLevelMonitor(WaterLevelConfig(clear_after_normal_readings=2))
        readings = [240, 199, 200, 185, 201, 202, 230, 180, 179]
        timeline = [(step, monitor.assess(value)) for step, value in enumerate(readings)]
        events = summarize_water_level_events(timeline)

        self.assertEqual(len(events), 2)
        first, second = events
        self.assertEqual((first.event_number, first.start_step, first.end_step), (1, 1, 5))
        self.assertEqual(first.lowest_reading, 185.0)
        self.assertEqual(first.readings_below_threshold, 2)
        self.assertFalse(first.is_active)
        self.assertEqual((second.event_number, second.start_step, second.end_step), (2, 7, None))
        self.assertTrue(second.is_active)
        self.assertEqual(second.lowest_reading, 179.0)

    def test_summary_is_empty_without_events(self) -> None:
        monitor = WaterLevelMonitor()
        timeline = [(step, monitor.assess(value)) for step, value in enumerate([240, 200, 230])]
        self.assertEqual(summarize_water_level_events(timeline), ())

    def test_summary_rejects_wrong_items(self) -> None:
        with self.assertRaises(TypeError):
            summarize_water_level_events([(0, "normal")])


class SimulationTests(unittest.TestCase):
    def test_simulated_reading_is_deterministic_for_a_seed(self) -> None:
        first = simulate_water_level_reading(step=3, event_scale=0.0, rng=water_level_rng(5))
        second = simulate_water_level_reading(step=3, event_scale=0.0, rng=water_level_rng(5))
        self.assertEqual(first, second)

    def test_leak_strength_lowers_simulated_level(self) -> None:
        normal = simulate_water_level_reading(step=0, event_scale=0.0, rng=np.random.default_rng(1))
        leaking = simulate_water_level_reading(step=0, event_scale=6.0, rng=np.random.default_rng(1))
        self.assertGreater(normal, DEFAULT_WATER_LEVEL_CONFIG.threshold)
        self.assertLess(leaking, DEFAULT_WATER_LEVEL_CONFIG.threshold)

    def test_simulation_input_validation(self) -> None:
        with self.assertRaises(ValueError):
            simulate_water_level_reading(step=-1, event_scale=0.0, rng=water_level_rng(1))
        with self.assertRaises(ValueError):
            simulate_water_level_reading(step=0, event_scale=-1.0, rng=water_level_rng(1))
        with self.assertRaises(TypeError):
            simulate_water_level_reading(step=0, event_scale=0.0, rng=None)
        with self.assertRaises(TypeError):
            water_level_rng(1.5)

    def test_every_frame_carries_a_water_level_reading(self) -> None:
        frames = generate_scenario(SimulatedScenario("wl-normal", seed=3), frame_count=20)
        self.assertTrue(all(frame.water_level_reading is not None for frame in frames))
        self.assertTrue(
            all(frame.water_level_reading > DEFAULT_WATER_LEVEL_CONFIG.threshold for frame in frames)
        )


class SeparationTests(unittest.TestCase):
    """Water level must stay out of model inputs and localization inputs."""

    def test_water_level_is_not_a_localization_sensor(self) -> None:
        self.assertNotIn(WATER_LEVEL_SENSOR_ID, DEFAULT_NETWORK.sensor_ids)
        frame = generate_scenario(SimulatedScenario("wl-sep", "P-02", onset_step=2), frame_count=6)[-1]
        self.assertNotIn(WATER_LEVEL_SENSOR_ID, frame.sensor_readings)
        self.assertNotIn(WATER_LEVEL_SENSOR_ID, frame.normal_reference)

    def test_estimator_rejects_water_level_as_an_extra_sensor(self) -> None:
        frame = generate_scenario(SimulatedScenario("wl-extra", seed=4), frame_count=1)[0]
        readings = dict(frame.sensor_readings)
        readings[WATER_LEVEL_SENSOR_ID] = 180.0
        with self.assertRaises(ValueError):
            estimate_affected_segment(readings, frame.normal_reference)

    def test_pipe_sensor_stream_is_unchanged_by_water_level_simulation(self) -> None:
        scenario = SimulatedScenario("wl-stream", "P-03", onset_step=8, seed=24)
        with_level = generate_scenario(scenario, frame_count=40)
        original = simulated_localization.simulate_water_level_reading
        try:
            simulated_localization.simulate_water_level_reading = lambda **_: 0.0
            without_level = generate_scenario(scenario, frame_count=40)
        finally:
            simulated_localization.simulate_water_level_reading = original
        for first, second in zip(with_level, without_level):
            self.assertEqual(dict(first.sensor_readings), dict(second.sensor_readings))
            self.assertEqual(dict(first.normal_reference), dict(second.normal_reference))

    def test_water_level_configuration_does_not_change_localization(self) -> None:
        scenario = SimulatedScenario("wl-config", "P-05", onset_step=5, seed=9)
        default = list(iter_localization_results(scenario, frame_count=25))
        custom = list(
            iter_localization_results(
                scenario,
                frame_count=25,
                water_level_config=WaterLevelConfig(baseline=500.0, leak_response_per_strength=1.0),
            )
        )
        for first, second in zip(default, custom):
            self.assertEqual(first.localization_status, second.localization_status)
            self.assertEqual(first.estimated_segment_id, second.estimated_segment_id)
            self.assertEqual(dict(first.sensor_readings), dict(second.sensor_readings))

    def test_saved_model_feature_schema_excludes_water_level(self) -> None:
        try:
            from sps.inference import SPSInference

            engine = SPSInference()
        except FileNotFoundError:
            self.skipTest("Saved model artifact is not present")
        names = tuple(engine.feature_names)
        self.assertEqual(len(names), 15)
        self.assertNotIn(WATER_LEVEL_SENSOR_ID, names)
        self.assertFalse(any("level" in name.lower() for name in names))


class AdapterIntegrationTests(unittest.TestCase):
    def test_leak_scenario_raises_exactly_one_water_level_alert(self) -> None:
        scenario = SimulatedScenario("wl-leak", "P-03", onset_step=8, leak_strength=6.0, seed=24)
        results = list(iter_localization_results(scenario, frame_count=60))
        triggered = [item.step for item in results if item.water_level_assessment.alert_triggered]
        self.assertEqual(len(triggered), 1)
        self.assertGreaterEqual(triggered[0], scenario.onset_step)
        events = summarize_water_level_events((item.step, item.water_level_assessment) for item in results)
        self.assertEqual(len(events), 1)
        self.assertTrue(events[0].is_active)

    def test_normal_scenario_raises_no_water_level_alert(self) -> None:
        results = list(iter_localization_results(SimulatedScenario("wl-ok", seed=24), frame_count=60))
        self.assertFalse(any(item.water_level_assessment.alert_triggered for item in results))
        self.assertTrue(
            all(item.water_level_assessment.condition is WaterLevelCondition.NORMAL for item in results)
        )

    def test_results_expose_reading_and_assessment(self) -> None:
        frame = generate_scenario(SimulatedScenario("wl-frame", seed=2), frame_count=1)[0]
        result = localize_frame(frame)
        self.assertEqual(result.water_level_reading, frame.water_level_reading)
        self.assertEqual(result.water_level_assessment.sensor_id, WATER_LEVEL_SENSOR_ID)

    def test_localize_frame_rejects_wrong_monitor_type(self) -> None:
        frame = generate_scenario(SimulatedScenario("wl-type", seed=2), frame_count=1)[0]
        with self.assertRaises(TypeError):
            localize_frame(frame, water_level_monitor=object())


if __name__ == "__main__":
    unittest.main()
