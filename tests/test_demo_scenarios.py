from __future__ import annotations

import unittest

from sps.demo_scenarios import (
    AlertStatus,
    SCENARIO_CHOICES,
    build_snapshot,
    section_status,
)
from sps.demo_visuals import make_network_figure, make_section_trend_figure


class DemoScenarioTests(unittest.TestCase):
    def test_scenario_list_is_complete_and_snapshots_are_consistent(self) -> None:
        expected = {
            "normal",
            "pressure",
            "flow",
            "leak",
            "acknowledged",
            "resolved",
        }
        self.assertEqual({scenario_id for scenario_id, _ in SCENARIO_CHOICES}, expected)

        for scenario_id, _ in SCENARIO_CHOICES:
            with self.subTest(scenario=scenario_id):
                snapshot = build_snapshot(scenario_id)
                self.assertEqual(len(snapshot.scenario.sections), 6)
                self.assertEqual(len(snapshot.telemetry), 6)
                self.assertEqual(
                    {item.section_id for item in snapshot.telemetry},
                    {item.section_id for item in snapshot.scenario.sections},
                )

    def test_default_leak_scenario_links_alert_section_and_sensor_series(self) -> None:
        snapshot = build_snapshot("leak")
        alert = snapshot.scenario.alert
        self.assertIsNotNone(alert)
        assert alert is not None
        affected = snapshot.telemetry_for(alert.section_id)

        self.assertEqual(alert.section_id, "S-04")
        self.assertEqual(affected.latest_pressure_kpa, alert.observed_pressure_kpa)
        self.assertEqual(affected.latest_flow_lpm, alert.observed_flow_lpm)
        self.assertEqual(affected.expected_pressure_kpa, alert.expected_pressure_kpa)
        self.assertEqual(affected.expected_flow_lpm, alert.expected_flow_lpm)
        self.assertEqual(section_status(snapshot, alert.section_id), "critical")
        self.assertEqual(snapshot.normal_sections, 5)
        self.assertEqual(len(snapshot.active_alerts), 1)
        self.assertEqual(len(snapshot.critical_alerts), 1)

    def test_specialized_scenarios_change_only_the_configured_signal_pattern(self) -> None:
        pressure = build_snapshot("pressure")
        pressure_alert = pressure.scenario.alert
        assert pressure_alert is not None
        pressure_data = pressure.telemetry_for(pressure_alert.section_id)
        self.assertEqual(pressure_data.latest_pressure_kpa, 374.0)
        self.assertAlmostEqual(
            pressure_data.latest_flow_lpm, pressure_data.expected_flow_lpm
        )

        flow = build_snapshot("flow")
        flow_alert = flow.scenario.alert
        assert flow_alert is not None
        flow_data = flow.telemetry_for(flow_alert.section_id)
        self.assertEqual(flow_data.latest_flow_lpm, 31.7)
        self.assertAlmostEqual(
            flow_data.latest_pressure_kpa, flow_data.expected_pressure_kpa
        )

    def test_acknowledged_and_resolved_statuses_have_coherent_section_state(self) -> None:
        acknowledged = build_snapshot("acknowledged")
        self.assertEqual(acknowledged.alert_status, AlertStatus.ACKNOWLEDGED)
        self.assertEqual(len(acknowledged.active_alerts), 1)
        self.assertEqual(len(acknowledged.critical_alerts), 0)
        self.assertEqual(section_status(acknowledged, "S-04"), "attention")

        resolved = build_snapshot("leak", AlertStatus.RESOLVED)
        self.assertEqual(len(resolved.active_alerts), 0)
        self.assertEqual(len(resolved.critical_alerts), 0)
        self.assertEqual(resolved.normal_sections, 6)
        self.assertEqual(section_status(resolved, "S-04"), "normal")
        self.assertEqual(
            resolved.telemetry_for("S-04").latest_pressure_kpa,
            resolved.telemetry_for("S-04").expected_pressure_kpa,
        )

    def test_figures_use_the_selected_scenario_series_and_section_state(self) -> None:
        snapshot = build_snapshot("flow")
        section = snapshot.scenario.alert.section_id
        trend = make_section_trend_figure(snapshot, section)
        network = make_network_figure(snapshot, section)

        self.assertEqual(len(trend.data), 4)
        self.assertEqual(trend.data[0].y[-1], snapshot.telemetry_for(section).latest_pressure_kpa)
        self.assertEqual(trend.data[2].y[-1], snapshot.telemetry_for(section).latest_flow_lpm)
        selected_section_markers = [
            trace
            for trace in network.data
            if trace.type == "scatter" and trace.mode == "markers"
        ]
        self.assertEqual(len(selected_section_markers), len(snapshot.scenario.sections))
        self.assertGreater(len(network.layout.annotations), 0)

    def test_unknown_scenario_or_section_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown demonstration scenario"):
            build_snapshot("surprise")
        with self.assertRaisesRegex(ValueError, "Unknown illustrative section"):
            build_snapshot("normal").telemetry_for("S-99")


if __name__ == "__main__":
    unittest.main()
