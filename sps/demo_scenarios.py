from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class AlertStatus(str, Enum):
    NEW = "New"
    INVESTIGATING = "Investigating"
    ACKNOWLEDGED = "Acknowledged"
    RESOLVED = "Resolved"


class ScenarioKind(str, Enum):
    NORMAL = "normal"
    PRESSURE = "pressure"
    FLOW = "flow"
    LEAK = "leak"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"


@dataclass(frozen=True)
class DemoNode:
    node_id: str
    name: str
    x: float
    y: float


@dataclass(frozen=True)
class DemoSection:
    section_id: str
    name: str
    start_node: str
    end_node: str
    route: tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class DemoAlert:
    alert_id: str
    title: str
    severity: str
    initial_status: AlertStatus
    section_id: str
    event_step: str
    observed_pressure_kpa: float
    expected_pressure_kpa: float
    observed_flow_lpm: float
    expected_flow_lpm: float
    explanation: str
    recommended_action: str
    resolution_note: str | None = None


@dataclass(frozen=True)
class DemoScenario:
    scenario_id: str
    name: str
    kind: ScenarioKind
    operating_status: str
    summary: str
    alert: DemoAlert | None
    sections: tuple[DemoSection, ...]
    nodes: tuple[DemoNode, ...]


@dataclass(frozen=True)
class SectionTelemetry:
    section_id: str
    pressure_kpa: tuple[float, ...]
    flow_lpm: tuple[float, ...]
    expected_pressure_kpa: float
    expected_flow_lpm: float

    @property
    def latest_pressure_kpa(self) -> float:
        return self.pressure_kpa[-1]

    @property
    def latest_flow_lpm(self) -> float:
        return self.flow_lpm[-1]


@dataclass(frozen=True)
class DemoSnapshot:
    scenario: DemoScenario
    telemetry: tuple[SectionTelemetry, ...]
    alert_status: AlertStatus | None

    @property
    def active_alerts(self) -> tuple[DemoAlert, ...]:
        if self.scenario.alert is None or self.alert_status is AlertStatus.RESOLVED:
            return ()
        return (self.scenario.alert,)

    @property
    def critical_alerts(self) -> tuple[DemoAlert, ...]:
        if (
            self.scenario.alert is None
            or self.scenario.alert.severity != "Critical"
            or self.alert_status
            not in (AlertStatus.NEW, AlertStatus.INVESTIGATING)
        ):
            return ()
        return (self.scenario.alert,)

    @property
    def normal_sections(self) -> int:
        if not self.active_alerts:
            return len(self.scenario.sections)
        affected = {alert.section_id for alert in self.active_alerts}
        return sum(section.section_id not in affected for section in self.scenario.sections)

    @property
    def overall_status(self) -> str:
        if self.critical_alerts:
            return "Investigation required"
        if self.active_alerts:
            return "Attention"
        return "Normal operation"

    def telemetry_for(self, section_id: str) -> SectionTelemetry:
        for item in self.telemetry:
            if item.section_id == section_id:
                return item
        raise ValueError(f"Unknown illustrative section: {section_id}")


NODES = (
    DemoNode("N-01", "Supply header", -4.5, 0.0),
    DemoNode("N-02", "North junction", -1.6, 0.0),
    DemoNode("N-03", "North branch", 0.5, 2.0),
    DemoNode("N-04", "North outlet", 3.5, 2.0),
    DemoNode("N-05", "South branch", 0.5, -2.0),
    DemoNode("N-06", "South outlet", 3.5, -2.0),
    DemoNode("N-07", "Return header", 5.0, 0.0),
)

SECTIONS = (
    DemoSection("S-01", "Inlet main", "N-01", "N-02", ((-4.5, 0.0), (-1.6, 0.0))),
    DemoSection("S-02", "North feed", "N-02", "N-03", ((-1.6, 0.0), (-0.7, 0.9), (0.5, 2.0))),
    DemoSection("S-03", "North riser", "N-03", "N-04", ((0.5, 2.0), (3.5, 2.0))),
    DemoSection("S-04", "South feed", "N-02", "N-05", ((-1.6, 0.0), (-0.7, -0.9), (0.5, -2.0))),
    DemoSection("S-05", "South riser", "N-05", "N-06", ((0.5, -2.0), (3.5, -2.0))),
    DemoSection("S-06", "Return main", "N-02", "N-07", ((-1.6, 0.0), (5.0, 0.0))),
)

_PRESSURE_BASE = (420, 422, 419, 421, 423, 420, 421, 418, 420, 422, 419, 421)
_FLOW_BASE = (18.2, 18.4, 18.1, 18.3, 18.5, 18.2, 18.4, 18.1, 18.3, 18.4, 18.2, 18.3)
_SECTION_OFFSETS = {
    "S-01": (14.0, -0.5),
    "S-02": (7.0, 0.2),
    "S-03": (0.0, 0.0),
    "S-04": (-8.0, -0.2),
    "S-05": (-16.0, 0.4),
    "S-06": (-23.0, -0.4),
}

SCENARIO_CHOICES = (
    ("normal", "Normal operation"),
    ("pressure", "Pressure fluctuation"),
    ("flow", "Abnormal flow"),
    ("leak", "Suspected leak"),
    ("acknowledged", "Acknowledged event"),
    ("resolved", "Resolved event"),
)

_ALERTS = {
    "pressure": DemoAlert(
        "ALT-2048",
        "Pressure below expected range",
        "High",
        AlertStatus.INVESTIGATING,
        "S-03",
        "DEMO SEQUENCE · 09",
        374.0,
        421.0,
        18.2,
        18.3,
        "Pressure fell below the illustrative operating band while flow remained near its reference.",
        "Inspect the north riser and verify pressure instrumentation before escalation.",
    ),
    "flow": DemoAlert(
        "ALT-2051",
        "Unexpected flow increase",
        "Medium",
        AlertStatus.NEW,
        "S-05",
        "DEMO SEQUENCE · 10",
        405.0,
        404.0,
        31.7,
        18.7,
        "Observed flow is elevated relative to the illustrative section baseline.",
        "Compare adjacent section readings and check for demand changes or instrumentation faults.",
    ),
    "leak": DemoAlert(
        "ALT-2054",
        "Suspected leak pattern",
        "Critical",
        AlertStatus.NEW,
        "S-04",
        "DEMO SEQUENCE · 11",
        352.0,
        413.0,
        29.8,
        18.1,
        "The demonstration profile combines a sustained pressure decline with elevated flow.",
        "Dispatch an operator to verify the south feed and inspect the section before any repair decision.",
    ),
    "acknowledged": DemoAlert(
        "ALT-2054",
        "Pressure and flow deviation",
        "Critical",
        AlertStatus.ACKNOWLEDGED,
        "S-04",
        "DEMO SEQUENCE · 11",
        352.0,
        413.0,
        29.8,
        18.1,
        "The demonstration profile combines a sustained pressure decline with elevated flow.",
        "Dispatch an operator to verify the south feed and inspect the section before any repair decision.",
    ),
    "resolved": DemoAlert(
        "ALT-2054",
        "Pressure and flow deviation",
        "Critical",
        AlertStatus.RESOLVED,
        "S-04",
        "DEMO SEQUENCE · 11",
        352.0,
        413.0,
        29.8,
        18.1,
        "This resolved example retains an earlier illustrative event after the profile returns to its normal band.",
        "Resolution is part of the demonstration workflow; it is not evidence of a real repair.",
        "Demo resolution: section returned to its illustrative operating band.",
    ),
}


def scenario_by_id(scenario_id: str) -> DemoScenario:
    if scenario_id not in dict(SCENARIO_CHOICES):
        raise ValueError(f"Unknown demonstration scenario: {scenario_id}")
    title = dict(SCENARIO_CHOICES)[scenario_id]
    kind = {
        "normal": ScenarioKind.NORMAL,
        "pressure": ScenarioKind.PRESSURE,
        "flow": ScenarioKind.FLOW,
        "leak": ScenarioKind.LEAK,
        "acknowledged": ScenarioKind.ACKNOWLEDGED,
        "resolved": ScenarioKind.RESOLVED,
    }[scenario_id]
    alert = _ALERTS.get(scenario_id)
    operating_status = (
        "Normal"
        if scenario_id == "normal" or scenario_id == "resolved"
        else "Investigation required"
        if scenario_id in ("leak", "acknowledged")
        else "Attention"
    )
    summaries = {
        "normal": "All illustrative sections are within their expected operating bands.",
        "pressure": "A pressure deviation on the north riser is awaiting operator review.",
        "flow": "An elevated flow pattern on the south riser requires verification.",
        "leak": "A combined pressure and flow pattern is flagged for investigation.",
        "acknowledged": "An operator has acknowledged the illustrative pressure/flow event.",
        "resolved": "A previous illustrative event is marked resolved; current values are normal.",
    }
    return DemoScenario(
        scenario_id=scenario_id,
        name=title,
        kind=kind,
        operating_status=operating_status,
        summary=summaries[scenario_id],
        alert=alert,
        sections=SECTIONS,
        nodes=NODES,
    )


def build_snapshot(
    scenario_id: str, alert_status: AlertStatus | None = None
) -> DemoSnapshot:
    scenario = scenario_by_id(scenario_id)
    alert = scenario.alert
    effective_status = alert_status if alert_status is not None else (
        alert.initial_status if alert is not None else None
    )

    telemetry: list[SectionTelemetry] = []
    for section in scenario.sections:
        pressure_offset, flow_offset = _SECTION_OFFSETS[section.section_id]
        pressure = [value + pressure_offset for value in _PRESSURE_BASE]
        flow = [value + flow_offset for value in _FLOW_BASE]

        if (
            scenario.kind is ScenarioKind.PRESSURE
            and section.section_id == "S-03"
            and effective_status is not AlertStatus.RESOLVED
        ):
            pressure[-5:] = (406.0, 399.0, 391.0, 382.0, 374.0)
        elif (
            scenario.kind is ScenarioKind.FLOW
            and section.section_id == "S-05"
            and effective_status is not AlertStatus.RESOLVED
        ):
            flow[-5:] = (21.4, 24.0, 27.1, 30.0, 31.7)
        elif (
            scenario.kind in (ScenarioKind.LEAK, ScenarioKind.ACKNOWLEDGED)
            and section.section_id == "S-04"
            and effective_status is not AlertStatus.RESOLVED
        ):
            pressure[-5:] = (397.0, 386.0, 375.0, 363.0, 352.0)
            flow[-5:] = (21.2, 23.4, 25.6, 27.8, 29.8)

        expected_pressure = round(_PRESSURE_BASE[-1] + pressure_offset, 1)
        expected_flow = round(_FLOW_BASE[-1] + flow_offset, 1)
        telemetry.append(
            SectionTelemetry(
                section_id=section.section_id,
                pressure_kpa=tuple(round(value, 1) for value in pressure),
                flow_lpm=tuple(round(value, 1) for value in flow),
                expected_pressure_kpa=expected_pressure,
                expected_flow_lpm=expected_flow,
            )
        )

    return DemoSnapshot(
        scenario=scenario,
        telemetry=tuple(telemetry),
        alert_status=effective_status,
    )


def section_status(snapshot: DemoSnapshot, section_id: str) -> str:
    for alert in snapshot.active_alerts:
        if alert.section_id == section_id:
            if snapshot.alert_status is AlertStatus.ACKNOWLEDGED:
                return "attention"
            if alert.severity == "Critical":
                return "critical"
            return "attention"
    if snapshot.scenario.alert and snapshot.scenario.alert.section_id == section_id:
        if snapshot.alert_status is AlertStatus.RESOLVED:
            return "normal"
    return "normal"
