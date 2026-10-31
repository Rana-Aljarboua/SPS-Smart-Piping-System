from __future__ import annotations

import math
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from numbers import Real
from types import MappingProxyType

import numpy as np

from sps.water_level import (
    DEFAULT_WATER_LEVEL_CONFIG,
    WaterLevelConfig,
    simulate_water_level_reading,
    water_level_rng,
)


@dataclass(frozen=True)
class PipeSegment:
    segment_id: str
    start_node: str
    end_node: str
    name: str


@dataclass(frozen=True)
class SensorSpec:
    sensor_id: str
    kind: str
    unit: str
    baseline: float
    noise_scale: float
    node_id: str | None = None
    segment_id: str | None = None


@dataclass(frozen=True)
class SegmentSignature:
    segment_id: str
    standardized_response: tuple[float, ...]


@dataclass(frozen=True)
class PipeNetwork:
    nodes: tuple[str, ...]
    segments: tuple[PipeSegment, ...]
    sensors: tuple[SensorSpec, ...]
    signatures: tuple[SegmentSignature, ...]

    def __post_init__(self) -> None:
        node_ids = set(self.nodes)
        segment_ids = {segment.segment_id for segment in self.segments}
        sensor_ids = {sensor.sensor_id for sensor in self.sensors}
        signature_ids = {signature.segment_id for signature in self.signatures}
        if len(node_ids) != len(self.nodes) or not node_ids:
            raise ValueError("Network node IDs must be unique and non-empty")
        if len(segment_ids) != len(self.segments) or not segment_ids:
            raise ValueError("Pipe segment IDs must be unique and non-empty")
        if len(sensor_ids) != len(self.sensors) or not sensor_ids:
            raise ValueError("Sensor IDs must be unique and non-empty")
        if signature_ids != segment_ids:
            raise ValueError("Every pipe segment must have exactly one response signature")
        for segment in self.segments:
            if segment.start_node not in node_ids or segment.end_node not in node_ids:
                raise ValueError(f"Unknown endpoint in pipe segment {segment.segment_id}")
        for sensor in self.sensors:
            if (sensor.node_id is None) == (sensor.segment_id is None):
                raise ValueError(
                    f"Sensor {sensor.sensor_id} must link to exactly one node or segment"
                )
            if sensor.node_id is not None and sensor.node_id not in node_ids:
                raise ValueError(f"Unknown node for sensor {sensor.sensor_id}")
            if sensor.segment_id is not None and sensor.segment_id not in segment_ids:
                raise ValueError(f"Unknown segment for sensor {sensor.sensor_id}")
            if (
                not math.isfinite(sensor.baseline)
                or not math.isfinite(sensor.noise_scale)
                or sensor.noise_scale <= 0
            ):
                raise ValueError(f"Invalid baseline/noise scale for sensor {sensor.sensor_id}")
        if any(len(signature.standardized_response) != len(self.sensors) for signature in self.signatures):
            raise ValueError("Each segment signature must contain one response per sensor")
        if any(
            not math.isfinite(value)
            for signature in self.signatures
            for value in signature.standardized_response
        ):
            raise ValueError("Segment signatures must contain finite values")

    @property
    def sensor_ids(self) -> tuple[str, ...]:
        return tuple(sensor.sensor_id for sensor in self.sensors)

    @property
    def segment_ids(self) -> tuple[str, ...]:
        return tuple(segment.segment_id for segment in self.segments)

    def signature_for(self, segment_id: str) -> tuple[float, ...]:
        for signature in self.signatures:
            if signature.segment_id == segment_id:
                return signature.standardized_response
        raise ValueError(f"Unknown pipe segment: {segment_id}")


DEFAULT_NETWORK = PipeNetwork(
    nodes=("SUPPLY", "JUNCTION", "NORTH", "NORTH_END", "SOUTH", "SOUTH_END"),
    segments=(
        PipeSegment("P-01", "SUPPLY", "JUNCTION", "Supply main"),
        PipeSegment("P-02", "JUNCTION", "NORTH", "North feed"),
        PipeSegment("P-03", "NORTH", "NORTH_END", "North branch"),
        PipeSegment("P-04", "JUNCTION", "SOUTH", "South feed"),
        PipeSegment("P-05", "SOUTH", "SOUTH_END", "South branch"),
    ),
    sensors=(
        SensorSpec("PRESSURE_SUPPLY", "pressure", "kPa", 420.0, 2.0, node_id="SUPPLY"),
        SensorSpec("PRESSURE_JUNCTION", "pressure", "kPa", 410.0, 2.0, node_id="JUNCTION"),
        SensorSpec("PRESSURE_NORTH", "pressure", "kPa", 405.0, 2.0, node_id="NORTH"),
        SensorSpec("PRESSURE_NORTH_END", "pressure", "kPa", 400.0, 2.0, node_id="NORTH_END"),
        SensorSpec("PRESSURE_SOUTH", "pressure", "kPa", 405.0, 2.0, node_id="SOUTH"),
        SensorSpec("PRESSURE_SOUTH_END", "pressure", "kPa", 400.0, 2.0, node_id="SOUTH_END"),
        SensorSpec("FLOW_MAIN", "flow", "L/min", 18.0, 0.5, segment_id="P-01"),
        SensorSpec("FLOW_NORTH_FEED", "flow", "L/min", 8.0, 0.3, segment_id="P-02"),
        SensorSpec("FLOW_NORTH_BRANCH", "flow", "L/min", 4.0, 0.2, segment_id="P-03"),
        SensorSpec("FLOW_SOUTH_FEED", "flow", "L/min", 8.0, 0.3, segment_id="P-04"),
        SensorSpec("FLOW_SOUTH_BRANCH", "flow", "L/min", 4.0, 0.2, segment_id="P-05"),
    ),
    signatures=(
        SegmentSignature(
            "P-01",
            (-0.35, -0.75, -0.70, -0.65, -0.70, -0.65, 1.00, 0.50, 0.35, 0.50, 0.35),
        ),
        SegmentSignature(
            "P-02",
            (-0.05, -0.20, -0.80, -0.72, -0.12, -0.10, 0.65, 1.00, 0.62, 0.08, 0.05),
        ),
        SegmentSignature(
            "P-03",
            (-0.03, -0.08, -0.24, -1.00, -0.05, -0.04, 0.42, 0.64, 1.00, 0.03, 0.02),
        ),
        SegmentSignature(
            "P-04",
            (-0.05, -0.20, -0.12, -0.10, -0.80, -0.72, 0.65, 0.08, 0.05, 1.00, 0.62),
        ),
        SegmentSignature(
            "P-05",
            (-0.03, -0.08, -0.05, -0.04, -0.24, -1.00, 0.42, 0.03, 0.02, 0.64, 1.00),
        ),
    ),
)


@dataclass(frozen=True)
class ScenarioTruth:
    scenario_id: str
    leak_segment_id: str | None
    onset_step: int | None
    leak_active: bool


@dataclass(frozen=True)
class SensorFrame:
    step: int
    time_seconds: float
    sensor_readings: Mapping[str, float]
    normal_reference: Mapping[str, float]
    ground_truth: ScenarioTruth
    water_level_reading: float | None = None


@dataclass(frozen=True)
class SimulatedScenario:
    scenario_id: str
    leak_segment_id: str | None = None
    onset_step: int | None = None
    leak_strength: float = 6.0
    ramp_steps: int = 3
    seed: int = 42

    def __post_init__(self) -> None:
        if not isinstance(self.scenario_id, str) or not self.scenario_id:
            raise ValueError("scenario_id must be a non-empty string")
        if self.leak_segment_id is None and self.onset_step is not None:
            raise ValueError("A normal scenario cannot have a leak onset step")
        if self.leak_segment_id is not None and self.onset_step is None:
            raise ValueError("A leak scenario must define its onset step")
        if self.onset_step is not None and (
            isinstance(self.onset_step, bool)
            or not isinstance(self.onset_step, int)
            or self.onset_step < 0
        ):
            raise ValueError("onset_step must be a non-negative integer")
        if (
            isinstance(self.leak_strength, bool)
            or not math.isfinite(self.leak_strength)
            or self.leak_strength <= 0
        ):
            raise ValueError("leak_strength must be finite and positive")
        if (
            isinstance(self.ramp_steps, bool)
            or not isinstance(self.ramp_steps, int)
            or self.ramp_steps < 1
        ):
            raise ValueError("ramp_steps must be a positive integer")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("seed must be an integer")


@dataclass(frozen=True)
class SensorEvidence:
    sensor_id: str
    observed_delta: float
    standardized_delta: float
    expected_standardized_delta: float


@dataclass(frozen=True)
class LocalizationEstimate:
    status: str
    estimated_segment_id: str | None
    similarity_score: float | None
    score_margin: float | None
    evidence: tuple[SensorEvidence, ...]
    segment_diagnostic_scores: Mapping[str, float] | None = None


def _validate_readings(
    readings: Mapping[str, Real],
    sensor_ids: tuple[str, ...],
    description: str,
) -> dict[str, float]:
    if not isinstance(readings, Mapping):
        raise TypeError(f"{description} must be a sensor-value mapping")
    if any(not isinstance(key, str) for key in readings):
        raise TypeError(f"{description} sensor names must be strings")
    missing = sorted(set(sensor_ids) - set(readings))
    extra = sorted(set(readings) - set(sensor_ids))
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing sensors: {', '.join(missing)}")
        if extra:
            details.append(f"unexpected sensors: {', '.join(extra)}")
        raise ValueError(f"{description} schema mismatch ({'; '.join(details)})")
    validated: dict[str, float] = {}
    for sensor_id in sensor_ids:
        value = readings[sensor_id]
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(f"{description} value for {sensor_id} must be numeric")
        numeric_value = float(value)
        if not math.isfinite(numeric_value):
            raise ValueError(f"{description} value for {sensor_id} must be finite")
        validated[sensor_id] = numeric_value
    return validated


def iter_scenario_frames(
    scenario: SimulatedScenario,
    *,
    frame_count: int = 30,
    sampling_interval_seconds: float = 1.0,
    network: PipeNetwork = DEFAULT_NETWORK,
    water_level_config: WaterLevelConfig = DEFAULT_WATER_LEVEL_CONFIG,
) -> Iterator[SensorFrame]:
    """Yield a deterministic synthetic sensor stream one timestamped frame at a time."""
    if not isinstance(scenario, SimulatedScenario):
        raise TypeError("scenario must be a SimulatedScenario")
    if (
        isinstance(frame_count, bool)
        or not isinstance(frame_count, int)
        or frame_count < 1
    ):
        raise ValueError("frame_count must be a positive integer")
    if scenario.onset_step is not None and scenario.onset_step >= frame_count:
        raise ValueError("onset_step must be within the generated frame range")
    if (
        isinstance(sampling_interval_seconds, bool)
        or not isinstance(sampling_interval_seconds, Real)
        or not math.isfinite(sampling_interval_seconds)
        or sampling_interval_seconds <= 0
    ):
        raise ValueError("sampling_interval_seconds must be finite and positive")
    if scenario.leak_segment_id is not None:
        network.signature_for(scenario.leak_segment_id)

    rng = np.random.default_rng(scenario.seed)
    # Separate stream: simulating water level must not change pipe-sensor noise.
    level_rng = water_level_rng(scenario.seed)
    sensor_ids = network.sensor_ids
    baselines = {sensor.sensor_id: sensor.baseline for sensor in network.sensors}
    noise_scales = {sensor.sensor_id: sensor.noise_scale for sensor in network.sensors}
    signature = (
        network.signature_for(scenario.leak_segment_id)
        if scenario.leak_segment_id is not None
        else tuple(0.0 for _ in network.sensors)
    )
    for step in range(frame_count):
        normal_reference = {
            sensor_id: baselines[sensor_id]
            + noise_scales[sensor_id] * 0.08 * math.sin(step / 5.0)
            for sensor_id in sensor_ids
        }
        event_scale = 0.0
        if scenario.onset_step is not None and step >= scenario.onset_step:
            event_scale = scenario.leak_strength * min(
                (step - scenario.onset_step + 1) / scenario.ramp_steps,
                1.0,
            )
        truth = ScenarioTruth(
            scenario.scenario_id,
            scenario.leak_segment_id,
            scenario.onset_step,
            event_scale > 0,
        )
        readings = {
            sensor_id: normal_reference[sensor_id]
            + noise_scales[sensor_id]
            * (
                event_scale * signature[index]
                + float(rng.normal(0.0, 0.08))
            )
            for index, sensor_id in enumerate(sensor_ids)
        }
        water_level_reading = simulate_water_level_reading(
            step=step,
            event_scale=event_scale,
            rng=level_rng,
            config=water_level_config,
        )
        yield SensorFrame(
            step=step,
            time_seconds=step * float(sampling_interval_seconds),
            sensor_readings=MappingProxyType(readings),
            normal_reference=MappingProxyType(normal_reference),
            ground_truth=truth,
            water_level_reading=water_level_reading,
        )


def generate_scenario(
    scenario: SimulatedScenario,
    *,
    frame_count: int = 30,
    sampling_interval_seconds: float = 1.0,
    network: PipeNetwork = DEFAULT_NETWORK,
    water_level_config: WaterLevelConfig = DEFAULT_WATER_LEVEL_CONFIG,
) -> tuple[SensorFrame, ...]:
    """Materialize a complete deterministic scenario for analysis and tests."""
    return tuple(
        iter_scenario_frames(
            scenario,
            frame_count=frame_count,
            sampling_interval_seconds=sampling_interval_seconds,
            network=network,
            water_level_config=water_level_config,
        )
    )


def estimate_affected_segment(
    sensor_readings: Mapping[str, Real],
    normal_reference: Mapping[str, Real],
    *,
    network: PipeNetwork = DEFAULT_NETWORK,
    minimum_signal_norm: float = 2.5,
    minimum_score_margin: float = 0.04,
    minimum_similarity: float = 0.55,
) -> LocalizationEstimate:
    """Localize using positive residuals on segment-linked flow sensors.

    Sensor-to-segment flow links are the localization evidence; pressure changes
    contribute to event-strength screening but are not assigned a segment. Scores
    are diagnostics, not calibrated probabilities.
    """
    if not math.isfinite(minimum_signal_norm) or minimum_signal_norm < 0:
        raise ValueError("minimum_signal_norm must be finite and non-negative")
    if not math.isfinite(minimum_score_margin) or minimum_score_margin < 0:
        raise ValueError("minimum_score_margin must be finite and non-negative")
    if not math.isfinite(minimum_similarity) or not -1 <= minimum_similarity <= 1:
        raise ValueError("minimum_similarity must be between -1 and 1")
    observed = _validate_readings(sensor_readings, network.sensor_ids, "sensor_readings")
    reference = _validate_readings(normal_reference, network.sensor_ids, "normal_reference")
    standardized = np.asarray(
        [
            (observed[sensor.sensor_id] - reference[sensor.sensor_id])
            / sensor.noise_scale
            for sensor in network.sensors
        ],
        dtype=float,
    )
    signal_norm = float(np.linalg.norm(standardized))
    if signal_norm < minimum_signal_norm:
        return LocalizationEstimate("no_leak_pattern", None, None, None, ())

    flow_sensors = {
        sensor.segment_id: sensor
        for sensor in network.sensors
        if sensor.kind == "flow" and sensor.segment_id is not None
    }
    if set(flow_sensors) != set(network.segment_ids) or len(flow_sensors) != len(
        network.segment_ids
    ):
        raise ValueError(
            "Localization requires exactly one segment-linked flow sensor "
            "per pipe segment"
        )
    pressure_sensors = {
        sensor.node_id: sensor
        for sensor in network.sensors
        if sensor.kind == "pressure" and sensor.node_id is not None
    }
    required_nodes = {
        node_id
        for segment in network.segments
        for node_id in (segment.start_node, segment.end_node)
    }
    if not required_nodes.issubset(pressure_sensors):
        raise ValueError(
            "Localization requires pressure sensors at every pipe-segment endpoint"
        )

    positive_flow_residuals = {
        segment_id: max(
            0.0,
            standardized[network.sensor_ids.index(sensor.sensor_id)],
        )
        for segment_id, sensor in flow_sensors.items()
    }
    minimum_flow_evidence = minimum_signal_norm / math.sqrt(len(network.sensor_ids))
    if max(positive_flow_residuals.values(), default=0.0) < minimum_flow_evidence:
        return LocalizationEstimate("insufficient_evidence", None, None, None, ())

    candidate_scores: list[tuple[str, float, SensorSpec]] = []
    for segment in network.segments:
        pressure_drop = sum(
            max(
                0.0,
                -standardized[
                    network.sensor_ids.index(pressure_sensors[node_id].sensor_id)
                ],
            )
            for node_id in (segment.start_node, segment.end_node)
        ) / 2.0
        score = positive_flow_residuals[segment.segment_id] + pressure_drop
        candidate_scores.append(
            (segment.segment_id, score, flow_sensors[segment.segment_id])
        )
    candidate_scores.sort(key=lambda result: result[1], reverse=True)
    score_norm = float(np.linalg.norm([candidate[1] for candidate in candidate_scores]))
    if score_norm == 0:
        return LocalizationEstimate("insufficient_evidence", None, None, None, ())
    segment_diagnostic_scores = MappingProxyType(
        {
            segment_id: float(score / score_norm)
            for segment_id, score, _sensor in candidate_scores
        }
    )
    best_segment, raw_best_score, best_sensor = candidate_scores[0]
    best_score = raw_best_score / score_norm
    runner_up_score = (
        candidate_scores[1][1] / score_norm if len(candidate_scores) > 1 else 0.0
    )
    margin = best_score - runner_up_score

    expected = np.zeros(len(network.sensors), dtype=float)
    expected[network.sensor_ids.index(best_sensor.sensor_id)] = (
        positive_flow_residuals[best_segment]
    )
    best_pipe = next(
        segment for segment in network.segments if segment.segment_id == best_segment
    )
    for node_id in (best_pipe.start_node, best_pipe.end_node):
        pressure_index = network.sensor_ids.index(pressure_sensors[node_id].sensor_id)
        expected[pressure_index] = min(0.0, standardized[pressure_index])
    evidence_indices = np.argsort(np.abs(standardized))[::-1][:5]
    evidence = tuple(
        SensorEvidence(
            sensor_id=network.sensors[index].sensor_id,
            observed_delta=float(
                observed[network.sensors[index].sensor_id]
                - reference[network.sensors[index].sensor_id]
            ),
            standardized_delta=float(standardized[index]),
            expected_standardized_delta=float(expected[index]),
        )
        for index in evidence_indices
    )
    if margin < minimum_score_margin or best_score < minimum_similarity:
        return LocalizationEstimate(
            "ambiguous",
            None,
            best_score,
            margin,
            evidence,
            segment_diagnostic_scores,
        )
    return LocalizationEstimate(
        "localized",
        best_segment,
        best_score,
        margin,
        evidence,
        segment_diagnostic_scores,
    )
