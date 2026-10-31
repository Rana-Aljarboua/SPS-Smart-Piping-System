from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from types import MappingProxyType

from sps.simulated_localization import (
    DEFAULT_NETWORK,
    LocalizationEstimate,
    PipeNetwork,
    ScenarioTruth,
    SensorEvidence,
    SensorFrame,
    SimulatedScenario,
    iter_scenario_frames,
    estimate_affected_segment,
)
from sps.water_level import (
    DEFAULT_WATER_LEVEL_CONFIG,
    WaterLevelAssessment,
    WaterLevelConfig,
    WaterLevelMonitor,
)


@dataclass(frozen=True)
class LocalizationTruth:
    """Evaluation-only simulator truth; never included in estimator inputs."""

    leak_active: bool
    leak_segment_id: str | None
    onset_step: int | None


@dataclass(frozen=True)
class LocalizationStepResult:
    step: int
    time_seconds: float
    scenario_id: str
    sensor_readings: Mapping[str, float]
    normal_reference: Mapping[str, float]
    estimated_segment_id: str | None
    localization_status: str
    diagnostic_scores: Mapping[str, float | None]
    segment_diagnostic_scores: Mapping[str, float]
    evidence: tuple[str, ...]
    sensor_ids: tuple[str, ...]
    segment_ids: tuple[str, ...]
    water_level_reading: float | None
    water_level_assessment: WaterLevelAssessment
    ground_truth: LocalizationTruth


def _truth_annotation(truth: ScenarioTruth) -> LocalizationTruth:
    return LocalizationTruth(
        leak_active=truth.leak_active,
        leak_segment_id=truth.leak_segment_id,
        onset_step=truth.onset_step,
    )


def _human_evidence(evidence: tuple[SensorEvidence, ...]) -> tuple[str, ...]:
    return tuple(
        (
            f"{item.sensor_id}: change {item.observed_delta:+.3f} "
            f"({item.standardized_delta:+.2f} noise-scaled units; "
            f"expected {item.expected_standardized_delta:+.2f})"
        )
        for item in evidence
    )


def localize_frame(
    frame: SensorFrame,
    *,
    network: PipeNetwork = DEFAULT_NETWORK,
    water_level_monitor: WaterLevelMonitor | None = None,
) -> LocalizationStepResult:
    """Estimate a frame from readings only; retain simulator truth separately."""
    if not isinstance(frame, SensorFrame):
        raise TypeError("frame must be a SensorFrame")
    if not isinstance(frame.ground_truth, ScenarioTruth):
        raise TypeError("frame ground_truth must be a ScenarioTruth")

    estimate: LocalizationEstimate = estimate_affected_segment(
        frame.sensor_readings,
        frame.normal_reference,
        network=network,
    )
    monitor = WaterLevelMonitor() if water_level_monitor is None else water_level_monitor
    if not isinstance(monitor, WaterLevelMonitor):
        raise TypeError("water_level_monitor must be a WaterLevelMonitor or None")
    water_level_assessment = monitor.assess(frame.water_level_reading)
    readings = MappingProxyType(dict(frame.sensor_readings))
    reference = MappingProxyType(dict(frame.normal_reference))
    scores = MappingProxyType(
        {
            "similarity_score": (
                float(estimate.similarity_score)
                if estimate.similarity_score is not None
                else None
            ),
            "score_margin": (
                float(estimate.score_margin)
                if estimate.score_margin is not None
                else None
            ),
        }
    )
    segment_scores = MappingProxyType(
        {
            segment_id: float(score)
            for segment_id, score in (
                estimate.segment_diagnostic_scores or {}
            ).items()
        }
    )
    return LocalizationStepResult(
        step=frame.step,
        time_seconds=frame.time_seconds,
        scenario_id=frame.ground_truth.scenario_id,
        sensor_readings=readings,
        normal_reference=reference,
        estimated_segment_id=estimate.estimated_segment_id,
        localization_status=estimate.status,
        diagnostic_scores=scores,
        segment_diagnostic_scores=segment_scores,
        evidence=_human_evidence(estimate.evidence),
        sensor_ids=network.sensor_ids,
        segment_ids=network.segment_ids,
        water_level_reading=frame.water_level_reading,
        water_level_assessment=water_level_assessment,
        ground_truth=_truth_annotation(frame.ground_truth),
    )


def iter_localization_results(
    scenario: SimulatedScenario,
    *,
    frame_count: int = 30,
    sampling_interval_seconds: float = 1.0,
    network: PipeNetwork = DEFAULT_NETWORK,
    water_level_config: WaterLevelConfig = DEFAULT_WATER_LEVEL_CONFIG,
) -> Iterator[LocalizationStepResult]:
    """Yield sequential estimates from the existing simulator, without truth leakage."""
    water_level_monitor = WaterLevelMonitor(water_level_config)
    for frame in iter_scenario_frames(
        scenario,
        frame_count=frame_count,
        sampling_interval_seconds=sampling_interval_seconds,
        network=network,
        water_level_config=water_level_config,
    ):
        yield localize_frame(
            frame,
            network=network,
            water_level_monitor=water_level_monitor,
        )
