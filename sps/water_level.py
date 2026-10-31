"""Threshold-based water-level monitoring for SPS.

Water level is a separate monitoring signal. It is never added to the saved
model's feature schema (``SPSInference``) or to the synthetic localization
sensor vector. Readings produced here are simulated; no physical water-level
sensor (for example the HW-038 from the original hardware concept) is
connected.

Rule (project requirement, threshold = 200 in project units):
* reading > 200  -> ``NORMAL``
* reading < 200  -> ``POTENTIAL_LEAK`` (potential leak indication, not a
  confirmed leak)
* reading == 200 -> ``BOUNDARY`` (explicit boundary condition: it neither
  opens a new alert event nor counts toward clearing an active one)
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
from numbers import Real

import numpy as np


WATER_LEVEL_SENSOR_ID = "WATER_LEVEL_RESERVOIR"
# Extra entropy so the simulated water-level noise uses its own random stream
# and never shifts the noise of the pipe-network sensors for the same seed.
_WATER_LEVEL_STREAM_KEY = 0x574C  # "WL"


class WaterLevelCondition(str, Enum):
    NORMAL = "normal"
    POTENTIAL_LEAK = "potential_leak_indication"
    BOUNDARY = "boundary_review"
    UNAVAILABLE = "unavailable"
    INVALID = "invalid"
    OUT_OF_RANGE = "out_of_range"


@dataclass(frozen=True)
class WaterLevelConfig:
    threshold: float = 200.0
    baseline: float = 240.0
    minimum: float = 0.0
    maximum: float | None = None
    sensor_id: str = WATER_LEVEL_SENSOR_ID
    unit: str = "project units"
    leak_response_per_strength: float = 10.0
    noise_scale: float = 1.5
    # Consecutive readings above the threshold needed to close an active alert
    # event. Prevents repeated alerts when readings hover around the threshold.
    clear_after_normal_readings: int = 3

    def __post_init__(self) -> None:
        if (
            isinstance(self.clear_after_normal_readings, bool)
            or not isinstance(self.clear_after_normal_readings, int)
            or self.clear_after_normal_readings < 1
        ):
            raise ValueError("clear_after_normal_readings must be a positive integer")
        numeric_values = (
            self.threshold,
            self.baseline,
            self.minimum,
            self.leak_response_per_strength,
            self.noise_scale,
        )
        if any(
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not math.isfinite(value)
            for value in numeric_values
        ):
            raise ValueError("Water-level configuration values must be finite numbers")
        if self.maximum is not None and (
            isinstance(self.maximum, bool)
            or not isinstance(self.maximum, Real)
            or not math.isfinite(self.maximum)
        ):
            raise ValueError("maximum must be a finite number or None")
        if self.minimum < 0:
            raise ValueError("minimum must be non-negative")
        if self.threshold < self.minimum:
            raise ValueError("threshold must be at least minimum")
        if self.maximum is not None and self.maximum <= self.threshold:
            raise ValueError("maximum must be greater than threshold")
        if self.baseline <= self.threshold:
            raise ValueError("baseline must be greater than threshold")
        if self.maximum is not None and self.baseline > self.maximum:
            raise ValueError("baseline must not exceed maximum")
        if self.leak_response_per_strength <= 0 or self.noise_scale < 0:
            raise ValueError("leak response must be positive and noise non-negative")
        if not self.sensor_id or not self.unit:
            raise ValueError("sensor_id and unit must be non-empty")


DEFAULT_WATER_LEVEL_CONFIG = WaterLevelConfig()

# Operator-facing wording for each condition (used by the dashboard).
WATER_LEVEL_STATUS_TEXT = {
    WaterLevelCondition.NORMAL: "Normal",
    WaterLevelCondition.POTENTIAL_LEAK: "Potential leak indication",
    WaterLevelCondition.BOUNDARY: "At threshold · review",
    WaterLevelCondition.UNAVAILABLE: "No reading",
    WaterLevelCondition.INVALID: "Invalid reading",
    WaterLevelCondition.OUT_OF_RANGE: "Out of range",
}


@dataclass(frozen=True)
class WaterLevelAssessment:
    sensor_id: str
    reading: float | None
    condition: WaterLevelCondition
    alert_triggered: bool
    # True while a potential-leak event is open (after this reading).
    event_active: bool = False
    # Ordinal of the open event (1, 2, ...); None when no event is open.
    event_number: int | None = None


class WaterLevelMonitor:
    """Apply the SPS project threshold without affecting model or localization inputs.

    The monitor is stateful so that one continuous low-level event raises
    exactly one alert (``alert_triggered`` is True only on the reading that
    opens the event). An event closes after
    ``config.clear_after_normal_readings`` consecutive readings above the
    threshold. Boundary readings (exactly at the threshold), unavailable,
    invalid and out-of-range readings neither open nor close an event.
    """

    def __init__(self, config: WaterLevelConfig = DEFAULT_WATER_LEVEL_CONFIG) -> None:
        if not isinstance(config, WaterLevelConfig):
            raise TypeError("config must be a WaterLevelConfig")
        self.config = config
        self.reset()

    def reset(self) -> None:
        """Forget any open event (e.g. when a simulation is restarted)."""
        self._alert_active = False
        self._normal_streak = 0
        self._event_count = 0

    @property
    def event_active(self) -> bool:
        return self._alert_active

    @property
    def event_count(self) -> int:
        return self._event_count

    def assess(self, reading: object | None) -> WaterLevelAssessment:
        if reading is None:
            return self._assessment(None, WaterLevelCondition.UNAVAILABLE)
        if isinstance(reading, bool) or not isinstance(reading, Real):
            return self._assessment(None, WaterLevelCondition.INVALID)

        value = float(reading)
        if not math.isfinite(value):
            return self._assessment(None, WaterLevelCondition.INVALID)
        if value < self.config.minimum or (
            self.config.maximum is not None and value > self.config.maximum
        ):
            return self._assessment(value, WaterLevelCondition.OUT_OF_RANGE)
        if value > self.config.threshold:
            if self._alert_active:
                self._normal_streak += 1
                if self._normal_streak >= self.config.clear_after_normal_readings:
                    self._alert_active = False
                    self._normal_streak = 0
            return self._assessment(value, WaterLevelCondition.NORMAL)
        if value == self.config.threshold:
            # Explicit boundary: not normal (does not count toward clearing),
            # not below threshold (does not open a new event).
            self._normal_streak = 0
            return self._assessment(value, WaterLevelCondition.BOUNDARY)

        self._normal_streak = 0
        alert_triggered = not self._alert_active
        if alert_triggered:
            self._alert_active = True
            self._event_count += 1
        return self._assessment(
            value,
            WaterLevelCondition.POTENTIAL_LEAK,
            alert_triggered=alert_triggered,
        )

    def _assessment(
        self,
        reading: float | None,
        condition: WaterLevelCondition,
        *,
        alert_triggered: bool = False,
    ) -> WaterLevelAssessment:
        return WaterLevelAssessment(
            sensor_id=self.config.sensor_id,
            reading=reading,
            condition=condition,
            alert_triggered=alert_triggered,
            event_active=self._alert_active,
            event_number=self._event_count if self._alert_active else None,
        )


@dataclass(frozen=True)
class WaterLevelEvent:
    """One continuous potential-leak indication from the water-level signal."""

    event_number: int
    start_step: int
    end_step: int | None  # step of the reading that cleared it; None if still open
    lowest_reading: float
    readings_below_threshold: int

    @property
    def is_active(self) -> bool:
        return self.end_step is None


def summarize_water_level_events(
    timeline: Iterable[tuple[int, WaterLevelAssessment]],
) -> tuple[WaterLevelEvent, ...]:
    """Collapse per-step assessments into one record per continuous event.

    ``timeline`` holds ``(step, assessment)`` pairs in step order, as produced
    by a single ``WaterLevelMonitor``. The result never contains duplicates
    for one continuous event, so it can back an alert list directly.
    """
    events: list[WaterLevelEvent] = []
    open_event: dict[str, object] | None = None
    for step, assessment in timeline:
        if not isinstance(assessment, WaterLevelAssessment):
            raise TypeError("timeline must contain WaterLevelAssessment items")
        if assessment.alert_triggered:
            if open_event is not None:  # defensive: previous event never closed
                events.append(_close(open_event, None))
            open_event = {
                "number": assessment.event_number or len(events) + 1,
                "start": step,
                "lowest": assessment.reading,
                "below": 0,
            }
        if open_event is None:
            continue
        if (
            assessment.condition is WaterLevelCondition.POTENTIAL_LEAK
            and assessment.reading is not None
        ):
            open_event["below"] = int(open_event["below"]) + 1
            open_event["lowest"] = min(float(open_event["lowest"]), assessment.reading)
        if not assessment.event_active:
            events.append(_close(open_event, step))
            open_event = None
    if open_event is not None:
        events.append(_close(open_event, None))
    return tuple(events)


def _close(state: dict[str, object], end_step: int | None) -> WaterLevelEvent:
    return WaterLevelEvent(
        event_number=int(state["number"]),
        start_step=int(state["start"]),
        end_step=end_step,
        lowest_reading=float(state["lowest"]),
        readings_below_threshold=int(state["below"]),
    )


def water_level_rng(seed: int) -> np.random.Generator:
    """Independent random stream for simulated water-level noise.

    Using a separate generator keeps the pipe-sensor readings of a scenario
    identical whether or not water level is simulated.
    """
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    return np.random.default_rng([seed, _WATER_LEVEL_STREAM_KEY])


def simulate_water_level_reading(
    *,
    step: int,
    event_scale: float,
    rng: np.random.Generator,
    config: WaterLevelConfig = DEFAULT_WATER_LEVEL_CONFIG,
) -> float:
    """Generate an illustrative level signal; its response is a synthetic assumption."""
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise ValueError("step must be a non-negative integer")
    if (
        isinstance(event_scale, bool)
        or not isinstance(event_scale, Real)
        or not math.isfinite(event_scale)
        or event_scale < 0
    ):
        raise ValueError("event_scale must be finite and non-negative")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a NumPy Generator")
    if not isinstance(config, WaterLevelConfig):
        raise TypeError("config must be a WaterLevelConfig")

    baseline = config.baseline + 2.0 * math.sin(step / 5.0)
    event_drop = float(event_scale) * config.leak_response_per_strength
    noise = float(rng.normal(0.0, config.noise_scale))
    reading = max(config.minimum, baseline - event_drop + noise)
    if config.maximum is not None:
        reading = min(config.maximum, reading)
    return reading
