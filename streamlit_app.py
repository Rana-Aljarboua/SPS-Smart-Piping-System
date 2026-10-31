from __future__ import annotations

from collections.abc import Sequence
from queue import Queue
from typing import Any

import plotly.graph_objects as go
import streamlit as st

from sps.company_requirements import load_requirement_groups
from sps.dashboard_state import DashboardHistory
from sps.inference import SPSInference
from sps.localization_adapter import LocalizationStepResult, iter_localization_results
from sps.simulated_localization import (
    DEFAULT_NETWORK,
    SimulatedScenario,
)
from sps.simulated_stream import (
    SimulatedSensorStream,
    StreamBatch,
    StreamState,
)
from sps.water_level import (
    DEFAULT_WATER_LEVEL_CONFIG,
    WATER_LEVEL_STATUS_TEXT,
    WaterLevelCondition,
    summarize_water_level_events,
)


_NAVIGATION = (
    "Overview",
    "3D Pipeline Monitoring",
    "Analytics & Charts",
    "Alerts & Investigation",
    "Demo Environment",
    "Company Configuration / Requirements",
    "WDSEventDB Replay",
)
_COLORS = {
    "background": "#11161b",
    "surface": "#1a2229",
    "border": "#2d3942",
    "text": "#e6edf2",
    "muted": "#9eacb6",
    "blue": "#63a6d8",
    "green": "#54bd91",
    "amber": "#e1ad4f",
    "red": "#e36565",
    "steel": "#708696",
}
_SENSOR_BY_ID = {sensor.sensor_id: sensor for sensor in DEFAULT_NETWORK.sensors}
_SEGMENT_BY_ID = {segment.segment_id: segment for segment in DEFAULT_NETWORK.segments}
_NODE_POSITIONS = {
    "SUPPLY": (0.0, 0.0, 0.0),
    "JUNCTION": (2.2, 0.0, 0.0),
    "NORTH": (4.4, 1.6, 0.8),
    "NORTH_END": (6.6, 1.6, 1.1),
    "SOUTH": (4.4, -1.6, -0.5),
    "SOUTH_END": (6.6, -1.6, -0.8),
}


st.set_page_config(
    page_title="SPS | Pipeline Operations",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    :root { color-scheme: dark; }
    .stApp { background: #0b141a; color: #e6edf2; }
    [data-testid="stHeader"] { background: rgba(11,20,26,.98); }
    [data-testid="stToolbar"], [data-testid="stDeployButton"], #MainMenu, footer {
      display: none; visibility: hidden; }
    [data-testid="stSidebar"] { display: none; }
    .block-container { max-width: 1720px; padding: 1.35rem 2.2rem 2.5rem; }
    h1, h2, h3 { letter-spacing: -.035em; }
    h1 { font-size: 2rem; }
    h2 { font-size: 1.12rem; }
    [data-testid="stMetric"] { background: #111e26; border: 1px solid #263944;
      padding: 1rem 1.15rem; border-radius: 12px; min-height: 108px; }
    [data-testid="stMetricLabel"] { color: #9aadb8; font-size: .78rem; }
    [data-testid="stMetricValue"] { color: #f0f5f7; font-weight: 650; }
    [data-testid="stMetricValue"] { font-size: 1.55rem; }
    div[data-testid="stPlotlyChart"], [data-testid="stDataFrame"] {
      border: 1px solid #263944; border-radius: 12px; overflow: hidden;
      background: #101b22; }
    .react-aria-ComboBox {
      background-color: #111e26 !important;
      border: 1px solid #38525e !important;
      border-radius: 8px;
    }
    .react-aria-ComboBox [role="combobox"] {
      background-color: #111e26 !important;
      color: #e6edf2 !important;
    }
    .react-aria-ComboBox button {
      background-color: #111e26 !important;
      border-color: #38525e !important;
      color: #e6edf2 !important;
    }
    .sps-eyebrow { color: #73c5cf; font-size: .7rem; letter-spacing: .15em;
      font-weight: 700; text-transform: uppercase; }
    .sps-note { color: #9aadb8; font-size: .9rem; line-height: 1.5; }
    .sps-pill { display: inline-flex; align-items: center; gap: .45rem;
      border: 1px solid #31515c; border-radius: 99px; padding: .42rem .8rem;
      color: #bce7e7; background: #102a31; font-size: .72rem;
      font-weight: 700; letter-spacing: .08em; }
    .sps-dot { color: #58c999; font-size: 1rem; }
    .sps-card { background: #111e26; border: 1px solid #263944;
      border-radius: 12px; padding: 1rem 1.1rem; }
    div.stButton > button { border-radius: 8px; min-height: 2.55rem;
      font-weight: 650; border-color: #38525e; }
    div.stButton > button[kind="primary"] { background: #18a6ad;
      border-color: #18a6ad; color: #07171b; }
    [data-testid="stProgressBar"] > div > div { background: #25b7bd; }
    hr { border-color: #263944; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def _get_inference() -> SPSInference:
    return SPSInference()


def _initialize_session() -> None:
    defaults: dict[str, Any] = {
        "navigation_page": "Overview",
        "demo_choice": "P-03",
        "demo_strength": 6.0,
        "demo_speed": 0.5,
        "demo_results": (),
        "demo_signature": None,
        "demo_step": 0,
        "demo_running": False,
        "selected_network_entity": "segment:P-03",
        "alert_workflow": {},
        "wdse_history": DashboardHistory(max_history=800),
        "wdse_queue": Queue(),
        "wdse_stream": None,
        "wdse_error": None,
        "wdse_feature": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _scenario_for_choice(choice: str, strength: float) -> SimulatedScenario:
    if choice == "normal":
        return SimulatedScenario("synthetic-normal", seed=24)
    if choice not in DEFAULT_NETWORK.segment_ids:
        raise ValueError(f"Unknown synthetic scenario choice: {choice}")
    return SimulatedScenario(
        scenario_id=f"synthetic-{choice.lower()}",
        leak_segment_id=choice,
        onset_step=8,
        leak_strength=strength,
        ramp_steps=3,
        seed=24,
    )


def _ensure_demo_results(choice: str, strength: float) -> None:
    signature = (choice, float(strength))
    if st.session_state.demo_signature == signature:
        return
    scenario = _scenario_for_choice(choice, strength)
    st.session_state.demo_results = tuple(
        iter_localization_results(
            scenario,
            frame_count=60,
            sampling_interval_seconds=1.0,
        )
    )
    st.session_state.demo_signature = signature
    st.session_state.demo_step = 0
    st.session_state.demo_running = False
    st.session_state.alert_workflow = {}


def _current_results() -> tuple[LocalizationStepResult, ...]:
    results = st.session_state.demo_results
    if not results:
        _ensure_demo_results(
            st.session_state.demo_choice,
            st.session_state.demo_strength,
        )
        results = st.session_state.demo_results
    return results


def _current_result() -> LocalizationStepResult:
    results = _current_results()
    index = min(max(int(st.session_state.demo_step), 0), len(results) - 1)
    return results[index]


def _title(eyebrow: str, title: str, description: str | None = None) -> None:
    st.markdown(f'<div class="sps-eyebrow">{eyebrow}</div>', unsafe_allow_html=True)
    st.title(title)
    if description:
        st.markdown(f'<div class="sps-note">{description}</div>', unsafe_allow_html=True)


def _status_text(result: LocalizationStepResult) -> str:
    return {
        "localized": f"Estimated segment {result.estimated_segment_id}",
        "ambiguous": "Ambiguous — no segment selected",
        "insufficient_evidence": "Insufficient evidence",
        "no_leak_pattern": "No leak pattern detected",
    }.get(result.localization_status, result.localization_status)


def _format_reading(value: float | None, unit: str) -> str:
    return "Unavailable" if value is None else f"{value:.3f} {unit}"


def _entity_options() -> tuple[str, ...]:
    return tuple(
        [f"segment:{segment_id}" for segment_id in DEFAULT_NETWORK.segment_ids]
        + [f"sensor:{sensor_id}" for sensor_id in DEFAULT_NETWORK.sensor_ids]
    )


def _entity_label(entity_id: str) -> str:
    kind, identifier = entity_id.split(":", maxsplit=1)
    if kind == "segment":
        return f"Pipe {identifier} · {_SEGMENT_BY_ID[identifier].name}"
    sensor = _SENSOR_BY_ID[identifier]
    location = sensor.sensor_id.partition("_")[2].replace("_", " ").title()
    return f"{sensor.kind.title()} · {location}"


def _sensor_location(sensor_id: str) -> tuple[float, float, float]:
    sensor = _SENSOR_BY_ID[sensor_id]
    if sensor.node_id is not None:
        x, y, z = _NODE_POSITIONS[sensor.node_id]
        return (x, y + 0.16, z + 0.26)
    if sensor.segment_id is None:
        raise ValueError(f"Sensor {sensor_id} is not linked to the network")
    segment = _SEGMENT_BY_ID[sensor.segment_id]
    start = _NODE_POSITIONS[segment.start_node]
    end = _NODE_POSITIONS[segment.end_node]
    return (
        (start[0] + end[0]) / 2.0,
        (start[1] + end[1]) / 2.0,
        (start[2] + end[2]) / 2.0,
    )


def _pipeline_figure(
    result: LocalizationStepResult,
    selected_entity: str,
) -> go.Figure:
    figure = go.Figure()
    selected_kind, selected_id = selected_entity.split(":", maxsplit=1)
    estimated = (
        result.estimated_segment_id
        if result.localization_status == "localized"
        else None
    )
    for segment in DEFAULT_NETWORK.segments:
        is_selected = selected_kind == "segment" and selected_id == segment.segment_id
        is_estimated = estimated == segment.segment_id
        color = (
            _COLORS["blue"]
            if is_selected
            else _COLORS["red"]
            if is_estimated
            else _COLORS["steel"]
        )
        start = _NODE_POSITIONS[segment.start_node]
        end = _NODE_POSITIONS[segment.end_node]
        figure.add_trace(
            go.Scatter3d(
                x=[start[0], end[0]],
                y=[start[1], end[1]],
                z=[start[2], end[2]],
                mode="lines",
                line={"color": color, "width": 9 if is_selected or is_estimated else 6},
                customdata=[[segment.segment_id, segment.name]] * 2,
                hovertemplate=(
                    "<b>%{customdata[0]} · %{customdata[1]}</b>"
                    "<extra>Pipe segment</extra>"
                ),
                name=segment.segment_id,
                showlegend=False,
            )
        )

    evidence_sensor_ids = {
        line.partition(":")[0]
        for line in result.evidence
        if result.localization_status in ("localized", "ambiguous")
    }
    for kind, marker_symbol, marker_color in (
        ("pressure", "circle", "#62b994"),
        ("flow", "diamond", "#7ca9d0"),
    ):
        sensors = [sensor for sensor in DEFAULT_NETWORK.sensors if sensor.kind == kind]
        positions = [_sensor_location(sensor.sensor_id) for sensor in sensors]
        colors = []
        sizes = []
        for sensor in sensors:
            is_selected = selected_kind == "sensor" and selected_id == sensor.sensor_id
            colors.append(
                _COLORS["blue"]
                if is_selected
                else _COLORS["amber"]
                if sensor.sensor_id in evidence_sensor_ids
                else marker_color
            )
            sizes.append(10 if is_selected else 8)
        labels = [
            _entity_label(f"sensor:{sensor.sensor_id}")
            if selected_kind == "sensor" and selected_id == sensor.sensor_id
            else ""
            for sensor in sensors
        ]
        figure.add_trace(
            go.Scatter3d(
                x=[point[0] for point in positions],
                y=[point[1] for point in positions],
                z=[point[2] for point in positions],
                mode="markers+text",
                text=labels,
                textposition="top center",
                textfont={"size": 9, "color": "#d9e2e8"},
                marker={
                    "size": sizes,
                    "color": colors,
                    "symbol": marker_symbol,
                    "line": {"color": "#10161b", "width": 1},
                },
                customdata=[
                    [
                        sensor.sensor_id,
                        sensor.kind,
                        result.sensor_readings[sensor.sensor_id],
                        sensor.unit,
                        sensor.segment_id or sensor.node_id,
                    ]
                    for sensor in sensors
                ],
                hovertemplate=(
                    "<b>%{customdata[0]}</b> · %{customdata[1]}"
                    "<br>Reading: %{customdata[2]:.3f} %{customdata[3]}"
                    "<br>Mapped to: %{customdata[4]}<extra></extra>"
                ),
                name=f"{kind.title()} sensors",
                showlegend=True,
            )
        )

    figure.add_trace(
        go.Scatter3d(
            x=[point[0] for point in _NODE_POSITIONS.values()],
            y=[point[1] for point in _NODE_POSITIONS.values()],
            z=[point[2] for point in _NODE_POSITIONS.values()],
            mode="markers",
            customdata=list(_NODE_POSITIONS),
            marker={"size": 5, "color": "#d2dbe1"},
            hovertemplate="<b>%{customdata}</b><extra>Network node</extra>",
            name="Network nodes",
            showlegend=False,
        )
    )
    figure.update_layout(
        height=500,
        margin={"l": 0, "r": 0, "t": 15, "b": 0},
        paper_bgcolor=_COLORS["background"],
        plot_bgcolor=_COLORS["background"],
        font={"color": _COLORS["text"], "family": "Segoe UI, sans-serif"},
        legend={"orientation": "h", "y": -0.03},
        hoverlabel={"bgcolor": _COLORS["surface"], "font": {"color": _COLORS["text"]}},
        scene={
            "bgcolor": _COLORS["background"],
            "aspectmode": "data",
            "xaxis": {"visible": False},
            "yaxis": {"visible": False},
            "zaxis": {"visible": False},
            "camera": {"eye": {"x": 1.8, "y": 1.6, "z": 1.35}},
        },
        uirevision="sps-network-topology-v1",
    )
    return figure


def _render_network_inspector(result: LocalizationStepResult) -> None:
    options = _entity_options()
    if st.session_state.selected_network_entity not in options:
        st.session_state.selected_network_entity = options[0]
    selected = st.selectbox(
        "Select a sensor or pipe segment",
        options,
        format_func=_entity_label,
        key="selected_network_entity",
    )
    kind, identifier = selected.split(":", maxsplit=1)
    if kind == "segment":
        segment = _SEGMENT_BY_ID[identifier]
        linked_sensors = [
            sensor
            for sensor in DEFAULT_NETWORK.sensors
            if sensor.segment_id == identifier
            or sensor.node_id in (segment.start_node, segment.end_node)
        ]
        st.subheader(f"{segment.segment_id} · {segment.name}")
        st.caption(
            f"Topology link: {segment.start_node} → {segment.end_node} · "
            "illustrative coordinates, not a surveyed asset map."
        )
        st.write(
            "Related sensors: "
            + ", ".join(sensor.sensor_id for sensor in linked_sensors)
        )
        if result.localization_status == "localized" and result.estimated_segment_id == identifier:
            st.error("This segment is the current synthetic estimator output.")
        elif result.localization_status == "ambiguous":
            st.warning("The current estimate is ambiguous; no segment is selected by the estimator.")
        elif result.localization_status == "insufficient_evidence":
            st.info("Current sensor evidence is insufficient to identify a segment.")
        elif result.localization_status == "no_leak_pattern":
            st.success("No leak pattern was returned for the current synthetic frame.")
        score = result.segment_diagnostic_scores.get(identifier)
        st.metric(
            "Segment diagnostic score",
            "Unavailable" if score is None else f"{score:.3f}",
        )
    else:
        sensor = _SENSOR_BY_ID[identifier]
        reading = result.sensor_readings.get(identifier)
        reference = result.normal_reference.get(identifier)
        st.subheader(f"{sensor.sensor_id} · {sensor.kind.title()} sensor")
        st.metric("Current reading", _format_reading(reading, sensor.unit))
        st.metric("Normal reference", _format_reading(reference, sensor.unit))
        if reading is not None and reference is not None:
            st.metric("Observed change", f"{reading - reference:+.3f} {sensor.unit}")
        location = sensor.segment_id or sensor.node_id or "Unmapped"
        evidence_status = (
            "Included in estimator evidence"
            if any(sensor.sensor_id in line for line in result.evidence)
            else "No estimator evidence for this sensor"
        )
        st.caption(f"Mapped to {location} · {evidence_status}.")
        matching = [line for line in result.evidence if sensor.sensor_id in line]
        if matching:
            st.write("Localization evidence")
            for line in matching:
                st.write(f"- {line}")


def _plot_sensor_trends(
    results: Sequence[LocalizationStepResult],
    sensor_ids: Sequence[str],
    *,
    title: str,
) -> go.Figure:
    figure = go.Figure()
    for sensor_id in sensor_ids:
        if sensor_id not in _SENSOR_BY_ID:
            continue
        sensor = _SENSOR_BY_ID[sensor_id]
        figure.add_trace(
            go.Scatter(
                x=[item.time_seconds for item in results],
                y=[item.sensor_readings[sensor_id] for item in results],
                mode="lines+markers",
                marker={"size": 3},
                name=f"{sensor_id}",
                hovertemplate=(
                    "Time %{x:.0f} s<br>Reading %{y:.3f} "
                    f"{sensor.unit}<extra>%{{fullData.name}}</extra>"
                ),
            )
        )
    figure.update_layout(
        title=title,
        height=360,
        margin={"l": 10, "r": 15, "t": 55, "b": 10},
        paper_bgcolor=_COLORS["background"],
        plot_bgcolor=_COLORS["surface"],
        font={"color": _COLORS["text"]},
        xaxis={"title": "Time (s)", "gridcolor": "#263944"},
        yaxis={"title": "Reading", "gridcolor": "#263944"},
        legend={"orientation": "h", "y": -0.25},
        hovermode="x unified",
    )
    return figure


def _localization_score_figure(result: LocalizationStepResult) -> go.Figure | None:
    if not result.segment_diagnostic_scores:
        return None
    colors = [
        _COLORS["red"]
        if result.localization_status == "localized"
        and segment_id == result.estimated_segment_id
        else _COLORS["blue"]
        if st.session_state.selected_network_entity == f"segment:{segment_id}"
        else _COLORS["steel"]
        for segment_id in DEFAULT_NETWORK.segment_ids
    ]
    figure = go.Figure(
        go.Bar(
            x=list(DEFAULT_NETWORK.segment_ids),
            y=[
                result.segment_diagnostic_scores.get(segment_id, 0.0)
                for segment_id in DEFAULT_NETWORK.segment_ids
            ],
            marker_color=colors,
            hovertemplate="Segment %{x}<br>Diagnostic score %{y:.4f}<extra></extra>",
        )
    )
    figure.update_layout(
        title="Segment diagnostic scores · not probabilities",
        height=330,
        margin={"l": 10, "r": 10, "t": 55, "b": 10},
        paper_bgcolor=_COLORS["background"],
        plot_bgcolor=_COLORS["surface"],
        font={"color": _COLORS["text"]},
        xaxis={"title": "Pipe segment"},
        yaxis={"title": "Diagnostic score", "rangemode": "tozero"},
    )
    return figure


def _render_overview() -> None:
    result = _current_result()
    history: DashboardHistory = st.session_state.wdse_history
    _title(
        "Operations overview",
        "Network status",
        "Synthetic localization and WDSEventDB model replay are separate data paths. No hardware telemetry is connected.",
    )
    metrics = st.columns(6)
    metrics[0].metric("Sensors in synthetic network", len(result.sensor_ids))
    metrics[1].metric("Active in current frame", len(result.sensor_readings))
    metrics[2].metric("WDSEventDB normal predictions", history.normal_predictions)
    metrics[3].metric("WDSEventDB anomalous predictions", history.anomalous_predictions)
    metrics[4].metric(
        "Synthetic leak active",
        "Yes" if result.ground_truth.leak_active else "No",
    )
    metrics[5].metric("Localization status", result.localization_status.replace("_", " "))

    st.caption(
        f"Synthetic scenario {result.scenario_id} · step {result.step} · "
        f"estimated segment: {result.estimated_segment_id or 'none'}"
    )
    left, right = st.columns([1.25, 1], gap="large")
    with left:
        st.subheader("Pipeline and sensor map")
        st.plotly_chart(
            _pipeline_figure(result, st.session_state.selected_network_entity),
            width="stretch",
            config={"displaylogo": False, "scrollZoom": True},
            key="overview_pipeline_3d",
        )
        st.caption(
            "Illustrative 3D geometry derived from network connectivity; it is not a surveyed engineering map."
        )
    with right:
        st.subheader("Network inspector")
        _render_network_inspector(result)
        st.divider()
        st.subheader("Recent sensor activity")
        recent = []
        for sensor in DEFAULT_NETWORK.sensors:
            recent.append(
                {
                    "Sensor": sensor.sensor_id,
                    "Type": sensor.kind,
                    "Reading": result.sensor_readings[sensor.sensor_id],
                    "Unit": sensor.unit,
                    "Change": (
                        result.sensor_readings[sensor.sensor_id]
                        - result.normal_reference[sensor.sensor_id]
                    ),
                }
            )
        st.dataframe(recent, width="stretch", hide_index=True)

    trend_col, alert_col = st.columns(2, gap="large")
    with trend_col:
        st.subheader("Recent pressure and flow trends")
        st.plotly_chart(
            _plot_sensor_trends(
                _current_results()[: result.step + 1],
                ("PRESSURE_JUNCTION",),
                title="Junction pressure trend (kPa)",
            ),
            width="stretch",
            config={"displaylogo": False},
            key="overview_pressure_trend",
        )
        st.plotly_chart(
            _plot_sensor_trends(
                _current_results()[: result.step + 1],
                ("FLOW_MAIN",),
                title="Main flow trend (L/min)",
            ),
            width="stretch",
            config={"displaylogo": False},
            key="overview_flow_trend",
        )
    with alert_col:
        st.subheader("Recent synthetic alerts")
        recent_alerts = _event_rows(_current_results(), result.step)[-5:]
        if recent_alerts:
            st.dataframe(recent_alerts, width="stretch", hide_index=True)
        else:
            st.info("No synthetic localization events at this step.")

    score_figure = _localization_score_figure(result)
    if score_figure is None:
        st.info("The estimator has no segment scores for this frame.")
    else:
        st.plotly_chart(
            score_figure,
            width="stretch",
            config={"displaylogo": False},
            key="overview_segment_scores",
        )
    st.info(
        "Localization is an experimental synthetic heuristic. It is not confirmed leak detection or field-validated localization."
    )


def _render_pipeline_page() -> None:
    result = _current_result()
    _title(
        "Topology explorer",
        "3D pipeline monitoring",
        "The topology and sensor-to-segment links come from the synthetic network definitions. Coordinates are illustrative only.",
    )
    left, right = st.columns([1.65, 1], gap="large")
    with left:
        st.plotly_chart(
            _pipeline_figure(result, st.session_state.selected_network_entity),
            width="stretch",
            config={"displaylogo": False, "scrollZoom": True},
            key="pipeline_3d",
        )
        st.caption(
            "Rotate, zoom, and pan with Plotly. Select a sensor or segment with the inspector control; hover shows IDs and current values."
        )
        if result.localization_status == "localized":
            st.error(f"Estimated segment: {result.estimated_segment_id}")
        elif result.localization_status == "ambiguous":
            st.warning("Ambiguous localization; no segment is highlighted as the estimate.")
        elif result.localization_status == "insufficient_evidence":
            st.info("Insufficient evidence; no segment is highlighted.")
        else:
            st.success("No leak pattern in the current synthetic frame.")
    with right:
        st.subheader("Network inspector")
        _render_network_inspector(result)
        st.subheader("Localization evidence")
        if result.evidence:
            for line in result.evidence:
                st.write(f"- {line}")
        else:
            st.caption("No sensor evidence was returned for this estimate.")
        st.caption(
            "Diagnostic scores and margins are not calibrated probabilities."
        )


def _render_analytics_page() -> None:
    results = _current_results()
    end_index = min(int(st.session_state.demo_step), len(results) - 1)
    visible = results[: end_index + 1]
    _title(
        "Signal analytics",
        "Analytics & charts",
        "Charts use sequential readings from the current synthetic scenario and show exact values on hover.",
    )
    if not visible:
        st.info("Advance the synthetic scenario to collect sensor history.")
        return
    if end_index > 0:
        start_step, selected_end = st.slider(
            "Simulation-step range",
            min_value=0,
            max_value=end_index,
            value=(0, end_index),
            key="analytics_step_range",
        )
        visible = visible[start_step : selected_end + 1]
    pressure_ids = [
        sensor.sensor_id
        for sensor in DEFAULT_NETWORK.sensors
        if sensor.kind == "pressure"
    ]
    flow_ids = [
        sensor.sensor_id
        for sensor in DEFAULT_NETWORK.sensors
        if sensor.kind == "flow"
    ]
    current_entity = st.session_state.selected_network_entity
    focused_segment = (
        current_entity.split(":", maxsplit=1)[1]
        if current_entity.startswith("segment:")
        else DEFAULT_NETWORK.segment_ids[0]
    )
    focused_pipe = _SEGMENT_BY_ID[focused_segment]
    segment_pressure_defaults = [
        sensor.sensor_id
        for sensor in DEFAULT_NETWORK.sensors
        if sensor.kind == "pressure"
        and sensor.node_id in (focused_pipe.start_node, focused_pipe.end_node)
    ]
    segment_flow_defaults = [
        sensor.sensor_id
        for sensor in DEFAULT_NETWORK.sensors
        if sensor.segment_id == focused_segment
    ]
    selected_focus = st.selectbox(
        "Synchronize analytics with segment",
        DEFAULT_NETWORK.segment_ids,
        index=DEFAULT_NETWORK.segment_ids.index(focused_segment),
        key="analytics_segment_focus",
    )
    st.session_state.selected_network_entity = f"segment:{selected_focus}"
    pressure_selection = st.multiselect(
        "Pressure sensors",
        pressure_ids,
        default=segment_pressure_defaults,
        key="analytics_pressure_sensors",
    )
    st.plotly_chart(
        _plot_sensor_trends(visible, pressure_selection, title="Pressure trend (kPa)"),
        width="stretch",
        config={"displaylogo": False},
        key="pressure_trend",
    )
    flow_selection = st.multiselect(
        "Flow sensors",
        flow_ids,
        default=segment_flow_defaults,
        key="analytics_flow_sensors",
    )
    st.plotly_chart(
        _plot_sensor_trends(visible, flow_selection, title="Flow trend (L/min)"),
        width="stretch",
        config={"displaylogo": False},
        key="flow_trend",
    )

    all_sensors = list(DEFAULT_NETWORK.sensor_ids)
    comparison_selection = st.multiselect(
        "Multi-sensor comparison",
        all_sensors,
        default=["PRESSURE_JUNCTION", "FLOW_MAIN", "FLOW_NORTH_FEED"],
        key="analytics_comparison_sensors",
    )
    st.plotly_chart(
        _plot_sensor_trends(visible, comparison_selection, title="Selected sensor comparison"),
        width="stretch",
        config={"displaylogo": False},
        key="sensor_comparison",
    )

    left, right = st.columns(2, gap="large")
    with left:
        st.subheader("Anomaly / localization timeline")
        timeline = []
        for item in visible:
            if item.localization_status == "no_leak_pattern":
                continue
            timeline.append(
                {
                    "Step": item.step,
                    "Result": item.localization_status,
                    "Estimated segment": item.estimated_segment_id or "None",
                }
            )
        if timeline:
            st.dataframe(timeline, width="stretch", hide_index=True)
        else:
            st.info("No synthetic event result has been emitted in the visible steps.")
    with right:
        st.subheader("Localization diagnostic scores")
        score_figure = _localization_score_figure(visible[-1])
        if score_figure is None:
            st.info("No segment scores are available for the current step.")
        else:
            st.plotly_chart(
                score_figure,
                width="stretch",
                config={"displaylogo": False},
                key="analytics_localization_scores",
            )
            st.caption(
                f"Status: {_status_text(visible[-1])}. Values are diagnostic scores, not probabilities."
            )


def _event_rows(
    results: Sequence[LocalizationStepResult],
    through_step: int,
) -> list[dict[str, Any]]:
    rows = []
    for result in results[: through_step + 1]:
        if result.localization_status == "no_leak_pattern":
            continue
        evidence_ids = [
            line.split(":", maxsplit=1)[0] for line in result.evidence
        ]
        rows.append(
            {
                "Alert ID": f"SIM-{result.scenario_id}-{result.step:03d}",
                "Step": result.step,
                "Scenario": result.scenario_id,
                "Result": result.localization_status,
                "Estimated segment": result.estimated_segment_id or "None",
                "Evidence sensors": ", ".join(evidence_ids),
                "Severity": (
                    "Localization estimate"
                    if result.localization_status == "localized"
                    else "Review required"
                    if result.localization_status == "ambiguous"
                    else "Insufficient evidence"
                ),
            }
        )
    return rows


def _render_alerts_page() -> None:
    result = _current_result()
    results = _current_results()
    rows = _event_rows(results, result.step)
    _title(
        "Event review",
        "Alerts & investigation",
        "Rows are generated only from synthetic estimator outputs. No real leak alerts or operator actions are represented.",
    )
    if not rows:
        st.info("No synthetic localization events to investigate at the current simulation step.")
        return

    first, second, third, fourth = st.columns(4)
    result_filter = first.multiselect(
        "Result status",
        sorted({str(row["Result"]) for row in rows}),
        default=sorted({str(row["Result"]) for row in rows}),
        key="alert_result_filter",
    )
    severity_filter = second.multiselect(
        "Severity / review class",
        sorted({str(row["Severity"]) for row in rows}),
        default=sorted({str(row["Severity"]) for row in rows}),
        key="alert_severity_filter",
    )
    segment_filter = third.multiselect(
        "Estimated segment",
        sorted({str(row["Estimated segment"]) for row in rows}),
        default=sorted({str(row["Estimated segment"]) for row in rows}),
        key="alert_segment_filter",
    )
    sensor_filter = fourth.multiselect(
        "Evidence sensor",
        list(DEFAULT_NETWORK.sensor_ids),
        key="alert_sensor_filter",
    )
    step_range = st.slider(
        "Simulation-step range",
        min_value=0,
        max_value=result.step,
        value=(0, result.step),
        key="alert_step_range",
    ) if result.step > 0 else (0, 0)
    filtered = [
        row
        for row in rows
        if row["Result"] in result_filter
        and row["Severity"] in severity_filter
        and row["Estimated segment"] in segment_filter
        and step_range[0] <= int(row["Step"]) <= step_range[1]
        and (
            not sensor_filter
            or any(sensor in row["Evidence sensors"] for sensor in sensor_filter)
        )
    ]
    st.dataframe(filtered, width="stretch", hide_index=True)
    if not filtered:
        st.info("No events match the selected filters.")
        return

    alert_ids = [str(row["Alert ID"]) for row in filtered]
    selected_alert = st.selectbox(
        "Select an event to inspect",
        alert_ids,
        key="selected_synthetic_alert",
    )
    row = next(item for item in filtered if item["Alert ID"] == selected_alert)
    selected_step = int(row["Step"])
    selected_result = results[selected_step]
    if selected_result.estimated_segment_id:
        st.session_state.selected_network_entity = (
            f"segment:{selected_result.estimated_segment_id}"
        )
    st.subheader(f"Investigation · {selected_alert}")
    st.write(
        f"**Step:** {selected_step} · **Result:** "
        f"{selected_result.localization_status} · **Estimated segment:** "
        f"{selected_result.estimated_segment_id or 'None'}"
    )
    workflow_key = f"workflow:{selected_alert}"
    workflow_options = ("New", "Reviewing", "Closed for demo")
    current_workflow = st.session_state.alert_workflow.get(
        workflow_key,
        workflow_options[0],
    )
    if current_workflow not in workflow_options:
        current_workflow = workflow_options[0]
    st.selectbox(
        "Demo workflow status (session only)",
        workflow_options,
        index=workflow_options.index(current_workflow),
        key=f"workflow_widget_{selected_alert}",
        on_change=lambda: st.session_state.alert_workflow.update(
            {
                workflow_key: st.session_state[f"workflow_widget_{selected_alert}"]
            }
        ),
    )
    left, right = st.columns(2, gap="large")
    with left:
        st.subheader("Sensor evidence")
        if selected_result.evidence:
            for evidence in selected_result.evidence:
                st.write(f"- {evidence}")
        else:
            st.caption("No evidence returned.")
        if selected_result.estimated_segment_id is None:
            st.warning(
                "The estimator did not identify a segment. Do not infer a leak location."
            )
    with right:
        st.subheader("Pressure and flow context")
        related_sensors: list[str] = []
        segment_id = selected_result.estimated_segment_id
        if segment_id is not None:
            segment = _SEGMENT_BY_ID[segment_id]
            related_sensors = [
                sensor.sensor_id
                for sensor in DEFAULT_NETWORK.sensors
                if sensor.segment_id == segment_id
                or sensor.node_id in (segment.start_node, segment.end_node)
            ]
        if related_sensors:
            st.plotly_chart(
                _plot_sensor_trends(
                    results[: selected_step + 1],
                    related_sensors,
                    title=f"Related sensor context · {segment_id}",
                ),
                width="stretch",
                config={"displaylogo": False},
                key=f"investigation_context_{selected_alert}",
            )
        else:
            st.info("No segment-specific chart is available for this unlocalized event.")
    st.caption(
        "Suggested check: verify the sensor-to-segment mapping and inspect the synthetic readings. No field action is executed."
    )


def _advance_demo() -> None:
    results = _current_results()
    if st.session_state.demo_step < len(results) - 1:
        st.session_state.demo_step += 1
    if st.session_state.demo_step >= len(results) - 1:
        st.session_state.demo_running = False


def _render_demo_frame() -> None:
    if st.session_state.demo_running:
        _advance_demo()
    result = _current_result()
    st.markdown(
        f"**Step {result.step} / {len(_current_results()) - 1}** · "
        f"{_status_text(result)}"
    )
    st.progress((result.step + 1) / len(_current_results()))
    metrics = st.columns(4)
    metrics[0].metric("Simulation time", f"{result.time_seconds:.1f} s")
    metrics[1].metric("Scenario", result.scenario_id)
    metrics[2].metric("Estimated segment", result.estimated_segment_id or "None")
    metrics[3].metric("Localization status", result.localization_status)
    with st.expander("Ground truth · evaluation/demo only", expanded=False):
        if result.ground_truth.leak_active:
            st.write(
                f"Simulated leak segment: {result.ground_truth.leak_segment_id}; "
                f"configured onset step: {result.ground_truth.onset_step}."
            )
        else:
            st.write("This is a synthetic no-leak scenario.")
        st.caption("Ground truth is displayed separately and is not estimator input.")
    left, right = st.columns([1.15, 1], gap="large")
    with left:
        st.subheader("Current sensors")
        st.dataframe(
            [
                {
                    "Sensor": sensor.sensor_id,
                    "Type": sensor.kind,
                    "Reading": result.sensor_readings[sensor.sensor_id],
                    "Unit": sensor.unit,
                    "Reference": result.normal_reference[sensor.sensor_id],
                }
                for sensor in DEFAULT_NETWORK.sensors
            ],
            width="stretch",
            hide_index=True,
        )
    with right:
        st.subheader("Localization evidence")
        if result.evidence:
            for evidence in result.evidence:
                st.write(f"- {evidence}")
        else:
            st.info("No localizing evidence was returned at this step.")
        st.plotly_chart(
            _pipeline_figure(result, st.session_state.selected_network_entity),
            width="stretch",
            config={"displaylogo": False, "scrollZoom": True},
            key="demo_pipeline_3d",
        )


def _render_demo_page() -> None:
    _title(
        "Synthetic test bench",
        "Demo environment",
        "Deterministic simulated sensors only. This environment is separate from WDSEventDB replay and has no live hardware connection.",
    )
    options = ("normal", *DEFAULT_NETWORK.segment_ids)
    choice = st.selectbox(
        "Scenario",
        options,
        format_func=lambda value: (
            "Normal operation" if value == "normal" else f"Leak starts at {value}"
        ),
        key="demo_choice",
    )
    strength = st.slider(
        "Synthetic event strength",
        min_value=1.0,
        max_value=10.0,
        value=float(st.session_state.demo_strength),
        step=0.5,
        key="demo_strength",
        disabled=choice == "normal",
    )
    _ensure_demo_results(choice, strength)
    controls = st.columns([1, 1, 1, 1, 1.2], vertical_alignment="center")
    if controls[0].button(
        "Start",
        type="primary",
        disabled=st.session_state.demo_running
        or st.session_state.demo_step >= len(_current_results()) - 1,
        key="demo_start",
    ):
        st.session_state.demo_running = True
        st.rerun()
    if controls[1].button(
        "Pause",
        disabled=not st.session_state.demo_running,
        key="demo_pause",
    ):
        st.session_state.demo_running = False
        st.rerun()
    if controls[2].button("Reset", key="demo_reset"):
        st.session_state.demo_running = False
        st.session_state.demo_step = 0
        st.rerun()
    if controls[3].button(
        "Step forward",
        disabled=st.session_state.demo_running
        or st.session_state.demo_step >= len(_current_results()) - 1,
        key="demo_step_forward",
    ):
        _advance_demo()
        st.rerun()
    speed = controls[4].selectbox(
        "Simulation interval",
        (0.25, 0.5, 1.0, 2.0),
        format_func=lambda value: f"{value:g} seconds / step",
        key="demo_speed",
    )
    st.divider()
    fragment = st.fragment(
        run_every=speed if st.session_state.demo_running else None
    )(_render_demo_frame)
    fragment()


def _render_requirements_page() -> None:
    _title(
        "Company workspace",
        "Company configuration & requirements",
        "Company-provided records are separate from synthetic monitoring and public research data.",
    )
    try:
        groups = load_requirement_groups()
    except ValueError as error:
        st.error(f"Requirements data could not be loaded: {error}")
        groups = None
    if groups is None:
        st.info("Requirements data not available yet")
        st.caption(
            "When the verified company dataset is provided, place it at "
            "`data/company/requirements.json` using the groups/requirements schema. "
            "No requirement records, assignments, or completion percentages are inferred."
        )
    elif not groups:
        st.info("The requirements source contains no groups.")
    else:
        group_ids = [group.group_id for group in groups]
        group_id = st.selectbox(
            "Requirement group",
            group_ids,
            format_func=lambda selected: next(
                group.title for group in groups if group.group_id == selected
            ),
        )
        group = next(group for group in groups if group.group_id == group_id)
        st.subheader(group.title)
        if not group.requirements:
            st.info("This source group contains no requirement records.")
        for requirement in group.requirements:
            with st.expander(
                f"{requirement.requirement_id} · {requirement.title}"
            ):
                st.write(requirement.description)
        st.caption("Records loaded from data/company/requirements.json.")

    st.divider()
    st.subheader("Company data inspection")
    category = st.selectbox(
        "Upload category",
        ("Product information", "Processes", "Policies", "Internal documents"),
        key="company_upload_category",
    )
    files = st.file_uploader(
        "Select company files for in-session inspection",
        accept_multiple_files=True,
        key="company_uploads",
    )
    if files:
        st.caption(
            f"{category} · uploaded content remains in this Streamlit session; SPS does not process or persist it."
        )
        st.dataframe(
            [
                {
                    "File": file.name,
                    "Type": file.type or "Unknown",
                    "Size (bytes)": file.size,
                    "Inspection": "Metadata only; no processing performed",
                }
                for file in files
            ],
            width="stretch",
            hide_index=True,
        )
    else:
        st.info("No company files uploaded in this session.")


def _start_wdseventdb_replay(*, restart: bool = False) -> None:
    stream = st.session_state.wdse_stream
    if stream is not None and stream.state is StreamState.RUNNING:
        return
    if restart:
        st.session_state.wdse_history = DashboardHistory(max_history=800)
        st.session_state.wdse_queue = Queue()
        st.session_state.wdse_error = None
    try:
        stream = SimulatedSensorStream.from_wdseventdb(
            inference=_get_inference(),
            batch_size=25,
            interval_seconds=0.5,
        )
        stream.start(st.session_state.wdse_queue.put)
        st.session_state.wdse_stream = stream
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        st.session_state.wdse_error = str(error)


def _stream_status(stream: SimulatedSensorStream | None) -> str:
    if stream is None:
        return "Idle"
    return {
        StreamState.READY: "Ready",
        StreamState.RUNNING: "Running",
        StreamState.STOPPED: "Stopped",
        StreamState.COMPLETED: "Completed · end of dataset",
        StreamState.FAILED: "Failed",
    }[stream.state]


def _render_wdseventdb_monitor() -> None:
    history: DashboardHistory = st.session_state.wdse_history
    history.drain(st.session_state.wdse_queue)
    stream: SimulatedSensorStream | None = st.session_state.wdse_stream
    if stream is not None and stream.state is StreamState.FAILED:
        try:
            stream.wait(timeout=0)
        except (RuntimeError, TimeoutError) as error:
            cause = error.__cause__
            st.session_state.wdse_error = str(cause or error)
    metrics = st.columns(4)
    metrics[0].metric("Rows received", history.total_readings)
    metrics[1].metric("Normal predictions", history.normal_predictions)
    metrics[2].metric("Anomalous-event predictions", history.anomalous_predictions)
    metrics[3].metric("Batches received", history.processed_batches)
    st.write(f"Replay status: **{_stream_status(stream)}**")
    if st.session_state.wdse_error:
        st.error(f"WDSEventDB replay error: {st.session_state.wdse_error}")
    if not history.readings:
        st.info("Start the replay to inspect recorded WDSEventDB rows.")
        return

    inference = _get_inference()
    feature_names = tuple(inference.feature_names)
    if st.session_state.wdse_feature not in feature_names:
        st.session_state.wdse_feature = feature_names[0]
    selected_feature = st.selectbox(
        "Sensor feature",
        feature_names,
        key="wdse_feature",
    )
    recent = list(history.readings)[-300:]
    chart = go.Figure(
        go.Scatter(
            x=list(range(len(recent))),
            y=[float(reading.sensor_readings[selected_feature]) for reading in recent],
            mode="lines",
            name=selected_feature,
            customdata=[
                [reading.source_file, reading.source_row_index, reading.prediction_status]
                for reading in recent
            ],
            hovertemplate=(
                "Replay sequence %{x}<br>Value %{y:.5f}"
                "<br>Source %{customdata[0]} row %{customdata[1]}"
                "<br>%{customdata[2]}<extra></extra>"
            ),
        )
    )
    chart.update_layout(
        title=f"WDSEventDB replay · {selected_feature}",
        height=330,
        paper_bgcolor=_COLORS["background"],
        plot_bgcolor=_COLORS["surface"],
        font={"color": _COLORS["text"]},
        xaxis_title="Replay sequence",
        yaxis_title=selected_feature,
    )
    st.plotly_chart(
        chart,
        width="stretch",
        config={"displaylogo": False},
        key="wdse_sensor_trend",
    )
    st.dataframe(
        [
            {
                "Source": reading.source_file,
                "Row": reading.source_row_index,
                "Original Labels metadata": reading.original_label,
                "Model output": reading.prediction_status,
                selected_feature: reading.sensor_readings[selected_feature],
            }
            for reading in reversed(recent[-25:])
        ],
        width="stretch",
        hide_index=True,
    )


def _render_wdseventdb_page() -> None:
    _title(
        "Recorded-data model replay",
        "WDSEventDB replay",
        "Uses the existing saved SPSInference model and Clean/Leakage workbook rows. This recorded-data replay is separate from synthetic localization.",
    )
    st.warning(
        "The model predicts normal versus anomalous source Labels. It does not confirm leaks or localize segments."
    )
    stream: SimulatedSensorStream | None = st.session_state.wdse_stream
    status = stream.state if stream is not None else None
    controls = st.columns(4)
    if controls[0].button(
        "Start replay",
        type="primary",
        disabled=status in (StreamState.RUNNING, StreamState.COMPLETED),
        key="wdse_start",
    ):
        _start_wdseventdb_replay()
        st.rerun()
    if controls[1].button(
        "Stop",
        disabled=status is not StreamState.RUNNING,
        key="wdse_stop",
    ):
        try:
            stream.stop(timeout=5)
        except (RuntimeError, TimeoutError) as error:
            st.session_state.wdse_error = str(error)
        st.rerun()
    if controls[2].button(
        "Replay from start",
        disabled=status not in (StreamState.STOPPED, StreamState.COMPLETED, StreamState.FAILED),
        key="wdse_restart",
    ):
        _start_wdseventdb_replay(restart=True)
        st.rerun()
    controls[3].caption("Saved artifact is read-only.")
    fragment = st.fragment(
        run_every=0.75 if status is StreamState.RUNNING else None
    )(_render_wdseventdb_monitor)
    fragment()


def _render_asset_details(result: LocalizationStepResult) -> None:
    options = _entity_options()
    if st.session_state.selected_network_entity not in options:
        st.session_state.selected_network_entity = options[0]
    selected = st.selectbox(
        "Asset",
        options,
        format_func=_entity_label,
        key="selected_network_entity",
        label_visibility="collapsed",
    )
    kind, identifier = selected.split(":", maxsplit=1)
    if kind == "segment":
        segment = _SEGMENT_BY_ID[identifier]
        linked_sensors = [
            sensor
            for sensor in DEFAULT_NETWORK.sensors
            if sensor.segment_id == identifier
            or sensor.node_id in (segment.start_node, segment.end_node)
        ]
        condition = (
            "Review"
            if result.estimated_segment_id == identifier
            else "Normal"
            if result.localization_status == "no_leak_pattern"
            else "Monitoring"
        )
        st.subheader(segment.name)
        st.caption(f"Section {segment.segment_id} · {segment.start_node} to {segment.end_node}")
        st.metric("Condition", condition)
        st.markdown("**Connected sensors**")
        for sensor in linked_sensors:
            reading = result.sensor_readings[sensor.sensor_id]
            st.write(
                f"{sensor.sensor_id} · {reading:.2f} {sensor.unit}"
            )
    else:
        sensor = _SENSOR_BY_ID[identifier]
        reading = result.sensor_readings[identifier]
        reference = result.normal_reference[identifier]
        has_signal = any(identifier in item for item in result.evidence)
        st.subheader(sensor.sensor_id.partition("_")[2].replace("_", " ").title())
        st.caption(f"{sensor.kind.title()} sensor · {sensor.unit}")
        st.metric("Current reading", _format_reading(reading, sensor.unit))
        st.metric(
            "Condition",
            "Review" if has_signal else "Online",
        )
        st.caption(f"Observed change {reading - reference:+.2f} {sensor.unit}")


_WATER_LEVEL_COLORS = {
    WaterLevelCondition.NORMAL: _COLORS["green"],
    WaterLevelCondition.POTENTIAL_LEAK: _COLORS["red"],
    WaterLevelCondition.BOUNDARY: _COLORS["amber"],
    WaterLevelCondition.UNAVAILABLE: _COLORS["muted"],
    WaterLevelCondition.INVALID: _COLORS["amber"],
    WaterLevelCondition.OUT_OF_RANGE: _COLORS["amber"],
}


def _water_level_events(results: Sequence[LocalizationStepResult]):
    """One record per continuous water-level event (no duplicate alerts)."""
    return summarize_water_level_events(
        (item.step, item.water_level_assessment) for item in results
    )


def _water_level_figure(results: Sequence[LocalizationStepResult]) -> go.Figure:
    config = DEFAULT_WATER_LEVEL_CONFIG
    points = [item for item in results if item.water_level_reading is not None]
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=[item.time_seconds for item in points],
            y=[item.water_level_reading for item in points],
            mode="lines+markers",
            line={"color": _COLORS["blue"], "width": 2},
            marker={
                "size": 5,
                "color": [
                    _WATER_LEVEL_COLORS[item.water_level_assessment.condition]
                    for item in points
                ],
            },
            customdata=[
                WATER_LEVEL_STATUS_TEXT[item.water_level_assessment.condition]
                for item in points
            ],
            name="Water level",
            hovertemplate=(
                "Time %{x:.0f} s<br>Level %{y:.1f} "
                f"{config.unit}<br>%{{customdata}}<extra></extra>"
            ),
        )
    )
    figure.add_hline(
        y=config.threshold,
        line={"color": _COLORS["amber"], "dash": "dash", "width": 1},
        annotation_text=f"Threshold {config.threshold:g} · below = potential leak indication",
        annotation_position="bottom right",
        annotation_font={"color": _COLORS["muted"], "size": 11},
    )
    figure.update_layout(
        title="Water level history (simulated)",
        height=330,
        margin={"l": 10, "r": 15, "t": 55, "b": 10},
        paper_bgcolor=_COLORS["background"],
        plot_bgcolor=_COLORS["surface"],
        font={"color": _COLORS["text"]},
        xaxis={"title": "Time (s)", "gridcolor": "#263944"},
        yaxis={"title": f"Level ({config.unit})", "gridcolor": "#263944"},
        showlegend=False,
        hovermode="x unified",
    )
    return figure


def _render_water_level_section(
    result: LocalizationStepResult,
    visible: Sequence[LocalizationStepResult],
) -> None:
    config = DEFAULT_WATER_LEVEL_CONFIG
    assessment = result.water_level_assessment
    status_text = WATER_LEVEL_STATUS_TEXT[assessment.condition]
    color = _WATER_LEVEL_COLORS[assessment.condition]
    st.subheader("Water level monitoring")
    st.caption(
        f"Separate threshold check on sensor {config.sensor_id}: above {config.threshold:g} is "
        f"normal, below {config.threshold:g} is a potential leak indication, exactly "
        f"{config.threshold:g} is a boundary reading. Not a model input and not used for "
        "segment localization. Values are simulated; no physical water-level sensor is connected."
    )
    chart_col, status_col = st.columns([1.6, 1], gap="large")
    with chart_col:
        st.plotly_chart(
            _water_level_figure(visible),
            width="stretch",
            config={"displaylogo": False},
            key="product_water_level_history",
        )
    with status_col:
        st.metric(
            f"Current level · {config.unit}",
            "Unavailable"
            if result.water_level_reading is None
            else f"{result.water_level_reading:.1f}",
        )
        st.markdown(
            f'<div class="sps-card"><span style="color:{color};font-weight:700">● '
            f"{status_text}</span></div>",
            unsafe_allow_html=True,
        )
        events = _water_level_events(visible)
        if events:
            st.markdown("**Water-level events**")
            st.dataframe(
                [
                    {
                        "Event": f"WL-{event.event_number:02d}",
                        "Started": f"{visible[event.start_step].time_seconds:05.1f} s",
                        "Status": "Active" if event.is_active else "Cleared",
                        "Lowest": f"{event.lowest_reading:.1f}",
                        "Readings below": event.readings_below_threshold,
                    }
                    for event in reversed(events)
                ],
                width="stretch",
                hide_index=True,
            )
        else:
            st.caption("No water-level events in the visible simulation steps.")


def _render_product_dashboard_controls() -> None:
    title_col, badge_col = st.columns([5, 1], vertical_alignment="center")
    with title_col:
        st.markdown('<div class="sps-eyebrow">SPS · NETWORK OPERATIONS</div>', unsafe_allow_html=True)
        st.title("Pipeline Operations")
    with badge_col:
        st.markdown(
            '<div class="sps-pill"><span class="sps-dot">●</span> SIMULATION</div>',
            unsafe_allow_html=True,
        )

    controls = st.columns([1.7, 1.0, 5.3], vertical_alignment="center")
    if st.session_state.demo_running:
        if controls[0].button("Pause", key="simulation_pause", use_container_width=True):
            st.session_state.demo_running = False
            st.rerun()
        if controls[1].button("Reset", key="simulation_reset_running", use_container_width=True):
            st.session_state.demo_step = 0
            st.session_state.demo_running = False
            st.rerun()
    else:
        if controls[0].button(
            "Start Simulation",
            type="primary",
            key="simulation_start",
            use_container_width=True,
        ):
            if st.session_state.demo_step >= len(_current_results()) - 1:
                st.session_state.demo_step = 0
            st.session_state.demo_running = True
            st.rerun()
        if st.session_state.demo_step > 0:
            if controls[1].button(
                "Reset",
                key="simulation_reset_paused",
                use_container_width=True,
            ):
                st.session_state.demo_step = 0
                st.session_state.demo_running = False
                st.rerun()
    controls[2].markdown(
        f'<div class="sps-note">{"Running" if st.session_state.demo_running else "Paused"}</div>',
        unsafe_allow_html=True,
    )


def _render_product_dashboard_content() -> None:
    if st.session_state.demo_running:
        _advance_demo()
        if not st.session_state.demo_running:
            st.rerun()
    result = _current_result()
    results = _current_results()
    visible = results[: result.step + 1]

    pressure = result.sensor_readings["PRESSURE_JUNCTION"]
    flow = result.sensor_readings["FLOW_MAIN"]
    condition = (
        "Review"
        if result.localization_status in ("ambiguous", "insufficient_evidence")
        else "Attention"
        if result.localization_status == "localized"
        else "Normal"
    )
    metrics = st.columns(5)
    metrics[0].metric("Junction pressure · kPa", f"{pressure:.1f}")
    metrics[1].metric("Main line flow · L/min", f"{flow:.1f}")
    metrics[2].metric("Sensors online", f"{len(result.sensor_readings)} / {len(DEFAULT_NETWORK.sensors)}")
    metrics[3].metric("Network condition", condition)
    metrics[4].metric(
        "Water level",
        "Unavailable"
        if result.water_level_reading is None
        else f"{result.water_level_reading:.1f}",
        help=WATER_LEVEL_STATUS_TEXT[result.water_level_assessment.condition],
    )
    metrics[4].caption(
        WATER_LEVEL_STATUS_TEXT[result.water_level_assessment.condition]
    )

    network_col, details_col = st.columns([1.7, 0.85], gap="large")
    with network_col:
        st.subheader("Pipeline network")
        st.plotly_chart(
            _pipeline_figure(result, st.session_state.selected_network_entity),
            width="stretch",
            config={"displaylogo": False, "scrollZoom": True},
            key="operations_pipeline",
        )
    with details_col:
        st.subheader("Asset details")
        _render_asset_details(result)
        if result.estimated_segment_id:
            st.warning(f"Attention area · Section {result.estimated_segment_id}")
        elif result.localization_status == "ambiguous":
            st.info("Condition requires review")

    st.divider()
    pressure_ids = [
        sensor.sensor_id
        for sensor in DEFAULT_NETWORK.sensors
        if sensor.kind == "pressure"
    ]
    flow_ids = [
        sensor.sensor_id
        for sensor in DEFAULT_NETWORK.sensors
        if sensor.kind == "flow"
    ]
    left_trend, right_trend = st.columns(2, gap="large")
    with left_trend:
        pressure_sensor = st.selectbox(
            "Pressure sensor",
            pressure_ids,
            index=pressure_ids.index("PRESSURE_JUNCTION"),
            key="product_pressure_sensor",
        )
        st.plotly_chart(
            _plot_sensor_trends(
                visible,
                (pressure_sensor,),
                title="Pressure history",
            ),
            width="stretch",
            config={"displaylogo": False},
            key="product_pressure_history",
        )
    with right_trend:
        flow_sensor = st.selectbox(
            "Flow sensor",
            flow_ids,
            index=flow_ids.index("FLOW_MAIN"),
            key="product_flow_sensor",
        )
        st.plotly_chart(
            _plot_sensor_trends(
                visible,
                (flow_sensor,),
                title="Flow history",
            ),
            width="stretch",
            config={"displaylogo": False},
            key="product_flow_history",
        )

    _render_water_level_section(result, visible)

    status_col, alerts_col = st.columns([1.1, 1], gap="large")
    with status_col:
        st.subheader("Sensor status")
        signal_sensors = {
            line.partition(":")[0]
            for line in result.evidence
        }
        st.dataframe(
            [
                {
                    "Sensor": sensor.sensor_id,
                    "Type": sensor.kind.title(),
                    "Reading": f"{result.sensor_readings[sensor.sensor_id]:.2f} {sensor.unit}",
                    "Status": "Review" if sensor.sensor_id in signal_sensors else "Online",
                    "Area": sensor.segment_id or sensor.node_id or "—",
                }
                for sensor in DEFAULT_NETWORK.sensors
            ],
            width="stretch",
            hide_index=True,
        )
    with alerts_col:
        st.subheader("Recent alerts")
        alert_rows = []
        for item in _current_results()[: result.step + 1]:
            if item.localization_status == "no_leak_pattern":
                continue
            alert_rows.append(
                {
                    "Time": f"{item.time_seconds:05.1f} s",
                    "Condition": "Network condition requires review",
                    "Area": item.estimated_segment_id or "Unresolved",
                    "Status": "Review",
                }
            )
        # One row per continuous water-level event, never one per reading.
        # Listed ahead of the latest localization rows so an open event stays visible.
        water_rows = [
            {
                "Time": f"{visible[event.start_step].time_seconds:05.1f} s",
                "Condition": "Water level below threshold · potential leak indication",
                "Area": "Water level (separate from segments)",
                "Status": "Active" if event.is_active else "Cleared",
            }
            for event in reversed(_water_level_events(visible))
        ]
        if alert_rows or water_rows:
            st.dataframe(
                water_rows + list(reversed(alert_rows[-6:])),
                width="stretch",
                hide_index=True,
            )
        else:
            st.markdown(
                '<div class="sps-card">No issues requiring attention.</div>',
                unsafe_allow_html=True,
            )


def _render_page(page: str) -> None:
    if page == "Overview":
        _render_overview()
    elif page == "3D Pipeline Monitoring":
        _render_pipeline_page()
    elif page == "Analytics & Charts":
        _render_analytics_page()
    elif page == "Alerts & Investigation":
        _render_alerts_page()
    elif page == "Demo Environment":
        _render_demo_page()
    elif page == "Company Configuration / Requirements":
        _render_requirements_page()
    elif page == "WDSEventDB Replay":
        _render_wdseventdb_page()
    else:
        raise ValueError(f"Unknown dashboard page: {page}")


_initialize_session()
_ensure_demo_results(
    st.session_state.demo_choice,
    st.session_state.demo_strength,
)
_render_product_dashboard_controls()
dashboard_content = st.fragment(
    run_every=1.0 if st.session_state.demo_running else None
)(_render_product_dashboard_content)
dashboard_content()
