from __future__ import annotations

import plotly.graph_objects as go

from sps.demo_scenarios import DemoSnapshot, NODES, SECTIONS, section_status


_STATUS_COLORS = {
    "normal": "#82a99c",
    "attention": "#d2a653",
    "critical": "#d87971",
}
_SECTION_LABEL_POSITIONS = {
    "S-01": (-3.05, 0.42),
    "S-02": (-0.55, 1.12),
    "S-03": (2.0, 2.42),
    "S-04": (-0.55, -1.12),
    "S-05": (2.0, -2.42),
    "S-06": (1.75, -0.4),
}
_NODE_LABEL_POSITIONS = {
    "N-01": "bottom center",
    "N-02": "bottom center",
    "N-03": "middle left",
    "N-04": "middle right",
    "N-05": "middle left",
    "N-06": "middle right",
    "N-07": "bottom center",
}


def _layout(figure: go.Figure, *, height: int) -> go.Figure:
    figure.update_layout(
        height=height,
        margin={"l": 8, "r": 16, "t": 8, "b": 8},
        paper_bgcolor="#101316",
        plot_bgcolor="#101316",
        font={"family": "Segoe UI, sans-serif", "color": "#c5cdd2", "size": 11},
        hoverlabel={
            "bgcolor": "#20272c",
            "bordercolor": "#46525a",
            "font": {"color": "#edf0f2", "family": "Segoe UI, sans-serif"},
        },
        showlegend=False,
    )
    return figure


def make_network_figure(
    snapshot: DemoSnapshot, selected_section_id: str | None = None
) -> go.Figure:
    """Draw the scenario's expressly illustrative 2D topology and section states."""
    figure = go.Figure()
    arrows: list[dict[str, object]] = []
    for section in snapshot.scenario.sections:
        status = section_status(snapshot, section.section_id)
        color = _STATUS_COLORS[status]
        selected = section.section_id == selected_section_id
        route_x = [point[0] for point in section.route]
        route_y = [point[1] for point in section.route]
        figure.add_trace(
            go.Scatter(
                x=route_x,
                y=route_y,
                mode="lines",
                line={
                    "color": "#0b0e11",
                    "width": 15 if selected else 12,
                    "shape": "spline",
                    "smoothing": 0.1,
                },
                hoverinfo="skip",
                showlegend=False,
            )
        )
        figure.add_trace(
            go.Scatter(
                x=route_x,
                y=route_y,
                mode="lines",
                line={
                    "color": "#edf0f2" if selected else color,
                    "width": 7 if selected else 5,
                    "shape": "spline",
                    "smoothing": 0.1,
                },
                customdata=[[section.section_id, section.name, status]] * len(route_x),
                hovertemplate=(
                    "<b>%{customdata[0]} · %{customdata[1]}</b>"
                    "<br>Status: %{customdata[2]}<extra></extra>"
                ),
                showlegend=False,
            )
        )
        arrow_start = section.route[-2]
        arrow_end = section.route[-1]
        arrows.append(
            {
                "x": arrow_start[0] + (arrow_end[0] - arrow_start[0]) * 0.68,
                "y": arrow_start[1] + (arrow_end[1] - arrow_start[1]) * 0.68,
                "ax": arrow_start[0] + (arrow_end[0] - arrow_start[0]) * 0.54,
                "ay": arrow_start[1] + (arrow_end[1] - arrow_start[1]) * 0.54,
                "xref": "x",
                "yref": "y",
                "axref": "x",
                "ayref": "y",
                "showarrow": True,
                "arrowhead": 2,
                "arrowsize": 1.1,
                "arrowwidth": 1.8,
                "arrowcolor": "#aebac1",
                "text": "",
            }
        )
        middle = section.route[len(section.route) // 2]
        figure.add_trace(
            go.Scatter(
                x=[middle[0]],
                y=[middle[1]],
                mode="markers",
                marker={
                    "size": 14 if status == "critical" else 11 if selected else 7,
                    "color": "#d2a653" if selected else color,
                    "line": {
                        "color": "#f0e2c8" if selected else "#303a42",
                        "width": 2 if selected else 1,
                    },
                },
                customdata=[[section.section_id, section.name, status]],
                hovertemplate=(
                    "<b>%{customdata[0]} · %{customdata[1]}</b>"
                    "<br>Status: %{customdata[2]}<extra></extra>"
                ),
                showlegend=False,
            )
        )
        label_x, label_y = _SECTION_LABEL_POSITIONS[section.section_id]
        figure.add_trace(
            go.Scatter(
                x=[label_x],
                y=[label_y],
                mode="text",
                text=[section.section_id],
                textfont={
                    "size": 10,
                    "color": "#edcc8f" if selected else "#b4bec4",
                },
                hoverinfo="skip",
                showlegend=False,
            )
        )

    node_status: dict[str, str] = {node.node_id: "normal" for node in NODES}
    for section in snapshot.scenario.sections:
        status = section_status(snapshot, section.section_id)
        if status != "normal":
            node_status[section.start_node] = status
            node_status[section.end_node] = status
    for node in NODES:
        color = _STATUS_COLORS[node_status[node.node_id]]
        figure.add_trace(
            go.Scatter(
                x=[node.x],
                y=[node.y],
                mode="markers+text",
                text=[node.node_id],
                textposition=_NODE_LABEL_POSITIONS[node.node_id],
                textfont={"size": 10, "color": "#d6dde1"},
                marker={
                    "size": 13,
                    "color": "#1a2024",
                    "line": {"color": color, "width": 2.5},
                },
                customdata=[[node.node_id, node.name, node_status[node.node_id]]],
                hovertemplate=(
                    "<b>%{customdata[0]} · %{customdata[1]}</b>"
                    "<br>Illustrative status: %{customdata[2]}<extra></extra>"
                ),
                showlegend=False,
            )
        )

    figure.update_layout(
        xaxis={
            "visible": False,
            "range": [-5.4, 6.0],
            "fixedrange": True,
            "zeroline": False,
        },
        yaxis={
            "visible": False,
            "range": [-3.1, 3.1],
            "fixedrange": True,
            "zeroline": False,
            "scaleanchor": "x",
            "scaleratio": 1,
        },
        annotations=arrows,
        uirevision=f"sps-demo-network-{snapshot.scenario.scenario_id}",
    )
    return _layout(figure, height=350)


def make_section_trend_figure(
    snapshot: DemoSnapshot,
    section_id: str,
    *,
    height: int = 225,
) -> go.Figure:
    telemetry = snapshot.telemetry_for(section_id)
    steps = list(range(1, len(telemetry.pressure_kpa) + 1))
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=steps,
            y=telemetry.pressure_kpa,
            name="Pressure",
            mode="lines",
            line={"color": "#94afbf", "width": 2.5},
            hovertemplate="Sequence %{x}<br>%{y:.1f} kPa<extra>Pressure</extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=steps,
            y=[telemetry.expected_pressure_kpa] * len(steps),
            name="Pressure reference",
            mode="lines",
            line={"color": "#74818a", "width": 1, "dash": "dot"},
            hovertemplate="Illustrative reference<br>%{y:.1f} kPa<extra></extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=steps,
            y=telemetry.flow_lpm,
            name="Flow",
            mode="lines",
            yaxis="y2",
            line={"color": "#789da2", "width": 2.5},
            hovertemplate="Sequence %{x}<br>%{y:.1f} L/min<extra>Flow</extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=steps,
            y=[telemetry.expected_flow_lpm] * len(steps),
            name="Flow reference",
            mode="lines",
            yaxis="y2",
            line={"color": "#74818a", "width": 1, "dash": "dot"},
            hovertemplate="Illustrative reference<br>%{y:.1f} L/min<extra></extra>",
        )
    )
    figure.update_layout(
        xaxis={
            "title": {"text": "Illustrative sequence step"},
            "tickmode": "array",
            "tickvals": [1, 4, 8, 12],
            "ticktext": ["01", "04", "08", "12"],
            "gridcolor": "#2a3339",
            "linecolor": "#3a454d",
        },
        yaxis={
            "title": {"text": "Pressure · kPa", "font": {"color": "#94afbf"}},
            "gridcolor": "#2a3339",
            "linecolor": "#3a454d",
            "tickfont": {"color": "#94afbf"},
        },
        yaxis2={
            "title": {"text": "Flow · L/min", "font": {"color": "#789da2"}},
            "overlaying": "y",
            "side": "right",
            "showgrid": False,
            "linecolor": "#3a454d",
            "tickfont": {"color": "#789da2"},
        },
        legend={
            "orientation": "h",
            "y": 1.12,
            "x": 0,
            "font": {"size": 10},
        },
        uirevision=f"sps-demo-trend-{snapshot.scenario.scenario_id}-{section_id}",
    )
    return _layout(figure, height=height)


def make_readings_figure(
    snapshot: DemoSnapshot,
    section_ids: tuple[str, ...],
) -> go.Figure:
    latest = [snapshot.telemetry_for(section_id) for section_id in section_ids]
    figure = go.Figure()
    figure.add_trace(
        go.Bar(
            x=[item.section_id for item in latest],
            y=[item.latest_pressure_kpa for item in latest],
            name="Pressure · kPa",
            marker_color="#94afbf",
            hovertemplate="%{x}<br>%{y:.1f} kPa<extra>Pressure</extra>",
        )
    )
    figure.add_trace(
        go.Bar(
            x=[item.section_id for item in latest],
            y=[item.latest_flow_lpm for item in latest],
            name="Flow · L/min",
            marker_color="#789da2",
            yaxis="y2",
            hovertemplate="%{x}<br>%{y:.1f} L/min<extra>Flow</extra>",
        )
    )
    figure.update_layout(
        barmode="group",
        xaxis={"title": {"text": "Illustrative pipeline section"}, "gridcolor": "#2a3339"},
        yaxis={"title": {"text": "Pressure · kPa"}, "gridcolor": "#2a3339"},
        yaxis2={
            "title": {"text": "Flow · L/min"},
            "overlaying": "y",
            "side": "right",
            "showgrid": False,
        },
        legend={"orientation": "h", "y": 1.12, "x": 0},
    )
    return _layout(figure, height=300)
