from __future__ import annotations

import math
from dataclasses import dataclass

import plotly.graph_objects as go


@dataclass(frozen=True)
class IllustrativeNode:
    node_id: str
    name: str
    position: tuple[float, float, float]
    description: str


NODES = (
    IllustrativeNode("N-01", "Supply header", (-4.0, 0.0, 0.0), "Illustrative network inlet."),
    IllustrativeNode("N-02", "Main junction", (-1.5, 0.0, 0.0), "Illustrative distribution junction."),
    IllustrativeNode("N-03", "North branch", (0.0, 2.0, 0.0), "Illustrative branch endpoint."),
    IllustrativeNode("N-04", "North riser", (2.5, 2.0, 1.2), "Illustrative elevated branch."),
    IllustrativeNode("N-05", "South branch", (0.0, -2.0, 0.0), "Illustrative branch endpoint."),
    IllustrativeNode("N-06", "South riser", (2.5, -2.0, -0.8), "Illustrative lower branch."),
    IllustrativeNode("N-07", "Return header", (4.5, 0.0, 0.0), "Illustrative network outlet."),
)

_EDGES = (
    (0, 1),
    (1, 2),
    (2, 3),
    (1, 4),
    (4, 5),
    (1, 6),
)


def _pipe_mesh(
    start: tuple[float, float, float],
    end: tuple[float, float, float],
    radius: float = 0.105,
    sides: int = 14,
) -> tuple[list[float], list[float], list[float], list[int], list[int], list[int]]:
    direction = tuple(end[i] - start[i] for i in range(3))
    length = math.sqrt(sum(component * component for component in direction))
    unit = tuple(component / length for component in direction)
    reference = (0.0, 0.0, 1.0) if abs(unit[2]) < 0.9 else (0.0, 1.0, 0.0)
    cross = (
        unit[1] * reference[2] - unit[2] * reference[1],
        unit[2] * reference[0] - unit[0] * reference[2],
        unit[0] * reference[1] - unit[1] * reference[0],
    )
    norm = math.sqrt(sum(component * component for component in cross))
    u = tuple(component / norm for component in cross)
    v = (
        unit[1] * u[2] - unit[2] * u[1],
        unit[2] * u[0] - unit[0] * u[2],
        unit[0] * u[1] - unit[1] * u[0],
    )

    x: list[float] = []
    y: list[float] = []
    z: list[float] = []
    for endpoint in (start, end):
        for side in range(sides):
            angle = 2 * math.pi * side / sides
            offset = tuple(
                radius * (math.cos(angle) * u[i] + math.sin(angle) * v[i])
                for i in range(3)
            )
            x.append(endpoint[0] + offset[0])
            y.append(endpoint[1] + offset[1])
            z.append(endpoint[2] + offset[2])

    triangles_i: list[int] = []
    triangles_j: list[int] = []
    triangles_k: list[int] = []
    for side in range(sides):
        next_side = (side + 1) % sides
        triangles_i.extend((side, next_side))
        triangles_j.extend((next_side, next_side + sides))
        triangles_k.extend((side + sides, next_side + sides))
    return x, y, z, triangles_i, triangles_j, triangles_k


def make_pipeline_figure(selected_node_id: str | None = None) -> go.Figure:
    """Build an illustrative Plotly 3D layout; it has no physical topology mapping."""
    figure = go.Figure()
    for start_index, end_index in _EDGES:
        mesh = _pipe_mesh(NODES[start_index].position, NODES[end_index].position)
        figure.add_trace(
            go.Mesh3d(
                x=mesh[0],
                y=mesh[1],
                z=mesh[2],
                i=mesh[3],
                j=mesh[4],
                k=mesh[5],
                name="Connected pipe",
                color="#647985",
                opacity=0.98,
                flatshading=False,
                lighting={
                    "ambient": 0.58,
                    "diffuse": 0.72,
                    "specular": 0.3,
                    "roughness": 0.52,
                    "fresnel": 0.08,
                },
                lightposition={"x": 100, "y": 150, "z": 220},
                hoverinfo="skip",
                showlegend=False,
            )
        )

    selected_index = next(
        (index for index, node in enumerate(NODES) if node.node_id == selected_node_id),
        None,
    )
    figure.add_trace(
        go.Scatter3d(
            x=[node.position[0] for node in NODES],
            y=[node.position[1] for node in NODES],
            z=[node.position[2] for node in NODES],
            mode="markers+text",
            text=[node.node_id for node in NODES],
            textposition="top center",
            textfont={"size": 14, "color": "#e4e9ec"},
            customdata=[[node.node_id, node.name, node.description] for node in NODES],
            hovertemplate=(
                "<b>%{customdata[0]} · %{customdata[1]}</b>"
                "<br>%{customdata[2]}<extra></extra>"
            ),
            marker={
                "size": [11 if index == selected_index else 8 for index in range(len(NODES))],
                "color": [
                    "#d2a653" if index == selected_index else "#91a4af"
                    for index in range(len(NODES))
                ],
                "line": {"color": "#e6ecef", "width": 1.4},
                "opacity": 1.0,
            },
            name="Network node",
            showlegend=False,
        )
    )

    figure.add_trace(
        go.Scatter3d(
            x=[None, None],
            y=[None, None],
            z=[None, None],
            mode="lines",
            line={"color": "#647985", "width": 8},
            name="Connected pipe",
            showlegend=True,
            hoverinfo="skip",
        )
    )
    figure.add_trace(
        go.Scatter3d(
            x=[None],
            y=[None],
            z=[None],
            mode="markers",
            marker={"size": 7, "color": "#91a4af", "line": {"color": "#e6ecef", "width": 1}},
            name="Illustrative node",
            showlegend=True,
            hoverinfo="skip",
        )
    )
    figure.add_trace(
        go.Scatter3d(
            x=[None],
            y=[None],
            z=[None],
            mode="markers",
            marker={"size": 8, "color": "#d2a653", "line": {"color": "#e6ecef", "width": 1}},
            name="Selected node",
            showlegend=True,
            hoverinfo="skip",
        )
    )

    figure.update_layout(
        paper_bgcolor="#101316",
        plot_bgcolor="#101316",
        font={"family": "Segoe UI, sans-serif", "color": "#d2d9de", "size": 11},
        hoverlabel={
            "bgcolor": "#20272c",
            "bordercolor": "#46525a",
            "font": {"color": "#edf0f2", "family": "Segoe UI, sans-serif", "size": 12},
        },
        legend={
            "orientation": "h",
            "x": 0.5,
            "xanchor": "center",
            "y": 0.02,
            "yanchor": "bottom",
            "bgcolor": "rgba(16,19,22,0.78)",
            "bordercolor": "#303a42",
            "borderwidth": 1,
            "font": {"color": "#c1cbd1", "size": 11},
        },
        margin={"l": 0, "r": 0, "t": 4, "b": 4},
        height=610,
        showlegend=True,
        clickmode="event+select",
        uirevision="sps-illustrative-network",
        scene={
            "bgcolor": "#101316",
            "xaxis": {"visible": False, "showgrid": False, "showbackground": False},
            "yaxis": {"visible": False, "showgrid": False, "showbackground": False},
            "zaxis": {"visible": False, "showgrid": False, "showbackground": False},
            "aspectmode": "data",
            "camera": {
                "eye": {"x": 1.8, "y": 1.75, "z": 1.45},
                "up": {"x": 0, "y": 0, "z": 1},
            },
        },
    )
    return figure


def get_node(node_id: str) -> IllustrativeNode:
    for node in NODES:
        if node.node_id == node_id:
            return node
    raise ValueError(f"Unknown illustrative node: {node_id}")
