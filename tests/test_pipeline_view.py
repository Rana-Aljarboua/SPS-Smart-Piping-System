from __future__ import annotations

import unittest

import plotly.graph_objects as go

from sps.pipeline_view import NODES, make_pipeline_figure


class PipelineViewTests(unittest.TestCase):
    def test_figure_contains_connected_pipes_labeled_nodes_and_legend(self) -> None:
        figure = make_pipeline_figure()

        self.assertEqual(
            sum(isinstance(trace, go.Mesh3d) for trace in figure.data),
            6,
        )
        node_trace = next(trace for trace in figure.data if trace.name == "Network node")
        self.assertEqual(len(node_trace.x), len(NODES))
        self.assertEqual(list(node_trace.text), [node.node_id for node in NODES])
        self.assertEqual(
            {trace.name for trace in figure.data if trace.showlegend},
            {"Connected pipe", "Illustrative node", "Selected node"},
        )
        self.assertEqual(figure.layout.scene.camera.eye.x, 1.8)
        self.assertEqual(figure.layout.paper_bgcolor, "#101316")

    def test_selected_node_is_visibly_highlighted(self) -> None:
        figure = make_pipeline_figure("N-04")
        node_trace = next(trace for trace in figure.data if trace.name == "Network node")

        self.assertEqual(list(node_trace.marker.color)[3], "#d2a653")
        self.assertEqual(list(node_trace.marker.size)[3], 11)
        self.assertEqual(node_trace.customdata[3][0], "N-04")


if __name__ == "__main__":
    unittest.main()
