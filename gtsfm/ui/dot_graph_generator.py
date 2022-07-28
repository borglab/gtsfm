"""
Creates a DOT graph based on the BlueNodes in RegistryHolder.REGISTRY.

Author: Kevin Fu
"""

from gtsfm.ui.registry import RegistryHolder

from pathlib import Path
import os

import pydot

REPO_ROOT = Path(__file__).resolve().parent.parent

class DotGraphGenerator:
    """Generates and saves a graph of all the blue nodes and gray nodes in REGISTRY."""

    def __init__(self, test_mode=False):
        """Create DotGraphGenerator.

        Args:
            test_mode - boolean flag, only set True when unit testing.
        """

        # create empty directed graph
        self._graph = pydot.Dot("my_graph", graph_type="digraph", bgcolor="white") 
        # TODO: fontname seems broken?

        # style constants
        # see http://www.graphviz.org/documentation/
        self._style = {
            "blue_fillcolor": "lightskyblue",
            "gray_fillcolor": "gainsboro",
            "node_style": '"rounded, filled, solid"',
            "node_shape": "box",
            "arrow_color": "gray75",
        }

        self._test_mode = test_mode

    def _build_graph(self):
        """
        Build graph of blue nodes and gray nodes based on RegistryHolder's
        REGISTRY.
        """

        # TODO: remove this
        print("!\n"*100)
        print(RegistryHolder.get_registry())

        for blue_node_name, blue_node_cls in RegistryHolder.get_registry().items():
            # don't add the base class to the graph
            if blue_node_name == "BlueNode":
                continue

            # don't add any test classes to the graph, unless in testing mode
            if not self._test_mode and blue_node_name.startswith("Fake"):
                continue

            # create a new instance of a blue node so we can access its gray nodes
            blue_node = blue_node_cls()

            # add blue node, all gray nodes to graph as Nodes
            self._graph.add_node(pydot.Node(blue_node_name, shape=self._style["node_shape"], style=self._style["node_style"], fillcolor=self._style["blue_fillcolor"]))
            for gray_node_name in (blue_node.input_gray_nodes + blue_node.output_gray_nodes):
                self._graph.add_node(pydot.Node(gray_node_name, shape=self._style["node_shape"], style=self._style["node_style"], fillcolor=self._style["gray_fillcolor"]))

            # add Edges for all input_gray_nodes -> blue node -> all output_gray_nodes
            for input_gray_node_name in blue_node.input_gray_nodes:
                self._graph.add_edge(pydot.Edge(input_gray_node_name, blue_node_name, color=self._style["arrow_color"]))
            for output_gray_node_name in blue_node.output_gray_nodes:
                self._graph.add_edge(pydot.Edge(blue_node_name, output_gray_node_name, color=self._style["arrow_color"]))

    def save_graph(self, filepath=os.path.join(REPO_ROOT, "ui", "output", "dot_graph_output.svg")):
        """Save graph to the path `gtsfm/ui/filename`."""

        # graph must be built first
        self._build_graph()

        if filepath.endswith(".png"):
            self._graph.write_png(filepath)
        elif filepath.endswith(".svg"):
            self._graph.write_svg(filepath)

