"""
Creates a DOT graph based on the classes that subclass GTSFMProcess.

Specifically, all classes that subclass GTSFMProcess are added to a central
registry on declaration. GTSFMProcess has the abstract class method
get_ui_metadata(), which returns a UiMetadata object. 

ProcessGraphGenerator combines all the UiMetadata of all the GTSFMProcesses into
one graph of blue and gray nodes, then saves to a file.

Author: Kevin Fu
"""

from gtsfm.ui.registry import RegistryHolder

from pathlib import Path
import os

import pydot
from collections import namedtuple

from typing import Tuple
from gtsfm.ui.registry import UiMetadata

JS_ROOT = os.path.join(Path(__file__).resolve().parent.parent.parent, "rtf_vis_tool")
DEFAULT_GRAPH_VIZ_OUTPUT_PATH = os.path.join(JS_ROOT, "src", "ui", "dot_graph_output.svg")


class ProcessGraphGenerator:
    """Generates and saves a graph of all the components in REGISTRY."""

    def __init__(self, test_mode: bool = False) -> None:
        """Create ProcessGraphGenerator.

        Args:
            test_mode: boolean flag, only set True when unit testing.
        """

        # create empty directed graph
        self._graph = pydot.Dot("my_graph", graph_type="digraph", bgcolor="white")
        # TODO: fontname seems broken?

        # style constants, see http://www.graphviz.org/documentation/
        style_consts = {
            "blue_fillcolor": "lightskyblue",
            "gray_fillcolor": "gainsboro",
            "node_style": '"rounded, filled, solid"',
            "node_shape": "box",
            "arrow_color": "gray75",
        }
        # namedtuple here is faster than default Dict and allows use of dot
        # operator
        StyleTuple = namedtuple("StyleTuple", style_consts)
        self._style = StyleTuple(**style_consts)

        self._test_mode = test_mode

    def _build_graph(self) -> None:
        """Build graph based on RegistryHolder's REGISTRY."""

        # TODO: remove this
        print("!\n" * 100)
        print(RegistryHolder.get_registry())

        seen_metadata = set()
        for cls_name, cls_type in RegistryHolder.get_registry().items():
            # don't add the base class to the graph
            if cls_name == "GTSFMProcess":
                continue

            # don't add any test classes to the graph, unless in testing mode
            if not self._test_mode and cls_name.startswith("Fake"):
                continue

            # get UI metadata of class
            metadata = cls_type.get_ui_metadata()

            # skip duplicates
            # happens when concrete classes implement an abstract class without overwriting get_ui_metadata()
            if metadata in seen_metadata:
                continue

            seen_metadata.add(metadata)

            self._add_metadata_to_graph(metadata)

    def _add_metadata_to_graph(self, metadata: UiMetadata) -> None:
        """Add UiMetadata to the graph as blue/gray nodes and corresponding edges.

        Auto-casts metadata.input_products and metadata.output_products to tuples of strings.

        Why? A common error is to define a one-element tuple like so:
          > input_products = ("Input Product")
        but this is actually parsed as a raw string. The correct usage is:
          > input_products = ("Input Product",)
        which is unintuitive. Auto-cast here prevents unexpected behavior for developers.

        Args:
            metadata: UiMetadata object
        """

        display_name = metadata.display_name

        # autocast strings to one-element tuples
        input_products = metadata.input_products
        if type(input_products) == str:
            input_products = (input_products,)

        output_products = metadata.output_products
        if type(output_products) == str:
            output_products = (output_products,)

        parent_plate = metadata.parent_plate

        # add cleaned-up metadata to graph
        self._add_nodes_and_edges(display_name, input_products, output_products, parent_plate)

    def _add_nodes_and_edges(
        self, display_name: str, input_products: Tuple[str], output_products: Tuple[str], parent_plate: str
    ) -> None:
        """Given the sanitized fields of a UiMetadata object, add blue/gray nodes and edges to the graph.

        Args:
            display_name: string display name (from UiMetadata.display_name)
            input_products: tuple of string names (from UiMetadata.input_products)
            output_products: tuple of string names (from UiMetadata.output_products)
            parent_plate: string name of parent plate (from UiMetadata.parent_plate)
        """

        style = self._style

        # add process, all products to graph as Nodes
        self._graph.add_node(
            pydot.Node(display_name, shape=style.node_shape, style=style.node_style, fillcolor=style.blue_fillcolor)
        )
        for product_name in input_products + output_products:
            self._graph.add_node(
                pydot.Node(product_name, shape=style.node_shape, style=style.node_style, fillcolor=style.gray_fillcolor)
            )

        # add Edges for all input_products -> blue node -> all output_products
        for input_product_name in input_products:
            self._graph.add_edge(pydot.Edge(input_product_name, display_name, color=style.arrow_color))
        for output_product_name in output_products:
            self._graph.add_edge(pydot.Edge(display_name, output_product_name, color=style.arrow_color))

    def save_graph(self, filepath: str = DEFAULT_GRAPH_VIZ_OUTPUT_PATH) -> None:
        """Save graph to the given filepath."""

        # graph must be built first
        self._build_graph()

        # make output directory if one does not exist
        save_dir = os.path.dirname(filepath)
        Path(save_dir).mkdir(parents=True, exist_ok=True)

        if filepath.endswith(".png"):
            self._graph.write_png(filepath)
        elif filepath.endswith(".svg"):
            self._graph.write_svg(filepath)
