"""
Creates a DOT graph based on the contents of RegistryHolder.REGISTRY.

Author: Kevin Fu
"""

from gtsfm.ui.registry import RegistryHolder

from pathlib import Path
import os

import pydot
from collections import namedtuple
from inspect import isabstract

JS_ROOT = os.path.join(Path(__file__).resolve().parent.parent.parent, "rtf_vis_tool")


class DotGraphGenerator:
    """Generates and saves a graph of all the components in REGISTRY."""

    def __init__(self, test_mode=False):
        """Create DotGraphGenerator.

        Args:
            test_mode - boolean flag, only set True when unit testing.
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

    def _build_graph(self):
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

            display_name = metadata.display_name

            # autocast strings to one-element tuples
            #
            # a common error is to define a one-element tuple like so:
            # >>> ("Input Product")
            # but Python auto-casts this to a raw str
            input_products = metadata.input_products
            if type(input_products) == str:
                input_products = (input_products,)

            output_products = metadata.output_products
            if type(output_products) == str:
                output_products = (output_products,)

            # parent_plate = metadata.parent_plate  # currently unused
            style = self._style

            # add process, all products to graph as Nodes
            self._graph.add_node(
                pydot.Node(display_name, shape=style.node_shape, style=style.node_style, fillcolor=style.blue_fillcolor)
            )
            for product_name in input_products + output_products:
                self._graph.add_node(
                    pydot.Node(
                        product_name, shape=style.node_shape, style=style.node_style, fillcolor=style.gray_fillcolor
                    )
                )

            # add Edges for all input_products -> blue node -> all output_products
            for input_product_name in input_products:
                self._graph.add_edge(pydot.Edge(input_product_name, display_name, color=style.arrow_color))
            for output_product_name in output_products:
                self._graph.add_edge(pydot.Edge(display_name, output_product_name, color=style.arrow_color))

    def save_graph(self, filepath=os.path.join(JS_ROOT, "src", "ui", "dot_graph_output.svg")):
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
