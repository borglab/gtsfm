"""
Creates a process graph based on the classes that subclass GTSFMProcess.

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

from typing import Set
from gtsfm.ui.gtsfm_process import UiMetadata

JS_ROOT = os.path.join(Path(__file__).resolve().parent.parent.parent, "rtf_vis_tool")
DEFAULT_GRAPH_VIZ_OUTPUT_PATH = os.path.join(JS_ROOT, "src", "ui", "process_graph_output.svg")


class ProcessGraphGenerator:
    """Generates and saves a graph of all the components in REGISTRY."""

    def __init__(self, test_mode: bool = False) -> None:
        """Create ProcessGraphGenerator.

        Args:
            test_mode: boolean flag, only set True when unit testing.
        """

        # create empty directed graph
        self._main_graph = pydot.Dot(graph_type="digraph", fontname="Veranda", bgcolor="white")

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

        # dict of pydot Clusters for plates
        # http://robertyu.com/wikiperdido/Pydot%20Clusters
        self._plate_to_cluster = {}

        self._test_mode = test_mode

    def _build_graph(self) -> None:
        """Build graph based on RegistryHolder's REGISTRY."""

        unique_metadata = self._get_metadata_from_registry()

        for metadata in unique_metadata:
            self._add_nodes_to_graph(metadata)

        for plate_name, cluster in self._plate_to_cluster.items():
            self._main_graph.add_subgraph(cluster)

    def _get_metadata_from_registry(self) -> Set[UiMetadata]:
        """Get a set of unique_metadata from the central registry.

        Also, create empty pydot clusters for later use as plate subgraphs, and store in self._plate_to_cluster.
        """
        unique_metadata = set()

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
            if metadata in unique_metadata:
                continue
            unique_metadata.add(metadata)

            # create an empty plate for each unique parent_plate name
            plate = metadata.parent_plate
            if plate not in self._plate_to_cluster:
                # if no plate, add straight to main graph
                if plate == "":
                    continue

                new_cluster = pydot.Cluster(plate, label=plate)
                self._plate_to_cluster[plate] = new_cluster

        return unique_metadata

    def _add_nodes_to_graph(self, metadata: UiMetadata) -> None:
        """Add connected blue/gray nodes to relevant subgraph based on UiMetadata.

        Also, auto-casts metadata.input_products and metadata.output_products to tuples of strings.

        Why? A common error is to define a one-element tuple like so:
          > input_products = ("Input Product")
        but this is actually parsed as a raw string. The correct usage is:
          > input_products = ("Input Product",)
        which is unintuitive. Auto-cast here prevents unexpected behavior for developers.

        Args:
            metadata: UiMetadata object to add nodes/edges for
        """

        display_name = metadata.display_name
        style = self._style

        # autocast strings to one-element tuples
        input_products = metadata.input_products
        if type(input_products) == str:
            input_products = (input_products,)
        output_products = metadata.output_products
        if type(output_products) == str:
            output_products = (output_products,)

        cluster = self._main_graph
        if metadata.parent_plate != "":
            cluster = self._plate_to_cluster[metadata.parent_plate]

        cluster.add_node(
            pydot.Node(display_name, shape=style.node_shape, style=style.node_style, fillcolor=style.blue_fillcolor)
        )
        for product_name in input_products + output_products:
            cluster.add_node(
                pydot.Node(product_name, shape=style.node_shape, style=style.node_style, fillcolor=style.gray_fillcolor)
            )

        # add Edges for all input_products -> blue node -> all output_products
        for input_product_name in input_products:
            self._main_graph.add_edge(pydot.Edge(input_product_name, display_name, color=style.arrow_color))
        for output_product_name in output_products:
            self._main_graph.add_edge(pydot.Edge(display_name, output_product_name, color=style.arrow_color))

    def save_graph(self, filepath: str = DEFAULT_GRAPH_VIZ_OUTPUT_PATH) -> None:
        """Save graph to the given filepath."""

        # graph must be built first
        self._build_graph()

        # make output directory if one does not exist
        save_dir = os.path.dirname(filepath)
        Path(save_dir).mkdir(parents=True, exist_ok=True)

        if filepath.endswith(".png"):
            self._main_graph.write_png(filepath)
        elif filepath.endswith(".svg"):
            self._main_graph.write_svg(filepath)
