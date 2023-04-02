"""
Defines GTSFMProcess, which holds a UiMetadata object for displaying this process in the process graph.

Author: Kevin Fu
"""

# When creating a new class, follow this template:

# 1: Import GTSFMProcess, UiMetadata
# ----------------------------------
# from gtsfm.ui.gtsfm_process import GTSFMProcess, UiMetadata

# 2: Subclass GTSFMProcess (this can replace ABCMeta, since it inherits from ABCMeta)
# ------------------------
# class ClassName(GTSFMProcess):

# 3: Implement get_ui_metadata()
# ------------------------------
#    def get_ui_metadata() -> UiMetadata:
#        """Returns data needed to display node and edge info for this process in the process graph."""
#
#        return UiMetadata(
#                   display_name="Display Name"
#                   input_products=("Input Product 1", "Input Product 2")
#                   output_products=("Output Product 1", "Output Product 2")
#                   parent_plate="Parent Plate Name"
#               )

import abc
from dataclasses import dataclass
from typing import Tuple, Optional

from gtsfm.ui.registry import AbstractableRegistryHolder


@dataclass(frozen=True, order=True)
class UiMetadata:
    """Holds all info needed to display a GTSFMProcess in the process graph (see ProcessGraphGenerator).

    frozen=True makes this dataclass hashable, used when adding it to a set to avoid duplicate nodes in the graph.
    order=True makes this sortable, used to add the nodes in the same order to avoid non-deterministic DOT graph output.

    Fields:
        display_name: string display_name of a GTSFMProcess
        input_products: tuple of strings representing all products this process consumes
        output_products: tuple of strings representing all products this process produces
        parent_plate: string parent_plate of a GTSFMProcess
    """

    display_name: str
    input_products: Tuple[str, ...]
    output_products: Tuple[str, ...]
    parent_plate: Optional[str] = None


class GTSFMProcess(metaclass=AbstractableRegistryHolder):
    """
    Base type that all classes the REGISTRY can see must inherit from.

    Built as a Mixin. For example usage see test cases for GTSFMProcess.
    """

    @staticmethod
    @abc.abstractmethod
    def get_ui_metadata() -> UiMetadata:
        """Returns data needed to display node and edge info for this process in the process graph."""
        ...
