"""
Defines GTSFMProcess, which holds a UiMetadata object for displaying this process in the process graph.

Author: Kevin Fu
"""

# When creating a new class, follow this template:

# 1: Import GTSFMProcess, UiMetadata
# ----------------------------------
# from gtsfm.ui.registry import GTSFMProcess, UiMetadata

# 2: Subclass GTSFMProcess (this can replace ABCMeta, since it inherits from ABCMeta)
# ------------------------
# class ClassName(GTSFMProcess):

# 3: Implement get_ui_metadata()
# ------------------------------
#    def get_ui_metadata() -> UiMetadata:
#        """Returns data needed to display this process in the process graph. See gtsfm/ui/registry.py for more info."""
#
#        return UiMetadata(
#                   "Display Name"
#                   "Parent Plate"
#                   ("Input Product 1", "Input Product 2")
#                   ("Output Product 1", "Output Product 2")
#               )

from gtsfm.ui.registry import AbstractableRegistryHolder

import abc
from dataclasses import dataclass

from typing import Tuple


@dataclass(frozen=True)
class UiMetadata:
    """
    Holds all info needed to display a GTSFMProcess in the process graph (see ProcessGraphGenerator).

    frozen=True makes this dataclass immutable and hashable.

    Fields:
        display_name: string display_name of a GTSFMProcess
        parent_plate: string parent_plate of a GTSFMProcess
        input_products: tuple of strings representing all products this process consumes
        output_products: tuple of strings representing all products this process produces
    """

    display_name: str
    parent_plate: str
    input_products: Tuple[str]
    output_products: Tuple[str]


class GTSFMProcess(metaclass=AbstractableRegistryHolder):
    """
    Base type that all classes the REGISTRY can see must inherit from.

    Built as a Mixin. For example usage see test cases for GTSFMProcess.
    """

    @staticmethod
    @abc.abstractmethod
    def get_ui_metadata() -> UiMetadata:
        """Returns data needed to display this process in the process graph."""
        ...
