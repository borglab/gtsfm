"""
Setup for registry design pattern, which will save all classes that subclass
GTSFMProcess for UI to display.

Heavy inspiration from:
https://charlesreid1.github.io/python-patterns-the-registry.html

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
#                   ("Input Product 1", "Input Product 2")
#                   ("Output Product 1", "Output Product 2")
#               )


import abc
from typing import Tuple, Dict

from dataclasses import dataclass


class RegistryHolder(type):
    """
    Class that defines central registry and automatically registers classes
    that extend GTSFMProcess.
    """

    REGISTRY = {}

    def __new__(cls: type, name: str, bases: Tuple, attrs: Dict) -> None:
        """
        Every time a new class that extends GTSFMProcess is **defined**,
        the REGISTRY here in RegistryHolder will be updated. This is thanks to
        the behavior of Python's built-in __new__().
        """

        new_cls = type.__new__(cls, name, bases, attrs)

        # TODO: article linked suggests trying a cast to lower, e.g.
        # cls.REGISTRY[new_cls.__name__.lower()] = new_cls
        cls.REGISTRY[new_cls.__name__] = new_cls
        return new_cls

    @classmethod
    def get_registry(cls: type) -> Dict:
        """Return current REGISTRY."""
        return dict(cls.REGISTRY)


class AbstractableRegistryHolder(abc.ABCMeta, RegistryHolder):
    """Extra class to ensure GTSFMProcess can use both ABCMeta and RegistryHolder metaclasses."""

    pass


@dataclass(frozen=True)
class UiMetadata:
    """
    Holds all info needed to display a GTSFMProcess in the process graph (see ProcessGraphGenerator).

    frozen=True makes this dataclass immutable and hashable.

    Fields:
        display_name: string display_name of a GTSFMProcess
        input_products: tuple of strings representing all products this process consumes
        output_products: tuple of strings representing all products this process produces
    """

    display_name: str
    input_products: Tuple[str]
    output_products: Tuple[str]


class GTSFMProcess(metaclass=AbstractableRegistryHolder):
    """
    Base type that all classes the REGISTRY can see must inherit from.

    Built as a Mixin. For example usage see `tests/ui/test_registry.py`.
    """

    @staticmethod
    @abc.abstractmethod
    def get_ui_metadata() -> UiMetadata:
        """Returns data needed to display this process in the process graph. See gtsfm/ui/registry.py for more info."""
        ...
