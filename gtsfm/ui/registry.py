"""
Setup for registry design pattern, which will save all classes that subclass
GTSFMProcess for UI to display.

Usage for developers:

# 1 - import GTSFMProcess, UiMetadata
from gtsfm.ui.registry import GTSFMProcess, UiMetadata

# 2 - subclass GTSFMProcess (which inherits from ABCMeta)
class ClassName(GTSFMProcess):

    # 3 - implement get_ui_metadata by returning a new UiMetadata object
    def get_ui_metadata() -> UiMetadata:
        return UiMetadata(...)

Heavy inspiration from:
https://charlesreid1.github.io/python-patterns-the-registry.html

Author: Kevin Fu
"""
import abc
from typing import Tuple

from dataclasses import dataclass


class RegistryHolder(type):
    """
    Class that defines central registry and automatically registers classes
    that extend GTSFMProcess.
    """

    REGISTRY = {}

    def __new__(cls, name, bases, attrs):
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
    def get_registry(cls):
        """Return current REGISTRY."""
        return dict(cls.REGISTRY)


class AbstractableRegistryHolder(abc.ABCMeta, RegistryHolder):
    """Extra class to ensure GTSFMProcess can use both ABCMeta and RegistryHolder metaclasses."""

    pass


@dataclass(frozen=True)
class UiMetadata:
    """
    Dataclass to hold UI metadata of a GTSFMProcess.

    frozen=True makes this dataclass immutable and hashable.
    """

    display_name: str
    input_products: Tuple[str]
    output_products: Tuple[str]
    parent_plate: str


class GTSFMProcess(metaclass=AbstractableRegistryHolder):
    """
    Base type that all classes the REGISTRY can see must inherit from.

    Built as a Mixin. For example usage see `tests/ui/test_registry.py`.
    """

    @staticmethod
    @abc.abstractmethod
    def get_ui_metadata() -> UiMetadata:
        """Return a new UiMetadata dataclass."""
        ...
