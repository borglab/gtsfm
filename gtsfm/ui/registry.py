"""
Setup for registry design pattern, which will save all classes that subclass
GTSFMProcess for UI to display.

Heavy inspiration from:
https://charlesreid1.github.io/python-patterns-the-registry.html

Author: Kevin Fu
"""
import abc
from typing import List, Optional

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


@dataclass
class UiMetadata:
    """
    Dataclass to hold UI metadata of a GTSFMProcess.
    """

    display_name: str
    input_products: List[str]
    output_products: List[str]
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
