"""
Setup for registry design pattern, which will save all classes that subclass
GTSFMProcess for UI to display.

Heavy inspiration from:
https://charlesreid1.github.io/python-patterns-the-registry.html

Author: Kevin Fu
"""

import abc
from typing import Any, Callable, Dict, Tuple


class RegistryHolder(type):
    """Class that defines central registry and automatically registers classes
    that extend GTSFMProcess.
    """

    REGISTRY = {}

    def __new__(cls: type, name: str, bases: Tuple[Any], attrs: Dict[str, Callable]) -> None:
        """
        Every time a new class that extends GTSFMProcess is **defined**,
        the REGISTRY here in RegistryHolder will be updated. This is thanks to
        the behavior of Python's built-in __new__().
        """

        new_cls = type.__new__(cls, name, bases, attrs)

        # note: article linked suggests trying a cast to lower, e.g.
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
