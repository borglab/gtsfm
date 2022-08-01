"""
Setup for registry design pattern, which will save all classes that subclass
BlueNode for UI to display.

Heavy inspiration from:
https://charlesreid1.github.io/python-patterns-the-registry.html

Author: Kevin Fu
"""
import abc
from typing import List


class RegistryHolder(type):
    """
    Class that defines central registry and automatically registers classes
    that extend BaseRegisteredClass.
    """

    REGISTRY = {}

    def __new__(cls, name, bases, attrs):
        """
        Every time a new class that extends BaseRegisteredClass is **defined**,
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
    """Extra class to ensure BlueNode can use both ABCMeta and RegistryHolder metaclasses."""

    pass


class BlueNode(metaclass=AbstractableRegistryHolder):
    """Base type that all classes the REGISTRY can see must inherit from."""

    # TODO: consider a design without __init__() and referencing these private
    # vars directly when called from DotGraphGenerator
    #
    # issue: how do you set different metadata in that case?
    def __init__(self):
        self._display_name: str = self.__class__.__name__  # defaults to cls name if none given
        self._input_gray_nodes: List[str] = []
        self._output_gray_nodes: List[str] = []
        self._parent_plate: str = None

        self._set_ui_metadata()

    @abc.abstractmethod
    def _set_ui_metadata(self):
        """
        Abstract method to force GTSFM developers to populate
        useful UI metadata.

        Copy-paste the following when implementing:

        self._display_name: str = <display name>  # defaults to cls name if none given
        self._input_gray_nodes: List[str] = [<gray in 1>, ...]
        self._output_gray_nodes: List[str] = [<gray out 1>, ...]
        self._parent_plate: str = <parent plate>
        """
        raise NotImplementedError

    @property
    def display_name(self):
        return self._display_name

    @property
    def input_gray_nodes(self):
        return self._input_gray_nodes

    @property
    def output_gray_nodes(self):
        return self._output_gray_nodes

    @property
    def parent_plate(self):
        return self._parent_plate

    def __repr__(self):
        return f"{self.display_name}:\n\t input_gray_nodes: {self.input_gray_nodes},\n\t output_gray_nodes: {self.output_gray_nodes},\n\t parent_plate: {self.parent_plate}\n"
