"""
Setup for registry design pattern, which will save all classes that subclass
BlueNode for UI to display.

Heavy inspiration from:
https://charlesreid1.github.io/python-patterns-the-registry.html

Author: Kevin Fu
"""
import abc

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

    def __init__(self):
        self._input_gray_nodes = []
        self._output_gray_nodes = []
        self._parent_plate = None

        self._set_gray_nodes()

    @abc.abstractmethod
    def _set_gray_nodes(self):
        """
        Abstract method to force GTSFM developers to populate
        self._*gray_nodes (or willfully ignore if needed).
        """
        raise NotImplementedError

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
        return f"{self.__class__.__name__}:\n\t input_gray_nodes: {self.input_gray_nodes},\n\t output_gray_nodes: {self.output_gray_nodes},\n\t parent_plate: {self.parent_plate}\n"

