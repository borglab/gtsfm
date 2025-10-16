"""Generic immutable rooted trees."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Generic, Iterator, Tuple, TypeVar

T = TypeVar("T")
U = TypeVar("U")


@dataclass(frozen=True)
class Tree(Generic[T]):
    """Immutable rooted tree node storing a payload and child subtrees."""

    value: T
    children: Tuple["Tree[T]", ...] = ()

    def is_leaf(self) -> bool:
        """Return True if this node has no children."""
        return len(self.children) == 0

    def leaves(self) -> Tuple["Tree[T]", ...]:
        """Return leaf nodes in depth-first order."""
        if self.is_leaf():
            return (self,)
        leaves: list[Tree[T]] = []
        for child in self.children:
            leaves.extend(child.leaves())
        return tuple(leaves)

    def traverse(self) -> Iterator["Tree[T]"]:
        """Yield nodes in a pre-order traversal."""
        yield self
        for child in self.children:
            yield from child.traverse()

    def map(self, fn: Callable[[T], U]) -> "Tree[U]":
        """Create a new tree by applying `fn` to every payload."""
        return Tree(value=fn(self.value), children=tuple(child.map(fn) for child in self.children))
