"""Generic immutable rooted trees."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Generic, Iterator, Tuple, TypeVar, cast

T = TypeVar("T")
U = TypeVar("U")
Self = TypeVar("Self", bound="Tree[Any]")


@dataclass(frozen=True)
class Tree(Generic[T]):
    """Immutable rooted tree node storing a payload and child subtrees."""

    value: T
    children: Tuple["Tree[T]", ...] = ()

    def is_leaf(self) -> bool:
        """Return True if this node has no children."""
        return len(self.children) == 0

    def leaves(self: Self) -> Tuple[Self, ...]:
        """Return leaf nodes in depth-first order."""
        if self.is_leaf():
            return (self,)
        leaves: list[Self] = []
        for child in self.children:
            leaves.extend(cast(Self, child).leaves())
        return tuple(leaves)

    def traverse(self) -> Iterator["Tree[T]"]:
        """Yield nodes in a pre-order traversal."""
        yield self
        for child in self.children:
            yield from child.traverse()

    def map(self, fn: Callable[[T], U]) -> "Tree[U]":
        """Create a new tree by applying `fn` to every payload."""
        mapped_children = tuple(child.map(fn) for child in self.children)
        return Tree[U](value=fn(self.value), children=mapped_children)

    def fold(self, fn: Callable[[T, Tuple[U, ...]], U]) -> U:
        """
        Aggregate the tree by reducing nodes bottom-up using the provided function `fn`.

        Args:
            fn: A callable that takes two arguments:
            - The value of the current node (`self.value`).
            - A tuple containing the results of folding each child node.
            It should return the aggregated result for the current node. The function `fn` should
            be able to handle cases where `children` is empty (i.e., for leaf nodes).

        Returns:
            The result of aggregating the entire tree using `fn`.

        Example:
            # Sum all values in the tree:
            #      1
            #     / \
            #    2   3
            #         \
            #          4
            tree = Tree(1, (Tree(2), Tree(3, (Tree(4),))))
            def fn(v, children):
                return v + sum(children)
            total = tree.fold(fn) # yields 10

        Note:
            There is no explicit initial value parameter. The aggregation starts from the leaves,
            where `children` is an empty tuple, and proceeds bottom-up.
        """
        child_results = tuple(child.fold(fn) for child in self.children)
        return fn(self.value, child_results)
