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

    def map(self, fn: Callable[[T], U]) -> "Tree[U]":
        """Create a new tree by applying `fn` to every payload."""
        mapped_children = tuple(child.map(fn) for child in self.children)
        return Tree[U](value=fn(self.value), children=mapped_children)

    @classmethod
    def zip(cls, a: Tree[T], b: Tree[U]) -> "Tree[Tuple[T, U]]":
        """Create a new tree by combining payload of two trees."""
        if len(a.children) != len(b.children):
            raise ValueError("Trees must have the same structure to zip.")
        zipped_children = tuple(cls.zip(a_child, b_child) for a_child, b_child in zip(a.children, b.children))
        return Tree(value=(a.value, b.value), children=zipped_children)

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

    def prune(self, predicate: Callable[[T], bool]) -> "Tree[T] | None":
        """
        Prune subtrees where the predicate is false for all values in the subtree.

        Args:
            predicate: A callable that takes a node value and returns True if the node should be kept.

        Returns:
            A new pruned tree or None if the entire tree is pruned.
        """
        pruned_children = []
        for child in self.children:
            pruned_child = child.prune(predicate)
            if pruned_child is not None:
                pruned_children.append(pruned_child)

        if predicate(self.value) or pruned_children:
            return Tree(value=self.value, children=tuple(pruned_children))
        else:
            return None

    def all(self, predicate: Callable[[T], bool]) -> bool:
        """Return True if predicate is true for all values in the tree."""
        return self.fold(lambda v, child_results: predicate(v) and all(child_results))


class PreOrderIter(Generic[T]):
    """Iterator over nodes in a pre-order traversal (parent before children)."""

    def __init__(self, root: Tree[T]) -> None:
        self.root = root

    def __iter__(self) -> Iterator[Tree[T]]:
        yield self.root
        for child in self.root.children:
            yield from PreOrderIter(child)


class PostOrderIter(Generic[T]):
    """Iterator over nodes in a post-order traversal (children before parent)."""

    def __init__(self, root: Tree[T]) -> None:
        self.root = root

    def __iter__(self) -> Iterator[Tree[T]]:
        for child in self.root.children:
            yield from PostOrderIter(child)
        yield self.root
