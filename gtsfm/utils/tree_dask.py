"""Utilities to evaluate tree computations using Dask futures."""

from __future__ import annotations

from typing import Callable, Tuple, TypeVar

from dask.distributed import Client, Future

from gtsfm.utils.tree import Tree

T = TypeVar("T")
U = TypeVar("U")


def _invoke_node(fn: Callable[[T, Tuple[U, ...]], U], value: T, child_results: Tuple[U, ...]) -> U:
    """Helper executed on workers to combine child outputs."""
    return fn(value, child_results)


def submit_tree_fold(client: "Client", tree: Tree[T], fn: Callable[[T, Tuple[U, ...]], U]) -> Future:
    """
    Submit a bottom-up fold of ``tree`` where each node is evaluated as a Dask future.

    Args:
        client: Active Dask client used to submit tasks.
        tree: Tree to traverse.
        fn: Callable invoked at every node, receiving the node value and child outputs.

    Returns:
        A future representing the folded value at the root.
    """
    child_futures = tuple(submit_tree_fold(client, child, fn) for child in tree.children)
    return client.submit(_invoke_node, fn, tree.value, child_futures)


def submit_tree_map(client: "Client", tree: Tree[T], fn: Callable[[T, Tuple[U, ...]], U]) -> Tree[Future]:
    """
    Submit a computation for every node and return a mirror tree of futures.

    Args:
        client: Active Dask client used to submit tasks.
        tree: Tree whose structure will be mirrored.
        fn: Callable invoked at each node with the node value and child outputs.

    Returns:
        A tree with identical topology where each node stores the future returned by ``fn``.
    """
    future_children = tuple(submit_tree_map(client, child, fn) for child in tree.children)
    child_outputs = tuple(child.value for child in future_children)
    future_value = client.submit(_invoke_node, fn, tree.value, child_outputs)
    return Tree(value=future_value, children=future_children)


def gather_future_tree(client: "Client", future_tree: Tree[Future]) -> Tree[object]:
    """
    Gather a mirrored tree of futures back into concrete values.

    Args:
        client: Active Dask client used to gather results.
        future_tree: Tree whose nodes contain futures to resolve.

    Returns:
        A tree mirroring the input structure with computed results instead of futures.
    """
    value = client.gather(future_tree.value)
    children = tuple(gather_future_tree(client, child) for child in future_tree.children)
    return Tree(value=value, children=children)
