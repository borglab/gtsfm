"""Utilities to evaluate tree computations using Dask futures."""

from __future__ import annotations

from typing import Callable, Tuple, TypeVar, cast

from dask.distributed import Client, Future

from gtsfm.utils.tree import Tree

TreePayloadT = TypeVar("TreePayloadT")
AggregatedT = TypeVar("AggregatedT")


def _invoke_fold_node(
    fn: Callable[[AggregatedT, Tuple[AggregatedT, ...]], AggregatedT],
    value: TreePayloadT,
    child_results: Tuple[AggregatedT, ...],
) -> AggregatedT:
    """Helper executed on workers to combine child outputs."""
    return fn(cast(AggregatedT, value), child_results)


def submit_tree_fold(
    client: Client,
    tree: Tree[TreePayloadT],
    fn: Callable[[AggregatedT, Tuple[AggregatedT, ...]], AggregatedT],
) -> Future:
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
    return client.submit(_invoke_fold_node, fn, tree.value, child_futures)


MapPayloadT = TypeVar("MapPayloadT")
MappedT = TypeVar("MappedT")


def submit_tree_map(client: Client, tree: Tree[MapPayloadT], fn: Callable[[MapPayloadT], MappedT]) -> Tree[Future]:
    """
    Submit an independent computation for every node and return a mirror tree of futures.

    Args:
        client: Active Dask client used to submit tasks.
        tree: Tree whose structure will be mirrored.
        fn: Callable invoked at each node with the node value.

    Returns:
        A tree with identical topology where each node stores the future returned by ``fn``.
    """
    future_children = tuple(submit_tree_map(client, child, fn) for child in tree.children)
    future_value = client.submit(fn, tree.value)
    return Tree(value=future_value, children=future_children)


def gather_future_tree(client: Client, future_tree: Tree[Future]) -> Tree[object]:
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
