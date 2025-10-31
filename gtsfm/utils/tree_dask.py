"""Utilities to evaluate tree computations using Dask futures."""

from __future__ import annotations

from typing import Callable, Tuple, TypeVar, cast

from dask.distributed import Client, Future

from gtsfm.utils.tree import Tree

TreePayloadT = TypeVar("TreePayloadT")
AggregatedT = TypeVar("AggregatedT")
MappedWithChildrenT = TypeVar("MappedWithChildrenT")


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


def submit_tree_map(
    client: Client,
    tree: Tree[MapPayloadT],
    fn: Callable[[MapPayloadT], MappedT],
    *,
    pure: bool | None = True,
) -> Tree[Future]:
    """
    Submit an independent computation for every node and return a mirror tree of futures.

    Args:
        client: Active Dask client used to submit tasks.
        tree: Tree whose structure will be mirrored.
        fn: Callable invoked at each node with the node value.

    Returns:
        A tree with identical topology where each node stores the future returned by ``fn``.
    """
    future_children = tuple(submit_tree_map(client, child, fn, pure=pure) for child in tree.children)
    submit_kwargs = {} if pure is None else {"pure": pure}
    future_value = client.submit(fn, tree.value, **submit_kwargs)
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


def _invoke_map_with_children_node(
    fn: Callable[[TreePayloadT, Tuple[MappedWithChildrenT, ...]], MappedWithChildrenT],
    value: object,
    child_results: Tuple[MappedWithChildrenT, ...],
) -> MappedWithChildrenT:
    """Helper to combine node payload with already-mapped child results."""
    return fn(cast(TreePayloadT, value), child_results)


def submit_tree_map_with_children(
    client: Client,
    tree: Tree[TreePayloadT],
    fn: Callable[[TreePayloadT, Tuple[MappedWithChildrenT, ...]], MappedWithChildrenT],
) -> Tree[Future]:
    """
    Submit computations where each node depends on its mapped child outputs.

    Args:
        client: Active Dask client used to submit tasks.
        tree: Tree whose payload will seed the computation.
        fn: Callable invoked at every node with the node payload and mapped child results.

    Returns:
        A tree mirroring the input topology whose nodes store futures with mapped results.
    """
    mapped_children = tuple(submit_tree_map_with_children(client, child, fn) for child in tree.children)
    child_results = tuple(child.value for child in mapped_children)
    mapped_value = client.submit(_invoke_map_with_children_node, fn, tree.value, child_results)
    return Tree(value=mapped_value, children=mapped_children)
