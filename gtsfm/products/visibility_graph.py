"""
Defining index image pair(s), a visibility graph, and an annotated visibility graph.
"""

from typing import TypeVar

ImageIndexPair = tuple[int, int]
ImageIndexPairs = list[ImageIndexPair]  # list of (i,j) index pairs

VisibilityGraph = ImageIndexPairs  # if we mean the graph and not a subset of edges

T = TypeVar("T")

AnnotatedGraph = dict[ImageIndexPair, T]  # (i,j) -> T
