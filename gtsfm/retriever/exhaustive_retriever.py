"""Exhaustive retriever, that proposes all possible image pairs (N choose 2).

Only useful for temporally ordered data.

Authors: John Lambert
"""

from gtsfm.retriever.sequential_retriever import SequentialRetriever

# For exhaustive matching, we limit the lookahead to 10,000 images.
MAX_POSSIBLE_FRAME_LOOKAHEAD = 10000


class ExhaustiveRetriever(SequentialRetriever):
    def __init__(self) -> None:
        """Constructor. All frames are considered for matching/co-visibility."""
        super().__init__(max_frame_lookahead=MAX_POSSIBLE_FRAME_LOOKAHEAD)
