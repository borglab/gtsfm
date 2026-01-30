# Short-name exports for retriever classes.
#
# Usage (Hydra/Python):
#   _target_: gtsfm.retriever.Exhaustive
#   _target_: gtsfm.retriever.JointSimilaritySequential
#   _target_: gtsfm.retriever.Sequential
#   _target_: gtsfm.retriever.Similarity

from .exhaustive_retriever import ExhaustiveRetriever
from .joint_similarity_sequential_retriever import JointSimilaritySequentialRetriever
from .sequential_retriever import SequentialRetriever
from .similarity_retriever import SimilarityRetriever
from .faiss_retriever import FaissRetriever

Exhaustive = ExhaustiveRetriever
JointSimilaritySequential = JointSimilaritySequentialRetriever
Sequential = SequentialRetriever
Similarity = SimilarityRetriever
BatchedSimilarity = BatchedSimilarityRetriever

__all__ = [
    "Exhaustive",
    "JointSimilaritySequential",
    "Sequential",
    "Similarity",
    "BatchedSimilarity"
]
