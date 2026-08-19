# Short-name exports for retriever classes.
#
# Usage (Hydra/Python):
#   _target_: gtsfm.retriever.Exhaustive
#   _target_: gtsfm.retriever.JointSimilaritySequential
#   _target_: gtsfm.retriever.Sequential
#   _target_: gtsfm.retriever.Similarity
#   _target_: gtsfm.retriever.ColmapDB
#   _target_: gtsfm.retriever.ColmapDBMegaLoc

from .colmap_db_megaloc_retriever import ColmapDBMegaLocRetriever
from .colmap_db_retriever import ColmapDBRetriever
from .exhaustive_retriever import ExhaustiveRetriever
from .joint_similarity_sequential_retriever import JointSimilaritySequentialRetriever
from .sequential_retriever import SequentialRetriever
from .similarity_retriever import SimilarityRetriever

ColmapDB = ColmapDBRetriever
ColmapDBMegaLoc = ColmapDBMegaLocRetriever
Exhaustive = ExhaustiveRetriever
JointSimilaritySequential = JointSimilaritySequentialRetriever
Sequential = SequentialRetriever
Similarity = SimilarityRetriever

__all__ = [
    "ColmapDB",
    "ColmapDBMegaLoc",
    "Exhaustive",
    "JointSimilaritySequential",
    "Sequential",
    "Similarity",
]
