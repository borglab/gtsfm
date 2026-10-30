# Short-name exports for partitioner classes.
#
# Usage (Hydra/Python):
#   _target_: gtsfm.graph_partitioner.Binary                    # BinaryTreePartitioner
#   _target_: gtsfm.graph_partitioner.Metis                     # MetisPartitioner
#   _target_: gtsfm.graph_partitioner.RecursiveMetis            # RecursiveMetisPartitioner
#   _target_: gtsfm.graph_partitioner.RecursiveWeightedMinCut   # RecursiveWeightedMinCutPartitioner
#   _target_: gtsfm.graph_partitioner.Single                    # SinglePartitioner

from .binary_tree_partitioner import BinaryTreePartitioner
from .metis_partitioner import MetisPartitioner
from .recursive_metis_partitioner import RecursiveMetisPartitioner
from .recursive_weighted_mincut_partitioner import RecursiveWeightedMinCutPartitioner
from .single_partitioner import SinglePartitioner

# Lightweight aliases so configs can reference shorter names.
Binary = BinaryTreePartitioner
Metis = MetisPartitioner
RecursiveMetis = RecursiveMetisPartitioner
RecursiveWeightedMinCut = RecursiveWeightedMinCutPartitioner
Single = SinglePartitioner

__all__ = [
    "Binary",
    "Metis",
    "RecursiveMetis",
    "RecursiveWeightedMinCut",
    "Single",
]
