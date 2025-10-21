# Short-name exports for partitioner classes.
#
# Usage (Hydra/Python):
#   _target_: gtsfm.graph_partitioner.Binary          # BinaryTreePartitioner
#   _target_: gtsfm.graph_partitioner.Metis           # MetisPartitioner
#   _target_: gtsfm.graph_partitioner.Single          # SinglePartitioner

from .binary_tree_partitioner import BinaryTreePartitioner
from .metis_partitioner import MetisPartitioner
from .single_partitioner import SinglePartitioner

# Lightweight aliases so configs can reference shorter names.
Binary = BinaryTreePartitioner
Metis = MetisPartitioner
Single = SinglePartitioner

__all__ = [
    "Binary",
    "Metis",
    "Single",
]
