"""Generate visibility graph for the frontend.

Authors: Ayush Baid
"""

from pathlib import Path
from typing import List, Optional

import numpy as np
from dask.distributed import Client, Future

import gtsfm.utils.logger as logger_utils
from gtsfm.common.image import Image
from gtsfm.frontend.global_descriptor.global_descriptor_base import GlobalDescriptorBase
from gtsfm.products.visibility_graph import VisibilityGraph
from gtsfm.retriever.retriever_base import RetrieverBase

logger = logger_utils.get_logger()


class ImagePairsGenerator:
    """Generates visibility graphs for structure-from-motion frontend processing."""

    def __init__(
        self, retriever: RetrieverBase, global_descriptor: Optional[GlobalDescriptorBase] = None, batch_size: int = 16
    ):
        """Initialize with a retriever and optional global descriptor for similarity matching."""
        self._global_descriptor: Optional[GlobalDescriptorBase] = global_descriptor  # Optional similarity descriptor
        self._retriever: RetrieverBase = retriever  # Core retriever that builds visibility graph
        self._batch_size = batch_size

    def __repr__(self) -> str:
        """Return string representation of the visibility graph generator configuration."""
        return f"""
            ImagePairsGenerator:
                {self._global_descriptor}
                {self._retriever}
        """

    def run(
        self, client: Client, images: List[Future], image_fnames: List[str], plots_output_dir: Optional[Path] = None
    ) -> VisibilityGraph:
        """Generate visibility graph using global descriptors and retriever logic."""

        def apply_global_descriptor_batch(
            global_descriptor: GlobalDescriptorBase, image_batch: torch.Tensor
        ) -> List[np.ndarray]:
            """Apply global descriptor to extract feature vectors from a batch of images."""
            # This will call the new method you need to create in your descriptor class.
            return global_descriptor.describe_batch(images=image_batch)

        image_batch_futures = images  # Image Pairs Generator assumes that we are getting batches now
        descriptors: Optional[List[np.ndarray]] = None  # Will hold global descriptors if computed

        if self._global_descriptor is not None:
            # Scatter descriptor to all workers for efficient parallel processing
            global_descriptor_future = client.scatter(self._global_descriptor, broadcast=True)

            descriptor_futures = [
                client.submit(apply_global_descriptor_batch, global_descriptor_future, batch_future)
                for batch_future in image_batch_futures
            ]

            logger.info(f"⏳ Computing global descriptors for all images in batches of {self._batch_size}...")
            batched_descriptors = client.gather(descriptor_futures)

            # Flatten the batched results
            descriptors = [desc for batch in batched_descriptors for desc in batch]

        # Use retriever to construct visibility graph based on descriptors and filenames
        logger.info("⏳ Computing visibility graph...")
        return self._retriever.get_image_pairs(
            global_descriptors=descriptors, image_fnames=image_fnames, plots_output_dir=plots_output_dir
        )
