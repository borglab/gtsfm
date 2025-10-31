"""Generate visibility graph for the frontend.

Authors: Ayush Baid
"""

import time
from pathlib import Path

import numpy as np
import torch
from dask.distributed import Client, Future
from torchvision.transforms import v2 as transforms  # type: ignore

import gtsfm.utils.logger as logger_utils

# from gtsfm.common.image import Image
from gtsfm.frontend.global_descriptor.global_descriptor_base import GlobalDescriptorBase
from gtsfm.loader.loader_base import BatchTransform, ResizeTransform
from gtsfm.products.visibility_graph import VisibilityGraph
from gtsfm.retriever.retriever_base import RetrieverBase

logger = logger_utils.get_logger()


class ImagePairsGenerator:
    """Generates visibility graphs for structure-from-motion frontend processing."""

    def __init__(
        self, retriever: RetrieverBase, global_descriptor: GlobalDescriptorBase | None = None, batch_size: int = 4
    ):
        """Initialize with a retriever and optional global descriptor for similarity matching."""
        self._global_descriptor: GlobalDescriptorBase | None = global_descriptor  # Optional similarity descriptor
        self._retriever: RetrieverBase = retriever  # Core retriever that builds visibility graph
        self._batch_size = batch_size

    def __repr__(self) -> str:
        """Return string representation of the visibility graph generator configuration."""
        return f"""
            ImagePairsGenerator:
                {self._global_descriptor}
                {self._retriever}
        """

    def get_preprocessing_transforms(self) -> tuple[ResizeTransform, BatchTransform | None]:
        """Get preprocessing transforms from the global descriptor, if available.

        Returns:
            A tuple of (ResizeTransform, BatchTransform) or (None, None) if no global descriptor is set.
        """
        if self._global_descriptor is not None:
            return self._global_descriptor.get_preprocessing_transforms()
        else:
            # No global descriptor; return identity transform converts to Tensor
            # This is purely to satisfy the loader's expected interface, because no descriptor will be computed.
            return transforms.Lambda(lambda x: torch.from_numpy(x)), None

    def run(
        self,
        client: Client,
        image_batch_futures: list[Future],
        image_fnames: list[str],
        plots_output_dir: Path | None = None,
    ) -> VisibilityGraph:
        """Generate visibility graph using global descriptors and retriever logic."""

        def apply_global_descriptor_batch(
            global_descriptor: GlobalDescriptorBase, image_batch: torch.Tensor
        ) -> list[np.ndarray]:
            """Apply global descriptor to extract feature vectors from a batch of images."""

            logger.info(
                "🟩 Computing global descriptors for batch of %d images",
                len(image_batch),
            )

            # This will call the new method you need to create in your descriptor class.
            return global_descriptor.describe_batch(images=image_batch)

        descriptors: list[np.ndarray] | None = None  # Will hold global descriptors if computed

        if self._global_descriptor is not None:
            logger.info("🟩 About to scatter descriptor")
            scatter_start = time.time()

            global_descriptor_future = client.scatter(self._global_descriptor, broadcast=True)

            logger.info(f"🟩 Scatter completed in {time.time()-scatter_start:.1f} seconds")

            # Submit descriptor extraction jobs for all images in parallel
            descriptor_futures: list[Future] = [
                client.submit(apply_global_descriptor_batch, global_descriptor_future, batch_future)
                for batch_future in image_batch_futures
            ]

            logger.info(f"⏳ Computing global descriptors for all images in batches of {self._batch_size}...")
            batched_descriptors = client.gather(descriptor_futures)

            # Flatten the batched results
            descriptors = [desc for batch in batched_descriptors for desc in batch]  # type: ignore

        # Use retriever to construct visibility graph based on descriptors and filenames
        logger.info("⏳ Computing visibility graph...")
        return self._retriever.get_image_pairs(
            global_descriptors=descriptors, image_fnames=image_fnames, plots_output_dir=plots_output_dir
        )
