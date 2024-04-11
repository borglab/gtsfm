"""Generate image pairs for the frontend

Authors: Ayush Baid
"""

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from dask.distributed import Client, Future
from gtsfm.common.image import Image
from gtsfm.evaluation.metrics import GtsfmMetric, GtsfmMetricsGroup
from gtsfm.frontend.global_descriptor.global_descriptor_base import GlobalDescriptorBase
from gtsfm.retriever.retriever_base import RetrieverBase


class ImagePairsGenerator:
    def __init__(
        self,
        retriever: RetrieverBase,
        global_descriptor: Optional[GlobalDescriptorBase] = None,
    ) -> None:
        self._global_descriptor: Optional[GlobalDescriptorBase] = global_descriptor
        self._retriever: RetrieverBase = retriever

    def __repr__(self) -> str:
        return f"""
            ImagePairGenerator:
                {self._global_descriptor}
                {self._retriever}
        """

    def generate_image_pairs(
        self,
        client: Client,
        images_future: List[Future],
        image_fnames: List[str],
        plots_output_dir: Optional[Path] = None,
    ) -> List[Tuple[int, int]]:
        def apply_global_descriptor(
            global_descriptor: GlobalDescriptorBase, image: Image
        ) -> np.ndarray:
            return global_descriptor.describe(image=image)

        descriptors: Optional[List[np.ndarray]] = None
        if self._global_descriptor is not None:
            global_descriptor_future = client.scatter(
                self._global_descriptor, broadcast=False
            )

            descriptor_futures = [
                client.submit(apply_global_descriptor, global_descriptor_future, image)
                for image in images_future
            ]

            descriptors = client.gather(descriptor_futures)

        return self._retriever.get_image_pairs(
            global_descriptors=descriptors,
            image_fnames=image_fnames,
            plots_output_dir=plots_output_dir,
        )

    @staticmethod
    def evaluate(
        num_images, image_pair_indices: List[Tuple[int, int]]
    ) -> GtsfmMetricsGroup:
        """Evaluates the retriever result.

        Args:
            num_images: the number of images in the dataset.
            image_pair_indices: (i1,i2) image pairs.

        Returns:
            Retriever metrics group.
        """
        metric_group_name = "retriever_metrics"
        retriever_metrics = GtsfmMetricsGroup(
            metric_group_name,
            [
                GtsfmMetric("num_input_images", num_images),
                GtsfmMetric("num_retrieved_image_pairs", len(image_pair_indices)),
            ],
        )
        return retriever_metrics
