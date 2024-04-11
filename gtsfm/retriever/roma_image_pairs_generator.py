"""Generate image pairs for the frontend

Authors: Ayush Baid
"""

import itertools
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import PIL
import torch
from dask.distributed import Client, Future
from gtsfm.common.image import Image
from gtsfm.evaluation.metrics import GtsfmMetric, GtsfmMetricsGroup
from gtsfm.frontend.global_descriptor.global_descriptor_base import GlobalDescriptorBase
from gtsfm.retriever.retriever_base import RetrieverBase

sys.path.append("thirdparty/RoMa")
from thirdparty.RoMa.roma import roma_indoor, roma_outdoor


class RoMaImagePairsGenerator:
    def __init__(
        self,
        retriever: Optional[RetrieverBase] = None,
        global_descriptor: Optional[GlobalDescriptorBase] = None,
        use_outdoor_model: bool = True,
        use_cuda: bool = True,
        num_matched: int = 10,
        min_score: float = 0.2,
    ) -> None:
        self._retriever: RetrieverBase = retriever
        self._global_descriptor: Optional[GlobalDescriptorBase] = global_descriptor
        self._num_matched = num_matched
        self._min_score = min_score
        self._use_cuda = use_cuda
        self._use_outdoor_model = use_outdoor_model

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
        def compute_similarity_score(
            matcher, image_i1: Image, image_i2: Image
        ) -> float:
            # with torch.no_grad():
            #    H, W = matcher.upsample_res
            #    im1 = PIL.Image.fromarray(image_i1.value_array).convert("RGB")
            #    im2 = PIL.Image.fromarray(image_i2.value_array).convert("RGB")
            #    _, certainty = matcher.match(im1, im2, device=self._device)
            #    score = (
            #        torch.max(
            #            torch.mean(certainty[:, :W]), torch.mean(certainty[:, W:])
            #        )
            #        .cpu()
            #        .numpy()
            #    )
            score = 0.5
            print(score)
            return score

        # Initialize model.
        device = torch.device(
            "cuda" if self._use_cuda and torch.cuda.is_available() else "cpu"
        )
        if self._use_outdoor_model:
            matcher = roma_outdoor(device).eval()
        else:
            matcher = roma_indoor(device).eval()

        images = client.gather(images_future)

        # Compute similarity scores.
        similarity_scores = {}
        H, W = matcher.upsample_res
        for i1, i2 in itertools.combinations(np.arange(len(images)), 2):
            im1 = PIL.Image.fromarray(images[i1].value_array).convert("RGB")
            im2 = PIL.Image.fromarray(images[i2].value_array).convert("RGB")
            _, certainty = matcher.match(im1, im2, device=device)
            score = (
                torch.max(torch.mean(certainty[:, :W]), torch.mean(certainty[:, W:]))
                .cpu()
                .numpy()
            )
            print(score)
            similarity_scores[(i1, i2)] = score

        retrieved_matches = {i: [] for i in range(len(images))}
        retrieved_scores = {i: [] for i in range(len(images))}
        for (i1, i2), score in similarity_scores.items():
            if score > self._min_score:
                retrieved_matches[i1].append(i2)
                retrieved_scores[i1].append(score)
                retrieved_matches[i1].append(i2)
                retrieved_scores[i1].append(score)

        image_pairs = []
        named_image_pairs = []
        for i in range(len(images)):
            scores = np.array(retrieved_scores[i])
            image_ids = np.array(retrieved_matches[i])
            ordered_idx = np.argsort(scores)
            if len(scores) > self._num_matched:
                ordered_idx = ordered_idx[-self._num_matched :]
            image_ids = image_ids[ordered_idx]
            scores = scores[ordered_idx]
            retrieved_matches[i] = image_ids
            retrieved_scores[i] = scores
            for j in image_ids:
                image_pairs.append((i, j))
                named_image_pairs.append((image_fnames[i], image_fnames[j]))

        # Save named pairs and scores.
        with open(plots_output_dir / "netvlad_named_pairs.txt", "w") as fid:
            for _named_pair, _pair_ind in zip(named_image_pairs, image_pairs):
                fid.write(
                    "%.4f %s %s\n"
                    % (
                        similarity_scores[_pair_ind[0], _pair_ind[1]],
                        _named_pair[0],
                        _named_pair[1],
                    )
                )

        return image_pairs

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
