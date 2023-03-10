"""Detector for the front end.

Authors: Ayush Baid
"""
import abc
import pickle
from pathlib import Path

import dask
from dask.distributed import Client, LocalCluster, performance_report
import numpy as np
import torch

import gtsfm.utils.images as image_utils
from gtsfm.common.image import Image
from gtsfm.common.keypoints import Keypoints
from gtsfm.loader.olsson_loader import OlssonLoader
from gtsfm.ui.gtsfm_process import GTSFMProcess, UiMetadata
from thirdparty.SuperGluePretrainedNetwork.models.superpoint import SuperPoint

ROOT_PATH = Path(__file__).resolve().parent.parent
MODEL_WEIGHTS_PATH = (
    ROOT_PATH / "thirdparty" / "SuperGluePretrainedNetwork" / "models" / "weights" / "superpoint_v1.pth"
)
TEST_DATA_PATH = ROOT_PATH / "tests" / "data" / "set1_lund_door"


class NewDetector(GTSFMProcess):
    @staticmethod
    def get_ui_metadata() -> UiMetadata:
        """Returns data needed to display node and edge info for this process in the process graph."""

        return UiMetadata(
            display_name="NewDetector",
            input_products=("Images",),
            output_products=("Keypoints",),
            parent_plate="DetDescCorrespondenceGenerator",
        )

    @abc.abstractmethod
    def apply(self, image: Image) -> Keypoints:
        """Apply the detector on the image to produce keypoints.

        Args:
            image: input image.

        Returns:
            detected keypoints, with maximum length of max_keypoints.
        """


class NewSuperpointDetector(NewDetector):
    def __init__(self):
        super(NewSuperpointDetector, self).__init__()
        # init the opencv object
        self._model = SuperPoint({"weights_path": MODEL_WEIGHTS_PATH}).cpu()
        self._model.eval()

    def apply(self, image: Image) -> Keypoints:
        # Compute features.
        image_tensor = torch.from_numpy(
            np.expand_dims(image_utils.rgb_to_gray_cv(image).value_array.astype(np.float32) / 255.0, (0, 1))
        )
        with torch.no_grad():
            model_results = self._model({"image": image_tensor})
        torch.cuda.empty_cache()

        # Unpack results.
        coordinates = model_results["keypoints"][0].detach().cpu().numpy()
        scores = model_results["scores"][0].detach().cpu().numpy()
        keypoints = Keypoints(coordinates, scales=None, responses=scores)
        # descriptors = model_results["descriptors"][0].detach().cpu().numpy().T

        return keypoints


@dask.delayed
def detect(detector: NewDetector, image: Image) -> Keypoints:
    return detector.apply(image)


if __name__ == "__main__":

    loader = OlssonLoader(str(TEST_DATA_PATH), image_extension="JPG")
    detector = NewSuperpointDetector()

    print("Serialized detector size: {}".format(dask.utils.format_bytes(len(pickle.dumps(detector)))))

    # create dask client
    cluster = LocalCluster(n_workers=2, threads_per_worker=1)

    delayed_images = loader.create_computation_graph_for_images()

    with Client(cluster) as client, performance_report(filename="retriever-dask-report.html"):
        detector_future = client.scatter(detector, broadcast=True)

        delayed_keypoints = [detect(detector_future, delayed_image) for delayed_image in delayed_images]
        keypoints_future = dask.compute(*delayed_keypoints)

        for i, kp in enumerate(keypoints_future):
            print("Idx {}: {} keypoints".format(i, len(kp)))