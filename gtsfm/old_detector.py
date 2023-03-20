"""Detector for the front end.

Authors: Ayush Baid
"""
import abc
import pickle
from pathlib import Path
from timeit import default_timer as timer

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
from gtsfm.frontend.detector_descriptor.superpoint import SuperPointDetectorDescriptor
from gtsfm.frontend.detector.detector_from_joint_detector_descriptor import DetectorFromDetectorDescriptor

ROOT_PATH = Path(__file__).resolve().parent.parent
MODEL_WEIGHTS_PATH = (
    ROOT_PATH / "thirdparty" / "SuperGluePretrainedNetwork" / "models" / "weights" / "superpoint_v1.pth"
)
TEST_DATA_PATH = ROOT_PATH / "tests" / "data" / "set1_lund_door"


if __name__ == "__main__":

    loader = OlssonLoader(str(TEST_DATA_PATH), image_extension="JPG")
    detector = DetectorFromDetectorDescriptor(SuperPointDetectorDescriptor())

    print("Serialized detector size: {}".format(dask.utils.format_bytes(len(pickle.dumps(detector)))))

    # create dask client
    cluster = LocalCluster(n_workers=1, threads_per_worker=1)

    delayed_images = loader.create_computation_graph_for_images_hack(num_images=100)

    with Client(cluster) as client, performance_report(filename="old-detector-dask-report.html"):

        start_time = timer()

        delayed_keypoints = [
            detector.create_computation_graph(x)
            for x in loader.create_computation_graph_for_images_hack(num_images=100)
        ]
        keypoints_future = dask.compute(*delayed_keypoints)

        end_time = timer()

        print("Time elapsed: {}".format(end_time - start_time))
