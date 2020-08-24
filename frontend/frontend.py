"""The front-end (detection-description-matching-verification) implementation.

This class combines the different components of front-ends and provides a
function to generate geometrically verified correspondences and pose-information
between pairs of images.

Authors: Ayush Baid
"""
import abc

from frontend.detector_descriptor.detector_descriptor_base import \
    DetectorDescriptorBase
from frontend.frontend_result import FrontEndResult
from frontend.matcher_verifier.matcher_verifier_base import MatcherVerifierBase
from loader.loader_base import LoaderBase


class FrontEnd(metaclass=abc.ABCMeta):
    """The complete front-end class (composed on different modules)."""

    def __init__(self,
                 detector_descriptor: DetectorDescriptorBase,
                 matcher_verifier: MatcherVerifierBase):
        """Initializes the front-end using different modules.

        Args:
            detector_descriptor (DetectorDescriptorBase): Detection-description
                                                          object
            matcher_verifier (MatcherVerifierBase): Matching-Verification object
        """
        self.detector_descriptor = detector_descriptor
        self.matcher_verifier = matcher_verifier

    def run(self, loader: LoaderBase, use_intrinsics: bool = False) -> FrontEndResult:
        """Runs the front-end for the loader.

        Args:
            loader (LoaderBase): The loader for the dataset on which the 
                                 front-end is to be run.
            use_intrinsics (bool, optional): flag to use intrinsics. Defaults 
                                             to False.

        Returns:
            FrontEndResult: the results of the front-end on the loader.
        """
        raise NotImplementedError
