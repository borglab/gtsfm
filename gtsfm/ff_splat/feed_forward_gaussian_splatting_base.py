"""Base class for feed forward Gaussian splatting.
Authors: Harneet Singh Khanuja
"""

from abc import abstractmethod
from pathlib import Path
from typing import List

from gtsfm.common.image import Image


class FeedForwardGaussianSplattingBase:
    """Base class for feed forward Gaussian splatting"""

    @abstractmethod
    def generate_splats(
        self,
        images: List[Image],
        save_gs_files_path: Path,
    ) -> None:
        """Apply feed forward Gaussian inference to generate Gaussian splats.
        Args:
            images: List of all images.
            save_gs_files_path: Path to save ply file and interpolated video with all Gaussians
        Returns:
            None
        """
