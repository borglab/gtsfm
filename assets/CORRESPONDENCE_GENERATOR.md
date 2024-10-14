# Correspondence Generator

![Alt text](gtsfm-overview-correspondence-generator.svg?raw=true)

## What is a Correspondence Generator?

The Correspondence Generator is responsible for taking in putative image pairs from the [`ImagePairsGenerator`](https://github.com/borglab/gtsfm/blob/master/gtsfm/retriever/image_pairs_generator.py) and returning keypoints for each image and correspondences between each specified image pair. Correspondence generation is implemented by the [`CorrespondenceGeneratorBase`](https://github.com/borglab/gtsfm/blob/master/gtsfm/frontend/correspondence_generator/correspondence_generator_base.py) class defined below.

```python
class CorrespondenceGeneratorBase:
    """Base class for correspondence generators."""

    @abstractmethod
    def generate_correspondences(
        self,
        client: Client,
        images: List[Future],
        image_pairs: List[Tuple[int, int]],
    ) -> Tuple[List[Keypoints], Dict[Tuple[int, int], np.ndarray]]:
        """Apply the correspondence generator to generate putative correspondences.

        Args:
            client: Dask client, used to execute the front-end as futures.
            images: List of all images, as futures.
            image_pairs: Indices of the pairs of images to estimate two-view pose and correspondences.

        Returns:
            List of keypoints, one entry for each input images.
            Putative correspondence as indices of keypoints, for pairs of images.
        """
```

## Types of Correspondence Generators

We provide support for two correspondence generation paradigms: [feature extraction _then_ matching](#feature-detection-and-description-then-matching) and [_detector-free_ matching](#detector-free-matching). The supported Correspondence Generator paradigms are detailed below.


### Feature Detection and Description _then_ Matching

This paradigm jointly computes feature detections and descriptors, typically using shared weights in a deep convolutional neural network, followed by feature matching. This is implemented by the [`DetDescCorrespondenceGenerator`](https://github.com/borglab/gtsfm/blob/master/gtsfm/frontend/correspondence_generator/det_desc_correspondence_generator.py) class, which wraps a feature detector and descriptor ([`DetectorDescriptorBase`](https://github.com/borglab/gtsfm/blob/master/gtsfm/frontend/detector_descriptor/detector_descriptor_base.py)) and a feature matcher ([`MatcherBase`](https://github.com/borglab/gtsfm/blob/master/gtsfm/frontend/matcher/matcher_base.py)).

The feature detector and descriptor module takes in a single image and outputs keypoints and feature descriptors. Joint detection and description is implemented by the [`DetectorDescriptorBase`](https://github.com/borglab/gtsfm/blob/master/gtsfm/frontend/detector_descriptor/detector_descriptor_base.py) class defined below. We also provide functionality for combining different keypoint detection and feature description modules to form a joint detector-descriptor module (see [`CombinationDetectorDescriptor`](https://github.com/borglab/gtsfm/blob/master/gtsfm/frontend/detector_descriptor/combination_detector_descriptor.py)). To create your own feature extractor, simply copy the contents of `detector_descriptor_base.py` to a new file corresponding to the new extractor's class name and implement the `detect_and_describe` method. 

The feature matcher takes in the keypoints and descriptors for image images and outputs indices for matching keypoints. Feature matching is implemented by the [`MatcherBase`](https://github.com/borglab/gtsfm/blob/master/gtsfm/frontend/matcher/matcher_base.py) class defined below. To create your own feature matcher, simply copy the contents of `matcher_base.py` to a new file corresponding to the new matcher's class name and implement the `match` method.

```python
class DetectorDescriptorBase(GTSFMProcess):
    """Base class for all methods which provide a joint detector-descriptor to work on a single image."""

    def __init__(self, max_keypoints: int = 5000):
        """Initialize the detector-descriptor.

        Args:
            max_keypoints: Maximum number of keypoints to detect. Defaults to 5000.
        """
        self.max_keypoints = max_keypoints

    @abc.abstractmethod
    def detect_and_describe(self, image: Image) -> Tuple[Keypoints, np.ndarray]:
        """Perform feature detection as well as their description.

        Refer to detect() in DetectorBase and describe() in DescriptorBase for
        details about the output format.

        Args:
            image: the input image.

        Returns:
            Detected keypoints, with length N <= max_keypoints.
            Corr. descriptors, of shape (N, D) where D is the dimension of each descriptor.
        """
```

```python
class MatcherBase(GTSFMProcess):
    """Base class for all matchers."""

    @abc.abstractmethod
    def match(
        self,
        keypoints_i1: Keypoints,
        keypoints_i2: Keypoints,
        descriptors_i1: np.ndarray,
        descriptors_i2: np.ndarray,
        im_shape_i1: Tuple[int, int, int],
        im_shape_i2: Tuple[int, int, int],
    ) -> np.ndarray:
        """Match descriptor vectors.

        # Some matcher implementations (such as SuperGlue) utilize keypoint coordinates as
        # positional encoding, so our matcher API provides them for optional use.

        Output format:
        1. Each row represents a match.
        2. First column represents keypoint index from image #i1.
        3. Second column represents keypoint index from image #i2.
        4. Matches are sorted in descending order of the confidence (score), if possible.

        Args:
            keypoints_i1: keypoints for image #i1, of length N1.
            keypoints_i2: keypoints for image #i2, of length N2.
            descriptors_i1: descriptors corr. to keypoints_i1.
            descriptors_i2: descriptors corr. to keypoints_i2.
            im_shape_i1: shape of image #i1, as (height,width,channel).
            im_shape_i2: shape of image #i2, as (height,width,channel).


        Returns:
            Match indices (sorted by confidence), as matrix of shape (N, 2), where N < min(N1, N2).
        """
```

<details><summary>Supported Feature Detectors & Descriptors</summary>
<ul>
  <li><strong>SIFT</strong>, D. G. Lowe, IJCV 2004. <a href="https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf">[paper]</a> <a href="https://github.com/borglab/gtsfm/blob/master/gtsfm/frontend/detector_descriptor/sift.py">[code]</a></li>
  <li><strong>BRISK</strong>, S. Leutenegger, <em>et al.</em>, ICCV 2011. <a href="https://margaritachli.com/papers/ICCV2011paper.pdf">[paper]</a> <a href="https://github.com/borglab/gtsfm/blob/master/gtsfm/frontend/detector_descriptor/brisk.py">[code]</a></li>
  <li><strong>ORB</strong>, E. Rublee <em>et al.</em>, ICCV 2011. <a href="https://ieeexplore.ieee.org/document/6126544">[paper]</a> <a href="https://github.com/borglab/gtsfm/blob/master/gtsfm/frontend/detector_descriptor/orb.py">[code]</a></li>
  <li><strong>KAZE</strong>, P. F. Alcantarilla <em>et al.</em>, ECCV 2012. <a href="https://link.springer.com/chapter/10.1007/978-3-642-33783-3_16">[paper]</a> <a href="https://github.com/borglab/gtsfm/blob/master/gtsfm/frontend/detector_descriptor/kaze.py">[code]</a></li>
  <li><strong>SuperPoint</strong>, D. DeTone <em>et al.</em>, CVPRW 2018. <a href="https://openaccess.thecvf.com/content_cvpr_2018_workshops/papers/w9/DeTone_SuperPoint_Self-Supervised_Interest_CVPR_2018_paper.pdf">[paper]</a> <a href="https://github.com/borglab/gtsfm/blob/master/gtsfm/frontend/detector_descriptor/superpoint.py">[code]</a></li>
  <li><strong>D2-Net</strong>, M. Dusmanu <em>et al.</em>, CVPR 2019. <a href="https://arxiv.org/abs/1905.03561">[paper]</a> <a href="https://github.com/borglab/gtsfm/blob/master/gtsfm/frontend/detector_descriptor/d2net.py">[code]</a></li>
  <li><strong>DISK</strong>, M. Tyszkiewicz <em>et al.</em>, NeurIPS 2020. <a href="https://proceedings.neurips.cc/paper/2020/file/a42a596fc71e17828440030074d15e74-Paper.pdf">[paper]</a> <a href="https://github.com/borglab/gtsfm/blob/master/gtsfm/frontend/detector_descriptor/disk.py">[code]</a></li>
</ul>
</details>

<details><summary>Supported Feature Matchers</summary>
<ul>
  <li><strong>Mutual Nearest Neighbors (MNN)</strong></li>
  <li><strong>SuperGlue (trained for SuperPoint)</strong>, P.-E. Sarlin <em>et al.</em>, CVPR 2020. <a href="http://openaccess.thecvf.com/content_CVPR_2020/papers/Sarlin_SuperGlue_Learning_Feature_Matching_With_Graph_Neural_Networks_CVPR_2020_paper.pdf">[paper]</a> <a href="https://github.com/borglab/gtsfm/blob/master/gtsfm/frontend/matcher/superglue_matcher.py">[code]</a></li>
  <li><strong>LightGlue (trained for SuperPoint and DISK)</strong>, P. Lindenberger <em>et al.</em>, CVPR 2023. <a href="https://openaccess.thecvf.com/content/ICCV2023/papers/Lindenberger_LightGlue_Local_Feature_Matching_at_Light_Speed_ICCV_2023_paper.pdf">[paper]</a> <a href="https://github.com/borglab/gtsfm/blob/master/gtsfm/frontend/matcher/lightglue_matcher.py">[code]</a></li>
</ul>
</details>


### Detector-Free Matching 

This paradigm directly regresses _per-pixel_ matches between two input images as opposed to generating detections for each image followed by matching. This is implemented by the [`ImageCorrespondenceGenerator`](https://github.com/borglab/gtsfm/blob/master/gtsfm/frontend/correspondence_generator/image_correspondence_generator.py) class, which siply wraps an detector-free matcher ([`ImageMatcherBase`](https://github.com/borglab/gtsfm/blob/master/gtsfm/frontend/matcher/image_matcher_base.py)). 

Detector-free matching is implemented by the `ImageMatcherBase` class defined below. To create your own detector-free matcher, simply copy the contents of `image_matcher_base.py` to a new file corresponding to the new matcher's class name and implement the `match` method.

```python
class ImageMatcherBase(GTSFMProcess):
    """Base class for matchers that accept an image pair, and directly generate keypoint matches.

    Note: these matchers do NOT use descriptors as input.
    """

    @abc.abstractmethod
    def match(
        self,
        image_i1: Image,
        image_i2: Image,
    ) -> Tuple[Keypoints, Keypoints]:
        """Identify feature matches across two images.

        Args:
            image_i1: first input image of pair.
            image_i2: second input image of pair.

        Returns:
            Keypoints from image 1 (N keypoints will exist).
            Corresponding keypoints from image 2 (there will also be N keypoints). These represent feature matches.
        """
```

<details><summary>Supported Detector-Free Matchers</summary>
<ul>
  <li><strong>LoFTR</strong>, J. Sun, Z. Shen, Y. Wang, <em>et al.</em>, CVPR 2021. <a href="https://zju3dv.github.io/loftr/">[paper]</a> <a href="https://github.com/borglab/gtsfm/blob/master/gtsfm/frontend/matcher/loftr.py">[code]</a></li>
  <li><strong>DKM</strong>, J. Edstedt <em>et al.</em>, CVPR 2021. <a href="https://parskatt.github.io/DKM/">[paper]</a> <a href="https://github.com/borglab/gtsfm/blob/master/gtsfm/frontend/matcher/dkm.py">[code]</a></li>
  <li><strong>RoMa</strong>, J. Edstedt <em>et al.</em>, CVPR 2024. <a href="https://parskatt.github.io/RoMa/">[paper]</a> <a href="https://github.com/borglab/gtsfm/blob/master/gtsfm/frontend/matcher/roma.py">[code]</a></li>
</ul>
</details>
