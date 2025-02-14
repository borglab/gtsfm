# Image Pairs Generator

![Alt text](gtsfm-overview-image-pairs-generator.svg?raw=true)

- [Loader](assets/LOADER.md)
- **Image Pairs Generator**
- [Correspondence Generator](assets/CORRESPONDENCE_GENERATOR.md)
- [Two View Estimator](assets/TWO_VIEW_ESTIMATOR.md)
- [Multiview Optimizer](assets/MULTIVIEW_OPTIMIZER.md)

## What is an Image Pair Generator?

The Image Pair Generator takes in images from the Loader and outputs putative image pairs for correspondence generation. Image pair generation is implemented by the [`ImagePairsGenerator`](https://github.com/borglab/gtsfm/blob/master/gtsfm/retriever/image_pairs_generator.py) class defined below, which wraps a specific [`Retriever`](https://github.com/borglab/gtsfm/blob/master/gtsfm/retriever/retriever_base.py), and, optionally, a [`GlobalDescriptor`](https://github.com/borglab/gtsfm/blob/master/gtsfm/frontend/global_descriptor/global_descriptor_base.py).

We support exhaustive, sequential, and descriptor-based image pair generation:
- **Exhaustive** matching compares every image pair, ensuring all possible matches are found but at a high computational cost.
- **Sequential** matching only compares neighboring images in a sequence (according to their image ID), making it faster but missing potentially valuable non-adjacent image pairs.
- **Descriptor-based** matching leverages global images descriptors to pre-filter and identify images pairs that have a higher probability of overlap, balancing efficiency with accuracy. This is discussed further in the [following section](#global-image-descriptors).


```python
class ImagePairsGenerator:
    def __init__(self, retriever: RetrieverBase, global_descriptor: Optional[GlobalDescriptorBase] = None):
        self._global_descriptor: Optional[GlobalDescriptorBase] = global_descriptor
        self._retriever: RetrieverBase = retriever

    def __repr__(self) -> str:
        return f"""
            ImagePairGenerator:
                {self._global_descriptor}
                {self._retriever}
        """

    def generate_image_pairs(
        self, client: Client, images: List[Future], image_fnames: List[str], plots_output_dir: Optional[Path] = None
    ) -> List[Tuple[int, int]]:
        def apply_global_descriptor(global_descriptor: GlobalDescriptorBase, image: Image) -> np.ndarray:
            return global_descriptor.describe(image=image)

        descriptors: Optional[List[np.ndarray]] = None
        if self._global_descriptor is not None:
            global_descriptor_future = client.scatter(self._global_descriptor, broadcast=False)

            descriptor_futures = [
                client.submit(apply_global_descriptor, global_descriptor_future, image) for image in images
            ]

            descriptors = client.gather(descriptor_futures)

        return self._retriever.get_image_pairs(
            global_descriptors=descriptors, image_fnames=image_fnames, plots_output_dir=plots_output_dir
        )
```

## Global Image Descriptors

Global desriptors work similar to local feature desriptors except that these methods generate a single descriptor for each image. Distances between these global image descriptors can then be used as a metric for the expected "matchability" of the image pairs during the correspondence generation phase, where a threshold can be used to reject potentially dissimilar image pairs before conducting correspondence generation. This reduces the likelihood of matching image pairs with little to no overlap that could cause erroneous correspondences to be inserted into the optimization process in the back-end while also significantly reducing the runtimes as compared to exhaustive matching. 

Global descriptor modules are implemented following the [`GlobalDescriptorBase`](https://github.com/borglab/gtsfm/blob/master/gtsfm/frontend/global_descriptor/global_descriptor_base.py) class and must be wrapped using a corresponding [`RetrieverBase`](https://github.com/borglab/gtsfm/blob/master/gtsfm/retriever/retriever_base.py) implementation, where the global descriptor module takes in individual images and outputs their corresponding descriptor and the retriever module takes these descriptors descriptors and computes the image pair similarity scores and outputs the putative image pairs based on a specified threshold (see [`NetVLADGlobalDescriptor`](https://github.com/borglab/gtsfm/blob/master/gtsfm/frontend/global_descriptor/netvlad_global_descriptor.py) and [`NetVLADRetriever`](https://github.com/borglab/gtsfm/blob/master/gtsfm/retriever/netvlad_retriever.py)).

```python
class RetrieverBase(GTSFMProcess):
    """Base class for image retriever implementations."""

    def __init__(self, matching_regime: ImageMatchingRegime) -> None:
        """
        Args:
            matching_regime: identifies type of matching used for image retrieval, e.g., exhaustive, descriptor-based.
        """
        self._matching_regime = matching_regime

    @abc.abstractmethod
    def get_image_pairs(
        self,
        global_descriptors: Optional[List[np.ndarray]],
        image_fnames: List[str],
        plots_output_dir: Optional[Path] = None,
    ) -> List[Tuple[int, int]]:
        """Compute potential image pairs.

        Args:
            global_descriptors: the global descriptors for the retriever, if needed.
            image_fnames: file names of the images
            plots_output_dir: Directory to save plots to. If None, plots are not saved.

        Returns:
            List of (i1,i2) image pairs.
        """
```

```python
class GlobalDescriptorBase:
    """Base class for all the global image descriptors.

    Global image descriptors assign a vector for each input image.
    """

    @abc.abstractmethod
    def describe(self, image: Image) -> np.ndarray:
        """Compute the global descriptor for a single image query.

        Args:
            image: input image.

        Returns:
            img_desc: array of shape (D,) representing global image descriptor.
        """

```

<details><summary>Supported Global Descriptors</summary>
<ul>
  <li><strong>NetVLAD</strong>, R. Arandjelovic <em>et al.</em>, CVPR 2016. <a href="https://arxiv.org/pdf/1511.07247.pdf">[paper]</a> <a href="https://github.com/borglab/gtsfm/blob/master/gtsfm/frontend/global_descriptor/netvlad_global_descriptor.py">[code]</a></li>
</ul>
</details>
