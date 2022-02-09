"""Unit tests for the NetVLAD retriever.

Authors: John Lambert
"""

from gtsfm.loader.colmap_loader import ColmapLoader
from gtsfm.retriever.netvlad_retriever import NetVLADRetriever

def test_deep_retriever() -> None:
    """ """
    colmap_files_dirpath = "/Users/jlambert/Downloads/skydio-501-colmap-pseudo-gt"
    images_dir = "/Users/jlambert/Downloads/skydio-501-images/skydio-crane-mast-501-images1"

    loader = ColmapLoader(
        colmap_files_dirpath=colmap_files_dirpath, images_dir=images_dir, max_frame_lookahead=100, max_resolution=760
    )
    num_images = 250

    retriever = NetVLADRetriever()
    retriever.run(loader=loader, num_images=num_images)


if __name__ == "__main__":
    """ """
    test_deep_retriever()
