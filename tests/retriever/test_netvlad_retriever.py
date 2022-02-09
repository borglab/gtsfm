"""Unit tests for the NetVLAD retriever.

Authors: John Lambert
"""

from pathlib import Path

from gtsfm.loader.colmap_loader import ColmapLoader
from gtsfm.loader.olsson_loader import OlssonLoader
from gtsfm.retriever.netvlad_retriever import NetVLADRetriever

DATA_ROOT_PATH = Path(__file__).resolve().parent.parent / "data"
DOOR_DATA_ROOT = DATA_ROOT_PATH / "set1_lund_door"
SKYDIO_DATA_ROOT = DATA_ROOT_PATH / "crane_mast_8imgs_colmap_output"


def test_netvlad_retriever_crane_mast() -> None:
    """Test the NetVLAD retriever on 2 frames of the Skydio Crane-Mast dataset."""
    colmap_files_dirpath = SKYDIO_DATA_ROOT
    images_dir = SKYDIO_DATA_ROOT / "images"

    loader = ColmapLoader(
        colmap_files_dirpath=colmap_files_dirpath, images_dir=images_dir, max_frame_lookahead=100, max_resolution=760
    )
    num_images = 2

    retriever = NetVLADRetriever(num_matched=2)
    pairs = retriever.run(loader=loader)

    # only 1 pair possible between frame 0 and frame 1
    assert len(pairs) == 1
    assert pairs == [(0,1)]

def test_netvlad_retriever_door() -> None:
    """Test the NetVLAD retriever on 12 frames of the Lund Door Dataset."""
    loader = OlssonLoader(folder=DOOR_DATA_ROOT, image_extension="JPG")
    retriever = NetVLADRetriever(num_matched=2)
    pairs = retriever.run(loader=loader)

    assert len(pairs) == 21

    for (i1,i2) in pairs:

        assert i1 != i2
        assert i1 < i2


if __name__ == "__main__":
    """ """
    test_netvlad_retriever_crane_mast()
    test_netvlad_retriever_door()
