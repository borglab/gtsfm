"""Unit tests for the NetVLAD retriever.

Authors: John Lambert
"""

from pathlib import Path

from gtsfm.loader.colmap_loader import ColmapLoader
from gtsfm.retriever.netvlad_retriever import NetVLADRetriever

SKYDIO_DATA_ROOT = Path(__file__).resolve().parent.parent / "data" / "crane_mast_8imgs_colmap_output"


def test_netvlad_retriever() -> None:
    """ """
    colmap_files_dirpath = SKYDIO_DATA_ROOT
    images_dir = SKYDIO_DATA_ROOT / "images"

    loader = ColmapLoader(
        colmap_files_dirpath=colmap_files_dirpath, images_dir=images_dir, max_frame_lookahead=100, max_resolution=760
    )
    num_images = 2

    retriever = NetVLADRetriever()
    retriever.run(loader=loader, num_images=num_images)


if __name__ == "__main__":
    """ """
    test_netvlad_retriever()
