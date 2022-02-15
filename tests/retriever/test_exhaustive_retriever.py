"""Unit tests for the exhaustive-matching retriever.

Authors: John Lambert
"""

from pathlib import Path

from dask.distributed import LocalCluster, Client

from gtsfm.loader.olsson_loader import OlssonLoader
from gtsfm.retriever.exhaustive_retriever import ExhaustiveRetriever

DATA_ROOT_PATH = Path(__file__).resolve().parent.parent / "data"
DOOR_DATA_ROOT = DATA_ROOT_PATH / "set1_lund_door"


def test_exhaustive_retriever_door() -> None:
    """Test the Exhaustive retriever on 12 frames of the Lund Door Dataset."""
    loader = OlssonLoader(folder=DOOR_DATA_ROOT, image_extension="JPG")
    retriever = ExhaustiveRetriever()

    # create dask client
    cluster = LocalCluster(n_workers=1, threads_per_worker=4)
    pairs_graph = retriever.create_computation_graph(loader=loader)
    with Client(cluster):
        pairs = pairs_graph.compute()

    # {12 \choose 2} = (12 * 11) / 2 = 66
    assert len(pairs) == 66

    for (i1, i2) in pairs:
        assert i1 < i2
