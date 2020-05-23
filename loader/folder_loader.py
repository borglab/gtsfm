""" Simple loader that reads from a folder on disk.

Authors: Frank Dellaert and Ayush Baid
"""

from loader.loader_base import LoaderBase


class FolderLoader(LoaderBase):
    """Simple loader class that reads from a folder on disk."""

    def __init__(self, folder: str):
        """Construct from input folder name."""
