"""
The base class for all front-ends
"""
import abc

from frontend.frontend_result import FrontEndResult
from loader.loader_base import LoaderBase


class FrontEndBase(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def run(self, loader: LoaderBase) -> FrontEndResult:
        """
        Runs the front-end for the loader.

        Args:
            loader (LoaderBase): The loader for the dataset on which the front-end is to be run.

        Returns: FrontEndResult:
        """
