# Lazy, short-name exports for loader classes.

# Usage (Hydra/Python): _target_: gtsfm.loader.Colmap  |  from gtsfm.loader import Colmap
import importlib
from typing import TYPE_CHECKING

__all__ = [
    "Argoverse",
    "Astrovision",
    "Colmap",
    "Hilti",
    "Mobilebrick",
    "Olsson",
    "OneDSFM",
    "TanksAndTemples",
    "YfccImb",
]

# For type checkers / IDEs: provide real symbols without importing at runtime.
if TYPE_CHECKING:
    from .argoverse_dataset_loader import ArgoverseDatasetLoader as Argoverse
    from .astrovision_loader import AstrovisionLoader as Astrovision
    from .colmap_loader import ColmapLoader as Colmap
    from .hilti_loader import HiltiLoader as Hilti
    from .mobilebrick_loader import MobilebrickLoader as Mobilebrick
    from .olsson_loader import OlssonLoader as Olsson
    from .one_d_sfm_loader import OneDSFMLoader as OneDSFM
    from .tanks_and_temples_loader import TanksAndTemplesLoader as TanksAndTemples
    from .yfcc_imb_loader import YfccImbLoader as YfccImb

# Map short public names to (module, class) for lazy loading.
_MOD_MAP = {
    "Argoverse": ("gtsfm.loader.argoverse_dataset_loader", "ArgoverseDatasetLoader"),
    "Astrovision": ("gtsfm.loader.astrovision_loader", "AstrovisionLoader"),
    "Colmap": ("gtsfm.loader.colmap_loader", "ColmapLoader"),
    "Hilti": ("gtsfm.loader.hilti_loader", "HiltiLoader"),
    "Mobilebrick": ("gtsfm.loader.mobilebrick_loader", "MobilebrickLoader"),
    "Olsson": ("gtsfm.loader.olsson_loader", "OlssonLoader"),
    "OneDSFM": ("gtsfm.loader.one_d_sfm_loader", "OneDSFMLoader"),
    "TanksAndTemples": ("gtsfm.loader.tanks_and_temples_loader", "TanksAndTemplesLoader"),
    "YfccImb": ("gtsfm.loader.yfcc_imb_loader", "YfccImbLoader"),
}


def __getattr__(name: str):
    """Lazily import loaders on first access using short names."""
    try:
        module_name, class_name = _MOD_MAP[name]
    except KeyError as e:
        raise AttributeError(name) from e
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def __dir__():
    # so that help(), dir(), and IDEs show the short names
    return sorted(list(globals().keys()) + __all__)
