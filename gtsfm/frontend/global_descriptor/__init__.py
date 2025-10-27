# Lazy, short-name exports for global descriptor classes.

# Usage (Hydra/Python):
# Hydra: _target_: gtsfm.frontend.global_descriptor.NetVLAD
# Python: from gtsfm.frontend.global_descriptor import NetVLAD
import importlib
from typing import TYPE_CHECKING

__all__ = ["MegaLoc", "NetVLAD"]

# For type checkers / IDEs: provide real symbols without importing at runtime.
if TYPE_CHECKING:
    from .megaloc_global_descriptor import MegaLocGlobalDescriptor as MegaLoc
    from .netvlad_global_descriptor import NetVLADGlobalDescriptor as NetVLAD

# Map short public names to (module, class) for lazy loading.
_MOD_MAP = {
    "MegaLoc": ("gtsfm.frontend.global_descriptor.megaloc_global_descriptor", "MegaLocGlobalDescriptor"),
    "NetVLAD": ("gtsfm.frontend.global_descriptor.netvlad_global_descriptor", "NetVLADGlobalDescriptor"),
}


def __getattr__(name: str):
    """Lazily import global descriptor classes on first access using short names."""
    try:
        module_name, class_name = _MOD_MAP[name]
    except KeyError as e:
        raise AttributeError(name) from e
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def __dir__():
    # so that help(), dir(), and IDEs show the short names
    return sorted(list(globals().keys()) + __all__)
