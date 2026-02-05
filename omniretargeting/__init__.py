"""OmniRetargeting: Generic motion retargeting for any humanoid URDF and terrain mesh."""

from .__version__ import __version__

__all__ = ["OmniRetargeter", "__version__"]


def __getattr__(name: str):
    if name == "OmniRetargeter":
        from .core import OmniRetargeter

        return OmniRetargeter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
