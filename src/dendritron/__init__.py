"""Dendritron reference implementations.

The Dendritron is the primitive. Boolean compilation, plastic RBF webs,
mixed-geometry tissues, and Transformer memory packs are realizations of it.
"""

from .boolean import BooleanDendritron, ParityTissue, boolean_cube
from .branches import LocalBranch
from .geometry import Chart, Geometry, expmap0, poincare_distance
from .memory import PPCA, MemoryPack, MemoryRegistry
from .mixed_geometry import MixedGeometryWeb
from .plasticity import PlasticDendritronWeb
from .primitive import Dendritron
from .tissue import DendritronTissue
from .types import Certificate, LifecycleState, RecallMode

__all__ = [
    "BooleanDendritron",
    "Certificate",
    "Chart",
    "Dendritron",
    "DendritronTissue",
    "Geometry",
    "LifecycleState",
    "LocalBranch",
    "MemoryPack",
    "MemoryRegistry",
    "MixedGeometryWeb",
    "PPCA",
    "ParityTissue",
    "PlasticDendritronWeb",
    "RecallMode",
    "boolean_cube",
    "expmap0",
    "poincare_distance",
]

__version__ = "0.1.0"
