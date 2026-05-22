"""
TurboQuant Search
=================
Vector compression for similarity search, inspired by
TurboQuant (Zandieh et al., arXiv:2504.19874, 2025).

Uses random orthogonal rotation + Lloyd-Max quantization + sign-bit refinement.
"""

from .core import TurboQuantSearchIndex, IVFTurboQuantIndex, FlatSearchIndex, ProductQuantizationIndex
from .adaptive import AdaptiveIVFTurboQuantIndex
from .faiss_baselines import FAISS_AVAILABLE
from .benchmarks import run_benchmark, compute_recall

try:
    from . import _tqs_cpp
    TQS_CPP_AVAILABLE = True
except ImportError:
    _tqs_cpp = None
    TQS_CPP_AVAILABLE = False

if FAISS_AVAILABLE:
    from .faiss_baselines import FAISSFlatIndex, FAISSPQIndex, FAISSIVFPQIndex

__version__ = "0.3.0"
__all__ = [
    "TurboQuantSearchIndex",
    "IVFTurboQuantIndex",
    "AdaptiveIVFTurboQuantIndex",
    "FlatSearchIndex",
    "ProductQuantizationIndex",
    "FAISS_AVAILABLE",
    "TQS_CPP_AVAILABLE",
    "run_benchmark",
    "compute_recall",
]
