"""
Core 3D Reconstruction Module
Clean implementation of Structure from Motion pipeline
"""

from .camera import Camera, CameraPose, load_calibration
from .features import FeatureExtractor, FeatureMatcher
from .geometry import triangulate_points, compute_essential_matrix, decompose_essential
from .sfm_pipeline import SfMPipeline
from .dense import DenseReconstructor
from .dense_stereo import DenseStereoReconstructor, create_combined_dense_cloud

# Optional neural matcher (requires LightGlue)
try:
    from .neural_matcher import NeuralMatcher
    _has_neural = True
except ImportError:
    _has_neural = False

__all__ = [
    'Camera',
    'CameraPose',
    'load_calibration', 
    'FeatureExtractor',
    'FeatureMatcher',
    'triangulate_points',
    'compute_essential_matrix',
    'decompose_essential',
    'SfMPipeline',
    'DenseReconstructor',
    'DenseStereoReconstructor',
    'create_combined_dense_cloud'
]

if _has_neural:
    __all__.append('NeuralMatcher')