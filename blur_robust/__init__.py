"""
blur_robust: Blur-robust pipeline for X-ray vertebra/pedicle detection.

Modules:
    deblur_preprocess   - Image deblurring and enhancement preprocessing
    blur_augmentation   - MMRotate-compatible blur augmentation transforms
    enhanced_config     - Enhanced ReDet configurations with blur augmentation
    inference_pipeline  - End-to-end inference with deblurring + ReDet detection
"""

from .deblur_preprocess import DeblurPreprocessor
from .enhanced_config import get_blur_robust_vertebra_config, get_blur_robust_pedicle_config

try:
    from .blur_augmentation import BlurAugmentation, AdaptiveSharpen
except ImportError:
    BlurAugmentation = None  # type: ignore[assignment]
    AdaptiveSharpen = None   # type: ignore[assignment]

try:
    from .inference_pipeline import VertebraPedicleDetector
except ImportError:
    VertebraPedicleDetector = None  # type: ignore[assignment]

__all__ = [
    'DeblurPreprocessor',
    'BlurAugmentation',
    'AdaptiveSharpen',
    'get_blur_robust_vertebra_config',
    'get_blur_robust_pedicle_config',
    'VertebraPedicleDetector',
]
