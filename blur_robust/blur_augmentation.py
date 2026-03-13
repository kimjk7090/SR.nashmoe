"""
blur_augmentation.py
MMRotate-compatible custom data augmentation transforms for blur robustness.

Transforms
----------
BlurAugmentation
    Registered as ``mmdet.BlurAugmentation``.
    Randomly applies one of:
    - MotionBlur  (random angle + kernel size)
    - GaussianBlur
    - MedianBlur
    - Defocus blur (disk kernel)

AdaptiveSharpen
    Registered as ``mmdet.AdaptiveSharpen``.
    Measures the blur level of the input image and applies proportional
    CLAHE + Unsharp Masking (more sharpening for blurrier images).

Both transforms follow the MMRotate/MMCV transform protocol:
    - decorated with ``@TRANSFORMS.register_module()``
    - implement ``transform(results: dict) -> dict``
    - read / write ``results['img']`` (BGR uint8 numpy array)
"""

import random
import numpy as np
import cv2

from mmdet.registry import TRANSFORMS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _motion_blur_kernel(size: int, angle: float) -> np.ndarray:
    """Build a motion blur convolution kernel."""
    kernel = np.zeros((size, size), dtype=np.float32)
    center = size // 2
    end_x = int(center + center * np.cos(np.radians(angle)))
    end_y = int(center + center * np.sin(np.radians(angle)))
    end_x = np.clip(end_x, 0, size - 1)
    end_y = np.clip(end_y, 0, size - 1)
    cv2.line(kernel, (center, center), (end_x, end_y), 1.0, 1)
    total = kernel.sum()
    if total > 0:
        kernel /= total
    return kernel


def _defocus_kernel(size: int) -> np.ndarray:
    """Build a disk-shaped (defocus) blur kernel."""
    kernel = np.zeros((size, size), dtype=np.float32)
    center = size // 2
    cv2.circle(kernel, (center, center), center, 1.0, -1)
    total = kernel.sum()
    if total > 0:
        kernel /= total
    return kernel


def _laplacian_variance(image: np.ndarray) -> float:
    """Return Laplacian variance (blur metric) for an image."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    return float(cv2.Laplacian(gray.astype(np.float64), cv2.CV_64F).var())


# ---------------------------------------------------------------------------
# BlurAugmentation
# ---------------------------------------------------------------------------

@TRANSFORMS.register_module(name='BlurAugmentation', force=True)
class BlurAugmentation:
    """Randomly apply one type of blur to teach the model to handle blurry input.

    Blur types:
        - ``'motion'``   : directional motion blur (random angle).
        - ``'gaussian'`` : isotropic Gaussian blur.
        - ``'median'``   : median blur (good for salt-and-pepper noise).
        - ``'defocus'``  : disk-shaped defocus blur.

    Args:
        prob (float): Probability of applying any augmentation. Default: 0.5.
        blur_types (list[str]): Blur types to sample from.
            Default: all four types.
        kernel_size_range (tuple[int, int]): Minimum and maximum kernel size
            (both must be odd). Default: (3, 9).
        motion_angle_range (tuple[float, float]): Range of motion blur angles
            in degrees. Default: (0.0, 360.0).
        gaussian_sigma_range (tuple[float, float]): Sigma range for Gaussian
            blur. Default: (0.5, 2.0).
    """

    def __init__(
        self,
        prob: float = 0.5,
        blur_types=None,
        kernel_size_range=(3, 9),
        motion_angle_range=(0.0, 360.0),
        gaussian_sigma_range=(0.5, 2.0),
    ):
        if blur_types is None:
            blur_types = ['motion', 'gaussian', 'median', 'defocus']
        self.prob = prob
        self.blur_types = blur_types
        self.kernel_size_range = kernel_size_range
        self.motion_angle_range = motion_angle_range
        self.gaussian_sigma_range = gaussian_sigma_range

    def _random_odd_kernel(self) -> int:
        lo, hi = self.kernel_size_range
        k = random.randint(lo, hi)
        return k if k % 2 == 1 else k + 1

    def _apply_motion_blur(self, image: np.ndarray) -> np.ndarray:
        k = self._random_odd_kernel()
        angle = random.uniform(*self.motion_angle_range)
        kernel = _motion_blur_kernel(k, angle)
        return cv2.filter2D(image, -1, kernel)

    def _apply_gaussian_blur(self, image: np.ndarray) -> np.ndarray:
        k = self._random_odd_kernel()
        sigma = random.uniform(*self.gaussian_sigma_range)
        return cv2.GaussianBlur(image, (k, k), sigma)

    def _apply_median_blur(self, image: np.ndarray) -> np.ndarray:
        k = self._random_odd_kernel()
        k = max(k, 3)
        return cv2.medianBlur(image, k)

    def _apply_defocus_blur(self, image: np.ndarray) -> np.ndarray:
        k = self._random_odd_kernel()
        kernel = _defocus_kernel(k)
        return cv2.filter2D(image, -1, kernel)

    def transform(self, results: dict) -> dict:
        if random.random() > self.prob:
            return results

        image = results['img']
        blur_type = random.choice(self.blur_types)

        if blur_type == 'motion':
            image = self._apply_motion_blur(image)
        elif blur_type == 'gaussian':
            image = self._apply_gaussian_blur(image)
        elif blur_type == 'median':
            image = self._apply_median_blur(image)
        elif blur_type == 'defocus':
            image = self._apply_defocus_blur(image)

        results['img'] = image
        return results

    def __repr__(self) -> str:
        return (
            f'{self.__class__.__name__}('
            f'prob={self.prob}, '
            f'blur_types={self.blur_types}, '
            f'kernel_size_range={self.kernel_size_range})'
        )


# ---------------------------------------------------------------------------
# AdaptiveSharpen
# ---------------------------------------------------------------------------

@TRANSFORMS.register_module(name='AdaptiveSharpen', force=True)
class AdaptiveSharpen:
    """Measure input blur and apply proportional CLAHE + Unsharp Masking.

    Blurrier images receive stronger sharpening.  Sharp images receive only
    light enhancement or none at all.

    Blur levels (Laplacian variance thresholds):
        - var ≥ sharp_thr   → no sharpening
        - var ≥ mild_thr    → light sharpening
        - var ≥ moderate_thr→ moderate sharpening
        - var < moderate_thr→ strong sharpening

    Args:
        sharp_thr (float): Laplacian variance above which image is considered
            sharp and no processing is applied. Default: 100.0.
        mild_thr (float): Threshold for mild blur. Default: 50.0.
        moderate_thr (float): Threshold for moderate blur. Default: 15.0.
        clahe_clip_sharp (float): CLAHE clip limit for sharp images.
            Default: 1.0.
        clahe_clip_mild (float): CLAHE clip limit for mild blur. Default: 1.5.
        clahe_clip_moderate (float): CLAHE clip limit for moderate blur.
            Default: 2.0.
        clahe_clip_severe (float): CLAHE clip limit for severe blur.
            Default: 3.0.
        clahe_tile_grid (tuple[int, int]): CLAHE tile grid size. Default: (8, 8).
        unsharp_amount_mild (float): Unsharp mask amount for mild blur.
            Default: 0.5.
        unsharp_amount_moderate (float): Unsharp mask amount for moderate blur.
            Default: 1.0.
        unsharp_amount_severe (float): Unsharp mask amount for severe blur.
            Default: 1.5.
        unsharp_radius (int): Gaussian blur radius for unsharp mask.
            Default: 5.
    """

    def __init__(
        self,
        sharp_thr: float = 100.0,
        mild_thr: float = 50.0,
        moderate_thr: float = 15.0,
        clahe_clip_sharp: float = 1.0,
        clahe_clip_mild: float = 1.5,
        clahe_clip_moderate: float = 2.0,
        clahe_clip_severe: float = 3.0,
        clahe_tile_grid=(8, 8),
        unsharp_amount_mild: float = 0.5,
        unsharp_amount_moderate: float = 1.0,
        unsharp_amount_severe: float = 1.5,
        unsharp_radius: int = 5,
    ):
        self.sharp_thr = sharp_thr
        self.mild_thr = mild_thr
        self.moderate_thr = moderate_thr
        self.clahe_clip_sharp = clahe_clip_sharp
        self.clahe_clip_mild = clahe_clip_mild
        self.clahe_clip_moderate = clahe_clip_moderate
        self.clahe_clip_severe = clahe_clip_severe
        self.clahe_tile_grid = tuple(clahe_tile_grid)
        self.unsharp_amount_mild = unsharp_amount_mild
        self.unsharp_amount_moderate = unsharp_amount_moderate
        self.unsharp_amount_severe = unsharp_amount_severe
        self.unsharp_radius = unsharp_radius

    def _apply_clahe(self, image: np.ndarray, clip_limit: float) -> np.ndarray:
        clahe = cv2.createCLAHE(
            clipLimit=clip_limit,
            tileGridSize=self.clahe_tile_grid,
        )
        if image.ndim == 2:
            return clahe.apply(image)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_ch, a_ch, b_ch = cv2.split(lab)
        l_ch = clahe.apply(l_ch)
        return cv2.cvtColor(cv2.merge([l_ch, a_ch, b_ch]), cv2.COLOR_LAB2BGR)

    def _apply_unsharp(self, image: np.ndarray, amount: float) -> np.ndarray:
        ksize = self.unsharp_radius * 2 + 1
        blurred = cv2.GaussianBlur(image, (ksize, ksize), 0)
        return cv2.addWeighted(image, 1.0 + amount, blurred, -amount, 0)

    def transform(self, results: dict) -> dict:
        image = results['img']
        var = _laplacian_variance(image)

        if var >= self.sharp_thr:
            # Image is already sharp; apply only mild CLAHE for consistency
            image = self._apply_clahe(image, self.clahe_clip_sharp)
        elif var >= self.mild_thr:
            image = self._apply_clahe(image, self.clahe_clip_mild)
            image = self._apply_unsharp(image, self.unsharp_amount_mild)
        elif var >= self.moderate_thr:
            image = self._apply_clahe(image, self.clahe_clip_moderate)
            image = self._apply_unsharp(image, self.unsharp_amount_moderate)
        else:
            image = self._apply_clahe(image, self.clahe_clip_severe)
            image = self._apply_unsharp(image, self.unsharp_amount_severe)

        results['img'] = image
        return results

    def __repr__(self) -> str:
        return (
            f'{self.__class__.__name__}('
            f'sharp_thr={self.sharp_thr}, '
            f'mild_thr={self.mild_thr}, '
            f'moderate_thr={self.moderate_thr})'
        )
