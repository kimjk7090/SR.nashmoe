"""
deblur_preprocess.py
X-ray image deblurring and enhancement preprocessing module.

Steps applied by DeblurPreprocessor.process():
1. Blur level assessment  (Laplacian variance)
2. Non-local Means Denoising
3. CLAHE contrast enhancement
4. Bilateral edge-preserving filter
5. Unsharp masking (sharpening)
6. Wiener deconvolution with motion PSF  (only when severe blur detected)

Batch helper:
    process_directory(src_dir, dst_dir, **kwargs)
"""

import os
import numpy as np
import cv2
from pathlib import Path
from typing import Optional, Tuple, Dict, Any


# ---------------------------------------------------------------------------
# Blur level thresholds (Laplacian variance)
# ---------------------------------------------------------------------------
BLUR_THRESHOLDS = {
    'sharp':    100.0,
    'mild':     50.0,
    'moderate': 15.0,
    # < 15.0  → severe
}


def assess_blur_level(image: np.ndarray) -> Tuple[str, float]:
    """Classify blur severity using Laplacian variance.

    Args:
        image: BGR or grayscale uint8 image.

    Returns:
        (level, variance) where level is one of
        'sharp', 'mild', 'moderate', 'severe'.
    """
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    variance = cv2.Laplacian(gray.astype(np.float64), cv2.CV_64F).var()

    if variance >= BLUR_THRESHOLDS['sharp']:
        level = 'sharp'
    elif variance >= BLUR_THRESHOLDS['mild']:
        level = 'mild'
    elif variance >= BLUR_THRESHOLDS['moderate']:
        level = 'moderate'
    else:
        level = 'severe'

    return level, float(variance)


def _build_motion_psf(size: int = 15, angle: float = 0.0) -> np.ndarray:
    """Create a motion-blur Point Spread Function (PSF).

    Args:
        size:  Length of the motion blur kernel (pixels).
        angle: Angle of motion in degrees (0 = horizontal).

    Returns:
        Normalised PSF as float32 array of shape (size, size).
    """
    psf = np.zeros((size, size), dtype=np.float32)
    center = size // 2
    cv2.line(
        psf,
        (center, center),
        (int(center + (size - 1) / 2 * np.cos(np.radians(angle))),
         int(center + (size - 1) / 2 * np.sin(np.radians(angle)))),
        1.0,
        1,
    )
    total = psf.sum()
    if total > 0:
        psf /= total
    return psf


def wiener_deconvolution(
    image: np.ndarray,
    psf: np.ndarray,
    noise_ratio: float = 0.01,
) -> np.ndarray:
    """Apply Wiener deconvolution to restore a blurred image.

    Operates in the frequency domain on each channel independently.

    Args:
        image:       BGR uint8 image.
        psf:         Point Spread Function (motion PSF).
        noise_ratio: Noise-to-signal ratio (regularisation strength).

    Returns:
        Restored BGR uint8 image.
    """
    img_float = image.astype(np.float32) / 255.0
    h, w = img_float.shape[:2]

    # Pad PSF to image size
    psf_pad = np.zeros((h, w), dtype=np.float32)
    ph, pw = psf.shape
    psf_pad[:ph, :pw] = psf
    psf_pad = np.roll(psf_pad, -ph // 2, axis=0)
    psf_pad = np.roll(psf_pad, -pw // 2, axis=1)
    PSF_F = np.fft.fft2(psf_pad)

    channels = cv2.split(img_float) if img_float.ndim == 3 else [img_float]
    restored_channels = []
    for ch in channels:
        CH_F = np.fft.fft2(ch)
        PSF_conj = np.conj(PSF_F)
        denom = np.abs(PSF_F) ** 2 + noise_ratio
        restored_F = PSF_conj * CH_F / denom
        restored = np.real(np.fft.ifft2(restored_F))
        restored = np.clip(restored, 0.0, 1.0)
        restored_channels.append(restored)

    if len(restored_channels) == 1:
        result = (restored_channels[0] * 255).astype(np.uint8)
    else:
        result = cv2.merge(restored_channels)
        result = (result * 255).astype(np.uint8)
    return result


class DeblurPreprocessor:
    """Full deblurring / enhancement pipeline for X-ray images.

    Usage::

        preprocessor = DeblurPreprocessor()
        enhanced = preprocessor.process(image)

        # Batch mode
        preprocessor.process_directory('raw_images/', 'enhanced_images/')

    Args:
        nlm_h:             Non-local Means filter strength (luminance).
        nlm_template_size: Template patch size for NLM (odd number).
        nlm_search_size:   Search window size for NLM (odd number).
        clahe_clip_limit:  Clip limit for CLAHE.
        clahe_tile_grid:   Tile grid size for CLAHE.
        bilateral_d:       Bilateral filter diameter.
        bilateral_sigma_color: Bilateral colour sigma.
        bilateral_sigma_space: Bilateral space sigma.
        unsharp_amount:    Unsharp mask strength.
        unsharp_radius:    Gaussian blur radius for unsharp mask.
        motion_psf_size:   Size of motion PSF kernel.
        motion_psf_angle:  Angle of estimated motion blur (degrees).
        wiener_noise_ratio: Wiener deconvolution regularisation.
    """

    def __init__(
        self,
        nlm_h: float = 10.0,
        nlm_template_size: int = 7,
        nlm_search_size: int = 21,
        clahe_clip_limit: float = 2.0,
        clahe_tile_grid: Tuple[int, int] = (8, 8),
        bilateral_d: int = 9,
        bilateral_sigma_color: float = 75.0,
        bilateral_sigma_space: float = 75.0,
        unsharp_amount: float = 1.5,
        unsharp_radius: int = 5,
        motion_psf_size: int = 15,
        motion_psf_angle: float = 0.0,
        wiener_noise_ratio: float = 0.01,
    ) -> None:
        self.nlm_h = nlm_h
        self.nlm_template_size = nlm_template_size
        self.nlm_search_size = nlm_search_size
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_tile_grid = clahe_tile_grid
        self.bilateral_d = bilateral_d
        self.bilateral_sigma_color = bilateral_sigma_color
        self.bilateral_sigma_space = bilateral_sigma_space
        self.unsharp_amount = unsharp_amount
        self.unsharp_radius = unsharp_radius
        self.motion_psf_size = motion_psf_size
        self.motion_psf_angle = motion_psf_angle
        self.wiener_noise_ratio = wiener_noise_ratio

    # ------------------------------------------------------------------
    # Individual processing steps
    # ------------------------------------------------------------------

    def denoise(self, image: np.ndarray) -> np.ndarray:
        """Apply Non-local Means denoising."""
        if image.ndim == 3:
            return cv2.fastNlMeansDenoisingColored(
                image,
                None,
                self.nlm_h,
                self.nlm_h,
                self.nlm_template_size,
                self.nlm_search_size,
            )
        return cv2.fastNlMeansDenoising(
            image,
            None,
            self.nlm_h,
            self.nlm_template_size,
            self.nlm_search_size,
        )

    def enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """Apply CLAHE contrast enhancement (per channel for colour images)."""
        clahe = cv2.createCLAHE(
            clipLimit=self.clahe_clip_limit,
            tileGridSize=self.clahe_tile_grid,
        )
        if image.ndim == 2:
            return clahe.apply(image)
        # Apply CLAHE to L channel in LAB colour space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_ch, a_ch, b_ch = cv2.split(lab)
        l_ch = clahe.apply(l_ch)
        lab = cv2.merge([l_ch, a_ch, b_ch])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    def smooth_edges(self, image: np.ndarray) -> np.ndarray:
        """Apply bilateral edge-preserving filter."""
        return cv2.bilateralFilter(
            image,
            self.bilateral_d,
            self.bilateral_sigma_color,
            self.bilateral_sigma_space,
        )

    def sharpen(self, image: np.ndarray) -> np.ndarray:
        """Apply unsharp masking for edge enhancement."""
        blurred = cv2.GaussianBlur(
            image,
            (self.unsharp_radius * 2 + 1, self.unsharp_radius * 2 + 1),
            0,
        )
        sharpened = cv2.addWeighted(
            image,
            1.0 + self.unsharp_amount,
            blurred,
            -self.unsharp_amount,
            0,
        )
        return sharpened

    def restore_motion_blur(self, image: np.ndarray) -> np.ndarray:
        """Apply Wiener deconvolution with a motion PSF."""
        psf = _build_motion_psf(self.motion_psf_size, self.motion_psf_angle)
        return wiener_deconvolution(image, psf, self.wiener_noise_ratio)

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def process(self, image: np.ndarray) -> Dict[str, Any]:
        """Run the complete deblurring / enhancement pipeline.

        Args:
            image: BGR uint8 image array.

        Returns:
            dict with keys:
                'image'       – processed BGR uint8 array
                'blur_level'  – 'sharp' | 'mild' | 'moderate' | 'severe'
                'blur_var'    – Laplacian variance (float)
                'wiener_applied' – whether Wiener deconvolution was used
        """
        blur_level, blur_var = assess_blur_level(image)

        # Step 1: Wiener deconvolution (only for severe blur)
        wiener_applied = False
        if blur_level == 'severe':
            image = self.restore_motion_blur(image)
            wiener_applied = True

        # Step 2: Denoise
        image = self.denoise(image)

        # Step 3: CLAHE contrast enhancement
        image = self.enhance_contrast(image)

        # Step 4: Bilateral edge-preserving smooth
        image = self.smooth_edges(image)

        # Step 5: Unsharp masking
        image = self.sharpen(image)

        return {
            'image': image,
            'blur_level': blur_level,
            'blur_var': blur_var,
            'wiener_applied': wiener_applied,
        }

    # ------------------------------------------------------------------
    # Batch processing
    # ------------------------------------------------------------------

    def process_directory(
        self,
        src_dir: str,
        dst_dir: str,
        extensions: Tuple[str, ...] = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'),
        verbose: bool = True,
    ) -> Dict[str, Dict[str, Any]]:
        """Preprocess all images in a directory.

        Args:
            src_dir:    Source image directory.
            dst_dir:    Destination directory for processed images.
            extensions: Accepted image file extensions (lowercase).
            verbose:    Print per-image status.

        Returns:
            Dictionary mapping filename → result metadata dict.
        """
        src_path = Path(src_dir)
        dst_path = Path(dst_dir)
        dst_path.mkdir(parents=True, exist_ok=True)

        results: Dict[str, Dict[str, Any]] = {}
        image_files = [
            f for f in src_path.iterdir()
            if f.suffix.lower() in extensions
        ]

        for img_file in sorted(image_files):
            image = cv2.imread(str(img_file))
            if image is None:
                if verbose:
                    print(f'[WARN] Cannot read: {img_file.name}')
                continue

            result = self.process(image)
            out_path = dst_path / img_file.name
            cv2.imwrite(str(out_path), result['image'])

            results[img_file.name] = {
                'blur_level': result['blur_level'],
                'blur_var': result['blur_var'],
                'wiener_applied': result['wiener_applied'],
            }

            if verbose:
                print(
                    f"[INFO] {img_file.name}: "
                    f"blur={result['blur_level']} "
                    f"(var={result['blur_var']:.2f}), "
                    f"wiener={result['wiener_applied']}"
                )

        return results
