"""
inference_pipeline.py
End-to-end inference: deblurring preprocessing + ReDet rotated detection.

Classes
-------
VertebraPedicleDetector
    Combines DeblurPreprocessor with two ReDet models (vertebra + pedicle).
    Supports single-image detection, cascaded full-pipeline, visualisation,
    and Test-Time Augmentation (TTA) with multi-scale inference + Soft-NMS.

Usage (CLI)
-----------
    python inference_pipeline.py \\
        --image path/to/xray.jpg \\
        --vertebra-ckpt work_dirs/vb_redet/epoch_10.pth \\
        --pedicle-ckpt  work_dirs/pd_redet/epoch_10.pth \\
        --config        redet-le90_re50_refpn_1x_dota.py \\
        --output        result.jpg
"""

import argparse
import copy
import os
from typing import Dict, List, Optional, Tuple, Any

import cv2
import numpy as np

from .deblur_preprocess import DeblurPreprocessor

# ---------------------------------------------------------------------------
# Optional imports — only required at runtime when models are loaded
# ---------------------------------------------------------------------------
try:
    from mmdet.apis import init_detector, inference_detector
    _MMDET_AVAILABLE = True
except ImportError:
    _MMDET_AVAILABLE = False


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _rotated_box_to_pts(cx: float, cy: float, w: float, h: float, angle: float) -> np.ndarray:
    """Convert rotated box (cx, cy, w, h, angle_rad) to 4 corner points."""
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    hw, hh = w / 2.0, h / 2.0
    corners = np.array([
        [-hw, -hh],
        [ hw, -hh],
        [ hw,  hh],
        [-hw,  hh],
    ], dtype=np.float32)
    rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32)
    rotated = corners @ rot.T + np.array([cx, cy])
    return rotated.astype(np.int32)


def _soft_nms_rotated(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float = 0.15,
    sigma: float = 0.5,
    score_threshold: float = 0.001,
) -> Tuple[np.ndarray, np.ndarray]:
    """Simple Gaussian Soft-NMS for rotated boxes.

    Args:
        boxes:  (N, 5) array of [cx, cy, w, h, angle].
        scores: (N,) confidence scores.
        iou_threshold: IoU above which to decay score.
        sigma: Gaussian decay factor.
        score_threshold: Minimum score to keep.

    Returns:
        Tuple of (kept_boxes, kept_scores).
    """
    if len(boxes) == 0:
        return boxes, scores

    boxes = boxes.copy()
    scores = scores.copy().astype(np.float32)
    keep = []

    indices = list(range(len(boxes)))
    while indices:
        best_idx = int(np.argmax(scores[indices]))
        best = indices[best_idx]
        keep.append(best)
        indices.pop(best_idx)

        remaining = []
        for idx in indices:
            # Approximate IoU using axis-aligned bounding boxes of the rotated boxes
            b1, b2 = boxes[best], boxes[idx]
            # Use overlap of bounding circles as a fast approximation
            dist = np.sqrt((b1[0] - b2[0]) ** 2 + (b1[1] - b2[1]) ** 2)
            r1 = np.sqrt(b1[2] ** 2 + b1[3] ** 2) / 2.0
            r2 = np.sqrt(b2[2] ** 2 + b2[3] ** 2) / 2.0
            iou_approx = max(0.0, (r1 + r2 - dist) / (r1 + r2 + 1e-6))
            if iou_approx > iou_threshold:
                scores[idx] *= np.exp(-(iou_approx ** 2) / sigma)
            if scores[idx] > score_threshold:
                remaining.append(idx)
        indices = remaining

    kept_boxes = boxes[keep]
    kept_scores = scores[keep]
    return kept_boxes, kept_scores


# ---------------------------------------------------------------------------
# Main detector class
# ---------------------------------------------------------------------------

class VertebraPedicleDetector:
    """End-to-end vertebra and pedicle detector with deblurring preprocessing.

    Args:
        vertebra_ckpt (str): Path to the vertebra ReDet model checkpoint.
        pedicle_ckpt  (str): Path to the pedicle  ReDet model checkpoint.
        config_path   (str): Path to the MMRotate config file.
        device        (str): Torch device string, e.g. ``'cuda:0'`` or ``'cpu'``.
        score_thr     (float): Minimum confidence score for detections.
        tta_scales    (list[float]): Scales used for Test-Time Augmentation.
        tta_flip      (bool): Whether to include horizontally flipped inference.
        deblur_kwargs (dict): Additional keyword arguments forwarded to
            :class:`DeblurPreprocessor`.
    """

    def __init__(
        self,
        vertebra_ckpt: str,
        pedicle_ckpt: str,
        config_path: str,
        device: str = 'cuda:0',
        score_thr: float = 0.05,
        tta_scales: Optional[List[float]] = None,
        tta_flip: bool = True,
        deblur_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not _MMDET_AVAILABLE:
            raise ImportError(
                'mmdet is not installed. '
                'Please install mmdet and mmrotate before using VertebraPedicleDetector.'
            )

        if tta_scales is None:
            tta_scales = [0.8, 1.0, 1.2]

        self.vertebra_ckpt = vertebra_ckpt
        self.pedicle_ckpt = pedicle_ckpt
        self.config_path = config_path
        self.device = device
        self.score_thr = score_thr
        self.tta_scales = tta_scales
        self.tta_flip = tta_flip

        self.preprocessor = DeblurPreprocessor(**(deblur_kwargs or {}))

        self._vertebra_model = None
        self._pedicle_model = None

    # ------------------------------------------------------------------
    # Model loading (lazy)
    # ------------------------------------------------------------------

    @property
    def vertebra_model(self):
        if self._vertebra_model is None:
            self._vertebra_model = init_detector(
                self.config_path,
                self.vertebra_ckpt,
                device=self.device,
            )
        return self._vertebra_model

    @property
    def pedicle_model(self):
        if self._pedicle_model is None:
            self._pedicle_model = init_detector(
                self.config_path,
                self.pedicle_ckpt,
                device=self.device,
            )
        return self._pedicle_model

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def _preprocess(self, image_path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Load image and run deblurring pipeline.

        Returns:
            Tuple of (processed_bgr_image, preprocessing_metadata).
        """
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f'Cannot read image: {image_path}')
        result = self.preprocessor.process(image)
        return result['image'], {
            'blur_level': result['blur_level'],
            'blur_var': result['blur_var'],
            'wiener_applied': result['wiener_applied'],
        }

    # ------------------------------------------------------------------
    # Single-scale inference helpers
    # ------------------------------------------------------------------

    def _run_inference(self, model, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Run mmdet inference on a BGR numpy image.

        Returns:
            (boxes, scores) where boxes is (N, 5) [cx, cy, w, h, angle]
            and scores is (N,).
        """
        result = inference_detector(model, image)
        pred = result.pred_instances

        if not hasattr(pred, 'bboxes') or len(pred.bboxes) == 0:
            return np.empty((0, 5), dtype=np.float32), np.empty((0,), dtype=np.float32)

        boxes = pred.bboxes.cpu().numpy()   # (N, 5)
        scores = pred.scores.cpu().numpy()  # (N,)

        # Filter by score threshold
        mask = scores >= self.score_thr
        return boxes[mask], scores[mask]

    # ------------------------------------------------------------------
    # Test-Time Augmentation
    # ------------------------------------------------------------------

    def _tta_inference(self, model, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Multi-scale + flip TTA inference merged with Soft-NMS.

        Args:
            model: MMDet model instance.
            image: BGR uint8 image.

        Returns:
            (boxes, scores) after Soft-NMS merging.
        """
        h, w = image.shape[:2]
        all_boxes: List[np.ndarray] = []
        all_scores: List[np.ndarray] = []

        augmented_images = []
        for scale in self.tta_scales:
            new_h, new_w = int(h * scale), int(w * scale)
            resized = cv2.resize(image, (new_w, new_h))
            augmented_images.append((resized, scale, False))
            if self.tta_flip:
                flipped = cv2.flip(resized, 1)
                augmented_images.append((flipped, scale, True))

        for aug_img, scale, flipped in augmented_images:
            boxes, scores = self._run_inference(model, aug_img)
            if len(boxes) == 0:
                continue

            # Rescale boxes back to original image size
            boxes_orig = boxes.copy()
            boxes_orig[:, 0] /= scale  # cx
            boxes_orig[:, 1] /= scale  # cy
            boxes_orig[:, 2] /= scale  # w
            boxes_orig[:, 3] /= scale  # h

            if flipped:
                boxes_orig[:, 0] = w - boxes_orig[:, 0]  # mirror cx
                boxes_orig[:, 4] = -boxes_orig[:, 4]     # negate angle

            all_boxes.append(boxes_orig)
            all_scores.append(scores)

        if not all_boxes:
            return np.empty((0, 5), dtype=np.float32), np.empty((0,), dtype=np.float32)

        merged_boxes = np.concatenate(all_boxes, axis=0)
        merged_scores = np.concatenate(all_scores, axis=0)
        return _soft_nms_rotated(merged_boxes, merged_scores)

    # ------------------------------------------------------------------
    # Public detection methods
    # ------------------------------------------------------------------

    def detect_vertebrae(self, image_path: str) -> Dict[str, Any]:
        """Detect vertebrae in an X-ray image.

        Args:
            image_path: Path to the input X-ray image.

        Returns:
            dict with keys:
                'boxes'      – (N, 5) rotated bboxes [cx, cy, w, h, angle]
                'scores'     – (N,) confidence scores
                'blur_level' – assessed blur level
                'blur_var'   – Laplacian variance
        """
        image, meta = self._preprocess(image_path)
        boxes, scores = self._tta_inference(self.vertebra_model, image)
        return {'boxes': boxes, 'scores': scores, **meta}

    def detect_pedicles(self, image_path: str) -> Dict[str, Any]:
        """Detect pedicles in an X-ray image.

        Args:
            image_path: Path to the input X-ray image.

        Returns:
            Same structure as :meth:`detect_vertebrae`.
        """
        image, meta = self._preprocess(image_path)
        boxes, scores = self._tta_inference(self.pedicle_model, image)
        return {'boxes': boxes, 'scores': scores, **meta}

    def run_full_pipeline(self, image_path: str) -> Dict[str, Any]:
        """Detect vertebrae first, then detect pedicles within each vertebra crop.

        Args:
            image_path: Path to the input X-ray image.

        Returns:
            dict with keys:
                'image'          – preprocessed BGR image
                'vertebra_boxes' – (N, 5) vertebra rotated bboxes
                'vertebra_scores'– (N,) vertebra scores
                'pedicle_results'– list of dicts, one per vertebra:
                    {'vertebra_idx': int,
                     'boxes': (M, 5),   # pedicle boxes in original image coords
                     'scores': (M,)}
                'blur_level'     – assessed blur level of the full image
                'blur_var'       – Laplacian variance
        """
        image, meta = self._preprocess(image_path)
        h, w = image.shape[:2]

        # Step 1: Detect vertebrae on the full image
        vb_boxes, vb_scores = self._tta_inference(self.vertebra_model, image)

        # Step 2: For each vertebra, crop and detect pedicles
        pedicle_results = []
        for i, (box, score) in enumerate(zip(vb_boxes, vb_scores)):
            cx, cy, bw, bh = box[0], box[1], box[2], box[3]
            # Axis-aligned crop bounding box (with some padding)
            pad = 0.1
            x1 = max(0, int(cx - bw / 2 * (1 + pad)))
            y1 = max(0, int(cy - bh / 2 * (1 + pad)))
            x2 = min(w, int(cx + bw / 2 * (1 + pad)))
            y2 = min(h, int(cy + bh / 2 * (1 + pad)))

            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            pe_boxes, pe_scores = self._tta_inference(self.pedicle_model, crop)

            # Translate pedicle box coordinates back to full-image space
            if len(pe_boxes) > 0:
                pe_boxes_full = pe_boxes.copy()
                pe_boxes_full[:, 0] += x1  # cx
                pe_boxes_full[:, 1] += y1  # cy
            else:
                pe_boxes_full = pe_boxes

            pedicle_results.append({
                'vertebra_idx': i,
                'boxes': pe_boxes_full,
                'scores': pe_scores,
            })

        return {
            'image': image,
            'vertebra_boxes': vb_boxes,
            'vertebra_scores': vb_scores,
            'pedicle_results': pedicle_results,
            **meta,
        }

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def visualize_results(
        self,
        image: np.ndarray,
        vertebra_results: Optional[Dict[str, Any]] = None,
        pedicle_results: Optional[Dict[str, Any]] = None,
        vertebra_color: Tuple[int, int, int] = (0, 255, 0),
        pedicle_color: Tuple[int, int, int] = (0, 0, 255),
        thickness: int = 2,
        font_scale: float = 0.5,
    ) -> np.ndarray:
        """Draw rotated bounding boxes on the image.

        Args:
            image:            BGR image to annotate (not modified in place).
            vertebra_results: Output from :meth:`detect_vertebrae` or
                              :meth:`run_full_pipeline` (may be None).
            pedicle_results:  Output from :meth:`detect_pedicles` or
                              pedicle sub-results from full pipeline (may be None).
            vertebra_color:   BGR colour for vertebra boxes. Default: green.
            pedicle_color:    BGR colour for pedicle boxes.  Default: red.
            thickness:        Box line thickness in pixels.
            font_scale:       Label font scale.

        Returns:
            Annotated copy of the input image.
        """
        vis = image.copy()

        def _draw_boxes(boxes, scores, color, label_prefix):
            for box, score in zip(boxes, scores):
                cx, cy, bw, bh, angle = box
                pts = _rotated_box_to_pts(cx, cy, bw, bh, angle)
                cv2.polylines(vis, [pts.reshape(-1, 1, 2)], True, color, thickness)
                label = f'{label_prefix}: {score:.2f}'
                cv2.putText(
                    vis, label,
                    (int(cx), int(cy)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, color, thickness, cv2.LINE_AA,
                )

        if vertebra_results is not None:
            vb_boxes = vertebra_results.get('vertebra_boxes', vertebra_results.get('boxes', np.empty((0, 5))))
            vb_scores = vertebra_results.get('vertebra_scores', vertebra_results.get('scores', np.empty((0,))))
            _draw_boxes(vb_boxes, vb_scores, vertebra_color, 'VB')

        if pedicle_results is not None:
            if isinstance(pedicle_results, list):
                # List of per-vertebra dicts (from run_full_pipeline)
                for pe in pedicle_results:
                    _draw_boxes(pe['boxes'], pe['scores'], pedicle_color, 'PE')
            else:
                pe_boxes = pedicle_results.get('boxes', np.empty((0, 5)))
                pe_scores = pedicle_results.get('scores', np.empty((0,)))
                _draw_boxes(pe_boxes, pe_scores, pedicle_color, 'PE')

        return vis


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Vertebra & Pedicle detection with deblurring preprocessing'
    )
    parser.add_argument('--image', required=True, help='Input X-ray image path')
    parser.add_argument('--vertebra-ckpt', required=True, help='Vertebra model checkpoint')
    parser.add_argument('--pedicle-ckpt', required=True, help='Pedicle model checkpoint')
    parser.add_argument(
        '--config',
        default='redet-le90_re50_refpn_1x_dota.py',
        help='MMRotate config file path',
    )
    parser.add_argument('--device', default='cuda:0', help='Device (e.g. cuda:0 or cpu)')
    parser.add_argument('--score-thr', type=float, default=0.05, help='Score threshold')
    parser.add_argument('--output', default='result.jpg', help='Output image path')
    parser.add_argument(
        '--mode',
        choices=['vertebra', 'pedicle', 'full'],
        default='full',
        help=(
            'Detection mode: '
            '"vertebra" – detect vertebrae only, '
            '"pedicle"  – detect pedicles only, '
            '"full"     – cascaded pipeline'
        ),
    )
    parser.add_argument(
        '--no-tta',
        action='store_true',
        help='Disable Test-Time Augmentation',
    )
    args = parser.parse_args()

    tta_scales = [1.0] if args.no_tta else [0.8, 1.0, 1.2]
    tta_flip = not args.no_tta

    detector = VertebraPedicleDetector(
        vertebra_ckpt=args.vertebra_ckpt,
        pedicle_ckpt=args.pedicle_ckpt,
        config_path=args.config,
        device=args.device,
        score_thr=args.score_thr,
        tta_scales=tta_scales,
        tta_flip=tta_flip,
    )

    image_bgr = cv2.imread(args.image)
    if image_bgr is None:
        raise SystemExit(f'Cannot read image: {args.image}')

    if args.mode == 'vertebra':
        result = detector.detect_vertebrae(args.image)
        vis = detector.visualize_results(image_bgr, vertebra_results=result)
        print(f"Detected {len(result['boxes'])} vertebrae  "
              f"| blur={result['blur_level']} (var={result['blur_var']:.2f})")

    elif args.mode == 'pedicle':
        result = detector.detect_pedicles(args.image)
        vis = detector.visualize_results(image_bgr, pedicle_results=result)
        print(f"Detected {len(result['boxes'])} pedicles  "
              f"| blur={result['blur_level']} (var={result['blur_var']:.2f})")

    else:  # full
        result = detector.run_full_pipeline(args.image)
        vis = detector.visualize_results(
            result['image'],
            vertebra_results=result,
            pedicle_results=result['pedicle_results'],
        )
        total_pedicles = sum(len(r['boxes']) for r in result['pedicle_results'])
        print(
            f"Detected {len(result['vertebra_boxes'])} vertebrae, "
            f"{total_pedicles} pedicles  "
            f"| blur={result['blur_level']} (var={result['blur_var']:.2f})"
        )

    cv2.imwrite(args.output, vis)
    print(f'Result saved to: {args.output}')


if __name__ == '__main__':
    main()
