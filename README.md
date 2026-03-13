# SR.nashmoe

Required dependencies: Python 3.8, PyTorch 2.0, NumPy, OpenCV-Python, mmcv 2.0.0, mmrotate 1.0.0rc1, mmdet 3.1.0, CUDA 11.7

## Files

1. `redet_train_vertebra.ipynb` — Training code for vertebra detection
2. `redet_train_pedicle.ipynb` — Training code for pedicle detection
3. `pedicle_model_perf_val.ipynb` — Pedicle model performance validation using the trained pedicle model, validation data and labels
4. `redet-le90_re50_refpn_1x_dota.py` — Configuration for ReDet object detection model
5. `re_resnet50_c8_batch256-25b16846.pth` — The Re-ResNet-50 backbone of the ReDet model is initialised with weights from this checkpoint, pre-trained on the ImageNet dataset

---

## Blur-Robust Pipeline (`blur_robust/`)

X-ray images frequently suffer from **motion blur** caused by patient movement, breathing, or equipment vibration during image capture. This makes it difficult to accurately detect vertebra bounding boxes and precisely localise pedicle regions. The `blur_robust/` module adds comprehensive blur-handling that integrates with the existing MMRotate/ReDet workflow.

### Problem

- Standard X-ray capture introduces directional motion blur and Gaussian noise.
- Blurry edges cause bounding-box regression errors and missed pedicle detections.
- The original training pipeline includes only `RandomFlip` augmentation — no blur-specific augmentations.

### Blur-Robust Pipeline Overview

```
Input X-ray image
       │
       ▼
┌──────────────────────────────────────────────────────────┐
│  DeblurPreprocessor (deblur_preprocess.py)               │
│                                                          │
│  1. Blur assessment   — Laplacian variance               │
│     → sharp / mild / moderate / severe                   │
│                                                          │
│  2. Wiener deconvolution (motion PSF)  ← severe only     │
│  3. Non-local Means denoising                            │
│  4. CLAHE contrast enhancement                           │
│  5. Bilateral edge-preserving filter                     │
│  6. Unsharp masking (sharpening)                         │
└──────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────┐
│  ReDet detection (ReResNet-50 + ReFPN + CascadeRoIHead)  │
│  (loaded via MMRotate / MMDet APIs)                      │
└──────────────────────────────────────────────────────────┘
       │
       ▼
  Rotated bounding boxes (vertebrae / pedicles)
```

### Module Structure

| File | Description |
|------|-------------|
| `blur_robust/__init__.py` | Package init — exports all public classes |
| `blur_robust/deblur_preprocess.py` | `DeblurPreprocessor` class: full deblurring pipeline + batch `process_directory()` |
| `blur_robust/blur_augmentation.py` | `BlurAugmentation` and `AdaptiveSharpen` — MMRotate-compatible TRANSFORMS |
| `blur_robust/enhanced_config.py` | `get_blur_robust_pedicle_config()` / `get_blur_robust_vertebra_config()` |
| `blur_robust/inference_pipeline.py` | `VertebraPedicleDetector` — end-to-end inference with TTA (multi-scale + flip) + Soft-NMS merging |
| `blur_robust/train_with_blur_augmentation.ipynb` | Notebook: full training workflow with blur augmentation |

### Training with Blur Augmentation

1. Open `blur_robust/train_with_blur_augmentation.ipynb`.
2. The notebook will:
   - Register `PEdataset` / `VBdataset`.
   - Import `BlurAugmentation` and `AdaptiveSharpen` (auto-registered via `@TRANSFORMS.register_module()`).
   - Load the enhanced config via `get_blur_robust_pedicle_config()`.
   - Show a pipeline diff (original vs. blur-robust).
   - Create an `mmengine.Runner` and start training.

Alternatively, load the config programmatically:

```python
from blur_robust.blur_augmentation import BlurAugmentation, AdaptiveSharpen  # registers transforms
from blur_robust.enhanced_config import get_blur_robust_pedicle_config
from mmengine.config import Config
from mmengine.runner import Runner

cfg = Config(get_blur_robust_pedicle_config())
runner = Runner.from_cfg(cfg)
runner.train()
```

Key changes vs. the original config:
- `BlurAugmentation` + `AdaptiveSharpen` injected into `train_pipeline`.
- NMS `iou_threshold` relaxed from 0.1 → 0.15 (standard rotated NMS with a looser threshold) for better handling of slightly overlapping detections on blurry images.
- Learning rate reduced from 2e-5 → 1e-5 with cosine annealing for stable convergence.

### Running Inference with Deblurring

**Full pipeline (vertebrae → pedicles):**

```bash
python blur_robust/inference_pipeline.py \
    --image        path/to/xray.jpg \
    --vertebra-ckpt work_dirs/vb_redet/epoch_10.pth \
    --pedicle-ckpt  work_dirs/pd_redet/epoch_10.pth \
    --config        redet-le90_re50_refpn_1x_dota.py \
    --output        result.jpg \
    --mode          full
```

**Vertebrae only:**

```bash
python blur_robust/inference_pipeline.py \
    --image path/to/xray.jpg \
    --vertebra-ckpt work_dirs/vb_redet/epoch_10.pth \
    --pedicle-ckpt  work_dirs/pd_redet/epoch_10.pth \
    --mode vertebra --output vertebra_result.jpg
```

**Disable TTA (faster, single-scale):**

```bash
python blur_robust/inference_pipeline.py \
    --image path/to/xray.jpg \
    --vertebra-ckpt work_dirs/vb_redet/epoch_10.pth \
    --pedicle-ckpt  work_dirs/pd_redet/epoch_10.pth \
    --no-tta --output result.jpg
```

**Batch preprocessing only:**

```python
from blur_robust.deblur_preprocess import DeblurPreprocessor

preprocessor = DeblurPreprocessor()
stats = preprocessor.process_directory('raw_images/', 'enhanced_images/')
# stats maps filename → {'blur_level', 'blur_var', 'wiener_applied'}
```

### Blur Augmentation Details

`BlurAugmentation` randomly applies one of four blur types with 50 % probability:

| Blur Type | Description |
|-----------|-------------|
| `motion`  | Directional motion blur (random angle, random kernel size) |
| `gaussian`| Isotropic Gaussian blur |
| `median`  | Median blur (robust against salt-and-pepper noise) |
| `defocus` | Disk-shaped defocus blur |

`AdaptiveSharpen` measures each training image's blur level (Laplacian variance) and applies proportional CLAHE + Unsharp Masking — ensuring blurrier training crops are enhanced more aggressively.

### Test-Time Augmentation (TTA)

During inference, `VertebraPedicleDetector` performs multi-scale inference at `[0.8×, 1.0×, 1.2×]` plus horizontal flip, then merges detections with Gaussian Soft-NMS (`iou_threshold=0.15`). This significantly improves recall on blurry images where single-scale inference may miss detections.
