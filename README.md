# Learning MONAI for Biomedical Image Processing

This repository is a simple starting point for learning [MONAI](https://monai.io/) in biomedical image processing.

## What you will learn

1. How to set up a MONAI environment
2. How to load and transform medical images
3. How to build and train a basic segmentation model
4. How to run inference and evaluate results

## Quick start

### 1) Create environment and install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install monai torch torchvision nibabel matplotlib
```

### 2) Minimal MONAI example (2D segmentation pipeline)

```python
from monai.transforms import (
    Compose,
    EnsureChannelFirstD,
    LoadImaged,
    ScaleIntensityD,
    ResizeD,
    ToTensorD,
)
from monai.data import Dataset, DataLoader

data = [
    {"image": "path/to/image1.nii.gz", "label": "path/to/label1.nii.gz"},
    {"image": "path/to/image2.nii.gz", "label": "path/to/label2.nii.gz"},
]

transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstD(keys=["image", "label"]),
        ScaleIntensityD(keys=["image"]),
        ResizeD(keys=["image", "label"], spatial_size=(256, 256)),
        ToTensorD(keys=["image", "label"]),
    ]
)

dataset = Dataset(data=data, transform=transforms)
loader = DataLoader(dataset, batch_size=2, shuffle=True)
```

## Suggested learning path

- **Step 1: Data & transforms**
  - Learn MONAI dictionary-based transforms (`*D`) and `Compose`.
- **Step 2: Model basics**
  - Start with `monai.networks.nets.UNet` for segmentation.
- **Step 3: Losses & metrics**
  - Explore `DiceLoss`, Dice score, and validation loops.
- **Step 4: Inference**
  - Use sliding window inference for large 3D volumes.
- **Step 5: Production readiness**
  - Add reproducibility, checkpointing, and experiment tracking.

## Recommended public datasets

- BTCV (abdomen multi-organ CT)
- BraTS (brain tumor MRI)
- MSD (Medical Segmentation Decathlon)

## Official resources

- MONAI docs: https://docs.monai.io/
- MONAI tutorials: https://github.com/Project-MONAI/tutorials
- MONAI model zoo: https://github.com/Project-MONAI/model-zoo
