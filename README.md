# Sky Segmentation for Embedded Systems

This repository contains training and inference code for sky segmentation models focused on embedded deployment.

## What is included

- TensorFlow training pipeline for the WNet/U2Net-style model.
- ONNX inference scripts for TensorFlow-trained WNet:
- `infer_wnet_tf2onnx_image.py` and `infer_wnet_tf2onnx_video.py`.
- ONNX inference scripts for PyTorch-trained DeepLab (custom normalization):
- `infer_deeplab_wnet_pytorch2onnx_image.py` and `infer_deeplab_wnet_pytorch2onnx_video.py`.
- Model weights and experiment outputs under `weights/` and `training/`.

## Quick start

Create and activate a virtual environment, then install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate    # Linux/macOS (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
```

Run the single-image WNet inference:

```bash
python inference/infer_wnet_tf2onnx_image.py
```

Run one of the scripts inside `inference/` according to your input type (image or video).

## Results

Images are stored in `docs/docs_imgs_results`:

![Stacked Article Shot](docs/docs_imgs_results/result_article_shot_stacked.png)
