# ColorRevive — Model Training

Offline pipeline for training the `unet-lab-v1` U-Net colorizer. **Nothing in
this folder is imported by the serving backend at runtime** — the API only
consumes the resulting checkpoint via `MODEL_CHECKPOINT_PATH`.

## Datasets

Any folder of RGB images works (COCO, an ImageNet subset, Places365, or your
own photos). Do **not** put large datasets under git; point `data.root` in
`configs/config.yaml` at a local path. The app never downloads data
automatically.

```bash
mkdir -p data && ln -s /path/to/coco/train2017 data/raw   # or just copy images in
```

## Train

```bash
pip install torch torchvision numpy scikit-image pillow pyyaml tensorboard albumentations
python src/train.py --config configs/config.yaml
# resume:
python src/train.py --config configs/config.yaml --resume checkpoints/last.pt
```

Features: seeded reproducibility, AdamW + cosine schedule, grad clipping,
train/val loops, `best.pt`/`last.pt` checkpointing, early stopping
(`train.early_stopping_patience`), TensorBoard (`checkpoints/runs/`), and
optional W&B (`train.wandb: true`). Loss is L1 on normalized ab channels;
a perceptual-loss extension point lives in `src/losses.py`.

## Evaluate

```bash
python src/evaluate.py --checkpoint checkpoints/best.pt --config configs/config.yaml \
    --split val --grid out/comparison-grid.png
```

Reports MAE/MSE (RGB), PSNR, SSIM, and mean ΔE (Lab), plus a 4-column grid:
`Original RGB | Grayscale input | Colorized output | Absolute difference`.

> Metrics alone do not measure color plausibility — many colors are reasonable
> for the same grayscale input. Always inspect the grid visually.

## Batch inference

```bash
python src/inference.py --checkpoint checkpoints/best.pt \
    --input ../sample-data --output out/colorized
```

## Serving a trained checkpoint

The backend builds `UNetLab` with `base_channels=64` (see
`backend/app/ml/model.py`). Either train with `model.base_channels: 64`, or
edit that constant to match your checkpoint, then:

```bash
MODEL_CHECKPOINT_PATH=/abs/path/to/best.pt uvicorn app.main:app --port 8000
```

`GET /health` should then report `"model_loaded": true, "fallback_mode": false`.

## Honest status

No trained checkpoint ships with this repository. Until you train one (or
supply a compatible third-party weight file), the application runs in its
clearly-labeled deterministic fallback mode. This training code has been
smoke-tested (forward/backward pass, dataset loading, checkpoint round-trip)
but has **not** been run to convergence on a real dataset.
