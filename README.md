# NYCU Computer Vision 2026 HW4 — Image Restoration

- **Student ID**: 314551069
- **Name**: Kao Yun-Ching

---

## Introduction

This repository implements **PromptIR** for all-in-one blind image restoration,
targeting two degradation types: **rain streaks** and **snow**.

PromptIR is a four-level U-Net transformer (based on Restormer) augmented with
**Prompt Generation Blocks (PGB)** at each decoder stage. Each PGB maintains a
learnable pool of prompt tensors whose weighted combination encodes
degradation-specific context, enabling a single unified model to handle multiple
degradation types without explicit degradation labels at test time.

At inference, **8-fold Test-Time Augmentation (TTA)** — averaging predictions
over horizontal flip × vertical flip × 90° rotation combinations — further
improves restoration quality.

---

## Environment Setup

```bash
conda create -n dl_vision python=3.10 -y
conda activate dl_vision

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install pillow tqdm numpy matplotlib
```

---

## Usage

### Training

```bash
python main.py --mode train \
    --data_dir /path/to/train \
    --save_dir ./checkpoints \
    --epochs 150 \
    --batch_size 16 \
    --patch_size 256 \
    --dim 64 \
    --device cuda:0
```

### Resume Training

```bash
python main.py --mode train \
    --data_dir /path/to/train \
    --save_dir ./checkpoints \
    --epochs 150 \
    --batch_size 16 \
    --patch_size 256 \
    --dim 64 \
    --device cuda:0 \
    --resume ./checkpoints/latest.pth
```

### Inference (with TTA)

```bash
python main.py --mode test \
    --test_dir /path/to/test/degraded \
    --checkpoint ./checkpoints/best.pth \
    --output pred.npz \
    --dim 64 \
    --device cuda:0
```

The output `pred.npz` contains restored images as `uint8` arrays of shape
`(3, H, W)`, keyed by filename (e.g., `'0.png'`, `'1.png'`, …).

### Generate Report Figures

```bash
python plot_report.py \
    --log ./checkpoints/train_log.csv \
    --data_dir /path/to/train \
    --checkpoint ./checkpoints/best.pth \
    --dim 64 \
    --device cuda:0
```

---

## Performance Snapshot

| Configuration | Val PSNR | Test PSNR (TTA) |
|---|---|---|
| PromptIR (d=48), no TTA | 29.22 dB | 29.76 dB |
| PromptIR (d=48) + TTA ×8 | 29.22 dB | 30.44 dB |
| PromptIR (d=64) + TTA ×8 | **29.77 dB** | **30.76 dB** |

> Public leaderboard score: **30.76 dB** (PSNR), exceeding the strong baseline of 30 dB.
