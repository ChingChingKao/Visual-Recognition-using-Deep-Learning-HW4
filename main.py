#!/usr/bin/env python3
"""
Visual Recognition using Deep Learning - HW4
Image Restoration using PromptIR (Rain & Snow removal)
Student: Kao Yun-Ching (314551069)

Usage:
    # Training
    python main.py --mode train \
        --data_dir /nfs_drive/yunching/hw4_realse_dataset/train \
        --save_dir ./checkpoints --epochs 150 --batch_size 8 --patch_size 256

    # Inference
    python main.py --mode test \
        --test_dir /nfs_drive/yunching/hw4_realse_dataset/test/degraded \
        --checkpoint ./checkpoints/best.pth --output pred.npz
"""

import csv
import math
import os
import random
import argparse
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import GradScaler, autocast
from PIL import Image
import torchvision.transforms.functional as TF
from tqdm import tqdm


# ══════════════════════════════ Utilities ═══════════════════════════════════

def set_seed(seed: int = 42) -> None:
    """Fix random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Compute PSNR (dB) between two image tensors in [0, 1]."""
    mse = F.mse_loss(pred.clamp(0, 1), target.clamp(0, 1)).item()
    return float('inf') if mse == 0 else 20 * math.log10(1.0 / math.sqrt(mse))


def pad_to_multiple(x: torch.Tensor, factor: int = 8):
    """Pad spatial dims to be divisible by factor (reflect padding)."""
    _, _, H, W = x.shape
    ph = (factor - H % factor) % factor
    pw = (factor - W % factor) % factor
    if ph > 0 or pw > 0:
        x = F.pad(x, (0, pw, 0, ph), mode='reflect')
    return x, H, W


# ══════════════════════════════ CSV Logger ══════════════════════════════════

class CSVLogger:
    """Append-mode CSV logger for training metrics.

    Columns: epoch, train_loss, val_psnr, lr
    Writes a header on first creation; resumes correctly when training is
    resumed from a checkpoint (skips re-writing the header).
    """

    def __init__(self, path: str):
        self.path = path
        self._fieldnames = ['epoch', 'train_loss', 'val_psnr', 'lr']
        # Write header only if the file does not exist yet
        if not os.path.isfile(path):
            with open(path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self._fieldnames)
                writer.writeheader()

    def write(self, epoch: int, train_loss: float,
              val_psnr: float, lr: float) -> None:
        with open(self.path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self._fieldnames)
            writer.writerow({
                'epoch': epoch,
                'train_loss': round(train_loss, 6),
                'val_psnr': round(val_psnr, 4),
                'lr': f'{lr:.2e}',
            })


# ══════════════════════════════ Dataset ═════════════════════════════════════

def build_pairs(root: str) -> list:
    """Collect (degraded_path, clean_path) pairs from the training root."""
    degraded_dir = Path(root) / 'degraded'
    clean_dir = Path(root) / 'clean'
    pairs = []
    for deg_path in sorted(degraded_dir.glob('*.png')):
        stem = deg_path.stem
        if stem.startswith('rain-'):
            idx = stem[5:]
            clean_name = f'rain_clean-{idx}.png'
        elif stem.startswith('snow-'):
            idx = stem[5:]
            clean_name = f'snow_clean-{idx}.png'
        else:
            continue
        clean_path = clean_dir / clean_name
        if clean_path.exists():
            pairs.append((str(deg_path), str(clean_path)))
    return pairs


class PatchDataset(Dataset):
    """Training dataset: random cropped + augmented (degraded, clean) patches."""

    def __init__(
            self,
            pairs: list,
            patch_size: int = 256,
            augment: bool = True):
        self.pairs = pairs
        self.patch_size = patch_size
        self.augment = augment

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        # Retry with a different sample on NFS / I/O errors
        for attempt in range(len(self.pairs)):
            try:
                real_idx = (idx + attempt) % len(self.pairs)
                deg = TF.to_tensor(
                    Image.open(self.pairs[real_idx][0]).convert('RGB')
                )
                clean = TF.to_tensor(
                    Image.open(self.pairs[real_idx][1]).convert('RGB')
                )
                break
            except OSError:
                continue
        else:
            raise RuntimeError('All samples failed — check NFS mount.')

        # Random crop
        _, H, W = deg.shape
        ps = self.patch_size
        if H >= ps and W >= ps:
            i = random.randint(0, H - ps)
            j = random.randint(0, W - ps)
            deg = deg[:, i:i + ps, j:j + ps]
            clean = clean[:, i:i + ps, j:j + ps]

        # Augmentation: horizontal flip, vertical flip, 90-degree rotation
        if self.augment:
            if random.random() > 0.5:
                deg, clean = TF.hflip(deg), TF.hflip(clean)
            if random.random() > 0.5:
                deg, clean = TF.vflip(deg), TF.vflip(clean)
            k = random.randint(0, 3)
            if k:
                deg = torch.rot90(deg, k, dims=[-2, -1])
                clean = torch.rot90(clean, k, dims=[-2, -1])

        return deg, clean


class TestDataset(Dataset):
    """Inference dataset: degraded test images (no ground truth)."""

    def __init__(self, test_dir: str):
        self.paths = sorted(
            Path(test_dir).glob('*.png'),
            key=lambda p: int(p.stem)
        )

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        path = self.paths[idx]
        img = TF.to_tensor(Image.open(str(path)).convert('RGB'))
        return img, path.name


# ══════════════════════════════ Model ═══════════════════════════════════════

# ── Layer Normalisation ───────────────────────────────────────────────────

class BiasFreeLayerNorm(nn.Module):
    def __init__(self, normalized_shape: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBiasLayerNorm(nn.Module):
    def __init__(self, normalized_shape: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    """Channel-wise layer norm for (B, C, H, W) tensors."""

    def __init__(self, dim: int, bias: bool = False):
        super().__init__()
        self.body = WithBiasLayerNorm(dim) if bias else BiasFreeLayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = x.shape[-2:]
        # (B,C,H,W) -> (B,H*W,C) -> norm -> (B,C,H,W)
        x_flat = x.flatten(2).transpose(1, 2)
        return self.body(x_flat).transpose(1, 2).view(*x.shape[:2], H, W)


# ── Attention & Feed-Forward ──────────────────────────────────────────────

class MDTA(nn.Module):
    """Multi-Dconv Head Transposed Attention (Restormer)."""

    def __init__(self, dim: int, num_heads: int, bias: bool = False):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=bias)
        self.qkv_dw = nn.Conv2d(
            dim * 3, dim * 3, 3, 1, 1, groups=dim * 3, bias=bias
        )
        self.project_out = nn.Conv2d(dim, dim, 1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        qkv = self.qkv_dw(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)
        head_dim = C // self.num_heads
        q = q.reshape(B, self.num_heads, head_dim, H * W)
        k = k.reshape(B, self.num_heads, head_dim, H * W)
        v = v.reshape(B, self.num_heads, head_dim, H * W)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = (attn @ v).reshape(B, C, H, W)
        return self.project_out(out)


class GDFN(nn.Module):
    """Gated-Dconv Feed-Forward Network (Restormer)."""

    def __init__(self, dim: int, expansion: float = 2.66, bias: bool = False):
        super().__init__()
        hidden = int(dim * expansion)
        self.project_in = nn.Conv2d(dim, hidden * 2, 1, bias=bias)
        self.dw = nn.Conv2d(
            hidden * 2, hidden * 2, 3, 1, 1, groups=hidden * 2, bias=bias
        )
        self.project_out = nn.Conv2d(hidden, dim, 1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = self.dw(self.project_in(x)).chunk(2, dim=1)
        return self.project_out(F.gelu(x1) * x2)


class TransformerBlock(nn.Module):
    """Restormer-style transformer block: MDTA + GDFN with residuals."""

    def __init__(self, dim: int, heads: int,
                 ffn_expansion: float = 2.66, bias: bool = False):
        super().__init__()
        self.norm1 = LayerNorm(dim, bias)
        self.attn = MDTA(dim, heads, bias)
        self.norm2 = LayerNorm(dim, bias)
        self.ffn = GDFN(dim, ffn_expansion, bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


# ── Down / Up sampling ────────────────────────────────────────────────────

class Downsample(nn.Module):
    """Halve spatial dims, double channels: (B,C,H,W) → (B,2C,H/2,W/2)."""

    def __init__(self, n_feat: int):
        super().__init__()
        # Conv(C→C/2) + PixelUnshuffle(2) → C/2 * 4 = 2C
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat // 2, 3, 1, 1, bias=False),
            nn.PixelUnshuffle(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class Upsample(nn.Module):
    """Double spatial dims, halve channels: (B,C,H,W) → (B,C/2,2H,2W)."""

    def __init__(self, n_feat: int):
        super().__init__()
        # Conv(C→2C) + PixelShuffle(2) → 2C / 4 = C/2
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat * 2, 3, 1, 1, bias=False),
            nn.PixelShuffle(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


# ── Prompt Generation Block ───────────────────────────────────────────────

class PromptGenBlock(nn.Module):
    """
    Prompt Generation Block — core contribution of PromptIR.

    Maintains a learnable pool of N prompt tensors.  For each input
    feature map, it computes attention weights via a linear layer over
    global-average-pooled features, produces a weighted-sum prompt,
    resizes it to the feature-map resolution, and returns it to be
    added into the decoder.

    Reference:
        Potlapalli et al., "PromptIR: Prompting for All-in-One Blind
        Image Restoration", NeurIPS 2023.
    """

    def __init__(self, prompt_dim: int, prompt_size: int,
                 lin_dim: int, n_prompts: int = 5):
        super().__init__()
        # Learnable prompt pool: (1, N, C_p, ps, ps)
        self.prompt_param = nn.Parameter(
            torch.rand(1, n_prompts, prompt_dim, prompt_size, prompt_size)
        )
        self.linear = nn.Linear(lin_dim, n_prompts)
        self.conv = nn.Conv2d(prompt_dim, prompt_dim, 3, 1, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        # Global average pool → (B, C)
        emb = x.mean(dim=(-2, -1))
        # Soft attention weights over prompt pool → (B, N)
        w = F.softmax(self.linear(emb), dim=-1)
        # Weighted sum over pool → (B, C_p, ps, ps)
        pool = self.prompt_param.expand(B, -1, -1, -1, -1)
        prompt = (w.view(B, -1, 1, 1, 1) * pool).sum(dim=1)
        # Resize to feature-map resolution
        prompt = F.interpolate(
            prompt, size=(H, W), mode='bilinear', align_corners=False
        )
        return self.conv(prompt)


# ── PromptIR ──────────────────────────────────────────────────────────────

class PromptIR(nn.Module):
    """
    PromptIR: Prompting for All-in-One Blind Image Restoration.

    A four-level U-Net transformer (based on Restormer) augmented with
    Prompt Generation Blocks at each decoder stage.  The prompts inject
    degradation-aware context that guides the decoder to restore images
    afflicted by different distortion types (rain, snow, …) with a
    single unified model.

    Channel schedule (dim=48):
        Encoder level 1 : 48   channels  (H   × W  )
        Encoder level 2 : 96   channels  (H/2 × W/2)
        Encoder level 3 : 192  channels  (H/4 × W/4)
        Bottleneck       : 384  channels  (H/8 × W/8)
        (decoder mirrors encoder with prompts injected at each level)
    """

    def __init__(
        self,
        inp_channels: int = 3,
        out_channels: int = 3,
        dim: int = 48,
        num_blocks: list = None,
        num_refinement_blocks: int = 4,
        heads: list = None,
        ffn_expansion: float = 2.66,
        bias: bool = False,
        prompt_size: int = 64,
        n_prompts: int = 5,
    ):
        super().__init__()
        if num_blocks is None:
            num_blocks = [4, 6, 6, 8]
        if heads is None:
            heads = [1, 2, 4, 8]

        # ── Initial embedding ──────────────────────────────────────
        self.patch_embed = nn.Conv2d(inp_channels, dim, 3, 1, 1, bias=bias)

        # ── Encoder ───────────────────────────────────────────────
        self.enc1 = nn.Sequential(*[
            TransformerBlock(dim, heads[0], ffn_expansion, bias)
            for _ in range(num_blocks[0])
        ])
        self.down1 = Downsample(dim)            # dim → dim*2

        self.enc2 = nn.Sequential(*[
            TransformerBlock(dim * 2, heads[1], ffn_expansion, bias)
            for _ in range(num_blocks[1])
        ])
        self.down2 = Downsample(dim * 2)        # dim*2 → dim*4

        self.enc3 = nn.Sequential(*[
            TransformerBlock(dim * 4, heads[2], ffn_expansion, bias)
            for _ in range(num_blocks[2])
        ])
        self.down3 = Downsample(dim * 4)        # dim*4 → dim*8

        # ── Bottleneck ─────────────────────────────────────────────
        self.bottleneck = nn.Sequential(*[
            TransformerBlock(dim * 8, heads[3], ffn_expansion, bias)
            for _ in range(num_blocks[3])
        ])

        # ── Decoder level 3 (dim*4) ────────────────────────────────
        self.up3 = Upsample(dim * 8)            # dim*8 → dim*4
        self.prompt3 = PromptGenBlock(
            prompt_dim=dim * 4, prompt_size=prompt_size,
            lin_dim=dim * 4, n_prompts=n_prompts,
        )
        # cat([up3+p3, enc3]) → dim*8 → reduce → dim*4
        self.reduce3 = nn.Conv2d(dim * 8, dim * 4, 1, bias=bias)
        self.dec3 = nn.Sequential(*[
            TransformerBlock(dim * 4, heads[2], ffn_expansion, bias)
            for _ in range(num_blocks[2])
        ])

        # ── Decoder level 2 (dim*2) ────────────────────────────────
        self.up2 = Upsample(dim * 4)            # dim*4 → dim*2
        self.prompt2 = PromptGenBlock(
            prompt_dim=dim * 2, prompt_size=prompt_size * 2,
            lin_dim=dim * 2, n_prompts=n_prompts,
        )
        self.reduce2 = nn.Conv2d(dim * 4, dim * 2, 1, bias=bias)
        self.dec2 = nn.Sequential(*[
            TransformerBlock(dim * 2, heads[1], ffn_expansion, bias)
            for _ in range(num_blocks[1])
        ])

        # ── Decoder level 1 (dim) ──────────────────────────────────
        self.up1 = Upsample(dim * 2)            # dim*2 → dim
        self.prompt1 = PromptGenBlock(
            prompt_dim=dim, prompt_size=prompt_size * 4,
            lin_dim=dim, n_prompts=n_prompts,
        )
        self.reduce1 = nn.Conv2d(dim * 2, dim, 1, bias=bias)
        self.dec1 = nn.Sequential(*[
            TransformerBlock(dim, heads[0], ffn_expansion, bias)
            for _ in range(num_blocks[0])
        ])

        # ── Refinement & output ────────────────────────────────────
        self.refinement = nn.Sequential(*[
            TransformerBlock(dim, heads[0], ffn_expansion, bias)
            for _ in range(num_refinement_blocks)
        ])
        self.output_conv = nn.Conv2d(dim, out_channels, 3, 1, 1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.patch_embed(x)

        # Encoder
        e1 = self.enc1(feat)                        # (B, dim,   H,   W)
        e2 = self.enc2(self.down1(e1))              # (B, dim*2, H/2, W/2)
        e3 = self.enc3(self.down2(e2))              # (B, dim*4, H/4, W/4)
        lat = self.bottleneck(self.down3(e3))       # (B, dim*8, H/8, W/8)

        # Decoder — level 3
        d3 = self.up3(lat)                          # (B, dim*4, H/4, W/4)
        p3 = self.prompt3(d3)                       # (B, dim*4, H/4, W/4)
        d3 = self.reduce3(torch.cat([d3 + p3, e3], dim=1))
        d3 = self.dec3(d3)                          # (B, dim*4, H/4, W/4)

        # Decoder — level 2
        d2 = self.up2(d3)                           # (B, dim*2, H/2, W/2)
        p2 = self.prompt2(d2)
        d2 = self.reduce2(torch.cat([d2 + p2, e2], dim=1))
        d2 = self.dec2(d2)                          # (B, dim*2, H/2, W/2)

        # Decoder — level 1
        d1 = self.up1(d2)                           # (B, dim,   H,   W)
        p1 = self.prompt1(d1)
        d1 = self.reduce1(torch.cat([d1 + p1, e1], dim=1))
        d1 = self.dec1(d1)                          # (B, dim,   H,   W)

        # Refinement + global residual
        out = self.output_conv(self.refinement(d1)) + x
        return out


# ══════════════════════════════ Loss Functions ═══════════════════════════════

class FrequencyLoss(nn.Module):
    """
    Frequency domain loss (FFT Loss).

    Penalises differences in the amplitude spectrum of pred vs target,
    encouraging the model to recover high-frequency details (edges,
    textures) that plain L1 loss tends to over-smooth.
    """

    def forward(
            self,
            pred: torch.Tensor,
            target: torch.Tensor) -> torch.Tensor:
        pred_fft = torch.fft.rfft2(pred, norm='ortho')
        target_fft = torch.fft.rfft2(target, norm='ortho')
        return F.l1_loss(torch.abs(pred_fft), torch.abs(target_fft))


class SSIMLoss(nn.Module):
    """
    SSIM-based loss for image restoration.

    Maximising SSIM (structural similarity) encourages the model to
    preserve luminance, contrast, and structure — complementary to L1
    which penalises pixel-level magnitude errors uniformly.
    Loss = 1 - SSIM, so minimising this maximises SSIM / PSNR.
    """

    def __init__(self, window_size: int = 11, sigma: float = 1.5):
        super().__init__()
        self.window_size = window_size
        self.sigma = sigma
        # Pre-build Gaussian window (fixed, not learnable)
        kernel = self._gaussian_kernel(window_size, sigma)
        # Shape: (1, 1, window_size, window_size) — broadcast over C
        self.register_buffer('window', kernel)

    @staticmethod
    def _gaussian_kernel(size: int, sigma: float) -> torch.Tensor:
        coords = torch.arange(size, dtype=torch.float32) - size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        kernel = g.outer(g)
        return kernel.unsqueeze(0).unsqueeze(0)

    def _ssim(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        C1, C2 = 0.01 ** 2, 0.03 ** 2
        B, C, H, W = x.shape
        win = self.window.expand(C, 1, -1, -1)
        pad = self.window_size // 2

        mu_x = F.conv2d(x, win, padding=pad, groups=C)
        mu_y = F.conv2d(y, win, padding=pad, groups=C)
        mu_x2, mu_y2, mu_xy = mu_x ** 2, mu_y ** 2, mu_x * mu_y

        sigma_x2 = F.conv2d(x * x, win, padding=pad, groups=C) - mu_x2
        sigma_y2 = F.conv2d(y * y, win, padding=pad, groups=C) - mu_y2
        sigma_xy = F.conv2d(x * y, win, padding=pad, groups=C) - mu_xy

        num = (2 * mu_xy + C1) * (2 * sigma_xy + C2)
        den = (mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2)
        return (num / den).mean()

    def forward(
            self,
            pred: torch.Tensor,
            target: torch.Tensor) -> torch.Tensor:
        return 1.0 - self._ssim(pred.clamp(0, 1), target.clamp(0, 1))

# ══════════════════════════════ Training ════════════════════════════════════


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    fft_weight: float = 0.0,
    ssim_weight: float = 0.0,
) -> float:
    model.train()
    freq_criterion = FrequencyLoss().to(device)
    ssim_criterion = SSIMLoss().to(device)
    total_loss = 0.0
    for deg, clean in tqdm(loader, desc='  train', leave=False):
        deg = deg.to(device, non_blocking=True)
        clean = clean.to(device, non_blocking=True)
        optimizer.zero_grad()
        with autocast('cuda'):
            pred = model(deg)
            loss = F.l1_loss(pred, clean)
            if fft_weight > 0:
                loss = loss + fft_weight * freq_criterion(pred, clean)
            if ssim_weight > 0:
                loss = loss + ssim_weight * ssim_criterion(pred, clean)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * deg.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def validate(
    model: nn.Module,
    val_pairs: list,
    device: torch.device,
    max_images: int = 100,
) -> float:
    """Evaluate PSNR on full-resolution validation images."""
    model.eval()
    subset = val_pairs[:max_images]
    psnr_vals = []
    for deg_path, clean_path in subset:
        deg = TF.to_tensor(
            Image.open(deg_path).convert('RGB')
        ).unsqueeze(0).to(device)
        clean = TF.to_tensor(
            Image.open(clean_path).convert('RGB')
        ).unsqueeze(0).to(device)
        deg_pad, H, W = pad_to_multiple(deg, factor=8)
        with autocast('cuda'):
            pred = model(deg_pad)[:, :, :H, :W]
        psnr_vals.append(compute_psnr(pred, clean))
    return float(np.mean(psnr_vals))


# ══════════════════════════════ Inference ═══════════════════════════════════

def _tta_predict(model: nn.Module, img: torch.Tensor) -> torch.Tensor:
    """8-fold TTA: average predictions over hflip x vflip x 2 rotations."""
    preds = []
    for hflip in [False, True]:
        for vflip in [False, True]:
            for k in [0, 1]:
                x = img
                if hflip:
                    x = torch.flip(x, dims=[-1])
                if vflip:
                    x = torch.flip(x, dims=[-2])
                if k:
                    x = torch.rot90(x, k, dims=[-2, -1])
                x_pad, H, W = pad_to_multiple(x, factor=8)
                with autocast('cuda'):
                    out = model(x_pad)[:, :, :H, :W]
                if k:
                    out = torch.rot90(out, -k, dims=[-2, -1])
                if vflip:
                    out = torch.flip(out, dims=[-2])
                if hflip:
                    out = torch.flip(out, dims=[-1])
                preds.append(out)
    return torch.stack(preds, dim=0).mean(dim=0).clamp(0, 1)


@torch.no_grad()
def run_inference(
    model: nn.Module,
    test_dir: str,
    device: torch.device,
    output_path: str,
    use_tta: bool = True,
) -> None:
    """Run inference (with optional 8-fold TTA) and save results as pred.npz."""
    model.eval()
    dataset = TestDataset(test_dir)
    results = {}
    mode = 'TTA x8' if use_tta else 'single'
    for img, name in tqdm(dataset, desc=f'inference [{mode}]'):
        img = img.unsqueeze(0).to(device)
        if use_tta:
            pred = _tta_predict(model, img)
        else:
            img_pad, H, W = pad_to_multiple(img, factor=8)
            with autocast('cuda'):
                pred = model(img_pad)[:, :, :H, :W].clamp(0, 1)
        # Store as (3, H, W) uint8
        arr = (
            pred.squeeze(0).cpu().float().numpy() *
            255).round().astype(
            np.uint8)
        results[name] = arr
    np.savez(output_path, **results)
    print(f'Saved {len(results)} images \u2192 {output_path}')


# ══════════════════════════════ Main ════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='HW4 — PromptIR Image Restoration')
    parser.add_argument('--mode', choices=['train', 'test'], default='train',
                        help='train or test')

    # Data paths
    parser.add_argument('--data_dir',
                        default='/nfs_drive/yunching/hw4_realse_dataset/train',
                        help='Training root (contains degraded/ and clean/)')
    parser.add_argument(
        '--test_dir',
        default='/nfs_drive/yunching/hw4_realse_dataset/test/degraded',
        help='Test images directory')
    parser.add_argument('--val_split', type=float, default=0.1,
                        help='Fraction of training pairs for validation')

    # Training hyper-parameters
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--patch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)

    # Model hyper-parameters
    parser.add_argument('--dim', type=int, default=48,
                        help='Base channel dimension')
    parser.add_argument('--n_prompts', type=int, default=5,
                        help='Number of prompt components')

    # Checkpoint / output
    parser.add_argument('--save_dir', default='./checkpoints')
    parser.add_argument('--checkpoint', default='./checkpoints/best.pth',
                        help='Checkpoint to load for --mode test')
    parser.add_argument('--resume', default='',
                        help='Checkpoint to resume training from')
    parser.add_argument('--output', default='pred.npz',
                        help='Output .npz for test mode')
    parser.add_argument('--device', default='cuda:2')
    parser.add_argument('--fft_weight', type=float, default=0.0,
                        help='Weight for frequency loss (0 = disabled)')
    parser.add_argument('--ssim_weight', type=float, default=0.0,
                        help='Weight for SSIM loss (0 = disabled)')
    parser.add_argument('--ft_lr', type=float, default=2e-5,
                        help='Learning rate when using --finetune')
    parser.add_argument(
        '--finetune',
        default='',
        help='Load model weights only (reset optimizer/scheduler)')
    parser.add_argument('--no_tta', action='store_true',
                        help='Disable TTA during inference')

    return parser.parse_args()


def build_model(args: argparse.Namespace, device: torch.device) -> PromptIR:
    model = PromptIR(
        dim=args.dim,
        num_blocks=[4, 6, 6, 8],
        num_refinement_blocks=4,
        heads=[1, 2, 4, 8],
        ffn_expansion=2.66,
        bias=False,
        prompt_size=64,
        n_prompts=args.n_prompts,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'PromptIR parameters: {n_params / 1e6:.2f} M')
    return model


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(
        args.device if torch.cuda.is_available() else 'cpu'
    )
    print(f'Device: {device}')

    model = build_model(args, device)

    # ── Inference mode ─────────────────────────────────────────────
    if args.mode == 'test':
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt['model'])
        print(f'Loaded checkpoint from {args.checkpoint}')
        run_inference(model, args.test_dir, device, args.output,
                      use_tta=not args.no_tta)
        return

    # ── Training mode ──────────────────────────────────────────────
    os.makedirs(args.save_dir, exist_ok=True)

    all_pairs = build_pairs(args.data_dir)
    random.shuffle(all_pairs)
    n_val = max(1, int(len(all_pairs) * args.val_split))
    val_pairs = all_pairs[:n_val]
    train_pairs = all_pairs[n_val:]
    print(f'Train: {len(train_pairs)} pairs | Val: {len(val_pairs)} pairs')

    train_ds = PatchDataset(
        train_pairs,
        patch_size=args.patch_size,
        augment=True)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # Use ft_lr when fine-tuning from an existing checkpoint
    effective_lr = args.ft_lr if args.finetune else args.lr
    optimizer = AdamW(
        model.parameters(),
        lr=effective_lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
    )
    scheduler = CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )
    scaler = GradScaler('cuda')
    best_psnr = 0.0
    start_epoch = 1

    # CSV log — resumes correctly if file already exists
    log_path = os.path.join(args.save_dir, 'train_log.csv')
    csv_logger = CSVLogger(log_path)
    print(f'Training log → {log_path}')

    # Optional resume
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        best_psnr = ckpt.get('best_psnr', 0.0)
        start_epoch = ckpt['epoch'] + 1
        print(
            f'Resumed from epoch {
                ckpt["epoch"]} (best PSNR: {
                best_psnr:.2f})')

    # Finetune: load model weights only, fresh optimizer/scheduler
    if args.finetune and os.path.isfile(args.finetune):
        ckpt = torch.load(args.finetune, map_location=device)
        model.load_state_dict(ckpt['model'])
        print(
            f'Fine-tuning from {args.finetune} (fresh optimizer + scheduler)')

    for epoch in range(start_epoch, args.epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scaler, device,
            fft_weight=args.fft_weight,
            ssim_weight=args.ssim_weight,
        )
        scheduler.step()

        val_psnr = validate(model, val_pairs, device, max_images=100)

        improved = '↑' if val_psnr > best_psnr else ' '
        current_lr = scheduler.get_last_lr()[0]
        print(
            f'Epoch [{epoch:3d}/{args.epochs}]  '
            f'Loss: {train_loss:.4f}  '
            f'Val PSNR: {val_psnr:.2f} dB  '
            f'LR: {current_lr:.2e}  {improved}'
        )
        csv_logger.write(epoch, train_loss, val_psnr, current_lr)

        state = {
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'best_psnr': best_psnr,
        }
        torch.save(state, os.path.join(args.save_dir, 'latest.pth'))

        if val_psnr > best_psnr:
            best_psnr = val_psnr
            state['best_psnr'] = best_psnr
            torch.save(state, os.path.join(args.save_dir, 'best.pth'))
            print(f'  → New best saved: {best_psnr:.2f} dB')


if __name__ == '__main__':
    main()
