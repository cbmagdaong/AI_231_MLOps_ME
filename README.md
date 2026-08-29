# AI_231_MLOps_ME

MNIST 3-layer CNN built **entirely with `einops` / `einsum`** — every core
operation (convolution, max-pooling, linear layers, softmax, cross-entropy and
gradients) is written by hand. No `nn.Conv2d`, `nn.Linear`, or `F.conv2d` in the
model. Includes the executed notebook, full training logs (CPU **and** GPU runs),
and 4×4 ground-truth-vs-prediction grids.

## Architecture (41,866 parameters)

```
input (N,1,28,28)
  → Conv1  1 → 32 filters, 3×3, ReLU   → MaxPool  → (N,32,13,13)
  → Conv2  32 → 64 filters, 3×3, ReLU  → MaxPool  → (N,64,6,6)
  → flatten (2304) → FC 2304 → 10 → softmax
```

| Operation | Hand-written with | Verified against PyTorch |
|-----------|-------------------|--------------------------|
| Convolution | `rearrange` im2col + `einsum(cols, wmat, 'n k m, o k -> n o m')` | `F.conv2d` (Δ 1.9e-06) |
| Max-pooling | `rearrange` to 2×2 blocks + `reduce(..., 'max')` | `F.max_pool2d` (Δ 0) |
| Linear | `einsum(x, w, 'n i, o i -> n o')` | `x@wᵀ+b` (Δ 0) |
| Softmax / Cross-entropy | `einsum(logp, onehot, 'n c, n c -> n')` | `F.softmax` / `F.cross_entropy` (Δ 0) |

## Contents

| File | Description |
|------|-------------|
| `mnist_einsum_cnn.ipynb` | Executed notebook (CPU run) — all ops, correctness checks, 5-epoch training, test accuracy, 4×4 grid |
| `mnist_grid.png` | 4×4 grid, 16 test samples, GT vs prediction (CPU run) |
| `mnist_grid_gpu0.png` | 4×4 grid, 16 test samples, GT vs prediction (GPU run, A100) |
| `training_logs/training_log.txt` | Per-epoch CPU training log |
| `training_logs/eval_test_accuracy.txt` | CPU test-split accuracy |
| `training_logs/gpu_run_gpu0.log` | Full GPU run log (device banner + per-epoch + test accuracy) |
| `training_logs/eval_test_accuracy_gpu.txt` | GPU test-split accuracy |
| `training_logs/op_correctness_checks.txt` | einops/einsum ops vs PyTorch reference |

## CPU vs GPU — comparison

Same model, same hyper-parameters (Adam, lr=1e-3, batch 128, seed 0), 5 epochs,
60k train / 10k test. The **only** difference is the compute device.

### Setup

| | CPU run | GPU run |
|---|---------|---------|
| Host | Local workstation (Apple Silicon, macOS) | `ai-n002.hpc.coe.upd.edu.ph` (UPD COE HPC) |
| Device | `cpu` | `cuda:0` — **NVIDIA A100-SXM4-40GB** (42.4 GB) |
| PyTorch | 2.8.0 (CPU build) | 2.5.1+cu121 (CUDA 12.1) |
| GPU pinning | — | `CUDA_VISIBLE_DEVICES=0` |

### Per-epoch wall-clock

| Epoch | CPU | GPU | Speedup |
|:-----:|:---:|:---:|:-------:|
| 1 | 16.7 s | 5.0 s | 3.3× |
| 2 | 17.3 s | 2.6 s | 6.7× |
| 3 | 16.8 s | 2.5 s | 6.7× |
| 4 | 17.9 s | 2.5 s | 7.2× |
| 5 | 16.8 s | 2.5 s | 6.7× |
| **Total** | **85.5 s** | **15.1 s** | **5.7×** |
| **Avg / epoch** | **17.1 s** | **3.0 s** | **5.7×** |

### Accuracy

| Metric | CPU | GPU | Δ |
|--------|:---:|:---:|:---:|
| Final train acc (epoch 5) | 99.11% | 99.21% | +0.10 pp |
| **Test-split accuracy** | **98.74%** (9874/10000) | **98.73%** (9873/10000) | −0.01 pp |

### Takeaways

- **~5.7× faster on the A100** (85.5 s → 15.1 s total for 5 epochs). The first
  GPU epoch (5.0 s) is slower than the rest because it absorbs one-time CUDA
  initialization / kernel compilation; steady-state epochs settle at ~2.5 s.
- **Accuracy is effectively identical** (98.74% vs 98.73%, within 1 sample).
  The tiny difference is expected — the two runs use different PyTorch builds and
  CPU/GPU floating-point accumulation order, so results are reproducible to
  within noise, not bit-for-bit.
- **Why the GPU run matters here:** the model is small (41,866 params) and the
  dataset is tiny (60k 28×28 images), so even the CPU run finishes in under 90 s.
  The speedup would grow substantially with larger networks, bigger batches, or
  higher-resolution inputs — the classic regime where an A100 pays off.

## Reproducing

```bash
# CPU (any machine with torch + einops + torchvision)
jupyter notebook mnist_einsum_cnn.ipynb

# GPU (on the HPC node, pinned to a free A100)
CUDA_VISIBLE_DEVICES=0 python run_cnn_gpu0.py
```

> Note on the GPU run: the HPC node ships an old `torchvision` whose `MNIST`
> downloader points at a dead mirror, so the GPU script reads the raw MNIST IDX
> files directly with NumPy. The model and every einops/einsum operation are
> unchanged.
