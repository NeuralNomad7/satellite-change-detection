# Architecture: Siamese U-Net for Change Detection

## Overview

Change detection from satellite imagery is framed as a pixel-level binary segmentation problem. Given two co-registered images of the same geographic area taken at different times (T1 and T2), the model predicts a binary mask where 1 indicates change and 0 indicates no change.

## Why Siamese U-Net?

Standard semantic segmentation models (plain U-Net, DeepLab) take a single input. Change detection requires comparing two images. The Siamese architecture solves this elegantly: two encoder branches with shared weights extract features from each time step independently, then a fusion module combines them before decoding. Weight sharing ensures both images are embedded in the same feature space, making the comparison meaningful.

U-Net's skip connections are critical here because change detection requires both high-level semantic understanding ("is this a building?") and low-level spatial precision ("exactly which pixels changed?"). The encoder captures the semantics, the decoder recovers spatial detail, and skip connections bridge the two.

## Architecture Diagram

```
T1 Image ──> [Encoder (ResNet-34)] ──> F1_1, F1_2, F1_3, F1_4, F1_5
                    │ (shared weights)
T2 Image ──> [Encoder (ResNet-34)] ──> F2_1, F2_2, F2_3, F2_4, F2_5
                                              │
                                    [Feature Fusion]
                                     |F1 - F2| + concat(F1, F2)
                                              │
                                    [Decoder with Skip Connections]
                                              │
                                    [1x1 Conv + Sigmoid]
                                              │
                                      Change Mask (H x W)
```

## Encoder

We use ResNet-34 pretrained on ImageNet as the encoder backbone. Even though ImageNet contains natural photos (not satellite imagery), the low-level features (edges, textures, color gradients) transfer surprisingly well to remote sensing. The encoder produces feature maps at 5 spatial scales (1/2, 1/4, 1/8, 1/16, 1/32 of input resolution).

The first convolutional layer is modified to accept the number of input spectral bands (e.g., 13 for full Sentinel-2, or 3 for RGB-only). When using more than 3 bands, the pretrained weights for the first layer are replicated and averaged across the extra channels.

## Feature Fusion

At each encoder scale, we compute two types of feature interactions:

1. **Absolute difference**: `|F1 - F2|` captures where features changed regardless of direction. This is the primary change signal.
2. **Concatenation**: `concat(F1, F2)` preserves the original feature context so the decoder can reason about what the area looked like before and after.

Both are concatenated and passed through a 1x1 convolution to reduce channel dimensionality before entering the decoder.

## Decoder

The decoder follows the standard U-Net pattern: each stage upsamples by 2x using transposed convolution, concatenates the corresponding skip connection from the fused encoder features, and applies two 3x3 conv + BatchNorm + ReLU blocks.

## Deep Supervision

During training, auxiliary classification heads are attached at each decoder scale. These produce intermediate change predictions at lower resolutions. The total loss is a weighted sum of all scales:

```
L_total = L_final + 0.4 * L_scale4 + 0.2 * L_scale3 + 0.1 * L_scale2
```

Deep supervision provides gradient signal throughout the network and helps the encoder learn meaningful features at every scale. It is disabled at inference time.

## Loss Function

We use a combination of Binary Cross-Entropy and Dice Loss:

```
L = 0.5 * BCE(pred, target) + 0.5 * DiceLoss(pred, target)
```

BCE handles pixel-level classification well but struggles with class imbalance (most pixels are typically "no change"). Dice Loss directly optimizes the F1 score and is robust to imbalance. The combination gives stable training and strong metrics.

## Training Details

- **Optimizer**: AdamW (lr=1e-4, weight_decay=1e-4)
- **Scheduler**: Cosine annealing with warm restarts
- **Batch size**: 16 (256x256 patches)
- **Augmentations**: Random flips, rotations, brightness/contrast jitter, Gaussian noise
- **Mixed precision**: FP16 via PyTorch AMP for 2x memory reduction and faster training
- **Early stopping**: Based on validation F1, patience of 15 epochs

## Design Decisions

**Why ResNet-34 over ResNet-50?** ResNet-34 is significantly lighter (21M vs 25M params) and trains faster, with negligible accuracy difference on change detection benchmarks. For a portfolio project where others will clone and run it, faster training matters.

**Why not a transformer backbone?** Vision transformers (ViT, Swin) can outperform CNNs on large datasets, but they need more data and compute. ResNet-34 trains well on modestly sized change detection datasets (LEVIR-CD has ~600 image pairs). The CNN approach is more practical and reproducible.

**Why patch-based training?** Sentinel-2 scenes are huge (10980x10980 pixels at 10m resolution). We can't fit a full scene in GPU memory. Instead, we extract 256x256 patches with overlap during inference and stitch the predictions back together. This is standard practice in remote sensing.
