# Hybrid CNN + Vision Transformer for CIFAR-100 Classification

## Table of Contents
- [Objective](#objective)
- [Problem Statement](#problem-statement)
- [Methodology](#methodology)
- [Architecture](#architecture)
- [Implementation Details](#implementation-details)
- [Code Structure](#code-structure)
- [Results](#results)
- [Analysis](#analysis)
- [Limitations](#limitations)
- [Future Work](#future-work)
- [References](#references)

---

## Objective

This project implements a hybrid architecture combining Convolutional Neural Networks (CNNs) and Vision Transformers (ViT) for image classification on the CIFAR-100 dataset. The primary objective is to leverage the inductive biases of CNNs for low-level feature extraction while utilizing the self-attention mechanisms of transformers for global context modeling.

---

## Problem Statement

### Challenge

Image classification on CIFAR-100 presents several challenges:

- **High inter-class similarity**: Many classes in CIFAR-100 share visual characteristics
- **Limited data per class**: Only 500 training images per class
- **Low resolution**: Images are only 32×32 pixels
- **Fine-grained classification**: 100 classes require learning subtle distinctions

### Traditional Approaches

Pure Vision Transformers, while powerful, often struggle with small datasets due to their lack of inductive biases. Standard CNNs, conversely, may miss long-range dependencies crucial for distinguishing similar classes.

---

## Methodology

### Hybrid Architecture Approach

The solution employs a hybrid architecture that:

1. **Replaces linear patch embedding** with a convolutional stem
2. **Extracts hierarchical features** using strided convolutions
3. **Applies transformer blocks** for global reasoning on extracted features
4. **Combines local and global processing** for improved performance

### Key Design Decisions

**Convolutional Stem**
- Three-layer CNN progressively downsamples the input
- BatchNorm and ReLU activations for stable training
- Reduces spatial dimensions from 32×32 to 8×8

**Transformer Configuration**
- 8 transformer blocks with 6 attention heads
- Embedding dimension of 384 for efficient computation
- Stochastic depth for regularization

**Training Strategy**
- Heavy data augmentation (AutoAugment, ColorJitter, RandomErasing)
- Label smoothing (0.1) to prevent overconfidence
- Cosine annealing learning rate schedule
- AdamW optimizer with weight decay

---

## Architecture

### Visual Overview

```
Input Image (32×32×3)
         ↓
┌─────────────────────┐
│  Convolutional Stem │
│  ─────────────────  │
│  Conv 3×3, 64       │
│  BatchNorm + ReLU   │
│  Conv 3×3, 128, s=2 │
│  BatchNorm + ReLU   │
│  Conv 3×3, 384, s=2 │
│  BatchNorm + ReLU   │
└─────────────────────┘
         ↓
    Feature Map (8×8×384)
         ↓
┌─────────────────────┐
│  Reshape to Patches │
│  (64 patches, 384d) │
└─────────────────────┘
         ↓
┌─────────────────────┐
│  Prepend CLS Token  │
│  + Positional Embed │
└─────────────────────┘
         ↓
┌─────────────────────┐
│ Transformer Encoder │
│  ─────────────────  │
│  8 × Transformer    │
│  Blocks             │
│  - Multi-Head Attn  │
│  - LayerNorm        │
│  - MLP              │
│  - Stochastic Depth │
└─────────────────────┘
         ↓
┌─────────────────────┐
│  Classification     │
│  Head (CLS Token)   │
└─────────────────────┘
         ↓
    100 Classes
```

### Convolutional Stem Details

The convolutional stem progressively processes the input:

| Layer | Input Size | Kernel | Stride | Output Size | Features |
|-------|-----------|--------|--------|-------------|----------|
| Conv1 | 32×32×3   | 3×3    | 1      | 32×32×64    | 64       |
| Conv2 | 32×32×64  | 3×3    | 2      | 16×16×128   | 128      |
| Conv3 | 16×16×128 | 3×3    | 2      | 8×8×384     | 384      |

### Transformer Block Architecture

Each transformer block consists of:
- Multi-Head Self-Attention (6 heads)
- Layer Normalization
- Feed-Forward Network (MLP with 4× expansion)
- Residual connections
- Stochastic depth (linearly increasing rate)

---

## Implementation Details

### Frameworks Used

- **PyTorch**: Deep learning framework for model implementation
- **torchvision**: Dataset loading and augmentation
- **einops**: Tensor manipulation utilities
- **tqdm**: Progress bar for training monitoring

### Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Embedding Dimension | 384 | Balance between capacity and efficiency |
| Number of Heads | 6 | Divides embedding dimension evenly |
| Depth | 8 | Sufficient capacity for CIFAR-100 |
| MLP Ratio | 4.0 | Standard transformer configuration |
| Dropout | 0.1 | Prevent overfitting |
| Stochastic Depth | 0.1 | Regularization through random layer dropping |
| Batch Size | 128 | Maximum for available GPU memory |
| Learning Rate | 3e-4 | Standard for AdamW optimizer |
| Weight Decay | 0.05 | L2 regularization |
| Label Smoothing | 0.1 | Reduce overconfidence |
| Epochs | 200 | Allow full convergence |

### Data Augmentation

**Training Augmentations:**
- Random resized crop (scale 0.8-1.0)
- Random horizontal flip
- Color jitter (brightness, contrast, saturation, hue)
- AutoAugment with CIFAR-10 policy
- Random erasing (p=0.25)

**Testing:**
- Center crop only
- Normalize using dataset statistics

---

## Code Structure

```
Hybrid_CNN+ViT.ipynb
│
├── Installation & Imports
│   └── PyTorch, torchvision, einops, tqdm
│
├── Configuration
│   ├── Device setup
│   ├── Hyperparameters
│   └── Random seed
│
├── Data Loading
│   ├── CIFAR-100 dataset
│   ├── Training transforms
│   └── DataLoaders
│
├── Model Architecture
│   ├── ConvPatchEmbed
│   │   └── 3-layer convolutional stem
│   ├── Attention Module
│   │   └── Multi-head self-attention
│   ├── MLP Module
│   │   └── Feed-forward network
│   ├── Transformer Block
│   │   ├── LayerNorm
│   │   ├── Attention
│   │   ├── MLP
│   │   └── Stochastic Depth
│   └── ViT Model
│       ├── Patch embedding (Conv)
│       ├── Position embedding
│       ├── Transformer blocks
│       └── Classification head
│
├── Training Components
│   ├── Optimizer (AdamW)
│   ├── Scheduler (CosineAnnealing)
│   └── Loss (CrossEntropy + Label Smoothing)
│
├── Training Loop
│   ├── train_one_epoch()
│   ├── evaluate()
│   └── Main training loop
│
└── Model Checkpointing
    └── Save best model
```

---

## Results

### Training Progress

The model was trained for 103 epochs (interrupted) with the following observations:

**Early Training (Epochs 0-20)**
- Rapid initial learning: accuracy increased from 5.08% to 51.14%
- Loss decreased from 4.37 to 2.34
- Model quickly learned basic feature representations

**Mid Training (Epochs 20-50)**
- Steady improvement: accuracy reached 63.75%
- Loss continued decreasing to ~1.98
- Learning rate gradually decreased with cosine schedule

**Late Training (Epochs 50-103)**
- Convergence phase: accuracy stabilized around 66.67%
- Loss plateaued at ~1.53
- Model refined learned representations

### Final Performance

| Metric | Value |
|--------|-------|
| Best Validation Accuracy | **66.67%** |
| Final Training Accuracy | 78.40% |
| Final Validation Loss | 1.9296 |
| Best Epoch | 93 |

### Training Curve Characteristics

**Observations:**
- Training accuracy reached 78.40%, showing the model learned the training data well
- Validation accuracy of 66.67% indicates moderate overfitting (~12% gap)
- The gap between training and validation suggests room for improved regularization
- Steady validation improvement throughout training with minimal fluctuation

---

## Analysis

### Strengths of the Hybrid Approach

**1. Improved Inductive Bias**
- Convolutional stem provides translation equivariance
- Better suited for image data than pure patch embedding
- More parameter-efficient feature extraction

**2. Hierarchical Feature Learning**
- Progressive downsampling captures multi-scale features
- Low-level features (edges, textures) extracted by CNN
- High-level features (objects, context) captured by transformer

**3. Computational Efficiency**
- Reduced sequence length (64 vs 256 patches for standard ViT)
- Faster attention computation
- Lower memory requirements

### Performance Analysis

**Comparison to Baselines:**
- Standard ResNet-50 on CIFAR-100: ~75-78%
- Pure ViT on CIFAR-100 (without pre-training): ~50-55%
- This hybrid approach: 66.67%

The hybrid model performs between pure CNNs and pure transformers, suggesting that while the combination is beneficial, there's room for optimization.

### Learning Dynamics

**Training Behavior:**
- Smooth convergence with cosine annealing
- No signs of instability or catastrophic forgetting
- Label smoothing prevented overconfidence
- Stochastic depth improved generalization

**Regularization Effectiveness:**
- ~12% train-validation gap suggests adequate regularization
- Data augmentation helped prevent severe overfitting
- Could benefit from additional regularization techniques

---

## Limitations

### Current Constraints

**1. Training Duration**
- Training interrupted at epoch 103 (target: 200 epochs)
- May not have reached full convergence potential
- Additional epochs could improve validation accuracy

**2. Model Capacity**
- Relatively small embedding dimension (384)
- Could benefit from wider or deeper architecture
- Limited by computational resources

**3. Data Efficiency**
- Still requires substantial training data
- CIFAR-100's 500 images per class may be limiting
- Could benefit from transfer learning or pre-training

**4. Overfitting**
- 12% train-validation gap indicates overfitting
- Despite heavy augmentation and regularization
- Suggests need for stronger regularization strategies

### Architectural Limitations

**Patch Size:**
- Fixed 4×4 effective patch size may not be optimal
- Could experiment with different stem configurations

**Attention Mechanism:**
- Standard self-attention may be too flexible for small datasets
- Could benefit from structured or local attention patterns

---

## Future Work

### Short-term Improvements

**1. Complete Training**
- Run full 200 epochs to assess convergence
- Potentially extend to 300 epochs with lower learning rate

**2. Enhanced Regularization**
```python
# Potential additions:
- Mixup augmentation
- CutMix augmentation
- Increased dropout (0.2-0.3)
- Stronger weight decay
- Gradient clipping adjustments
```

**3. Architecture Tuning**
- Experiment with embedding dimensions: 512, 768
- Try different depths: 6, 10, 12 blocks
- Adjust MLP ratio: 3.0, 4.0, 6.0

### Medium-term Enhancements

**1. Advanced Training Techniques**
- Knowledge distillation from larger models
- Self-supervised pre-training on unlabeled data
- Progressive training strategies

**2. Architectural Variations**
```python
# Potential modifications:
- Multi-scale feature fusion
- Deformable attention mechanisms
- Conditional computation (MoE)
- Hierarchical transformers
```

**3. Optimization Strategies**
- AdamW with different betas
- Learning rate warmup refinement
- Cosine annealing with restarts

### Long-term Research Directions

**1. Transfer Learning**
- Pre-train on ImageNet-1K
- Fine-tune on CIFAR-100
- Investigate optimal transfer strategies

**2. Neural Architecture Search**
- Automated stem design
- Optimal depth and width
- Attention head configuration

**3. Efficiency Optimization**
- Model quantization
- Knowledge distillation to smaller models
- Pruning and sparsity techniques

**4. Scaling Studies**
- Performance vs. model size
- Data requirements analysis
- Computational efficiency benchmarks

---

## References

### Primary Reference

**An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale**
- Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, et al.
- International Conference on Learning Representations (ICLR), 2021
- [Paper Link](https://arxiv.org/abs/2010.11929)

**Key Contributions:**
- Introduced Vision Transformer (ViT) architecture
- Demonstrated transformers can match or exceed CNN performance
- Showed importance of pre-training at scale
- Established patch-based image processing paradigm

### Related Work

**Hybrid Architectures:**
- Early Convolutions Help Transformers See Better (Xiao et al., 2021)
- Tokens-to-Token ViT (Yuan et al., 2021)
- ConViT: Improving Vision Transformers with Soft Convolutional Inductive Biases (d'Ascoli et al., 2021)

**Vision Transformers:**
- DeiT: Data-efficient Image Transformers (Touvron et al., 2021)
- Swin Transformer (Liu et al., 2021)
- CaiT: Going Deeper with Image Transformers (Touvron et al., 2021)

### Implementation References

- PyTorch Documentation: https://pytorch.org/docs/
- torchvision Transforms: https://pytorch.org/vision/stable/transforms.html
- Timm Library: https://github.com/rwightman/pytorch-image-models

---

## Acknowledgments

This implementation draws inspiration from the original Vision Transformer paper and various hybrid architecture approaches in the literature. The convolutional stem design is influenced by research showing that early convolutions improve transformer performance on vision tasks.

---

## License

This project is available for educational and research purposes. Please cite the original Vision Transformer paper when using this code or building upon this work.
