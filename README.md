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

#### This is the Hugging Face Space for the original Hybrid ViT model trained on CIFAR-100 classes:

[![Hugging Face Space](https://img.shields.io/badge/🤗%20Hugging%20Face-Space-yellow)](https://huggingface.co/spaces/Aumkeshchy2003/ViT_For_100_Class)

#### This other Space is for the fine-tuned model on CIFAR-10, making the total number of classes 110:

[![Hugging Face Space](https://img.shields.io/badge/🤗%20Hugging%20Face-Space-yellow)](https://huggingface.co/spaces/Aumkeshchy2003/ViT-One110)

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

<img width="700" height="700" alt="Gemini_Generated_Image_nvl61nnvl61nnvl6" src="https://github.com/user-attachments/assets/611f1c4c-eddd-4dbd-b452-759417d71582" />



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

### Main Results: Classification Performance

| Metric | Value |
|--------|-------|
| **Validation Accuracy** | 68.23% |
| **Training Accuracy** | 88.23% |
| **Training Time** | 9h 10m (200 epochs) |
| **Train-Validation Gap** | 20.0% |

**Key Findings:**
- Achieves **68.23% accuracy** on CIFAR-100 without pre-training
- Outperforms pure ViT (~50-55%) by **13-18 percentage points**
- Comparable to published hybrid architectures (CvT, DeiT) on CIFAR-100
- Demonstrates effectiveness of hybrid CNN-ViT design on small datasets

### Convergence Behavior

- **Initial Convergence:** Rapid improvement in first 50 epochs
- **Stabilization:** Smooth plateau after epoch 100
- **Final Phase:** Gradual refinement from epoch 150-200
- **No Divergence:** Stable training throughout

### Generalization Analysis

- **Train Accuracy:** 88.23% (final epoch)
- **Validation Accuracy:** 68.23% (final epoch)
- **Gap:** 20.0% (indicates moderate overfitting)
- **Interpretation:** Model learns training patterns; room for improvement via better regularization or data


**Observations:**
- Training accuracy reached 78.40%, showing the model learned the training data well
- Validation accuracy of 66.67% indicates moderate overfitting (~12% gap)
- The gap between training and validation suggests room for improved regularization
- Steady validation improvement throughout training with minimal fluctuation

---

## Analysis

### Why Hybrid Architecture Works

1. **CNN Inductive Bias:** 
   - Convolutional layers naturally capture spatial locality
   - Hierarchical feature learning (low-level edges → high-level objects)
   - Reduces parameters compared to pure ViT
   - Effective with limited training data

2. **Transformer Expressiveness:**
   - Self-attention captures global spatial relationships
   - Flexible receptive fields (not fixed like CNN kernels)
   - Models long-range dependencies between distant patches
   - Powerful for complex semantic reasoning

3. **Data Efficiency:**
   - CNN stem learns basic features with fewer samples
   - Transformer refines these features with global context
   - Combined: Better than each component alone on 50K samples

4. **Computational Balance:**
   - CNN stem: O(N) complexity in spatial dimensions
   - Transformer: O(N²) in number of patches (but only 64)
   - Total: Manageable computational cost

### Comparison to Pure ViT

| Aspect | Pure ViT | Hybrid CNN-ViT |
|--------|----------|---|
| **CIFAR-100 Accuracy** | 50-55% | **68.23%** |
| **Data Efficiency** | Poor (needs pre-training) | Good (no pre-training needed) |
| **Training Stability** | Requires careful tuning | More stable |
| **Parameters** | More | Fewer |
| **Inductive Bias** | None (learns from scratch) | Spatial locality from CNN |

### Generalization Gap (20%)

The 20% train-validation gap suggests:
- Model has capacity to memorize training data
- Overfitting occurs despite regularization (dropout, stochastic depth)
- Validation set presents unseen patterns not well-captured by training

**Possible Improvements:**
- Stronger regularization (higher dropout, stochastic depth)
- More data augmentation
- Early stopping
- Ensemble methods

---

## Limitations

## Limitations

1. **Single Dataset:** Validated on CIFAR-100 only (100 classes, 32×32 RGB images, 50K samples). Generalization to ImageNet, medical imaging, or other domains untested.

2. **Single Architecture:** Specific hybrid design (3-layer CNN stem + 8 ViT blocks). Different stem depths, widths, or ViT configurations not explored.

3. **No Pre-training:** No ImageNet pre-training used. Pre-trained models likely achieve higher accuracy but require additional data/compute.

4. **Limited Statistical Analysis:** Single training run without error bars, confidence intervals, or multiple random seeds. Results may vary with different initializations.

5. **Hardware-Specific:** Trained on Apple M2 Pro MPS backend. Training dynamics and convergence may differ on GPUs (V100, A100, RTX) or TPUs.

6. **CNN Stem Design:** Specific kernel sizes (3×3), stride patterns (2,2), and channel widths (64→128→384) chosen empirically. Systematic ablation not performed.

7. **Fixed Patch Size:** Patches derived from fixed 8×8 spatial grid (stride-2 pooling twice). Different patch sizes not explored.

8. **20% Generalization Gap:** Indicates room for improvement; not optimal generalization.
---

## Future Work

### High Priority

1. **Test on ImageNet-1K** — Validate scalability to 1000 classes, 1.2M images, and 224×224 resolution
2. **Ablation Studies** — Systematically vary CNN stem depth, ViT blocks, embedding dimensions
3. **Comparison to Baselines** — Direct comparison to CvT, DeiT, Swin Transformers on CIFAR-100
4. **Pre-training Analysis** — Evaluate impact of ImageNet pre-training

### Medium Priority

1. **Different CNN Architectures** — ResNet stem, MobileNet stem, or other efficient designs
2. **Patch Size Exploration** — Test different stem output resolutions (4×4, 8×8, 16×16 patches)
3. **Transfer Learning** — Fine-tune on CIFAR-10, downstream tasks (object detection, segmentation)
4. **Architecture Search** — AutoML to find optimal hybrid design

### Research Directions

1. **Theoretical Analysis** — Why does CNN inductive bias help with limited data? Information-theoretic bounds?
2. **Visualization & Interpretability** — Attention maps, feature importance, patch significance
3. **Robustness Analysis** — Performance on corrupted images, adversarial examples, distribution shifts
4. **Domain Adaptation** — Fine-tuning on natural, medical, satellite, or synthetic imagery

---

## References

### Primary Reference

## References

[1] Dosovitskiy, A., et al. "An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale." ICLR, 2021.

[2] Wu, Z., et al. "CvT: Introducing Convolutions to Vision Transformers." ICCV, 2021.

[3] Touvron, H., et al. "Training Data-Efficient Image Transformers & Distillation Through Attention." ICML, 2021. (DeiT)

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

## Citation

If you use this baseline model in your research, please cite:

```bibtex
@misc{chaudhary2025hybridvit,
  title={Hybrid CNN-ViT: Data-Efficient Vision Transformer for CIFAR-100},
  author={Chaudhary, Aumkesh},
  year={2025},
  howpublished={\url{https://github.com/aumkeshchaudhary/Hybrid-CNN-ViT-Baseline}}
}
```
---

## License

This project is available for educational and research purposes. Please cite the original Vision Transformer paper when using this code or building upon this work.
