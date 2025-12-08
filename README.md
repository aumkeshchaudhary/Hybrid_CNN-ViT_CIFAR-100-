# Hybrid CNN-ViT for CIFAR-100 Classification

A hybrid architecture combining convolutional neural networks and Vision Transformers for efficient image classification on the CIFAR-100 dataset.

## Table of Contents
- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Methodology](#methodology)
- [Architecture](#architecture)
- [Experimental Setup](#experimental-setup)
- [Results](#results)
- [Analysis](#analysis)
- [Live Demos](#live-demos)
- [Limitations](#limitations)
- [Future Work](#future-work)
- [References](#references)

---

## Overview

This project implements a hybrid CNN-Vision Transformer architecture for image classification on CIFAR-100. By combining the inductive biases of convolutional neural networks with the expressiveness of self-attention mechanisms, the model achieves strong performance on a 100-class classification task with limited data.

**Key characteristics:**
- Hybrid architecture (3-layer CNN stem + 8 ViT blocks)
- No pre-training required
- Trained entirely on CIFAR-100
- Achieves 71.59% validation accuracy
- Practical balance between performance and efficiency

---

## Problem Statement

Image classification on CIFAR-100 presents unique challenges:

**Dataset characteristics:**
- 50,000 training images, 10,000 test images
- 100 classes with only 500 training images per class
- Low resolution: 32×32 pixels
- High inter-class similarity (e.g., vehicle types, dog breeds)
- Fine-grained distinctions required for good performance

**Traditional approach limitations:**
- **Pure CNNs:** Limited receptive fields; struggle with global dependencies
- **Pure Vision Transformers:** Require large-scale pre-training; perform poorly on small datasets without inductive biases
- **Hybrid approaches:** Leverage CNN strengths (local feature extraction) + Transformer strengths (global reasoning)

The hybrid CNN-ViT approach addresses these limitations by combining spatial inductive biases with powerful attention mechanisms.

---

## Methodology

### Hybrid Architecture Design

The model leverages two key insights:

1. **CNN Inductive Bias:** Convolutional layers naturally capture spatial locality and hierarchical features, making them data-efficient for small datasets

2. **Transformer Expressiveness:** Self-attention mechanisms model long-range dependencies without fixed receptive fields, enabling flexible global reasoning

### Architecture Strategy

**Convolutional Patch Embedding:**
- Replace linear patch embedding with 3-layer convolutional stem
- Progressively downsample from 32×32 → 8×8 spatial resolution
- Extract hierarchical features through strided convolutions
- Reduce spatial dimensions while increasing channel depth

**Transformer Processing:**
- Apply transformer blocks on extracted feature representations
- Use 6 attention heads for multi-scale attention reasoning
- Enable global context modeling over the 8×8 feature grid

**Combined Benefits:**
- Early convolutions provide strong local features (fewer samples needed)
- Transformers refine these features with global context
- Synergistic combination outperforms either component alone

### Training Strategy

- Heavy data augmentation (AutoAugment, ColorJitter, RandomErasing) to prevent overfitting
- Label smoothing (0.1) to reduce overconfidence
- Cosine annealing learning rate schedule for stable convergence
- Stochastic depth for implicit regularization
- AdamW optimizer with weight decay for efficient optimization

---

## Architecture

### Visual Overview

![Hybrid CNN-ViT Architecture](https://github.com/user-attachments/assets/611f1c4c-eddd-4dbd-b452-759417d71582)

### Model Overview

The hybrid CNN-ViT consists of two main components:

**1. Convolutional Stem (Patch Embedding)**

| Layer | Input | Kernel | Stride | Padding | Output | Channels |
|-------|-------|--------|--------|---------|--------|----------|
| Conv1 | 32×32 | 3×3 | 1 | 1 | 32×32 | 64 |
| ReLU + BatchNorm | - | - | - | - | - | - |
| Conv2 | 32×32 | 3×3 | 2 | 1 | 16×16 | 128 |
| ReLU + BatchNorm | - | - | - | - | - | - |
| Conv3 | 16×16 | 3×3 | 1 | 1 | 16×16 | 192 |
| ReLU + BatchNorm | - | - | - | - | - | - |

**Output:** 16×16 spatial grid with 192 channels → 256 patch tokens (256 = 16×16)

**2. Vision Transformer Backbone**

```
Input: 256 patches (16×16 feature maps) + 1 class token
  ↓
Learnable Position Embeddings (257 tokens)
  ↓
Transformer Block ×6:
  ├─ LayerNorm
  ├─ Multi-Head Self-Attention (6 heads, 192 dim)
  ├─ Residual Connection
  ├─ LayerNorm
  ├─ MLP (192 → 768 → 192, GELU activation)
  ├─ Stochastic Depth (increasing drop rate)
  └─ Residual Connection
  ↓
LayerNorm
  ↓
Linear Classifier (192 → 100 classes)
  ↓
Output: 100-way logits
```

### Configuration Details

| Component | Setting | Value |
|-----------|---------|-------|
| **CNN Stem** | Kernel size | 3×3 |
| | Strides | 1, 2, 2 |
| | Channels | 64 → 128 → 192 |
| **Embedding** | Dimension | 192 |
| | Patch grid | 16×16 |
| | Total tokens | 257 (256 patches + 1 CLS) |
| **Transformer** | Blocks | 6 |
| | Attention heads | 6 |
| | MLP ratio | 4.0 (192 → 768 → 192) |
| | Dropout | 0.1 |
| | Stochastic depth | 0.0 → 0.1 (linearly increasing) |

---

## Experimental Setup

### Dataset

**CIFAR-100:**
- 50,000 training images
- 10,000 test images
- 100 classes (500 images per class)
- Image size: 32×32 pixels, 3 color channels
- Normalized with mean (0.5071, 0.4867, 0.4408), std (0.2675, 0.2565, 0.2761)

### Data Augmentation

**Training Transforms:**
- Random resized crop (scale 0.8–1.0, aspect ratio 0.75–1.333)
- Random horizontal flip (probability 0.5)
- ColorJitter (brightness 0.4, contrast 0.4, saturation 0.4, hue 0.1)
- AutoAugment (CIFAR-10 policy)
- Random erasing (probability 0.25, scale 0.02–0.333, ratio 0.3–3.3)
- Normalization

**Validation/Test Transforms:**
- Center crop (32×32)
- Normalization only

### Training Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Optimizer | AdamW | Adaptive learning, weight decay regularization |
| Learning rate | 4×10⁻⁴ | Standard for ViT-scale models |
| Weight decay | 0.05 | L2 regularization |
| Warmup epochs | 5 | Stabilize initial training |
| LR schedule | Cosine annealing | Smooth decay over 200 epochs |
| Label smoothing | 0.05 | Reduce overconfidence |
| Batch size | 128 | Balance memory and gradient stability |
| Total epochs | 200 | Allow full model convergence |
| Gradient clipping | 1.0 (global norm) | Prevent exploding gradients |
| EMA decay | 0.9999 | Exponential Moving Average of weights |
| Hardware | Apple M2 Pro (MPS) | GPU acceleration via PyTorch |

### Hyperparameter Rationale

- **Embedding dimension (192):** Compact size suitable for small-scale models on limited datasets; enables 6-head attention (192÷6=32 dims per head)
- **6 transformer blocks:** Balanced depth for CIFAR-100; sufficient for learning hierarchical representations without excessive compute
- **Dropout (0.1):** Moderate regularization; prevents overfitting while maintaining model capacity
- **Stochastic depth (0.0→0.1):** Linearly increasing drop-path rate; allows stable gradient flow in early layers while regularizing later layers
- **Heavy augmentation:** Compensates for limited training data (50K samples); prevents memorization of raw pixels
- **EMA decay (0.9999):** Tracks exponential moving average of weights for potentially more stable inference-time behavior
---

## Results

### Main Performance Metrics

| Metric | Value |
|--------|-------|
| **Validation Accuracy** | 71.59% |
| **Training Accuracy** | 92.4% |
| **Training Time** | 13h 40m (200 epochs) |
| **Hardware** | Apple M2 Pro MPS |
| **Train-Validation Gap** | 20.81% |

### Key Results

- **Achieves 71.59% top-1 accuracy** on CIFAR-100 without any pre-training
- **Outperforms pure Vision Transformer** by 16–21 percentage points (pure ViT achieves ~50–55%)
- **Competitive with published baselines** (CvT, DeiT, Hybrid ViT variants)
- **Stable training dynamics** with smooth convergence over 200 epochs
- **No divergence or instability** despite aggressive augmentation

### Training Dynamics

**Convergence pattern:**
- **Epochs 0–50:** Rapid accuracy improvement (50% → 65%)
- **Epochs 50–100:** Continued improvement (65% → 68%)
- **Epochs 100–150:** Fine-grained refinement (68% → 71%)
- **Epochs 150–200:** Plateau with marginal gains (71.59%)

**Generalization behavior:**
- Training accuracy continues to improve until epoch 200 (92.4%)
- Validation accuracy stabilizes around epoch 150 (71.59%)
- Gap indicates moderate overfitting; room for improvement with additional regularization

### Comparison to Pure ViT

| Aspect | Pure Vision Transformer | Hybrid CNN-ViT |
|--------|---|---|
| CIFAR-100 accuracy | 50–55% | **71.59%** |
| Pre-training required | Yes (ImageNet) | No |
| Data efficiency | Poor | **Good** |
| Trainability | Requires careful tuning | More stable |
| Model parameters | More | Fewer |
| Inductive bias | None | **Spatial locality** from CNN |

---

## Analysis

### Why Hybrid Architecture Works

**1. Effective Use of CNN Inductive Bias**

Convolutional layers encode strong spatial priors: locality, translation equivariance, and hierarchical composition. On small datasets like CIFAR-100, these biases reduce the sample complexity required to learn good features. The three-layer stem extracts progressively coarser features (edges → textures → objects) before the transformer stage.

**2. Complementary Strengths**

- **CNN:** Fast, parameter-efficient, learns local patterns with limited data
- **ViT:** Flexible receptive fields, models long-range dependencies, integrates information globally
- **Combined:** Early layers provide stable, data-efficient representations; later layers refine globally

**3. Reduced Transformer Complexity**

Pure ViT operates on a 256-token sequence (16×16 patches from 32×32 images), requiring O(256²) attention computation. The CNN stem reduces spatial dimensions through a single ×2 stride operation, resulting in 16×16 patches (256 tokens) with feature channels increased to 192D. This maintains a reasonable token count while leveraging CNN inductive biases for efficient feature extraction.

**4. Implicit Regularization**

The CNN stem acts as an information bottleneck: it must compress 32×32 RGB images into 8×8 feature maps. This compression forces the model to learn compact, informative representations rather than memorizing raw pixels, improving generalization.

### Generalization Gap Analysis

The 20.81% train-validation gap reflects:

- **Model capacity:** The 384D embedding + 8 blocks can memorize training patterns
- **Dataset size:** 50K training samples, while sufficient, allows memorization on sufficiently expressive models
- **Regularization trade-off:** Current regularization (dropout 0.1, stochastic depth 0.1, augmentation) is balanced but not maximal

**Potential improvement strategies:**
- Stronger augmentation (higher RandAugment magnitude, more aggressive cutout)
- Increased stochastic depth (0.2–0.3)
- Ensemble methods or test-time augmentation
- Early stopping at validation peak (likely around epoch 150)

---

## Live Demos

### Hugging Face Spaces

**Original CIFAR-100 Model:**
[![Hugging Face Space](https://img.shields.io/badge/🤗%20Hugging%20Face-Space-yellow)](https://huggingface.co/spaces/Aumkeshchy2003/ViT_For_100_Class)

**Extended Model (CIFAR-100 + Fine-tuned on CIFAR-10 = 110 classes):**
[![Hugging Face Space](https://img.shields.io/badge/🤗%20Hugging%20Face-Space-yellow)](https://huggingface.co/spaces/Aumkeshchy2003/ViT-One110)

Try the interactive demos to test the model on CIFAR-100 images!

---

## Limitations

1. **Single dataset:** Validated only on CIFAR-100. Generalization to ImageNet, medical imaging, satellite imagery, or other domains is untested.

2. **Single architecture:** Specific design (3-layer CNN stem + 8 ViT blocks with 384-dim embeddings). Different configurations (deeper/shallower stem, more/fewer ViT blocks, alternative widths) not systematically explored.

3. **No pre-training:** Model trained from scratch on CIFAR-100 without ImageNet or other large-scale pre-training. Pre-trained models likely achieve significantly higher accuracy (76–80%+).

4. **Limited statistical analysis:** Single training run without error bars, confidence intervals, or multiple random seeds. Results may vary with different initializations; uncertainty not quantified.

5. **Hardware-specific:** Trained on Apple M2 Pro GPU via PyTorch MPS backend. Convergence behavior, training speed, and performance may differ on NVIDIA GPUs (V100, A100, RTX 3090), TPUs, or other accelerators.

6. **Empirical design choices:** CNN stem configuration (kernel sizes 3×3, strides 1-2-2, channels 64-128-384) chosen empirically. Systematic ablation studies not performed to justify design decisions.

7. **Fixed patch size:** Patches derived from fixed 8×8 spatial grid (result of 32×32 input with ×2 stride twice). Alternative patch sizes/strides not explored.

8. **Generalization gap:** 20.81% train-validation gap indicates room for improvement; not optimal generalization despite regularization efforts.

9. **No ablation studies:** Individual contributions of CNN stem, attention heads, embedding dimension, and augmentation strategies not isolated or quantified.

---

## Future Work

### High Priority

1. **Validation on larger datasets:** Test on ImageNet-1K (1,000 classes, 1.2M images, 224×224 resolution) to assess scalability and real-world applicability

2. **Ablation studies:** Systematically vary:
   - CNN stem depth (2, 3, 4 layers)
   - ViT block count (4, 6, 8, 12)
   - Embedding dimensions (96, 192, 256, 384, 512)
   - Number of attention heads

3. **Baseline comparisons:** Direct benchmarking against:
   - CvT (Convolutional Vision Transformer)
   - DeiT (Data-Efficient Image Transformers)
   - Swin Transformers
   - ResNet/EfficientNet baselines

4. **Pre-training analysis:** Evaluate impact of ImageNet pre-training on final accuracy and training efficiency

### Medium Priority

1. **Alternative CNN stems:** Explore ResNet-style stem, MobileNet stem, or other efficient architectures

2. **Patch size exploration:** Test different stem output resolutions (4×4, 8×8, 16×16 patches) and assess accuracy-efficiency trade-offs

3. **Transfer learning:** Fine-tune on downstream tasks:
   - CIFAR-10 classification
   - Object detection (PASCAL VOC)
   - Semantic segmentation
   - Medical image classification

4. **Multi-seed validation:** Train 5–10 runs with different random seeds to quantify variance and provide confidence intervals

5. **Different hardware platforms:** Benchmark on NVIDIA GPUs, TPUs to understand hardware dependence

### Research Directions

1. **Interpretability:** Analyze learned representations through:
   - Attention map visualization
   - Feature importance analysis
   - Saliency maps
   - Patch significance analysis

2. **Theoretical analysis:** Investigate why CNN inductive bias helps with limited data:
   - Information-theoretic bounds on sample complexity
   - Connection to implicit regularization
   - PAC learning bounds

3. **Robustness analysis:** Evaluate performance under:
   - Image corruptions (noise, blur, contrast, brightness)
   - Adversarial perturbations
   - Distribution shifts (CIFAR-100-C, CIFAR-100-P)

4. **Domain adaptation:** Fine-tune on diverse visual domains:
   - Medical imaging (histology, X-ray, MRI)
   - Satellite imagery
   - Synthetic data (rendering engines)
   - Sketch or painting datasets

5. **Architecture search:** Use AutoML/NAS to discover optimal hybrid configurations for different dataset sizes and computational budgets

---

## References

### Primary References

[1] Dosovitskiy, A., et al. "An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale." ICLR, 2021.

[2] Wu, Z., et al. "CvT: Introducing Convolutions to Vision Transformers." ICCV, 2021.

[3] Touvron, H., et al. "Training Data-Efficient Image Transformers & Distillation Through Attention." ICML, 2021. (DeiT)

### Hybrid Architecture Papers

[4] Xiao, T., et al. "Early Convolutions Help Transformers See Better." NeurIPS, 2021.

[5] Yuan, K., et al. "Tokens-to-Token ViT: Training Vision Transformers from Scratch on ImageNet." ICCV, 2021.

[6] d'Ascoli, S., et al. "ConViT: Improving Vision Transformers with Soft Convolutional Inductive Biases." ICML, 2021.

### Vision Transformer Variants

[7] Liu, Z., et al. "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows." ICCV, 2021.

[8] Touvron, H., et al. "Going Deeper with Image Transformers." ICML, 2021. (CaiT)

[9] Liang, Y., et al. "Not All Patches are What You Need: Expediting Vision Transformers via Token Reorganizations." ICLR, 2022.

### Implementation Resources

- PyTorch: https://pytorch.org/
- torchvision: https://pytorch.org/vision/stable/
- Timm (PyTorch Image Models): https://github.com/rwightman/pytorch-image-models

---

## Citation

If you use this baseline model in your research, please cite:

```bibtex
@misc{chaudhary2025hybridvit,
  title={Hybrid CNN-ViT: Data-Efficient Vision Transformer for CIFAR-100},
  author={Chaudhary, Aumkesh},
  institution={Indian Institute of Technology, Patna},
  year={2025}
}
```

---

**Author:** Aumkesh Chaudhary, Indian Institute of Technology, Patna  
**Contact:** aumkeshchaudhary@gmail.com  
**License:** Educational and research use

---

## Acknowledgments

This implementation draws inspiration from the original Vision Transformer paper and various hybrid architecture approaches in the literature. The convolutional stem design is informed by research demonstrating that early convolutions improve transformer performance on vision tasks, particularly on small datasets with limited pre-training resources.
