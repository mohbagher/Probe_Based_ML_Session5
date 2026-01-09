# 📚 PhD Research Dashboard - Complete User Guide

## Table of Contents
1. [System Parameters](#1-system-parameters)
2. [Model Architectures](#2-model-architectures)
3. [Training Configuration](#3-training-configuration)
4. [Evaluation Metrics](#4-evaluation-metrics)
5. [Visualization Options](#5-visualization-options)
6. [Advanced Features](#6-advanced-features)

---

## 1. System Parameters

### 1.1 N (Number of RIS Elements)

**What it is:** Physical antenna elements in the Reconfigurable Intelligent Surface.

**Technical Background:**
- RIS is an array of passive reflecting elements
- Each element can independently control phase shift
- More elements → higher beamforming gain but more complexity

**Mathematical Impact:**
- Channel dimension: h ∈ ℂ^N (BS-RIS), g ∈ ℂ^N (RIS-UE)
- Optimal beamforming gain: O(N²)
- Computational complexity: O(NK) for all probe evaluations

**When to use different values:**
- **N=8-16:** Quick prototyping, low-complexity scenarios
- **N=32:** Standard research baseline (good balance)
- **N=64-128:** Realistic deployments, high-performance systems
- **N=256+:** Advanced research, asymptotic analysis

**Trade-offs:**
- ✅ Larger N: Better performance, higher capacity
- ❌ Larger N: More hardware cost, higher training time, larger models

**Practical Recommendations:**
- Start with N=32 for initial experiments
- Use N=16 for quick debugging
- Use N=64+ for final paper results

---

### 1.2 K (Codebook Size)

**What it is:** Number of predefined phase configurations (probes) in your library.

**Technical Background:**
- Codebook = set of K possible RIS configurations
- Each probe k has N phase values: θₖ ∈ [0, 2π)^N
- Exhaustive search: measure all K probes → infeasible for large K
- Our approach: ML predicts best probe from limited measurements

**Mathematical Impact:**
- Search space: 2^(N·bits) possible configurations (discrete phase)
- Codebook reduces to K << 2^(N·bits)
- Classification problem: predict best among K options

**When to use different values:**
- **K=16-32:** Simple scenarios, proof-of-concept
- **K=64:** Standard baseline (used in most papers)
- **K=128-256:** Complex environments, higher capacity
- **K=512+:** Research on scalability limits

**Trade-offs:**
- ✅ Larger K: More optimization freedom, better optimal performance
- ❌ Larger K: Harder ML task, needs more training data, slower inference

**Design Considerations:**
- K should be large enough to contain near-optimal solutions
- K vs N ratio: typically K = 2N to 4N is good
- K must be >> M (sensing budget) for meaningful ML problem

---

### 1.3 M (Sensing Budget)

**What it is:** Number of probes you can actually measure (test) before making a decision.

**Technical Background:**
- Limited probing constraint: can only try M << K probes
- Motivation: Each measurement costs time, energy, signaling overhead
- Classic trade-off: exploration vs exploitation
- M/K ratio defines problem difficulty

**Mathematical Impact:**
- Input features: only M observed powers out of K total
- Sparse observation: 1-M/K fraction missing
- ML model must infer from partial information

**When to use different values:**
- **M=2-4:** Extremely limited (hard problem, practical for fast systems)
- **M=8:** Standard baseline (good for research)
- **M=16-32:** More information available (easier ML task)
- **M=K/2:** Moderate observation (for comparison with full knowledge)

**Trade-offs:**
- ✅ Larger M: Easier ML problem, better predictions, safer decisions
- ❌ Larger M: Higher overhead, slower adaptation, more energy

**Key Relationships:**
- M → 1: ML must extrapolate heavily (hard)
- M → K: Approaches full information (easy but defeats purpose)
- Typical sweet spot: M = K/8 to K/4

**Performance Metrics:**
- η (eta) = P_predicted / P_best: measures how close to optimal
- Best observed baseline: max power among M measured probes
- ML improvement: how much better than best observed

---

## 2. Model Architectures

### 2.1 MLP (Multi-Layer Perceptron)

**Architecture:** Fully connected neural network.

**How it works:**
```
Input (2K) → Dense(512) → ReLU → Dropout → 
           → Dense(256) → ReLU → Dropout →
           → Dense(128) → ReLU → Dropout →
           → Dense(K) → Softmax
```

**Technical Background:**
- Universal approximation theorem: can learn any continuous function
- Each layer transforms: h = ReLU(Wx + b)
- Learns hierarchical representations
- No spatial/temporal structure assumed

**Advantages:**
- ✅ Simple, well-understood
- ✅ Fast training and inference
- ✅ Works well for tabular/vector data
- ✅ Few hyperparameters to tune
- ✅ Good baseline for comparison

**Disadvantages:**
- ❌ No built-in structure awareness
- ❌ Treats all inputs independently
- ❌ Can overfit on small datasets

**When to use:**
- Default choice for vector inputs
- When data has no obvious structure
- When you need fast training
- For establishing baselines

**Hyperparameters:**
- **Hidden sizes:** [512, 256, 128] is good default
  - Deeper = more capacity but harder to train
  - Wider = more parameters but parallelizable
- **Dropout:** 0.1-0.3 typical range
  - Higher if overfitting
  - Lower if underfitting
- **Batch norm:** almost always beneficial

**Expected Performance:**
- Should achieve η ≈ 0.85-0.92 (M=8, K=64, N=32)
- Training time: ~2-5 min (50 epochs, CPU)
- Parameters: ~500K for [512,256,128]

---

### 2.2 CNN (Convolutional Neural Network)

**Architecture:** 1D convolutions over probe features.

**How it works:**
```
Input (2 channels × K length) → 
  Conv1D(32, kernel=5) → ReLU → Pool →
  Conv1D(64, kernel=5) → ReLU → Pool →
  Conv1D(128, kernel=3) → ReLU → Pool →
  Flatten → Dense(256) → Dense(K)
```

**Technical Background:**
- Exploits local correlations in probe space
- Kernel slides across probe indices
- Parameter sharing reduces model size
- Hierarchical feature extraction

**Key Concept - Why CNN for Probes:**
- Probes with similar indices may have similar powers
- Spatial structure in phase configurations
- Local patterns more important than global
- Weight sharing = inductive bias

**Advantages:**
- ✅ Captures local structure
- ✅ Fewer parameters than MLP (weight sharing)
- ✅ Translation invariant
- ✅ Good for structured probe banks (e.g., Hadamard)

**When to use:**
- Structured codebooks (Hadamard, Sobol)
- When probe order matters
- Larger K (spatial structure more apparent)

**Hyperparameters:**
- **Num conv layers:** 3-5 typical
- **Channels:** [32, 64, 128] progressive increase
- **Kernel size:** 3-7 (odd numbers)
  - Larger kernel = more context
  - Smaller kernel = more local

**Expected Performance:**
- η ≈ 0.87-0.94 for structured probes
- Best for: Hadamard, Sobol probe methods

---

### 2.3 LSTM (Long Short-Term Memory)

**Architecture:** Recurrent network treating probes as sequence.

**How it works:**
```
Input sequence (K timesteps, 2 features) →
  BiLSTM(128, 2 layers) →
  Flatten → Dense(256) → Dense(K)
```

**Technical Background:**
- Treats K probes as temporal sequence
- Each probe = 1 timestep with [power, mask] features
- LSTM cell: can remember long-term dependencies
- Bidirectional: processes sequence both directions

**Advantages:**
- ✅ Models sequential dependencies
- ✅ Handles variable-length sequences
- ✅ Captures long-range correlations
- ✅ Bidirectional sees full context

**Disadvantages:**
- ❌ Sequential computation (slow)
- ❌ Can't parallelize across time
- ❌ Needs more data than MLP

**When to use:**
- Ordered probe sequences
- Time-varying channels
- When probe history matters

**Expected Performance:**
- η ≈ 0.84-0.91
- Slower training (5-10× vs MLP)

---

### 2.4 GRU (Gated Recurrent Unit)

**Architecture:** Simplified LSTM with fewer gates.

**Advantages over LSTM:**
- ✅ Faster training (fewer parameters)
- ✅ Similar performance in many cases
- ✅ Less prone to overfitting

**When to use:**
- Similar to LSTM but want faster training
- When LSTM shows overfitting

---

### 2.5 Attention MLP

**Architecture:** MLP with multi-head attention mechanism.

**How it works:**
```
Input → Dense(512) →
  MultiHeadAttention(4 heads) →
  Dense(256) → Dense(128) → Dense(K)
```

**Why combine MLP + Attention:**
- Attention finds important features
- MLP processes attended features
- Best of both worlds

**Expected Performance:**
- η ≈ 0.88-0.94
- Good balance: simpler than Transformer, better than MLP

---

### 2.6 Transformer

**Architecture:** Self-attention mechanism.

**How it works:**
```
Input (K tokens, 2 features) →
  Embedding → Positional Encoding →
  Transformer Encoder (3 layers):
    MultiHeadAttention(8 heads) →
    FeedForward(512) →
  →  Global Pool → Dense(K)
```

**Technical Background:**
- Attention: learns which probes are most relevant
- Self-attention: each probe attends to all others
- Position-agnostic: doesn't assume order
- Parallel computation: very fast on GPU

**Advantages:**
- ✅ Most powerful architecture (SOTA)
- ✅ Learns complex relationships
- ✅ Parallel computation (fast on GPU)
- ✅ Interpretable attention weights

**Disadvantages:**
- ❌ Most parameters (largest model)
- ❌ Needs more data
- ❌ Can overfit easily

**When to use:**
- Large datasets (50K+ samples)
- Large K (128+)
- GPU available
- Pushing state-of-the-art

**Expected Performance:**
- η ≈ 0.90-0.96 (best overall with enough data)
- Needs: 50K+ training samples

---

### 2.7 ResNet-Style MLP

**Architecture:** MLP with skip connections.

**How it works:**
```
Input → Dense(512) → [ResBlock → ResBlock → ...] → Dense(K)

ResBlock:
  x → Dense → ReLU → Dense → (+) → ReLU
  └─────────────────────────┘ (skip)
```

**Advantages:**
- ✅ Can go much deeper (10+ layers)
- ✅ Faster convergence
- ✅ Less prone to vanishing gradients

**When to use:**
- Want deeper models
- Plain MLP plateaus

---

### 2.8 Hybrid CNN-LSTM

**Architecture:** CNN for feature extraction, LSTM for sequence modeling.

**Use Case:**
- Time-varying channels
- Structured + sequential patterns

**Expected Performance:**
- η ≈ 0.86-0.92 for time-varying scenarios

---

## 3. Training Configuration

### 3.1 Optimizers

#### Adam (Adaptive Moment Estimation)

**Default choice** - Works well out-of-box.

**Hyperparameters:**
- **Learning rate:** 1e-4 to 1e-2 (start with 1e-3)
- **β₁:** 0.9 (typical, rarely changed)
- **β₂:** 0.999 (typical, rarely changed)

**When to use:**
- Default choice (90% of cases)
- Quick prototyping

#### AdamW

**Better regularization** - Decoupled weight decay.

**When to use:**
- Large models (Transformer)
- Need better generalization

#### SGD

**Classic optimizer** - Better generalization but needs tuning.

**Hyperparameters:**
- **Learning rate:** 0.01-0.1 (much higher than Adam!)
- **Momentum:** 0.9-0.99

**When to use:**
- Final paper results
- Care about generalization
- Use with good scheduler

---

### 3.2 Learning Rate Schedulers

#### ReduceLROnPlateau

**Default choice** - Automatic adaptation.

**Parameters:**
- **Patience:** how many epochs to wait
- **Factor:** multiply LR by this (e.g., 0.5)

#### CosineAnnealing

**Smooth decay** - LR follows cosine curve.

**When to use:**
- Know total epochs
- Want smooth schedule

#### OneCycleLR

**SOTA schedule** - Combines warmup + annealing.

**When to use:**
- Want fastest training
- SGD optimizer

---

### 3.3 Loss Functions

#### CrossEntropy

**Standard choice** for classification.

#### FocalLoss

**For imbalanced classes** - Focuses on hard examples.

#### LabelSmoothing

**Regularization technique** - Prevents overconfidence.

**When to use:**
- Model overfitting
- Want better calibration

---

## 4. Evaluation Metrics

### Top-K Accuracy

Fraction of samples where oracle best probe is in top-K predictions.

- **Top-1:** Most important (exact match)
- **Top-2, Top-4, Top-8:** Show prediction quality

### Eta (η) - Power Ratio

η = P_predicted / P_optimal

- **η = 1.0:** Perfect prediction
- **η ≈ 0.9:** Excellent (90% of optimal power)
- **η ≈ 0.8:** Good (80% of optimal power)
- **η < 0.7:** Poor performance

### Baseline Comparisons

- **Random-1:** Pick 1 probe randomly from K
- **Random-M:** Best of M random probes
- **Best-Observed:** Best among actually measured M probes
- **Oracle:** True best probe (upper bound)

---

## 5. Visualization Options

### Training Plots

- **Training Curves:** Loss and accuracy over epochs
- **Learning Rate Schedule:** LR evolution
- **Gradient Flow:** Gradient magnitudes per layer

### Performance Plots

- **Eta Distribution:** Histogram/PDF of η values
- **CDF:** Cumulative distribution function
- **Box/Violin Plot:** Statistical distribution comparison
- **Bar/Scatter:** Model comparison

### Probe Analysis

- **Probe Heatmap:** Phase configurations visualization
- **Correlation Matrix:** Probe similarity
- **Diversity Analysis:** Probe bank quality metrics

### Advanced Plots

- **3D Surface:** Performance vs two parameters
- **ROC/Precision-Recall:** Classification metrics
- **Confusion Matrix:** Prediction patterns
- **Convergence Analysis:** Multi-model comparison

---

## 6. Advanced Features

### Multi-Model Comparison

Compare multiple architectures in one run:
- Enable "Multi-Model Comparison Mode"
- Select models to compare
- Get side-by-side performance metrics

### Multi-Seed Runs

Statistical analysis with confidence intervals:
- Enable "Multi-Seed Runs"
- Set number of seeds
- Get mean ± std results

### Preset Configurations

Quick start with optimized presets:
- **Quick Test:** Fast configuration for testing
- **Standard Research:** Default research setup
- **High Capacity:** Large-scale deployment
- **Limited Budget:** Extreme sensing constraint

### Configuration Management

- **Save Config:** Export current settings to JSON/YAML
- **Load Config:** Import previously saved settings
- **Validation:** Auto-check for common issues

---

## Quick Start Examples

### Example 1: Basic Experiment
```python
1. Set N=32, K=64, M=8
2. Choose MLP model
3. Default training settings
4. Click "RUN EXPERIMENT"
```

### Example 2: Model Comparison
```python
1. Enable "Multi-Model Comparison Mode"
2. Select: MLP, CNN, LSTM, Transformer
3. Set epochs=50
4. Click "RUN EXPERIMENT"
5. View comparison plots
```

### Example 3: Statistical Analysis
```python
1. Enable "Multi-Seed Runs"
2. Set num_seeds=5
3. Configure your experiment
4. Click "RUN EXPERIMENT"
5. Get mean ± std results
```

---

## Tips & Best Practices

### For Quick Testing
- Use N=16, K=32, M=4
- 10K training samples
- 20 epochs
- MLP model

### For Research Papers
- Use N=32-64, K=64-128
- 50K+ training samples
- 50-100 epochs
- Compare multiple models
- Multi-seed runs for error bars

### For Large-Scale
- Use N=64-128, K=128-256
- 100K+ training samples
- Transformer model
- GPU recommended

---

## Troubleshooting

### Issue: Model not converging
- **Solution:** Increase learning rate, try different optimizer

### Issue: Overfitting
- **Solution:** Increase dropout, add weight decay, more training data

### Issue: Poor performance
- **Solution:** Check M vs K ratio, try different model, more epochs

### Issue: Slow training
- **Solution:** Reduce batch size, use simpler model, check device (GPU/CPU)

---

For more details, see:
- [Developer Guide](DEVELOPER_GUIDE.md) - How to extend the system
- [Tutorial](TUTORIAL.md) - Step-by-step examples
- [API Reference](API_REFERENCE.md) - Function documentation
