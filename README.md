# RIS Probe-Based Control with Limited Probing

A machine learning framework for optimizing Reconfigurable Intelligent Surface (RIS) phase configurations using limited probe measurements.

## 🎯 Problem Overview

In RIS-assisted wireless communications, we have:
- **N** RIS elements (reflective surface) - default: 32
- **K** pre-defined probe configurations (phase patterns) - default: 64
- **M** sensing budget (probes we can actually measure) - default: 8

**Challenge**: Predict the best probe among all K options while only observing M << K measurements.

## 🏗️ Architecture

```
Input: [masked_powers, binary_mask] ∈ ℝ^{2K}
      ↓
   MLP (512 → 256 → 128)
      ↓
Output: logits ∈ ℝ^K (probability over all probes)
```

The model uses a **Masked Vector Approach**:
- `masked_powers`: K-dimensional vector with observed powers at their indices, zeros elsewhere
- `binary_mask`: K-dimensional vector with 1s at observed indices, 0s elsewhere

## 📦 Installation

```bash
pip install -r requirements.txt
```

## 🚀 Quick Start

### Python API

```python
from run import run_default

# Run with default parameters
results = run_default()

# Custom parameters
from run import run_custom
results = run_custom(N=64, K=128, M=16, n_train=100000)
```

### Command Line

```bash
python main.py --N 32 --K 64 --M 8 --n_train 50000 --epochs 100
```

## 💻 Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--N` | 32 | Number of RIS elements |
| `--K` | 64 | Total probes in bank |
| `--M` | 8 | Sensing budget (probes measured per sample) |
| `--n_train` | 50000 | Number of training samples |
| `--n_val` | 5000 | Number of validation samples |
| `--n_test` | 5000 | Number of test samples |
| `--epochs` | 100 | Maximum number of epochs |
| `--batch_size` | 128 | Batch size |
| `--lr` | 1e-3 | Learning rate |
| `--hidden` | [512, 256, 128] | Hidden layer sizes |
| `--dropout` | 0.1 | Dropout probability |
| `--save_dir` | results | Directory to save results |

## 📊 Key Metrics

- **η (eta)**: Power ratio = P_selected / P_best_probe (higher is better, max 1.0)
- **Top-m Accuracy**: Fraction where oracle best probe is in top-m predictions
- **Baselines**:
  - Random 1/K: Pick 1 probe randomly
  - Random M/K: Best of random M probes
  - Best Observed: Best among actually measured M probes
  - Oracle: Theoretical best (η = 1.0)

## 📁 Project Structure

```
├── config.py           # Configuration dataclasses
├── data_generation.py  # Channel simulation &amp; dataset creation
├── model.py            # MLP neural network architecture
├── training.py         # Training loop with early stopping
├── evaluation.py       # Metrics computation &amp; baselines
├── utils.py            # Visualization &amp; result saving
├── main.py             # CLI entry point
├── run.py              # Quick experiment scripts
└── requirements.txt    # Dependencies
```

## 📚 Dependencies

- PyTorch ≥ 1.12.0
- NumPy ≥ 1.21.0
- Matplotlib ≥ 3.5.0
- Seaborn ≥ 0.11.0
- tqdm ≥ 4.62.0
- pandas ≥ 1.3.0

## 📈 Example Results

After training, the model outputs:
- Training history plots (loss, accuracy, η over epochs)
- η distribution comparison (ML model vs baselines)
- Top-m accuracy and power ratio bar charts
- Saved model checkpoint and metrics

## 📄 License

MIT License
