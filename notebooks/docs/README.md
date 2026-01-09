# 📚 PhD Research Dashboard - Documentation

## Overview

This directory contains comprehensive documentation for the PhD Research Dashboard, a complete interactive system for RIS (Reconfigurable Intelligent Surface) probe-based machine learning research.

## Documentation Files

### [USER_GUIDE.md](USER_GUIDE.md) (16KB, 623 lines)
**Complete technical user manual**

Covers:
- System Parameters (N, K, M) - detailed technical background
- Model Architectures (8 models with mathematical foundations)
- Training Configuration (optimizers, schedulers, loss functions)
- Evaluation Metrics (accuracy, eta, baselines)
- Visualization Options (25+ plot types)
- Advanced Features (multi-model comparison, multi-seed runs)
- Quick start examples
- Tips and best practices
- Troubleshooting guide

**Target Audience:** Researchers, PhD students, ML practitioners

---

### [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md) (8KB, 368 lines)
**Extension and customization guide**

Covers:
- Adding new widgets
- Adding new ML models
- Adding new probe methods
- Adding new plot types
- Adding new optimizers and schedulers
- Adding new loss functions
- Adding new channel models
- Code style guidelines
- Testing procedures
- Common issues and solutions

**Target Audience:** Developers extending the system

---

### [TUTORIAL.md](TUTORIAL.md) (8KB, 398 lines)
**Step-by-step practical examples**

Contains 8 complete tutorials:
1. Your First Experiment
2. Comparing Multiple Models
3. Statistical Analysis with Multiple Seeds
4. Exploring Different Probe Methods
5. Hyperparameter Tuning
6. Scaling to Larger Systems
7. Debugging Poor Performance
8. Production Deployment Workflow

Plus:
- Common workflows (quick test, standard, production)
- Tips and tricks
- Common mistakes to avoid

**Target Audience:** All users, especially beginners

---

### [API_REFERENCE.md](API_REFERENCE.md) (14KB, 556 lines)
**Complete function and class reference**

Documents:
- All dashboard modules and functions
- All advanced model architectures
- All plot functions
- Configuration objects
- Data structures
- Usage examples
- Parameter descriptions

**Target Audience:** Developers and advanced users

---

## Quick Navigation

### For First-Time Users:
1. Start with [TUTORIAL.md](TUTORIAL.md) → Tutorial 1
2. Read relevant sections of [USER_GUIDE.md](USER_GUIDE.md)
3. Experiment with the dashboard
4. Refer to [API_REFERENCE.md](API_REFERENCE.md) for specific functions

### For Researchers:
1. Read [USER_GUIDE.md](USER_GUIDE.md) sections on:
   - System Parameters (understand N, K, M)
   - Model Architectures (choose the right model)
   - Evaluation Metrics (interpret results)
2. Follow [TUTORIAL.md](TUTORIAL.md) → Tutorial 2 (model comparison)
3. Follow [TUTORIAL.md](TUTORIAL.md) → Tutorial 3 (statistical analysis)

### For Developers:
1. Read [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md)
2. Check [API_REFERENCE.md](API_REFERENCE.md) for existing functions
3. Follow code style guidelines
4. Test your additions

### For Quick Testing:
1. Open notebook: `notebooks/PhD_Research_Dashboard.ipynb`
2. Run Cell 0 (auto-setup)
3. Run Cell 1 (load dashboard)
4. Run helper: `use_preset('quick_test')`
5. Click "RUN EXPERIMENT"

---

## Key Features Documented

### ✅ 8 Model Architectures
- MLP (Multi-Layer Perceptron)
- CNN (1D Convolutional)
- LSTM (Bidirectional)
- GRU (Bidirectional)
- Attention MLP
- Transformer
- ResNet-style MLP
- Hybrid CNN-LSTM

### ✅ 7 Optimizers
- Adam
- AdamW
- SGD
- RMSprop
- AdaGrad
- Adadelta
- Adamax

### ✅ 8 LR Schedulers
- StepLR
- MultiStepLR
- ExponentialLR
- CosineAnnealingLR
- CosineAnnealingWarmRestarts
- ReduceLROnPlateau
- OneCycleLR
- None (constant LR)

### ✅ 3 Loss Functions
- CrossEntropy
- Focal Loss
- Label Smoothing

### ✅ 6 Probe Methods
- continuous (random)
- binary ({0, π})
- 2bit ({0, π/2, π, 3π/2})
- hadamard (structured orthogonal)
- sobol (low-discrepancy)
- halton (low-discrepancy)

### ✅ 25+ Plot Types
Training:
- Training curves
- Learning rate schedule
- Gradient flow

Performance:
- Eta distribution
- CDF (Cumulative Distribution Function)
- PDF histogram
- Box plot
- Violin plot
- Scatter comparison
- Bar comparison
- Radar chart

Probe Analysis:
- Probe heatmap
- Correlation matrix
- Diversity analysis
- Probe power distribution

Advanced:
- Heatmap comparison
- 3D surface
- ROC curves
- Precision-Recall curves
- Confusion matrix
- Top-M comparison
- Baseline comparison
- Convergence analysis
- Parameter sensitivity
- Model complexity vs performance

### ✅ 10 Export Formats
Data: CSV, JSON, YAML, HDF5, Excel, Pickle
Plots: PNG, PDF, SVG, EPS

---

## System Requirements

### Software:
- Python 3.8+
- Jupyter Notebook or JupyterLab
- See requirements.txt for dependencies

### Hardware:
- CPU: Any modern processor
- RAM: 8GB minimum, 16GB recommended
- GPU: Optional but recommended for large models (Transformer)
- Storage: 1GB for code + generated results

### Dependencies:
Auto-installed by Cell 0 (auto-setup):
- torch >= 1.10.0
- numpy >= 1.21.0
- matplotlib >= 3.4.0
- seaborn >= 0.11.0
- scipy >= 1.7.0
- pandas >= 1.3.0
- ipywidgets >= 8.0.0
- tqdm >= 4.62.0
- pyyaml >= 5.4.0
- plotly >= 5.0.0
- scikit-learn >= 1.0.0
- openpyxl >= 3.0.0
- h5py >= 3.0.0

---

## Getting Started

### Method 1: Interactive Dashboard (Recommended)
```bash
# Open the notebook
jupyter notebook notebooks/PhD_Research_Dashboard.ipynb

# Run Cell 0 (auto-setup)
# Run Cell 1 (load dashboard)
# Configure and click "RUN EXPERIMENT"
```

### Method 2: Command Line
```python
from config import Config
from advanced_models import create_advanced_model
from dashboard_runner import run_single_experiment

# Create config and run
config = Config()
result = run_single_experiment(config)
```

### Method 3: Presets
```python
from notebooks.dashboard_utils import apply_preset

# Apply preset and run
apply_preset(widgets_dict, 'quick_test')
# Click "RUN EXPERIMENT"
```

---

## Common Use Cases

### Research Paper Experiment
1. Use preset 'standard' (N=32, K=64, M=8)
2. Enable multi-seed runs (5 seeds)
3. Compare 3-4 models
4. Generate comparison plots
5. Export results to CSV and PNG

### Quick Prototyping
1. Use preset 'quick_test' (N=16, K=32, M=4)
2. 10K samples, 20 epochs
3. MLP model
4. Fast iteration (~2 minutes)

### Large-Scale Study
1. Use preset 'high_capacity' (N=64, K=128, M=16)
2. 100K samples, 100 epochs
3. Transformer model
4. GPU recommended

### Ablation Study
1. Fix all parameters except one
2. Create parameter sweep
3. Run multiple experiments
4. Use parameter sensitivity plot

---

## Troubleshooting

### Setup Issues
**Problem:** Packages not installing
**Solution:** Re-run Cell 0, check internet connection

**Problem:** Widgets not displaying
**Solution:** Run `jupyter nbextension enable --py widgetsnbextension`

### Performance Issues
**Problem:** Training too slow
**Solution:** Use GPU, reduce model size, smaller dataset

**Problem:** Poor η values
**Solution:** Check M/K ratio, try different model/probe method

### Import Errors
**Problem:** Module not found
**Solution:** Ensure project root in Python path (Cell 0 does this)

---

## Support and Resources

### Documentation
- **This directory:** Complete documentation
- **Main README:** `../README.md`
- **Implementation notes:** `../IMPLEMENTATION_SUMMARY.md`

### Code Examples
- **Tutorial examples:** [TUTORIAL.md](TUTORIAL.md)
- **API examples:** [API_REFERENCE.md](API_REFERENCE.md)
- **Developer examples:** [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md)

### Repository
- **GitHub:** https://github.com/mohbagher/Probe_Based_ML_Session5
- **Issues:** Report bugs and request features
- **Discussions:** Ask questions and share ideas

---

## Version Information

**Dashboard Version:** 1.0.0 (Complete Implementation)
**Last Updated:** January 2026
**Author:** Mohammad Bagher
**License:** MIT

---

## Contributing

See [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md) for:
- Code style guidelines
- How to add new features
- Testing procedures
- Pull request process

---

## Citation

If you use this dashboard in your research, please cite:

```bibtex
@software{phd_research_dashboard,
  title = {PhD Research Dashboard: Interactive System for RIS Probe-Based ML},
  author = {Bagher, Mohammad},
  year = {2026},
  url = {https://github.com/mohbagher/Probe_Based_ML_Session5}
}
```

---

## Changelog

### Version 1.0.0 (January 2026)
- ✅ Complete implementation of all features
- ✅ 8 model architectures
- ✅ 7 optimizers, 8 schedulers
- ✅ 6 probe methods
- ✅ 25+ plot types
- ✅ Auto-setup cell
- ✅ Comprehensive documentation (46KB)
- ✅ Multi-model comparison
- ✅ Multi-seed statistical analysis
- ✅ Configuration management
- ✅ Zero incomplete code

---

## License

MIT License - see LICENSE file in repository root

---

**Happy Researching! 🎯🚀📊**
