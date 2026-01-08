# 📖 Tutorial - Step-by-Step Examples

## Tutorial 1: Your First Experiment

### Goal
Run a basic experiment with default settings.

### Steps

1. **Open the notebook**
   - Navigate to `notebooks/PhD_Research_Dashboard.ipynb`
   - Run Cell 0 (auto-setup) - wait for completion

2. **Run Cell 1**
   - This loads all widgets and creates the interface

3. **Configure basic parameters**
   - In "System & Physics" tab:
     - N = 32 (keep default)
     - K = 64 (keep default)
     - M = 8 (keep default)
     - Probe Type = continuous

4. **Select model**
   - In "Model" tab:
     - Model Architecture = MLP
     - Keep other defaults

5. **Set training**
   - In "Training" tab:
     - Epochs = 50
     - Batch Size = 128
     - Optimizer = Adam
     - Learning Rate = 0.001

6. **Configure evaluation**
   - In "Data & Evaluation" tab:
     - Select plots you want to see
     - Check "Save Results"

7. **Run the experiment**
   - Scroll to control panel
   - Click "🚀 RUN EXPERIMENT"
   - Wait for completion (~5 minutes on CPU)

8. **View results**
   - Check printed summary
   - View generated plots
   - Find saved results in output directory

### Expected Results
- η (eta) ≈ 0.85-0.92
- Top-1 accuracy ≈ 40-60%
- Training completes in 3-5 minutes

---

## Tutorial 2: Comparing Multiple Models

### Goal
Compare MLP, CNN, and Transformer performance.

### Steps

1. **Enable comparison mode**
   - Go to "Multi-Experiment" tab
   - Check "Multi-Model Comparison Mode"

2. **Select models to compare**
   - In "Models to Compare" list:
     - Select: MLP, CNN, Transformer
     - (Hold Ctrl/Cmd to select multiple)

3. **Configure system**
   - N = 32, K = 64, M = 8
   - Probe Type = hadamard (structured probes work well with CNN)

4. **Set training parameters**
   - Epochs = 50
   - Batch Size = 128
   - Optimizer = Adam
   - Learning Rate = 0.001

5. **Select comparison plots**
   - Box Plot (compare distributions)
   - Convergence Analysis (training curves)
   - Bar Comparison (final metrics)

6. **Run experiment**
   - Click "🚀 RUN EXPERIMENT"
   - Each model trains sequentially
   - Wait for all models to complete

7. **Analyze results**
   - Compare η values across models
   - Check training time for each
   - View comparison plots
   - Identify best model for your case

### Expected Results
- MLP: Fast training, good baseline
- CNN: Better with structured probes
- Transformer: Best performance with enough data

---

## Tutorial 3: Statistical Analysis with Multiple Seeds

### Goal
Get confidence intervals with multi-seed runs.

### Steps

1. **Enable multi-seed mode**
   - Go to "Multi-Experiment" tab
   - Check "Multi-Seed Runs"

2. **Configure seeds**
   - Number of Seeds = 5
   - Starting Seed = 42

3. **Select your configuration**
   - Model: MLP (or your choice)
   - N = 32, K = 64, M = 8
   - Epochs = 50

4. **Run experiment**
   - Click "🚀 RUN EXPERIMENT"
   - Watch as each seed runs
   - 5 complete experiments will execute

5. **Analyze statistical results**
   - Mean η ± std
   - Variance across seeds
   - Confidence in results

### Expected Results
- Mean η ≈ 0.88
- Std η ≈ 0.02-0.03
- Good for publication error bars

---

## Tutorial 4: Exploring Different Probe Methods

### Goal
Compare different probe generation methods.

### Method 1: Continuous (Random)
```
Probe Type = continuous
Good for: General testing
```

### Method 2: Binary
```
Probe Type = binary
Phases = {0, π}
Good for: Hardware with binary phase shifters
```

### Method 3: Hadamard
```
Probe Type = hadamard
Good for: Structured, orthogonal patterns
Best with: CNN model
```

### Method 4: Sobol (Low-Discrepancy)
```
Probe Type = sobol
Good for: Better coverage of phase space
```

### Experiment Plan
1. Set everything else constant
2. Run 4 experiments, one per probe type
3. Use comparison mode if desired
4. Compare η values

---

## Tutorial 5: Hyperparameter Tuning

### Goal
Find optimal learning rate and architecture.

### Experiment Matrix

| Run | LR    | Hidden Sizes    | Dropout |
|-----|-------|-----------------|---------|
| 1   | 1e-4  | 512,256,128     | 0.1     |
| 2   | 1e-3  | 512,256,128     | 0.1     |
| 3   | 1e-2  | 512,256,128     | 0.1     |
| 4   | 1e-3  | 1024,512,256    | 0.1     |
| 5   | 1e-3  | 512,256,128     | 0.2     |

### Steps for Each Run
1. Set parameters from table
2. Run experiment
3. Record η value
4. Save configuration

### Analysis
- Plot η vs learning rate
- Identify best configuration
- Use for final experiments

---

## Tutorial 6: Scaling to Larger Systems

### Goal
Test performance on larger RIS arrays.

### Progressive Scaling

**Stage 1: Small (Baseline)**
```
N = 16, K = 32, M = 4
Training: 10K samples, 30 epochs
Model: MLP
Purpose: Quick validation
```

**Stage 2: Medium (Standard)**
```
N = 32, K = 64, M = 8
Training: 50K samples, 50 epochs
Model: MLP or CNN
Purpose: Research experiments
```

**Stage 3: Large (Deployment)**
```
N = 64, K = 128, M = 16
Training: 100K samples, 100 epochs
Model: Transformer
Purpose: High-performance systems
```

**Stage 4: Very Large (Advanced)**
```
N = 128, K = 256, M = 32
Training: 200K samples, 100 epochs
Model: Transformer with GPU
Purpose: Asymptotic analysis
```

### Key Observations
- Training time scales ~quadratically with N
- Model capacity needs increase with K
- Transformer shines at large scale

---

## Tutorial 7: Debugging Poor Performance

### Scenario: Model η < 0.7

#### Step 1: Check Data
```python
# In evaluation tab, check:
- Training samples: Should be >> K
- M/K ratio: Should be ≤ 0.5
- Normalization: Should be enabled
```

#### Step 2: Check Model
```python
# Try:
- Larger model (more hidden units)
- Different architecture (try Transformer)
- More dropout if overfitting
```

#### Step 3: Check Training
```python
# Adjust:
- More epochs (try 100)
- Lower learning rate (try 1e-4)
- Different optimizer (try AdamW)
```

#### Step 4: Check Problem Setup
```python
# Verify:
- Probe type matches model (structured→CNN)
- M is not too small
- K has good coverage
```

---

## Tutorial 8: Production Deployment Workflow

### Goal
Create reproducible results for publication.

### Full Workflow

1. **Design Experiments**
   - Define research questions
   - Choose parameter ranges
   - Plan comparison matrix

2. **Quick Prototyping**
   - Use small N, K, fewer samples
   - Test all configurations quickly
   - Identify promising approaches

3. **Full-Scale Runs**
   - Use final parameters
   - Run multi-seed for statistics
   - Save all configurations

4. **Generate Results**
   - Enable all relevant plots
   - Export to multiple formats
   - Save models for reproducibility

5. **Documentation**
   - Save configurations (JSON/YAML)
   - Record experimental notes
   - Document any issues

6. **Publication Figures**
   - Export high-resolution plots (PDF/SVG)
   - Create comparison tables (CSV)
   - Include error bars (multi-seed)

---

## Common Workflows

### Quick Test Run
```
Time: 2-3 minutes
Config: N=16, K=32, M=4, 10K samples, 20 epochs
Purpose: Verify setup works
```

### Standard Experiment
```
Time: 5-10 minutes
Config: N=32, K=64, M=8, 50K samples, 50 epochs
Purpose: Research experiments
```

### Production Run
```
Time: 30-60 minutes
Config: N=64, K=128, M=16, 100K samples, 100 epochs, 5 seeds
Purpose: Publication results
```

---

## Tips & Tricks

### Speed Up Experiments
1. Use GPU if available
2. Reduce batch size for memory
3. Use early stopping
4. Start with smaller models

### Improve Performance
1. More training data
2. Larger models
3. Better probe methods (Hadamard, Sobol)
4. Hyperparameter tuning

### Avoid Common Mistakes
1. M > K (invalid)
2. Too few training samples
3. Learning rate too high
4. No early stopping → overfitting

---

## Next Steps

After completing these tutorials:
1. Read [USER_GUIDE.md](USER_GUIDE.md) for detailed explanations
2. Check [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md) to add features
3. See [API_REFERENCE.md](API_REFERENCE.md) for function details
4. Experiment with your own configurations!

---

## Getting Help

If you encounter issues:
1. Check documentation files
2. Review error messages carefully
3. Try simpler configuration first
4. Check GitHub issues
5. Start with Tutorial 1 to verify setup
