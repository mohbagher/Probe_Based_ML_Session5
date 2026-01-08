# 📋 API Reference

## Dashboard Modules

### dashboard_widgets.py

#### create_system_widgets()
Creates widgets for system parameters (N, K, M, probe types, channel models).

**Returns:** Dict of widgets

**Widgets:**
- `N`: IntSlider - Number of RIS elements (8-256)
- `K`: IntSlider - Codebook size (16-512)
- `M`: IntSlider - Sensing budget (2-64)
- `probe_type`: Dropdown - Probe generation method
- `phase_mode`: Dropdown - Continuous or discrete
- `phase_bits`: IntSlider - Phase quantization bits

#### create_model_widgets()
Creates widgets for model architecture selection and hyperparameters.

**Returns:** Dict of widgets

**Widgets:**
- `model_type`: Dropdown - Architecture selection
- `hidden_sizes`: Text - MLP hidden layer sizes
- `dropout_prob`: FloatSlider - Dropout probability
- CNN, LSTM, Transformer specific parameters

#### create_training_widgets()
Creates widgets for training configuration.

**Returns:** Dict of widgets

**Widgets:**
- `epochs`: IntSlider - Training epochs
- `batch_size`: Dropdown - Batch size
- `optimizer`: Dropdown - Optimizer selection
- `learning_rate`: FloatText - Learning rate
- `scheduler`: Dropdown - LR scheduler
- Loss function configuration

#### create_data_widgets()
Creates widgets for data generation parameters.

**Returns:** Dict of widgets

#### create_evaluation_widgets()
Creates widgets for evaluation and visualization.

**Returns:** Dict of widgets

#### create_multi_experiment_widgets()
Creates widgets for multi-experiment mode.

**Returns:** Dict of widgets

#### create_control_widgets()
Creates control buttons and status display.

**Returns:** Dict of widgets

#### create_all_widgets()
Creates all widgets organized by category.

**Returns:** Dict with all widget categories

---

### dashboard_callbacks.py

#### create_callbacks(widgets_dict)
Create and register all callback functions for widget interactions.

**Args:**
- `widgets_dict`: Dictionary of all widgets

**Returns:** Dict of callback functions

**Callbacks:**
- `on_model_type_change`: Show/hide model-specific parameters
- `on_optimizer_change`: Show/hide optimizer-specific parameters
- `on_scheduler_change`: Show/hide scheduler-specific parameters
- `validate_M_vs_K`: Ensure M ≤ K
- `on_comparison_mode_change`: Toggle comparison mode
- `on_multi_seed_change`: Toggle multi-seed mode

#### create_button_callbacks(widgets_dict, runner_func)
Create callbacks for control buttons.

**Args:**
- `widgets_dict`: Dictionary of all widgets
- `runner_func`: Function to run experiments

**Returns:** Dict of button callbacks

---

### dashboard_runner.py

#### extract_config_from_widgets(widgets_dict)
Extract configuration from widget values.

**Args:**
- `widgets_dict`: Dictionary of all widgets

**Returns:** `Config` object

#### create_optimizer(model, config_widgets)
Create optimizer based on widget selection.

**Args:**
- `model`: PyTorch model
- `config_widgets`: Training configuration widgets

**Returns:** PyTorch optimizer

**Supported Optimizers:**
- Adam, AdamW, SGD, RMSprop, AdaGrad, Adadelta, Adamax

#### create_scheduler(optimizer, config_widgets, steps_per_epoch)
Create learning rate scheduler.

**Args:**
- `optimizer`: PyTorch optimizer
- `config_widgets`: Training configuration widgets
- `steps_per_epoch`: Steps per epoch (optional)

**Returns:** PyTorch scheduler or None

**Supported Schedulers:**
- StepLR, MultiStepLR, ExponentialLR
- CosineAnnealingLR, CosineAnnealingWarmRestarts
- ReduceLROnPlateau, OneCycleLR

#### create_loss_function(config_widgets, num_classes)
Create loss function.

**Args:**
- `config_widgets`: Training configuration widgets
- `num_classes`: Number of output classes

**Returns:** PyTorch loss function

**Supported Loss Functions:**
- CrossEntropy, LabelSmoothing, FocalLoss

#### run_single_experiment(widgets_dict, model_type, seed)
Run a single experiment.

**Args:**
- `widgets_dict`: Dictionary of all widgets
- `model_type`: Override model type (optional)
- `seed`: Override seed (optional)

**Returns:** Dict with results, history, model, config

**Result Structure:**
```python
{
    'results': EvaluationResults,
    'history': {'train_loss': [], 'val_loss': [], ...},
    'model': PyTorch model,
    'config': Config object,
    'probe_bank': ProbeBank,
    'model_type': str
}
```

#### run_experiments(widgets_dict)
Main experiment runner - handles single/multi-model/multi-seed experiments.

**Args:**
- `widgets_dict`: Dictionary of all widgets

**Returns:** Dict of all results

#### save_all_results(all_results, widgets_dict)
Save all results in various formats.

**Args:**
- `all_results`: Dict of experiment results
- `widgets_dict`: Dictionary of all widgets

**Saved Formats:** CSV, JSON, YAML, Pickle, Excel, HDF5

#### generate_plots(all_results, widgets_dict)
Generate selected plots.

**Args:**
- `all_results`: Dict of experiment results
- `widgets_dict`: Dictionary of all widgets

---

### dashboard_utils.py

#### create_tab_layout(widgets_dict)
Create organized tab layout for the dashboard.

**Args:**
- `widgets_dict`: Dictionary of all widgets

**Returns:** Tab widget with all panels

**Tabs:**
1. System & Physics
2. Model Architecture
3. Training Configuration
4. Data & Evaluation
5. Multi-Experiment

#### create_control_panel(widgets_dict)
Create control panel with buttons and status.

**Args:**
- `widgets_dict`: Dictionary of all widgets

**Returns:** VBox with control elements

#### create_header()
Create dashboard header with title and description.

**Returns:** HTML widget

#### create_quick_stats_panel()
Create panel showing system info.

**Returns:** HTML widget

#### save_config_to_file(widgets_dict, filepath)
Save current widget configuration to file.

**Args:**
- `widgets_dict`: Dictionary of all widgets
- `filepath`: Path to save (.json or .yaml)

**Example:**
```python
save_config_to_file(widgets_dict, 'my_config.json')
```

#### load_config_from_file(widgets_dict, filepath)
Load widget configuration from file.

**Args:**
- `widgets_dict`: Dictionary of all widgets
- `filepath`: Path to load from

**Example:**
```python
load_config_from_file(widgets_dict, 'my_config.json')
```

#### validate_configuration(widgets_dict)
Validate current configuration.

**Args:**
- `widgets_dict`: Dictionary of all widgets

**Returns:** Tuple (is_valid, list of errors)

**Example:**
```python
valid, errors = validate_configuration(widgets_dict)
if not valid:
    print("Errors:", errors)
```

#### print_configuration_summary(widgets_dict)
Print summary of current configuration.

**Args:**
- `widgets_dict`: Dictionary of all widgets

#### create_preset_configs()
Create preset configurations.

**Returns:** Dict of presets

**Available Presets:**
- `quick_test`: Fast configuration for testing
- `standard`: Default research setup
- `high_capacity`: Large-scale deployment
- `limited_budget`: Extreme sensing constraint

#### apply_preset(widgets_dict, preset_name)
Apply a preset configuration.

**Args:**
- `widgets_dict`: Dictionary of all widgets
- `preset_name`: Name of preset

**Example:**
```python
apply_preset(widgets_dict, 'quick_test')
```

---

## Core Modules

### advanced_models.py

#### create_advanced_model(model_type, K, config)
Factory function to create model architectures.

**Args:**
- `model_type`: Type of model ('mlp', 'cnn', 'lstm', 'gru', 'attention', 'transformer', 'resnet', 'hybrid')
- `K`: Number of probes (output size)
- `config`: Dict with model-specific parameters

**Returns:** PyTorch model instance

**Example:**
```python
model = create_advanced_model('transformer', K=64, config={
    'd_model': 256,
    'num_heads': 8,
    'num_layers': 3
})
```

#### AdvancedMLP(K, hidden_sizes, dropout_prob, use_batch_norm)
Enhanced MLP architecture.

**Args:**
- `K`: Number of output classes
- `hidden_sizes`: List of hidden layer sizes
- `dropout_prob`: Dropout probability (default: 0.1)
- `use_batch_norm`: Use batch normalization (default: True)

#### CNN1D(K, num_filters, kernel_sizes, dropout_prob)
1D CNN for probe sequences.

**Args:**
- `K`: Number of output classes
- `num_filters`: List of filter counts per layer
- `kernel_sizes`: List of kernel sizes per layer
- `dropout_prob`: Dropout probability

#### BiLSTM(K, hidden_size, num_layers, dropout_prob)
Bidirectional LSTM.

**Args:**
- `K`: Number of output classes
- `hidden_size`: LSTM hidden size
- `num_layers`: Number of LSTM layers
- `dropout_prob`: Dropout probability

#### BiGRU(K, hidden_size, num_layers, dropout_prob)
Bidirectional GRU.

**Args:** Similar to BiLSTM

#### AttentionMLP(K, d_model, num_heads, hidden_sizes, dropout_prob)
MLP with multi-head attention.

**Args:**
- `K`: Number of output classes
- `d_model`: Model dimension
- `num_heads`: Number of attention heads
- `hidden_sizes`: FC layer sizes
- `dropout_prob`: Dropout probability

#### TransformerModel(K, d_model, num_heads, num_layers, dim_feedforward, dropout_prob)
Transformer encoder.

**Args:**
- `K`: Number of output classes
- `d_model`: Model dimension
- `num_heads`: Number of attention heads
- `num_layers`: Number of transformer layers
- `dim_feedforward`: FFN dimension
- `dropout_prob`: Dropout probability

#### ResNetMLP(K, hidden_size, num_blocks, dropout_prob)
ResNet-style MLP with skip connections.

**Args:**
- `K`: Number of output classes
- `hidden_size`: Hidden layer size
- `num_blocks`: Number of residual blocks
- `dropout_prob`: Dropout probability

#### HybridCNNLSTM(K, num_filters, kernel_size, lstm_hidden, lstm_layers, dropout_prob)
Hybrid CNN-LSTM architecture.

**Args:**
- `K`: Number of output classes
- `num_filters`: Number of CNN filters
- `kernel_size`: CNN kernel size
- `lstm_hidden`: LSTM hidden size
- `lstm_layers`: Number of LSTM layers
- `dropout_prob`: Dropout probability

---

### plot_registry.py

**Plot Functions:** All plot functions follow the pattern:

```python
plot_function(results, save_path=None)
```

**Available Plots:**
- `plot_training_history(history)`: Training curves
- `plot_learning_rate_schedule(history)`: LR over epochs
- `plot_gradient_flow(model)`: Gradient magnitudes
- `plot_eta_distribution(results)`: Eta histogram
- `plot_cdf(results)`: Cumulative distribution
- `plot_pdf_histogram(results)`: Probability density
- `plot_box_comparison(results_dict)`: Box plot comparison
- `plot_violin(results_dict)`: Violin plot
- `plot_scatter_comparison(results_list, labels)`: Scatter plot
- `plot_bar_comparison(results_dict, metric)`: Bar chart
- `plot_radar_chart(results_dict, metrics)`: Radar chart
- `plot_heatmap(probe_bank)`: Probe phase heatmap
- `plot_correlation_matrix(probe_bank)`: Probe similarity
- `plot_diversity_analysis(probe_bank)`: Diversity metrics
- `plot_3d_surface(results_grid, param1, param2, ...)`: 3D surface
- `plot_roc_curves(results_list, labels)`: ROC curves
- `plot_precision_recall(results_list, labels)`: PR curves
- `plot_confusion_matrix(predictions, targets, K)`: Confusion matrix
- `plot_convergence_analysis(history_list, labels)`: Multi-model convergence
- `plot_parameter_sensitivity(results_list, param_values, param_name)`: Sensitivity
- `plot_model_complexity_vs_performance(results_dict, param_counts)`: Complexity plot

---

## Configuration Objects

### SystemConfig
```python
SystemConfig(
    N=32,                    # Number of RIS elements
    K=64,                    # Codebook size
    M=8,                     # Sensing budget
    P_tx=1.0,               # Transmit power
    sigma_h_sq=1.0,         # BS-RIS channel variance
    sigma_g_sq=1.0,         # RIS-UE channel variance
    probe_type='continuous', # Probe type
    phase_mode='continuous', # Phase mode
    phase_bits=3             # Phase bits (discrete)
)
```

### DataConfig
```python
DataConfig(
    n_train=50000,          # Training samples
    n_val=5000,             # Validation samples
    n_test=5000,            # Test samples
    seed=42,                # Random seed
    normalize_input=True,   # Normalize inputs
    normalization_type='mean' # Normalization type
)
```

### TrainingConfig
```python
TrainingConfig(
    epochs=50,              # Training epochs
    batch_size=128,         # Batch size
    learning_rate=0.001,    # Learning rate
    weight_decay=0.0,       # Weight decay
    early_stopping=True,    # Early stopping
    patience=10             # ES patience
)
```

---

## Usage Examples

### Basic Experiment
```python
from dashboard_widgets import create_all_widgets
from dashboard_callbacks import create_callbacks
from dashboard_runner import run_experiments

# Create widgets
widgets = create_all_widgets()

# Setup callbacks
callbacks = create_callbacks(widgets)

# Run experiment
results = run_experiments(widgets)
```

### Save/Load Configuration
```python
from dashboard_utils import save_config_to_file, load_config_from_file

# Save
save_config_to_file(widgets, 'my_experiment.json')

# Load
load_config_from_file(widgets, 'my_experiment.json')
```

### Custom Model
```python
from advanced_models import create_advanced_model

model = create_advanced_model('transformer', K=64, config={
    'd_model': 256,
    'num_heads': 8,
    'num_layers': 3,
    'dim_feedforward': 512,
    'dropout_prob': 0.1
})
```

---

## Data Structures

### EvaluationResults
```python
@dataclass
class EvaluationResults:
    accuracy_top1: float    # Top-1 accuracy
    accuracy_top2: float    # Top-2 accuracy
    accuracy_top4: float    # Top-4 accuracy
    accuracy_top8: float    # Top-8 accuracy
    eta_top1: float         # η for top-1
    eta_top2: float         # η for top-2
    eta_top4: float         # η for top-4
    eta_top8: float         # η for top-8
    eta_random_1: float     # Random baseline
    eta_random_M: float     # Best of M random
    eta_best_observed: float # Best observed
    eta_oracle: float       # Oracle (should be 1.0)
    M: int                  # Sensing budget
    K: int                  # Codebook size
```

### ProbeBank
```python
@dataclass
class ProbeBank:
    phases: np.ndarray      # (K, N) phase values
    K: int                  # Number of probes
    N: int                  # Number of elements
    probe_type: str         # Probe type
```

---

For usage instructions, see [USER_GUIDE.md](USER_GUIDE.md)
For examples, see [TUTORIAL.md](TUTORIAL.md)
For extending, see [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md)
