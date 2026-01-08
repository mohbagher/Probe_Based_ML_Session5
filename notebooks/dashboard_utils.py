"""
Dashboard Utilities for PhD Research Dashboard.

Helper functions for configuration management, validation, and UI layout.
"""

from pathlib import Path
import json
import yaml
from ipywidgets import VBox, HBox, HTML, Tab, Accordion
from IPython.display import display, clear_output


def create_tab_layout(widgets_dict):
    """
    Create organized tab layout for the dashboard.
    
    Args:
        widgets_dict: Dictionary of all widgets
        
    Returns:
        Tab widget with all organized panels
    """
    # System & Physics Tab
    system_tab = VBox([
        HTML("<h2>🌐 System Parameters & Physics</h2>"),
        HTML("<hr>"),
        HTML("<h3>RIS System</h3>"),
        widgets_dict['system']['N'],
        widgets_dict['system']['K'],
        widgets_dict['system']['M'],
        HTML("<br><h3>Channel Parameters</h3>"),
        widgets_dict['system']['P_tx'],
        widgets_dict['system']['sigma_h_sq'],
        widgets_dict['system']['sigma_g_sq'],
        HTML("<br><h3>Probe Configuration</h3>"),
        widgets_dict['system']['probe_type'],
        widgets_dict['system']['phase_mode'],
        widgets_dict['system']['phase_bits'],
    ], layout={'padding': '20px'})
    
    # Model Architecture Tab
    model_tab = VBox([
        HTML("<h2>🧠 Model Architecture</h2>"),
        HTML("<hr>"),
        HTML("<h3>Architecture Selection</h3>"),
        widgets_dict['model']['model_type'],
        HTML("<br><h3>General Hyperparameters</h3>"),
        widgets_dict['model']['hidden_sizes'],
        widgets_dict['model']['dropout_prob'],
        widgets_dict['model']['use_batch_norm'],
        HTML("<br><h3>CNN-Specific</h3>"),
        widgets_dict['model']['cnn_filters'],
        widgets_dict['model']['cnn_kernels'],
        HTML("<br><h3>RNN-Specific (LSTM/GRU/Hybrid)</h3>"),
        widgets_dict['model']['rnn_hidden_size'],
        widgets_dict['model']['rnn_num_layers'],
        HTML("<br><h3>Transformer-Specific</h3>"),
        widgets_dict['model']['transformer_d_model'],
        widgets_dict['model']['transformer_num_heads'],
        widgets_dict['model']['transformer_num_layers'],
        widgets_dict['model']['transformer_dim_feedforward'],
        HTML("<br><h3>ResNet-Specific</h3>"),
        widgets_dict['model']['resnet_hidden_size'],
        widgets_dict['model']['resnet_num_blocks'],
    ], layout={'padding': '20px'})
    
    # Training Configuration Tab
    training_tab = VBox([
        HTML("<h2>⚙️ Training Configuration</h2>"),
        HTML("<hr>"),
        HTML("<h3>Basic Training</h3>"),
        widgets_dict['training']['epochs'],
        widgets_dict['training']['batch_size'],
        HTML("<br><h3>Optimization</h3>"),
        widgets_dict['training']['optimizer'],
        widgets_dict['training']['learning_rate'],
        widgets_dict['training']['weight_decay'],
        widgets_dict['training']['momentum'],
        HTML("<br><h3>Learning Rate Scheduling</h3>"),
        widgets_dict['training']['scheduler'],
        widgets_dict['training']['scheduler_step_size'],
        widgets_dict['training']['scheduler_gamma'],
        widgets_dict['training']['scheduler_patience'],
        HTML("<br><h3>Loss Function</h3>"),
        widgets_dict['training']['loss_function'],
        widgets_dict['training']['label_smoothing'],
        HTML("<br><h3>Early Stopping</h3>"),
        widgets_dict['training']['early_stopping'],
        widgets_dict['training']['early_stopping_patience'],
    ], layout={'padding': '20px'})
    
    # Data & Evaluation Tab
    data_eval_tab = VBox([
        HTML("<h2>📊 Data & Evaluation</h2>"),
        HTML("<hr>"),
        HTML("<h3>Dataset Configuration</h3>"),
        widgets_dict['data']['n_train'],
        widgets_dict['data']['n_val'],
        widgets_dict['data']['n_test'],
        widgets_dict['data']['seed'],
        widgets_dict['data']['normalize_input'],
        widgets_dict['data']['normalization_type'],
        HTML("<br><h3>Evaluation & Visualization</h3>"),
        widgets_dict['evaluation']['plot_types'],
        HTML("<br>"),
        widgets_dict['evaluation']['export_format'],
        widgets_dict['evaluation']['save_results'],
        widgets_dict['evaluation']['save_model'],
        widgets_dict['evaluation']['output_dir'],
    ], layout={'padding': '20px'})
    
    # Multi-Experiment Tab
    multi_tab = VBox([
        HTML("<h2>🔬 Multi-Experiment Mode</h2>"),
        HTML("<hr>"),
        HTML("<h3>Model Comparison</h3>"),
        HTML("<p>Enable this to compare multiple models in a single run.</p>"),
        widgets_dict['multi_experiment']['comparison_mode'],
        widgets_dict['multi_experiment']['models_to_compare'],
        HTML("<br><h3>Statistical Analysis</h3>"),
        HTML("<p>Run experiments with multiple random seeds for confidence intervals.</p>"),
        widgets_dict['multi_experiment']['multi_seed'],
        widgets_dict['multi_experiment']['num_seeds'],
        widgets_dict['multi_experiment']['seed_start'],
    ], layout={'padding': '20px'})
    
    # Create tab widget
    tabs = Tab()
    tabs.children = [system_tab, model_tab, training_tab, data_eval_tab, multi_tab]
    tabs.set_title(0, '🌐 System & Physics')
    tabs.set_title(1, '🧠 Model')
    tabs.set_title(2, '⚙️ Training')
    tabs.set_title(3, '📊 Data & Evaluation')
    tabs.set_title(4, '🔬 Multi-Experiment')
    
    return tabs


def create_control_panel(widgets_dict):
    """
    Create control panel with buttons and status.
    
    Args:
        widgets_dict: Dictionary of all widgets
        
    Returns:
        VBox with control elements
    """
    button_row = HBox([
        widgets_dict['control']['run_button'],
        widgets_dict['control']['stop_button'],
        widgets_dict['control']['clear_button'],
    ], layout={'justify_content': 'center', 'padding': '20px'})
    
    status_panel = VBox([
        widgets_dict['control']['status_text'],
        widgets_dict['control']['progress_text'],
    ])
    
    control_panel = VBox([
        HTML("<h2>🎮 Experiment Control</h2>"),
        HTML("<hr>"),
        button_row,
        status_panel,
    ], layout={'padding': '20px', 'border': '2px solid #ccc', 'border_radius': '5px'})
    
    return control_panel


def create_header():
    """Create dashboard header with title and description."""
    header_html = """
    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 30px; border-radius: 10px; color: white; margin-bottom: 20px;'>
        <h1 style='margin: 0; font-size: 36px;'>🎯 PhD Research Dashboard</h1>
        <h2 style='margin: 10px 0 0 0; font-weight: 300;'>
            RIS Probe-Based Machine Learning - Complete Interactive Interface
        </h2>
        <p style='margin: 15px 0 0 0; font-size: 14px; opacity: 0.9;'>
            ✨ Ultra-customizable • 🚀 Production-ready • 📊 Publication-quality results
        </p>
    </div>
    """
    return HTML(header_html)


def create_quick_stats_panel():
    """Create panel showing quick statistics and system info."""
    import torch
    
    gpu_info = "🚀 GPU Available" if torch.cuda.is_available() else "💻 CPU Mode"
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_info += f": {gpu_name}"
    
    stats_html = f"""
    <div style='background: #f8f9fa; padding: 15px; border-radius: 5px; margin-bottom: 20px;'>
        <h3 style='margin-top: 0;'>📈 Quick Info</h3>
        <ul style='list-style: none; padding-left: 0;'>
            <li><strong>Device:</strong> {gpu_info}</li>
            <li><strong>Available Models:</strong> MLP, CNN, LSTM, GRU, Attention, Transformer, ResNet, Hybrid</li>
            <li><strong>Probe Methods:</strong> continuous, binary, 2bit, hadamard, sobol, halton</li>
            <li><strong>Plot Types:</strong> 25+ visualization options</li>
            <li><strong>Export Formats:</strong> CSV, JSON, YAML, HDF5, Excel, PNG, PDF, SVG, EPS</li>
        </ul>
    </div>
    """
    return HTML(stats_html)


def create_documentation_links():
    """Create panel with links to documentation."""
    docs_html = """
    <div style='background: #e7f3ff; padding: 15px; border-radius: 5px; margin-bottom: 20px;'>
        <h3 style='margin-top: 0;'>📚 Documentation</h3>
        <ul>
            <li><a href='notebooks/docs/USER_GUIDE.md' target='_blank'>User Guide</a> - Comprehensive usage manual</li>
            <li><a href='notebooks/docs/DEVELOPER_GUIDE.md' target='_blank'>Developer Guide</a> - Extension instructions</li>
            <li><a href='notebooks/docs/TUTORIAL.md' target='_blank'>Tutorial</a> - Step-by-step examples</li>
            <li><a href='notebooks/docs/API_REFERENCE.md' target='_blank'>API Reference</a> - Function documentation</li>
        </ul>
    </div>
    """
    return HTML(docs_html)


def save_config_to_file(widgets_dict, filepath):
    """
    Save current widget configuration to file.
    
    Args:
        widgets_dict: Dictionary of all widgets
        filepath: Path to save configuration
    """
    config = {}
    
    for category, widgets in widgets_dict.items():
        if category == 'control':
            continue
        config[category] = {}
        for name, widget in widgets.items():
            try:
                config[category][name] = widget.value
            except:
                pass
    
    filepath = Path(filepath)
    if filepath.suffix == '.json':
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)
    elif filepath.suffix in ['.yaml', '.yml']:
        with open(filepath, 'w') as f:
            yaml.dump(config, f)
    else:
        raise ValueError("Unsupported file format. Use .json or .yaml")
    
    print(f"✅ Configuration saved to {filepath}")


def load_config_from_file(widgets_dict, filepath):
    """
    Load widget configuration from file.
    
    Args:
        widgets_dict: Dictionary of all widgets
        filepath: Path to load configuration from
    """
    filepath = Path(filepath)
    
    if filepath.suffix == '.json':
        with open(filepath, 'r') as f:
            config = json.load(f)
    elif filepath.suffix in ['.yaml', '.yml']:
        with open(filepath, 'r') as f:
            config = yaml.safe_load(f)
    else:
        raise ValueError("Unsupported file format. Use .json or .yaml")
    
    # Apply configuration to widgets
    for category, settings in config.items():
        if category in widgets_dict:
            for name, value in settings.items():
                if name in widgets_dict[category]:
                    try:
                        widgets_dict[category][name].value = value
                    except:
                        print(f"Warning: Could not set {category}.{name}")
    
    print(f"✅ Configuration loaded from {filepath}")


def validate_configuration(widgets_dict):
    """
    Validate current configuration for common issues.
    
    Args:
        widgets_dict: Dictionary of all widgets
        
    Returns:
        Tuple of (is_valid, list of error messages)
    """
    errors = []
    
    # Check M <= K
    M = widgets_dict['system']['M'].value
    K = widgets_dict['system']['K'].value
    if M > K:
        errors.append(f"M ({M}) cannot be greater than K ({K})")
    
    # Check reasonable values
    N = widgets_dict['system']['N'].value
    if N < 4:
        errors.append(f"N ({N}) is too small, recommended minimum is 8")
    
    if K < 8:
        errors.append(f"K ({K}) is too small, recommended minimum is 16")
    
    # Check training parameters
    epochs = widgets_dict['training']['epochs'].value
    if epochs < 10:
        errors.append(f"Epochs ({epochs}) might be too few for convergence")
    
    lr = widgets_dict['training']['learning_rate'].value
    if lr > 0.1:
        errors.append(f"Learning rate ({lr}) might be too large")
    if lr < 1e-6:
        errors.append(f"Learning rate ({lr}) might be too small")
    
    # Check data sizes
    n_train = widgets_dict['data']['n_train'].value
    if n_train < 1000:
        errors.append(f"Training size ({n_train}) might be too small")
    
    return len(errors) == 0, errors


def print_configuration_summary(widgets_dict):
    """Print a summary of current configuration."""
    print("\n" + "="*70)
    print("CONFIGURATION SUMMARY")
    print("="*70)
    
    print("\n🌐 System:")
    print(f"  N = {widgets_dict['system']['N'].value}")
    print(f"  K = {widgets_dict['system']['K'].value}")
    print(f"  M = {widgets_dict['system']['M'].value}")
    print(f"  Probe Type = {widgets_dict['system']['probe_type'].value}")
    
    print("\n🧠 Model:")
    print(f"  Architecture = {widgets_dict['model']['model_type'].value}")
    print(f"  Dropout = {widgets_dict['model']['dropout_prob'].value}")
    
    print("\n⚙️ Training:")
    print(f"  Epochs = {widgets_dict['training']['epochs'].value}")
    print(f"  Batch Size = {widgets_dict['training']['batch_size'].value}")
    print(f"  Optimizer = {widgets_dict['training']['optimizer'].value}")
    print(f"  Learning Rate = {widgets_dict['training']['learning_rate'].value}")
    print(f"  Scheduler = {widgets_dict['training']['scheduler'].value}")
    
    print("\n📊 Data:")
    print(f"  Training Samples = {widgets_dict['data']['n_train'].value}")
    print(f"  Validation Samples = {widgets_dict['data']['n_val'].value}")
    print(f"  Test Samples = {widgets_dict['data']['n_test'].value}")
    print(f"  Seed = {widgets_dict['data']['seed'].value}")
    
    print("\n" + "="*70 + "\n")


def create_preset_configs():
    """
    Create preset configurations for common use cases.
    
    Returns:
        Dictionary of preset configurations
    """
    presets = {
        'quick_test': {
            'name': 'Quick Test',
            'description': 'Fast configuration for testing',
            'system': {'N': 16, 'K': 32, 'M': 4},
            'data': {'n_train': 10000, 'n_val': 1000, 'n_test': 1000},
            'training': {'epochs': 20, 'batch_size': 256},
        },
        'standard': {
            'name': 'Standard Research',
            'description': 'Default configuration for research',
            'system': {'N': 32, 'K': 64, 'M': 8},
            'data': {'n_train': 50000, 'n_val': 5000, 'n_test': 5000},
            'training': {'epochs': 50, 'batch_size': 128},
        },
        'high_capacity': {
            'name': 'High Capacity',
            'description': 'Large-scale deployment scenario',
            'system': {'N': 64, 'K': 128, 'M': 16},
            'data': {'n_train': 100000, 'n_val': 10000, 'n_test': 10000},
            'training': {'epochs': 100, 'batch_size': 256},
        },
        'limited_budget': {
            'name': 'Limited Budget',
            'description': 'Extreme sensing constraint',
            'system': {'N': 32, 'K': 64, 'M': 4},
            'data': {'n_train': 50000, 'n_val': 5000, 'n_test': 5000},
            'training': {'epochs': 50, 'batch_size': 128},
        },
    }
    return presets


def apply_preset(widgets_dict, preset_name):
    """
    Apply a preset configuration to widgets.
    
    Args:
        widgets_dict: Dictionary of all widgets
        preset_name: Name of the preset to apply
    """
    presets = create_preset_configs()
    
    if preset_name not in presets:
        print(f"Unknown preset: {preset_name}")
        return
    
    preset = presets[preset_name]
    print(f"Applying preset: {preset['name']}")
    print(f"Description: {preset['description']}")
    
    for category, settings in preset.items():
        if category in ['name', 'description']:
            continue
        if category in widgets_dict:
            for param, value in settings.items():
                if param in widgets_dict[category]:
                    widgets_dict[category][param].value = value
    
    print("✅ Preset applied")
