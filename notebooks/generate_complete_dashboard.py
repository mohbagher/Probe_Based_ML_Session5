"""
Generate the complete PhD Research Dashboard notebook with auto-setup cell.
"""

import json
from pathlib import Path

# Auto-setup cell content
auto_setup_cell = '''"""
🔧 AUTOMATIC SETUP CELL
Run this cell first - it will check and install everything needed!
"""

import sys
import subprocess
import os
from pathlib import Path

print("="*70)
print("🔧 SETTING UP PHD RESEARCH DASHBOARD")
print("="*70)

# 1. Verify we're in correct directory
notebook_dir = Path.cwd()
if notebook_dir.name == 'notebooks':
    project_root = notebook_dir.parent
else:
    project_root = notebook_dir

print(f"\\n📁 Project Root: {project_root}")

# 2. Add project root to path
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
print(f"✅ Python path configured")

# 3. Check for required files
required_files = [
    'config.py',
    'model.py',
    'training.py',
    'evaluation.py',
    'data_generation.py',
    'advanced_models.py',
    'plot_registry.py',
    'experiments/probe_generators.py',
    'notebooks/dashboard_widgets.py',
    'notebooks/dashboard_callbacks.py',
    'notebooks/dashboard_runner.py',
    'notebooks/dashboard_utils.py',
]

missing_files = []
for file in required_files:
    if not (project_root / file).exists():
        missing_files.append(file)
        print(f"❌ Missing: {file}")
    else:
        print(f"✅ Found: {file}")

if missing_files:
    print(f"\\n⚠️  Missing {len(missing_files)} required files!")
    print("Please ensure all project files are present.")
else:
    print(f"\\n✅ All {len(required_files)} required files found!")

# 4. Check Python version
py_version = sys.version_info
print(f"\\n🐍 Python Version: {py_version.major}.{py_version.minor}.{py_version.micro}")
if py_version.major < 3 or (py_version.major == 3 and py_version.minor < 8):
    print("⚠️  Warning: Python 3.8+ recommended")
else:
    print("✅ Python version compatible")

# 5. Check/Install required packages
required_packages = {
    'torch': 'torch>=1.10.0',
    'numpy': 'numpy>=1.21.0',
    'matplotlib': 'matplotlib>=3.4.0',
    'seaborn': 'seaborn>=0.11.0',
    'scipy': 'scipy>=1.7.0',
    'pandas': 'pandas>=1.3.0',
    'ipywidgets': 'ipywidgets>=8.0.0',
    'tqdm': 'tqdm>=4.62.0',
    'yaml': 'pyyaml>=5.4.0',
    'plotly': 'plotly>=5.0.0',
    'sklearn': 'scikit-learn>=1.0.0',
}

print(f"\\n📦 Checking packages...")
missing_packages = []
for package_name, package_spec in required_packages.items():
    try:
        __import__(package_name)
        print(f"✅ {package_name} installed")
    except ImportError:
        missing_packages.append(package_spec)
        print(f"❌ {package_name} not found")

if missing_packages:
    print(f"\\n⚙️  Installing {len(missing_packages)} missing packages...")
    for package in missing_packages:
        print(f"   Installing {package}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])
            print(f"   ✅ {package} installed successfully")
        except Exception as e:
            print(f"   ⚠️  Failed to install {package}: {e}")
    print("\\n✅ Package installation completed!")
else:
    print("✅ All packages already installed!")

# 6. GPU Check
try:
    import torch
    if torch.cuda.is_available():
        print(f"\\n🚀 GPU Available: {torch.cuda.get_device_name(0)}")
        print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print(f"\\n💻 Running on CPU (GPU not available)")
except Exception as e:
    print(f"\\n💻 PyTorch check failed: {e}")

# 7. Enable widget extensions
try:
    from IPython.display import display, Javascript
    display(Javascript('IPython.OutputArea.prototype._should_scroll = function(lines) { return false; }'))
    print("\\n✅ Jupyter widgets configured")
except:
    print("\\n⚠️  Could not configure widget display (non-critical)")

print("\\n" + "="*70)
print("✅ SETUP COMPLETE! Ready to use the dashboard.")
print("="*70)
print("\\n💡 Next: Run the next cell to load the dashboard")
'''

# Main dashboard cell
main_cell = '''# ============================================================================
# 🎯 PHD RESEARCH DASHBOARD - MAIN INTERFACE
# ============================================================================

from IPython.display import display, HTML, clear_output
import sys
from pathlib import Path

# Ensure project root is in path
notebook_dir = Path.cwd()
project_root = notebook_dir.parent if notebook_dir.name == 'notebooks' else notebook_dir
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import dashboard modules
from notebooks.dashboard_widgets import create_all_widgets
from notebooks.dashboard_callbacks import create_callbacks, create_button_callbacks
from notebooks.dashboard_runner import run_experiments
from notebooks.dashboard_utils import (
    create_tab_layout, 
    create_control_panel, 
    create_header,
    create_quick_stats_panel,
    create_documentation_links
)

# Create all widgets
print("Creating dashboard interface...")
widgets_dict = create_all_widgets()

# Setup callbacks
callbacks = create_callbacks(widgets_dict)

# Create UI layout
header = create_header()
stats_panel = create_quick_stats_panel()
docs_panel = create_documentation_links()
tabs = create_tab_layout(widgets_dict)
control_panel = create_control_panel(widgets_dict)

# Setup button callbacks
button_callbacks = create_button_callbacks(widgets_dict, run_experiments)

# Display the dashboard
display(header)
display(stats_panel)
display(docs_panel)
display(tabs)
display(control_panel)

print("✅ Dashboard loaded! Configure your experiment and click 'RUN EXPERIMENT'")
'''

# Helper functions cell
helper_cell = '''# ============================================================================
# 📊 HELPER FUNCTIONS (Optional - for advanced users)
# ============================================================================

from notebooks.dashboard_utils import (
    save_config_to_file,
    load_config_from_file,
    validate_configuration,
    print_configuration_summary,
    apply_preset
)

# Example: Save current configuration
def save_config(filename='my_config.json'):
    """Save current dashboard configuration to file."""
    save_config_to_file(widgets_dict, filename)

# Example: Load configuration
def load_config(filename='my_config.json'):
    """Load configuration from file."""
    load_config_from_file(widgets_dict, filename)

# Example: Validate current configuration
def check_config():
    """Validate current configuration and show any errors."""
    valid, errors = validate_configuration(widgets_dict)
    if valid:
        print("✅ Configuration is valid")
    else:
        print("❌ Configuration has errors:")
        for error in errors:
            print(f"  - {error}")
    return valid

# Example: Print configuration summary
def show_config():
    """Print a summary of current configuration."""
    print_configuration_summary(widgets_dict)

# Example: Apply preset
def use_preset(preset_name='standard'):
    """Apply a preset configuration.
    
    Available presets:
    - 'quick_test': Fast configuration for testing
    - 'standard': Default research setup
    - 'high_capacity': Large-scale deployment
    - 'limited_budget': Extreme sensing constraint
    """
    apply_preset(widgets_dict, preset_name)

print("✅ Helper functions loaded!")
print("Available functions: save_config(), load_config(), check_config(), show_config(), use_preset()")
'''

def create_notebook():
    """Create the complete notebook JSON structure."""
    
    notebook = {
        "cells": [
            # Cell 0: Header markdown
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# 🎯 PhD Research Dashboard - ULTRA-CUSTOMIZABLE INTERACTIVE NOTEBOOK\\n",
                    "\\n",
                    "## RIS Probe-Based ML Research Interface\\n",
                    "\\n",
                    "This comprehensive dashboard provides **COMPLETE CONTROL** over all experimental parameters through an intuitive widget-based interface.\\n",
                    "\\n",
                    "### 🌟 Key Features:\\n",
                    "\\n",
                    "- **5 Organized Tabs**: System & Physics, Model Architecture, Training, Evaluation, Visualization\\n",
                    "- **8 Model Architectures**: MLP, CNN, LSTM, GRU, Attention, Transformer, ResNet, Hybrid\\n",
                    "- **7 Optimizers**: Adam, AdamW, SGD, RMSprop, AdaGrad, Adadelta, Adamax\\n",
                    "- **8 LR Schedulers**: Step, MultiStep, Exponential, Cosine, CosineWarmup, ReduceLROnPlateau, OneCycle\\n",
                    "- **6 Probe Methods**: continuous, binary, 2bit, hadamard, sobol, halton\\n",
                    "- **25+ Plot Types**: Training curves, CDF, PDF, Box, Violin, Heatmap, 3D Surface, ROC, etc.\\n",
                    "- **Multi-Format Export**: CSV, JSON, YAML, HDF5, Excel, PNG, PDF, SVG, EPS\\n",
                    "- **Dynamic Interactions**: Smart widget dependencies and auto-validation\\n",
                    "- **Multi-Model Comparison**: Run and compare multiple architectures\\n",
                    "- **Statistical Analysis**: Multi-seed runs with confidence intervals\\n",
                    "- **Reproducibility**: Save/load configs, track all parameters\\n",
                    "\\n",
                    "### 📋 Quick Start:\\n",
                    "\\n",
                    "1. **Run Cell 0** (below) - Auto-setup with package checking\\n",
                    "2. **Run Cell 1** - Load the dashboard interface\\n",
                    "3. **Configure** parameters in the widget tabs\\n",
                    "4. **Click** '🚀 RUN EXPERIMENT'\\n",
                    "5. **View** real-time progress and results\\n",
                    "\\n",
                    "### 📚 Documentation:\\n",
                    "\\n",
                    "- [User Guide](docs/USER_GUIDE.md) - Comprehensive technical documentation\\n",
                    "- [Developer Guide](docs/DEVELOPER_GUIDE.md) - How to extend the system\\n",
                    "- [Tutorial](docs/TUTORIAL.md) - Step-by-step examples\\n",
                    "- [API Reference](docs/API_REFERENCE.md) - Function documentation\\n",
                    "\\n",
                    "### 🎓 Ideal for:\\n",
                    "\\n",
                    "- PhD research exploration\\n",
                    "- Hyperparameter sweeps\\n",
                    "- Model architecture comparison\\n",
                    "- Publication-quality results\\n",
                    "\\n",
                    "---\\n",
                    "\\n",
                    "**Author**: Mohammad Bagher  \\n",
                    "**Repository**: [Probe_Based_ML_Session5](https://github.com/mohbagher/Probe_Based_ML_Session5)  \\n",
                    "**License**: MIT"
                ]
            },
            # Cell 0: Auto-setup
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": auto_setup_cell.split('\n')
            },
            # Cell 1: Main dashboard
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": main_cell.split('\n')
            },
            # Cell 2: Helper functions
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": helper_cell.split('\n')
            },
            # Cell 3: Examples markdown
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 📖 Quick Examples\\n",
                    "\\n",
                    "### Example 1: Basic Experiment\\n",
                    "```python\\n",
                    "# Already configured with defaults!\\n",
                    "# Just click 'RUN EXPERIMENT' in the control panel above\\n",
                    "```\\n",
                    "\\n",
                    "### Example 2: Apply a Preset\\n",
                    "```python\\n",
                    "use_preset('quick_test')  # Fast configuration\\n",
                    "# Then click 'RUN EXPERIMENT'\\n",
                    "```\\n",
                    "\\n",
                    "### Example 3: Check Your Configuration\\n",
                    "```python\\n",
                    "show_config()  # Print current settings\\n",
                    "check_config()  # Validate configuration\\n",
                    "```\\n",
                    "\\n",
                    "### Example 4: Save Configuration\\n",
                    "```python\\n",
                    "save_config('my_experiment.json')\\n",
                    "```\\n",
                    "\\n",
                    "### Example 5: Load Configuration\\n",
                    "```python\\n",
                    "load_config('my_experiment.json')\\n",
                    "```\\n",
                    "\\n",
                    "---\\n",
                    "\\n",
                    "## 💡 Tips\\n",
                    "\\n",
                    "- **For quick testing**: Use preset 'quick_test' (N=16, K=32, 20 epochs)\\n",
                    "- **For research**: Use preset 'standard' (N=32, K=64, 50 epochs)\\n",
                    "- **Compare models**: Enable 'Multi-Model Comparison Mode' in Multi-Experiment tab\\n",
                    "- **Get statistics**: Enable 'Multi-Seed Runs' in Multi-Experiment tab\\n",
                    "- **Save time**: Start with MLP, then try advanced models\\n",
                    "- **GPU training**: Dashboard auto-detects and uses GPU if available\\n",
                    "\\n",
                    "---\\n",
                    "\\n",
                    "## 🔧 Troubleshooting\\n",
                    "\\n",
                    "**Issue: Missing packages**\\n",
                    "→ Re-run Cell 0 (auto-setup)\\n",
                    "\\n",
                    "**Issue: Widgets not showing**\\n",
                    "→ Run: `jupyter nbextension enable --py widgetsnbextension`\\n",
                    "\\n",
                    "**Issue: Slow training**\\n",
                    "→ Check Device in setup output (GPU vs CPU)\\n",
                    "→ Reduce batch size or model size\\n",
                    "\\n",
                    "**Issue: Poor performance**\\n",
                    "→ Check M < K (sensing budget < codebook size)\\n",
                    "→ Try different model or probe method\\n",
                    "→ Increase training data or epochs"
                ]
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.8.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    return notebook

if __name__ == "__main__":
    # Create the notebook
    notebook = create_notebook()
    
    # Save to file
    output_path = Path(__file__).parent / "PhD_Research_Dashboard.ipynb"
    with open(output_path, 'w') as f:
        json.dump(notebook, f, indent=2)
    
    print(f"✅ Complete notebook created: {output_path}")
