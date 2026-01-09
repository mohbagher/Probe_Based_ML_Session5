"""
Test all imports to catch errors before running dashboard.
Run this before using the dashboard to ensure everything works.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

print("="*70)
print("🧪 TESTING ALL IMPORTS")
print("="*70)

errors = []

# Test 1: Core config
print("\n1️⃣ Testing config.py...")
try:
    from config import Config, SystemConfig, DataConfig, ModelConfig, TrainingConfig, EvalConfig
    print("   ✅ Config classes imported")
    try:
        from config import get_config
        print("   ✅ get_config imported")
    except ImportError:
        print("   ⚠️  get_config not found (will use manual config)")
except Exception as e:
    errors.append(f"config.py: {e}")
    print(f"   ❌ ERROR: {e}")

# Test 2: Data generation
print("\n2️⃣ Testing data_generation.py...")
try:
    from data_generation import create_dataloaders
    print("   ✅ create_dataloaders imported")
    try:
        from data_generation import generate_channel_realization, compute_probe_powers
        print("   ✅ generate_channel_realization and compute_probe_powers imported")
    except ImportError as e:
        print(f"   ⚠️  Some data_generation functions not found: {e}")
except Exception as e:
    errors.append(f"data_generation.py: {e}")
    print(f"   ❌ ERROR: {e}")

# Test 3: Probe generators
print("\n3️⃣ Testing experiments/probe_generators.py...")
try:
    from experiments.probe_generators import get_probe_bank, ProbeBank
    print("   ✅ Probe generators imported")
except Exception as e:
    errors.append(f"probe_generators.py: {e}")
    print(f"   ❌ ERROR: {e}")

# Test 4: Models
print("\n4️⃣ Testing model.py...")
try:
    from model import LimitedProbingMLP, create_model, count_parameters
    print("   ✅ Model classes imported")
except Exception as e:
    errors.append(f"model.py: {e}")
    print(f"   ❌ ERROR: {e}")

# Test 5: Training
print("\n5️⃣ Testing training.py...")
try:
    from training import train, TrainingHistory, EarlyStopping
    print("   ✅ Training functions imported")
except Exception as e:
    errors.append(f"training.py: {e}")
    print(f"   ❌ ERROR: {e}")

# Test 6: Evaluation
print("\n6️⃣ Testing evaluation.py...")
try:
    from evaluation import evaluate_model, EvaluationResults
    print("   ✅ Evaluation functions imported")
except Exception as e:
    errors.append(f"evaluation.py: {e}")
    print(f"   ❌ ERROR: {e}")

# Test 7: Advanced models (optional)
print("\n7️⃣ Testing advanced_models.py...")
try:
    from advanced_models import create_advanced_model
    print("   ✅ create_advanced_model imported")
except Exception as e:
    print(f"   ⚠️  advanced_models.py: {e}")

# Test 8: Plot registry (optional)
print("\n8️⃣ Testing plot_registry.py...")
try:
    from plot_registry import PLOT_REGISTRY, get_plot_function
    print("   ✅ PLOT_REGISTRY and get_plot_function imported")
except Exception as e:
    print(f"   ⚠️  plot_registry.py: {e}")

# Test 9: Model registry (optional)
print("\n9️⃣ Testing model_registry.py...")
try:
    from model_registry import MODEL_REGISTRY
    print("   ✅ MODEL_REGISTRY imported")
except Exception as e:
    print(f"   ⚠️  model_registry.py: {e}")

# Test 10: Extended channel models (optional - expected to fail)
print("\n🔟 Testing extended_channel_models.py...")
try:
    from extended_channel_models import get_channel_generator
    print("   ✅ get_channel_generator imported")
except Exception as e:
    print(f"   ⚠️  extended_channel_models.py not found (expected)")

# Test 11: Dashboard modules (if they exist)
print("\n1️⃣1️⃣ Testing dashboard modules...")
dashboard_files = [
    'notebooks.dashboard_widgets',
    'notebooks.dashboard_callbacks',
    'notebooks.dashboard_runner',
    'notebooks.dashboard_utils',
]
for module_name in dashboard_files:
    file_path = module_name.replace('.', '/') + '.py'
    if (project_root / file_path).exists():
        try:
            __import__(module_name)
            print(f"   ✅ {module_name} imported")
        except Exception as e:
            errors.append(f"{module_name}: {e}")
            print(f"   ❌ {module_name}: {e}")
    else:
        print(f"   ⚠️  {file_path} not found")

# Summary
print("\n" + "="*70)
if errors:
    print(f"❌ FAILED: {len(errors)} import errors found")
    print("="*70)
    for error in errors:
        print(f"  - {error}")
    sys.exit(1)
else:
    print("✅ SUCCESS: All critical imports working correctly!")
    print("="*70)
    sys.exit(0)
