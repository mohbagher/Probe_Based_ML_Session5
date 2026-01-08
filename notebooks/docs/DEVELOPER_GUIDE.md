# 🛠️ Developer Guide - Extending the Dashboard

## Overview

This guide explains how to extend the PhD Research Dashboard with new features.

---

## Adding New Widgets

### Step 1: Define Widget in `dashboard_widgets.py`

```python
def create_new_widget_category():
    """Create widgets for new feature"""
    return {
        'new_param1': IntSlider(
            min=0, max=100, value=50,
            description='New Param 1',
            style={'description_width': 'initial'}
        ),
    }
```

### Step 2: Add to main widget dictionary

In your notebook:
```python
widgets_dict['new_category'] = create_new_widget_category()
```

### Step 3: Add callback in `dashboard_callbacks.py`

```python
def on_new_param_change(change):
    new_value = change['new']
    # React to changes
    
widgets_dict['new_category']['new_param1'].observe(
    on_new_param_change, names='value'
)
```

---

## Adding New ML Models

### Step 1: Implement model in `advanced_models.py`

```python
class MyNewModel(nn.Module):
    def __init__(self, K, custom_param=64):
        super().__init__()
        self.K = K
        # ... implement architecture
        
    def forward(self, x):
        # ... implement forward pass
        return logits
```

### Step 2: Add to model factory

```python
def create_advanced_model(model_type, K, config):
    # ... existing cases ...
    elif model_type == "mynewmodel":
        return MyNewModel(
            K=K,
            custom_param=config.get('custom_param', 64)
        )
```

### Step 3: Add widgets for model-specific params

In `dashboard_widgets.py`:
```python
'mynewmodel_custom': IntSlider(min=16, max=256, value=64),
```

### Step 4: Update model type dropdown

```python
model_type = Dropdown(
    options=[... , 'MyNewModel'],
    description='Model Type'
)
```

---

## Adding New Probe Methods

### Step 1: Implement in `experiments/probe_generators.py`

```python
def generate_probe_bank_mynewmethod(N, K, seed=None):
    """Generate probe bank using my new method."""
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.RandomState()
    
    # Implement your probe generation logic
    phases = ... # Shape: (K, N)
    
    return ProbeBank(
        phases=phases,
        K=K, N=N,
        probe_type="mynewmethod"
    )
```

### Step 2: Register in factory function

```python
def get_probe_bank(probe_type, N, K, seed=None):
    # ... existing cases ...
    elif probe_type == "mynewmethod":
        return generate_probe_bank_mynewmethod(N, K, seed)
```

### Step 3: Add to probe method dropdown

```python
probe_methods = Dropdown(
    options=[... , 'mynewmethod']
)
```

---

## Adding New Plot Types

### Step 1: Implement plot function in `plot_registry.py`

```python
def plot_mynewplot(results, config, save_path=None):
    """Generate my new plot type."""
    import matplotlib.pyplot as plt
    
    # Extract data
    data = results.eta_top1_distribution
    
    # Create plot
    plt.figure(figsize=(10, 6))
    # ... plotting code ...
    plt.title('My New Plot')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
```

### Step 2: Register in plot registry

```python
PLOT_REGISTRY = {
    # ... existing ...
    'mynewplot': plot_mynewplot,
}
```

### Step 3: Add to plot selection widget

```python
plot_options = [
    # ... existing ...
    'My New Plot',
]
```

---

## Adding New Optimizers

In `dashboard_runner.py`, extend `create_optimizer` function:

```python
def create_optimizer(model, config_widgets):
    optimizer_name = config_widgets['optimizer'].value
    lr = config_widgets['learning_rate'].value
    
    # ... existing cases ...
    elif optimizer_name == 'MyOptimizer':
        return optim.MyOptimizer(model.parameters(), lr=lr)
```

---

## Adding New Learning Rate Schedulers

In `dashboard_runner.py`, extend `create_scheduler` function:

```python
def create_scheduler(optimizer, config_widgets, steps_per_epoch=None):
    scheduler_name = config_widgets['scheduler'].value
    
    # ... existing cases ...
    elif scheduler_name == 'MyScheduler':
        return optim.lr_scheduler.MyScheduler(optimizer, ...)
```

---

## Adding New Loss Functions

In `dashboard_runner.py`, extend `create_loss_function`:

```python
def create_loss_function(config_widgets, num_classes):
    loss_name = config_widgets['loss_function'].value
    
    # ... existing cases ...
    elif loss_name == 'MyLoss':
        return MyCustomLoss()
```

---

## Adding New Channel Models

### Step 1: Implement in `data_generation.py`

```python
def generate_channel_mymodel(N, sigma_sq, rng):
    """Generate channel using my model."""
    # Implement channel generation
    return channel  # Shape: (N,) complex
```

### Step 2: Add to channel generation function

```python
def generate_channels(config, rng):
    # ... existing code ...
    elif config.system.channel_model == 'mymodel':
        h = generate_channel_mymodel(N, sigma_h_sq, rng)
        g = generate_channel_mymodel(N, sigma_g_sq, rng)
```

---

## Adding New Evaluation Metrics

### Step 1: Add to `evaluation.py`

In `EvaluationResults` dataclass:
```python
@dataclass
class EvaluationResults:
    # ... existing fields ...
    my_new_metric: float
```

### Step 2: Compute in `evaluate_model` function

```python
def evaluate_model(model, test_loader, config, device):
    # ... existing evaluation ...
    
    # Compute new metric
    my_new_metric = compute_my_metric(predictions, targets)
    
    return EvaluationResults(
        # ... existing metrics ...
        my_new_metric=my_new_metric
    )
```

---

## Adding Presets

In `dashboard_utils.py`, extend `create_preset_configs`:

```python
def create_preset_configs():
    presets = {
        # ... existing presets ...
        'my_preset': {
            'name': 'My Preset',
            'description': 'Custom configuration',
            'system': {'N': 32, 'K': 64, 'M': 8},
            'data': {'n_train': 50000},
            'training': {'epochs': 50},
        },
    }
    return presets
```

---

## Code Style Guidelines

1. **Follow existing patterns:** Look at similar code for reference
2. **Add docstrings:** Document all public functions
3. **Type hints:** Use type annotations where helpful
4. **Error handling:** Add try-except for user-facing code
5. **Validation:** Validate inputs before processing
6. **Logging:** Use print statements for progress updates

---

## Testing New Features

### Manual Testing Checklist

- [ ] Widget displays correctly
- [ ] Callbacks work as expected
- [ ] Model trains successfully
- [ ] Plots generate without errors
- [ ] Results save correctly
- [ ] Configuration can be saved/loaded
- [ ] Works with comparison mode
- [ ] Works with multi-seed mode

### Example Test Script

```python
# Test new model
widgets_dict['model']['model_type'].value = 'MyNewModel'
result = run_single_experiment(widgets_dict)
print(f"Test passed: η = {result['results'].eta_top1:.4f}")
```

---

## Common Issues and Solutions

### Issue: Import errors
**Solution:** Check `sys.path` includes project root

### Issue: Widget not responding
**Solution:** Check callback is registered with `.observe()`

### Issue: Model shape mismatch
**Solution:** Verify K dimension matches everywhere

### Issue: Plot not showing
**Solution:** Call `plt.show()` or save with `save_path`

---

## Contributing

When contributing new features:

1. Test thoroughly with different configurations
2. Update documentation (USER_GUIDE.md, API_REFERENCE.md)
3. Add examples to TUTORIAL.md
4. Follow code style guidelines
5. Add comments for complex logic

---

## Resources

- **PyTorch Documentation:** https://pytorch.org/docs/
- **ipywidgets Documentation:** https://ipywidgets.readthedocs.io/
- **Matplotlib Gallery:** https://matplotlib.org/stable/gallery/
- **Project Repository:** https://github.com/mohbagher/Probe_Based_ML_Session5

---

For usage instructions, see [USER_GUIDE.md](USER_GUIDE.md)
For step-by-step examples, see [TUTORIAL.md](TUTORIAL.md)
For function reference, see [API_REFERENCE.md](API_REFERENCE.md)
