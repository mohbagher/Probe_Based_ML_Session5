"""
Dashboard Runner for PhD Research Dashboard. 

Handles experiment execution, training, evaluation, and result saving.
"""

import sys
import os
from pathlib import Path
import torch
import torch.nn as nn
import torch. optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import json
import yaml
import pickle
from datetime import datetime
from tqdm. auto import tqdm

# Add project root to path
project_root = Path. cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()
if str(project_root) not in sys.path:
    sys. path.insert(0, str(project_root))

from config import Config, SystemConfig, DataConfig, TrainingConfig, ModelConfig
from data_generation import create_dataloaders
from model import LimitedProbingMLP
from evaluation import evaluate_model
from experiments.probe_generators import get_probe_bank

# Try to import advanced models and plot registry
try:
    from advanced_models import create_advanced_model
    HAS_ADVANCED_MODELS = True
except ImportError: 
    HAS_ADVANCED_MODELS = False
    print("⚠️  advanced_models. py not found - only MLP available")

try:
    import plot_registry
    HAS_PLOT_REGISTRY = True
except ImportError:
    HAS_PLOT_REGISTRY = False
    print("⚠️  plot_registry.py not found - plotting disabled")


def extract_config_from_widgets(widgets_dict):
    """
    Extract configuration from widget values.
    
    Args:
        widgets_dict: Dictionary of all widgets
        
    Returns: 
        Config object
    """
    # System config
    system = SystemConfig(
        N=widgets_dict['system']['N'].value,
        K=widgets_dict['system']['K'].value,
        M=widgets_dict['system']['M']. value,
        P_tx=widgets_dict['system']['P_tx'].value,
        sigma_h_sq=widgets_dict['system']['sigma_h_sq'].value,
        sigma_g_sq=widgets_dict['system']['sigma_g_sq'].value,
        probe_type=widgets_dict['system']['probe_type'].value,
        phase_mode=widgets_dict['system']['phase_mode'].value,
        phase_bits=widgets_dict['system']['phase_bits'].value,
    )
    
    # Data config
    data = DataConfig(
        n_train=widgets_dict['data']['n_train'].value,
        n_val=widgets_dict['data']['n_val'].value,
        n_test=widgets_dict['data']['n_test'].value,
        seed=widgets_dict['data']['seed'].value,
        normalize_input=widgets_dict['data']['normalize_input'].value,
        normalization_type=widgets_dict['data']['normalization_type'].value,
    )
    
    # Parse hidden sizes for MLP
    hidden_sizes_str = widgets_dict['model']['hidden_sizes'].value
    try:
        hidden_sizes = [int(x. strip()) for x in hidden_sizes_str.split(',')]
    except: 
        hidden_sizes = [512, 256, 128]  # Default
    
    # Model config
    model = ModelConfig(
        hidden_sizes=hidden_sizes,
        dropout_prob=widgets_dict['model']['dropout_prob'].value,
        use_batch_norm=widgets_dict['model']['use_batch_norm'].value,
    )
    
    # Training config - FIXED parameter names
    training = TrainingConfig(
        n_epochs=widgets_dict['training']['epochs'].value,
        batch_size=widgets_dict['training']['batch_size'].value,
        learning_rate=widgets_dict['training']['learning_rate'].value,
        weight_decay=widgets_dict['training']['weight_decay'].value,
        early_stop_patience=widgets_dict['training']['early_stopping_patience'].value,
    )
    
    config = Config(
        system=system,
        data=data,
        model=model,
        training=training,
    )
    
    return config


def create_optimizer(model, config_widgets):
    """Create optimizer based on widget selection."""
    optimizer_name = config_widgets['optimizer'].value
    lr = config_widgets['learning_rate'].value
    weight_decay = config_widgets['weight_decay']. value
    
    if optimizer_name == 'Adam':
        return optim.Adam(model. parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'AdamW':
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'SGD':
        momentum = config_widgets['momentum'].value if 'momentum' in config_widgets else 0.9
        return optim.SGD(model. parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif optimizer_name == 'RMSprop':
        return optim.RMSprop(model. parameters(), lr=lr, weight_decay=weight_decay)
    else:
        return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)


def create_scheduler(optimizer, config_widgets, steps_per_epoch=None):
    """Create learning rate scheduler based on widget selection."""
    scheduler_name = config_widgets['scheduler'].value if 'scheduler' in config_widgets else 'None'
    
    if scheduler_name == 'None' or scheduler_name is None:
        return None
    elif scheduler_name == 'StepLR':
        step_size = config_widgets['scheduler_step_size'].value if 'scheduler_step_size' in config_widgets else 10
        gamma = config_widgets['scheduler_gamma'].value if 'scheduler_gamma' in config_widgets else 0.1
        return optim. lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif scheduler_name == 'ReduceLROnPlateau': 
        patience = config_widgets['scheduler_patience'].value if 'scheduler_patience' in config_widgets else 5
        gamma = config_widgets['scheduler_gamma'].value if 'scheduler_gamma' in config_widgets else 0.5
        return optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                     patience=patience, factor=gamma)
    return None


def create_loss_function(config_widgets, num_classes):
    """Create loss function based on widget selection."""
    loss_name = config_widgets['loss_function'].value if 'loss_function' in config_widgets else 'CrossEntropy'
    
    if loss_name == 'CrossEntropy':
        return nn.CrossEntropyLoss()
    elif loss_name == 'LabelSmoothing':
        smoothing = config_widgets['label_smoothing'].value if 'label_smoothing' in config_widgets else 0.1
        return nn.CrossEntropyLoss(label_smoothing=smoothing)
    else:
        return nn. CrossEntropyLoss()


def run_single_experiment(widgets_dict, model_type=None, seed=None):
    """
    Run a single experiment with given configuration.
    
    Args:
        widgets_dict: Dictionary of all widgets
        model_type: Override model type (for comparison mode)
        seed: Override seed (for multi-seed mode)
        
    Returns: 
        Dictionary with results, history, model, and config
    """
    # Update status
    status_msg = f'Running experiment'
    if model_type:
        status_msg += f' with {model_type}'
    if seed:
        status_msg += f' (seed={seed})'
    
    if 'control' in widgets_dict and 'progress_text' in widgets_dict['control']: 
        widgets_dict['control']['progress_text'].value = f'<p>{status_msg}...</p>'
    
    # Extract configuration
    config = extract_config_from_widgets(widgets_dict)
    if seed:
        config.data.seed = seed
    
    # Get model type
    if model_type is None:
        model_type = widgets_dict['model']['model_type'].value
    
    # Generate probe bank
    print(f"\n{'='*70}")
    print(f"Generating probe bank ({config.system.probe_type})...")
    probe_bank = get_probe_bank(
        config.system.probe_type,
        config.system.N,
        config.system.K,
        config.data.seed
    )
    
    # Generate dataset - FIXED:  correct unpacking (4 values not 3)
    print(f"Generating dataset (N={config.system.N}, K={config.system.K}, M={config.system.M})...")
    train_loader, val_loader, test_loader, metadata = create_dataloaders(config, probe_bank)
    
    # Create model
    print(f"Creating {model_type} model...")
    if model_type == 'MLP':
        model = LimitedProbingMLP(
            K=config.system.K,
            hidden_sizes=config.model.hidden_sizes,
            dropout_prob=config.model.dropout_prob,
            use_batch_norm=config.model.use_batch_norm
        )
    else:
        if not HAS_ADVANCED_MODELS:
            print(f"⚠️  Advanced models not available, falling back to MLP")
            model = LimitedProbingMLP(
                K=config. system.K,
                hidden_sizes=config.model.hidden_sizes,
                dropout_prob=config.model.dropout_prob,
                use_batch_norm=config.model.use_batch_norm
            )
        else:
            model_config = {'dropout_prob': config.model.dropout_prob}
            model = create_advanced_model(model_type. lower(), config.system.K, model_config)
    
    # Move to device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create optimizer, scheduler, loss
    optimizer = create_optimizer(model, widgets_dict['training'])
    scheduler = create_scheduler(optimizer, widgets_dict['training'], len(train_loader))
    criterion = create_loss_function(widgets_dict['training'], config.system.K)
    
    # Training loop
    print(f"\nTraining for {config.training.n_epochs} epochs...")
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': [],
        'lr': []
    }
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(config.training.n_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config. training.n_epochs}')
        for batch_x, batch_y in pbar: 
            batch_x, batch_y = batch_x. to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += batch_y.size(0)
            train_correct += predicted.eq(batch_y).sum().item()
            
            pbar.set_postfix({'loss': loss.item(), 'acc': 100. * train_correct / train_total})
        
        train_loss /= len(train_loader)
        train_acc = 100. * train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                
                val_loss += loss.item()
                _, predicted = outputs. max(1)
                val_total += batch_y.size(0)
                val_correct += predicted.eq(batch_y).sum().item()
        
        val_loss /= len(val_loader)
        val_acc = 100. * val_correct / val_total
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss']. append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['lr']. append(optimizer.param_groups[0]['lr'])
        
        print(f'Epoch {epoch+1}:  Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, '
              f'Train Acc={train_acc:.2f}%, Val Acc={val_acc:.2f}%')
        
        # Learning rate scheduling
        if scheduler is not None:
            if isinstance(scheduler, optim.lr_scheduler. ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.training. early_stop_patience:
                print(f'Early stopping at epoch {epoch+1}')
                break
    
    # Evaluation - FIXED: correct parameters
    print(f"\nEvaluating on test set...")
    results = evaluate_model(
        model=model,
        test_loader=test_loader,
        config=config,
        powers_full=metadata['test_powers_full'],
        labels=metadata['test_labels'],
        observed_indices=metadata['test_observed_indices'],
        optimal_powers=metadata['test_optimal_powers']
    )
    results. print_summary()
    
    return {
        'results': results,
        'history': history,
        'model': model,
        'config': config,
        'probe_bank': probe_bank,
        'model_type': model_type,
        'metadata': metadata,
    }


def run_experiments(widgets_dict):
    """
    Main experiment runner - handles single/multi-model/multi-seed experiments. 
    
    Args:
        widgets_dict: Dictionary of all widgets
    """
    try:
        # Check if multi-experiment widgets exist and get values safely
        comparison_mode = False
        multi_seed = False
        
        if 'multi_experiment' in widgets_dict:
            if 'comparison_mode' in widgets_dict['multi_experiment']:
                comparison_mode = widgets_dict['multi_experiment']['comparison_mode']. value
            if 'multi_seed' in widgets_dict['multi_experiment']:
                multi_seed = widgets_dict['multi_experiment']['multi_seed'].value
        
        all_results = {}
        
        if comparison_mode:
            # Multi-model comparison
            models = widgets_dict['multi_experiment']['models_to_compare'].value
            for model_type in models:
                print(f"\n{'='*70}")
                print(f"Running experiment with {model_type}")
                print(f"{'='*70}")
                result = run_single_experiment(widgets_dict, model_type=model_type)
                all_results[model_type] = result
        elif multi_seed:
            # Multi-seed runs
            num_seeds = widgets_dict['multi_experiment']['num_seeds'].value
            seed_start = widgets_dict['multi_experiment']['seed_start'].value
            model_type = widgets_dict['model']['model_type'].value
            
            for i in range(num_seeds):
                seed = seed_start + i
                print(f"\n{'='*70}")
                print(f"Running experiment with seed {seed} ({i+1}/{num_seeds})")
                print(f"{'='*70}")
                result = run_single_experiment(widgets_dict, seed=seed)
                all_results[f'{model_type}_seed{seed}'] = result
        else: 
            # Single experiment
            result = run_single_experiment(widgets_dict)
            model_type = widgets_dict['model']['model_type'].value
            all_results[model_type] = result
        
        # Save results if requested
        if 'evaluation' in widgets_dict and 'save_results' in widgets_dict['evaluation']:
            if widgets_dict['evaluation']['save_results'].value:
                save_all_results(all_results, widgets_dict)
        
        # Generate plots if requested
        if HAS_PLOT_REGISTRY and 'evaluation' in widgets_dict and 'plot_types' in widgets_dict['evaluation']: 
            plot_types = widgets_dict['evaluation']['plot_types']. value
            if plot_types:
                generate_plots(all_results, widgets_dict)
        
        print(f"\n{'='*70}")
        print("✅ ALL EXPERIMENTS COMPLETED SUCCESSFULLY!")
        print(f"{'='*70}")
        
        return all_results
        
    except Exception as e:
        print(f"\n{'='*70}")
        print(f"❌ Error:  {str(e)}")
        print(f"{'='*70}")
        import traceback
        traceback.print_exc()
        raise


def save_all_results(all_results, widgets_dict):
    """Save all results in various formats."""
    output_dir = Path(widgets_dict['evaluation']['output_dir'].value if 'output_dir' in widgets_dict['evaluation'] else 'results')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_dir = output_dir / f'experiment_{timestamp}'
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSaving results to {exp_dir}...")
    
    export_formats = widgets_dict['evaluation']['export_format'].value if 'export_format' in widgets_dict['evaluation'] else ['JSON']
    
    for name, result in all_results.items():
        results_dict = result['results']. to_dict()
        
        if 'JSON' in export_formats:
            with open(exp_dir / f'{name}_results.json', 'w') as f:
                json.dump(results_dict, f, indent=2)
        
        if 'Pickle' in export_formats:
            with open(exp_dir / f'{name}_results.pkl', 'wb') as f:
                pickle.dump(result, f)
        
        # Save model
        if 'save_model' in widgets_dict['evaluation']:
            if widgets_dict['evaluation']['save_model'].value:
                torch.save(result['model'].state_dict(), exp_dir / f'{name}_model.pth')
    
    print(f"✅ Results saved to {exp_dir}")


def generate_plots(all_results, widgets_dict):
    """Generate selected plots."""
    if not HAS_PLOT_REGISTRY:
        print("⚠️  Plot registry not available, skipping plots")
        return
    
    print(f"\n⚠️  Plot generation not yet fully implemented")
