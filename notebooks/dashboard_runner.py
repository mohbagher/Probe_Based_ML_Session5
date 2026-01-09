"""
Dashboard Runner for PhD Research Dashboard.

Handles experiment execution, training, evaluation, and result saving.
"""

import sys
import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import json
import yaml
import pickle
from datetime import datetime
from tqdm.auto import tqdm

# Add project root to path
project_root = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from config import Config, SystemConfig, DataConfig, TrainingConfig, ModelConfig
from data_generation import create_dataloaders
from advanced_models import create_advanced_model
from model import LimitedProbingMLP
from evaluation import evaluate_model
from experiments.probe_generators import get_probe_bank
import plot_registry


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
        M=widgets_dict['system']['M'].value,
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
        hidden_sizes = [int(x.strip()) for x in hidden_sizes_str.split(',')]
    except:
        hidden_sizes = [512, 256, 128]  # Default
    
    # Model config
    model = ModelConfig(
        hidden_sizes=hidden_sizes,
        dropout_prob=widgets_dict['model']['dropout_prob'].value,
        use_batch_norm=widgets_dict['model']['use_batch_norm'].value,
    )
    
    # Training config
    training = TrainingConfig(
        epochs=widgets_dict['training']['epochs'].value,
        batch_size=widgets_dict['training']['batch_size'].value,
        learning_rate=widgets_dict['training']['learning_rate'].value,
        weight_decay=widgets_dict['training']['weight_decay'].value,
        early_stopping=widgets_dict['training']['early_stopping'].value,
        patience=widgets_dict['training']['early_stopping_patience'].value,
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
    weight_decay = config_widgets['weight_decay'].value
    
    if optimizer_name == 'Adam':
        return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'AdamW':
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'SGD':
        momentum = config_widgets['momentum'].value
        return optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif optimizer_name == 'RMSprop':
        return optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'AdaGrad':
        return optim.Adagrad(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'Adadelta':
        return optim.Adadelta(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'Adamax':
        return optim.Adamax(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)


def create_scheduler(optimizer, config_widgets, steps_per_epoch=None):
    """Create learning rate scheduler based on widget selection."""
    scheduler_name = config_widgets['scheduler'].value
    
    if scheduler_name == 'None':
        return None
    elif scheduler_name == 'StepLR':
        step_size = config_widgets['scheduler_step_size'].value
        gamma = config_widgets['scheduler_gamma'].value
        return optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif scheduler_name == 'MultiStepLR':
        step_size = config_widgets['scheduler_step_size'].value
        gamma = config_widgets['scheduler_gamma'].value
        milestones = [step_size * i for i in range(1, 10)]
        return optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
    elif scheduler_name == 'ExponentialLR':
        gamma = config_widgets['scheduler_gamma'].value
        return optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    elif scheduler_name == 'CosineAnnealingLR':
        epochs = config_widgets['epochs'].value
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    elif scheduler_name == 'CosineAnnealingWarmRestarts':
        step_size = config_widgets['scheduler_step_size'].value
        return optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=step_size)
    elif scheduler_name == 'ReduceLROnPlateau':
        patience = config_widgets['scheduler_patience'].value
        gamma = config_widgets['scheduler_gamma'].value
        return optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                     patience=patience, factor=gamma)
    elif scheduler_name == 'OneCycleLR':
        if steps_per_epoch is None:
            steps_per_epoch = 100  # Default
        epochs = config_widgets['epochs'].value
        max_lr = config_widgets['learning_rate'].value * 10
        return optim.lr_scheduler.OneCycleLR(optimizer, max_lr=max_lr, 
                                              steps_per_epoch=steps_per_epoch, 
                                              epochs=epochs)
    return None


def create_loss_function(config_widgets, num_classes):
    """Create loss function based on widget selection."""
    loss_name = config_widgets['loss_function'].value
    
    if loss_name == 'CrossEntropy':
        return nn.CrossEntropyLoss()
    elif loss_name == 'LabelSmoothing':
        smoothing = config_widgets['label_smoothing'].value
        return nn.CrossEntropyLoss(label_smoothing=smoothing)
    elif loss_name == 'FocalLoss':
        # Implement focal loss
        class FocalLoss(nn.Module):
            def __init__(self, alpha=1.0, gamma=2.0):
                super().__init__()
                self.alpha = alpha
                self.gamma = gamma
                self.ce = nn.CrossEntropyLoss(reduction='none')
            
            def forward(self, inputs, targets):
                ce_loss = self.ce(inputs, targets)
                pt = torch.exp(-ce_loss)
                focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
                return focal_loss.mean()
        
        return FocalLoss()
    else:
        return nn.CrossEntropyLoss()


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
    
    # Generate dataset
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
        # Use advanced models
        model_config = {
            'dropout_prob': config.model.dropout_prob,
        }
        
        if model_type == 'CNN':
            cnn_filters_str = widgets_dict['model']['cnn_filters'].value
            cnn_kernels_str = widgets_dict['model']['cnn_kernels'].value
            model_config['num_filters'] = [int(x.strip()) for x in cnn_filters_str.split(',')]
            model_config['kernel_sizes'] = [int(x.strip()) for x in cnn_kernels_str.split(',')]
        elif model_type in ['LSTM', 'GRU']:
            model_config['hidden_size'] = widgets_dict['model']['rnn_hidden_size'].value
            model_config['num_layers'] = widgets_dict['model']['rnn_num_layers'].value
        elif model_type == 'Transformer':
            model_config['d_model'] = widgets_dict['model']['transformer_d_model'].value
            model_config['num_heads'] = widgets_dict['model']['transformer_num_heads'].value
            model_config['num_layers'] = widgets_dict['model']['transformer_num_layers'].value
            model_config['dim_feedforward'] = widgets_dict['model']['transformer_dim_feedforward'].value
        elif model_type == 'ResNet':
            model_config['hidden_size'] = widgets_dict['model']['resnet_hidden_size'].value
            model_config['num_blocks'] = widgets_dict['model']['resnet_num_blocks'].value
        elif model_type == 'Hybrid':
            model_config['lstm_hidden'] = widgets_dict['model']['rnn_hidden_size'].value
            model_config['lstm_layers'] = widgets_dict['model']['rnn_num_layers'].value
        elif model_type == 'Attention':
            model_config['hidden_sizes'] = config.model.hidden_sizes
        
        model = create_advanced_model(model_type.lower(), config.system.K, model_config)
    
    # Move to device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create optimizer, scheduler, loss
    optimizer = create_optimizer(model, widgets_dict['training'])
    scheduler = create_scheduler(optimizer, widgets_dict['training'], len(train_loader))
    criterion = create_loss_function(widgets_dict['training'], config.system.K)
    
    # Training loop
    print(f"\nTraining for {config.training.epochs} epochs...")
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': [],
        'lr': []
    }
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(config.training.epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config.training.epochs}')
        for batch_x, batch_y in pbar:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
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
                _, predicted = outputs.max(1)
                val_total += batch_y.size(0)
                val_correct += predicted.eq(batch_y).sum().item()
        
        val_loss /= len(val_loader)
        val_acc = 100. * val_correct / val_total
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['lr'].append(optimizer.param_groups[0]['lr'])
        
        print(f'Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, '
              f'Train Acc={train_acc:.2f}%, Val Acc={val_acc:.2f}%')
        
        # Learning rate scheduling
        if scheduler is not None:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()
        
        # Early stopping
        if config.training.early_stopping:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= config.training.patience:
                    print(f'Early stopping at epoch {epoch+1}')
                    break
    
    # Evaluation
    print(f"\nEvaluating on test set...")
    
    # Validate metadata has required keys
    required_keys = ['test_powers_full', 'test_labels', 'test_observed_indices', 'test_optimal_powers']
    missing_keys = [key for key in required_keys if key not in metadata]
    if missing_keys:
        raise KeyError(f"Metadata missing required keys: {missing_keys}")
    
    results = evaluate_model(
        model, test_loader, config,
        metadata['test_powers_full'],
        metadata['test_labels'],
        metadata['test_observed_indices'],
        metadata['test_optimal_powers']
    )
    results.print_summary()
    
    return {
        'results': results,
        'history': history,
        'model': model,
        'config': config,
        'probe_bank': probe_bank,
        'model_type': model_type,
    }


def run_experiments(widgets_dict):
    """
    Main experiment runner - handles single/multi-model/multi-seed experiments.
    
    Args:
        widgets_dict: Dictionary of all widgets
    """
    comparison_mode = widgets_dict['multi_experiment']['comparison_mode'].value
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
    
    # Save results and generate plots
    if widgets_dict['evaluation']['save_results'].value:
        save_all_results(all_results, widgets_dict)
    
    # Generate plots
    generate_plots(all_results, widgets_dict)
    
    return all_results


def save_all_results(all_results, widgets_dict):
    """Save all results in various formats."""
    output_dir = Path(widgets_dict['evaluation']['output_dir'].value)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_dir = output_dir / f'experiment_{timestamp}'
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSaving results to {exp_dir}...")
    
    export_formats = widgets_dict['evaluation']['export_format'].value
    
    for name, result in all_results.items():
        # Save results dict
        results_dict = result['results'].to_dict()
        
        if 'CSV' in export_formats:
            import pandas as pd
            df = pd.DataFrame([results_dict])
            df.to_csv(exp_dir / f'{name}_results.csv', index=False)
        
        if 'JSON' in export_formats:
            with open(exp_dir / f'{name}_results.json', 'w') as f:
                json.dump(results_dict, f, indent=2)
        
        if 'YAML' in export_formats:
            with open(exp_dir / f'{name}_results.yaml', 'w') as f:
                yaml.dump(results_dict, f)
        
        if 'Pickle' in export_formats:
            with open(exp_dir / f'{name}_results.pkl', 'wb') as f:
                pickle.dump(result, f)
        
        if 'Excel' in export_formats:
            import pandas as pd
            df = pd.DataFrame([results_dict])
            df.to_excel(exp_dir / f'{name}_results.xlsx', index=False)
        
        # Save model
        if widgets_dict['evaluation']['save_model'].value:
            torch.save(result['model'].state_dict(), exp_dir / f'{name}_model.pth')
    
    print(f"✅ Results saved to {exp_dir}")


def generate_plots(all_results, widgets_dict):
    """Generate selected plots."""
    selected_plots = widgets_dict['evaluation']['plot_types'].value
    output_dir = Path(widgets_dict['evaluation']['output_dir'].value)
    export_formats = widgets_dict['evaluation']['export_format'].value
    
    # Determine save extension
    save_ext = None
    for fmt in ['PNG', 'PDF', 'SVG', 'EPS']:
        if fmt in export_formats:
            save_ext = fmt.lower()
            break
    
    print(f"\nGenerating {len(selected_plots)} plots...")
    
    # Map plot names to registry keys
    plot_map = {
        'Training Curves': 'training_curves',
        'Learning Rate Schedule': 'learning_rate_schedule',
        'Gradient Flow': 'gradient_flow',
        'Eta Distribution': 'eta_distribution',
        'CDF': 'cdf',
        'PDF Histogram': 'pdf_histogram',
        'Box Plot': 'box',
        'Violin Plot': 'violin',
        'Scatter Comparison': 'scatter',
        'Bar Comparison': 'bar_comparison',
        'Radar Chart': 'radar_chart',
        'Heatmap Comparison': 'heatmap_comparison',
        'Probe Heatmap': 'heatmap',
        'Correlation Matrix': 'correlation_matrix',
        'Diversity Analysis': 'diversity_analysis',
        'Probe Power Distribution': 'probe_power_distribution',
        '3D Surface': '3d_surface',
        'ROC Curves': 'roc_curves',
        'Precision-Recall': 'precision_recall',
        'Confusion Matrix': 'confusion_matrix',
        'Top-M Comparison': 'top_m_comparison',
        'Baseline Comparison': 'baseline_comparison',
        'Convergence Analysis': 'convergence_analysis',
        'Parameter Sensitivity': 'parameter_sensitivity',
        'Model Complexity vs Performance': 'model_complexity_vs_performance'
    }
    
    for plot_name in selected_plots:
        if plot_name in plot_map:
            plot_key = plot_map[plot_name]
            
            try:
                # Prepare arguments based on plot type
                if len(all_results) == 1:
                    result = list(all_results.values())[0]
                    
                    if plot_key in ['training_curves', 'learning_rate_schedule']:
                        plot_registry.PLOT_REGISTRY[plot_key](result['history'])
                    elif plot_key in ['gradient_flow']:
                        plot_registry.PLOT_REGISTRY[plot_key](result['model'])
                    elif plot_key in ['heatmap', 'correlation_matrix', 'diversity_analysis']:
                        plot_registry.PLOT_REGISTRY[plot_key](result['probe_bank'])
                    else:
                        plot_registry.PLOT_REGISTRY[plot_key](result['results'])
                else:
                    # Multiple results - comparison plots
                    results_dict = {name: r['results'] for name, r in all_results.items()}
                    
                    if plot_key in ['box', 'violin', 'bar_comparison']:
                        plot_registry.PLOT_REGISTRY[plot_key](results_dict)
                    elif plot_key == 'convergence_analysis':
                        histories = [r['history'] for r in all_results.values()]
                        labels = list(all_results.keys())
                        plot_registry.PLOT_REGISTRY[plot_key](histories, labels)
                    else:
                        # Use first result for single-result plots
                        result = list(all_results.values())[0]
                        if plot_key in ['training_curves', 'learning_rate_schedule']:
                            plot_registry.PLOT_REGISTRY[plot_key](result['history'])
                        elif plot_key == 'gradient_flow':
                            plot_registry.PLOT_REGISTRY[plot_key](result['model'])
                        elif plot_key in ['heatmap', 'correlation_matrix', 'diversity_analysis']:
                            plot_registry.PLOT_REGISTRY[plot_key](result['probe_bank'])
                        else:
                            plot_registry.PLOT_REGISTRY[plot_key](result['results'])
            except Exception as e:
                print(f"Warning: Could not generate {plot_name}: {e}")
    
    print("✅ Plots generated")
