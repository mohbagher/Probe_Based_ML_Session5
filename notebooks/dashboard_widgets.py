"""
Dashboard Widgets for PhD Research Dashboard.

Contains all widget definitions organized by category.
"""

import ipywidgets as widgets
from ipywidgets import (
    IntSlider, FloatSlider, Dropdown, SelectMultiple, Checkbox, 
    Text, Textarea, IntText, FloatText, Button, HTML, VBox, HBox, Tab
)


def create_system_widgets():
    """Create widgets for system parameters (N, K, M, probe types, channel models)."""
    return {
        'N': IntSlider(
            min=8, max=256, step=8, value=32,
            description='N (RIS Elements):',
            style={'description_width': '180px'},
            layout=widgets.Layout(width='500px')
        ),
        'K': IntSlider(
            min=16, max=512, step=16, value=64,
            description='K (Codebook Size):',
            style={'description_width': '180px'},
            layout=widgets.Layout(width='500px')
        ),
        'M': IntSlider(
            min=2, max=64, step=2, value=8,
            description='M (Sensing Budget):',
            style={'description_width': '180px'},
            layout=widgets.Layout(width='500px')
        ),
        'P_tx': FloatSlider(
            min=0.1, max=10.0, step=0.1, value=1.0,
            description='P_tx (Transmit Power):',
            style={'description_width': '180px'},
            layout=widgets.Layout(width='500px')
        ),
        'sigma_h_sq': FloatSlider(
            min=0.1, max=5.0, step=0.1, value=1.0,
            description='σ²_h (BS-RIS variance):',
            style={'description_width': '180px'},
            layout=widgets.Layout(width='500px')
        ),
        'sigma_g_sq': FloatSlider(
            min=0.1, max=5.0, step=0.1, value=1.0,
            description='σ²_g (RIS-UE variance):',
            style={'description_width': '180px'},
            layout=widgets.Layout(width='500px')
        ),
        'probe_type': Dropdown(
            options=['continuous', 'binary', '2bit', 'hadamard', 'sobol', 'halton'],
            value='continuous',
            description='Probe Type:',
            style={'description_width': '180px'},
            layout=widgets.Layout(width='500px')
        ),
        'phase_mode': Dropdown(
            options=['continuous', 'discrete'],
            value='continuous',
            description='Phase Mode:',
            style={'description_width': '180px'},
            layout=widgets.Layout(width='500px')
        ),
        'phase_bits': IntSlider(
            min=1, max=8, step=1, value=3,
            description='Phase Bits (discrete):',
            style={'description_width': '180px'},
            layout=widgets.Layout(width='500px')
        ),
    }


def create_model_widgets():
    """Create widgets for model architecture selection and hyperparameters."""
    return {
        'model_type': Dropdown(
            options=['MLP', 'CNN', 'LSTM', 'GRU', 'Attention', 'Transformer', 'ResNet', 'Hybrid'],
            value='MLP',
            description='Model Architecture:',
            style={'description_width': '180px'},
            layout=widgets.Layout(width='500px')
        ),
        'hidden_sizes': Text(
            value='512,256,128',
            description='Hidden Sizes (MLP):',
            style={'description_width': '180px'},
            layout=widgets.Layout(width='500px')
        ),
        'dropout_prob': FloatSlider(
            min=0.0, max=0.5, step=0.05, value=0.1,
            description='Dropout Probability:',
            style={'description_width': '180px'},
            layout=widgets.Layout(width='500px')
        ),
        'use_batch_norm': Checkbox(
            value=True,
            description='Use Batch Normalization',
            style={'description_width': '180px'}
        ),
        # CNN-specific
        'cnn_filters': Text(
            value='32,64,128',
            description='CNN Filters:',
            style={'description_width': '180px'},
            layout=widgets.Layout(width='500px')
        ),
        'cnn_kernels': Text(
            value='5,5,3',
            description='CNN Kernel Sizes:',
            style={'description_width': '180px'},
            layout=widgets.Layout(width='500px')
        ),
        # LSTM/GRU-specific
        'rnn_hidden_size': IntSlider(
            min=32, max=512, step=32, value=128,
            description='RNN Hidden Size:',
            style={'description_width': '180px'},
            layout=widgets.Layout(width='500px')
        ),
        'rnn_num_layers': IntSlider(
            min=1, max=5, step=1, value=2,
            description='RNN Layers:',
            style={'description_width': '180px'},
            layout=widgets.Layout(width='500px')
        ),
        # Transformer-specific
        'transformer_d_model': IntSlider(
            min=64, max=512, step=64, value=256,
            description='Transformer d_model:',
            style={'description_width': '180px'},
            layout=widgets.Layout(width='500px')
        ),
        'transformer_num_heads': IntSlider(
            min=2, max=16, step=2, value=8,
            description='Attention Heads:',
            style={'description_width': '180px'},
            layout=widgets.Layout(width='500px')
        ),
        'transformer_num_layers': IntSlider(
            min=1, max=6, step=1, value=3,
            description='Transformer Layers:',
            style={'description_width': '180px'},
            layout=widgets.Layout(width='500px')
        ),
        'transformer_dim_feedforward': IntSlider(
            min=128, max=2048, step=128, value=512,
            description='FFN Dimension:',
            style={'description_width': '180px'},
            layout=widgets.Layout(width='500px')
        ),
        # ResNet-specific
        'resnet_hidden_size': IntSlider(
            min=128, max=1024, step=128, value=512,
            description='ResNet Hidden Size:',
            style={'description_width': '180px'},
            layout=widgets.Layout(width='500px')
        ),
        'resnet_num_blocks': IntSlider(
            min=2, max=10, step=1, value=4,
            description='ResNet Blocks:',
            style={'description_width': '180px'},
            layout=widgets.Layout(width='500px')
        ),
    }


def create_training_widgets():
    """Create widgets for training configuration."""
    return {
        'epochs': IntSlider(
            min=10, max=200, step=10, value=50,
            description='Epochs:',
            style={'description_width': '180px'},
            layout=widgets.Layout(width='500px')
        ),
        'batch_size': Dropdown(
            options=[32, 64, 128, 256, 512],
            value=128,
            description='Batch Size:',
            style={'description_width': '180px'},
            layout=widgets.Layout(width='500px')
        ),
        'optimizer': Dropdown(
            options=['Adam', 'AdamW', 'SGD', 'RMSprop', 'AdaGrad', 'Adadelta', 'Adamax'],
            value='Adam',
            description='Optimizer:',
            style={'description_width': '180px'},
            layout=widgets.Layout(width='500px')
        ),
        'learning_rate': FloatText(
            value=1e-3,
            description='Learning Rate:',
            style={'description_width': '180px'},
            layout=widgets.Layout(width='500px')
        ),
        'weight_decay': FloatText(
            value=0.0,
            description='Weight Decay:',
            style={'description_width': '180px'},
            layout=widgets.Layout(width='500px')
        ),
        'momentum': FloatSlider(
            min=0.0, max=0.99, step=0.01, value=0.9,
            description='Momentum (SGD):',
            style={'description_width': '180px'},
            layout=widgets.Layout(width='500px')
        ),
        'scheduler': Dropdown(
            options=['None', 'StepLR', 'MultiStepLR', 'ExponentialLR', 'CosineAnnealingLR', 
                    'CosineAnnealingWarmRestarts', 'ReduceLROnPlateau', 'OneCycleLR'],
            value='ReduceLROnPlateau',
            description='LR Scheduler:',
            style={'description_width': '180px'},
            layout=widgets.Layout(width='500px')
        ),
        'scheduler_step_size': IntSlider(
            min=5, max=50, step=5, value=10,
            description='Step Size:',
            style={'description_width': '180px'},
            layout=widgets.Layout(width='500px')
        ),
        'scheduler_gamma': FloatSlider(
            min=0.1, max=0.9, step=0.05, value=0.5,
            description='Gamma:',
            style={'description_width': '180px'},
            layout=widgets.Layout(width='500px')
        ),
        'scheduler_patience': IntSlider(
            min=3, max=20, step=1, value=5,
            description='Patience:',
            style={'description_width': '180px'},
            layout=widgets.Layout(width='500px')
        ),
        'loss_function': Dropdown(
            options=['CrossEntropy', 'FocalLoss', 'LabelSmoothing'],
            value='CrossEntropy',
            description='Loss Function:',
            style={'description_width': '180px'},
            layout=widgets.Layout(width='500px')
        ),
        'label_smoothing': FloatSlider(
            min=0.0, max=0.3, step=0.01, value=0.1,
            description='Label Smoothing:',
            style={'description_width': '180px'},
            layout=widgets.Layout(width='500px')
        ),
        'early_stopping': Checkbox(
            value=True,
            description='Early Stopping',
            style={'description_width': '180px'}
        ),
        'early_stopping_patience': IntSlider(
            min=5, max=30, step=5, value=10,
            description='ES Patience:',
            style={'description_width': '180px'},
            layout=widgets.Layout(width='500px')
        ),
    }


def create_data_widgets():
    """Create widgets for data generation parameters."""
    return {
        'n_train': IntText(
            value=50000,
            description='Training Samples:',
            style={'description_width': '180px'},
            layout=widgets.Layout(width='500px')
        ),
        'n_val': IntText(
            value=5000,
            description='Validation Samples:',
            style={'description_width': '180px'},
            layout=widgets.Layout(width='500px')
        ),
        'n_test': IntText(
            value=5000,
            description='Test Samples:',
            style={'description_width': '180px'},
            layout=widgets.Layout(width='500px')
        ),
        'seed': IntText(
            value=42,
            description='Random Seed:',
            style={'description_width': '180px'},
            layout=widgets.Layout(width='500px')
        ),
        'normalize_input': Checkbox(
            value=True,
            description='Normalize Input',
            style={'description_width': '180px'}
        ),
        'normalization_type': Dropdown(
            options=['mean', 'std', 'log'],
            value='mean',
            description='Normalization Type:',
            style={'description_width': '180px'},
            layout=widgets.Layout(width='500px')
        ),
    }


def create_evaluation_widgets():
    """Create widgets for evaluation and visualization."""
    return {
        'plot_types': SelectMultiple(
            options=[
                'Training Curves',
                'Learning Rate Schedule',
                'Gradient Flow',
                'Eta Distribution',
                'CDF',
                'PDF Histogram',
                'Box Plot',
                'Violin Plot',
                'Scatter Comparison',
                'Bar Comparison',
                'Radar Chart',
                'Heatmap Comparison',
                'Probe Heatmap',
                'Correlation Matrix',
                'Diversity Analysis',
                'Probe Power Distribution',
                '3D Surface',
                'ROC Curves',
                'Precision-Recall',
                'Confusion Matrix',
                'Top-M Comparison',
                'Baseline Comparison',
                'Convergence Analysis',
                'Parameter Sensitivity',
                'Model Complexity vs Performance'
            ],
            value=['Training Curves', 'Eta Distribution', 'CDF', 'Box Plot'],
            rows=10,
            description='Plot Types:',
            style={'description_width': '180px'},
            layout=widgets.Layout(width='500px', height='250px')
        ),
        'export_format': SelectMultiple(
            options=['CSV', 'JSON', 'YAML', 'HDF5', 'Excel', 'Pickle', 
                    'PNG', 'PDF', 'SVG', 'EPS'],
            value=['CSV', 'PNG'],
            rows=5,
            description='Export Formats:',
            style={'description_width': '180px'},
            layout=widgets.Layout(width='500px', height='150px')
        ),
        'save_results': Checkbox(
            value=True,
            description='Save Results',
            style={'description_width': '180px'}
        ),
        'save_model': Checkbox(
            value=True,
            description='Save Model Checkpoint',
            style={'description_width': '180px'}
        ),
        'output_dir': Text(
            value='results/dashboard_experiments',
            description='Output Directory:',
            style={'description_width': '180px'},
            layout=widgets.Layout(width='500px')
        ),
    }


def create_multi_experiment_widgets():
    """Create widgets for running multiple experiments (comparisons, sweeps)."""
    return {
        'comparison_mode': Checkbox(
            value=False,
            description='Multi-Model Comparison Mode',
            style={'description_width': '250px'}
        ),
        'models_to_compare': SelectMultiple(
            options=['MLP', 'CNN', 'LSTM', 'GRU', 'Attention', 'Transformer', 'ResNet', 'Hybrid'],
            value=['MLP', 'CNN', 'LSTM'],
            rows=5,
            description='Models to Compare:',
            style={'description_width': '180px'},
            layout=widgets.Layout(width='500px', height='150px')
        ),
        'multi_seed': Checkbox(
            value=False,
            description='Multi-Seed Runs',
            style={'description_width': '180px'}
        ),
        'num_seeds': IntSlider(
            min=3, max=10, step=1, value=5,
            description='Number of Seeds:',
            style={'description_width': '180px'},
            layout=widgets.Layout(width='500px')
        ),
        'seed_start': IntText(
            value=42,
            description='Starting Seed:',
            style={'description_width': '180px'},
            layout=widgets.Layout(width='500px')
        ),
    }


def create_control_widgets():
    """Create control buttons and status display."""
    return {
        'run_button': Button(
            description='🚀 RUN EXPERIMENT',
            button_style='success',
            layout=widgets.Layout(width='300px', height='50px'),
            style={'font_weight': 'bold'}
        ),
        'stop_button': Button(
            description='⏹ STOP',
            button_style='danger',
            layout=widgets.Layout(width='150px', height='50px')
        ),
        'clear_button': Button(
            description='🗑 CLEAR OUTPUT',
            button_style='warning',
            layout=widgets.Layout(width='150px', height='50px')
        ),
        'status_text': HTML(
            value='<h3 style="color: #555;">Status: Ready</h3>',
            layout=widgets.Layout(width='100%')
        ),
        'progress_text': HTML(
            value='',
            layout=widgets.Layout(width='100%')
        ),
    }


def create_all_widgets():
    """Create all widgets organized by category."""
    return {
        'system': create_system_widgets(),
        'model': create_model_widgets(),
        'training': create_training_widgets(),
        'data': create_data_widgets(),
        'evaluation': create_evaluation_widgets(),
        'multi_experiment': create_multi_experiment_widgets(),
        'control': create_control_widgets(),
    }
