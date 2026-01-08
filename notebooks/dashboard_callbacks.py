"""
Dashboard Callbacks for PhD Research Dashboard.

Contains callback functions for widget interactions and dynamic UI updates.
"""

from ipywidgets import HTML


def create_callbacks(widgets_dict):
    """
    Create and register all callback functions for widget interactions.
    
    Args:
        widgets_dict: Dictionary of all widgets organized by category
    """
    
    # Model type change callback
    def on_model_type_change(change):
        """Show/hide model-specific parameters based on selected model type."""
        model_type = change['new']
        
        # Get model widgets
        model_widgets = widgets_dict['model']
        
        # Hide all model-specific widgets
        for key in ['cnn_filters', 'cnn_kernels', 'rnn_hidden_size', 'rnn_num_layers',
                    'transformer_d_model', 'transformer_num_heads', 'transformer_num_layers',
                    'transformer_dim_feedforward', 'resnet_hidden_size', 'resnet_num_blocks']:
            if key in model_widgets:
                model_widgets[key].layout.display = 'none'
        
        # Show relevant widgets based on model type
        if model_type == 'CNN':
            model_widgets['cnn_filters'].layout.display = 'flex'
            model_widgets['cnn_kernels'].layout.display = 'flex'
        elif model_type in ['LSTM', 'GRU', 'Hybrid']:
            model_widgets['rnn_hidden_size'].layout.display = 'flex'
            model_widgets['rnn_num_layers'].layout.display = 'flex'
        elif model_type == 'Transformer':
            model_widgets['transformer_d_model'].layout.display = 'flex'
            model_widgets['transformer_num_heads'].layout.display = 'flex'
            model_widgets['transformer_num_layers'].layout.display = 'flex'
            model_widgets['transformer_dim_feedforward'].layout.display = 'flex'
        elif model_type == 'ResNet':
            model_widgets['resnet_hidden_size'].layout.display = 'flex'
            model_widgets['resnet_num_blocks'].layout.display = 'flex'
        
        # MLP and Attention always show hidden_sizes
        if model_type in ['MLP', 'Attention']:
            model_widgets['hidden_sizes'].layout.display = 'flex'
    
    widgets_dict['model']['model_type'].observe(on_model_type_change, names='value')
    
    # Optimizer change callback
    def on_optimizer_change(change):
        """Show/hide optimizer-specific parameters."""
        optimizer = change['new']
        
        training_widgets = widgets_dict['training']
        
        # SGD-specific: momentum
        if optimizer == 'SGD':
            training_widgets['momentum'].layout.display = 'flex'
        else:
            training_widgets['momentum'].layout.display = 'none'
        
        # AdamW-specific: higher weight decay recommended
        if optimizer == 'AdamW':
            if training_widgets['weight_decay'].value == 0.0:
                training_widgets['weight_decay'].value = 0.01
    
    widgets_dict['training']['optimizer'].observe(on_optimizer_change, names='value')
    
    # Scheduler change callback
    def on_scheduler_change(change):
        """Show/hide scheduler-specific parameters."""
        scheduler = change['new']
        
        training_widgets = widgets_dict['training']
        
        # Hide all scheduler-specific widgets
        training_widgets['scheduler_step_size'].layout.display = 'none'
        training_widgets['scheduler_gamma'].layout.display = 'none'
        training_widgets['scheduler_patience'].layout.display = 'none'
        
        # Show relevant widgets
        if scheduler in ['StepLR', 'MultiStepLR']:
            training_widgets['scheduler_step_size'].layout.display = 'flex'
            training_widgets['scheduler_gamma'].layout.display = 'flex'
        elif scheduler == 'ExponentialLR':
            training_widgets['scheduler_gamma'].layout.display = 'flex'
        elif scheduler == 'ReduceLROnPlateau':
            training_widgets['scheduler_patience'].layout.display = 'flex'
            training_widgets['scheduler_gamma'].layout.display = 'flex'
    
    widgets_dict['training']['scheduler'].observe(on_scheduler_change, names='value')
    
    # Loss function change callback
    def on_loss_change(change):
        """Show/hide loss-specific parameters."""
        loss_fn = change['new']
        
        training_widgets = widgets_dict['training']
        
        if loss_fn == 'LabelSmoothing':
            training_widgets['label_smoothing'].layout.display = 'flex'
        else:
            training_widgets['label_smoothing'].layout.display = 'none'
    
    widgets_dict['training']['loss_function'].observe(on_loss_change, names='value')
    
    # Phase mode change callback
    def on_phase_mode_change(change):
        """Show/hide phase bits parameter for discrete mode."""
        phase_mode = change['new']
        
        system_widgets = widgets_dict['system']
        
        if phase_mode == 'discrete':
            system_widgets['phase_bits'].layout.display = 'flex'
        else:
            system_widgets['phase_bits'].layout.display = 'none'
    
    widgets_dict['system']['phase_mode'].observe(on_phase_mode_change, names='value')
    
    # M validation against K
    def validate_M_vs_K(change):
        """Ensure M <= K."""
        M = widgets_dict['system']['M'].value
        K = widgets_dict['system']['K'].value
        
        if M > K:
            widgets_dict['system']['M'].value = K
            widgets_dict['control']['status_text'].value = \
                '<h4 style="color: orange;">⚠️ Warning: M adjusted to K (M cannot exceed K)</h4>'
    
    widgets_dict['system']['M'].observe(validate_M_vs_K, names='value')
    widgets_dict['system']['K'].observe(validate_M_vs_K, names='value')
    
    # Comparison mode callback
    def on_comparison_mode_change(change):
        """Show/hide multi-model comparison widgets."""
        comparison_mode = change['new']
        
        multi_widgets = widgets_dict['multi_experiment']
        
        if comparison_mode:
            multi_widgets['models_to_compare'].layout.display = 'flex'
            widgets_dict['model']['model_type'].disabled = True
            widgets_dict['control']['status_text'].value = \
                '<h4 style="color: blue;">ℹ️ Comparison Mode: Multiple models will be trained</h4>'
        else:
            multi_widgets['models_to_compare'].layout.display = 'none'
            widgets_dict['model']['model_type'].disabled = False
            widgets_dict['control']['status_text'].value = \
                '<h3 style="color: #555;">Status: Ready</h3>'
    
    widgets_dict['multi_experiment']['comparison_mode'].observe(on_comparison_mode_change, names='value')
    
    # Multi-seed callback
    def on_multi_seed_change(change):
        """Show/hide multi-seed parameters."""
        multi_seed = change['new']
        
        multi_widgets = widgets_dict['multi_experiment']
        
        if multi_seed:
            multi_widgets['num_seeds'].layout.display = 'flex'
            multi_widgets['seed_start'].layout.display = 'flex'
            widgets_dict['data']['seed'].disabled = True
        else:
            multi_widgets['num_seeds'].layout.display = 'none'
            multi_widgets['seed_start'].layout.display = 'none'
            widgets_dict['data']['seed'].disabled = False
    
    widgets_dict['multi_experiment']['multi_seed'].observe(on_multi_seed_change, names='value')
    
    # Early stopping callback
    def on_early_stopping_change(change):
        """Show/hide early stopping patience."""
        early_stopping = change['new']
        
        training_widgets = widgets_dict['training']
        
        if early_stopping:
            training_widgets['early_stopping_patience'].layout.display = 'flex'
        else:
            training_widgets['early_stopping_patience'].layout.display = 'none'
    
    widgets_dict['training']['early_stopping'].observe(on_early_stopping_change, names='value')
    
    # Initialize visibility based on default values
    # Trigger all callbacks once to set initial state
    on_model_type_change({'new': widgets_dict['model']['model_type'].value})
    on_optimizer_change({'new': widgets_dict['training']['optimizer'].value})
    on_scheduler_change({'new': widgets_dict['training']['scheduler'].value})
    on_loss_change({'new': widgets_dict['training']['loss_function'].value})
    on_phase_mode_change({'new': widgets_dict['system']['phase_mode'].value})
    on_comparison_mode_change({'new': widgets_dict['multi_experiment']['comparison_mode'].value})
    on_multi_seed_change({'new': widgets_dict['multi_experiment']['multi_seed'].value})
    on_early_stopping_change({'new': widgets_dict['training']['early_stopping'].value})
    
    return {
        'on_model_type_change': on_model_type_change,
        'on_optimizer_change': on_optimizer_change,
        'on_scheduler_change': on_scheduler_change,
        'on_loss_change': on_loss_change,
        'on_phase_mode_change': on_phase_mode_change,
        'validate_M_vs_K': validate_M_vs_K,
        'on_comparison_mode_change': on_comparison_mode_change,
        'on_multi_seed_change': on_multi_seed_change,
        'on_early_stopping_change': on_early_stopping_change,
    }


def create_button_callbacks(widgets_dict, runner_func):
    """
    Create callbacks for control buttons.
    
    Args:
        widgets_dict: Dictionary of all widgets
        runner_func: Function to run experiments
    """
    
    def on_run_click(b):
        """Handle run button click."""
        # Update status
        widgets_dict['control']['status_text'].value = \
            '<h3 style="color: blue;">🚀 Running experiment...</h3>'
        widgets_dict['control']['run_button'].disabled = True
        widgets_dict['control']['stop_button'].disabled = False
        
        try:
            # Run the experiment
            runner_func(widgets_dict)
            
            # Update status on success
            widgets_dict['control']['status_text'].value = \
                '<h3 style="color: green;">✅ Experiment completed successfully!</h3>'
        except Exception as e:
            # Update status on error
            widgets_dict['control']['status_text'].value = \
                f'<h3 style="color: red;">❌ Error: {str(e)}</h3>'
        finally:
            widgets_dict['control']['run_button'].disabled = False
            widgets_dict['control']['stop_button'].disabled = True
    
    def on_stop_click(b):
        """Handle stop button click."""
        widgets_dict['control']['status_text'].value = \
            '<h3 style="color: orange;">⏹ Stopping experiment...</h3>'
        # Note: Actual stopping logic would need to be implemented in runner
        widgets_dict['control']['stop_button'].disabled = True
    
    def on_clear_click(b):
        """Handle clear output button click."""
        from IPython.display import clear_output
        clear_output(wait=False)
        widgets_dict['control']['status_text'].value = \
            '<h3 style="color: #555;">Status: Output cleared, ready for new run</h3>'
    
    widgets_dict['control']['run_button'].on_click(on_run_click)
    widgets_dict['control']['stop_button'].on_click(on_stop_click)
    widgets_dict['control']['clear_button'].on_click(on_clear_click)
    
    # Disable stop button initially
    widgets_dict['control']['stop_button'].disabled = True
    
    return {
        'on_run_click': on_run_click,
        'on_stop_click': on_stop_click,
        'on_clear_click': on_clear_click,
    }
