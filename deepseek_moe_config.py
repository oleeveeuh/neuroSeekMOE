"""
DeepSeek-MoE Recommended Configuration Templates

This module provides pre-configured hyperparameters following DeepSeek-MoE best practices
for different dataset sizes. Use these configurations to ensure reproducibility and
optimal performance.

Usage:
    from deepseek_moe_config import DeepSeekMoEConfig
    
    # Auto-select based on dataset size
    config_dict = DeepSeekMoEConfig.get_config(len(train_dataset))
    model = SimpleMoEModel(**config_dict)
    
    # Or use a specific configuration
    model = SimpleMoEModel(**DeepSeekMoEConfig.SMALL)
"""


class DeepSeekMoEConfig:
    """DeepSeek-MoE Recommended Configuration for Different Dataset Sizes
    
    These configurations follow DeepSeek-MoE best practices and are tuned for:
    - Small datasets (< 10k samples): Fewer experts, higher regularization
    - Medium datasets (10k - 1M samples): Balanced configuration
    - Large datasets (> 1M samples): More experts, lower regularization
    
    All configurations use:
    - Expert Choice routing
    - Learnable shared expert weighting
    - Temperature scheduling
    - DeepSeek-aligned auxiliary losses
    """
    
    # Small dataset (< 10k samples) - typical for research/prototyping
    SMALL = {
        # Model architecture
        'num_shared_experts': 2,
        'num_routed_experts': 4,
        'top_k': 2,
        'noise_scale': 0.5,
        'load_balance_loss_weight': 0.1,
        'z_loss_weight': 0.001,
        'capacity_factor': 1.5,
        'residual_factor': 0.1,
        'temperature_start': 5.0,  # Higher start for better exploration
        'temperature_end': 0.01,
        'temperature_schedule': 'linear',
        'temperature_steps': 1000,
        # Training parameters (for reference, not passed to model)
        'dropout_rate': 0.1,  # Note: Currently hardcoded in model, included for reference
        'learning_rate': 1e-4,
        'warmup_ratio': 0.1,  # 10% of total steps
    }
    
    # Medium dataset (10k - 1M samples) - typical for production
    MEDIUM = {
        # Model architecture
        'num_shared_experts': 4,
        'num_routed_experts': 16,
        'top_k': 4,
        'noise_scale': 0.3,
        'load_balance_loss_weight': 0.01,
        'z_loss_weight': 0.001,
        'capacity_factor': 1.5,
        'residual_factor': 0.1,
        'temperature_start': 2.0,
        'temperature_end': 0.1,
        'temperature_schedule': 'cosine',
        'temperature_steps': 5000,
        # Training parameters
        'dropout_rate': 0.15,
        'learning_rate': 5e-5,
        'warmup_ratio': 0.05,  # 5% of total steps
    }
    
    # Large dataset (> 1M samples) - DeepSeek scale
    LARGE = {
        # Model architecture
        'num_shared_experts': 8,
        'num_routed_experts': 64,
        'top_k': 8,
        'noise_scale': 0.1,
        'load_balance_loss_weight': 0.01,
        'z_loss_weight': 0.001,
        'capacity_factor': 2.0,
        'residual_factor': 0.1,
        'temperature_start': 1.0,
        'temperature_end': 0.05,
        'temperature_schedule': 'cosine',
        'temperature_steps': 10000,
        # Training parameters
        'dropout_rate': 0.2,
        'learning_rate': 1e-5,
        'warmup_ratio': 0.01,  # 1% of total steps
    }
    
    @classmethod
    def get_config(cls, dataset_size: int, model_only: bool = True):
        """Auto-select configuration based on dataset size.
        
        Args:
            dataset_size: Number of training samples in the dataset
            model_only: If True, return only model parameters (exclude training params)
                        If False, return all parameters including training config
            
        Returns:
            Dictionary of configuration parameters for SimpleMoEModel
            If model_only=False, includes training parameters (learning_rate, warmup_ratio, etc.)
            
        Examples:
            >>> # Get only model parameters (default)
            >>> config = DeepSeekMoEConfig.get_config(5000)
            >>> model = SimpleMoEModel(**config)
            
            >>> # Get all parameters including training config
            >>> config = DeepSeekMoEConfig.get_config(5000, model_only=False)
            >>> model = SimpleMoEModel(**{k: v for k, v in config.items() 
            ...                            if k not in ['dropout_rate', 'learning_rate', 'warmup_ratio']})
            >>> learning_rate = config['learning_rate']
        """
        if dataset_size < 10000:
            config = cls.SMALL.copy()
        elif dataset_size < 1000000:
            config = cls.MEDIUM.copy()
        else:
            config = cls.LARGE.copy()
        
        # Remove training parameters if model_only=True
        if model_only:
            config = {k: v for k, v in config.items() 
                     if k not in ['dropout_rate', 'learning_rate', 'warmup_ratio']}
        
        return config
    
    @classmethod
    def get_training_config(cls, dataset_size: int, num_epochs: int, batch_size: int):
        """Get both model and training configuration.
        
        Args:
            dataset_size: Number of training samples
            num_epochs: Number of training epochs
            batch_size: Training batch size
            
        Returns:
            Tuple of (model_config_dict, training_config_dict)
            - model_config: Parameters for SimpleMoEModel
            - training_config: Parameters for training (learning_rate, warmup_ratio, etc.)
        """
        model_config = cls.get_config(dataset_size)
        
        # Training configs vary by dataset size
        total_steps = (dataset_size // batch_size) * num_epochs
        
        if dataset_size < 10000:
            training_config = {
                'learning_rate': 1e-4,
                'warmup_ratio': 0.1,  # 10% of total steps
                'warmup_steps': max(1, int(total_steps * 0.1)),
                'weight_decay': 1e-3,
            }
        elif dataset_size < 1000000:
            training_config = {
                'learning_rate': 5e-5,
                'warmup_ratio': 0.05,  # 5% of total steps
                'warmup_steps': max(1, int(total_steps * 0.05)),
                'weight_decay': 1e-4,
            }
        else:
            training_config = {
                'learning_rate': 1e-5,
                'warmup_ratio': 0.01,  # 1% of total steps
                'warmup_steps': max(1, int(total_steps * 0.01)),
                'weight_decay': 1e-5,
            }
        
        return model_config, training_config


# Example usage
if __name__ == "__main__":
    # Example 1: Auto-select configuration
    print("Example 1: Auto-select based on dataset size")
    small_config = DeepSeekMoEConfig.get_config(5000)
    print(f"Small dataset config: {small_config}")
    
    medium_config = DeepSeekMoEConfig.get_config(50000)
    print(f"Medium dataset config: {medium_config}")
    
    large_config = DeepSeekMoEConfig.get_config(5000000)
    print(f"Large dataset config: {large_config}")
    
    # Example 2: Get both model and training config
    print("\nExample 2: Get model and training config")
    model_config, training_config = DeepSeekMoEConfig.get_training_config(
        dataset_size=5000,
        num_epochs=10,
        batch_size=8
    )
    print(f"Model config: {model_config}")
    print(f"Training config: {training_config}")

