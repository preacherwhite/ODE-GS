"""
Configuration utilities for loading and managing experiment configurations.
"""

import os
import yaml
import argparse
from typing import Dict, Any, Optional
from omegaconf import OmegaConf, DictConfig
import sys


class ConfigLoader:
    """Handles loading and merging of configuration files."""
    
    @staticmethod
    def load_config(config_path: str) -> DictConfig:
        """Load a YAML configuration file."""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        return OmegaConf.create(config_dict)
    
    @staticmethod
    def merge_configs(base_config: DictConfig, override_config: Optional[DictConfig] = None) -> DictConfig:
        """Merge two configurations, with override taking precedence."""
        if override_config is None:
            return base_config
        
        return OmegaConf.merge(base_config, override_config)
    
    @staticmethod
    def from_cli_args(args: argparse.Namespace) -> DictConfig:
        """Convert command-line arguments to OmegaConf configuration."""
        # Filter out None values and system arguments
        args_dict = {k: v for k, v in vars(args).items() 
                     if v is not None and not k.startswith('_')}
        
        return OmegaConf.create(args_dict)
    
    @staticmethod
    def save_config(config: DictConfig, save_path: str):
        """Save configuration to a YAML file."""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            yaml.dump(OmegaConf.to_container(config, resolve=True), f, default_flow_style=False)


def setup_config(args: argparse.Namespace) -> DictConfig:
    """
    Setup configuration from default config, optional override config, and CLI args.
    
    Priority (highest to lowest):
    1. Command-line arguments
    2. Override config file (if specified)
    3. Default config file
    """
    # Load default config
    default_config_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'default_config.yaml')
    default_config = ConfigLoader.load_config(default_config_path)
    
    # Load override config if specified
    override_config = None
    if hasattr(args, 'config') and args.config:
        override_config = ConfigLoader.load_config(args.config)
    
    # Convert CLI args to config
    cli_config = ConfigLoader.from_cli_args(args)
    
    # Merge configurations
    config = ConfigLoader.merge_configs(default_config, override_config)
    config = ConfigLoader.merge_configs(config, cli_config)
    
    # Ensure all paths are absolute
    if 'source_path' in config:
        config.source_path = os.path.abspath(config.source_path)
    if 'model_path' in config:
        config.model_path = os.path.abspath(config.model_path)
    
    return config


class DotDict:
    """
    A dictionary that supports dot notation access.
    This allows accessing config values as config.model.latent_dim
    instead of config['model']['latent_dim'].
    """
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            if isinstance(value, dict):
                value = DotDict(value)
            self.__dict__[key] = value
    
    def __getattr__(self, key):
        return self.__dict__.get(key, None)
    
    def __setattr__(self, key, value):
        self.__dict__[key] = value
    
    def get(self, key, default=None):
        return self.__dict__.get(key, default)
    
    def to_dict(self):
        """Convert back to regular dictionary."""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, DotDict):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result 