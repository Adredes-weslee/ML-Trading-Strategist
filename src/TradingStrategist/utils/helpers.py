"""
Helper utilities for TradingStrategist.
"""
import yaml
import os
import datetime as dt
from pathlib import Path


def load_config(config_path):
    """
    Load configuration from YAML file.
    
    Parameters:
    -----------
    config_path : str
        Path to YAML configuration file
        
    Returns:
    --------
    dict
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def save_config(config, config_path):
    """
    Save configuration to YAML file.
    
    Parameters:
    -----------
    config : dict
        Configuration dictionary
    config_path : str
        Path to save YAML configuration file
    """
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def get_output_path():
    """
    Get the path to save output files and ensure it exists.
    
    Returns:
    --------
    str
        Path to output directory
    """
    # Create output directory relative to project root
    output_dir = Path(__file__).parent.parent.parent.parent / "output"
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def get_config_path(config_name=None):
    """
    Get the path to configuration files.
    
    Parameters:
    -----------
    config_name : str, optional
        Name of specific config file (without .yaml extension)
        
    Returns:
    --------
    str or Path
        Path to config directory or specific config file
    """
    config_dir = Path(__file__).parent.parent.parent.parent / "configs"
    
    if config_name:
        return config_dir / f"{config_name}.yaml"
    
    return config_dir


def create_timestamp_folder():
    """
    Create a timestamped folder for experiment outputs.
    
    Returns:
    --------
    Path
        Path to created folder
    """
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_path = get_output_path() / timestamp
    os.makedirs(folder_path, exist_ok=True)
    return folder_path


def merge_configs(base_config, override_config):
    """
    Merge two configuration dictionaries with override_config taking precedence.
    
    Parameters:
    -----------
    base_config : dict
        Base configuration
    override_config : dict
        Override configuration
        
    Returns:
    --------
    dict
        Merged configuration
    """
    result = base_config.copy()
    
    for key, value in override_config.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
            
    return result

# Add this function to helpers.py

def validate_config(config, required_sections=None, required_params=None):
    """
    Validate that configuration has required sections and parameters.
    
    Parameters:
    -----------
    config : dict
        Configuration dictionary to validate
    required_sections : list, optional
        List of section names that must exist in config
    required_params : dict, optional
        Dictionary mapping section names to lists of required parameters
        
    Returns:
    --------
    bool
        True if config is valid
        
    Raises:
    -------
    ValueError
        If config is missing required sections or parameters
    """
    if required_sections is None:
        required_sections = []
        
    if required_params is None:
        required_params = {}
    
    # Check required sections
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required section: {section}")
    
    # Check required parameters
    for section, params in required_params.items():
        if section in config:
            for param in params:
                if param not in config[section]:
                    raise ValueError(f"Missing required parameter: {section}.{param}")
    
    return True