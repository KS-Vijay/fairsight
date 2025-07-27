"""
Tests for the Fairsight CLI functionality.
"""

import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest
import sys
import os

# Add the fairsight package to the path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from fairsight.cli import (
    get_config_dir, 
    get_config_file, 
    load_config, 
    save_config,
    configure_command,
    show_key_command,
    list_features_command,
    version_command
)


class TestCLI:
    """Test cases for CLI functionality."""
    
    def setup_method(self):
        """Set up test environment."""
        # Create a temporary directory for testing
        self.temp_dir = tempfile.mkdtemp()
        self.original_home = Path.home()
        
        # Mock the home directory to use our temp directory
        self.home_patcher = patch('pathlib.Path.home')
        self.mock_home = self.home_patcher.start()
        self.mock_home.return_value = Path(self.temp_dir)
    
    def teardown_method(self):
        """Clean up test environment."""
        self.home_patcher.stop()
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_get_config_dir(self):
        """Test getting the configuration directory."""
        config_dir = get_config_dir()
        expected_dir = Path(self.temp_dir) / ".fairsight"
        assert config_dir == expected_dir
    
    def test_get_config_file(self):
        """Test getting the configuration file path."""
        config_file = get_config_file()
        expected_file = Path(self.temp_dir) / ".fairsight" / "config.json"
        assert config_file == expected_file
    
    def test_load_config_empty(self):
        """Test loading config when file doesn't exist."""
        config = load_config()
        assert config == {}
    
    def test_save_and_load_config(self):
        """Test saving and loading configuration."""
        test_config = {"api_key": "test_key_123", "setting": "value"}
        
        # Save config
        save_config(test_config)
        
        # Verify file was created
        config_file = get_config_file()
        assert config_file.exists()
        
        # Load config
        loaded_config = load_config()
        assert loaded_config == test_config
    
    def test_configure_command(self):
        """Test the configure command."""
        args = MagicMock()
        args.api_key = "test_api_key_456"
        
        # Run configure command
        configure_command(args)
        
        # Verify config was saved
        config = load_config()
        assert config["api_key"] == "test_api_key_456"
    
    def test_show_key_command_with_key(self):
        """Test show-key command when API key is set."""
        # Set up config with API key
        test_config = {"api_key": "test_key_789"}
        save_config(test_config)
        
        args = MagicMock()
        
        # Capture stdout
        with patch('builtins.print') as mock_print:
            show_key_command(args)
            mock_print.assert_called_with("🔑 API Key: test_key_789")
    
    def test_show_key_command_without_key(self):
        """Test show-key command when no API key is set."""
        args = MagicMock()
        
        # Capture stdout
        with patch('builtins.print') as mock_print:
            show_key_command(args)
            mock_print.assert_called_with("❌ No API key set. Use 'fairsight configure --api-key YOUR_KEY' to set one.")
    
    def test_version_command(self):
        """Test the version command."""
        args = MagicMock()
        
        with patch('builtins.print') as mock_print:
            version_command(args)
            mock_print.assert_called_with("fairsight version 1.0.0")
    
    def test_list_features_command(self):
        """Test the list-features command."""
        args = MagicMock()
        
        with patch('builtins.print') as mock_print:
            list_features_command(args)
            
            # Verify that print was called multiple times (for features list)
            assert mock_print.call_count > 5  # Should print multiple lines
    
    def test_load_config_corrupted_file(self):
        """Test loading config with corrupted JSON file."""
        # Create a corrupted config file
        config_file = get_config_file()
        config_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_file, 'w') as f:
            f.write("invalid json content")
        
        # Should return empty dict and not crash
        config = load_config()
        assert config == {}


if __name__ == "__main__":
    pytest.main([__file__]) 