#!/usr/bin/env python3
"""
Fairsight CLI - Command Line Interface for Fairsight Toolkit
==========================================================

Provides command-line access to Fairsight configuration and features.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any


def get_config_dir() -> Path:
    """Get the configuration directory path."""
    home = Path.home()
    config_dir = home / ".fairsight"
    return config_dir


def get_config_file() -> Path:
    """Get the configuration file path."""
    config_dir = get_config_dir()
    return config_dir / "config.json"


def load_config() -> Dict[str, Any]:
    """Load configuration from file."""
    config_file = get_config_file()
    if not config_file.exists():
        return {}
    
    try:
        with open(config_file, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(f"⚠️  Warning: Could not load config file: {e}")
        return {}


def save_config(config: Dict[str, Any]) -> None:
    """Save configuration to file."""
    config_dir = get_config_dir()
    config_file = get_config_file()
    
    # Create config directory if it doesn't exist
    config_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
    except IOError as e:
        print(f"❌ Error: Could not save config file: {e}")
        sys.exit(1)


def configure_command(args: argparse.Namespace) -> None:
    """Handle the configure command."""
    api_key = args.api_key
    
    if not api_key:
        print("❌ Error: API key is required. Use --api-key to specify it.")
        sys.exit(1)
    
    # Load existing config
    config = load_config()
    
    # Update API key
    config['api_key'] = api_key
    
    # Save config
    save_config(config)
    
    print("✅ API key saved successfully.")


def show_key_command(args: argparse.Namespace) -> None:
    """Handle the show-key command."""
    config = load_config()
    api_key = config.get('api_key')
    
    if api_key:
        print(f"🔑 API Key: {api_key}")
    else:
        print("❌ No API key set. Use 'fairsight configure --api-key YOUR_KEY' to set one.")


def list_features_command(args: argparse.Namespace) -> None:
    """Handle the list-features command."""
    from .auth import list_free_features, list_premium_features
    
    print("🎯 Fairsight Features:")
    print()
    
    print("🆓 Free Features:")
    free_features = list_free_features()
    for feature, description in free_features.items():
        print(f"  • {feature}: {description}")
    
    print()
    print("⭐ Premium Features:")
    premium_features = list_premium_features()
    for feature, description in premium_features.items():
        print(f"  • {feature}: {description}")
    
    print()
    print("💡 To use premium features, configure your API key:")
    print("   fairsight configure --api-key YOUR_KEY")


def version_command(args: argparse.Namespace) -> None:
    """Handle the version command."""
    from . import __version__
    print(f"fairsight version {__version__}")


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="fairsight",
        description="Fairsight Toolkit - AI Bias Detection and Fairness Auditing CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  fairsight configure --api-key XYZ123
  fairsight show-key
  fairsight list-features
  fairsight version
        """
    )
    
    # Create subparsers for commands
    subparsers = parser.add_subparsers(
        dest='command',
        help='Available commands'
    )
    
    # Configure command
    configure_parser = subparsers.add_parser(
        'configure',
        help='Configure Fairsight settings'
    )
    configure_parser.add_argument(
        '--api-key',
        required=True,
        help='API key for premium features'
    )
    configure_parser.set_defaults(func=configure_command)
    
    # Show key command
    show_key_parser = subparsers.add_parser(
        'show-key',
        help='Display the currently configured API key'
    )
    show_key_parser.set_defaults(func=show_key_command)
    
    # List features command
    list_features_parser = subparsers.add_parser(
        'list-features',
        help='List available free and premium features'
    )
    list_features_parser.set_defaults(func=list_features_command)
    
    # Version command
    version_parser = subparsers.add_parser(
        'version',
        help='Show Fairsight version'
    )
    version_parser.set_defaults(func=version_command)
    
    # Parse arguments
    args = parser.parse_args()
    
    # If no command specified, show help
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Execute the command
    try:
        args.func(args)
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main() 