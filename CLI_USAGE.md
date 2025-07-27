# Fairsight CLI Usage Guide

The Fairsight CLI provides command-line access to configure and manage your Fairsight Toolkit installation.

## Installation

After installing the Fairsight package, the CLI is automatically available as the `fairsight` command.

### Development Installation

If you're developing Fairsight locally:

```bash
# Install in development mode
pip install -e .

# The CLI will be available as 'fairsight'
fairsight --help
```

## Available Commands

### 1. Configure API Key

Set up your API key for premium features:

```bash
fairsight configure --api-key YOUR_API_KEY
```

**Example:**
```bash
fairsight configure --api-key XYZ123456789
```

**What it does:**
- Saves the API key to `~/.fairsight/config.json`
- Creates the config directory if it doesn't exist
- Displays a success message

### 2. Show API Key

Display the currently configured API key:

```bash
fairsight show-key
```

**Output examples:**
```
🔑 API Key: XYZ123456789
```

Or if no key is set:
```
❌ No API key set. Use 'fairsight configure --api-key YOUR_KEY' to set one.
```

### 3. List Features

View all available free and premium features:

```bash
fairsight list-features
```

**Output:**
```
🎯 Fairsight Features:

🆓 Free Features:
  • basic_dataset_audit: Basic dataset bias and fairness analysis
  • basic_model_audit: Basic model performance and bias evaluation
  • basic_bias_detection: Core bias detection algorithms
  • utility_functions: Data preprocessing and utility functions
  • data_preprocessing: Data cleaning and preprocessing tools
  • data_fingerprinting: Data fingerprinting and duplicate detection

⭐ Premium Features:
  • comprehensive_audit: Complete Fairsight audit with all components
  • advanced_reporting: Advanced report generation with custom templates
  • dashboard_integration: SAP HANA Cloud dashboard integration
  • sap_hana_integration: Full SAP HANA Cloud connectivity
  • illegal_data_detection: AI-powered illegal data detection
  • advanced_explainability: Advanced SHAP and LIME explainability features

💡 To use premium features, configure your API key:
   fairsight configure --api-key YOUR_KEY
```

### 4. Version

Display the current Fairsight version:

```bash
fairsight version
```

**Output:**
```
fairsight version 1.0.0
```

## Configuration File

The CLI stores configuration in `~/.fairsight/config.json`:

```json
{
  "api_key": "YOUR_API_KEY_HERE"
}
```

### File Location

- **Windows:** `%USERPROFILE%\.fairsight\config.json`
- **macOS/Linux:** `~/.fairsight/config.json`

## Automatic API Key Loading

Once you've configured your API key using the CLI, premium features will automatically use it without requiring you to pass it as an argument.

**Example:**
```python
from fairsight import IllegalDataDetector

# This will automatically load the API key from ~/.fairsight/config.json
detector = IllegalDataDetector()
result = detector.detect_illegal_content("path/to/image.jpg")
```

## Error Handling

The CLI provides clear error messages:

- **Missing API key:** Prompts you to configure one
- **Invalid configuration:** Warns about corrupted config files
- **Permission errors:** Clear messages about file access issues

## Future Commands

The CLI is designed to be easily extensible. Future commands may include:

- `fairsight audit-model` - Run model audits from command line
- `fairsight audit-dataset` - Run dataset audits from command line
- `fairsight generate-report` - Generate reports from command line
- `fairsight dashboard-push` - Push results to SAP HANA dashboard

## Troubleshooting

### CLI not found after installation

If the `fairsight` command is not found:

1. Ensure you installed with `pip install -e .`
2. Check that your Python environment is in your PATH
3. Try running with `python -m fairsight.cli` instead

### Configuration file issues

If you encounter configuration file problems:

1. Delete `~/.fairsight/config.json`
2. Re-run `fairsight configure --api-key YOUR_KEY`

### Permission errors

On Unix-like systems, ensure proper permissions:

```bash
chmod 600 ~/.fairsight/config.json
```

## Examples

### Complete Setup Workflow

```bash
# 1. Install Fairsight
pip install -e .

# 2. Configure API key
fairsight configure --api-key YOUR_PREMIUM_API_KEY

# 3. Verify configuration
fairsight show-key

# 4. Check available features
fairsight list-features

# 5. Check version
fairsight version
```

### Using Premium Features

After configuring your API key, you can use premium features in your Python code:

```python
from fairsight import IllegalDataDetector, FSAuditor

# These will automatically use the configured API key
detector = IllegalDataDetector()
auditor = FSAuditor()

# No need to pass API key explicitly
result = detector.detect_illegal_content("image.jpg")
audit_result = auditor.run_audit(dataset, model)
``` 