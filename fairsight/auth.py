import requests
import json
from pathlib import Path
from typing import Optional, Dict, Any
from enum import Enum

class FeatureTier(Enum):
    """Enumeration of feature tiers."""
    FREE = "free"
    PREMIUM = "premium"

class APIKeyVerificationError(Exception):
    pass

class TieredAccessError(Exception):
    """Exception raised when trying to access premium features without valid API key."""
    pass


def _load_api_key_from_config() -> Optional[str]:
    """Load API key from the configuration file."""
    try:
        home = Path.home()
        config_file = home / ".fairsight" / "config.json"
        
        if config_file.exists():
            with open(config_file, 'r') as f:
                config = json.load(f)
                return config.get('api_key')
    except (json.JSONDecodeError, IOError, KeyError):
        pass
    
    return None

# Define which features require premium access
PREMIUM_FEATURES = {
    "comprehensive_audit": FeatureTier.PREMIUM,  # FSAuditor.run_audit()
    "advanced_reporting": FeatureTier.PREMIUM,   # ReportGenerator
    "dashboard_integration": FeatureTier.PREMIUM, # Dashboard.push_to_dashboard()
    "sap_hana_integration": FeatureTier.PREMIUM, # SAP HANA features
    "illegal_data_detection": FeatureTier.PREMIUM, # IllegalDataDetector
    "advanced_explainability": FeatureTier.PREMIUM, # Advanced SHAP/LIME features
}

# Define which features are available for free
FREE_FEATURES = {
    "basic_dataset_audit": FeatureTier.FREE,     # DatasetAuditor.audit()
    "basic_model_audit": FeatureTier.FREE,       # ModelAuditor.audit()
    "basic_bias_detection": FeatureTier.FREE,    # Basic bias detection functions
    "basic_fairness_metrics": FeatureTier.FREE,  # Basic fairness metrics
    "utility_functions": FeatureTier.FREE,       # Utils class methods
    "data_preprocessing": FeatureTier.FREE,      # preprocess_data, calculate_privilege_groups
    "data_fingerprinting": FeatureTier.FREE,     # DataFingerprintEngine
}

def verify(api_key: str, base_url="http://localhost:5000") -> bool:
    """
    Verifies the API key by calling the backend verification endpoint.
    Raises APIKeyVerificationError if invalid or on error.
    Returns True if valid, otherwise raises.

    Example:
        from fairsight.auth import verify, APIKeyVerificationError
        try:
            if verify("my_api_key", base_url="http://localhost:5000"):
                print("API key is valid!")
        except APIKeyVerificationError as e:
            print(f"API key error: {e}")
    """
    try:
        url = f"{base_url}/verify_key"
        resp = requests.get(url, params={"api_key": api_key}, timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            if data.get("valid"):
                return True
            else:
                raise APIKeyVerificationError(data.get("reason", "Invalid API key"))
        else:
            try:
                data = resp.json()
                reason = data.get("reason", "API key verification failed")
            except Exception:
                reason = f"API key verification failed with status {resp.status_code}"
            raise APIKeyVerificationError(reason)
    except requests.RequestException as e:
        raise APIKeyVerificationError(f"API key verification request failed: {e}")

def require_premium_access(feature_name: str, api_key: Optional[str] = None, 
                          api_base_url: str = "http://localhost:5000") -> bool:
    """
    Check if a feature requires premium access and verify API key if needed.
    
    Args:
        feature_name: Name of the feature being accessed
        api_key: API key for premium features (if None, will try to load from config)
        api_base_url: Base URL for API verification
        
    Returns:
        bool: True if access is granted
        
    Raises:
        TieredAccessError: If premium feature is accessed without valid API key
        APIKeyVerificationError: If API key verification fails
    """
    # Check if feature is premium
    if feature_name in PREMIUM_FEATURES:
        # If no API key provided, try to load from config
        if not api_key:
            api_key = _load_api_key_from_config()
        
        if not api_key:
            raise TieredAccessError(
                f"Feature '{feature_name}' requires premium access. "
                f"Please configure your API key using: fairsight configure --api-key YOUR_KEY"
            )
        
        try:
            if not verify(api_key, base_url=api_base_url):
                raise TieredAccessError(f"Invalid API key for premium feature '{feature_name}'")
        except APIKeyVerificationError as e:
            raise TieredAccessError(f"API key verification failed for '{feature_name}': {e}")
    
    return True

def is_premium_feature(feature_name: str) -> bool:
    """
    Check if a feature requires premium access.
    
    Args:
        feature_name: Name of the feature
        
    Returns:
        bool: True if feature requires premium access
    """
    return feature_name in PREMIUM_FEATURES

def get_feature_tier(feature_name: str) -> FeatureTier:
    """
    Get the tier level for a specific feature.
    
    Args:
        feature_name: Name of the feature
        
    Returns:
        FeatureTier: The tier level of the feature
    """
    if feature_name in PREMIUM_FEATURES:
        return FeatureTier.PREMIUM
    elif feature_name in FREE_FEATURES:
        return FeatureTier.FREE
    else:
        # Default to free for unknown features
        return FeatureTier.FREE

def list_premium_features() -> Dict[str, str]:
    """
    Get a list of all premium features.
    
    Returns:
        Dict[str, str]: Dictionary mapping feature names to descriptions
    """
    premium_descriptions = {
        "comprehensive_audit": "Complete Fairsight audit with all components",
        "advanced_reporting": "Advanced report generation with custom templates",
        "dashboard_integration": "SAP HANA Cloud dashboard integration",
        "sap_hana_integration": "Full SAP HANA Cloud connectivity",
        "illegal_data_detection": "AI-powered illegal data detection",
        "advanced_explainability": "Advanced SHAP and LIME explainability features"
    }
    
    return {feature: premium_descriptions.get(feature, "Premium feature") 
            for feature in PREMIUM_FEATURES.keys()}

def list_free_features() -> Dict[str, str]:
    """
    Get a list of all free features.
    
    Returns:
        Dict[str, str]: Dictionary mapping feature names to descriptions
    """
    free_descriptions = {
        "basic_dataset_audit": "Basic dataset bias and fairness analysis",
        "basic_model_audit": "Basic model performance and bias evaluation",
        "basic_bias_detection": "Core bias detection algorithms",
        "basic_fairness_metrics": "Essential fairness metrics computation",
        "utility_functions": "Data preprocessing and utility functions",
        "data_preprocessing": "Data cleaning and preprocessing tools",
        "data_fingerprinting": "Data fingerprinting and duplicate detection"
    }
    
    return {feature: free_descriptions.get(feature, "Free feature") 
            for feature in FREE_FEATURES.keys()}
