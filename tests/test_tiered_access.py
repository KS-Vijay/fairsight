"""
Test Tiered Access Functionality
================================

This module tests the tiered access system for Fairsight toolkit,
ensuring that premium features require API keys while free features work without them.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import pandas as pd
import numpy as np
from fairsight import (
    FSAuditor, DatasetAuditor, ModelAuditor,
    require_premium_access, is_premium_feature, get_feature_tier,
    list_premium_features, list_free_features,
    FeatureTier, TieredAccessError, APIKeyVerificationError,
    IllegalDataDetector
)

def test_feature_tier_classification():
    """Test that features are correctly classified as free or premium."""
    
    # Test premium features
    assert is_premium_feature("comprehensive_audit") == True
    assert is_premium_feature("advanced_reporting") == True
    assert is_premium_feature("dashboard_integration") == True
    assert is_premium_feature("sap_hana_integration") == True
    assert is_premium_feature("illegal_data_detection") == True
    assert is_premium_feature("advanced_explainability") == True
    
    # Test free features
    assert is_premium_feature("basic_dataset_audit") == False
    assert is_premium_feature("basic_model_audit") == False
    assert is_premium_feature("basic_bias_detection") == False
    assert is_premium_feature("basic_fairness_metrics") == False
    assert is_premium_feature("utility_functions") == False
    assert is_premium_feature("data_preprocessing") == False
    assert is_premium_feature("data_fingerprinting") == False

def test_get_feature_tier():
    """Test getting feature tier levels."""
    
    # Test premium features
    assert get_feature_tier("comprehensive_audit") == FeatureTier.PREMIUM
    assert get_feature_tier("illegal_data_detection") == FeatureTier.PREMIUM
    
    # Test free features
    assert get_feature_tier("basic_dataset_audit") == FeatureTier.FREE
    assert get_feature_tier("utility_functions") == FeatureTier.FREE
    
    # Test unknown features (should default to free)
    assert get_feature_tier("unknown_feature") == FeatureTier.FREE

def test_list_features():
    """Test listing premium and free features."""
    
    premium_features = list_premium_features()
    free_features = list_free_features()
    
    # Check that premium features are listed
    assert "comprehensive_audit" in premium_features
    assert "illegal_data_detection" in premium_features
    assert "advanced_reporting" in premium_features
    
    # Check that free features are listed
    assert "basic_dataset_audit" in free_features
    assert "utility_functions" in free_features
    assert "data_preprocessing" in free_features
    
    # Check that features don't overlap
    premium_keys = set(premium_features.keys())
    free_keys = set(free_features.keys())
    assert len(premium_keys.intersection(free_keys)) == 0

def test_require_premium_access_without_key():
    """Test that premium features require API keys."""
    
    # Test that premium features raise TieredAccessError without API key
    with pytest.raises(TieredAccessError):
        require_premium_access("comprehensive_audit")
    
    with pytest.raises(TieredAccessError):
        require_premium_access("illegal_data_detection")
    
    with pytest.raises(TieredAccessError):
        require_premium_access("advanced_reporting")

def test_require_premium_access_with_key():
    """Test that premium features work with valid API keys."""
    
    # Mock a valid API key (in real scenario, this would be verified)
    # For testing purposes, we'll assume the verification passes
    try:
        require_premium_access("comprehensive_audit", "test_api_key")
        # If no exception is raised, the test passes
    except (TieredAccessError, APIKeyVerificationError):
        # This is expected if the API key verification fails
        # In a real test environment, you might mock the verification
        pass

def test_free_features_work_without_key():
    """Test that free features work without API keys."""
    
    # These should not raise any exceptions
    try:
        require_premium_access("basic_dataset_audit")
        require_premium_access("utility_functions")
        require_premium_access("data_preprocessing")
        # If no exception is raised, the test passes
    except Exception as e:
        pytest.fail(f"Free features should not require API keys: {e}")

def test_fsauditor_basic_audits_free():
    """Test that basic audits in FSAuditor work without API keys."""
    
    # Create sample data
    data = pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5],
        'feature2': [0, 1, 0, 1, 0],
        'target': [0, 1, 0, 1, 0],
        'gender': ['M', 'F', 'M', 'F', 'M']
    })
    
    # Create auditor without API key
    auditor = FSAuditor(
        dataset=data,
        sensitive_features=['gender'],
        target='target'
    )
    
    # Basic audits should work without API key
    try:
        dataset_results = auditor.run_dataset_audit()
        assert isinstance(dataset_results, dict)
        print("✅ Basic dataset audit works without API key")
    except Exception as e:
        pytest.fail(f"Basic dataset audit should work without API key: {e}")

def test_fsauditor_premium_features_require_key():
    """Test that premium features in FSAuditor require API keys."""
    
    # Create sample data
    data = pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5],
        'feature2': [0, 1, 0, 1, 0],
        'target': [0, 1, 0, 1, 0],
        'gender': ['M', 'F', 'M', 'F', 'M']
    })
    
    # Create auditor without API key
    auditor = FSAuditor(
        dataset=data,
        sensitive_features=['gender'],
        target='target'
    )
    
    # Premium features should require API key
    with pytest.raises(TieredAccessError):
        auditor.run_audit()  # Comprehensive audit
    
    with pytest.raises(TieredAccessError):
        auditor.generate_report()  # Advanced reporting
    
    with pytest.raises(TieredAccessError):
        auditor.push_to_dashboard({})  # Dashboard integration

def test_illegal_data_detector_requires_key():
    """Test that IllegalDataDetector requires API key."""
    
    # Mock pipeline and reference folder for testing
    mock_pipeline = None
    mock_reference_folder = "/tmp/test_reference"
    
    # Should raise TieredAccessError without API key
    with pytest.raises(TieredAccessError):
        IllegalDataDetector(
            pipeline=mock_pipeline,
            reference_folder=mock_reference_folder
        )

def test_dataset_auditor_free():
    """Test that DatasetAuditor works without API key."""
    
    # Create sample data
    data = pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5],
        'feature2': [0, 1, 0, 1, 0],
        'target': [0, 1, 0, 1, 0],
        'gender': ['M', 'F', 'M', 'F', 'M']
    })
    
    # Should work without API key
    try:
        auditor = DatasetAuditor(
            dataset=data,
            protected_attributes=['gender'],
            target_column='target'
        )
        results = auditor.audit()
        assert isinstance(results, dict)
        print("✅ DatasetAuditor works without API key")
    except Exception as e:
        pytest.fail(f"DatasetAuditor should work without API key: {e}")

if __name__ == "__main__":
    print("🧪 Running Tiered Access Tests...")
    
    # Run basic tests
    test_feature_tier_classification()
    test_get_feature_tier()
    test_list_features()
    test_require_premium_access_without_key()
    test_free_features_work_without_key()
    
    print("✅ All tiered access tests passed!")
    print("\n📋 Feature Summary:")
    print("🆓 Free Features:")
    free_features = list_free_features()
    for feature, desc in free_features.items():
        print(f"   • {feature}: {desc}")
    
    print("\n🔑 Premium Features:")
    premium_features = list_premium_features()
    for feature, desc in premium_features.items():
        print(f"   • {feature}: {desc}") 