import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

#!/usr/bin/env python3
"""
Fairsight Tiered Access Demo
============================

This script demonstrates the tiered access system for Fairsight toolkit,
showing which features are free and which require premium API keys.
"""

import pandas as pd
import numpy as np
from fairsight import (
    FSAuditor, DatasetAuditor,
    require_premium_access, is_premium_feature, get_feature_tier,
    list_premium_features, list_free_features,
    FeatureTier, TieredAccessError, APIKeyVerificationError,
    IllegalDataDetector
)

def print_feature_summary():
    """Print a summary of all available features and their tiers."""
    print("🎯 FAIRSIGHT TIERED ACCESS SYSTEM")
    print("=" * 50)
    
    print("\n🆓 FREE FEATURES (No API Key Required):")
    print("-" * 40)
    free_features = list_free_features()
    for feature, desc in free_features.items():
        print(f"   • {feature}: {desc}")
    
    print("\n🔑 PREMIUM FEATURES (API Key Required):")
    print("-" * 40)
    premium_features = list_premium_features()
    for feature, desc in premium_features.items():
        print(f"   • {feature}: {desc}")

def demo_free_features():
    """Demonstrate free features that work without API keys."""
    print("\n🚀 DEMONSTRATING FREE FEATURES")
    print("=" * 40)
    
    # Create sample data
    data = pd.DataFrame({
        'age': [25, 30, 35, 40, 45, 50, 55, 60],
        'income': [30000, 45000, 60000, 75000, 90000, 105000, 120000, 135000],
        'education': ['HS', 'College', 'College', 'Masters', 'Masters', 'PhD', 'PhD', 'PhD'],
        'gender': ['M', 'F', 'M', 'F', 'M', 'F', 'M', 'F'],
        'hired': [0, 1, 1, 1, 1, 1, 1, 1]
    })
    
    print("📊 Sample dataset created with 8 records")
    print(f"   Features: {list(data.columns)}")
    print(f"   Shape: {data.shape}")
    
    # Test basic dataset audit (FREE)
    print("\n📋 Testing Basic Dataset Audit (FREE)...")
    try:
        auditor = DatasetAuditor(
            dataset=data,
            protected_attributes=['gender'],
            target_column='hired'
        )
        results = auditor.audit()
        print("✅ Basic dataset audit completed successfully!")
        print(f"   Task type: {results['dataset_info']['task_type']}")
        print(f"   Protected attributes: {results['dataset_info']['protected_attributes']}")
        print(f"   Recommendations: {len(results['recommendations'])} generated")
    except Exception as e:
        print(f"❌ Basic dataset audit failed: {e}")
    
    # Test FSAuditor basic features (FREE)
    print("\n🤖 Testing FSAuditor Basic Features (FREE)...")
    try:
        fs_auditor = FSAuditor(
            dataset=data,
            sensitive_features=['gender'],
            target='hired'
        )
        
        # Basic dataset audit
        dataset_results = fs_auditor.run_dataset_audit()
        print("✅ FSAuditor basic dataset audit completed!")
        
        # Basic model audit (if model provided)
        print("✅ FSAuditor basic features work without API key!")
        
    except Exception as e:
        print(f"❌ FSAuditor basic features failed: {e}")

def demo_premium_features():
    """Demonstrate premium features that require API keys."""
    print("\n🔑 DEMONSTRATING PREMIUM FEATURES")
    print("=" * 40)
    
    # Create sample data
    data = pd.DataFrame({
        'age': [25, 30, 35, 40, 45, 50, 55, 60],
        'income': [30000, 45000, 60000, 75000, 90000, 105000, 120000, 135000],
        'education': ['HS', 'College', 'College', 'Masters', 'Masters', 'PhD', 'PhD', 'PhD'],
        'gender': ['M', 'F', 'M', 'F', 'M', 'F', 'M', 'F'],
        'hired': [0, 1, 1, 1, 1, 1, 1, 1]
    })
    
    # Test comprehensive audit (PREMIUM)
    print("\n📊 Testing Comprehensive Audit (PREMIUM)...")
    try:
        fs_auditor = FSAuditor(
            dataset=data,
            sensitive_features=['gender'],
            target='hired'
        )
        results = fs_auditor.run_audit()
        print("❌ This should not work without API key!")
    except TieredAccessError as e:
        print(f"✅ Correctly blocked: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
    
    # Test advanced reporting (PREMIUM)
    print("\n📝 Testing Advanced Reporting (PREMIUM)...")
    try:
        fs_auditor = FSAuditor(
            dataset=data,
            sensitive_features=['gender'],
            target='hired'
        )
        report = fs_auditor.generate_report()
        print("❌ This should not work without API key!")
    except TieredAccessError as e:
        print(f"✅ Correctly blocked: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
    
    # Test dashboard integration (PREMIUM)
    print("\n📊 Testing Dashboard Integration (PREMIUM)...")
    try:
        fs_auditor = FSAuditor(
            dataset=data,
            sensitive_features=['gender'],
            target='hired'
        )
        session_id = fs_auditor.push_to_dashboard({})
        print("❌ This should not work without API key!")
    except TieredAccessError as e:
        print(f"✅ Correctly blocked: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
    
    # Test illegal data detector (PREMIUM)
    print("\n🚨 Testing Illegal Data Detector (PREMIUM)...")
    try:
        # Mock pipeline and reference folder
        mock_pipeline = None
        mock_reference_folder = "/tmp/test_reference"
        
        detector = IllegalDataDetector(
            pipeline=mock_pipeline,
            reference_folder=mock_reference_folder
        )
        print("❌ This should not work without API key!")
    except TieredAccessError as e:
        print(f"✅ Correctly blocked: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

def demo_api_key_verification():
    """Demonstrate API key verification system."""
    print("\n🔐 DEMONSTRATING API KEY VERIFICATION")
    print("=" * 40)
    
    # Test feature tier checking
    print("\n📋 Feature Tier Checking:")
    features_to_test = [
        "comprehensive_audit",
        "basic_dataset_audit", 
        "illegal_data_detection",
        "utility_functions"
    ]
    
    for feature in features_to_test:
        tier = get_feature_tier(feature)
        is_premium = is_premium_feature(feature)
        print(f"   • {feature}: {tier.value.upper()} ({'🔑' if is_premium else '🆓'})")
    
    # Test require_premium_access function
    print("\n🔍 Testing Premium Access Requirements:")
    
    # Test free features
    free_features = ["basic_dataset_audit", "utility_functions"]
    for feature in free_features:
        try:
            require_premium_access(feature)
            print(f"   ✅ {feature}: Access granted (FREE)")
        except Exception as e:
            print(f"   ❌ {feature}: Unexpected error - {e}")
    
    # Test premium features
    premium_features = ["comprehensive_audit", "illegal_data_detection"]
    for feature in premium_features:
        try:
            require_premium_access(feature)
            print(f"   ❌ {feature}: Should have been blocked!")
        except TieredAccessError as e:
            print(f"   ✅ {feature}: Correctly blocked - {e}")

def main():
    """Main demo function."""
    print_feature_summary()
    demo_free_features()
    demo_premium_features()
    demo_api_key_verification()
    
    print("\n🎉 TIERED ACCESS DEMO COMPLETED!")
    print("=" * 50)
    print("\n💡 Key Takeaways:")
    print("   • Free features work without API keys")
    print("   • Premium features require valid API keys")
    print("   • Clear error messages guide users")
    print("   • Easy to check feature tiers programmatically")
    
    print("\n📚 Usage Examples:")
    print("   • Basic auditing: No API key needed")
    print("   • Advanced features: Provide user_api_key parameter")
    print("   • Check tiers: Use is_premium_feature() or get_feature_tier()")

if __name__ == "__main__":
    main() 