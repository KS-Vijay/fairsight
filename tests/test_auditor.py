import pandas as pd
import numpy as np
from fairsight import FSAuditor, Auditor, TieredAccessError, APIKeyVerificationError

# def test_fsauditor_basic_features_free():
#     """Test that basic FSAuditor features work without API key (FREE)."""
#     print("\n🆓 Testing Basic FSAuditor Features (FREE)...")
    
#     # Create sample data
#     df = pd.DataFrame({
#         'category': ['A', 'B', 'A', 'B', 'A', 'B'],
#         'intent': [1, 0, 1, 0, 1, 0],
#         'feature': [10, 20, 10, 30, 15, 25],
#         'gender': ['M', 'F', 'M', 'F', 'M', 'F']
#     })
    
#     # Test without API key - basic features should work
#     try:
#         auditor = FSAuditor(
#             dataset=df, 
#             sensitive_features=['category', 'gender'], 
#             target='intent'
#         )
        
#         # Test basic dataset audit (FREE)
#         dataset_results = auditor.run_dataset_audit()
#         print("✅ Basic dataset audit works without API key")
#         assert isinstance(dataset_results, dict)
        
#         # Test basic model audit (FREE) - if model is provided
#         # Note: This would require a model to be passed to FSAuditor
#         print("✅ Basic FSAuditor features work without API key")
        
#     except Exception as e:
#         print(f"❌ Basic features failed: {e}")
#         raise

def test_fsauditor_premium_features_without_key():
    """Test that premium FSAuditor features require API key."""
    print("\n🔒 Testing Premium FSAuditor Features (API Key Required)...")
    
    # Create sample data
    df = pd.DataFrame({
        'category': ['A', 'B', 'A', 'B', 'A', 'B'],
        'intent': [1, 0, 1, 0, 1, 0],
        'feature': [10, 20, 10, 30, 15, 25],
        'gender': ['M', 'F', 'M', 'F', 'M', 'F']
    })
    
    # Test without API key - premium features should fail
    auditor = FSAuditor(
        dataset=df, 
        sensitive_features=['category', 'gender'], 
        target='intent'
    )
    
    # Test comprehensive audit (PREMIUM)
    try:
        results = auditor.run_audit()
        print("❌ Comprehensive audit should have failed without API key")
    except TieredAccessError as e:
        print(f"✅ Comprehensive audit correctly blocked: {e}")
    
    # Test advanced reporting (PREMIUM)
    try:
        report = auditor.generate_report()
        print("❌ Advanced reporting should have failed without API key")
    except TieredAccessError as e:
        print(f"✅ Advanced reporting correctly blocked: {e}")
    
    # Test dashboard integration (PREMIUM)
    try:
        session_id = auditor.push_to_dashboard({})
        print("❌ Dashboard integration should have failed without API key")
    except TieredAccessError as e:
        print(f"✅ Dashboard integration correctly blocked: {e}")

def test_fsauditor_premium_features_with_key():
    """Test that premium FSAuditor features work with valid API key."""
    print("\n🔑 Testing Premium FSAuditor Features with API Key...")
    
    # Placeholder API key - replace with your actual API key for testing
    TEST_API_KEY = "c910089f-1037-44e1-a3a0-8192a6ded2f9"  # TODO: Replace with actual API key
    
    # Create sample data
    df = pd.DataFrame({
        'category': ['A', 'B', 'A', 'B', 'A', 'B'],
        'intent': [1, 0, 1, 0, 1, 0],
        'feature': [10, 20, 10, 30, 15, 25],
        'gender': ['M', 'F', 'M', 'F', 'M', 'F']
    })
    
    # Test with API key
    try:
        auditor = FSAuditor(
            dataset=df, 
            sensitive_features=['category', 'gender'], 
            target='intent',
            user_api_key=TEST_API_KEY
        )
        
        # Test comprehensive audit with API key
        try:
            results = auditor.run_audit(
                include_dataset=True,
                include_model=False,  # No model provided
                include_bias_detection=True,
                generate_report=False,  # Test separately
                push_to_dashboard=False  # Test separately
            )
            print("✅ Comprehensive audit works with API key")
            assert isinstance(results, dict)
            print(f"   Ethical score: {results.get('ethical_score', 'N/A')}")
            print(f"   Executive summary: {results.get('executive_summary', {}).get('overall_assessment', 'N/A')}")
            
        except (TieredAccessError, APIKeyVerificationError) as e:
            print(f"⚠️ Comprehensive audit failed (expected if using placeholder): {e}")
        
        # Test advanced reporting with API key
        try:
            report = auditor.generate_report()
            print("✅ Advanced reporting works with API key")
            assert isinstance(report, dict)
            
        except (TieredAccessError, APIKeyVerificationError) as e:
            print(f"⚠️ Advanced reporting failed (expected if using placeholder): {e}")
        
        # Test dashboard integration with API key (requires connection params)
        try:
            # This will fail due to missing connection params, but should pass API key check
            auditor.push_to_dashboard({}, connection_params={})
            print("❌ This should not work without connection params")
            
        except (TieredAccessError, APIKeyVerificationError) as e:
            print(f"⚠️ Dashboard integration failed (expected if using placeholder): {e}")
        except ValueError as e:
            if "connection_params" in str(e):
                print("✅ Dashboard integration API key check passed (connection params error expected)")
            else:
                print(f"❌ Unexpected error: {e}")
        
    except Exception as e:
        print(f"❌ FSAuditor initialization failed: {e}")

# def test_fsauditor_with_model():
#     """Test FSAuditor with a model for comprehensive testing."""
#     print("\n🤖 Testing FSAuditor with Model...")
    
#     # Placeholder API key - replace with your actual API key for testing 
    
#     # Create sample data
#     df = pd.DataFrame({
#         'category': ['A', 'B', 'A', 'B', 'A', 'B'],
#         'intent': [1, 0, 1, 0, 1, 0],
#         'feature': [10, 20, 10, 30, 15, 25],
#         'gender': ['M', 'F', 'M', 'F', 'M', 'F']
#     })
    
#     try:
#         # Create a simple mock model
#         from sklearn.ensemble import RandomForestClassifier
#         from sklearn.model_selection import train_test_split
        
#         # Prepare data for model
#         X = df[['feature']].values
#         y = df['intent'].values
        
#         # Train a simple model
#         model = RandomForestClassifier(n_estimators=10, random_state=42)
#         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
#         model.fit(X_train, y_train)
        
#         # Test FSAuditor with model and API key
#         auditor = FSAuditor(
#             dataset=df,
#             model=model,
#             X_test=pd.DataFrame(X_test, columns=['feature']),
#             y_test=y_test,
#             sensitive_features=['category', 'gender'],
#             target='intent'
#         )
        
#         # Test comprehensive audit with model
#         try:
#             results = auditor.run_audit(
#                 include_dataset=True,
#                 include_model=True,
#                 include_bias_detection=True,
#                 generate_report=False,
#                 push_to_dashboard=False
#             )
#             print("✅ Comprehensive audit with model works with API key")
#             assert isinstance(results, dict)
#             print(f"   Ethical score: {results.get('ethical_score', 'N/A')}")
            
#         except (TieredAccessError, APIKeyVerificationError) as e:
#             print(f"⚠️ Comprehensive audit with model failed (expected if using placeholder): {e}")
        
#         # Test model audit separately
#         try:
#             model_results = auditor.run_model_audit()
#             print("✅ Model audit works (FREE feature)")
#             assert isinstance(model_results, dict)
            
#         except Exception as e:
#             print(f"❌ Model audit failed: {e}")
        
#     except Exception as e:
#         print(f"❌ Model testing failed: {e}")

# def test_auditor_legacy_class():
#     """Test the legacy Auditor class (alias for FSAuditor)."""
#     print("\n📋 Testing Legacy Auditor Class...")
    
#     # Create sample data
#     df = pd.DataFrame({
#         'category': ['A', 'B', 'A', 'B'],
#         'intent': [1, 0, 1, 0],
#         'feature': [10, 20, 10, 30]
#     })
    
#     try:
#         # Test legacy Auditor class (should work the same as FSAuditor)
#         auditor = Auditor(
#             dataset=df, 
#             sensitive_features=['category'], 
#             target='intent'
#         )
        
#         # Test basic functionality
#         dataset_results = auditor.run_dataset_audit()
#         print("✅ Legacy Auditor class works without API key")
#         assert isinstance(dataset_results, dict)
        
#     except Exception as e:
#         print(f"❌ Legacy Auditor class failed: {e}")

# def demo_fsauditor():
#     """Original demo function - now with API key support."""
#     print("\n🎯 Running Original Demo with API Key Support...")
    
#     # Placeholder API key - replace with your actual API key for testing
#     TEST_API_KEY = "c910089f-1037-44e1-a3a0-8192a6ded2f9"  # TODO: Replace with actual API key
    
#     df = pd.DataFrame({
#         'category': ['A', 'B', 'A', 'B'],
#         'intent': [1, 0, 1, 0],
#         'feature': [10, 20, 10, 30]
#     })
    
#     try:
#         # Test without API key (basic features only)
#         print("Testing without API key (basic features)...")
#         auditor = FSAuditor(
#             dataset=df, 
#             sensitive_features=['category'], 
#             target='intent'
#         )
        
#         # Basic dataset audit should work
#         dataset_results = auditor.run_dataset_audit()
#         print(f"✅ Basic dataset audit completed")
        
#         # Comprehensive audit should fail
#         try:
#             results = auditor.run_audit(generate_report=False, push_to_dashboard=False)
#             print("❌ Comprehensive audit should have failed without API key")
#         except TieredAccessError as e:
#             print(f"✅ Comprehensive audit correctly blocked: {e}")
        
#         # Test with API key
#         print("\nTesting with API key (premium features)...")
#         auditor_with_key = FSAuditor(
#             dataset=df, 
#             sensitive_features=[s'category'], 
#             target='intent',
#             user_api_key=TEST_API_KEY
#         )
        
#         try:
#             results = auditor_with_key.run_audit(generate_report=False, push_to_dashboard=False)
#             print(f"✅ Comprehensive audit with API key completed")
#             print(f"   Ethical score: {results.get('ethical_score', 'N/A')}")
#             print(f"   Executive summary: {results.get('executive_summary', {}).get('overall_assessment', 'N/A')}")
            
#         except (TieredAccessError, APIKeyVerificationError) as e:
#             print(f"⚠️ Comprehensive audit failed (expected if using placeholder): {e}")
        
#     except Exception as e:
#         print(f"❌ Demo failed: {e}")

if __name__ == '__main__':
    print("🧪 Running FSAuditor Tests with API Key Support")
    print("=" * 60)
    
    # Run all tests
    #test_fsauditor_basic_features_free()
    test_fsauditor_premium_features_without_key()
    test_fsauditor_premium_features_with_key()
    #test_fsauditor_with_model()
    #test_auditor_legacy_class()
    #demo_fsauditor()
    
    print("\n" + "=" * 60)
    print("✅ All FSAuditor tests completed!")
    print("\n💡 Note: Replace 'your_api_key_here' with actual API key to test premium features")
    print("🆓 Free features: Basic dataset audit, basic model audit")
    print("🔑 Premium features: Comprehensive audit, advanced reporting, dashboard integration")