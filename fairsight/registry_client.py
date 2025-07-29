"""
Fairsight Registry Client
========================

Client for submitting Fairsight audit results to the registry.
"""

import requests
import json
import logging
import numpy as np
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle numpy data types and custom objects."""
    
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif hasattr(obj, 'to_dict'):
            return obj.to_dict()
        return super().default(obj)
    
    def encode(self, obj):
        # Handle Infinity and NaN values
        if isinstance(obj, float):
            if obj == float('inf'):
                return 'null'
            elif obj == float('-inf'):
                return 'null'
            elif obj != obj:  # NaN
                return 'null'
        return super().encode(obj)

def load_api_key() -> Optional[str]:
    """Load API key from CLI configuration."""
    try:
        config_file = Path.home() / ".fairsight" / "config.json"
        if config_file.exists():
            with open(config_file, 'r') as f:
                config = json.load(f)
                return config.get('api_key')
    except Exception as e:
        logger.warning(f"Could not load API key from config: {e}")
    return None

def preprocess_audit_results(audit_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Preprocess audit results to ensure JSON serialization compatibility.
    
    Args:
        audit_results: Raw audit results dictionary
        
    Returns:
        Preprocessed audit results with numpy types converted to native Python types
    """
    def convert_numpy_types(obj):
        """Recursively convert numpy types to native Python types."""
        if isinstance(obj, dict):
            return {key: convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        elif isinstance(obj, np.integer):
            val = float(obj)
            # Handle Infinity and NaN
            if val == float('inf') or val == float('-inf') or val != val:
                return None
            return val
        elif isinstance(obj, np.floating):
            val = float(obj)
            # Handle Infinity and NaN
            if val == float('inf') or val == float('-inf') or val != val:
                return None
            return val
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, float):
            # Handle regular Python floats too
            if obj == float('inf') or obj == float('-inf') or obj != obj:
                return None
            return obj
        else:
            return obj
    
    return convert_numpy_types(audit_results)

class FairsightRegistryClient:
    """
    Client for submitting Fairsight audit results to the registry.
    
    This client handles communication with the fairsight-api backend
    to upload audit results, generate PDFs, and store data in Supabase.
    """
    
    def __init__(self, api_base_url: str = "http://localhost:3000", api_key: Optional[str] = None):
        """
        Initialize the registry client.
        
        Args:
            api_base_url: Base URL for the fairsight-api backend
            api_key: API key for authentication (if None, will try to load from config)
        """
        self.api_base_url = api_base_url.rstrip('/')
        self.session = requests.Session()
        
        # Load API key if not provided
        if api_key is None:
            api_key = load_api_key()
        
        # Set default headers
        headers = {
            'Content-Type': 'application/json',
            'User-Agent': 'Fairsight-Registry-Client/1.0'
        }
        
        # Add API key to headers if available
        if api_key:
            headers['x-api-key'] = api_key
            logger.info(f"🔑 Using API key: {api_key[:8]}...")
        else:
            logger.warning("⚠️ No API key provided - using default key")
        
        self.session.headers.update(headers)
        
        logger.info(f"🔗 FairsightRegistryClient initialized for {self.api_base_url}")
    
    def submit_audit_results(self, 
                           audit_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Submit comprehensive audit results to the registry.
        
        Args:
            audit_results: Complete audit results from FSAuditor
            
        Returns:
            Response from the registry API
        """
        try:
            # Preprocess audit results to handle numpy types
            processed_results = preprocess_audit_results(audit_results)
            
            # Prepare the request payload
            payload = processed_results
            
            # Submit to registry
            url = f"{self.api_base_url}/api/upload-json-report"
            
            logger.info(f"📤 Submitting audit results to registry: {url}")
            
            # Test JSON serialization first
            try:
                json_str = json.dumps(payload, cls=NumpyEncoder)
                logger.info(f"✅ JSON serialization successful, length: {len(json_str)}")
                
                # Additional cleanup for any remaining Infinity values
                json_str = json_str.replace('Infinity', 'null').replace('-Infinity', 'null').replace('NaN', 'null')
                logger.info(f"✅ JSON cleanup completed, final length: {len(json_str)}")
                
            except Exception as e:
                logger.error(f"❌ JSON serialization failed: {e}")
                return {
                    'error': True,
                    'message': f'JSON serialization error: {str(e)}'
                }
            
            # Use custom encoder for JSON serialization
            response = self.session.post(
                url, 
                data=json_str,
                headers={'Content-Type': 'application/json'},
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                logger.info(f"✅ Successfully submitted to registry: {result.get('message', 'OK')}")
                return result
            else:
                error_msg = f"Registry submission failed: {response.status_code}"
                try:
                    error_data = response.json()
                    error_msg += f" - {error_data.get('message', 'Unknown error')}"
                except:
                    error_msg += f" - {response.text}"
                
                logger.error(f"❌ {error_msg}")
                return {
                    'error': True,
                    'message': error_msg,
                    'status_code': response.status_code
                }
                
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Network error during registry submission: {e}")
            return {
                'error': True,
                'message': f'Network error: {str(e)}'
            }
        except Exception as e:
            logger.error(f"❌ Unexpected error during registry submission: {e}")
            return {
                'error': True,
                'message': f'Unexpected error: {str(e)}'
            }
    
    def test_connection(self) -> bool:
        """
        Test connection to the registry.
        
        Returns:
            True if connection is successful, False otherwise
        """
        try:
            url = f"{self.api_base_url}/api/hello"
            response = self.session.get(url, timeout=10)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"❌ Connection test failed: {e}")
            return False
    
    def generate_detailed_reasoning(self, 
                                  bias_results: List[Dict[str, Any]], 
                                  fairness_results: Dict[str, Any],
                                  dataset_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate detailed reasoning for the audit results.
        
        Args:
            bias_results: List of bias detection results
            fairness_results: Fairness analysis results
            dataset_info: Dataset information
            
        Returns:
            Detailed reasoning object
        """
        reasoning = {
            'timestamp': datetime.now().isoformat(),
            'bias_analysis': {
                'total_attributes_analyzed': len(bias_results),
                'biased_attributes': len([b for b in bias_results if b.get('biased', False)]),
                'justified_attributes': len([b for b in bias_results if b.get('justified', False)]),
                'critical_issues': len([b for b in bias_results if b.get('biased', False) and not b.get('justified', False)])
            },
            'fairness_analysis': {
                'metrics_analyzed': len(fairness_results) if fairness_results else 0,
                'overall_fairness_score': self._calculate_fairness_score(fairness_results)
            },
            'dataset_info': dataset_info or {},
            'recommendations': self._generate_recommendations(bias_results, fairness_results)
        }
        
        return reasoning
    
    def _calculate_fairness_score(self, fairness_results: Dict[str, Any]) -> float:
        """Calculate overall fairness score from fairness results."""
        if not fairness_results:
            return 0.0
        
        scores = []
        for metric_name, metric_data in fairness_results.items():
            if isinstance(metric_data, dict) and 'value' in metric_data:
                scores.append(metric_data['value'])
            elif isinstance(metric_data, (int, float)):
                scores.append(float(metric_data))
        
        return sum(scores) / len(scores) if scores else 0.0
    
    def _generate_recommendations(self, 
                                bias_results: List[Dict[str, Any]], 
                                fairness_results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on audit results."""
        recommendations = []
        
        # Analyze bias results
        critical_bias = [b for b in bias_results if b.get('biased', False) and not b.get('justified', False)]
        if critical_bias:
            recommendations.append(f"Address bias in {len(critical_bias)} critical attributes")
        
        # Analyze fairness results
        if fairness_results:
            low_fairness = [m for m in fairness_results.values() 
                          if isinstance(m, dict) and m.get('value', 1.0) < 0.8]
            if low_fairness:
                recommendations.append("Improve fairness metrics for better model performance")
        
        # General recommendations
        if not recommendations:
            recommendations.append("Model shows good fairness characteristics")
        
        return recommendations


def submit_to_registry(audit_results: Dict[str, Any],
                      api_base_url: str = "http://localhost:3000") -> Dict[str, Any]:
    """
    Convenience function to submit audit results to registry.
    
    Args:
        audit_results: Complete audit results from FSAuditor
        api_base_url: Base URL for the fairsight-api backend
        
    Returns:
        Registry response
    """
    client = FairsightRegistryClient(api_base_url)
    return client.submit_audit_results(audit_results)