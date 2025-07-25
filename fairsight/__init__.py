"""
Fairsight Toolkit - AI Bias Detection and Fairness Auditing Library
================================================================

A comprehensive toolkit for detecting bias and ensuring fairness in AI models and datasets.
Includes support for justified attributes, SAP HANA Cloud integration, and automated reporting.

Author: Vijay K S, Abhay Pratap Singh
Version: 1.0.0
License: MIT
"""

__version__ = "1.0.0"
__author__ = "Vijay K S, Abhay Pratap Singh"
__email__ = "ksvijay2005@gmail.com"
__license__ = "MIT"

# Core Classes
from .auditor import FSAuditor, Auditor
from .dataset_audit import DatasetAuditor
from .model_audit import ModelAuditor
from .bias_detection import BiasDetector, BiasDetectionResult
from .report_generator import ReportGenerator, Report, generate_html_report
from .dashboard_push import Dashboard #SAPHANAConnector
from .explainability import ExplainabilityEngine, ExplainabilityResult, explain_with_shap, explain_with_lime
from .fairness_metrics import FairnessEngine, FairnessMetrics
from .utils import Utils, preprocess_data, calculate_privilege_groups
from .data_fingerprint import DataFingerprintEngine, DuplicateRecord
from .illegal_data import IllegalDataDetector
from .reweighing import Reweighing

# Convenience imports for quick access
from .bias_detection import (
    detect_dataset_bias,
    detect_model_bias,
    compute_disparate_impact,
    compute_statistical_parity
)

from .fairness_metrics import (
    compute_demographic_parity,
    compute_equal_opportunity,
    compute_predictive_parity
)

# Main API Classes
__all__ = [
    # Core auditing classes
    "FSAuditor",
    "Auditor", 
    "DatasetAuditor",
    "ModelAuditor",
    
    # Bias detection
    "BiasDetector",
    "BiasDetectionResult",
    "detect_dataset_bias",
    "detect_model_bias",
    "compute_disparate_impact",
    "compute_statistical_parity",
    
    # Fairness metrics
    "FairnessEngine",
    "FairnessMetrics", 
    "compute_demographic_parity",
    "compute_equal_opportunity",
    "compute_predictive_parity",
    
    # Explainability
    "ExplainabilityEngine",
    "ExplainabilityResult",
    
    # Reporting and dashboards
    "ReportGenerator",
    "Report",
    "Dashboard",
    #"SAPHANAConnector",
    
    # Utilities
    "Utils",

    #Data Fingerprint
    "DataFingerprintEngine",
    "DuplicateRecord",

    #Illegal Data
    "IllegalDataDetector",
    # Bias mitigation
    "Reweighing",
    # Standalone utilities
    "explain_with_shap",
    "explain_with_lime",
    "preprocess_data",
    "calculate_privilege_groups",
    "generate_html_report",
]

# Package metadata
__package_info__ = {
    "name": "fairsight",
    "version": __version__,
    "description": "AI Bias Detection and Fairness Auditing Toolkit with SAP Integration",
    "author": __author__,
    "email": __email__,
    "license": __license__,
    "url": "https://github.com/KS-Vijay/fairsight",
    "keywords": ["AI", "bias", "fairness", "audit", "machine learning", "SAP"],
    "python_requires": ">=3.7",
}

def get_version():
    """Return the current version of Fairsight Toolkit."""
    return __version__

def get_info():
    """Return package information."""
    return __package_info__

# Initialize logging for the package
import logging
logging.getLogger(__name__).addHandler(logging.NullHandler())
