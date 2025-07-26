"""
Fairsight Toolkit - Main Auditor
================================

This module provides the main FSAuditor class that orchestrates comprehensive
fairness and bias auditing for both datasets and machine learning models.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union
import logging
from datetime import datetime
import json
import warnings
from .auth import verify, APIKeyVerificationError, require_premium_access, TieredAccessError

logger = logging.getLogger(__name__)

class FSAuditor:
    """
    Fairsight Main Auditor - Comprehensive AI Ethics and Fairness Auditing.

    Features:
    - Unified interface for dataset and model auditing
    - Support for justified attributes
    - Comprehensive reporting
    - SAP HANA Cloud integration
    - Automated recommendations
    """

    def __init__(self, 
                 dataset: Optional[Union[str, pd.DataFrame]] = None,
                 model: Optional[Any] = None,
                 X_test: Optional[pd.DataFrame] = None,
                 y_test: Optional[Union[pd.Series, np.ndarray]] = None,
                 sensitive_features: Optional[List[str]] = None,
                 target: Optional[str] = None,
                 justified_attributes: Optional[List[str]] = None,
                 privileged_groups: Optional[Dict[str, Any]] = None,
                 fairness_threshold: float = 0.8,
                 enable_sap_integration: bool = True,
                 user_api_key: Optional[str] = None,
                 api_base_url: str = "http://localhost:5000"):
        """
        Initialize FSAuditor.

        Args:
            dataset: Dataset path or DataFrame
            model: Trained model for auditing
            X_test: Test features (if separate from dataset)
            y_test: Test targets (if separate from dataset)  
            sensitive_features: List of sensitive/protected attributes
            target: Target column name
            justified_attributes: Attributes justified for discrimination
            privileged_groups: Dict mapping attributes to privileged values
            fairness_threshold: Threshold for fairness metrics (default 0.8)
            enable_sap_integration: Whether to enable SAP HANA integration
        """
        # Core parameters
        self.dataset = dataset
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.sensitive_features = sensitive_features or []
        self.target = target
        self.justified_attributes = justified_attributes or []
        self.privileged_groups = privileged_groups or {}
        self.fairness_threshold = fairness_threshold
        self.enable_sap_integration = enable_sap_integration
        self.user_api_key = user_api_key
        self.api_base_url = api_base_url

        # Results storage
        self.audit_results = {}
        self.session_id = None

        # Validate inputs
        self._validate_initialization()

        logger.info("🚀 FSAuditor initialized successfully")
        logger.info(f"📋 Sensitive features: {self.sensitive_features}")
        logger.info(f"📋 Justified attributes: {self.justified_attributes}")

    def _validate_initialization(self):
        """Validate initialization parameters."""
        if self.dataset is None and (self.X_test is None or self.y_test is None):
            raise ValueError("Either dataset or (X_test, y_test) must be provided")

        if not self.sensitive_features:
            warnings.warn("No sensitive features provided. Bias detection will be limited.")

        if self.model is None and self.dataset is None:
            raise ValueError("At least one of dataset or model must be provided")

    def set_justified_attributes(self, attributes: List[str]):
        """
        Set attributes that are justified for discrimination.

        Args:
            attributes: List of justified attribute names
        """
        self.justified_attributes = attributes
        logger.info(f"📋 Updated justified attributes: {attributes}")

    def run_dataset_audit(self) -> Dict[str, Any]:
        """
        Run comprehensive dataset audit.

        Returns:
            Dataset audit results
        """
        if self.dataset is None:
            logger.warning("⚠️ No dataset provided, skipping dataset audit")
            return {}

        # Basic dataset audit is free - no API key required
        logger.info("📊 Running dataset audit...")

        try:
            from .dataset_audit import DatasetAuditor

            dataset_auditor = DatasetAuditor(
                dataset=self.dataset,
                protected_attributes=self.sensitive_features,
                target_column=self.target,
                justified_attributes=self.justified_attributes
            )

            dataset_results = dataset_auditor.audit()
            logger.info("✅ Dataset audit completed")
            return dataset_results

        except Exception as e:
            logger.error(f"❌ Dataset audit failed: {e}")
            return {'error': str(e)}

    def run_model_audit(self) -> Dict[str, Any]:
        """
        Run comprehensive model audit.

        Returns:
            Model audit results
        """
        if self.model is None:
            logger.warning("⚠️ No model provided, skipping model audit")
            return {}

        # Basic model audit is free - no API key required
        logger.info("🤖 Running model audit...")

        try:
            from .model_audit import ModelAuditor

            model_auditor = ModelAuditor(
                model=self.model,
                dataset=self.dataset,
                X_test=self.X_test,
                y_test=self.y_test,
                protected_attributes=self.sensitive_features,
                target_column=self.target,
                justified_attributes=self.justified_attributes
            )

            model_results = model_auditor.audit()
            logger.info("✅ Model audit completed")
            return model_results

        except Exception as e:
            logger.error(f"❌ Model audit failed: {e}")
            return {'error': str(e)}

    def run_comprehensive_bias_detection(self) -> Dict[str, Any]:
        """
        Run comprehensive bias detection across dataset and model.

        Returns:
            Combined bias detection results
        """
        logger.info("🔍 Running comprehensive bias detection...")

        try:
            from .bias_detection import BiasDetector

            # Initialize bias detector
            detector = BiasDetector(
                dataset=self.dataset,
                model=self.model,
                sensitive_features=self.sensitive_features,
                target=self.target,
                privileged_values=self.privileged_groups,
                justified_attributes=self.justified_attributes,
                threshold=self.fairness_threshold
            )

            # Run detection
            bias_results = detector.detect()

            # Generate summary report
            summary_report = detector.get_summary_report(bias_results)

            return {
                'detailed_results': [r.to_dict() for r in bias_results],
                'summary': summary_report
            }

        except Exception as e:
            logger.error(f"❌ Comprehensive bias detection failed: {e}")
            return {'error': str(e)}

    def generate_report(self, results: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate comprehensive audit report.

        Args:
            results: Audit results (uses self.audit_results if None)

        Returns:
            Generated report
        """
        # Advanced reporting is a premium feature - requires API key
        try:
            require_premium_access("advanced_reporting", self.user_api_key, self.api_base_url)
        except TieredAccessError as e:
            logger.error(f"❌ {e}")
            raise

        logger.info("📝 Generating comprehensive report...")

        if results is None:
            results = self.audit_results

        try:
            from .report_generator import Report

            report_generator = Report(results)
            report = report_generator.generate()

            logger.info("✅ Report generation completed")
            return report

        except Exception as e:
            logger.error(f"❌ Report generation failed: {e}")
            return {'error': str(e), 'results': results}

    def push_to_dashboard(self, results: Optional[Dict[str, Any]] = None, connection_params: Optional[Dict[str, str]] = None) -> Optional[str]:
        """
        Push audit results to SAP HANA Cloud dashboard.

        Args:
            results: Results to push (uses self.audit_results if None)
            connection_params: Dictionary with SAP HANA connection details (required)

        Returns:
            Session ID if successful, None if failed
        """
        # Dashboard integration is a premium feature - requires API key
        try:
            require_premium_access("dashboard_integration", self.user_api_key, self.api_base_url)
        except TieredAccessError as e:
            logger.error(f"❌ {e}")
            raise

        if not self.enable_sap_integration:
            logger.info("📊 SAP integration disabled, skipping dashboard push")
            return None

        if results is None:
            results = self.audit_results

        if not connection_params or not all(k in connection_params for k in ["host", "port", "user", "password"]):
            raise ValueError("connection_params must be provided with keys: host, port, user, password for dashboard push")

        logger.info("📊 Pushing results to SAP HANA Cloud...")

        try:
            from .dashboard_push import Dashboard

            with Dashboard(connection_params=connection_params) as dashboard:
                session_id = dashboard.push(results)
                self.session_id = session_id
                logger.info(f"✅ Successfully pushed to dashboard. Session ID: {session_id}")
                return session_id

        except Exception as e:
            logger.error(f"❌ Dashboard push failed: {e}")
            return None

    def get_audit_history(self, limit: int = 10, connection_params: Optional[Dict[str, str]] = None) -> Optional[pd.DataFrame]:
        """
        Get audit history from SAP HANA Cloud.

        Args:
            limit: Number of recent audits to retrieve
            connection_params: Dictionary with SAP HANA connection details (required)

        Returns:
            DataFrame with audit history or None if failed
        """
        # SAP HANA integration is a premium feature - requires API key
        try:
            require_premium_access("sap_hana_integration", self.user_api_key, self.api_base_url)
        except TieredAccessError as e:
            logger.error(f"❌ {e}")
            raise

        if not self.enable_sap_integration:
            logger.info("📊 SAP integration disabled")
            return None

        if not connection_params or not all(k in connection_params for k in ["host", "port", "user", "password"]):
            raise ValueError("connection_params must be provided with keys: host, port, user, password for audit history retrieval")

        try:
            from .dashboard_push import Dashboard

            with Dashboard(connection_params=connection_params) as dashboard:
                history = dashboard.get_audit_history(limit)
                logger.info(f"✅ Retrieved {len(history)} audit records")
                return history

        except Exception as e:
            logger.error(f"❌ Failed to retrieve audit history: {e}")
            return None

    def run_audit(self, 
                  include_dataset: bool = True,
                  include_model: bool = True,
                  include_bias_detection: bool = True,
                  generate_report: bool = True,
                  push_to_dashboard: bool = True,
                  connection_params: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Run comprehensive audit with all components.

        Args:
            include_dataset: Whether to include dataset audit
            include_model: Whether to include model audit
            include_bias_detection: Whether to include bias detection
            generate_report: Whether to generate report
            push_to_dashboard: Whether to push to dashboard
            connection_params: Dictionary with SAP HANA connection details (required if push_to_dashboard is True)

        Returns:
            Complete audit results
        """
        # Comprehensive audit is a premium feature - requires API key
        try:
            require_premium_access("comprehensive_audit", self.user_api_key, self.api_base_url)
        except TieredAccessError as e:
            logger.error(f"❌ {e}")
            raise

        logger.info("🚀 Starting comprehensive Fairsight audit...")

        audit_results = {
            'audit_metadata': {
                'timestamp': datetime.now().isoformat(),
                'fairsight_version': '1.0.0',
                'dataset_provided': self.dataset is not None,
                'model_provided': self.model is not None,
                'sensitive_features': self.sensitive_features,
                'justified_attributes': self.justified_attributes,
                'fairness_threshold': self.fairness_threshold
            }
        }

        # Dataset audit
        if include_dataset and self.dataset is not None:
            dataset_results = self.run_dataset_audit()
            audit_results['dataset_audit'] = dataset_results

        # Model audit
        if include_model and self.model is not None:
            model_results = self.run_model_audit()
            audit_results['model_audit'] = model_results

        # Comprehensive bias detection
        if include_bias_detection:
            bias_results = self.run_comprehensive_bias_detection()
            audit_results['bias_detection'] = bias_results

        # Calculate overall ethical score
        audit_results['ethical_score'] = self._calculate_overall_ethical_score(audit_results)

        # Generate executive summary
        audit_results['executive_summary'] = self._generate_executive_summary(audit_results)

        # Store results
        self.audit_results = audit_results

        # Generate report
        if generate_report:
            report = self.generate_report(audit_results)
            audit_results['generated_report'] = report

        # Push to dashboard
        if push_to_dashboard:
            session_id = self.push_to_dashboard(audit_results, connection_params=connection_params)
            if session_id:
                audit_results['session_id'] = session_id

        logger.info("🎉 Comprehensive Fairsight audit completed!")
        return audit_results

    def _calculate_overall_ethical_score(self, results: Dict[str, Any]) -> int:
        """Calculate overall ethical score from audit results."""
        base_score = 100
        deductions = 0

        # Dataset-based deductions
        if 'dataset_audit' in results:
            dataset_bias = results['dataset_audit'].get('bias_detection', [])
            for bias_result in dataset_bias:
                if bias_result.get('biased', False) and not bias_result.get('justified', False):
                    deductions += 10 if 'disparate_impact' in bias_result.get('metric_name', '').lower() else 5

        # Model-based deductions
        if 'model_audit' in results:
            model_bias = results['model_audit'].get('bias_detection', [])
            for bias_result in model_bias:
                if bias_result.get('biased', False) and not bias_result.get('justified', False):
                    deductions += 15 if 'disparate_impact' in bias_result.get('metric_name', '').lower() else 8

        # Performance-based deductions
        if 'model_audit' in results:
            performance = results['model_audit'].get('performance_metrics', {})
            if performance.get('accuracy', 1.0) < 0.7:
                deductions += 10  # Poor performance can indicate bias

        return max(base_score - deductions, 0)

    def _generate_executive_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate executive summary of audit results."""
        summary = {
            'overall_assessment': 'GOOD',
            'key_findings': [],
            'critical_issues': [],
            'recommendations': [],
            'ethical_score': results.get('ethical_score', 0)
        }

        ethical_score = summary['ethical_score']

        # Overall assessment
        if ethical_score >= 80:
            summary['overall_assessment'] = 'EXCELLENT'
        elif ethical_score >= 60:
            summary['overall_assessment'] = 'GOOD'
        elif ethical_score >= 40:
            summary['overall_assessment'] = 'CONCERNING'
        else:
            summary['overall_assessment'] = 'CRITICAL'

        # Key findings
        if 'dataset_audit' in results:
            dataset_results = results['dataset_audit']
            if dataset_results.get('data_quality', {}).get('class_imbalance', {}).get('imbalanced', False):
                summary['key_findings'].append("Class imbalance detected in dataset")

        if 'model_audit' in results:
            model_results = results['model_audit']
            performance = model_results.get('performance_metrics', {})
            if performance.get('accuracy', 0) > 0.9:
                summary['key_findings'].append("High model performance achieved")

        # Critical issues
        if 'bias_detection' in results:
            bias_summary = results['bias_detection'].get('summary', {})
            biased_count = bias_summary.get('biased_metrics', 0)
            if biased_count > 0:
                summary['critical_issues'].append(f"{biased_count} bias issues detected")

        # Recommendations
        if summary['ethical_score'] < 60:
            summary['recommendations'].append("Immediate bias mitigation required")

        if self.justified_attributes:
            summary['recommendations'].append(
                f"Justified attributes ({', '.join(self.justified_attributes)}) "
                f"are excluded from bias concerns"
            )

        return summary

# Legacy compatibility class
class Auditor(FSAuditor):
    """Legacy class name for backward compatibility."""
    pass

# Convenience function
def audit_complete_pipeline(dataset: Union[str, pd.DataFrame],
                           model: Any,
                           sensitive_features: List[str],
                           target_column: str,
                           justified_attributes: Optional[List[str]] = None,
                           **kwargs) -> Dict[str, Any]:
    """
    Convenience function to audit complete ML pipeline.

    Args:
        dataset: Dataset path or DataFrame
        model: Trained model
        sensitive_features: List of sensitive attributes
        target_column: Target column name
        justified_attributes: Justified attributes list
        **kwargs: Additional arguments for FSAuditor

    Returns:
        Complete audit results
    """
    auditor = FSAuditor(
        dataset=dataset,
        model=model,
        sensitive_features=sensitive_features,
        target=target_column,
        justified_attributes=justified_attributes,
        **kwargs
    )

    return auditor.run_audit()
