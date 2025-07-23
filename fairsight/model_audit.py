"""
Fairsight Toolkit - Model Audit
===============================

This module provides comprehensive auditing capabilities for machine learning models,
including bias detection, fairness analysis, explainability, and performance evaluation.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                           roc_auc_score, mean_squared_error, mean_absolute_error, r2_score)
from sklearn.base import BaseEstimator
from typing import Dict, Any, List, Optional, Union, Tuple
import logging
import warnings
from .utils import Utils

logger = logging.getLogger(__name__)

class ModelAuditor:
    """
    Comprehensive model auditing class.

    Features:
    - Model performance evaluation
    - Bias detection in predictions
    - Fairness metrics computation
    - Explainability analysis
    - Comprehensive reporting
    """

    def __init__(self, 
                 model: BaseEstimator,
                 dataset: Union[str, pd.DataFrame, None] = None,
                 X_test: Optional[pd.DataFrame] = None,
                 y_test: Optional[Union[pd.Series, np.ndarray]] = None,
                 protected_attributes: Optional[List[str]] = None,
                 target_column: Optional[str] = None,
                 justified_attributes: Optional[List[str]] = None,
                 task_type: Optional[str] = None):
        """
        Initialize ModelAuditor.

        Args:
            model: Trained model to audit
            dataset: Full dataset (will be split if X_test/y_test not provided)
            X_test: Test features
            y_test: Test targets
            protected_attributes: List of protected/sensitive attributes
            target_column: Target column name
            justified_attributes: Attributes justified for discrimination
            task_type: 'classification' or 'regression' (auto-detected if None)
        """
        self.model = model
        self.protected_attributes = protected_attributes or []
        self.justified_attributes = justified_attributes or []
        self.target_column = target_column
        self.task_type = task_type

        # Data setup
        self._setup_data(dataset, X_test, y_test)

        # Model validation
        self._validate_model()

        # Auto-detect task type if not provided
        if not self.task_type:
            self.task_type = Utils.is_classification_task(self.y_test)
            self.task_type = 'classification' if self.task_type else 'regression'

        logger.info(f"🤖 ModelAuditor initialized for {self.task_type} task")
        logger.info(f"📋 Justified attributes: {self.justified_attributes}")

    def _setup_data(self, dataset: Union[str, pd.DataFrame, None], 
                   X_test: Optional[pd.DataFrame], 
                   y_test: Optional[Union[pd.Series, np.ndarray]]):
        """Setup test data for model auditing."""

        if X_test is not None and y_test is not None:
            # Use provided test data
            self.X_test = X_test.copy() if isinstance(X_test, pd.DataFrame) else pd.DataFrame(X_test)
            self.y_test = y_test.copy() if hasattr(y_test, 'copy') else y_test
            logger.info(f"📊 Using provided test data: {self.X_test.shape}")

        elif dataset is not None:
            # Load dataset and split
            if isinstance(dataset, str):
                df = pd.read_csv(dataset)
            else:
                df = dataset.copy()

            # Infer target column if not provided
            if not self.target_column:
                self.target_column = Utils.infer_target_column(df)

            # Split data (use last 20% as test set)
            test_size = int(0.2 * len(df))
            df_test = df.tail(test_size)

            self.X_test = df_test.drop(columns=[self.target_column])
            self.y_test = df_test[self.target_column]

            logger.info(f"📊 Split dataset for testing: {self.X_test.shape}")

        else:
            raise ValueError("Either provide (X_test, y_test) or dataset")

        # Ensure protected attributes are in test data
        missing_attrs = [attr for attr in self.protected_attributes 
                        if attr not in self.X_test.columns]
        if missing_attrs:
            warnings.warn(f"Protected attributes not found in test data: {missing_attrs}")
            self.protected_attributes = [attr for attr in self.protected_attributes 
                                       if attr in self.X_test.columns]

    def _validate_model(self):
        """Validate that the model has required methods."""
        if not hasattr(self.model, 'predict'):
            raise ValueError("Model must have a 'predict' method")

        # Check if model is fitted by trying a prediction
        try:
            sample_input = self.X_test.iloc[:1] if hasattr(self.X_test, 'iloc') else self.X_test[:1]
            self.model.predict(sample_input)
            logger.info("✅ Model validation passed")
        except Exception as e:
            raise ValueError(f"Model appears to be unfitted or incompatible: {e}")

    def evaluate_performance(self) -> Dict[str, float]:
        """
        Evaluate model performance on test data.

        Returns:
            Dictionary with performance metrics
        """
        logger.info("📊 Evaluating model performance...")

        try:
            # Get predictions
            y_pred = self.model.predict(self.X_test)

            if self.task_type == 'classification':
                metrics = self._classification_metrics(self.y_test, y_pred)

                # Add probability-based metrics if available
                if hasattr(self.model, 'predict_proba'):
                    y_proba = self.model.predict_proba(self.X_test)
                    if y_proba.shape[1] == 2:  # Binary classification
                        metrics['roc_auc'] = roc_auc_score(self.y_test, y_proba[:, 1])

            else:  # regression
                metrics = self._regression_metrics(self.y_test, y_pred)

            logger.info(f"✅ Performance evaluation complete")
            return metrics

        except Exception as e:
            logger.error(f"❌ Performance evaluation failed: {e}")
            return {}

    def _classification_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Compute classification metrics."""
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(y_true, y_pred, average='weighted', zero_division=0)
        }

    def _regression_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Compute regression metrics."""
        return {
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2_score': r2_score(y_true, y_pred)
        }

    def detect_bias(self) -> List[Dict[str, Any]]:
        """
        Detect bias in model predictions.

        Returns:
            List of bias detection results
        """
        logger.info("🔍 Running bias detection on model predictions...")

        try:
            from .bias_detection import BiasDetector

            # Get model predictions
            y_pred = self.model.predict(self.X_test)

            # Create dataframe with predictions and protected attributes
            df_with_preds = self.X_test.copy()
            df_with_preds['predictions'] = y_pred
            df_with_preds[self.target_column] = self.y_test

            # Create bias detector
            detector = BiasDetector(
                dataset=df_with_preds,
                sensitive_features=self.protected_attributes,
                target=self.target_column,
                justified_attributes=self.justified_attributes
            )

            # Detect bias in predictions
            results = detector.detect_bias_on_model_predictions(
                df_with_preds, 
                'predictions', 
                self.target_column
            )

            return [result.to_dict() for result in results]

        except Exception as e:
            logger.error(f"❌ Bias detection failed: {e}")
            return []

    def compute_fairness_metrics(self) -> Dict[str, Any]:
        """
        Compute fairness metrics for model predictions.

        Returns:
            Dictionary with fairness metrics by protected attribute
        """
        logger.info("⚖️ Computing fairness metrics for model...")

        fairness_metrics = {}

        try:
            from .fairness_metrics import FairnessMetrics

            # Get predictions
            y_pred = self.model.predict(self.X_test)

            # For each protected attribute, compute fairness metrics
            for attr in self.protected_attributes:
                if attr not in self.X_test.columns:
                    continue

                # Determine privileged group (most common value)
                privileged_group = self.X_test[attr].mode().iloc[0]

                fairness = FairnessMetrics(
                    y_true=self.y_test,
                    y_pred=y_pred,
                    protected_attr=self.X_test[attr].values,
                    privileged_group=privileged_group
                )

                fairness_metrics[attr] = fairness.evaluate()
                fairness_metrics[attr]['is_justified'] = attr in self.justified_attributes

                # Add group-specific performance metrics
                fairness_metrics[attr]['group_performance'] = self._compute_group_performance(
                    attr, privileged_group
                )

        except Exception as e:
            logger.error(f"❌ Fairness metrics computation failed: {e}")

        return fairness_metrics

    def _compute_group_performance(self, attr: str, privileged_group: Any) -> Dict[str, Dict[str, float]]:
        """Compute performance metrics for different groups."""
        try:
            y_pred = self.model.predict(self.X_test)

            # Split by groups
            privileged_mask = self.X_test[attr] == privileged_group
            unprivileged_mask = ~privileged_mask

            # Performance for privileged group
            priv_performance = {}
            unpriv_performance = {}

            if self.task_type == 'classification':
                priv_performance = self._classification_metrics(
                    self.y_test[privileged_mask], y_pred[privileged_mask]
                )
                unpriv_performance = self._classification_metrics(
                    self.y_test[unprivileged_mask], y_pred[unprivileged_mask]
                )
            else:
                priv_performance = self._regression_metrics(
                    self.y_test[privileged_mask], y_pred[privileged_mask]
                )
                unpriv_performance = self._regression_metrics(
                    self.y_test[unprivileged_mask], y_pred[unprivileged_mask]
                )

            return {
                'privileged_group': priv_performance,
                'unprivileged_group': unpriv_performance
            }

        except Exception as e:
            logger.error(f"Group performance computation failed: {e}")
            return {}

    def explain_model(self, sample_size: int = 100, 
                     methods: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Generate model explanations using SHAP and LIME.

        Args:
            sample_size: Number of samples for explanation
            methods: List of explanation methods ('SHAP', 'LIME')

        Returns:
            Dictionary with explanation results
        """
        logger.info("🔍 Generating model explanations...")

        try:
            from .explainability import ExplainabilityEngine

            # Sample data for explanation
            sample_size = min(sample_size, len(self.X_test))
            X_sample = self.X_test.sample(n=sample_size, random_state=42)

            # Create explainability engine
            explainer = ExplainabilityEngine(
                model=self.model,
                training_data=self.X_test,  # Use test data as reference
                feature_names=list(self.X_test.columns),
                mode=self.task_type
            )

            # Generate explanations
            explanation_results = explainer.explain(X_sample, methods)

            # Convert to serializable format
            explanations = {}
            for result in explanation_results:
                explanations[result.method] = result.to_dict()

            return explanations

        except Exception as e:
            logger.error(f"❌ Model explanation failed: {e}")
            return {}

    def analyze_feature_importance(self) -> Dict[str, Any]:
        """
        Analyze feature importance for the model.

        Returns:
            Dictionary with feature importance analysis
        """
        logger.info("📊 Analyzing feature importance...")

        try:
            importance_analysis = {}

            # Model-based feature importance (if available)
            if hasattr(self.model, 'feature_importances_'):
                importances = self.model.feature_importances_
                feature_importance = dict(zip(self.X_test.columns, importances))
                importance_analysis['model_feature_importance'] = feature_importance

            # Permutation-based importance (simplified)
            importance_analysis['protected_attribute_importance'] = {}
            for attr in self.protected_attributes:
                if attr in self.X_test.columns:
                    # Simple correlation with predictions
                    y_pred = self.model.predict(self.X_test)
                    if self.task_type == 'classification':
                        # For classification, use correlation with predictions
                        corr = np.corrcoef(self.X_test[attr], y_pred)[0, 1]
                    else:
                        # For regression, use correlation coefficient
                        corr = np.corrcoef(self.X_test[attr], y_pred)[0, 1]

                    importance_analysis['protected_attribute_importance'][attr] = {
                        'correlation_with_predictions': corr if not np.isnan(corr) else 0.0,
                        'is_justified': attr in self.justified_attributes
                    }

            return importance_analysis

        except Exception as e:
            logger.error(f"❌ Feature importance analysis failed: {e}")
            return {}

    def audit(self) -> Dict[str, Any]:
        """
        Run comprehensive model audit.

        Returns:
            Complete audit results dictionary
        """
        logger.info("🚀 Starting comprehensive model audit...")

        # Model performance evaluation
        performance_metrics = self.evaluate_performance()

        # Bias detection
        bias_results = self.detect_bias()

        # Fairness metrics
        fairness_metrics = self.compute_fairness_metrics()

        # Model explanations
        explanations = self.explain_model()

        # Feature importance analysis
        feature_importance = self.analyze_feature_importance()

        # Compile complete audit results
        audit_results = {
            'model_info': {
                'model_type': type(self.model).__name__,
                'task_type': self.task_type,
                'test_data_shape': self.X_test.shape,
                'protected_attributes': self.protected_attributes,
                'justified_attributes': self.justified_attributes,
                'target_column': self.target_column
            },
            'performance_metrics': performance_metrics,
            'bias_detection': bias_results,
            'fairness_metrics': fairness_metrics,
            'explanations': explanations,
            'feature_importance': feature_importance,
            'audit_timestamp': pd.Timestamp.now().isoformat(),
            'recommendations': self._generate_model_recommendations(
                performance_metrics, bias_results, fairness_metrics
            )
        }

        logger.info("✅ Model audit completed successfully")
        return audit_results

    def _generate_model_recommendations(self, performance: Dict[str, float],
                                      bias_results: List[Dict], 
                                      fairness_metrics: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations based on model audit results."""
        recommendations = []

        # Performance-based recommendations
        if self.task_type == 'classification':
            if performance.get('accuracy', 0) < 0.7:
                recommendations.append(
                    "📉 Low model accuracy (<70%). Consider feature engineering, "
                    "hyperparameter tuning, or trying different algorithms."
                )
            if performance.get('f1_score', 0) < 0.6:
                recommendations.append(
                    "⚖️ Low F1 score suggests poor precision-recall balance. "
                    "Consider threshold tuning or class rebalancing."
                )
        else:  # regression
            if performance.get('r2_score', 0) < 0.5:
                recommendations.append(
                    "📉 Low R² score (<0.5) indicates poor model fit. "
                    "Consider feature engineering or different modeling approaches."
                )

        # Bias-related recommendations
        biased_attributes = [r['attribute'] for r in bias_results 
                           if r['biased'] and not r['justified']]
        if biased_attributes:
            recommendations.append(
                f"🚨 Model bias detected for attributes: {', '.join(set(biased_attributes))}. "
                f"Consider bias mitigation techniques like adversarial debiasing or "
                f"post-processing adjustments."
            )

        # Fairness-related recommendations
        problematic_attrs = []
        for attr, metrics in fairness_metrics.items():
            if not metrics.get('is_justified', False):
                disparities = metrics.get('group_disparities', {})
                if any(abs(val) > 0.1 for val in disparities.values()):
                    problematic_attrs.append(attr)

        if problematic_attrs:
            recommendations.append(
                f"⚖️ Significant fairness disparities found for: {', '.join(problematic_attrs)}. "
                f"Consider fairness-aware algorithms or post-processing techniques."
            )

        # Justified attributes note
        if self.justified_attributes:
            recommendations.append(
                f"📋 Note: {', '.join(self.justified_attributes)} are marked as justified "
                f"attributes. Disparities in these features may be acceptable for business reasons."
            )

        return recommendations if recommendations else [
            "✅ No major issues detected in model audit."
        ]

# Convenience functions for backward compatibility
def audit_model(model: BaseEstimator,
               X_test: pd.DataFrame,
               y_test: Union[pd.Series, np.ndarray],
               protected_attributes: List[str],
               **kwargs) -> Dict[str, Any]:
    """Convenience function for model auditing."""
    auditor = ModelAuditor(
        model=model,
        X_test=X_test,
        y_test=y_test,
        protected_attributes=protected_attributes,
        **kwargs
    )
    return auditor.audit()

def run_model_audit(model: BaseEstimator, 
                   X_test: pd.DataFrame, 
                   y_test: Union[pd.Series, np.ndarray],
                   protected_attributes: List[str], 
                   task_type: str = "classification") -> Dict[str, Any]:
    """Legacy function for backward compatibility."""
    return audit_model(model, X_test, y_test, protected_attributes, task_type=task_type)
