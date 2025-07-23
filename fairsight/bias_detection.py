"""
Fairsight Toolkit - Enhanced Bias Detection
===========================================

This module provides comprehensive bias detection capabilities with support for
justified attributes that should not be flagged as discriminatory bias.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from typing import List, Optional, Union, Dict, Any, Tuple
import logging
from .utils import Utils

logger = logging.getLogger(__name__)

class BiasDetectionResult:
    """Container for bias detection results."""

    def __init__(self, metric_name: str, value: float, threshold: float, 
                 biased: bool, attribute: str, justified: bool = False, 
                 details: Optional[Dict[str, Any]] = None):
        self.metric_name = metric_name
        self.value = value
        self.threshold = threshold
        self.biased = biased
        self.attribute = attribute
        self.justified = justified  # New field for justified attributes
        self.details = details or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metric_name": self.metric_name,
            "value": self.value,
            "threshold": self.threshold,
            "biased": self.biased,
            "attribute": self.attribute,
            "justified": self.justified,
            "details": self.details
        }

    def __repr__(self) -> str:
        status = "JUSTIFIED" if self.justified else ("BIASED" if self.biased else "FAIR")
        return f"BiasResult({self.attribute}.{self.metric_name}: {self.value:.3f} [{status}])"

class BiasDetector:
    """
    Enhanced bias detection with support for justified attributes.

    Key features:
    - Comprehensive fairness metrics
    - Justified attributes handling
    - Configurable thresholds
    - Detailed reporting
    """

    def __init__(self, 
                 dataset: Optional[Union[str, pd.DataFrame]] = None,
                 model: Optional[Any] = None,
                 sensitive_features: Optional[List[str]] = None,
                 target: Optional[str] = None,
                 privileged_values: Optional[Dict[str, Any]] = None,
                 justified_attributes: Optional[List[str]] = None,
                 threshold: float = 0.8):
        """
        Initialize BiasDetector.

        Args:
            dataset: Dataset path or DataFrame
            model: Trained model for prediction-based bias detection
            sensitive_features: List of sensitive/protected attributes
            target: Target column name
            privileged_values: Dict mapping attributes to privileged values
            justified_attributes: List of attributes justified for discrimination
            threshold: Fairness threshold (default 0.8 for 80% rule)
        """
        self.dataset = self._load_dataset(dataset) if dataset is not None else None
        self.model = model
        self.sensitive_features = sensitive_features or []
        self.target = target
        self.privileged_values = privileged_values or {}
        self.justified_attributes = justified_attributes or []
        self.threshold = threshold

        # Auto-determine privileged groups if not provided
        if self.dataset is not None and not self.privileged_values:
            self.privileged_values = Utils.calculate_privilege_groups(
                self.dataset, self.sensitive_features
            )

        logger.info(f"🔍 BiasDetector initialized")
        logger.info(f"📋 Justified attributes: {self.justified_attributes}")

    def _load_dataset(self, dataset: Union[str, pd.DataFrame]) -> pd.DataFrame:
        """Load dataset from path or return DataFrame."""
        if isinstance(dataset, str):
            return pd.read_csv(dataset)
        return dataset.copy()

    def set_justified_attributes(self, attributes: List[str]):
        """Set attributes that are justified for discrimination."""
        self.justified_attributes = attributes
        logger.info(f"📋 Updated justified attributes: {attributes}")

    def _get_binary_group_mask(self, df: pd.DataFrame, feature: str, 
                              privileged_val: Union[str, int]) -> Tuple[pd.Series, pd.Series]:
        """Get binary masks for privileged and unprivileged groups."""
        privileged_mask = df[feature] == privileged_val
        unprivileged_mask = ~privileged_mask
        return privileged_mask, unprivileged_mask

    def _safe_divide(self, a: float, b: float) -> float:
        """Safely divide two numbers."""
        return Utils.safe_divide(a, b)

    def _compute_disparate_impact(self, df: pd.DataFrame, target: str, 
                                 feature: str, privileged_val: Any) -> BiasDetectionResult:
        """Compute disparate impact (80% rule)."""
        try:
            privileged_mask, unprivileged_mask = self._get_binary_group_mask(df, feature, privileged_val)

            favorable_rate_priv = df[privileged_mask][target].mean()
            favorable_rate_unpriv = df[unprivileged_mask][target].mean()

            di = self._safe_divide(favorable_rate_unpriv, favorable_rate_priv)
            biased = di < self.threshold
            justified = feature in self.justified_attributes

            return BiasDetectionResult(
                "Disparate Impact", di, self.threshold, biased, feature, justified,
                {
                    "favorable_rate_privileged": favorable_rate_priv,
                    "favorable_rate_unprivileged": favorable_rate_unpriv,
                    "privileged_group": privileged_val,
                    "privileged_count": privileged_mask.sum(),
                    "unprivileged_count": unprivileged_mask.sum()
                }
            )
        except Exception as e:
            return BiasDetectionResult(
                "Disparate Impact", 0.0, self.threshold, True, feature, False,
                {"error": str(e)}
            )

    def _compute_statistical_parity_difference(self, df: pd.DataFrame, target: str, 
                                              feature: str, privileged_val: Any) -> BiasDetectionResult:
        """Compute statistical parity difference."""
        try:
            privileged_mask, unprivileged_mask = self._get_binary_group_mask(df, feature, privileged_val)

            rate_priv = df[privileged_mask][target].mean()
            rate_unpriv = df[unprivileged_mask][target].mean()

            spd = rate_unpriv - rate_priv
            biased = abs(spd) > (1 - self.threshold)
            justified = feature in self.justified_attributes

            return BiasDetectionResult(
                "Statistical Parity Difference", spd, 1 - self.threshold, biased, feature, justified,
                {
                    "rate_privileged": rate_priv,
                    "rate_unprivileged": rate_unpriv,
                    "privileged_group": privileged_val
                }
            )
        except Exception as e:
            return BiasDetectionResult(
                "Statistical Parity Difference", 0.0, 1 - self.threshold, True, feature, False,
                {"error": str(e)}
            )

    def _compute_equal_opportunity_difference(self, df: pd.DataFrame, prediction: str, 
                                            target: str, feature: str, privileged_val: Any) -> BiasDetectionResult:
        """Compute equal opportunity difference (TPR difference)."""
        try:
            privileged_mask, unprivileged_mask = self._get_binary_group_mask(df, feature, privileged_val)

            # True Positive Rate for privileged group
            tp_rate_priv = self._safe_divide(
                ((df[privileged_mask][prediction] == 1) & (df[privileged_mask][target] == 1)).sum(),
                (df[privileged_mask][target] == 1).sum()
            )

            # True Positive Rate for unprivileged group  
            tp_rate_unpriv = self._safe_divide(
                ((df[unprivileged_mask][prediction] == 1) & (df[unprivileged_mask][target] == 1)).sum(),
                (df[unprivileged_mask][target] == 1).sum()
            )

            eod = tp_rate_unpriv - tp_rate_priv
            biased = abs(eod) > (1 - self.threshold)
            justified = feature in self.justified_attributes

            return BiasDetectionResult(
                "Equal Opportunity Difference", eod, 1 - self.threshold, biased, feature, justified,
                {
                    "tpr_privileged": tp_rate_priv,
                    "tpr_unprivileged": tp_rate_unpriv,
                    "privileged_group": privileged_val
                }
            )
        except Exception as e:
            return BiasDetectionResult(
                "Equal Opportunity Difference", 0.0, 1 - self.threshold, True, feature, False,
                {"error": str(e)}
            )

    def _compute_predictive_parity_difference(self, df: pd.DataFrame, prediction: str, 
                                            target: str, feature: str, privileged_val: Any) -> BiasDetectionResult:
        """Compute predictive parity difference (PPV difference)."""
        try:
            privileged_mask, unprivileged_mask = self._get_binary_group_mask(df, feature, privileged_val)

            # Positive Predictive Value for privileged group
            ppv_priv = self._safe_divide(
                ((df[privileged_mask][prediction] == 1) & (df[privileged_mask][target] == 1)).sum(),
                (df[privileged_mask][prediction] == 1).sum()
            )

            # Positive Predictive Value for unprivileged group
            ppv_unpriv = self._safe_divide(
                ((df[unprivileged_mask][prediction] == 1) & (df[unprivileged_mask][target] == 1)).sum(),
                (df[unprivileged_mask][prediction] == 1).sum()
            )

            ppd = ppv_unpriv - ppv_priv
            biased = abs(ppd) > (1 - self.threshold)
            justified = feature in self.justified_attributes

            return BiasDetectionResult(
                "Predictive Parity Difference", ppd, 1 - self.threshold, biased, feature, justified,
                {
                    "ppv_privileged": ppv_priv,
                    "ppv_unprivileged": ppv_unpriv,
                    "privileged_group": privileged_val
                }
            )
        except Exception as e:
            return BiasDetectionResult(
                "Predictive Parity Difference", 0.0, 1 - self.threshold, True, feature, False,
                {"error": str(e)}
            )

    def _compute_equalized_odds_difference(self, df: pd.DataFrame, prediction: str, 
                                         target: str, feature: str, privileged_val: Any) -> List[BiasDetectionResult]:
        """Compute equalized odds (both TPR and FPR differences)."""
        results = []

        try:
            privileged_mask, unprivileged_mask = self._get_binary_group_mask(df, feature, privileged_val)

            # True Positive Rates
            tpr_priv = self._safe_divide(
                ((df[privileged_mask][prediction] == 1) & (df[privileged_mask][target] == 1)).sum(),
                (df[privileged_mask][target] == 1).sum()
            )
            tpr_unpriv = self._safe_divide(
                ((df[unprivileged_mask][prediction] == 1) & (df[unprivileged_mask][target] == 1)).sum(),
                (df[unprivileged_mask][target] == 1).sum()
            )

            # False Positive Rates
            fpr_priv = self._safe_divide(
                ((df[privileged_mask][prediction] == 1) & (df[privileged_mask][target] == 0)).sum(),
                (df[privileged_mask][target] == 0).sum()
            )
            fpr_unpriv = self._safe_divide(
                ((df[unprivileged_mask][prediction] == 1) & (df[unprivileged_mask][target] == 0)).sum(),
                (df[unprivileged_mask][target] == 0).sum()
            )

            tpr_diff = tpr_unpriv - tpr_priv
            fpr_diff = fpr_unpriv - fpr_priv

            justified = feature in self.justified_attributes

            results.append(BiasDetectionResult(
                "Equalized Odds (TPR)", tpr_diff, 1 - self.threshold, 
                abs(tpr_diff) > (1 - self.threshold), feature, justified,
                {"tpr_privileged": tpr_priv, "tpr_unprivileged": tpr_unpriv}
            ))

            results.append(BiasDetectionResult(
                "Equalized Odds (FPR)", fpr_diff, 1 - self.threshold, 
                abs(fpr_diff) > (1 - self.threshold), feature, justified,
                {"fpr_privileged": fpr_priv, "fpr_unprivileged": fpr_unpriv}
            ))

        except Exception as e:
            results.append(BiasDetectionResult(
                "Equalized Odds", 0.0, 1 - self.threshold, True, feature, False,
                {"error": str(e)}
            ))

        return results

    def detect_bias_on_dataset(self, df: Optional[pd.DataFrame] = None, 
                              target_col: Optional[str] = None) -> List[BiasDetectionResult]:
        """
        Detect bias on dataset (without model predictions).

        Args:
            df: DataFrame to analyze (uses self.dataset if None)
            target_col: Target column name (uses self.target if None)

        Returns:
            List of BiasDetectionResult objects
        """
        df = df if df is not None else self.dataset
        target_col = target_col if target_col is not None else self.target

        if df is None or target_col is None:
            raise ValueError("Dataset and target column must be provided")

        results = []
        logger.info("🔍 Running dataset bias detection...")

        for feature, privileged_val in self.privileged_values.items():
            if feature not in df.columns or target_col not in df.columns:
                logger.warning(f"⚠️ Skipping {feature}: column not found")
                continue

            # Check if this is a justified attribute
            is_justified = feature in self.justified_attributes
            if is_justified:
                logger.info(f"📋 {feature} is a justified attribute - will be marked accordingly")

            results.append(self._compute_disparate_impact(df, target_col, feature, privileged_val))
            results.append(self._compute_statistical_parity_difference(df, target_col, feature, privileged_val))

        return results

    def detect_bias_on_model_predictions(self, df: Optional[pd.DataFrame] = None, 
                                        prediction_col: str = "predictions", 
                                        label_col: Optional[str] = None) -> List[BiasDetectionResult]:
        """
        Detect bias on model predictions.

        Args:
            df: DataFrame with predictions (uses self.dataset if None)
            prediction_col: Column name containing predictions
            label_col: True label column name (uses self.target if None)

        Returns:
            List of BiasDetectionResult objects
        """
        df = df if df is not None else self.dataset
        label_col = label_col if label_col is not None else self.target

        if df is None or label_col is None:
            raise ValueError("Dataset and label column must be provided")

        results = []
        logger.info("🔍 Running model prediction bias detection...")

        for feature, privileged_val in self.privileged_values.items():
            if feature not in df.columns or prediction_col not in df.columns or label_col not in df.columns:
                logger.warning(f"⚠️ Skipping {feature}: required columns not found")
                continue

            is_justified = feature in self.justified_attributes
            if is_justified:
                logger.info(f"📋 {feature} is a justified attribute - will be marked accordingly")

            results.append(self._compute_disparate_impact(df, prediction_col, feature, privileged_val))
            results.append(self._compute_statistical_parity_difference(df, prediction_col, feature, privileged_val))
            results.append(self._compute_equal_opportunity_difference(df, prediction_col, label_col, feature, privileged_val))
            results.append(self._compute_predictive_parity_difference(df, prediction_col, label_col, feature, privileged_val))
            results.extend(self._compute_equalized_odds_difference(df, prediction_col, label_col, feature, privileged_val))

        return results

    def detect(self, include_model_predictions: bool = True) -> List[BiasDetectionResult]:
        """
        Main detect method that runs comprehensive bias detection.

        Args:
            include_model_predictions: Whether to include model-based bias detection

        Returns:
            List of all bias detection results
        """
        all_results = []

        # Dataset-level bias detection
        if self.dataset is not None and self.target is not None:
            dataset_results = self.detect_bias_on_dataset()
            all_results.extend(dataset_results)

        # Model prediction bias detection
        if include_model_predictions and self.model is not None and self.dataset is not None:
            try:
                # Make predictions
                X = self.dataset.drop(columns=[self.target])
                predictions = self.model.predict(X)

                # Add predictions to dataframe
                df_with_preds = self.dataset.copy()
                df_with_preds['predictions'] = predictions

                model_results = self.detect_bias_on_model_predictions(df_with_preds)
                all_results.extend(model_results)

            except Exception as e:
                logger.error(f"❌ Model prediction bias detection failed: {e}")

        # Log summary
        total_results = len(all_results)
        biased_results = sum(1 for r in all_results if r.biased and not r.justified)
        justified_results = sum(1 for r in all_results if r.justified)

        logger.info(f"📊 Bias detection complete:")
        logger.info(f"   - Total metrics: {total_results}")
        logger.info(f"   - Biased (concerning): {biased_results}")
        logger.info(f"   - Justified attributes: {justified_results}")

        return all_results

    def get_summary_report(self, results: Optional[List[BiasDetectionResult]] = None) -> Dict[str, Any]:
        """
        Generate summary report from bias detection results.

        Args:
            results: List of BiasDetectionResult objects

        Returns:
            Dictionary containing summary report
        """
        if results is None:
            results = self.detect()

        # Group results by attribute
        by_attribute = {}
        for result in results:
            attr = result.attribute
            if attr not in by_attribute:
                by_attribute[attr] = {
                    'results': [],
                    'is_justified': result.justified,
                    'bias_count': 0,
                    'total_count': 0
                }

            by_attribute[attr]['results'].append(result)
            by_attribute[attr]['total_count'] += 1
            if result.biased and not result.justified:
                by_attribute[attr]['bias_count'] += 1

        # Calculate overall metrics
        total_metrics = len(results)
        biased_metrics = sum(1 for r in results if r.biased and not r.justified)
        justified_metrics = sum(1 for r in results if r.justified)

        ethical_score = max(0, 100 - (biased_metrics * 10))  # Simple scoring

        return {
            'summary': {
                'total_metrics': total_metrics,
                'biased_metrics': biased_metrics,
                'justified_metrics': justified_metrics,
                'ethical_score': ethical_score,
                'attributes_analyzed': list(by_attribute.keys()),
                'justified_attributes': self.justified_attributes
            },
            'by_attribute': by_attribute,
            'detailed_results': [r.to_dict() for r in results]
        }

# Convenience functions for backward compatibility
def detect_dataset_bias(df: pd.DataFrame, protected_attributes: List[str], 
                       target_column: str, **kwargs) -> List[BiasDetectionResult]:
    """Convenience function for dataset bias detection."""
    detector = BiasDetector(
        dataset=df,
        sensitive_features=protected_attributes,
        target=target_column,
        **kwargs
    )
    return detector.detect_bias_on_dataset()

def detect_model_bias(model: Any, df: pd.DataFrame, protected_attributes: List[str], 
                     target_column: str, **kwargs) -> List[BiasDetectionResult]:
    """Convenience function for model bias detection."""
    detector = BiasDetector(
        dataset=df,
        model=model,
        sensitive_features=protected_attributes,
        target=target_column,
        **kwargs
    )
    return detector.detect(include_model_predictions=True)

def compute_disparate_impact(df: pd.DataFrame, target: str, feature: str, privileged_val: Any, threshold: float = 0.8, justified_attributes: Optional[list] = None) -> BiasDetectionResult:
    """Public wrapper to compute disparate impact (80% rule) for a feature."""
    detector = BiasDetector(dataset=df, sensitive_features=[feature], target=target, threshold=threshold, justified_attributes=justified_attributes)
    return detector._compute_disparate_impact(df, target, feature, privileged_val)

def compute_statistical_parity(df: pd.DataFrame, target: str, feature: str, privileged_val: Any, threshold: float = 0.8, justified_attributes: Optional[list] = None) -> BiasDetectionResult:
    """Public wrapper to compute statistical parity difference for a feature."""
    detector = BiasDetector(dataset=df, sensitive_features=[feature], target=target, threshold=threshold, justified_attributes=justified_attributes)
    return detector._compute_statistical_parity_difference(df, target, feature, privileged_val)
