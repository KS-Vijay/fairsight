"""
Fairsight Toolkit - Dataset Audit
=================================

This module provides comprehensive auditing capabilities for datasets,
including bias detection, fairness analysis, and data quality assessment.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from typing import List, Dict, Any, Optional, Union, Tuple
import logging
import warnings
from .utils import Utils

logger = logging.getLogger(__name__)

class DatasetAuditor:
    """
    Comprehensive dataset auditing class.

    Features:
    - Automatic data loading and preprocessing
    - Bias detection with justified attributes support
    - Fairness metrics computation
    - Data quality assessment
    - Statistical analysis
    """

    def __init__(self, 
                 dataset: Union[str, pd.DataFrame], 
                 protected_attributes: Optional[List[str]] = None, 
                 target_column: Optional[str] = None,
                 justified_attributes: Optional[List[str]] = None):
        """
        Initialize DatasetAuditor.

        Args:
            dataset: Dataset path (CSV) or DataFrame
            protected_attributes: List of protected/sensitive attributes
            target_column: Target column name (auto-inferred if None)
            justified_attributes: Attributes justified for discrimination
        """
        self.dataset_path = dataset if isinstance(dataset, str) else None
        self.protected_attributes = protected_attributes or []
        self.justified_attributes = justified_attributes or []
        self.df = None
        self.original_df = None
        self.target_column = target_column
        self.label_encoders = {}
        self.task_type = None

        # Load and preprocess dataset
        self._load_dataset(dataset)
        self._validate_inputs()

    def _load_dataset(self, dataset: Union[str, pd.DataFrame]):
        """Load dataset from path or DataFrame."""
        try:
            if isinstance(dataset, str):
                self.df = pd.read_csv(dataset)
                logger.info(f"📄 Loaded dataset from {dataset}")
            else:
                self.df = dataset.copy()
                logger.info("📄 Loaded dataset from DataFrame")

            self.original_df = self.df.copy()
            logger.info(f"📊 Dataset shape: {self.df.shape[0]} rows × {self.df.shape[1]} columns")

        except Exception as e:
            raise ValueError(f"❌ Failed to load dataset: {e}")

    def _validate_inputs(self):
        """Validate input parameters."""
        if self.df is None or self.df.empty:
            raise ValueError("Dataset is empty or not loaded")

        # Infer target column if not provided
        if not self.target_column:
            self.target_column = Utils.infer_target_column(self.df)
            logger.info(f"🎯 Inferred target column: {self.target_column}")

        # Validate target column exists
        if self.target_column not in self.df.columns:
            raise ValueError(f"Target column '{self.target_column}' not found in dataset")

        # Validate protected attributes exist
        missing_attrs = [attr for attr in self.protected_attributes if attr not in self.df.columns]
        if missing_attrs:
            warnings.warn(f"Protected attributes not found in dataset: {missing_attrs}")
            self.protected_attributes = [attr for attr in self.protected_attributes if attr in self.df.columns]

        # Log justified attributes
        if self.justified_attributes:
            logger.info(f"📋 Justified attributes: {self.justified_attributes}")

    def preprocess(self, handle_missing: bool = True, encode_categorical: bool = True) -> Dict[str, Any]:
        """
        Preprocess dataset for analysis.

        Args:
            handle_missing: Whether to handle missing values
            encode_categorical: Whether to encode categorical variables

        Returns:
            Dictionary with preprocessing information
        """
        logger.info("🔄 Preprocessing dataset...")

        preprocessing_info = {
            'original_shape': self.df.shape,
            'missing_values_before': self.df.isnull().sum().sum(),
            'categorical_columns': [],
            'encoded_columns': [],
            'label_encoders': {}
        }

        # Handle missing values
        if handle_missing:
            self._handle_missing_values()

        # Encode categorical variables
        if encode_categorical:
            preprocessing_info.update(self._encode_categorical_variables())

        # Detect task type
        self.task_type = self._detect_task_type()

        preprocessing_info.update({
            'final_shape': self.df.shape,
            'missing_values_after': self.df.isnull().sum().sum(),
            'task_type': self.task_type
        })

        logger.info(f"✅ Preprocessing complete. Task type: {self.task_type}")
        return preprocessing_info

    def _handle_missing_values(self):
        """Handle missing values in the dataset."""
        missing_counts = self.df.isnull().sum()
        total_missing = missing_counts.sum()

        if total_missing > 0:
            logger.info(f"🔧 Handling {total_missing} missing values...")

            for col in self.df.columns:
                if self.df[col].isnull().sum() > 0:
                    if self.df[col].dtype == 'object' or self.df[col].dtype.name == 'category':
                        # Fill categorical with mode or 'Unknown'
                        mode_val = self.df[col].mode()
                        fill_val = mode_val[0] if len(mode_val) > 0 else 'Unknown'
                        self.df[col] = self.df[col].fillna(fill_val)
                    else:
                        # Fill numerical with median
                        self.df[col] = self.df[col].fillna(self.df[col].median())

    def _encode_categorical_variables(self) -> Dict[str, Any]:
        """Encode categorical variables."""
        categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
        encoded_cols = []

        logger.info(f"🏷️ Encoding {len(categorical_cols)} categorical columns...")

        for col in categorical_cols:
            if col != self.target_column or self.task_type == 'classification':
                try:
                    self.label_encoders[col] = LabelEncoder()
                    self.df[col] = self.label_encoders[col].fit_transform(self.df[col].astype(str))
                    encoded_cols.append(col)
                except Exception as e:
                    logger.warning(f"⚠️ Failed to encode column {col}: {e}")

        return {
            'categorical_columns': categorical_cols,
            'encoded_columns': encoded_cols,
            'label_encoders': list(self.label_encoders.keys())
        }

    def _detect_task_type(self) -> str:
        """Detect whether this is a classification or regression task."""
        y = self.df[self.target_column]
        if Utils.is_classification_task(y):
            return 'classification'
        else:
            return 'regression'

    def analyze_data_quality(self) -> Dict[str, Any]:
        """
        Perform comprehensive data quality analysis.

        Returns:
            Dictionary with data quality metrics
        """
        logger.info("🔍 Analyzing data quality...")

        # Basic statistics
        basic_stats = Utils.create_summary_stats(self.df)

        # Missing values analysis
        missing_analysis = {
            'columns_with_missing': self.df.isnull().sum()[self.df.isnull().sum() > 0].to_dict(),
            'missing_percentage': (self.df.isnull().sum() / len(self.df) * 100).to_dict()
        }

        # Duplicate analysis
        duplicate_analysis = {
            'duplicate_rows': self.df.duplicated().sum(),
            'duplicate_percentage': (self.df.duplicated().sum() / len(self.df) * 100)
        }

        # Distribution analysis for protected attributes
        protected_distributions = {}
        for attr in self.protected_attributes:
            if attr in self.df.columns:
                protected_distributions[attr] = self.df[attr].value_counts().to_dict()

        # Target distribution
        target_distribution = self.df[self.target_column].value_counts().to_dict()

        return {
            'basic_statistics': basic_stats,
            'missing_analysis': missing_analysis,
            'duplicate_analysis': duplicate_analysis,
            'protected_attribute_distributions': protected_distributions,
            'target_distribution': target_distribution,
            'class_imbalance': self._check_class_imbalance()
        }

    def _check_class_imbalance(self) -> Dict[str, Any]:
        """Check for class imbalance in the target variable."""
        if self.task_type != 'classification':
            return {'imbalanced': False, 'reason': 'Not a classification task'}

        value_counts = self.df[self.target_column].value_counts()
        total_samples = len(self.df)

        # Calculate imbalance ratio
        majority_class_size = value_counts.iloc[0]
        minority_class_size = value_counts.iloc[-1]
        imbalance_ratio = majority_class_size / minority_class_size

        # Consider imbalanced if ratio > 2:1
        is_imbalanced = imbalance_ratio > 2.0

        return {
            'imbalanced': is_imbalanced,
            'imbalance_ratio': imbalance_ratio,
            'majority_class': value_counts.index[0],
            'majority_class_size': majority_class_size,
            'minority_class': value_counts.index[-1],
            'minority_class_size': minority_class_size,
            'class_distribution': value_counts.to_dict()
        }

    def detect_bias(self) -> List[Dict[str, Any]]:
        """
        Detect bias in the dataset using the enhanced BiasDetector.

        Returns:
            List of bias detection results
        """
        logger.info("🔍 Running bias detection on dataset...")

        try:
            from .bias_detection import BiasDetector

            # Create bias detector with justified attributes
            detector = BiasDetector(
                dataset=self.df,
                sensitive_features=self.protected_attributes,
                target=self.target_column,
                justified_attributes=self.justified_attributes
            )

            # Detect bias
            results = detector.detect_bias_on_dataset()

            # Convert to dictionaries for JSON serialization
            return [result.to_dict() for result in results]

        except Exception as e:
            logger.error(f"❌ Bias detection failed: {e}")
            return []

    def compute_fairness_metrics(self) -> Dict[str, Any]:
        """
        Compute fairness metrics for the dataset.

        Returns:
            Dictionary with fairness metrics by protected attribute
        """
        logger.info("⚖️ Computing fairness metrics...")

        fairness_metrics = {}

        try:
            from .fairness_metrics import FairnessMetrics

            # For each protected attribute, compute fairness metrics
            for attr in self.protected_attributes:
                if attr not in self.df.columns:
                    continue

                # Determine privileged group (most common value)
                privileged_group = self.df[attr].mode().iloc[0]

                # Create dummy predictions (use target for dataset-only analysis)
                y_true = self.df[self.target_column].values
                y_pred = y_true  # For dataset analysis, we use actual values
                protected_attr = self.df[attr].values

                fairness = FairnessMetrics(
                    y_true=y_true,
                    y_pred=y_pred,
                    protected_attr=protected_attr,
                    privileged_group=privileged_group
                )

                fairness_metrics[attr] = fairness.evaluate()
                fairness_metrics[attr]['is_justified'] = attr in self.justified_attributes

        except Exception as e:
            logger.error(f"❌ Fairness metrics computation failed: {e}")

        return fairness_metrics

    def generate_statistical_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive statistical report.

        Returns:
            Dictionary with statistical analysis
        """
        logger.info("📊 Generating statistical report...")

        try:
            # Descriptive statistics
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            descriptive_stats = self.df[numeric_cols].describe().to_dict()

            # Correlation analysis (excluding target for correlation matrix)
            correlation_matrix = {}
            if len(numeric_cols) > 1:
                corr_df = self.df[numeric_cols].corr()
                correlation_matrix = corr_df.to_dict()

                # Find highly correlated features (>0.8)
                high_correlations = []
                for i in range(len(corr_df.columns)):
                    for j in range(i+1, len(corr_df.columns)):
                        if abs(corr_df.iloc[i, j]) > 0.8:
                            high_correlations.append({
                                'feature1': corr_df.columns[i],
                                'feature2': corr_df.columns[j],
                                'correlation': corr_df.iloc[i, j]
                            })

            # Feature importance (simple correlation with target)
            feature_importance = {}
            if self.task_type == 'classification':
                for col in numeric_cols:
                    if col != self.target_column:
                        try:
                            corr = self.df[col].corr(self.df[self.target_column])
                            if not np.isnan(corr):
                                feature_importance[col] = abs(corr)
                        except:
                            pass

            return {
                'descriptive_statistics': descriptive_stats,
                'correlation_matrix': correlation_matrix,
                'high_correlations': high_correlations,
                'feature_importance': feature_importance
            }

        except Exception as e:
            logger.error(f"❌ Statistical report generation failed: {e}")
            return {}

    def audit(self) -> Dict[str, Any]:
        """
        Run comprehensive dataset audit.

        Returns:
            Complete audit results dictionary
        """
        logger.info("🚀 Starting comprehensive dataset audit...")

        # Preprocess data
        preprocessing_info = self.preprocess()

        # Data quality analysis
        data_quality = self.analyze_data_quality()

        # Bias detection
        bias_results = self.detect_bias()

        # Fairness metrics
        fairness_metrics = self.compute_fairness_metrics()

        # Statistical analysis
        statistical_report = self.generate_statistical_report()

        # Compile complete audit results
        audit_results = {
            'dataset_info': {
                'source': self.dataset_path or 'DataFrame',
                'original_shape': self.original_df.shape,
                'processed_shape': self.df.shape,
                'target_column': self.target_column,
                'protected_attributes': self.protected_attributes,
                'justified_attributes': self.justified_attributes,
                'task_type': self.task_type
            },
            'preprocessing': preprocessing_info,
            'data_quality': data_quality,
            'bias_detection': bias_results,
            'fairness_metrics': fairness_metrics,
            'statistical_analysis': statistical_report,
            'audit_timestamp': pd.Timestamp.now().isoformat(),
            'recommendations': self._generate_recommendations(bias_results, data_quality)
        }

        logger.info("✅ Dataset audit completed successfully")
        return audit_results

    def _generate_recommendations(self, bias_results: List[Dict], 
                                 data_quality: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations based on audit results."""
        recommendations = []

        # Bias-related recommendations
        biased_attributes = [r['attribute'] for r in bias_results 
                           if r['biased'] and not r['justified']]
        if biased_attributes:
            recommendations.append(
                f"🚨 Bias detected in attributes: {', '.join(set(biased_attributes))}. "
                f"Consider data collection improvements or bias mitigation techniques."
            )

        # Data quality recommendations
        if data_quality['duplicate_analysis']['duplicate_percentage'] > 5:
            recommendations.append(
                "🔄 High percentage of duplicate rows detected. Consider deduplication."
            )

        if data_quality['class_imbalance']['imbalanced']:
            recommendations.append(
                "⚖️ Class imbalance detected. Consider resampling techniques or "
                "cost-sensitive learning methods."
            )

        missing_cols = data_quality['missing_analysis']['columns_with_missing']
        if missing_cols:
            high_missing = [col for col, count in missing_cols.items() 
                          if (count / len(self.df) * 100) > 20]
            if high_missing:
                recommendations.append(
                    f"❓ High missing values (>20%) in columns: {', '.join(high_missing)}. "
                    f"Consider feature engineering or removal."
                )

        # Justified attributes note
        if self.justified_attributes:
            recommendations.append(
                f"📋 Note: {', '.join(self.justified_attributes)} are marked as justified "
                f"attributes and bias in these features may be acceptable for business reasons."
            )

        return recommendations if recommendations else ["✅ No major issues detected in dataset audit."]

# Convenience functions for backward compatibility
def audit_dataset(dataset: Union[str, pd.DataFrame], 
                 protected_attributes: List[str],
                 target_column: Optional[str] = None,
                 **kwargs) -> Dict[str, Any]:
    """Convenience function for dataset auditing."""
    auditor = DatasetAuditor(
        dataset=dataset,
        protected_attributes=protected_attributes,
        target_column=target_column,
        **kwargs
    )
    return auditor.audit()
