"""
Fairsight Toolkit - Utility Functions
====================================

This module provides essential utility functions used across the Fairsight toolkit
for data validation, preprocessing, and common operations.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Union, Optional, Tuple
import logging
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Utils:
    """Utility class containing helper functions for the Fairsight toolkit."""

    @staticmethod
    def validate_inputs(X: pd.DataFrame, y: Union[pd.Series, np.ndarray], 
                       protected_attributes: List[str]) -> bool:
        """
        Validate input data for fairness analysis.

        Args:
            X: Feature dataset
            y: Target variable
            protected_attributes: List of protected attribute column names

        Returns:
            bool: True if validation passes

        Raises:
            ValueError: If validation fails
        """
        try:
            # Check if X is DataFrame
            if not isinstance(X, pd.DataFrame):
                raise ValueError("X must be a pandas DataFrame")

            # Check if y has same length as X
            if len(X) != len(y):
                raise ValueError("X and y must have the same number of samples")

            # Check if protected attributes exist in X
            missing_attrs = [attr for attr in protected_attributes if attr not in X.columns]
            if missing_attrs:
                raise ValueError(f"Protected attributes not found in X: {missing_attrs}")

            # Check for missing values in critical columns
            if X[protected_attributes].isnull().any().any():
                warnings.warn("Missing values found in protected attributes")

            logger.info("✅ Input validation passed")
            return True

        except Exception as e:
            logger.error(f"❌ Input validation failed: {e}")
            raise

    @staticmethod
    def infer_target_column(df: pd.DataFrame) -> str:
        """
        Infer the most likely target column from a dataset.

        Args:
            df: Input DataFrame

        Returns:
            str: Name of inferred target column
        """
        # Common target column names
        target_candidates = [
            'target', 'label', 'class', 'outcome', 'result', 'prediction',
            'approved', 'accepted', 'hired', 'admitted', 'selected',
            'y', 'response', 'dependent', 'output'
        ]

        # Check for exact matches first
        for candidate in target_candidates:
            if candidate in df.columns:
                logger.info(f"🎯 Found target column: {candidate}")
                return candidate

        # Check for partial matches
        for candidate in target_candidates:
            matching_cols = [col for col in df.columns if candidate in col.lower()]
            if matching_cols:
                logger.info(f"🎯 Inferred target column: {matching_cols[0]}")
                return matching_cols[0]

        # If no matches, assume last column
        logger.warning(f"⚠️ Could not infer target column, using last column: {df.columns[-1]}")
        return df.columns[-1]

    @staticmethod
    def is_classification_task(y: Union[pd.Series, np.ndarray]) -> bool:
        """
        Determine if the target variable represents a classification task.

        Args:
            y: Target variable

        Returns:
            bool: True if classification, False if regression
        """
        try:
            # Convert to pandas Series if numpy array
            if isinstance(y, np.ndarray):
                y = pd.Series(y)

            # Check data type
            if y.dtype == 'object' or y.dtype.name == 'category':
                return True

            # Check number of unique values
            unique_values = y.nunique()
            total_values = len(y)

            # If less than 20 unique values or less than 5% unique, likely classification
            if unique_values < 20 or (unique_values / total_values) < 0.05:
                return True

            # Check if all values are integers
            if y.dtype in ['int64', 'int32', 'int16', 'int8']:
                return True

            return False

        except Exception as e:
            logger.error(f"Error determining task type: {e}")
            return True  # Default to classification

    @staticmethod
    def is_regression_task(y: Union[pd.Series, np.ndarray]) -> bool:
        """
        Determine if the target variable represents a regression task.

        Args:
            y: Target variable

        Returns:
            bool: True if regression, False if classification
        """
        return not Utils.is_classification_task(y)

    @staticmethod
    def preprocess_data(df: pd.DataFrame, 
                       target_column: str,
                       protected_attributes: List[str],
                       justified_attributes: Optional[List[str]] = None) -> Tuple[pd.DataFrame, Dict[str, LabelEncoder]]:
        """
        Preprocess dataset for fairness analysis.

        Args:
            df: Input DataFrame
            target_column: Name of target column
            protected_attributes: List of protected attributes
            justified_attributes: List of justified attributes (won't be flagged for bias)

        Returns:
            Tuple of processed DataFrame and label encoders dict
        """
        try:
            df_processed = df.copy()
            label_encoders = {}

            # Handle missing values
            for col in df_processed.columns:
                if df_processed[col].dtype == 'object':
                    df_processed[col] = df_processed[col].fillna('Unknown')
                else:
                    df_processed[col] = df_processed[col].fillna(df_processed[col].median())

            # Encode categorical variables
            categorical_cols = df_processed.select_dtypes(include=['object', 'category']).columns

            for col in categorical_cols:
                label_encoders[col] = LabelEncoder()
                df_processed[col] = label_encoders[col].fit_transform(df_processed[col].astype(str))

            # Log justified attributes if provided
            if justified_attributes:
                logger.info(f"📋 Justified attributes (won't be flagged for bias): {justified_attributes}")

            logger.info("✅ Data preprocessing completed")
            return df_processed, label_encoders

        except Exception as e:
            logger.error(f"❌ Data preprocessing failed: {e}")
            raise

    @staticmethod
    def calculate_privilege_groups(df: pd.DataFrame, 
                                 protected_attributes: List[str]) -> Dict[str, Any]:
        """
        Automatically determine privileged groups for each protected attribute.

        Args:
            df: Input DataFrame
            protected_attributes: List of protected attributes

        Returns:
            Dict mapping attribute names to privileged values
        """
        privileged_groups = {}

        for attr in protected_attributes:
            if attr not in df.columns:
                continue

            # Get value counts
            value_counts = df[attr].value_counts()

            # For binary attributes, assume majority is privileged
            if len(value_counts) == 2:
                privileged_groups[attr] = value_counts.index[0]  # Most common value
            else:
                # For multi-class, use domain knowledge or default to most common
                # This should be configurable by user in practice
                privileged_groups[attr] = value_counts.index[0]

            logger.info(f"🏷️ Privileged group for {attr}: {privileged_groups[attr]}")

        return privileged_groups

    @staticmethod
    def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
        """
        Safely divide two numbers, returning default if division by zero.

        Args:
            numerator: Numerator value
            denominator: Denominator value  
            default: Default value to return if division by zero

        Returns:
            float: Result of division or default value
        """
        try:
            if denominator == 0 or np.isnan(denominator) or np.isinf(denominator):
                return default
            result = numerator / denominator
            if np.isnan(result) or np.isinf(result):
                return default
            return result
        except:
            return default

    @staticmethod
    def format_percentage(value: float) -> str:
        """Format a decimal value as a percentage string."""
        return f"{value * 100:.2f}%"

    @staticmethod
    def create_summary_stats(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Create summary statistics for a dataset.

        Args:
            df: Input DataFrame

        Returns:
            Dict containing summary statistics
        """
        return {
            'rows': len(df),
            'columns': len(df.columns),
            'missing_values': df.isnull().sum().sum(),
            'numeric_columns': len(df.select_dtypes(include=[np.number]).columns),
            'categorical_columns': len(df.select_dtypes(include=['object', 'category']).columns),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024
        }
