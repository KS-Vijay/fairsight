"""
Explainability Engine for Fairsight Toolkit
==========================================

Provides model explainability using SHAP, LIME, and custom feature importance methods.
Includes support for both local and global explanations with bias-aware analysis.
"""

import shap
import lime
import lime.lime_tabular
import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Any, Dict, List, Optional, Union, Tuple
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ExplainabilityResult:
    """Container for explainability analysis results."""
    
    def __init__(
        self, 
        method: str, 
        global_importance: Optional[Dict[str, float]] = None,
        local_explanations: Optional[List[Dict[str, float]]] = None,
        feature_interactions: Optional[Dict[str, float]] = None,
        bias_explanation: Optional[Dict[str, Any]] = None,
        visualization_paths: Optional[List[str]] = None
    ):
        self.method = method
        self.global_importance = global_importance or {}
        self.local_explanations = local_explanations or []
        self.feature_interactions = feature_interactions or {}
        self.bias_explanation = bias_explanation or {}
        self.visualization_paths = visualization_paths or []
    
    def to_dict(self):
        """Convert result to dictionary format."""
        return {
            "method": self.method,
            "global_importance": self.global_importance,
            "local_explanations": self.local_explanations,
            "feature_interactions": self.feature_interactions,
            "bias_explanation": self.bias_explanation,
            "visualization_paths": self.visualization_paths
        }
    
    def get_top_features(self, n: int = 10) -> List[Tuple[str, float]]:
        """Get top N most important features."""
        if not self.global_importance:
            return []
        
        sorted_features = sorted(
            self.global_importance.items(), 
            key=lambda x: abs(x[1]), 
            reverse=True
        )
        return sorted_features[:n]


class ExplainabilityEngine:
    """
    Comprehensive explainability engine with bias-aware analysis.
    
    Features:
    - SHAP explanations (global and local)
    - LIME explanations
    - Permutation importance
    - Feature interaction detection
    - Bias-aware explanations for protected attributes
    - Visualization generation
    """
    
    def __init__(
        self, 
        model: BaseEstimator, 
        training_data: pd.DataFrame, 
        feature_names: List[str], 
        mode: str = "classification",
        protected_attributes: Optional[List[str]] = None,
        justified_attributes: Optional[List[str]] = None,
        output_dir: str = "/tmp/fairsight_explanations"
    ):
        self.model = model
        self.training_data = training_data
        self.feature_names = feature_names
        self.mode = mode.lower()
        self.protected_attributes = protected_attributes or []
        self.justified_attributes = justified_attributes or []
        self.output_dir = output_dir
        
        self._validate_model()
        self._ensure_output_dir()
    
    def _validate_model(self):
        """Validate model compatibility."""
        if not hasattr(self.model, "predict"):
            raise ValueError("Model must implement a predict method.")
        
        if self.mode not in ["classification", "regression"]:
            raise ValueError("Mode must be either 'classification' or 'regression'.")
        
        try:
            # Test prediction on small sample
            test_sample = self.training_data.iloc[:1]
            self.model.predict(test_sample)
        except NotFittedError as e:
            raise ValueError("Model appears to be unfitted. Fit the model before explainability.") from e
        except Exception as e:
            logger.warning(f"Model validation warning: {e}")
    
    def _ensure_output_dir(self):
        """Create output directory for visualizations."""
        import os
        os.makedirs(self.output_dir, exist_ok=True)
    
    def explain_with_shap(
        self, 
        X: pd.DataFrame, 
        sample_size: int = 100,
        create_plots: bool = True
    ) -> ExplainabilityResult:
        """
        Generate SHAP explanations with bias analysis.
        
        Args:
            X: Data to explain
            sample_size: Number of samples for explanation
            create_plots: Whether to create visualization plots
            
        Returns:
            ExplainabilityResult with SHAP analysis
        """
        try:
            logger.info("Starting SHAP explanation...")
            
            # Sample data for performance
            X_sample = X.sample(n=min(sample_size, len(X)), random_state=42)
            
            # Create SHAP explainer based on model type
            if hasattr(self.model, 'predict_proba') and self.mode == "classification":
                explainer = shap.Explainer(self.model.predict_proba, self.training_data)
            else:
                explainer = shap.Explainer(self.model.predict, self.training_data)
            
            # Generate SHAP values
            shap_values = explainer(X_sample)
            
            # Handle multi-class case
            if len(shap_values.shape) == 3 and shap_values.shape[2] > 1:
                # Use values for positive class in binary classification
                if shap_values.shape[2] == 2:
                    shap_vals = shap_values.values[:, :, 1]
                else:
                    # For multi-class, use mean across classes
                    shap_vals = np.mean(shap_values.values, axis=2)
            else:
                shap_vals = shap_values.values
            
            # Global importance (mean absolute SHAP values)
            global_importance = {}
            for i, feature in enumerate(self.feature_names):
                global_importance[feature] = float(np.mean(np.abs(shap_vals[:, i])))
            
            # Local explanations
            local_explanations = []
            for i in range(len(X_sample)):
                local_exp = {}
                for j, feature in enumerate(self.feature_names):
                    local_exp[feature] = float(shap_vals[i, j])
                local_explanations.append(local_exp)
            
            # Bias-aware analysis
            bias_explanation = self._analyze_shap_bias(shap_vals, X_sample)
            
            # Create visualizations
            viz_paths = []
            if create_plots:
                viz_paths = self._create_shap_visualizations(shap_values, X_sample)
            
            logger.info("SHAP explanation completed successfully")
            
            return ExplainabilityResult(
                method="SHAP",
                global_importance=global_importance,
                local_explanations=local_explanations,
                bias_explanation=bias_explanation,
                visualization_paths=viz_paths
            )
            
        except Exception as e:
            logger.error(f"SHAP explanation failed: {e}")
            return ExplainabilityResult(
                method="SHAP",
                global_importance={},
                local_explanations=[],
                bias_explanation={"error": str(e)}
            )
    
    def explain_with_lime(
        self, 
        X: pd.DataFrame, 
        num_samples: int = 5,
        num_features: int = 10
    ) -> ExplainabilityResult:
        """
        Generate LIME explanations.
        
        Args:
            X: Data to explain
            num_samples: Number of instances to explain
            num_features: Number of features to show in explanation
            
        Returns:
            ExplainabilityResult with LIME analysis
        """
        try:
            logger.info("Starting LIME explanation...")
            
            # Create LIME explainer
            lime_explainer = lime.lime_tabular.LimeTabularExplainer(
                training_data=self.training_data.values,
                feature_names=self.feature_names,
                class_names=["negative", "positive"] if self.mode == "classification" else ["prediction"],
                mode=self.mode
            )
            
            # Sample instances to explain
            X_sample = X.sample(n=min(num_samples, len(X)), random_state=42)
            
            local_explanations = []
            for idx, (_, row) in enumerate(X_sample.iterrows()):
                try:
                    # Generate explanation for this instance
                    if self.mode == "classification" and hasattr(self.model, 'predict_proba'):
                        exp = lime_explainer.explain_instance(
                            data_row=row.values,
                            predict_fn=self.model.predict_proba,
                            num_features=num_features
                        )
                    else:
                        exp = lime_explainer.explain_instance(
                            data_row=row.values,
                            predict_fn=self.model.predict,
                            num_features=num_features
                        )
                    
                    # Convert to dict format
                    exp_dict = dict(exp.as_list())
                    local_explanations.append(exp_dict)
                    
                except Exception as e:
                    logger.warning(f"LIME explanation failed for instance {idx}: {e}")
                    continue
            
            # Bias analysis for LIME
            bias_explanation = self._analyze_lime_bias(local_explanations)
            
            logger.info("LIME explanation completed successfully")
            
            return ExplainabilityResult(
                method="LIME",
                global_importance={},  # LIME doesn't provide global importance
                local_explanations=local_explanations,
                bias_explanation=bias_explanation
            )
            
        except Exception as e:
            logger.error(f"LIME explanation failed: {e}")
            return ExplainabilityResult(
                method="LIME",
                global_importance={},
                local_explanations=[],
                bias_explanation={"error": str(e)}
            )
    
    def explain_with_permutation_importance(
        self, 
        X: pd.DataFrame, 
        y: pd.Series,
        n_repeats: int = 10
    ) -> ExplainabilityResult:
        """
        Generate permutation importance explanations.
        
        Args:
            X: Feature data
            y: Target data
            n_repeats: Number of permutation repeats
            
        Returns:
            ExplainabilityResult with permutation importance
        """
        try:
            logger.info("Starting permutation importance analysis...")
            
            # Compute permutation importance
            perm_importance = permutation_importance(
                self.model, X, y, n_repeats=n_repeats, random_state=42
            )
            
            # Create importance dictionary
            global_importance = {}
            for i, feature in enumerate(self.feature_names):
                global_importance[feature] = float(perm_importance.importances_mean[i])
            
            # Bias analysis
            bias_explanation = self._analyze_permutation_bias(global_importance)
            
            logger.info("Permutation importance completed successfully")
            
            return ExplainabilityResult(
                method="PermutationImportance",
                global_importance=global_importance,
                bias_explanation=bias_explanation
            )
            
        except Exception as e:
            logger.error(f"Permutation importance failed: {e}")
            return ExplainabilityResult(
                method="PermutationImportance",
                global_importance={},
                bias_explanation={"error": str(e)}
            )
    
    def _analyze_shap_bias(
        self, 
        shap_values: np.ndarray, 
        X_sample: pd.DataFrame
    ) -> Dict[str, Any]:
        """Analyze SHAP values for bias in protected attributes."""
        bias_analysis = {
            "protected_attribute_importance": {},
            "justified_attribute_importance": {},
            "bias_risk_score": 0.0,
            "recommendations": []
        }
        
        try:
            # Analyze protected attributes
            for attr in self.protected_attributes:
                if attr in self.feature_names:
                    attr_idx = self.feature_names.index(attr)
                    importance = float(np.mean(np.abs(shap_values[:, attr_idx])))
                    bias_analysis["protected_attribute_importance"][attr] = importance
                    
                    # High importance in protected attributes indicates potential bias
                    if importance > 0.1:  # Threshold can be adjusted
                        bias_analysis["bias_risk_score"] += importance
                        bias_analysis["recommendations"].append(
                            f"High importance detected for protected attribute '{attr}'. "
                            f"Consider bias mitigation techniques."
                        )
            
            # Analyze justified attributes
            for attr in self.justified_attributes:
                if attr in self.feature_names:
                    attr_idx = self.feature_names.index(attr)
                    importance = float(np.mean(np.abs(shap_values[:, attr_idx])))
                    bias_analysis["justified_attribute_importance"][attr] = importance
            
            # Overall bias risk assessment
            if bias_analysis["bias_risk_score"] > 0.3:
                bias_analysis["overall_assessment"] = "HIGH_RISK"
            elif bias_analysis["bias_risk_score"] > 0.1:
                bias_analysis["overall_assessment"] = "MEDIUM_RISK"
            else:
                bias_analysis["overall_assessment"] = "LOW_RISK"
                
        except Exception as e:
            bias_analysis["error"] = str(e)
        
        return bias_analysis
    
    def _analyze_lime_bias(self, local_explanations: List[Dict]) -> Dict[str, Any]:
        """Analyze LIME explanations for bias patterns."""
        bias_analysis = {
            "protected_attribute_frequency": {},
            "recommendations": []
        }
        
        try:
            # Count how often protected attributes appear in explanations
            for attr in self.protected_attributes:
                count = sum(1 for exp in local_explanations if attr in exp)
                bias_analysis["protected_attribute_frequency"][attr] = count / len(local_explanations) if local_explanations else 0
                
                if bias_analysis["protected_attribute_frequency"][attr] > 0.5:
                    bias_analysis["recommendations"].append(
                        f"Protected attribute '{attr}' frequently appears in LIME explanations. "
                        f"This may indicate bias."
                    )
                    
        except Exception as e:
            bias_analysis["error"] = str(e)
        
        return bias_analysis
    
    def _analyze_permutation_bias(self, importance_dict: Dict[str, float]) -> Dict[str, Any]:
        """Analyze permutation importance for bias."""
        bias_analysis = {
            "protected_attribute_importance": {},
            "justified_attribute_importance": {},
            "recommendations": []
        }
        
        try:
            # Check protected attributes
            for attr in self.protected_attributes:
                if attr in importance_dict:
                    importance = importance_dict[attr]
                    bias_analysis["protected_attribute_importance"][attr] = importance
                    
                    if importance > 0.05:  # Threshold for concern
                        bias_analysis["recommendations"].append(
                            f"Protected attribute '{attr}' shows significant permutation importance. "
                            f"Consider bias mitigation."
                        )
            
            # Track justified attributes
            for attr in self.justified_attributes:
                if attr in importance_dict:
                    bias_analysis["justified_attribute_importance"][attr] = importance_dict[attr]
                    
        except Exception as e:
            bias_analysis["error"] = str(e)
        
        return bias_analysis
    
    def _create_shap_visualizations(
        self, 
        shap_values, 
        X_sample: pd.DataFrame
    ) -> List[str]:
        """Create SHAP visualization plots."""
        viz_paths = []
        
        try:
            # Summary plot
            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values, X_sample, feature_names=self.feature_names, show=False)
            summary_path = f"{self.output_dir}/shap_summary.png"
            plt.savefig(summary_path, bbox_inches='tight', dpi=300)
            plt.close()
            viz_paths.append(summary_path)
            
            # Feature importance plot
            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_values, X_sample, feature_names=self.feature_names, plot_type="bar", show=False)
            importance_path = f"{self.output_dir}/shap_importance.png"
            plt.savefig(importance_path, bbox_inches='tight', dpi=300)
            plt.close()
            viz_paths.append(importance_path)
            
        except Exception as e:
            logger.warning(f"SHAP visualization creation failed: {e}")
        
        return viz_paths
    
    def explain(
        self, 
        X: pd.DataFrame, 
        y: Optional[pd.Series] = None,
        methods: Optional[List[str]] = None
    ) -> List[ExplainabilityResult]:
        """
        Run comprehensive explainability analysis.
        
        Args:
            X: Feature data to explain
            y: Target data (required for permutation importance)
            methods: List of methods to use ['SHAP', 'LIME', 'PermutationImportance']
            
        Returns:
            List of ExplainabilityResult objects
        """
        if methods is None:
            methods = ["SHAP", "LIME"]
            if y is not None:
                methods.append("PermutationImportance")
        
        results = []
        
        if "SHAP" in methods:
            logger.info("Running SHAP explanation...")
            results.append(self.explain_with_shap(X))
        
        if "LIME" in methods:
            logger.info("Running LIME explanation...")
            results.append(self.explain_with_lime(X))
        
        if "PermutationImportance" in methods and y is not None:
            logger.info("Running Permutation Importance...")
            results.append(self.explain_with_permutation_importance(X, y))
        
        return results
    
    def generate_bias_explanation_report(
        self, 
        results: List[ExplainabilityResult]
    ) -> str:
        """Generate a comprehensive bias explanation report."""
        report_lines = [
            "# 🔍 Model Explainability & Bias Analysis Report\n",
            f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
            "---\n"
        ]
        
        for result in results:
            report_lines.extend([
                f"## {result.method} Analysis\n",
                f"### Global Feature Importance\n"
            ])
            
            # Top features
            top_features = result.get_top_features(10)
            if top_features:
                report_lines.append("| Feature | Importance |")
                report_lines.append("|---------|------------|")
                for feature, importance in top_features:
                    report_lines.append(f"| {feature} | {importance:.4f} |")
                report_lines.append("")
            
            # Bias analysis
            if result.bias_explanation:
                report_lines.extend([
                    f"### Bias Analysis\n",
                    f"**Protected Attributes:** {', '.join(self.protected_attributes)}\n",
                    f"**Justified Attributes:** {', '.join(self.justified_attributes)}\n"
                ])
                
                # Risk assessment
                if "overall_assessment" in result.bias_explanation:
                    assessment = result.bias_explanation["overall_assessment"]
                    emoji = "🔴" if assessment == "HIGH_RISK" else "🟡" if assessment == "MEDIUM_RISK" else "🟢"
                    report_lines.append(f"**Risk Level:** {emoji} {assessment}\n")
                
                # Recommendations
                if "recommendations" in result.bias_explanation:
                    report_lines.append("**Recommendations:**\n")
                    for rec in result.bias_explanation["recommendations"]:
                        report_lines.append(f"- {rec}")
                    report_lines.append("")
        
        report_content = "\n".join(report_lines)
        
        # Save report
        report_path = f"{self.output_dir}/explainability_bias_report.md"
        with open(report_path, 'w') as f:
            f.write(report_content)
        
        return report_path


# Convenience functions for quick access
def explain_model_shap(model, X_train, X_test, feature_names, **kwargs):
    """Quick SHAP explanation function."""
    engine = ExplainabilityEngine(model, X_train, feature_names, **kwargs)
    return engine.explain_with_shap(X_test)

def explain_model_lime(model, X_train, X_test, feature_names, **kwargs):
    """Quick LIME explanation function."""
    engine = ExplainabilityEngine(model, X_train, feature_names, **kwargs)
    return engine.explain_with_lime(X_test)


def explain_with_shap(model, X, feature_names, mode="classification", training_data=None, protected_attributes=None, justified_attributes=None):
    """
    Standalone SHAP explanation function.
    Args:
        model: Trained model
        X: DataFrame to explain
        feature_names: List of feature names
        mode: 'classification' or 'regression'
        training_data: DataFrame used for training (defaults to X)
        protected_attributes: List of protected attribute names
        justified_attributes: List of justified attribute names
    Returns:
        ExplainabilityResult
    """
    if training_data is None:
        training_data = X
    engine = ExplainabilityEngine(
        model=model,
        training_data=training_data,
        feature_names=feature_names,
        mode=mode,
        protected_attributes=protected_attributes,
        justified_attributes=justified_attributes
    )
    return engine.explain_with_shap(X)

def explain_with_lime(model, X, feature_names, mode="classification", training_data=None, protected_attributes=None, justified_attributes=None, num_samples=5, num_features=10):
    """
    Standalone LIME explanation function.
    Args:
        model: Trained model
        X: DataFrame to explain
        feature_names: List of feature names
        mode: 'classification' or 'regression'
        training_data: DataFrame used for training (defaults to X)
        protected_attributes: List of protected attribute names
        justified_attributes: List of justified attribute names
        num_samples: Number of samples to explain
        num_features: Number of features to show in explanation
    Returns:
        ExplainabilityResult
    """
    if training_data is None:
        training_data = X
    engine = ExplainabilityEngine(
        model=model,
        training_data=training_data,
        feature_names=feature_names,
        mode=mode,
        protected_attributes=protected_attributes,
        justified_attributes=justified_attributes
    )
    return engine.explain_with_lime(X, num_samples=num_samples, num_features=num_features)
