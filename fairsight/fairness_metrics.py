"""
Fairness Metrics Engine for Fairsight Toolkit
=============================================

Comprehensive fairness metrics computation with support for justified attributes
and multiple fairness definitions. Includes demographic parity, equalized odds,
predictive parity, and more.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Tuple, Dict, Union, List, Any
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    confusion_matrix, roc_auc_score, mean_squared_error, mean_absolute_error
)
from scipy import stats
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FairnessResult:
    """Container for fairness analysis results."""
    
    def __init__(
        self,
        metric_name: str,
        privileged_value: float,
        unprivileged_value: float,
        difference: float,
        ratio: float,
        threshold_met: bool,
        interpretation: str,
        details: Optional[Dict] = None
    ):
        self.metric_name = metric_name
        self.privileged_value = privileged_value
        self.unprivileged_value = unprivileged_value
        self.difference = difference
        self.ratio = ratio
        self.threshold_met = threshold_met
        self.interpretation = interpretation
        self.details = details or {}
    
    def to_dict(self):
        """Convert result to dictionary format."""
        return {
            "metric_name": self.metric_name,
            "privileged_value": self.privileged_value,
            "unprivileged_value": self.unprivileged_value,
            "difference": self.difference,
            "ratio": self.ratio,
            "threshold_met": self.threshold_met,
            "interpretation": self.interpretation,
            "details": self.details
        }

class FairnessMetrics:
    """
    Comprehensive fairness metrics calculator.
    
    Supports multiple fairness definitions:
    - Demographic Parity (Statistical Parity)
    - Equalized Odds (Equal Opportunity & Equal False Positive Rate)
    - Predictive Parity (Precision Parity)
    - Calibration
    - Individual Fairness approximations
    """
    
    def __init__(
        self, 
        y_true: Union[np.ndarray, pd.Series], 
        y_pred: Union[np.ndarray, pd.Series],
        y_prob: Optional[Union[np.ndarray, pd.Series]] = None,
        protected_attr: Union[np.ndarray, pd.Series] = None,
        privileged_group: Union[int, str] = 1,
        positive_label: Union[int, str] = 1,
        fairness_threshold: float = 0.8,
        justified_disparity: bool = False
    ):
        self.y_true = np.array(y_true)
        self.y_pred = np.array(y_pred)
        self.y_prob = np.array(y_prob) if y_prob is not None else None
        self.protected_attr = np.array(protected_attr) if protected_attr is not None else None
        self.privileged_group = privileged_group
        self.positive_label = positive_label
        self.fairness_threshold = fairness_threshold
        self.justified_disparity = justified_disparity
        
        self._validate_inputs()
        self.unprivileged_group = self._get_unprivileged_group()
    
    def _validate_inputs(self):
        """Validate input data."""
        if len(self.y_true) != len(self.y_pred):
            raise ValueError("y_true and y_pred must have the same length.")
        
        if self.protected_attr is not None and len(self.y_true) != len(self.protected_attr):
            raise ValueError("y_true and protected_attr must have the same length.")
        
        if self.y_prob is not None and len(self.y_true) != len(self.y_prob):
            raise ValueError("y_true and y_prob must have the same length.")
    
    def _get_unprivileged_group(self):
        """Identify unprivileged group."""
        if self.protected_attr is None:
            return None
            
        unique_vals = np.unique(self.protected_attr)
        unprivileged_groups = [val for val in unique_vals if val != self.privileged_group]
        
        if not unprivileged_groups:
            raise ValueError("Only one group found in protected attribute; need at least two for fairness analysis.")
        
        # Return the first unprivileged group (can be extended for multiple groups)
        return unprivileged_groups[0]
    
    def _safe_divide(self, numerator: float, denominator: float) -> float:
        """Safe division with zero handling."""
        if denominator == 0:
            return 0.0 if numerator == 0 else float('inf')
        return numerator / denominator
    
    def _group_metrics(self, group_value: Union[int, str]) -> Dict[str, float]:
        """Compute basic metrics for a specific group."""
        if self.protected_attr is None:
            raise ValueError("Protected attribute not provided.")
            
        mask = self.protected_attr == group_value
        y_true_group = self.y_true[mask]
        y_pred_group = self.y_pred[mask]
        
        if len(y_true_group) == 0:
            return {
                "positive_rate": 0.0,
                "true_positive_rate": 0.0,
                "false_positive_rate": 0.0,
                "true_negative_rate": 0.0,
                "false_negative_rate": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1_score": 0.0,
                "accuracy": 0.0,
                "size": 0
            }
        
        # Basic counts
        tp = np.sum((y_pred_group == self.positive_label) & (y_true_group == self.positive_label))
        fp = np.sum((y_pred_group == self.positive_label) & (y_true_group != self.positive_label))
        fn = np.sum((y_pred_group != self.positive_label) & (y_true_group == self.positive_label))
        tn = np.sum((y_pred_group != self.positive_label) & (y_true_group != self.positive_label))
        
        total = len(y_true_group)
        positive_actual = np.sum(y_true_group == self.positive_label)
        negative_actual = total - positive_actual
        
        return {
            "positive_rate": self._safe_divide(tp + fp, total),  # P(Y_hat = 1)
            "true_positive_rate": self._safe_divide(tp, positive_actual),  # Sensitivity/Recall
            "false_positive_rate": self._safe_divide(fp, negative_actual),  # 1 - Specificity
            "true_negative_rate": self._safe_divide(tn, negative_actual),  # Specificity
            "false_negative_rate": self._safe_divide(fn, positive_actual),  # Miss rate
            "precision": self._safe_divide(tp, tp + fp),  # Positive predictive value
            "recall": self._safe_divide(tp, tp + fn),  # Same as TPR
            "f1_score": self._safe_divide(2 * tp, 2 * tp + fp + fn),
            "accuracy": self._safe_divide(tp + tn, total),
            "size": total
        }
    
    def demographic_parity(self) -> FairnessResult:
        """
        Compute Demographic Parity (Statistical Parity).
        Measures whether positive prediction rates are equal across groups.
        """
        privileged_metrics = self._group_metrics(self.privileged_group)
        unprivileged_metrics = self._group_metrics(self.unprivileged_group)
        
        priv_rate = privileged_metrics["positive_rate"]
        unpriv_rate = unprivileged_metrics["positive_rate"]
        
        difference = unpriv_rate - priv_rate
        ratio = self._safe_divide(unpriv_rate, priv_rate)
        
        # Check fairness threshold (80% rule)
        threshold_met = ratio >= self.fairness_threshold and ratio <= (1/self.fairness_threshold)
        
        if self.justified_disparity:
            interpretation = f"Disparity is business-justified. Ratio: {ratio:.3f}"
        elif threshold_met:
            interpretation = f"Demographic parity achieved. Ratio: {ratio:.3f}"
        else:
            interpretation = f"Demographic parity violated. Ratio: {ratio:.3f} (should be ≥ {self.fairness_threshold})"
        
        return FairnessResult(
            metric_name="Demographic Parity",
            privileged_value=priv_rate,
            unprivileged_value=unpriv_rate,
            difference=difference,
            ratio=ratio,
            threshold_met=threshold_met or self.justified_disparity,
            interpretation=interpretation,
            details={
                "privileged_group_size": privileged_metrics["size"],
                "unprivileged_group_size": unprivileged_metrics["size"]
            }
        )
    
    def equalized_odds(self) -> Tuple[FairnessResult, FairnessResult]:
        """
        Compute Equalized Odds (Equal Opportunity + Equal False Positive Rate).
        Returns tuple of (Equal Opportunity, Equal FPR) results.
        """
        privileged_metrics = self._group_metrics(self.privileged_group)
        unprivileged_metrics = self._group_metrics(self.unprivileged_group)
        
        # Equal Opportunity (TPR equality)
        priv_tpr = privileged_metrics["true_positive_rate"]
        unpriv_tpr = unprivileged_metrics["true_positive_rate"]
        tpr_diff = unpriv_tpr - priv_tpr
        tpr_ratio = self._safe_divide(unpriv_tpr, priv_tpr)
        
        tpr_threshold_met = abs(tpr_diff) <= (1 - self.fairness_threshold)
        
        if self.justified_disparity:
            tpr_interpretation = f"TPR disparity is business-justified. Difference: {tpr_diff:.3f}"
        else:
            tpr_interpretation = f"Equal opportunity {'achieved' if tpr_threshold_met else 'violated'}. TPR difference: {tpr_diff:.3f}"
        
        equal_opportunity = FairnessResult(
            metric_name="Equal Opportunity",
            privileged_value=priv_tpr,
            unprivileged_value=unpriv_tpr,
            difference=tpr_diff,
            ratio=tpr_ratio,
            threshold_met=tpr_threshold_met or self.justified_disparity,
            interpretation=tpr_interpretation
        )
        
        # Equal False Positive Rate
        priv_fpr = privileged_metrics["false_positive_rate"]
        unpriv_fpr = unprivileged_metrics["false_positive_rate"]
        fpr_diff = unpriv_fpr - priv_fpr
        fpr_ratio = self._safe_divide(unpriv_fpr, priv_fpr)
        
        fpr_threshold_met = abs(fpr_diff) <= (1 - self.fairness_threshold)
        
        if self.justified_disparity:
            fpr_interpretation = f"FPR disparity is business-justified. Difference: {fpr_diff:.3f}"
        else:
            fpr_interpretation = f"Equal FPR {'achieved' if fpr_threshold_met else 'violated'}. FPR difference: {fpr_diff:.3f}"
        
        equal_fpr = FairnessResult(
            metric_name="Equal False Positive Rate",
            privileged_value=priv_fpr,
            unprivileged_value=unpriv_fpr,
            difference=fpr_diff,
            ratio=fpr_ratio,
            threshold_met=fpr_threshold_met or self.justified_disparity,
            interpretation=fpr_interpretation
        )
        
        return equal_opportunity, equal_fpr
    
    def predictive_parity(self) -> FairnessResult:
        """
        Compute Predictive Parity (Precision Parity).
        Measures whether precision is equal across groups.
        """
        privileged_metrics = self._group_metrics(self.privileged_group)
        unprivileged_metrics = self._group_metrics(self.unprivileged_group)
        
        priv_precision = privileged_metrics["precision"]
        unpriv_precision = unprivileged_metrics["precision"]
        
        difference = unpriv_precision - priv_precision
        ratio = self._safe_divide(unpriv_precision, priv_precision)
        
        threshold_met = abs(difference) <= (1 - self.fairness_threshold)
        
        if self.justified_disparity:
            interpretation = f"Precision disparity is business-justified. Difference: {difference:.3f}"
        else:
            interpretation = f"Predictive parity {'achieved' if threshold_met else 'violated'}. Precision difference: {difference:.3f}"
        
        return FairnessResult(
            metric_name="Predictive Parity",
            privileged_value=priv_precision,
            unprivileged_value=unpriv_precision,
            difference=difference,
            ratio=ratio,
            threshold_met=threshold_met or self.justified_disparity,
            interpretation=interpretation
        )
    
    def calibration(self) -> Optional[FairnessResult]:
        """
        Compute Calibration fairness.
        Requires probability predictions (y_prob).
        """
        if self.y_prob is None:
            logger.warning("Calibration requires probability predictions (y_prob)")
            return None
        
        # Group data
        priv_mask = self.protected_attr == self.privileged_group
        unpriv_mask = self.protected_attr == self.unprivileged_group
        
        priv_probs = self.y_prob[priv_mask]
        priv_actual = self.y_true[priv_mask]
        unpriv_probs = self.y_prob[unpriv_mask]
        unpriv_actual = self.y_true[unpriv_mask]
        
        # Compute calibration using binning
        n_bins = 10
        
        def compute_calibration_error(probs, actual):
            bin_boundaries = np.linspace(0, 1, n_bins + 1)
            bin_lowers = bin_boundaries[:-1]
            bin_uppers = bin_boundaries[1:]
            
            ece = 0
            for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
                in_bin = (probs > bin_lower) & (probs <= bin_upper)
                prop_in_bin = in_bin.mean()
                
                if prop_in_bin > 0:
                    accuracy_in_bin = actual[in_bin].mean()
                    avg_confidence_in_bin = probs[in_bin].mean()
                    ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            
            return ece
        
        priv_cal_error = compute_calibration_error(priv_probs, priv_actual)
        unpriv_cal_error = compute_calibration_error(unpriv_probs, unpriv_actual)
        
        difference = unpriv_cal_error - priv_cal_error
        
        # Lower calibration error is better
        threshold_met = abs(difference) <= 0.1  # 10% threshold
        
        if self.justified_disparity:
            interpretation = f"Calibration disparity is business-justified. Difference: {difference:.3f}"
        else:
            interpretation = f"Calibration fairness {'achieved' if threshold_met else 'violated'}. Error difference: {difference:.3f}"
        
        return FairnessResult(
            metric_name="Calibration",
            privileged_value=priv_cal_error,
            unprivileged_value=unpriv_cal_error,
            difference=difference,
            ratio=self._safe_divide(unpriv_cal_error, priv_cal_error),
            threshold_met=threshold_met or self.justified_disparity,
            interpretation=interpretation,
            details={
                "n_bins": n_bins,
                "privileged_calibration_error": priv_cal_error,
                "unprivileged_calibration_error": unpriv_cal_error
            }
        )
    
    def overall_performance_gap(self) -> Dict[str, FairnessResult]:
        """Compute performance gaps across different metrics."""
        privileged_metrics = self._group_metrics(self.privileged_group)
        unprivileged_metrics = self._group_metrics(self.unprivileged_group)
        
        performance_gaps = {}
        
        metrics_to_compare = ["accuracy", "precision", "recall", "f1_score"]
        
        for metric in metrics_to_compare:
            priv_val = privileged_metrics[metric]
            unpriv_val = unprivileged_metrics[metric]
            
            difference = unpriv_val - priv_val
            ratio = self._safe_divide(unpriv_val, priv_val)
            
            # Performance gaps should be small
            threshold_met = abs(difference) <= 0.1  # 10% threshold
            
            if self.justified_disparity:
                interpretation = f"{metric.capitalize()} gap is business-justified. Difference: {difference:.3f}"
            else:
                interpretation = f"{metric.capitalize()} gap {'acceptable' if threshold_met else 'concerning'}. Difference: {difference:.3f}"
            
            performance_gaps[metric] = FairnessResult(
                metric_name=f"{metric.capitalize()} Gap",
                privileged_value=priv_val,
                unprivileged_value=unpriv_val,
                difference=difference,
                ratio=ratio,
                threshold_met=threshold_met or self.justified_disparity,
                interpretation=interpretation
            )
        
        return performance_gaps
    
    def evaluate(self):
        """Alias for compute_all_metrics for user convenience."""
        return self.compute_all_metrics()
    
    def compute_all_metrics(self) -> Dict[str, Union[FairnessResult, Dict[str, FairnessResult]]]:
        """Compute all available fairness metrics."""
        results = {}
        
        if self.protected_attr is None:
            logger.warning("No protected attribute provided. Cannot compute group fairness metrics.")
            return results
        
        # Core fairness metrics
        results["demographic_parity"] = self.demographic_parity()
        
        equal_opp, equal_fpr = self.equalized_odds()
        results["equal_opportunity"] = equal_opp
        results["equal_false_positive_rate"] = equal_fpr
        
        results["predictive_parity"] = self.predictive_parity()
        
        # Calibration (if probabilities available)
        calibration_result = self.calibration()
        if calibration_result:
            results["calibration"] = calibration_result
        
        # Performance gaps
        results["performance_gaps"] = self.overall_performance_gap()
        
        return results


class FairnessEngine:
    """
    High-level fairness analysis engine with visualization and reporting capabilities.
    """
    
    def __init__(
        self,
        justified_attributes: Optional[List[str]] = None,
        fairness_threshold: float = 0.8,
        output_dir: str = "/tmp/fairsight_fairness"
    ):
        self.justified_attributes = justified_attributes or []
        self.fairness_threshold = fairness_threshold
        self.output_dir = output_dir
        self._ensure_output_dir()
    
    def _ensure_output_dir(self):
        """Create output directory for visualizations."""
        import os
        os.makedirs(self.output_dir, exist_ok=True)
    
    def analyze_fairness(
        self,
        y_true: Union[np.ndarray, pd.Series],
        y_pred: Union[np.ndarray, pd.Series],
        protected_attributes: Dict[str, Union[np.ndarray, pd.Series]],
        y_prob: Optional[Union[np.ndarray, pd.Series]] = None,
        privileged_groups: Optional[Dict[str, Union[int, str]]] = None
    ) -> Dict[str, Dict[str, Union[FairnessResult, Dict[str, FairnessResult]]]]:
        """
        Comprehensive fairness analysis across multiple protected attributes.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            protected_attributes: Dict of {attr_name: attr_values}
            y_prob: Predicted probabilities (optional)
            privileged_groups: Dict of {attr_name: privileged_value} (optional)
            
        Returns:
            Nested dict of fairness results
        """
        results = {}
        
        for attr_name, attr_values in protected_attributes.items():
            logger.info(f"Analyzing fairness for attribute: {attr_name}")
            
            # Determine if this attribute should be treated as justified
            is_justified = attr_name in self.justified_attributes
            
            # Determine privileged group
            if privileged_groups and attr_name in privileged_groups:
                privileged_group = privileged_groups[attr_name]
            else:
                # Default: assume the most frequent class is privileged
                unique_vals, counts = np.unique(attr_values, return_counts=True)
                privileged_group = unique_vals[np.argmax(counts)]
                logger.info(f"Auto-detected privileged group for {attr_name}: {privileged_group}")
            
            # Create fairness calculator
            fairness_calc = FairnessMetrics(
                y_true=y_true,
                y_pred=y_pred,
                y_prob=y_prob,
                protected_attr=attr_values,
                privileged_group=privileged_group,
                fairness_threshold=self.fairness_threshold,
                justified_disparity=is_justified
            )
            
            # Compute all metrics
            attr_results = fairness_calc.compute_all_metrics()
            results[attr_name] = attr_results
        
        return results
    
    def create_fairness_dashboard(
        self,
        fairness_results: Dict[str, Dict[str, Union[FairnessResult, Dict[str, FairnessResult]]]],
        title: str = "Fairness Analysis Dashboard"
    ) -> str:
        """
        Create a comprehensive fairness dashboard visualization.
        
        Args:
            fairness_results: Results from analyze_fairness()
            title: Dashboard title
            
        Returns:
            Path to saved dashboard image
        """
        n_attrs = len(fairness_results)
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # Flatten axes for easier indexing
        axes = axes.flatten()
        
        # Colors for different fairness levels
        colors = {
            'pass': '#2E8B57',  # Sea Green
            'concern': '#FF8C00',  # Dark Orange
            'fail': '#DC143C'   # Crimson
        }
        
        # 1. Demographic Parity Ratios
        attr_names = list(fairness_results.keys())
        dp_ratios = []
        dp_colors = []
        
        for attr_name in attr_names:
            dp_result = fairness_results[attr_name].get("demographic_parity")
            if dp_result:
                ratio = dp_result.ratio
                dp_ratios.append(ratio)
                
                if dp_result.threshold_met:
                    dp_colors.append(colors['pass'])
                elif 0.6 <= ratio <= 1.67:  # Moderate disparity
                    dp_colors.append(colors['concern'])
                else:
                    dp_colors.append(colors['fail'])
            else:
                dp_ratios.append(0)
                dp_colors.append(colors['fail'])
        
        axes[0].bar(attr_names, dp_ratios, color=dp_colors)
        axes[0].set_title('Demographic Parity Ratios')
        axes[0].set_ylabel('Ratio (Unprivileged/Privileged)')
        axes[0].axhline(y=0.8, color='red', linestyle='--', alpha=0.7, label='Min Threshold')
        axes[0].axhline(y=1.25, color='red', linestyle='--', alpha=0.7, label='Max Threshold')
        axes[0].legend()
        axes[0].set_ylim(0, 2)
        
        # 2. Equal Opportunity Differences
        eo_diffs = []
        eo_colors = []
        
        for attr_name in attr_names:
            eo_result = fairness_results[attr_name].get("equal_opportunity")
            if eo_result:
                diff = abs(eo_result.difference)
                eo_diffs.append(diff)
                
                if eo_result.threshold_met:
                    eo_colors.append(colors['pass'])
                elif diff <= 0.15:
                    eo_colors.append(colors['concern'])
                else:
                    eo_colors.append(colors['fail'])
            else:
                eo_diffs.append(0)
                eo_colors.append(colors['fail'])
        
        axes[1].bar(attr_names, eo_diffs, color=eo_colors)
        axes[1].set_title('Equal Opportunity Differences (Absolute)')
        axes[1].set_ylabel('|TPR Difference|')
        axes[1].axhline(y=0.1, color='red', linestyle='--', alpha=0.7, label='Threshold')
        axes[1].legend()
        
        # 3. Performance Gaps Summary
        performance_metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        gap_data = []
        
        for attr_name in attr_names:
            attr_gaps = []
            perf_gaps = fairness_results[attr_name].get("performance_gaps", {})
            
            for metric in performance_metrics:
                if metric in perf_gaps:
                    attr_gaps.append(abs(perf_gaps[metric].difference))
                else:
                    attr_gaps.append(0)
            
            gap_data.append(attr_gaps)
        
        gap_df = pd.DataFrame(gap_data, index=attr_names, columns=performance_metrics)
        
        im = axes[2].imshow(gap_df.values, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=0.2)
        axes[2].set_title('Performance Gaps Heatmap')
        axes[2].set_xticks(range(len(performance_metrics)))
        axes[2].set_xticklabels(performance_metrics, rotation=45)
        axes[2].set_yticks(range(len(attr_names)))
        axes[2].set_yticklabels(attr_names)
        
        # Add colorbar
        plt.colorbar(im, ax=axes[2], label='Absolute Difference')
        
        # 4. Overall Fairness Score
        overall_scores = []
        score_colors = []
        
        for attr_name in attr_names:
            attr_results = fairness_results[attr_name]
            
            # Simple scoring: count passed metrics
            total_metrics = 0
            passed_metrics = 0
            
            for result_key, result_val in attr_results.items():
                if isinstance(result_val, FairnessResult):
                    total_metrics += 1
                    if result_val.threshold_met:
                        passed_metrics += 1
                elif isinstance(result_val, dict):
                    for sub_result in result_val.values():
                        if isinstance(sub_result, FairnessResult):
                            total_metrics += 1
                            if sub_result.threshold_met:
                                passed_metrics += 1
            
            score = (passed_metrics / total_metrics * 100) if total_metrics > 0 else 0
            overall_scores.append(score)
            
            if score >= 80:
                score_colors.append(colors['pass'])
            elif score >= 60:
                score_colors.append(colors['concern'])
            else:
                score_colors.append(colors['fail'])
        
        axes[3].bar(attr_names, overall_scores, color=score_colors)
        axes[3].set_title('Overall Fairness Scores')
        axes[3].set_ylabel('Score (%)')
        axes[3].set_ylim(0, 100)
        axes[3].axhline(y=80, color='green', linestyle='--', alpha=0.7, label='Good Threshold')
        axes[3].axhline(y=60, color='orange', linestyle='--', alpha=0.7, label='Acceptable Threshold')
        axes[3].legend()
        
        plt.tight_layout()
        
        # Save dashboard
        dashboard_path = f"{self.output_dir}/fairness_dashboard.png"
        plt.savefig(dashboard_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return dashboard_path
    
    def generate_fairness_report(
        self,
        fairness_results: Dict[str, Dict[str, Union[FairnessResult, Dict[str, FairnessResult]]]],
        model_name: str = "Model"
    ) -> str:
        """Generate a comprehensive fairness analysis report."""
        report_lines = [
            f"# 🎯 Fairness Analysis Report: {model_name}\n",
            f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
            f"Fairness Threshold: {self.fairness_threshold * 100:.0f}% (80% Rule)\n",
            f"Justified Attributes: {', '.join(self.justified_attributes) if self.justified_attributes else 'None'}\n",
            "---\n"
        ]
        
        for attr_name, attr_results in fairness_results.items():
            is_justified = attr_name in self.justified_attributes
            justified_note = " (Business Justified)" if is_justified else ""
            
            report_lines.extend([
                f"## 📊 {attr_name.title()}{justified_note}\n"
            ])
            
            # Overall assessment
            passed_metrics = 0
            total_metrics = 0
            
            for result_key, result_val in attr_results.items():
                if isinstance(result_val, FairnessResult):
                    total_metrics += 1
                    if result_val.threshold_met:
                        passed_metrics += 1
                elif isinstance(result_val, dict):
                    for sub_result in result_val.values():
                        if isinstance(sub_result, FairnessResult):
                            total_metrics += 1
                            if sub_result.threshold_met:
                                passed_metrics += 1
            
            fairness_score = (passed_metrics / total_metrics * 100) if total_metrics > 0 else 0
            
            if fairness_score >= 80:
                assessment = "🟢 GOOD"
            elif fairness_score >= 60:
                assessment = "🟡 ACCEPTABLE"
            else:
                assessment = "🔴 CONCERNING"
            
            report_lines.append(f"**Overall Assessment:** {assessment} ({fairness_score:.0f}% metrics passed)\n")
            
            # Individual metrics
            report_lines.append("### Detailed Metrics\n")
            
            for result_key, result_val in attr_results.items():
                if isinstance(result_val, FairnessResult):
                    status = "✅" if result_val.threshold_met else "❌"
                    report_lines.extend([
                        f"#### {status} {result_val.metric_name}",
                        f"- **Privileged Group:** {result_val.privileged_value:.3f}",
                        f"- **Unprivileged Group:** {result_val.unprivileged_value:.3f}",
                        f"- **Difference:** {result_val.difference:.3f}",
                        f"- **Ratio:** {result_val.ratio:.3f}",
                        f"- **Interpretation:** {result_val.interpretation}\n"
                    ])
                elif isinstance(result_val, dict) and result_key == "performance_gaps":
                    report_lines.append("#### Performance Gaps\n")
                    for gap_name, gap_result in result_val.items():
                        status = "✅" if gap_result.threshold_met else "❌"
                        report_lines.append(
                            f"- {status} **{gap_result.metric_name}:** {gap_result.difference:.3f}"
                        )
                    report_lines.append("")
            
            report_lines.append("---\n")
        
        # Summary and recommendations
        report_lines.extend([
            "## 📋 Summary & Recommendations\n"
        ])
        
        total_attrs = len(fairness_results)
        concerning_attrs = []
        
        for attr_name, attr_results in fairness_results.items():
            passed = sum(
                1 for result in attr_results.values()
                if isinstance(result, FairnessResult) and result.threshold_met
            )
            total = sum(
                1 for result in attr_results.values()
                if isinstance(result, FairnessResult)
            )
            
            if total > 0 and (passed / total) < 0.6:
                concerning_attrs.append(attr_name)
        
        if not concerning_attrs:
            report_lines.append("✅ **Overall Assessment:** Your model demonstrates good fairness across all analyzed attributes.")
        else:
            report_lines.extend([
                f"⚠️ **Concerning Attributes:** {', '.join(concerning_attrs)}",
                "",
                "**Recommendations:**",
                "1. Review data collection and preprocessing for biased patterns",
                "2. Consider bias mitigation techniques during model training",
                "3. Implement post-processing fairness interventions",
                "4. Monitor fairness metrics continuously in production",
            ])
        
        report_content = "\n".join(report_lines)
        
        # Save report
        report_path = f"{self.output_dir}/fairness_analysis_report.md"
        with open(report_path, 'w') as f:
            f.write(report_content)
        
        return report_path


# Convenience functions
def compute_demographic_parity(y_true, y_pred, protected_attr, privileged_group=1):
    """Quick demographic parity calculation."""
    fm = FairnessMetrics(y_true, y_pred, protected_attr=protected_attr, privileged_group=privileged_group)
    return fm.demographic_parity()

def compute_equal_opportunity(y_true, y_pred, protected_attr, privileged_group=1):
    """Quick equal opportunity calculation."""
    fm = FairnessMetrics(y_true, y_pred, protected_attr=protected_attr, privileged_group=privileged_group)
    equal_opp, _ = fm.equalized_odds()
    return equal_opp

def compute_predictive_parity(y_true, y_pred, protected_attr, privileged_group=1):
    """Quick predictive parity calculation."""
    fm = FairnessMetrics(y_true, y_pred, protected_attr=protected_attr, privileged_group=privileged_group)
    return fm.predictive_parity()
