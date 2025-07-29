"""
Report Generator for Fairsight Toolkit
=====================================

Comprehensive report generation with support for multiple formats (HTML, PDF, Markdown),
visualizations, and SAP HANA integration. Handles bias analysis, fairness metrics,
and explainability results.
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Circle, Rectangle
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Tuple
import base64
from io import BytesIO
import logging
import pkg_resources

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import additional dependencies
try:
    from jinja2 import Template, Environment, FileSystemLoader
    JINJA2_AVAILABLE = True
except ImportError:
    JINJA2_AVAILABLE = False
    logger.warning("Jinja2 not available. Advanced templating will be disabled.")

# Remove WeasyPrint imports and logic
# Add FPDF import
from fpdf import FPDF


class ReportGenerator:
    """
    Comprehensive report generator for fairness and bias analysis.
    
    Features:
    - Multiple output formats (HTML, PDF, Markdown, JSON)
    - Interactive visualizations
    - Executive summaries
    - Detailed technical analysis
    - SAP HANA integration
    - Template-based generation
    """
    
    def __init__(
        self,
        output_dir: str = "/tmp/fairsight_reports",
        template_dir: Optional[str] = None,
        company_name: str = "Your Organization",
        logo_path: Optional[str] = None
    ):
        self.output_dir = output_dir
        self.template_dir = template_dir
        self.company_name = company_name
        self.logo_path = logo_path
        
        self._ensure_output_dir()
        self._setup_templates()
        
        # Styling configuration
        self.colors = {
            'primary': '#2E8B57',      # Sea Green
            'secondary': '#4682B4',    # Steel Blue
            'success': '#28A745',      # Green
            'warning': '#FFC107',      # Amber
            'danger': '#DC3545',       # Red
            'info': '#17A2B8',         # Cyan
            'light': '#F8F9FA',        # Light Gray
            'dark': '#343A40'          # Dark Gray
        }
    
    def _ensure_output_dir(self):
        """Create output directory structure."""
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "assets"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "visualizations"), exist_ok=True)
    
    def _setup_templates(self):
        """Setup Jinja2 templates."""
        if JINJA2_AVAILABLE and self.template_dir:
            self.jinja_env = Environment(loader=FileSystemLoader(self.template_dir))
        else:
            self.jinja_env = None
    
    def calculate_ethical_score(
        self,
        bias_results: Dict[str, Any],
        fairness_results: Dict[str, Any],
        justified_attributes: Optional[List[str]] = None
    ) -> Tuple[int, Dict[str, Any]]:
        """
        Calculate overall ethical score based on bias and fairness analysis.
        
        Args:
            bias_results: Results from bias detection
            fairness_results: Results from fairness analysis
            justified_attributes: List of business-justified attributes
            
        Returns:
            Tuple of (ethical_score, score_breakdown)
        """
        justified_attributes = justified_attributes or []
        score = 100
        deductions = 0
        breakdown = {
            'bias_deductions': 0,
            'fairness_deductions': 0,
            'performance_deductions': 0,
            'total_deductions': 0,
            'final_score': 100,
            'grade': 'A+',
            'assessment': 'Excellent'
        }
        
        try:
            # Analyze bias results
            for attr_name, attr_results in bias_results.items():
                # Skip justified attributes or apply lighter penalties
                is_justified = attr_name in justified_attributes
                penalty_multiplier = 0.3 if is_justified else 1.0
                
                if isinstance(attr_results, dict):
                    # Disparate Impact violations
                    di = attr_results.get("disparate_impact", 1.0)
                    if di < 0.6:
                        deductions += 20 * penalty_multiplier
                    elif di < 0.8:
                        deductions += 10 * penalty_multiplier
                    
                    # Statistical Parity violations
                    spd = abs(attr_results.get("statistical_parity_difference", 0))
                    if spd > 0.2:
                        deductions += 15 * penalty_multiplier
                    elif spd > 0.1:
                        deductions += 7 * penalty_multiplier
                    
                    # Equal Opportunity violations
                    eod = abs(attr_results.get("equal_opportunity_difference", 0))
                    if eod > 0.15:
                        deductions += 12 * penalty_multiplier
                    elif eod > 0.1:
                        deductions += 6 * penalty_multiplier
            
            breakdown['bias_deductions'] = deductions
            
            # Analyze fairness results
            fairness_deductions = 0
            for attr_name, attr_results in fairness_results.items():
                is_justified = attr_name in justified_attributes
                penalty_multiplier = 0.3 if is_justified else 1.0
                
                if isinstance(attr_results, dict):
                    # Count failed fairness metrics
                    failed_metrics = 0
                    total_metrics = 0
                    
                    for metric_name, metric_result in attr_results.items():
                        if hasattr(metric_result, 'threshold_met'):
                            total_metrics += 1
                            if not metric_result.threshold_met and not is_justified:
                                failed_metrics += 1
                        elif isinstance(metric_result, dict):
                            for sub_metric in metric_result.values():
                                if hasattr(sub_metric, 'threshold_met'):
                                    total_metrics += 1
                                    if not sub_metric.threshold_met and not is_justified:
                                        failed_metrics += 1
                    
                    if total_metrics > 0:
                        failure_rate = failed_metrics / total_metrics
                        fairness_deductions += failure_rate * 25 * penalty_multiplier
            
            breakdown['fairness_deductions'] = fairness_deductions
            deductions += fairness_deductions
            
            # Calculate final score
            breakdown['total_deductions'] = deductions
            final_score = max(0, int(score - deductions))
            breakdown['final_score'] = final_score
            
            # Assign grade and assessment
            if final_score >= 90:
                breakdown['grade'] = 'A+'
                breakdown['assessment'] = 'Excellent - Model demonstrates exceptional fairness'
            elif final_score >= 80:
                breakdown['grade'] = 'A'
                breakdown['assessment'] = 'Good - Model shows strong ethical integrity'
            elif final_score >= 70:
                breakdown['grade'] = 'B'
                breakdown['assessment'] = 'Acceptable - Minor fairness concerns identified'
            elif final_score >= 60:
                breakdown['grade'] = 'C'
                breakdown['assessment'] = 'Concerning - Moderate bias detected, review recommended'
            elif final_score >= 50:
                breakdown['grade'] = 'D'
                breakdown['assessment'] = 'Poor - Significant bias issues require immediate attention'
            else:
                breakdown['grade'] = 'F'
                breakdown['assessment'] = 'Critical - Model fails fairness standards, do not deploy'
            
            return final_score, breakdown
            
        except Exception as e:
            logger.error(f"Error calculating ethical score: {e}")
            return 50, {"error": str(e), "final_score": 50, "grade": "Unknown", "assessment": "Error in calculation"}
    
    def create_ethical_badge(
        self, 
        ethical_score: int, 
        grade: str = "A", 
        size: Tuple[int, int] = (300, 300)
    ) -> str:
        """
        Create an ethical score badge visualization.
        
        Args:
            ethical_score: Score from 0-100
            grade: Letter grade (A+, A, B, C, D, F)
            size: Badge size (width, height)
            
        Returns:
            Path to saved badge image
        """
        fig, ax = plt.subplots(figsize=(size[0]/100, size[1]/100))
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Color based on score
        if ethical_score >= 90:
            color = self.colors['success']
            bg_color = '#E8F5E8'
        elif ethical_score >= 80:
            color = self.colors['primary']
            bg_color = '#E8F4F0'
        elif ethical_score >= 70:
            color = self.colors['info']
            bg_color = '#E3F2FD'
        elif ethical_score >= 60:
            color = self.colors['warning']
            bg_color = '#FFF8E1'
        else:
            color = self.colors['danger']
            bg_color = '#FFEBEE'
        
        # Background circle
        bg_circle = Circle((0, 0), 1.1, facecolor=bg_color, edgecolor='none', alpha=0.3)
        ax.add_patch(bg_circle)
        
        # Main circle
        main_circle = Circle((0, 0), 1, facecolor='white', edgecolor=color, linewidth=8)
        ax.add_patch(main_circle)
        
        # Inner progress ring
        theta = np.linspace(0, 2 * np.pi * (ethical_score / 100), 100)
        x_ring = 0.85 * np.cos(theta)
        y_ring = 0.85 * np.sin(theta)
        ax.plot(x_ring, y_ring, color=color, linewidth=6)
        
        # Score text
        ax.text(0, 0.3, f"{ethical_score}", ha='center', va='center', 
                fontsize=48, fontweight='bold', color=color)
        ax.text(0, 0.05, "ETHICAL", ha='center', va='center', 
                fontsize=14, fontweight='bold', color=self.colors['dark'])
        ax.text(0, -0.15, "SCORE", ha='center', va='center', 
                fontsize=14, fontweight='bold', color=self.colors['dark'])
        
        # Grade
        ax.text(0, -0.4, f"Grade: {grade}", ha='center', va='center', 
                fontsize=16, fontweight='bold', color=color)
        
        # Save badge
        badge_path = os.path.join(self.output_dir, "assets", "ethical_badge.png")
        plt.savefig(badge_path, bbox_inches='tight', dpi=300, facecolor='white', 
                   edgecolor='none', transparent=False)
        plt.close()
        
        return badge_path
    
    def create_bias_summary_chart(
        self, 
        bias_results: Dict[str, Any],
        justified_attributes: Optional[List[str]] = None
    ) -> str:
        """Create a summary chart of bias analysis results."""
        justified_attributes = justified_attributes or []
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Bias Analysis Summary', fontsize=16, fontweight='bold')
        
        # Data preparation
        attributes = list(bias_results.keys())
        disparate_impacts = []
        stat_parity_diffs = []
        equal_opp_diffs = []
        colors_list = []
        
        for attr in attributes:
            attr_data = bias_results[attr]
            is_justified = attr in justified_attributes
            
            # Color coding
            if is_justified:
                color = self.colors['info']  # Blue for justified
            else:
                di = attr_data.get('disparate_impact', 1.0)
                if di >= 0.8:
                    color = self.colors['success']  # Green for good
                elif di >= 0.6:
                    color = self.colors['warning']  # Yellow for concerning
                else:
                    color = self.colors['danger']   # Red for critical
            
            colors_list.append(color)
            disparate_impacts.append(attr_data.get('disparate_impact', 1.0))
            stat_parity_diffs.append(abs(attr_data.get('statistical_parity_difference', 0)))
            equal_opp_diffs.append(abs(attr_data.get('equal_opportunity_difference', 0)))
        
        # 1. Disparate Impact Ratios
        bars1 = ax1.bar(attributes, disparate_impacts, color=colors_list)
        ax1.set_title('Disparate Impact Ratios')
        ax1.set_ylabel('Ratio (Should be ≥ 0.8)')
        ax1.axhline(y=0.8, color='red', linestyle='--', alpha=0.7, label='Min Threshold')
        ax1.axhline(y=1.0, color='green', linestyle='-', alpha=0.7, label='Perfect Parity')
        ax1.legend()
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars1, disparate_impacts):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{value:.2f}', ha='center', va='bottom', fontweight='bold')
        
        # 2. Statistical Parity Differences
        bars2 = ax2.bar(attributes, stat_parity_diffs, color=colors_list)
        ax2.set_title('Statistical Parity Differences (Absolute)')
        ax2.set_ylabel('Absolute Difference')
        ax2.axhline(y=0.1, color='orange', linestyle='--', alpha=0.7, label='Concern Threshold')
        ax2.axhline(y=0.2, color='red', linestyle='--', alpha=0.7, label='Critical Threshold')
        ax2.legend()
        ax2.tick_params(axis='x', rotation=45)
        
        for bar, value in zip(bars2, stat_parity_diffs):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 3. Equal Opportunity Differences
        bars3 = ax3.bar(attributes, equal_opp_diffs, color=colors_list)
        ax3.set_title('Equal Opportunity Differences (Absolute)')
        ax3.set_ylabel('Absolute Difference')
        ax3.axhline(y=0.1, color='orange', linestyle='--', alpha=0.7, label='Concern Threshold')
        ax3.axhline(y=0.15, color='red', linestyle='--', alpha=0.7, label='Critical Threshold')
        ax3.legend()
        ax3.tick_params(axis='x', rotation=45)
        
        for bar, value in zip(bars3, equal_opp_diffs):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 4. Legend/Summary
        ax4.axis('off')
        legend_elements = [
            plt.Rectangle((0, 0), 1, 1, facecolor=self.colors['success'], 
                         label='✅ Passes Fairness Tests'),
            plt.Rectangle((0, 0), 1, 1, facecolor=self.colors['warning'], 
                         label='⚠️ Concerning Bias Levels'),
            plt.Rectangle((0, 0), 1, 1, facecolor=self.colors['danger'], 
                         label='❌ Critical Bias Issues'),
            plt.Rectangle((0, 0), 1, 1, facecolor=self.colors['info'], 
                         label='ℹ️ Business Justified')
        ]
        ax4.legend(handles=legend_elements, loc='center', fontsize=12)
        ax4.set_title('Legend', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        chart_path = os.path.join(self.output_dir, "visualizations", "bias_summary_chart.png")
        plt.savefig(chart_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return chart_path
    
    def create_fairness_metrics_chart(
        self,
        fairness_results: Dict[str, Any]
    ) -> str:
        """Create a comprehensive fairness metrics visualization."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Fairness Metrics Analysis', fontsize=16, fontweight='bold')
        
        axes = axes.flatten()
        
        # Extract data for visualization
        attributes = list(fairness_results.keys())
        
        # 1. Demographic Parity
        dp_ratios = []
        dp_colors = []
        
        for attr in attributes:
            attr_data = fairness_results[attr]
            dp_result = attr_data.get('demographic_parity')
            
            if dp_result and hasattr(dp_result, 'ratio'):
                ratio = dp_result.ratio
                dp_ratios.append(ratio)
                
                if dp_result.threshold_met:
                    dp_colors.append(self.colors['success'])
                elif 0.6 <= ratio <= 1.67:
                    dp_colors.append(self.colors['warning'])
                else:
                    dp_colors.append(self.colors['danger'])
            else:
                dp_ratios.append(0)
                dp_colors.append(self.colors['danger'])
        
        axes[0].bar(attributes, dp_ratios, color=dp_colors)
        axes[0].set_title('Demographic Parity Ratios')
        axes[0].set_ylabel('Ratio (Unprivileged/Privileged)')
        axes[0].axhline(y=0.8, color='red', linestyle='--', alpha=0.7)
        axes[0].axhline(y=1.25, color='red', linestyle='--', alpha=0.7)
        axes[0].tick_params(axis='x', rotation=45)
        
        # 2. Equal Opportunity
        eo_diffs = []
        eo_colors = []
        
        for attr in attributes:
            attr_data = fairness_results[attr]
            eo_result = attr_data.get('equal_opportunity')
            
            if eo_result and hasattr(eo_result, 'difference'):
                diff = abs(eo_result.difference)
                eo_diffs.append(diff)
                
                if eo_result.threshold_met:
                    eo_colors.append(self.colors['success'])
                elif diff <= 0.15:
                    eo_colors.append(self.colors['warning'])
                else:
                    eo_colors.append(self.colors['danger'])
            else:
                eo_diffs.append(0)
                eo_colors.append(self.colors['danger'])
        
        axes[1].bar(attributes, eo_diffs, color=eo_colors)
        axes[1].set_title('Equal Opportunity Differences')
        axes[1].set_ylabel('|TPR Difference|')
        axes[1].axhline(y=0.1, color='orange', linestyle='--', alpha=0.7)
        axes[1].tick_params(axis='x', rotation=45)
        
        # 3. Performance Gaps Heatmap
        performance_data = []
        performance_metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        
        for attr in attributes:
            attr_data = fairness_results[attr]
            perf_gaps = attr_data.get('performance_gaps', {})
            
            attr_perf = []
            for metric in performance_metrics:
                if metric in perf_gaps and hasattr(perf_gaps[metric], 'difference'):
                    attr_perf.append(abs(perf_gaps[metric].difference))
                else:
                    attr_perf.append(0)
            
            performance_data.append(attr_perf)
        
        if performance_data:
            perf_df = pd.DataFrame(performance_data, 
                                 index=attributes, 
                                 columns=performance_metrics)
            
            im = axes[2].imshow(perf_df.values, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=0.2)
            axes[2].set_title('Performance Gaps Heatmap')
            axes[2].set_xticks(range(len(performance_metrics)))
            axes[2].set_xticklabels(performance_metrics, rotation=45)
            axes[2].set_yticks(range(len(attributes)))
            axes[2].set_yticklabels(attributes)
            
            # Add values to heatmap
            for i in range(len(attributes)):
                for j in range(len(performance_metrics)):
                    text = axes[2].text(j, i, f'{perf_df.values[i, j]:.3f}',
                                      ha="center", va="center", color="black", fontweight='bold')
        
        # 4. Overall Fairness Scores
        overall_scores = []
        score_colors = []
        
        for attr in attributes:
            attr_data = fairness_results[attr]
            
            passed = 0
            total = 0
            
            for result_key, result_val in attr_data.items():
                if hasattr(result_val, 'threshold_met'):
                    total += 1
                    if result_val.threshold_met:
                        passed += 1
                elif isinstance(result_val, dict):
                    for sub_result in result_val.values():
                        if hasattr(sub_result, 'threshold_met'):
                            total += 1
                            if sub_result.threshold_met:
                                passed += 1
            
            score = (passed / total * 100) if total > 0 else 0
            overall_scores.append(score)
            
            if score >= 80:
                score_colors.append(self.colors['success'])
            elif score >= 60:
                score_colors.append(self.colors['warning'])
            else:
                score_colors.append(self.colors['danger'])
        
        axes[3].bar(attributes, overall_scores, color=score_colors)
        axes[3].set_title('Overall Fairness Scores')
        axes[3].set_ylabel('Score (%)')
        axes[3].set_ylim(0, 100)
        axes[3].axhline(y=80, color='green', linestyle='--', alpha=0.7)
        axes[3].axhline(y=60, color='orange', linestyle='--', alpha=0.7)
        axes[3].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        chart_path = os.path.join(self.output_dir, "visualizations", "fairness_metrics_chart.png")
        plt.savefig(chart_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return chart_path
    
    def generate_executive_summary(
        self,
        model_name: str,
        ethical_score: int,
        score_breakdown: Dict[str, Any],
        bias_results: Dict[str, Any],
        fairness_results: Dict[str, Any],
        justified_attributes: Optional[List[str]] = None
    ) -> str:
        """Generate an executive summary section."""
        justified_attributes = justified_attributes or []
        
        summary_lines = [
            f"## 📋 Executive Summary\n",
            f"**Model:** {model_name}  ",
            f"**Analysis Date:** {datetime.now().strftime('%B %d, %Y')}  ",
            f"**Ethical Score:** {ethical_score}/100 (Grade: {score_breakdown.get('grade', 'N/A')})  ",
            f"**Assessment:** {score_breakdown.get('assessment', 'Assessment not available')}\n",
        ]
        
        # Key findings
        summary_lines.append("### 🎯 Key Findings\n")
        
        # Analyze critical issues
        critical_issues = []
        concerning_issues = []
        good_attributes = []
        
        for attr_name, attr_data in bias_results.items():
            is_justified = attr_name in justified_attributes
            
            di = attr_data.get('disparate_impact', 1.0)
            spd = abs(attr_data.get('statistical_parity_difference', 0))
            
            if not is_justified:
                if di < 0.6 or spd > 0.2:
                    critical_issues.append(f"**{attr_name}** (DI: {di:.2f}, SPD: {spd:.3f})")
                elif di < 0.8 or spd > 0.1:
                    concerning_issues.append(f"**{attr_name}** (DI: {di:.2f}, SPD: {spd:.3f})")
                else:
                    good_attributes.append(f"**{attr_name}**")
            else:
                good_attributes.append(f"**{attr_name}** (Business Justified)")
        
        if critical_issues:
            summary_lines.extend([
                "#### 🔴 Critical Issues Requiring Immediate Attention:",
                f"- {', '.join(critical_issues)}",
                ""
            ])
        
        if concerning_issues:
            summary_lines.extend([
                "#### 🟡 Areas of Concern:",
                f"- {', '.join(concerning_issues)}",
                ""
            ])
        
        if good_attributes:
            summary_lines.extend([
                "#### ✅ Attributes Meeting Fairness Standards:",
                f"- {', '.join(good_attributes)}",
                ""
            ])
        
        # Recommendations
        summary_lines.append("### 📌 Primary Recommendations\n")
        
        if ethical_score >= 90:
            summary_lines.extend([
                "✅ **Excellent fairness performance.** Your model demonstrates strong ethical integrity.",
                "- Continue monitoring fairness metrics in production",
                "- Consider this model as a best practice example",
                ""
            ])
        elif ethical_score >= 80:
            summary_lines.extend([
                "✅ **Good fairness performance with minor areas for improvement.**",
                "- Address any remaining fairness gaps through post-processing",
                "- Implement continuous monitoring",
                ""
            ])
        elif ethical_score >= 70:
            summary_lines.extend([
                "⚠️ **Acceptable performance with moderate fairness concerns.**",
                "- Review data preprocessing and feature engineering",
                "- Consider bias mitigation techniques during training",
                "- Implement fairness-aware algorithms",
                ""
            ])
        elif ethical_score >= 50:
            summary_lines.extend([
                "🔴 **Significant fairness issues detected.**",
                "- **Do not deploy without bias mitigation**",
                "- Comprehensive review of training data required",
                "- Consider alternative modeling approaches",
                "- Implement strict fairness constraints",
                ""
            ])
        else:
            summary_lines.extend([
                "🚨 **Critical fairness violations - Model fails ethical standards.**",
                "- **STOP: Do not deploy this model**",
                "- Immediate remediation required",
                "- Complete retraining with fairness-first approach",
                "- Seek expert consultation on bias mitigation",
                ""
            ])
        
        return "\n".join(summary_lines)
    
    def generate_detailed_analysis(
        self,
        bias_results: Dict[str, Any],
        fairness_results: Dict[str, Any],
        explainability_results: Optional[List[Any]] = None,
        justified_attributes: Optional[List[str]] = None
    ) -> str:
        """Generate detailed technical analysis section."""
        justified_attributes = justified_attributes or []
        
        analysis_lines = [
            "## 🔬 Detailed Technical Analysis\n"
        ]
        
        # Bias Analysis Section
        analysis_lines.append("### 🎯 Bias Detection Analysis\n")
        
        for attr_name, attr_data in bias_results.items():
            is_justified = attr_name in justified_attributes
            justified_note = " (Business Justified)" if is_justified else ""
            
            # Safely format values
            def fmt(val):
                return f"{val:.3f}" if isinstance(val, (int, float)) else str(val)
            analysis_lines.extend([
                f"#### {attr_name.title()}{justified_note}\n",
                f"**Disparate Impact:** {fmt(attr_data.get('disparate_impact', 'N/A'))}  ",
                f"**Statistical Parity Difference:** {fmt(attr_data.get('statistical_parity_difference', 'N/A'))}  ",
                f"**Equal Opportunity Difference:** {fmt(attr_data.get('equal_opportunity_difference', 'N/A'))}  ",
            ])
            
            # Interpretation
            interpretation = attr_data.get('interpretation', 'No interpretation available')
            analysis_lines.append(f"**Interpretation:** {interpretation}\n")
            
            # Add recommendations if not justified
            if not is_justified:
                di = attr_data.get('disparate_impact', 1.0)
                if isinstance(di, (int, float)) and di < 0.8:
                    analysis_lines.append("⚠️ **Action Required:** Consider preprocessing techniques to reduce disparate impact.\n")
            else:
                analysis_lines.append("ℹ️ **Note:** Disparities in this attribute are considered business-justified.\n")
        
        # Fairness Analysis Section
        analysis_lines.append("### ⚖️ Fairness Metrics Analysis\n")
        
        for attr_name, attr_results in fairness_results.items():
            is_justified = attr_name in justified_attributes
            justified_note = " (Business Justified)" if is_justified else ""
            
            analysis_lines.append(f"#### {attr_name.title()}{justified_note}\n")
            
            # Individual fairness metrics
            for metric_name, metric_result in attr_results.items():
                if hasattr(metric_result, 'metric_name'):
                    status = "✅" if metric_result.threshold_met else "❌"
                    analysis_lines.extend([
                        f"**{status} {metric_result.metric_name}**  ",
                        f"- Privileged Group: {metric_result.privileged_value:.3f}  ",
                        f"- Unprivileged Group: {metric_result.unprivileged_value:.3f}  ",
                        f"- Difference: {metric_result.difference:.3f}  ",
                        f"- Ratio: {metric_result.ratio:.3f}  ",
                        f"- Interpretation: {metric_result.interpretation}  ",
                        ""
                    ])
                elif isinstance(metric_result, dict) and metric_name == "performance_gaps":
                    analysis_lines.append("**Performance Gaps:**  ")
                    for gap_name, gap_result in metric_result.items():
                        status = "✅" if gap_result.threshold_met else "❌"
                        analysis_lines.append(f"- {status} {gap_result.metric_name}: {gap_result.difference:.3f}  ")
                    analysis_lines.append("")
        
        # Explainability Analysis (if available)
        if explainability_results:
            analysis_lines.append("### 🔍 Explainability Analysis\n")
            
            for exp_result in explainability_results:
                analysis_lines.append(f"#### {exp_result.method} Analysis\n")
                
                # Top features
                if exp_result.global_importance:
                    top_features = sorted(
                        exp_result.global_importance.items(),
                        key=lambda x: abs(x[1]),
                        reverse=True
                    )[:10]
                    
                    analysis_lines.append("**Top Important Features:**\n")
                    analysis_lines.append("| Feature | Importance |")
                    analysis_lines.append("|---------|------------|")
                    
                    for feature, importance in top_features:
                        is_protected = feature in [attr for attr in bias_results.keys()]
                        is_justified_feat = feature in justified_attributes
                        
                        status = ""
                        if is_protected and not is_justified_feat:
                            status = " ⚠️"  # Protected attribute
                        elif is_justified_feat:
                            status = " ℹ️"   # Justified attribute
                        
                        analysis_lines.append(f"| {feature}{status} | {importance:.4f} |")
                    
                    analysis_lines.append("")
                
                # Bias explanation
                if hasattr(exp_result, 'bias_explanation') and exp_result.bias_explanation:
                    bias_exp = exp_result.bias_explanation
                    
                    if 'overall_assessment' in bias_exp:
                        assessment = bias_exp['overall_assessment']
                        emoji = "🔴" if assessment == "HIGH_RISK" else "🟡" if assessment == "MEDIUM_RISK" else "🟢"
                        analysis_lines.append(f"**Bias Risk Assessment:** {emoji} {assessment}\n")
                    
                    if 'recommendations' in bias_exp:
                        analysis_lines.append("**Explainability Recommendations:**\n")
                        for rec in bias_exp['recommendations']:
                            analysis_lines.append(f"- {rec}")
                        analysis_lines.append("")
        
        return "\n".join(analysis_lines)
    
    def generate_markdown_report(
        self,
        model_name: str,
        dataset_info: Optional[Dict[str, Any]] = None,
        bias_results: Optional[Dict[str, Any]] = None,
        fairness_results: Optional[Dict[str, Any]] = None,
        explainability_results: Optional[List[Any]] = None,
        justified_attributes: Optional[List[str]] = None,
        additional_info: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate a comprehensive markdown report.
        
        Args:
            model_name: Name of the model being analyzed
            dataset_info: Information about the dataset
            bias_results: Results from bias detection
            fairness_results: Results from fairness analysis
            explainability_results: Results from explainability analysis
            justified_attributes: List of business-justified attributes
            additional_info: Any additional information to include
            
        Returns:
            Path to generated markdown report
        """
        # Calculate ethical score
        ethical_score = 75  # Default
        score_breakdown = {"grade": "B", "assessment": "Acceptable performance"}
        
        if bias_results and fairness_results:
            ethical_score, score_breakdown = self.calculate_ethical_score(
                bias_results, fairness_results, justified_attributes
            )
        
        # Create visualizations
        badge_path = self.create_ethical_badge(ethical_score, score_breakdown['grade'])
        
        viz_paths = []
        if bias_results:
            bias_chart = self.create_bias_summary_chart(bias_results, justified_attributes)
            viz_paths.append(bias_chart)
        
        if fairness_results:
            fairness_chart = self.create_fairness_metrics_chart(fairness_results)
            viz_paths.append(fairness_chart)
        
        # Start report content
        report_lines = [
            f"# 🤖 AI Fairness & Bias Audit Report: {model_name}\n",
            f"**Generated by Fairsight Toolkit**  ",
            f"**Report Date:** {datetime.now().strftime('%B %d, %Y at %I:%M %p')}  ",
            f"**Organization:** {self.company_name}  ",
            f"**Ethical Score:** {ethical_score}/100 ({score_breakdown['grade']})  ",
            "",
            f"![Ethical Badge](assets/ethical_badge.png)\n",
        ]
        
        # Executive Summary
        if bias_results and fairness_results:
            exec_summary = self.generate_executive_summary(
                model_name, ethical_score, score_breakdown,
                bias_results, fairness_results, justified_attributes
            )
            report_lines.append(exec_summary)
        
        # Dataset Information
        if dataset_info:
            report_lines.extend([
                "## 📊 Dataset Information\n",
                f"**Dataset Size:** {dataset_info.get('n_samples', 'N/A')} samples, {dataset_info.get('n_features', 'N/A')} features  ",
                f"**Target Variable:** {dataset_info.get('target_column', 'N/A')}  ",
                f"**Protected Attributes:** {', '.join(dataset_info.get('protected_attributes', []))}  ",
                f"**Justified Attributes:** {', '.join(justified_attributes) if justified_attributes else 'None'}  ",
                ""
            ])
        
        # Visualizations
        if viz_paths:
            report_lines.append("## 📈 Key Visualizations\n")
            
            if bias_results:
                report_lines.extend([
                    "### Bias Analysis Summary",
                    "![Bias Summary](visualizations/bias_summary_chart.png)\n"
                ])
            
            if fairness_results:
                report_lines.extend([
                    "### Fairness Metrics Analysis", 
                    "![Fairness Metrics](visualizations/fairness_metrics_chart.png)\n"
                ])
        
        # Detailed Analysis
        if bias_results or fairness_results:
            detailed_analysis = self.generate_detailed_analysis(
                bias_results or {}, fairness_results or {},
                explainability_results, justified_attributes
            )
            report_lines.append(detailed_analysis)
        
        # Additional Information
        if additional_info:
            report_lines.append("## 📝 Additional Information\n")
            for key, value in additional_info.items():
                report_lines.append(f"**{key.title()}:** {value}  ")
            report_lines.append("")
        
        # Footer
        report_lines.extend([
            "---\n",
            "## 🛠 Technical Details\n",
            f"**Generated with:** Fairsight Toolkit v1.0.0  ",
            f"**Bias Detection Methods:** Disparate Impact, Statistical Parity, Equal Opportunity  ",
            f"**Fairness Standards:** 80% Rule (Disparate Impact), 10% Threshold (Differences)  ",
            "",
            "*This report was automatically generated by the Fairsight Toolkit. ",
            "For questions or support, please contact your AI Ethics team.*",
        ])
        
        # Save report
        report_content = "\n".join(report_lines)
        report_path = os.path.join(self.output_dir, f"{model_name}_fairness_report.md")
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        logger.info(f"Markdown report generated: {report_path}")
        return report_path
    
    def generate_html_report(
        self,
        model_name: str,
        markdown_report_path: str
    ) -> str:
        """Convert markdown report to HTML with styling."""
        try:
            import markdown
            from markdown.extensions import tables, toc
            
            # Read markdown content
            with open(markdown_report_path, 'r', encoding='utf-8') as f:
                md_content = f.read()
            
            # Convert to HTML
            md = markdown.Markdown(extensions=['tables', 'toc', 'fenced_code'])
            html_body = md.convert(md_content)
            
            # CSS Styling
            css_style = f"""
            <style>
                body {{
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                    line-height: 1.6;
                    color: #333;
                    max-width: 1200px;
                    margin: 0 auto;
                    padding: 20px;
                    background-color: #f8f9fa;
                }}
                .container {{
                    background-color: white;
                    padding: 30px;
                    border-radius: 10px;
                    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                }}
                h1 {{
                    color: {self.colors['primary']};
                    border-bottom: 3px solid {self.colors['primary']};
                    padding-bottom: 10px;
                }}
                h2 {{
                    color: {self.colors['secondary']};
                    border-bottom: 2px solid #e9ecef;
                    padding-bottom: 5px;
                }}
                h3 {{
                    color: {self.colors['dark']};
                }}
                .executive-summary {{
                    background-color: #e8f4f0;
                    padding: 20px;
                    border-radius: 8px;
                    border-left: 5px solid {self.colors['primary']};
                    margin: 20px 0;
                }}
                .critical-issue {{
                    background-color: #f8d7da;
                    padding: 15px;
                    border-radius: 5px;
                    border-left: 4px solid {self.colors['danger']};
                    margin: 10px 0;
                }}
                .warning-issue {{
                    background-color: #fff3cd;
                    padding: 15px;
                    border-radius: 5px;
                    border-left: 4px solid {self.colors['warning']};
                    margin: 10px 0;
                }}
                .success-issue {{
                    background-color: #d4edda;
                    padding: 15px;
                    border-radius: 5px;
                    border-left: 4px solid {self.colors['success']};
                    margin: 10px 0;
                }}
                table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin: 20px 0;
                }}
                th, td {{
                    border: 1px solid #dee2e6;
                    padding: 12px;
                    text-align: left;
                }}
                th {{
                    background-color: {self.colors['light']};
                    font-weight: bold;
                    color: {self.colors['dark']};
                }}
                tr:nth-child(even) {{
                    background-color: #f8f9fa;
                }}
                .metric-pass {{
                    color: {self.colors['success']};
                    font-weight: bold;
                }}
                .metric-fail {{
                    color: {self.colors['danger']};
                    font-weight: bold;
                }}
                .badge {{
                    display: inline-block;
                    padding: 4px 8px;
                    border-radius: 4px;
                    font-size: 0.875em;
                    font-weight: bold;
                }}
                .badge-success {{
                    background-color: {self.colors['success']};
                    color: white;
                }}
                .badge-warning {{
                    background-color: {self.colors['warning']};
                    color: white;
                }}
                .badge-danger {{
                    background-color: {self.colors['danger']};
                    color: white;
                }}
                img {{
                    max-width: 100%;
                    height: auto;
                    border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    margin: 20px 0;
                }}
                .footer {{
                    margin-top: 40px;
                    padding-top: 20px;
                    border-top: 1px solid #dee2e6;
                    color: #6c757d;
                    font-size: 0.9em;
                }}
            </style>
            """
            
            # Full HTML document
            html_content = f"""
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Fairness Report - {model_name}</title>
                {css_style}
            </head>
            <body>
                <div class="container">
                    {html_body}
                </div>
            </body>
            </html>
            """
            
            # Save HTML report
            html_path = os.path.join(self.output_dir, f"{model_name}_fairness_report.html")
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            logger.info(f"HTML report generated: {html_path}")
            return html_path
            
        except ImportError:
            logger.warning("Markdown library not available. HTML generation skipped.")
            return ""
        except Exception as e:
            logger.error(f"Error generating HTML report: {e}")
            return ""
    
    def generate_pdf_report(
        self,
        markdown_report_path: str,
        model_name: str
    ) -> str:
        """Generate PDF report from markdown using FPDF and bundled Unicode font."""
        def safe_unicode(text):
            # Remove characters above U+FFFF (most emoji, rare symbols)
            return ''.join(c for c in text if ord(c) <= 0xFFFF)
        try:
            pdf_path = os.path.join(self.output_dir, f"{model_name}_fairness_report.pdf")
            with open(markdown_report_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            pdf = FPDF()
            pdf.set_auto_page_break(auto=True, margin=15)
            pdf.add_page()
            # Load bundled DejaVuSans.ttf font
            font_path = pkg_resources.resource_filename('fairsight', 'fonts/DejaVuSans.ttf')
            pdf.add_font('DejaVu', '', font_path, uni=True)
            pdf.set_font('DejaVu', '', 12)

            for line in lines:
                text = safe_unicode(line.strip())
                if line.startswith('# '):
                    pdf.set_font('DejaVu', '', 18)
                    pdf.cell(0, 10, safe_unicode(text[2:]), ln=True)
                    pdf.set_font('DejaVu', '', 12)
                elif line.startswith('## '):
                    pdf.set_font('DejaVu', '', 14)
                    pdf.cell(0, 8, safe_unicode(text[3:]), ln=True)
                    pdf.set_font('DejaVu', '', 12)
                elif line.startswith('### '):
                    pdf.set_font('DejaVu', '', 12)
                    pdf.cell(0, 7, safe_unicode(text[4:]), ln=True)
                    pdf.set_font('DejaVu', '', 12)
                elif text.startswith('!['):
                    import re
                    m = re.match(r'!\[.*\]\((.*)\)', text)
                    if m:
                        img_path = m.group(1)
                        img_full_path = os.path.join(self.output_dir, img_path) if not os.path.isabs(img_path) else img_path
                        if os.path.exists(img_full_path):
                            pdf.ln(2)
                            pdf.image(img_full_path, w=100)
                            pdf.ln(2)
                elif text == '---':
                    pdf.ln(5)
                    pdf.cell(0, 0, '', ln=True)
                    pdf.ln(5)
                else:
                    pdf.multi_cell(0, 7, text)
            pdf.output(pdf_path)
            return pdf_path
        except Exception as e:
            logger.error(f"Error generating PDF report: {e}")
            return ""
    
    def generate_json_summary(
        self,
        model_name: str,
        ethical_score: int,
        score_breakdown: Dict[str, Any],
        bias_results: Optional[Dict[str, Any]] = None,
        fairness_results: Optional[Dict[str, Any]] = None,
        justified_attributes: Optional[List[str]] = None
    ) -> str:
        """Generate a JSON summary for API consumption or SAP HANA integration."""
        
        # Convert complex objects to dictionaries
        def serialize_results(results):
            if isinstance(results, dict):
                serialized = {}
                for key, value in results.items():
                    if hasattr(value, 'to_dict'):
                        serialized[key] = value.to_dict()
                    elif isinstance(value, dict):
                        serialized[key] = serialize_results(value)
                    else:
                        serialized[key] = value
                return serialized
            return results
        
        summary_data = {
            "report_metadata": {
                "model_name": model_name,
                "generated_at": datetime.now().isoformat(),
                "toolkit_version": "1.0.0",
                "organization": self.company_name
            },
            "ethical_assessment": {
                "overall_score": ethical_score,
                "grade": score_breakdown.get('grade', 'N/A'),
                "assessment": score_breakdown.get('assessment', 'N/A'),
                "score_breakdown": score_breakdown
            },
            "configuration": {
                "justified_attributes": justified_attributes or [],
                "fairness_threshold": 0.8
            },
            "bias_analysis": serialize_results(bias_results) if bias_results else {},
            "fairness_analysis": serialize_results(fairness_results) if fairness_results else {},
            "summary_statistics": {
                "total_attributes_analyzed": len(bias_results) if bias_results else 0,
                "critical_issues": 0,
                "concerning_issues": 0,
                "passing_attributes": 0
            }
        }
        
        # Calculate summary statistics
        if bias_results:
            for attr_name, attr_data in bias_results.items():
                if attr_name not in (justified_attributes or []):
                    di = attr_data.get('disparate_impact', 1.0)
                    spd = abs(attr_data.get('statistical_parity_difference', 0))
                    
                    if di < 0.6 or spd > 0.2:
                        summary_data['summary_statistics']['critical_issues'] += 1
                    elif di < 0.8 or spd > 0.1:
                        summary_data['summary_statistics']['concerning_issues'] += 1
                    else:
                        summary_data['summary_statistics']['passing_attributes'] += 1
                else:
                    summary_data['summary_statistics']['passing_attributes'] += 1
        
        # Save JSON summary
        json_path = os.path.join(self.output_dir, f"{model_name}_fairness_summary.json")
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"JSON summary generated: {json_path}")
        return json_path
    
    def generate_complete_report(
        self,
        model_name: str,
        dataset_info: Optional[Dict[str, Any]] = None,
        bias_results: Optional[Dict[str, Any]] = None,
        fairness_results: Optional[Dict[str, Any]] = None,
        explainability_results: Optional[List[Any]] = None,
        justified_attributes: Optional[List[str]] = None,
        additional_info: Optional[Dict[str, Any]] = None,
        formats: List[str] = ["markdown", "html", "json"],
        send_to_registry: bool = False,
        registry_client: Optional[Any] = None,
        user_id: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Generate complete report in multiple formats.
        
        Args:
            model_name: Name of the model
            dataset_info: Dataset information
            bias_results: Bias analysis results
            fairness_results: Fairness analysis results
            explainability_results: Explainability analysis results
            justified_attributes: Business-justified attributes
            additional_info: Additional information to include
            formats: List of formats to generate ["markdown", "html", "pdf", "json"]
            
        Returns:
            Dictionary of format -> file_path mappings
        """
        logger.info(f"Generating complete report for {model_name}...")
        
        generated_files = {}
        
        # Calculate ethical score for JSON summary
        ethical_score = 75
        score_breakdown = {"grade": "B", "assessment": "Acceptable performance"}
        
        if bias_results and fairness_results:
            ethical_score, score_breakdown = self.calculate_ethical_score(
                bias_results, fairness_results, justified_attributes
            )
        
        try:
            # Generate Markdown report (base for other formats)
            if "markdown" in formats:
                md_path = self.generate_markdown_report(
                    model_name=model_name,
                    dataset_info=dataset_info,
                    bias_results=bias_results,
                    fairness_results=fairness_results,
                    explainability_results=explainability_results,
                    justified_attributes=justified_attributes,
                    additional_info=additional_info
                )
                generated_files["markdown"] = md_path
            else:
                # Generate markdown for HTML/PDF conversion even if not requested
                md_path = self.generate_markdown_report(
                    model_name=model_name,
                    dataset_info=dataset_info,
                    bias_results=bias_results,
                    fairness_results=fairness_results,
                    explainability_results=explainability_results,
                    justified_attributes=justified_attributes,
                    additional_info=additional_info
                )
            
            # Generate HTML report
            if "html" in formats:
                html_path = self.generate_html_report(model_name, md_path)
                if html_path:
                    generated_files["html"] = html_path
            
            # Generate PDF report
            if "pdf" in formats:
                if "html" not in generated_files:
                    html_path = self.generate_html_report(model_name, md_path)
                else:
                    html_path = generated_files["html"]
                
                if html_path:
                    pdf_path = self.generate_pdf_report(html_path, model_name)
                    if pdf_path:
                        generated_files["pdf"] = pdf_path
            
            # Generate JSON summary
            if "json" in formats:
                json_path = self.generate_json_summary(
                    model_name=model_name,
                    ethical_score=ethical_score,
                    score_breakdown=score_breakdown,
                    bias_results=bias_results,
                    fairness_results=fairness_results,
                    justified_attributes=justified_attributes
                )
                generated_files["json"] = json_path
            
            logger.info(f"Report generation completed. Generated formats: {list(generated_files.keys())}")
            
            # Submit to registry if requested
            if send_to_registry and registry_client and user_id:
                try:
                    logger.info("Submitting report to registry...")
                    
                    # Prepare data for registry submission
                    bias_results_list = []
                    if bias_results:
                        # Convert bias results to list format for registry
                        for attr, results in bias_results.items():
                            if isinstance(results, dict):
                                bias_results_list.append({
                                    'attribute': attr,
                                    'metric_name': results.get('metric_name', 'Unknown'),
                                    'value': results.get('value', 0),
                                    'biased': results.get('biased', False),
                                    'justified': attr in (justified_attributes or []),
                                    'details': results
                                })
                    
                    # Submit to registry
                    registry_response = registry_client.submit_audit_results(
                        bias_results=bias_results_list,
                        fairness_results=fairness_results or {},
                        detailed_reasoning=registry_client.generate_detailed_reasoning(
                            bias_results_list, 
                            fairness_results or {}, 
                            dataset_info
                        ),
                        model_name=model_name,
                        user_id=user_id,
                        ethical_score=ethical_score,
                        bias_score=score_breakdown.get('bias_score', 0),
                        fairness_score=score_breakdown.get('fairness_score', 0),
                        grade=score_breakdown.get('grade', 'UNKNOWN')
                    )
                    
                    logger.info(f"✅ Successfully submitted to registry: {registry_response.get('message', 'OK')}")
                    generated_files["registry"] = registry_response
                    
                except Exception as e:
                    logger.error(f"❌ Failed to submit to registry: {e}")
                    generated_files["registry_error"] = str(e)
            
        except Exception as e:
            logger.error(f"Error during report generation: {e}")
            generated_files["error"] = str(e)
        
        return generated_files


# Convenience class for backward compatibility
class Report:
    """Backward compatibility wrapper for ReportGenerator."""
    
    def __init__(self, audit_results: Dict[str, Any], **kwargs):
        self.generator = ReportGenerator(**kwargs)
        self.audit_results = audit_results
    
    def generate(self, model_name: str = "Model", formats: List[str] = ["markdown"]) -> Dict[str, str]:
        """Generate report from audit results."""
        return self.generator.generate_complete_report(
            model_name=model_name,
            bias_results=self.audit_results.get('bias', {}),
            fairness_results=self.audit_results.get('fairness', {}),
            explainability_results=self.audit_results.get('explainability', []),
            formats=formats
        )


# Convenience functions
def generate_quick_report(
    model_name: str,
    bias_results: Dict[str, Any],
    fairness_results: Dict[str, Any],
    output_dir: str = "/tmp/fairsight_reports"
) -> str:
    """Quick report generation function."""
    generator = ReportGenerator(output_dir=output_dir)
    
    files = generator.generate_complete_report(
        model_name=model_name,
        bias_results=bias_results,
        fairness_results=fairness_results,
        formats=["markdown"]
    )
    
    return files.get("markdown", "")

def calculate_model_ethical_score(
    bias_results: Dict[str, Any],
    fairness_results: Dict[str, Any],
    justified_attributes: Optional[List[str]] = None
) -> Tuple[int, str]:
    """Quick ethical score calculation."""
    generator = ReportGenerator()
    score, breakdown = generator.calculate_ethical_score(
        bias_results, fairness_results, justified_attributes
    )
    return score, breakdown['assessment']

def generate_html_report(bias_results, fairness_results, model_name='Model', output_dir='/tmp/fairsight_reports'):
    """
    Standalone HTML report generation function.
    Args:
        bias_results: Bias analysis results
        fairness_results: Fairness analysis results
        model_name: Name of the model
        output_dir: Directory to save the report
    Returns:
        Path to generated HTML report
    """
    generator = ReportGenerator(output_dir=output_dir)
    files = generator.generate_complete_report(
        model_name=model_name,
        bias_results=bias_results,
        fairness_results=fairness_results,
        formats=["html"]
    )
    return files.get("html", "")
