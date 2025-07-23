# 🧠 Fairsight Toolkit

> **Comprehensive AI Ethics and Bias Detection Toolkit with SAP Integration**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![SAP HANA](https://img.shields.io/badge/SAP%20HANA-Cloud-blue)](https://www.sap.com/products/hana.html)

Fairsight is a production-ready Python toolkit for detecting bias, ensuring fairness, and maintaining ethical standards in machine learning models and datasets. Built with enterprise integration in mind, it features seamless SAP HANA Cloud and SAP Analytics Cloud connectivity.

## 🌟 Key Features

- **🔍 Comprehensive Bias Detection**: Statistical parity, disparate impact, equal opportunity, and more
- **⚖️ Fairness Metrics**: Demographic parity, equalized odds, predictive parity
- **🔮 Model Explainability**: SHAP and LIME integration for interpretable AI
- **📊 Enterprise Integration**: Native SAP HANA Cloud and SAP Analytics Cloud support
- **📋 Justified Attributes**: Smart handling of business-justified discriminatory features
- **🚀 Easy to Use**: Simple API for both datasets and trained models
- **📈 Automated Reporting**: Beautiful, actionable audit reports
- **🏢 Production Ready**: Enterprise-grade logging, error handling, and scalability

## 🛠️ Installation

### Basic Installation
```bash
pip install fairsight-toolkit
```

### With SAP Integration
```bash
pip install fairsight-toolkit[sap]
```

### Development Installation
```bash
git clone https://github.com/vijayk/fairsight-toolkit.git
cd fairsight-toolkit
pip install -e .[dev,sap]
```

## 🚀 Quick Start

### Basic Dataset Audit
```python
from fairsight import FSAuditor

# Simple dataset audit
auditor = FSAuditor(
    dataset="data/hiring_data.csv",
    sensitive_features=["gender", "race"],
    target="hired",
    justified_attributes=["experience_years"]  # Job-relevant factors
)

results = auditor.run_audit()
print(f"Ethical Score: {results['ethical_score']}/100")
```

### Model + Dataset Audit
```python
from fairsight import FSAuditor
from sklearn.ensemble import RandomForestClassifier

# Train your model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Comprehensive audit
auditor = FSAuditor(
    dataset="data/loan_data.csv",
    model=model,
    sensitive_features=["gender", "race", "age"],
    target="loan_approved",
    justified_attributes=["credit_score", "income"],  # Financially relevant
    fairness_threshold=0.8
)

# Run complete audit
audit_results = auditor.run_audit()

# Export results
auditor.export_results("audit_report.json")
```

### Handling "Justified" Attributes

The key innovation of Fairsight is handling **justified attributes** - features that may appear discriminatory but are business-justified:

```python
# Example: House loan approval
auditor = FSAuditor(
    dataset="house_loans.csv",
    sensitive_features=["gender", "race", "job"],  
    justified_attributes=["job"],  # Job status is legally justified for loans
    target="approved"
)

results = auditor.run_audit()

# Job-related disparities won't be flagged as bias
# Gender/race disparities will still be detected
```

## 🏗️ Architecture

```
fairsight/
├── __init__.py              # Main package exports  
├── auditor.py              # FSAuditor main class
├── bias_detection.py       # Enhanced bias detection with justified attributes
├── dataset_audit.py        # Comprehensive dataset auditing
├── model_audit.py          # Model performance and bias auditing  
├── explainability.py       # SHAP/LIME model explanations
├── fairness_metrics.py     # Fairness metric computations
├── report_generator.py     # Automated report generation
├── dashboard_push.py       # SAP HANA Cloud integration
└── utils.py               # Utility functions
```

## 📊 SAP Integration

### SAP HANA Cloud Setup
```python
from fairsight import Dashboard

# Configure SAP HANA connection
dashboard = Dashboard({
    "host": "your-hana-instance.hanacloud.ondemand.com",
    "port": 443,
    "user": "DBADMIN", 
    "password": "your_password",
    "encrypt": True
})

# Audit results automatically pushed to HANA
auditor = FSAuditor(
    dataset="data.csv",
    sensitive_features=["gender"],
    enable_sap_integration=True
)

results = auditor.run_audit()  # Automatically pushes to HANA
```

### SAP Analytics Cloud Dashboard
```python
# Generate SAP Analytics Cloud configuration
dashboard_config = dashboard.create_sac_dashboard_config()

# Use this configuration to set up your SAC dashboard
print(dashboard_config)
```

## 🔍 Comprehensive Example

```python
import pandas as pd
from fairsight import FSAuditor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load data
df = pd.read_csv("hiring_dataset.csv")

# Define protected and justified attributes
protected_attrs = ["gender", "race", "age"]
justified_attrs = ["years_experience", "education_level"]  # Job-relevant

# Split data
X = df.drop("hired", axis=1)
y = df["hired"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Comprehensive audit
auditor = FSAuditor(
    model=model,
    X_test=X_test,
    y_test=y_test,
    sensitive_features=protected_attrs,
    justified_attributes=justified_attrs,
    fairness_threshold=0.8,
    enable_sap_integration=True
)

# Run audit with all components
results = auditor.run_audit(
    include_dataset=True,
    include_model=True,
    include_bias_detection=True,
    generate_report=True,
    push_to_dashboard=True
)

# Print summary
print("=" * 50)
print(f"🏆 ETHICAL SCORE: {results['ethical_score']}/100")
print(f"📊 OVERALL ASSESSMENT: {results['executive_summary']['overall_assessment']}")
print("=" * 50)

# Key findings
for finding in results['executive_summary']['key_findings']:
    print(f"✅ {finding}")

# Critical issues  
for issue in results['executive_summary']['critical_issues']:
    print(f"🚨 {issue}")

# Recommendations
for rec in results['executive_summary']['recommendations']:
    print(f"💡 {rec}")

# Export detailed results
auditor.export_results("detailed_audit_results.json")

# View audit history
history = auditor.get_audit_history(limit=5)
print(history)
```

## 📋 Key Metrics

### Bias Detection Metrics
- **Disparate Impact**: 80% rule compliance
- **Statistical Parity Difference**: Difference in positive rates
- **Equal Opportunity Difference**: Difference in TPR across groups  
- **Predictive Parity**: Difference in precision across groups
- **Equalized Odds**: Both TPR and FPR differences

### Fairness Metrics  
- **Demographic Parity**: Equal positive prediction rates
- **Equal Opportunity**: Equal TPR for qualified individuals
- **Predictive Equality**: Equal FPR across groups
- **Overall Accuracy Equality**: Equal accuracy across groups

## 🎯 Use Cases

### 1. **Hiring & Recruitment**
```python
# Audit hiring algorithms
auditor = FSAuditor(
    dataset="hiring_data.csv",
    sensitive_features=["gender", "race", "age"],
    justified_attributes=["experience", "education", "skills_score"],
    target="hired"
)
```

### 2. **Financial Services**
```python  
# Audit loan approval models
auditor = FSAuditor(
    model=loan_model,
    sensitive_features=["gender", "race", "marital_status"], 
    justified_attributes=["credit_score", "income", "debt_ratio"],
    target="loan_approved"
)
```

### 3. **Healthcare**
```python
# Audit medical diagnosis systems
auditor = FSAuditor(
    model=diagnosis_model,
    sensitive_features=["gender", "race", "age"],
    justified_attributes=["symptoms", "medical_history", "test_results"],
    target="diagnosis"
)
```

## 📊 Example Output

```
🧠 AI Fairness & Bias Audit Report
===================================

**Ethical Score**: 87/100

🔍 Attribute-wise Bias Analysis
--------------------------------

➤ Gender
- Disparate Impact: 0.85
- Equal Opportunity Difference: 0.08  
- Statistical Parity Difference: 0.12
- **Interpretation**: Minor disparity detected, within acceptable range.

➤ Job (justified attribute)  
- Disparate Impact: 0.62
- Equal Opportunity Difference: 0.28
- **Interpretation**: This feature is justified for decision-making per business requirements.

📊 Fairness Metric Gaps
------------------------

| Attribute | Precision Gap | Recall Gap | F1 Score Gap |
|-----------|---------------|-----------|-------------|
| Gender    | 0.05          | 0.07      | 0.06        |
| Job       | 0.15          | 0.18      | 0.16        |

📌 Final Ethical Assessment
----------------------------

✅ The model demonstrates strong ethical integrity with low bias across protected groups.

📋 Note: job is marked as a justified attribute and disparities here are acceptable per business configuration.
```

## 🔧 Advanced Configuration

### Custom Fairness Thresholds
```python
auditor = FSAuditor(
    dataset="data.csv",
    fairness_threshold=0.85,  # Stricter 85% rule
    sensitive_features=["gender", "race"]
)
```

### Custom Privileged Groups
```python
auditor = FSAuditor(
    dataset="data.csv",
    sensitive_features=["gender", "race"],
    privileged_groups={
        "gender": "male",      # Specify privileged group
        "race": "white"
    }
)
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`) 
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **SAP HANA Cloud** for enterprise data integration
- **SHAP & LIME** for model explainability  
- **scikit-learn** for machine learning utilities
- **pandas & numpy** for data processing

## 📞 Support

- 📧 Email: support@fairsight-toolkit.com
- 💬 GitHub Issues: [Create an issue](https://github.com/vijayk/fairsight-toolkit/issues)
- 📖 Documentation: [fairsight-toolkit.readthedocs.io](https://fairsight-toolkit.readthedocs.io/)

---

**Made with ❤️ for Ethical AI**

*Fairsight Toolkit - Making AI Fair, Transparent, and Accountable*
