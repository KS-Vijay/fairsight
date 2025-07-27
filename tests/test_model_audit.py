import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
from sklearn.linear_model import LogisticRegression
from fairsight import ModelAuditor

def demo_model_auditor():
    df = pd.DataFrame({
        'category': [0, 1, 0, 1],
        'feature': [10, 20, 10, 30],
        'intent': [1, 0, 1, 0]
    })
    X = df[['category', 'feature']]
    y = df['intent']
    model = LogisticRegression().fit(X, y)
    auditor = ModelAuditor(model=model, X_test=X, y_test=y, protected_attributes=['category'], target_column='intent')
    print('Audit:', auditor.audit())
    print('Performance:', auditor.evaluate_performance())
    print('Bias:', auditor.detect_bias())
    print('Fairness Metrics:', auditor.compute_fairness_metrics())
    print('Explanations:', auditor.explain_model(sample_size=2, methods=['SHAP']))
    print('Feature Importance:', auditor.analyze_feature_importance())

if __name__ == '__main__':
    print('--- Demo: ModelAuditor ---')
    demo_model_auditor() 