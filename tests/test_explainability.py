import pandas as pd
from sklearn.linear_model import LogisticRegression
from fairsight import ExplainabilityEngine

def demo_explainability():
    df = pd.DataFrame({
        'feature1': [1, 2, 3, 4],
        'feature2': [10, 20, 10, 30],
        'target': [1, 0, 1, 0]
    })
    X = df[['feature1', 'feature2']]
    y = df['target']
    model = LogisticRegression().fit(X, y)
    engine = ExplainabilityEngine(model=model, training_data=X, feature_names=['feature1', 'feature2'])
    try:
        shap_result = engine.explain_with_shap(X)
        print('SHAP explanation:', shap_result)
    except ImportError:
        print('SHAP not installed')
    try:
        lime_result = engine.explain_with_lime(X)
        print('LIME explanation:', lime_result)
    except ImportError:
        print('LIME not installed')

if __name__ == '__main__':
    print('--- Demo: ExplainabilityEngine ---')
    demo_explainability() 