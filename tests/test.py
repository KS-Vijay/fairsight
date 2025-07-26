import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fairsight import ExplainabilityEngine
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

# Load the adult income dataset
df = pd.read_csv('tests/adult_income.csv')

# Separate features and target
X = df.drop('income', axis=1)
y = df['income']

# Handle categorical variables using one-hot encoding
categorical_columns = ['workclass', 'education', 'marital-status', 'occupation', 
                      'relationship', 'race', 'gender', 'native-country']
X = pd.get_dummies(X, columns=categorical_columns)

# Train a classification model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Create explainability engine for classification
engine = ExplainabilityEngine(
    model=model, 
    training_data=X, 
    feature_names=list(X.columns),
    mode="classification"
)

# Generate SHAP explanations
shap_result = engine.explain_with_shap(X.sample(n=50, random_state=42))
print("SHAP Analysis Results:")
print(f"Method: {shap_result.method}")
print(f"Top 5 most important features:")
for feature, importance in shap_result.get_top_features(5):
    print(f"  {feature}: {importance:.4f}")

# Generate LIME explanations for a few samples
lime_result = engine.explain_with_lime(X.sample(n=5, random_state=42))
print(f"\nLIME Analysis Results:")
print(f"Method: {lime_result.method}")
print(f"Number of local explanations: {len(lime_result.local_explanations)}")

# Print model performance summary
from sklearn.metrics import accuracy_score, classification_report
y_pred = model.predict(X.sample(n=100, random_state=42))
y_true = y.sample(n=100, random_state=42)
print(f"\nModel Performance (sample):")
print(f"Accuracy: {accuracy_score(y_true, y_pred):.3f}")
print(f"Classification Report:")
print(classification_report(y_true, y_pred))