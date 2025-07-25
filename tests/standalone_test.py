if __name__ == "__main__":
    import pandas as pd
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from fairsight import (
        explain_with_shap, explain_with_lime, detect_illegal_data,
        preprocess_data, calculate_privilege_groups, generate_html_report
    )

    print("\n=== Standalone Function Demo: Fairsight Toolkit ===\n")

    # Synthetic dataset
    data = {
        'gender': ['male', 'female', 'female', 'male', 'female', 'male', 'male', 'female'],
        'age': [25, 30, 22, 40, 28, 35, 45, 23],
        'outcome': [1, 0, 1, 1, 0, 0, 1, 0]
    }
    df = pd.DataFrame(data)
    print("Synthetic dataset:")
    print(df)

    # --- Preprocessing ---
    print("\n--- Preprocessing Data ---")
    df_processed, encoders = preprocess_data(df, target_column='outcome', protected_attributes=['gender'])
    print("Processed DataFrame:")
    print(df_processed)
    print("Label Encoders:", encoders)

    # --- Privilege Group Calculation ---
    print("\n--- Calculating Privilege Groups ---")
    priv_groups = calculate_privilege_groups(df, ['gender'])
    print("Privilege Groups:", priv_groups)

    # --- Illegal Data Detection ---
    print("\n--- Illegal Data Detection ---")
    # (Assume no illegal data in this synthetic example)
    illegal_results = detect_illegal_data(df)
    print("Illegal Data Results:", illegal_results)

    # --- Explainability (SHAP & LIME) ---
    print("\n--- Explainability (SHAP & LIME) ---")
    # Train a simple model
    X = df_processed[['gender', 'age']]
    y = df_processed['outcome']
    model = LogisticRegression().fit(X, y)
    feature_names = ['gender', 'age']

    # SHAP
    print("\nSHAP Explanation:")
    shap_result = explain_with_shap(model, X, feature_names)
    print(shap_result)

    # LIME
    print("\nLIME Explanation:")
    lime_result = explain_with_lime(model, X, feature_names)
    print(lime_result)

    # --- Reporting ---
    print("\n--- HTML Report Generation ---")
    # Dummy bias/fairness results for demo
    bias_results = {'gender': {'statistical_parity': 0.1, 'disparate_impact': 0.85}}
    fairness_results = {'gender': {'demographic_parity': 0.12, 'equal_opportunity': 0.09}}
    report_path = generate_html_report(bias_results, fairness_results, model_name='DemoModel')
    print(f"HTML report generated at: {report_path}") 