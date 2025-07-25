import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from fairsight import FSAuditor
from fairsight.report_generator import Report
# Use the actual CSV and columns present in the test data

# Define SAP HANA Cloud connection parameters (replace with real values for actual test)
connection_params = {
    "host": "d4749caf-d293-4be5-8cde-fdd920efefac.hana.trial-us10.hanacloud.ondemand.com",
    "port": 443,
    "user": "DBADMIN",
    "password": "FairSight000!",
    "encrypt": True
}

auditor = FSAuditor(
    dataset="tests/synthetic_complaints.csv",
    sensitive_features=["category"],
    target="intent",
    justified_attributes=[]
)
results = auditor.run_audit(connection_params=connection_params)
print(results["ethical_score"])
#auditor.export_results("audit_report.json")
reporter = Report(auditor.audit_results)
files = reporter.generate(formats=["markdown", "html", "pdf", "json"])
print(files)

if __name__ == "__main__":
    import pandas as pd
    from fairsight.reweighing import Reweighing
    from fairsight.bias_detection import BiasDetector
    from fairsight.fairness_metrics import generalized_entropy_index, FairnessMetrics

    # Create a small synthetic dataset
    data = {
        'gender': ['male', 'female', 'female', 'male', 'female', 'male', 'male', 'female'],
        'outcome': [1, 0, 1, 1, 0, 0, 1, 0]
    }
    df = pd.DataFrame(data)
    print("Synthetic dataset:")
    print(df)

    # --- Audit BEFORE mitigation ---
    print("\n=== Bias Metrics BEFORE Mitigation ===")
    detector = BiasDetector(dataset=df, sensitive_features=['gender'], target='outcome', privileged_values={'gender': 'male'})
    bias_results = detector.detect_bias_on_dataset()
    for res in bias_results:
        print(res)
    fm = FairnessMetrics(y_true=df['outcome'], y_pred=df['outcome'], protected_attr=df['gender'], privileged_group='male')
    print("Demographic Parity:", fm.demographic_parity())
    print("Generalized Entropy Index (outcome, alpha=2):", generalized_entropy_index(df['outcome'], alpha=2))

    # --- Apply Reweighing Mitigation ---
    print("\nApplying Reweighing mitigation...")
    rw = Reweighing('gender', 'outcome', privileged_value='male', unprivileged_value='female')
    weights = rw.compute_weights(df)
    print("Instance weights:")
    print(weights)

    # --- Audit AFTER mitigation ---
    print("\n=== Bias Metrics AFTER Mitigation (weights available for model training) ===")
    print("(To see the effect, retrain your model using these weights and re-audit)")