import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from fairsight import FSAuditor

# Use the actual CSV and columns present in the test data
auditor = FSAuditor(
    dataset="tests/synthetic_complaints.csv",
    sensitive_features=["category"],
    target="intent",
    justified_attributes=[]
)
results = auditor.run_audit()
print(results["ethical_score"])
auditor.export_results("audit_report.json")