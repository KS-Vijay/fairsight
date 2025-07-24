import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from fairsight import FSAuditor

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
auditor.export_results("audit_report.json")