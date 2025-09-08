import pandas as pd
from fairsight import IllegalDataDetector, FSAuditor

# HANA Cloud Connection Parameters Template
# Replace these with your actual HANA Cloud credentials
connection_params = {
    "host": "d4749caf-d293-4be5-8cde-fdd920efefac.hana.trial-us10.hanacloud.ondemand.com",  # Replace with your HANA Cloud instance URL
    "port": 443,
    "user": "DBADMIN",
    "password": "FairSight000!",  # Replace with your HANA Cloud password
    "encrypt": True
}

df = pd.read_csv('tests/Medical_insurance.csv')
auditor = FSAuditor(
            dataset=df, 
            sensitive_features=['sex', 'children'], 
            justified_attributes='smoker',
            target='charges')

audit_result = auditor.run_audit(
    push_to_dashboard=True,
    push_to_registry=False,
    connection_params=connection_params
)