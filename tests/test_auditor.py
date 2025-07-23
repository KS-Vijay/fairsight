import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
from fairsight import FSAuditor, Auditor

def demo_fsauditor():
    df = pd.DataFrame({
        'category': ['A', 'B', 'A', 'B'],
        'intent': [1, 0, 1, 0],
        'feature': [10, 20, 10, 30]
    })
    auditor = FSAuditor(dataset=df, sensitive_features=['category'], target='intent')
    results = auditor.run_audit(generate_report=False, push_to_dashboard=False)
    print('FSAuditor ethical_score:', results.get('ethical_score'))
    auditor.export_results('auditor_results.json')
    print('Results exported to auditor_results.json')

def demo_auditor_legacy():
    df = pd.DataFrame({
        'category': ['A', 'B', 'A', 'B'],
        'intent': [1, 0, 1, 0],
        'feature': [10, 20, 10, 30]
    })
    auditor = Auditor(dataset=df, sensitive_features=['category'], target='intent')
    results = auditor.run_audit(generate_report=False, push_to_dashboard=False)
    print('Auditor (legacy) ethical_score:', results.get('ethical_score'))

if __name__ == '__main__':
    print('--- Demo: FSAuditor ---')
    demo_fsauditor()
    print('\n--- Demo: Auditor (legacy) ---')
    demo_auditor_legacy() 