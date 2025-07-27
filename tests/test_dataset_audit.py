import pandas as pd
from fairsight import DatasetAuditor

def demo_dataset_auditor():
    df = pd.DataFrame({
        'category': ['A', 'B', 'A', 'B'],
        'intent': [1, 0, 1, 0],
        'feature': [10, 20, 10, 30]
    })
    auditor = DatasetAuditor(dataset=df, protected_attributes=['category'], target_column='intent')
    print('Audit:', auditor.audit())
    print('Preprocess:', auditor.preprocess())
    print('Data Quality:', auditor.analyze_data_quality())
    print('Bias:', auditor.detect_bias())
    print('Fairness Metrics:', auditor.compute_fairness_metrics())
    print('Statistical Report:', auditor.generate_statistical_report())

if __name__ == '__main__':
    print('--- Demo: DatasetAuditor ---')
    demo_dataset_auditor() 