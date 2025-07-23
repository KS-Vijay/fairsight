import numpy as np
from fairsight import FairnessMetrics, FairnessEngine, compute_demographic_parity, compute_equal_opportunity, compute_predictive_parity

def demo_fairness_metrics():
    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([1, 0, 0, 0])
    protected = np.array([0, 1, 0, 1])
    fm = FairnessMetrics(y_true, y_pred, protected_attr=protected, privileged_group=0)
    print('Demographic Parity:', fm.demographic_parity())
    eo, fpr = fm.equalized_odds()
    print('Equal Opportunity:', eo)
    print('Predictive Parity:', fm.predictive_parity())
    fe = FairnessEngine()
    results = fe.analyze_fairness(y_true, y_pred, {'category': protected})
    print('FairnessEngine results:', results)
    print('Quick metrics:')
    print('compute_demographic_parity:', compute_demographic_parity(y_true, y_pred, protected, privileged_group=0))
    print('compute_equal_opportunity:', compute_equal_opportunity(y_true, y_pred, protected, privileged_group=0))
    print('compute_predictive_parity:', compute_predictive_parity(y_true, y_pred, protected, privileged_group=0))

if __name__ == '__main__':
    print('--- Demo: FairnessMetrics & FairnessEngine ---')
    demo_fairness_metrics() 