import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
import numpy as np
from fairsight import Utils

def demo_utils():
    X = pd.DataFrame({'a': [1, 2], 'b': [3, 4], 'cat': ['x', 'y']})
    y = pd.Series([0, 1])
    print('validate_inputs:', Utils.validate_inputs(X, y, ['cat']))
    df = pd.DataFrame({'feature': [1, 2], 'target': [0, 1]})
    print('infer_target_column:', Utils.infer_target_column(df))
    y_class = pd.Series([0, 1, 0, 1])
    print('is_classification_task:', Utils.is_classification_task(y_class))
    y_reg = pd.Series([0.1, 0.2, 0.3, 0.4])
    print('is_regression_task:', Utils.is_regression_task(y_reg))
    processed, encoders = Utils.preprocess_data(df, 'feature', ['target'])
    print('preprocess_data encoders:', encoders)
    df2 = pd.DataFrame({'cat': ['x', 'y', 'x', 'x']})
    print('calculate_privilege_groups:', Utils.calculate_privilege_groups(df2, ['cat']))
    print('safe_divide (1, 0):', Utils.safe_divide(1, 0))
    print('safe_divide (4, 2):', Utils.safe_divide(4, 2))
    print('format_percentage (0.25):', Utils.format_percentage(0.25))
    df3 = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
    print('create_summary_stats:', Utils.create_summary_stats(df3))

if __name__ == '__main__':
    print('--- Demo: Utils ---')
    demo_utils() 