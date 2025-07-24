import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
from fairsight import BiasDetector, detect_dataset_bias

def demo_bias_detection():
    df = pd.DataFrame({
        'category': [0, 1, 0, 1],
        'intent': [1, 0, 1, 0]
    })
    detector = BiasDetector(dataset=df, sensitive_features=['category'], target='intent')
    print('BiasDetector results:', detector.detect_bias_on_dataset())
    print('detect_dataset_bias results:', detect_dataset_bias(df, ['category'], 'intent'))

if __name__ == '__main__':
    print('--- Demo: Bias Detection ---')
    demo_bias_detection() 