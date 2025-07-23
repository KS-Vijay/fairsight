import pandas as pd
from fairsight import ReportGenerator
import os

def demo_report_generator():
    sample_results = {
        'bias': {'category': {'disparate_impact': 0.7, 'statistical_parity_difference': 0.1}},
        'fairness': {'category': {'demographic_parity': type('obj', (object,), {'ratio': 0.7, 'threshold_met': False})()}},
        'explainability': []
    }
    output_dir = 'demo_reports'
    os.makedirs(output_dir, exist_ok=True)
    rg = ReportGenerator(output_dir=output_dir)
    md_path = rg.generate_markdown_report('TestModel', bias_results=sample_results['bias'], fairness_results=sample_results['fairness'])
    print('Markdown report generated at:', md_path)
    html_path = rg.generate_html_report('TestModel', md_path)
    print('HTML report generated at:', html_path)
    pdf_path = rg.generate_pdf_report(md_path, 'TestModel')
    print('PDF report generated at:', pdf_path)

if __name__ == '__main__':
    print('--- Demo: ReportGenerator ---')
    demo_report_generator() 