"""
Fairsight Toolkit Setup
======================

Setup script for the Fairsight AI Ethics and Bias Detection Toolkit.
"""

from setuptools import setup, find_packages
import os

# Read README for long description
def read_readme():
    """Read README file for long description."""
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements from requirements.txt
def read_requirements():
    """Read requirements from requirements.txt."""
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="fairsight",
    version="1.0.0",
    author="Vijay K S",
    author_email="ksvijay2005@gmail.com",
    description="Comprehensive AI Ethics and Bias Detection Toolkit with SAP Integration",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/KS-Vijay/fairsight",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Software Development :: Quality Assurance",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme>=0.5",
            "sphinx-autodoc-typehints>=1.12",
        ],
        "sap": [
            "hdbcli>=2.15",
            "sap-hana>=1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "fairsight-audit=fairsight.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "fairsight": [
            "templates/*.html",
            "templates/*.md",
            "config/*.json",
            "fonts/*.ttf",
        ],
    },
    zip_safe=False,
    keywords=[
        "ai ethics",
        "bias detection", 
        "fairness",
        "machine learning",
        "audit",
        "sap hana",
        "explainability",
        "responsible ai",
    ],
    project_urls={
        "Bug Reports": "https://github.com/KS-Vijayk/fairsight/issues",
        "Source": "https://github.com/KS-Vijay/fairsight",
        "Documentation": "https://fairsight-toolkit.readthedocs.io/",
    },
)
