"""
Setup configuration for DRL Trading Strategies package.

This setup enables installation via: pip install -e .
"""

from setuptools import setup, find_packages
from pathlib import Path

# 读取README
readme_file = Path(__file__).parent / "README_SUBMISSION.md"
long_description = readme_file.read_text() if readme_file.exists() else ""

setup(
    name="drl-trading-strategies",
    version="1.0.0",
    author="IEOR 4733 Project Team",
    author_email="",
    description="A modular Python framework for deep reinforcement learning trading strategies",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/IEOR4733_Project",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Office/Business :: Financial",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "scipy>=1.7.0",
        "torch>=1.9.0",
        "scikit-learn>=0.24.0",
        "yfinance>=0.1.70",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "plotly>=5.0.0",
        "statsmodels>=0.13.0",
        "pyyaml>=5.4.0",
        "pytest>=6.2.0",
        "jupyter>=1.0.0",
        "tqdm>=4.62.0",
    ],
    extras_require={
        "dev": [
            "black>=21.7b0",
            "flake8>=3.9.0",
            "mypy>=0.910",
            "pytest-cov>=2.12.0",
            "pytest-xdist>=2.3.0",
        ],
        "ml": [
            "optuna>=2.9.0",
            "ray>=1.10.0",
            "scikit-optimize>=0.9.0",
        ],
        "viz": [
            "plotly>=5.0.0",
            "tensorboard>=2.7.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "drl-baseline=src.scripts.run_baseline_strategies:main",
            "drl-dqn=src.scripts.run_dqn_model:main",
            "drl-evaluate=src.scripts.evaluate_all_strategies:main",
            "drl-report=src.scripts.generate_report:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
