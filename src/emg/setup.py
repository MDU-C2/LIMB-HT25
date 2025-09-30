#!/usr/bin/env python3
"""Setup script for EMG package."""

from setuptools import setup, find_packages
import os

# Read README file
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return "EMG Signal Processing and Intent Classification Package"

# Read requirements
def read_requirements():
    requirements_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    if os.path.exists(requirements_path):
        with open(requirements_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return []

setup(
    name="emg-signal-processing",
    version="1.0.0",
    description="LSTM-based EMG signal processing and intent classification",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    author="EMG Research Team",
    author_email="emg@example.com",
    url="https://github.com/example/emg-signal-processing",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.2.0",
            "pytest-cov>=2.12.0",
            "black>=21.0.0",
            "flake8>=3.9.0",
        ],
        "visualization": [
            "matplotlib>=3.4.0",
            "seaborn>=0.11.0",
        ],
        "jupyter": [
            "jupyter>=1.0.0",
            "ipywidgets>=7.6.0",
        ],
        "audio": [
            "librosa>=0.8.0",
            "sounddevice>=0.4.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "emg-train=emg.example_usage:main",
            "emg-test=emg.test_emg_package:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
