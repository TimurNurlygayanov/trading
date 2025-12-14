"""
Setup script for algo_trading package.
"""
from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_path = Path(__file__).parent / "README.md"
long_description = ""
if readme_path.exists():
    long_description = readme_path.read_text(encoding="utf-8")

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_path.exists():
    requirements = [
        line.strip()
        for line in requirements_path.read_text().splitlines()
        if line.strip() and not line.startswith("#") and not line.startswith("-")
    ]

setup(
    name="algo_trading",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Modular algorithmic trading framework with RL support",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/algo_trading",
    packages=find_packages(exclude=["tests", "tests.*"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Office/Business :: Financial :: Investment",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.3.0",
            "pytest-cov>=4.1.0",
            "black>=23.3.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
            "mypy>=1.3.0",
        ],
        "rl": [
            "stable-baselines3>=2.0.0",
            "gymnasium>=0.28.0",
            "torch>=2.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "algo-backtest=benchmarks.runner:main",
            "algo-dashboard=ui.dashboard:main",
            "algo-train=strategies.rl_ppo.training:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
