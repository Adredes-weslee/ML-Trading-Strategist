from setuptools import setup, find_packages

setup(
    name="TradingStrategist",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy",
        "pandas",
        "matplotlib",
        "scikit-learn",
        "pyyaml",
        "scipy",
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="ML Trading Strategist package",
)