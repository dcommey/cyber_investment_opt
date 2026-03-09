from setuptools import setup, find_packages

setup(
    name="cyber_investment",
    version="0.1.0",
    description="Power-system-informed cybersecurity investment optimization for transmission-grid control networks",
    packages=find_packages(where=".", include=["src", "src.*"]),
    package_dir={"": "."},
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.21",
        "networkx>=2.6",
        "scipy>=1.7",
        "pandas>=1.3",
        "matplotlib>=3.7",
    ],
)