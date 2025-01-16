from setuptools import setup, find_packages

setup(
    name="blink",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.0",
    install_requires=[
        "numpy",
        "pandas",
        "scipy",
        "scikit-learn==1.0.2",
        "seaborn",
        "jupyter",
        "matplotlib",
        "rdkit-pypi",  # rdkit from PyPI
        "matchms",
        "networkx",
        "pymzml",
        "pyteomics",
        "torch",  # Specified under pip in environment.yaml
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    entry_points={
        "console_scripts": [
            # Add CLI scripts here if needed, e.g., "blink-cli = blink.cli:main"
        ]
    },
)
