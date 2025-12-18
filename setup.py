from pathlib import Path
from setuptools import setup, find_packages

ROOT = Path(__file__).resolve().parent
readme_path = ROOT / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

setup(
    name="fastDSA",
    version="1.0.1",
    description="Fast Dynamical Similarity Analysis (fastDSA)",
    long_description=long_description,
    long_description_content_type="text/markdown" if long_description else None,
    url="https://github.com/CMC-lab/fastDSA",
    author="CMC lab",
    author_email="",
    license="MIT",  # adjust if your LICENSE differs
    python_requires=">=3.9",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    include_package_data=True,
    install_requires=[
        "numpy>=1.24.0",
        "torch>=1.3.0",
        "pot",                 # provides `ot`
        "scikit-learn>=1.0.0", # RegularizationTerm uses sklearn (e.g., TruncatedSVD)
        "tqdm>=4.0.0",         # used for progress bars (LandingAlgorithm)
    ],
    extras_require={
        # Riemannian manifold backend (method="rim")
        "rim": [
            "geoopt>=0.5.0",
        ],
        # Kernel-Wasserstein backend (method="kw")
        "kw": [
            "kooplearn>=1.1.0",
        ],
        # Notebook/tutorial conveniences (ODE generation, plotting, MDS, etc.)
        "tutorial": [
            "scipy>=1.8.0",
            "matplotlib>=3.5.0",
        ],
        # Developer / test dependencies
        "dev": [
            "pytest>=7.0",
        ],
        # One-shot install for all optional features
        "all": [
            "geoopt>=0.5.0",
            "kooplearn>=1.1.0",
            "scipy>=1.8.0",
            "matplotlib>=3.5.0",
            "pytest>=7.0",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "License :: OSI Approved :: MIT License",
    ],
)
