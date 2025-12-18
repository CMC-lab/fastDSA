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
    license="MIT",  # adjust if needed
    python_requires=">=3.9",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    include_package_data=True,
    install_requires=[
        # core
        "numpy>=1.24.0",
        "torch>=1.3.0",

        # optimal transport (imported as `ot`)
        "pot>=0.9.0",

        # used in RegularizationTerm (e.g., TruncatedSVD)
        "scikit-learn>=1.0.0",

        # progress bars (LandingAlgorithm)
        "tqdm>=4.0.0",

        # required by RiemannianManifold (method='rim')
        "geoopt>=0.5.0",

        # required by kwDSA path (method='kw')
        "kooplearn>=1.1.0",

        # practical runtime dependency (POT depends on SciPy anyway, but make it explicit)
        "scipy>=1.6.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
        ],
        # Optional notebook conveniences (not required for the library itself)
        "tutorial": [
            "matplotlib>=3.5.0",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "License :: OSI Approved :: MIT License",
    ],
)
