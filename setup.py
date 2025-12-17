from pathlib import Path
from setuptools import setup, find_packages

ROOT = Path(__file__).resolve().parent

# Optional: pull long description from README if present
readme_path = ROOT / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

setup(
    name="fastDSA",
    version="1.0.1",
    description="Faster version of Dynamical Similarity Analysis (DSA)",
    long_description=long_description,
    long_description_content_type="text/markdown" if long_description else None,
    url="https://github.com/CMC-lab/fastDSA",
    author="CMC lab",
    author_email="",  # add a real email if you want it on PyPI
    license="MIT",    # change if your LICENSE is different
    python_requires=">=3.9",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    include_package_data=True,
    install_requires=[
        "numpy>=1.24.0",
        "torch>=1.3.0",
        "pot",
        "kooplearn>=1.1.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
