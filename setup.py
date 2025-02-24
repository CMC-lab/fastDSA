import setuptools

setuptools.setup(
    name="fastDSA",
    version="1.0.1",
    url="https://github.com/CMC-lab/fastDSA",

    author="CMClab",
    author_email="_",

    description="faster version of  Dynamical Similarity Analysis(DSA)",
    packages=setuptools.find_packages(),
    install_requires=[
        'numpy>=1.24.0',
        'torch>=1.3.0',
        'kooplearn>=1.1.0',
        'pot'
    ],
    extras_require={
        'dev': [
            'pytest>=3.7'
        ]
    },
)