from setuptools import find_packages, setup

setup(
    name="arabert",
    version="1.0.0",
    author="AUB MIND Lab",
    maintainer="Wissam Antoun",
    maintainer_email="wissam.antoun@gmail.com",
    url="https://github.com/aub-mind/arabert",
    description="AraBERT is a Python library that contains the"
    "code for the AraBERT, AraGPT2 and AraELECTRA models with"
    "the preprocessing scripts.",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    install_requires=["PyArabic", "farasapy"],
    py_modules=["arabert.preprocess"],
    package_dir={"arabert": "."},
    packages=[
        "arabert.{}".format(p)
        for p in find_packages(
            where=".",
            exclude=["testing", "data", "examples"],
        )
    ],
    python_requires=">=3.6.0",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
