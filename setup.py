from setuptools import find_packages, setup


def get_long_description():
    with open("README.md", "r", encoding="utf-8") as f:
        long_description = f.read()

    long_description = long_description.replace(
        "[Read More...](#AraBERT)",
        "[Read More...](https://github.com/aub-mind/arabert/tree/master/arabert)",
    )
    long_description = long_description.replace(
        "[Read More...](#AraGPT2)",
        "[Read More...](https://github.com/aub-mind/arabert/tree/master/aragpt2)",
    )
    long_description = long_description.replace(
        "[Read More...](#AraELECTRA)",
        "[Read More...](https://github.com/aub-mind/arabert/tree/master/araelectra)",
    )
    long_description = long_description.replace(
        "[preprocessing function](#Preprocessing)",
        "https://github.com/aub-mind/arabert#preprocessing",
    )
    long_description = long_description.replace(
        "[Dataset Section](#Dataset)", "https://github.com/aub-mind/arabert#Dataset"
    )
    long_description = long_description.replace(
        "https://github.com/aub-mind/arabert/blob/master/",
        "https://raw.githubusercontent.com/aub-mind/arabert/master/",
    )
    return long_description


setup(
    name="arabert",
    version="1.0.1",
    author="AUB MIND Lab",
    maintainer="Wissam Antoun",
    maintainer_email="wissam.antoun@gmail.com",
    url="https://github.com/aub-mind/arabert",
    description="AraBERT is a Python library that contains the"
    "code for the AraBERT, AraGPT2 and AraELECTRA models with"
    "the preprocessing scripts.",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    install_requires=["PyArabic", "farasapy", "emoji==1.4.2"],
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
