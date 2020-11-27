from setuptools import setup, find_packages
from protopipe import __version__


def readme():
    with open("README.rst") as f:
        return f.read()


extras_require = {
    "docs": [
        "sphinx_rtd_theme",
        "sphinx-issues",
        "sphinx_automodapi",
        "nbsphinx",
        "numpydoc",
        "ipython",
        "gammapy == 0.8",
        "pytest",
        "numpy",
    ],
    "tests": [
        "pytest",
        "pytest-cov",
        "codecov",
    ],
}

extras_require["all"] = list(set(extras_require["tests"] + extras_require["docs"]))


setup(
    name="protopipe",
    version=__version__,
    description="Prototype pipeline for the Cherenkov Telescope Array (CTA)",
    url="https://github.com/cta-observatory/protopipe",
    author="Michele Peresano",
    author_email="michele.peresano@cea.fr",
    license="MIT",
    packages=find_packages(),
    package_data={"protopipe": ["aux/example_config_files/protopipe/analysis.yaml"]},
    include_package_data=True,
    install_requires=["ctapipe"],
    zip_safe=False,
    extras_require={
        "all": extras_require["all"],
        "tests": extras_require["tests"],
        "docs": extras_require["docs"],
    },
)
