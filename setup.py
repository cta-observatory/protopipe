from setuptools import setup, find_packages
from protopipe import __version__


def readme():
    with open("README.rst") as f:
        return f.read()


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
)
