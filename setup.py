from setuptools import setup, find_packages
from protopipe import __version__


def readme():
    with open("README.rst") as f:
        return f.read()


setup(
    name="protopipe",
    version=__version__,
    description="Pipeline to process events from DL0 to DL3",
    url="https://github.com/cta-observatory/protopipe",
    author="Michele Peresano, Karl Kosack, Thierry Stolarczyk, Alice Donini, Thomas Vuillaume",
    author_email="michele.peresano@cea.fr",
    license="MIT",
    packages=find_packages(),
    install_requires=["ctapipe"],
    zip_safe=False,
)
