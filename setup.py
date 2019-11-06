from setuptools import setup, find_packages


def readme():
    with open("README.rst") as f:
        return f.read()


setup(
    name="protopipe",
    version="0.2.1-dev",
    description="Pipeline to process events from DL0 to DL3",
    url="https://github.com/cta-observatory/protopipe",
    author="CEA",
    author_email="michele.peresano@cea.fr",
    license="MIT",
    packages=find_packages(),
    install_requires=["ctapipe"],
    zip_safe=False,
)
