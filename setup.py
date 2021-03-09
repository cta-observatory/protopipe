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
        "gammapy",
        "pytest",
        "numpy",
        "ctapipe>=0.10.1",
        "pyirf",
    ],
    "tests": [
        "pytest",
        "pytest-cov",
        "codecov",
        "ctapipe-extra @ https://github.com/cta-observatory/ctapipe-extra/archive/v0.3.1.tar.gz",
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
    license="CeCILL-B Free Software License Agreement",
    packages=find_packages(),
    package_data={"protopipe": ["aux/example_config_files/protopipe/analysis.yaml"]},
    include_package_data=True,
    install_requires=["ctapipe>=0.10.1", "pyirf"],
    zip_safe=False,
    extras_require={
        "all": extras_require["all"],
        "tests": extras_require["tests"],
        "docs": extras_require["docs"],
    },
    entry_points={
        'console_scripts': [
            'protopipe-STEP1=protopipe.scripts.step1_tool_draft:main',
            'protopipe-TRAINING=protopipe.scripts.data_training:main',
            'protopipe-MODEL=protopipe.scripts.build_model:main',
            'protopipe-DL2=protopipe.scripts.write_dl2:main',
            'protopipe-DL3-EventDisplay=protopipe.scripts.make_performance_EventDisplay:main',
        ],
    },
)
