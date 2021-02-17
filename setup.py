from setuptools import setup, find_packages
from protopipe import __version__

extras_require = {
    "docs": [
        "sphinx_rtd_theme",
        "sphinx-issues",
        "sphinx_automodapi",
        "nbsphinx",
        "numpydoc",
        "ipython",
        "gammapy==0.8",
        "pytest",
        "numpy",
        "ctapipe==0.9.1",
        "pyirf",
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
    keywords='cta pipeline simulations grid',
    url="https://github.com/cta-observatory/protopipe",
    project_urls={
        'Documentation': 'https://cta-observatory.github.io/protopipe/',
        'Source': 'https://github.com/cta-observatory/protopipe',
        'Tracker': 'https://github.com/cta-observatory/protopipe/issues',
        'Projects': 'https://github.com/cta-observatory/protopipe/projects'
    },
    author="Michele Peresano",
    author_email="michele.peresano@cea.fr",
    license="MIT",
    packages=find_packages(),
    package_data={"protopipe": ["aux/example_config_files/analysis.yaml"]},
    include_package_data=True,
    install_requires=["ctapipe"],
    zip_safe=False,
    classifiers=[
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Astronomy',
        'Topic :: Scientific/Engineering :: Physics',
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    python_requires='>=3',
    extras_require={
        "all": extras_require["all"],
        "tests": extras_require["tests"],
        "docs": extras_require["docs"],
    },
    entry_points={
        'console_scripts': [
            'protopipe-TRAINING=protopipe.scripts.write_dl1:main',
            'protopipe-MODEL=protopipe.scripts.build_model:main',
            'protopipe-DL2=protopipe.scripts.write_dl2:main',
            'protopipe-DL3=protopipe.scripts.make_performance:main',
            ],
    },
)
