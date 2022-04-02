import os
from setuptools import setup, find_packages

ctapipe = "ctapipe==0.11.0"
gammapy = "gammapy==0.18.2"
pyirf = "pyirf==0.5.0"
pandas = "pandas==1.3.5"
scikit_learn = "scikit-learn==1.0.2"

extras_require = {
    "docs": [
        "jinja2==3.0.3",
        "pydata-sphinx-theme~=0.8",
        "sphinx-issues",
        "sphinx_automodapi",
        "numpydoc",
        "ipython",
        gammapy,
        "pytest",
        "numpy",
        ctapipe,
        pyirf,
        pandas,
    ],
    "tests": [
        # pipeline tests
        "pytest",
        "pytest-cov",
        "pytest-dependency",
        "codecov",
    ],
}

extras_require["all"] = list(set(extras_require["tests"] + extras_require["docs"]))


setup(
    name="protopipe",
    description="Prototype pipeline for the Cherenkov Telescope Array (CTA)",
    keywords="cta pipeline simulations grid",
    url="https://github.com/cta-observatory/protopipe",
    project_urls={
        "Documentation": "https://cta-observatory.github.io/protopipe/",
        "Source": "https://github.com/cta-observatory/protopipe",
        "Tracker": "https://github.com/cta-observatory/protopipe/issues",
        "Projects": "https://github.com/cta-observatory/protopipe/projects",
    },
    author="Michele Peresano",
    author_email="michele.peresano@cea.fr",
    license="CeCILL-B Free Software License Agreement",
    packages=find_packages(),
    package_data={"protopipe": ["aux/example_config_files/analysis.yaml"]},
    include_package_data=True,
    install_requires=[
        ctapipe,
        pyirf,
        pandas,
        scikit_learn,
        "jupyterlab>=3.1.10",
        "jupyter-book",
        "papermill==2.3.4",
    ],
    zip_safe=False,
    use_scm_version={"write_to": os.path.join("protopipe", "_version.py")},
    tests_require=extras_require["tests"],
    classifiers=[
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Astronomy",
        "Topic :: Scientific/Engineering :: Physics",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.8",
    ],
    python_requires=">=3.7",
    extras_require={
        "all": extras_require["tests"] + extras_require["docs"],
        "tests": extras_require["tests"],
        "docs": extras_require["docs"],
    },
    entry_points={
        "console_scripts": [
            "protopipe-TRAINING=protopipe.scripts.data_training:main",
            "protopipe-MODEL=protopipe.scripts.build_model:main",
            "protopipe-DL2=protopipe.scripts.write_dl2:main",
            "protopipe-DL3-EventDisplay=protopipe.scripts.make_performance_EventDisplay:main",
            "protopipe-BENCHMARK=protopipe.scripts.launch_benchmark:main",
        ]
    },
)
