#!/bin/bash

echo "Using conda"
source $CONDA/etc/profile.d/conda.sh
conda config --set always_yes yes --set changeps1 no
conda update -q conda  # get latest conda version
# Useful for debugging any issues with conda
conda info -a

sed -i -e "s/- python=.*/- python=$PYTHON_VERSION/g" environment.yml
conda install -c conda-forge mamba
mamba env create -n protopipe --file environment.yml
conda activate protopipe
