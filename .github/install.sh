#!/bin/bash

if [[ "$INSTALL_METHOD" == "conda" ]]; then
  echo "Using conda"
  source $CONDA/etc/profile.d/conda.sh
  conda config --set always_yes yes --set changeps1 no
  conda update -q conda  # get latest conda version
  # Useful for debugging any issues with conda
  conda info -a

  sed -i -e "s/- python=.*/- python=$PYTHON_VERSION/g" environment_development.yml
  conda install -c conda-forge mamba
  # Temporary workaround: https://github.com/mamba-org/mamba/issues/488
  rm -rf /usr/share/miniconda/pkgs/cache/*.json
  mamba env create -n protopipe --file environment_development.yml
  conda activate protopipe
else
  echo "Using pip"
  pip install -U pip setuptools wheel
fi
