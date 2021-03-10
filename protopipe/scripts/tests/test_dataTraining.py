"""Test the data training script.

TODO
----

- test only diffuse data (more general case)
- add Paranal diffuse test file
- add Prod5 test files

"""
from os import path, system
from pkg_resources import resource_filename

import tables
import pytest

from protopipe.scripts import data_training
from protopipe.pipeline.temp import get_dataset_path

# TEST FILES

# PROD 3b

PROD3B_CTA_NORTH = get_dataset_path("gamma_LaPalma_baseline_20Zd_180Az_prod3b_test.simtel.gz")
PROD3B_CTA_SOUTH = get_dataset_path("gamma_Paranal_baseline_20Zd_180Az_prod3_test.simtel.gz")


@pytest.mark.parametrize("input_file", [PROD3B_CTA_NORTH, PROD3B_CTA_SOUTH])
def test_dataTraining_noImages(input_file):
    """Very bare test to see if the script reaches the end correctly.

    WARNING: some of the cuts in the example config file are not optimized for
    cameras other than LSTCam and NectarCam.
    In any case, it is expected that in absence of fatal bugs, the script
    ends successfully.
    """

    # the difference is only the 'site' key as a check for the user
    if "Paranal" in str(input_file):
        ana_config = resource_filename("protopipe", "scripts/tests/test_config_analysis_south.yaml")
    else:
        ana_config = resource_filename("protopipe", "scripts/tests/test_config_analysis_north.yaml")

    exit_status = system(
        f"python {data_training.__file__}\
        --config_file {ana_config}\
        -o test_training_noImages.h5\
        -m 10\
        -i {path.dirname(input_file)}\
        -f {path.basename(input_file)}"
    )

    # check that the script ends without crashing
    assert exit_status == 0

    # check that the produced HDF5 file is non-empty
    with tables.open_file("test_training_noImages.h5") as file:
        assert file.get_filesize() > 0


@pytest.mark.parametrize("input_file", [PROD3B_CTA_NORTH, PROD3B_CTA_SOUTH])
def test_dataTraining_withImages(input_file):
    """Very bare test to see if the script reaches the end correctly.

    WARNING: some of the cuts in the example config file are not optimized for
    cameras other than LSTCam and NectarCam.
    In any case, it is expected that in absence of fatal bugs, the script
    ends successfully.
    """

    # the difference is only the 'site' key as a check for the user
    if "Paranal" in str(input_file):
        ana_config = resource_filename("protopipe", "scripts/tests/test_config_analysis_south.yaml")
    else:
        ana_config = resource_filename("protopipe", "scripts/tests/test_config_analysis_north.yaml")

    exit_status = system(
        f"python {data_training.__file__}\
        --config_file {ana_config}\
        -o test_training_withImages.h5\
        -m 10\
        --save_images\
        -i {path.dirname(input_file)}\
        -f {path.basename(input_file)}"
    )

    # check that the script ends without crashing
    assert exit_status == 0

    # check that the produced HDF5 file is non-empty
    with tables.open_file("test_training_noImages.h5") as file:
        assert file.get_filesize() > 0
