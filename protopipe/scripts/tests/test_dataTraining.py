"""Test the data training script.

TODO
----

- test only diffuse data (more general case)
- add Paranal diffuse test file
- add Prod5 test files

"""
from os import path, system
from pkg_resources import resource_filename
from protopipe.scripts import data_training
from protopipe.pipeline.temp import get_dataset_path

# configuration files
ana_config = resource_filename("protopipe", "aux/example_config_files/analysis.yaml")

# TEST FILES

# Prod 2

# CTA_SOUTH = get_dataset_path("gamma_test_large.simtel.gz")

# PROD 3b

CTA_NORTH = get_dataset_path("gamma_LaPalma_baseline_20Zd_180Az_prod3b_test.simtel.gz")




def test_dataTraining_noImages():
    """Very bare test to see if the script reaches the end correctly.

    WARNING: some of the cuts in the example config file are not optimized for
    cameras other than LSTCam and NectarCam.
    In any case, it is expected that in absence of fatal bugs, the script
    ends successfully.
    """
    exit_status = system(
        f"python {data_training.__file__}\
        --config_file {ana_config}\
        -o test_training_noImages.h5\
        -m 10\
        --debug\
        -i {path.dirname(CTA_NORTH)}\
        -f {path.basename(CTA_NORTH)}"
    )
    assert exit_status == 0


def test_dataTraining_withImages():
    """Very bare test to see if the script reaches the end correctly.

    WARNING: some of the cuts in the example config file are not optimized for
    cameras other than LSTCam and NectarCam.
    In any case, it is expected that in absence of fatal bugs, the script
    ends successfully.
    """
    exit_status = system(
        f"python {data_training.__file__}\
        --config_file {ana_config}\
        -o test_training_withImages.h5\
        -m 10\
        --save_images\
        -i {path.dirname(CTA_NORTH)}\
        -f {path.basename(CTA_NORTH)}"
    )
    assert exit_status == 0
