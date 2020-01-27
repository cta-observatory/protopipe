"""Test the write_dl1 script in a variety of conditions."""
from os import path, system
from pkg_resources import resource_filename
from protopipe.scripts import write_dl1
from ctapipe.utils import get_dataset_path

# TEST FILES
# 110 events, 98 telescopes at Paranal.
# Instruments tested: LST_LST_LSTCam, MST_MST_FlashCam, SST_ASTRI_ASTRICam
GAMMA_TEST_LARGE = get_dataset_path("gamma_test_large.simtel.gz")
# WARNING: absolutely not sufficient!
# This is just the only file easily usable without external resources.
# Later on, we will need a sub-simtel file from each of the
# MC productions expected to be analyzed with protopipe.

# configuration files
ana_config = resource_filename(
    "protopipe", "aux/example_config_files/protopipe/analysis.yaml"
)


def test_write_dl1():
    """Very bare test to see if the script reaches the end correctly.

    WARNING: some of the cuts in the example config file are not optimized for
    cameras other than LSTCam and NectarCam.
    In any case, it is expected that in absence of fatal bugs, the script
    ends successfully.
    """
    exit_status = system(
        f"python {write_dl1.__file__}\
        --config_file {ana_config}\
        -o test_dl1.h5\
        -i {path.dirname(GAMMA_TEST_LARGE)}\
        -f {path.basename(GAMMA_TEST_LARGE)}"
    )
    assert exit_status == 0
