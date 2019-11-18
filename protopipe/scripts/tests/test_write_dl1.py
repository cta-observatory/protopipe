"""Test the write_dl1 script."""
import os
import protopipe
from protopipe.scripts import write_dl1
from ctapipe.utils import get_dataset_path

# TEST FILES
# Instruments tested: LST_LST_LSTCam, MST_MST_FlashCam, SST_ASTRI_ASTRICam
GAMMA_TEST_LARGE = get_dataset_path("gamma_test_large.simtel.gz")

parentdir = os.path.dirname(protopipe.__path__[0])
configs = os.path.join(parentdir, "aux/example_config_files/protopipe/")


def test_write_dl1():
    """Very bare test to see if the script reaches the end correctly.

    WARNING: this is unfortunately not sufficient, since we work with many
    different simulation setups, but for the moment in protopipe we provide
    automatic tests made only using GAMMA_TEST_LARGE.
    """
    exit_status = os.system(
        f"python {write_dl1.__file__}\
        --config_file {os.path.join(configs, 'analysis.yaml')}\
        -o test_dl1.h5\
        -i {os.path.dirname(GAMMA_TEST_LARGE)}\
        -f {os.path.basename(GAMMA_TEST_LARGE)}\
        -m 10"
    )
    assert exit_status == 0
