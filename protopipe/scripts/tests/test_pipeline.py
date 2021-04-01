from os import system
from pkg_resources import resource_filename

import tables
import pytest

from protopipe.pipeline.temp import get_dataset_path
from protopipe.scripts import data_training, build_model


# PROD 3B

# CONFIG FILES
config_prod3b_CTAN = resource_filename("protopipe", "scripts/tests/test_config_analysis_north.yaml")
config_prod3b_CTAS = resource_filename("protopipe", "scripts/tests/test_config_analysis_south.yaml")

# TEST FILES

URL_TEST_DATA = "http://cccta-dataserver.in2p3.fr/data/protopipe/testData/"
URL_PROD3B_CTAN = f"{URL_TEST_DATA}/prod3_laPalma_baseline_Az180_Zd20"
URL_PROD3B_CTAS = f"{URL_TEST_DATA}/prod3_Paranal_baseline_Az180_Zd20"

input_data = {

    "PROD3B_CTA_NORTH": {"config": config_prod3b_CTAN,
                         "gamma1": get_dataset_path("gamma1.simtel.gz",
                                                    url=f"{URL_PROD3B_CTAN}"),
                         "gamma2": get_dataset_path("gamma2.simtel.gz",
                                                    url=f"{URL_PROD3B_CTAN}"),
                         "proton1": get_dataset_path("proton1.simtel.gz",
                                                     url=f"{URL_PROD3B_CTAN}"),
                         },

    "PROD3B_CTA_SOUTH": {"config": config_prod3b_CTAS,
                         "gamma1": get_dataset_path("gamma1.simtel.gz",
                                                    url=f"{URL_PROD3B_CTAS}"),
                         "gamma2": get_dataset_path("gamma2.simtel.gz",
                                                    url=f"{URL_PROD3B_CTAS}"),
                         "proton1": get_dataset_path("proton1.simtel.gz",
                                                     url=f"{URL_PROD3B_CTAS}"),
                         }

}


@pytest.mark.parametrize("test_case", ["PROD3B_CTA_NORTH", "PROD3B_CTA_SOUTH"])
def test_GET_GAMMAS_FOR_ENERGY_MODEL_WITH_IMAGES(test_case, pipeline_testdir):

    outpath = pipeline_testdir / f"test_training_withImages_{test_case}.h5"

    command = f"python {data_training.__file__}\
    --config_file {input_data[test_case]['config']}\
    -o {outpath}\
    --save_images\
    -i {input_data[test_case]['gamma1'].parent}\
    -f {input_data[test_case]['gamma1'].name}"

    print(  # only with "pytest -s"
        f'''
        /n You can reproduce this test by running the following command,

        {command}
        '''
    )

    exit_status = system(command)

    # check that the script ends without crashing
    assert exit_status == 0

    # check that the produced HDF5 file is non-empty
    with tables.open_file(outpath) as file:
        assert file.get_filesize() > 0


@pytest.mark.parametrize("test_case", [
    pytest.param("PROD3B_CTA_NORTH", marks=pytest.mark.dependency(name="g1N")),
    pytest.param("PROD3B_CTA_SOUTH", marks=pytest.mark.dependency(name="g1S")),
])
def test_GET_GAMMAS_FOR_ENERGY_MODEL(test_case, pipeline_testdir):

    outpath = pipeline_testdir / f"test_gamma1_noImages_{test_case}.h5"

    command = f"python {data_training.__file__}\
    --config_file {input_data[test_case]['config']}\
    -o {outpath}\
    -i {input_data[test_case]['gamma1'].parent}\
    -f {input_data[test_case]['gamma1'].name}"

    print(  # only with "pytest -s"
        f'''
        /n You can reproduce this test by running the following command,

        {command}
        '''
    )

    exit_status = system(command)

    # check that the script ends without crashing
    assert exit_status == 0

    # check that the produced HDF5 file is non-empty
    with tables.open_file(outpath) as file:
        assert file.get_filesize() > 0


@pytest.mark.parametrize("test_case", [
    pytest.param("PROD3B_CTA_NORTH", marks=pytest.mark.dependency(name="EN",
                                                                  depends=["g1N"])),
    pytest.param("PROD3B_CTA_SOUTH", marks=pytest.mark.dependency(name="ES",
                                                                  depends=["g1S"])),
])
def test_BUILD_ENERGY_MODEL_AdaBoost_DecisionTreeRegressor(test_case, pipeline_testdir):
    """Launch protopipe.scripts.build_model for a AdaBoost DecisionTreeRegressor."""

    infile = pipeline_testdir / f"test_gamma1_noImages_{test_case}.h5"
    outdir = pipeline_testdir / f"energy_model_{test_case}"

    config = resource_filename("protopipe", "scripts/tests/test_regressor.yaml")

    command = f"python {build_model.__file__}\
    --config_file {config}\
    --infile_signal {infile}\
    --outdir {outdir}\
    --cameras_from_file"

    print(  # only with "pytest -s"
        f'''
        /n You can reproduce this test by running the following command,

        {command}
        '''
    )

    exit_status = system(command)
    assert exit_status == 0


@pytest.mark.parametrize("test_case", [
    pytest.param("PROD3B_CTA_NORTH", marks=pytest.mark.dependency(name="g2N",
                                                                  depends=["EN"])),
    pytest.param("PROD3B_CTA_SOUTH", marks=pytest.mark.dependency(name="g2S",
                                                                  depends=["ES"])),
])
def test_GET_GAMMAS_FOR_CLASSIFICATION_MODEL(test_case, pipeline_testdir):

    modelpath = pipeline_testdir / f"energy_model_{test_case}"
    outpath = pipeline_testdir / f"test_gamma2_noImages_{test_case}.h5"

    command = f"python {data_training.__file__}\
    --config_file {input_data[test_case]['config']}\
    -o {outpath}\
    -i {input_data[test_case]['gamma2'].parent}\
    -f {input_data[test_case]['gamma2'].name}\
    --estimate_energy True\
    --regressor_dir {modelpath}"

    print(  # only with "pytest -s"
        f'''
        /n You can reproduce this test by running the following command,

        {command}
        '''
    )

    exit_status = system(command)

    # check that the script ends without crashing
    assert exit_status == 0

    # check that the produced HDF5 file is non-empty
    with tables.open_file(outpath) as file:
        assert file.get_filesize() > 0


@pytest.mark.parametrize("test_case", [
    pytest.param("PROD3B_CTA_NORTH", marks=pytest.mark.dependency(name="p1N",
                                                                  depends=["EN"])),
    pytest.param("PROD3B_CTA_SOUTH", marks=pytest.mark.dependency(name="p1S",
                                                                  depends=["ES"])),
])
def test_GET_PROTONS_FOR_CLASSIFICATION_MODEL(test_case, pipeline_testdir):

    modelpath = pipeline_testdir / f"energy_model_{test_case}"
    outpath = pipeline_testdir / f"test_proton1_noImages_{test_case}.h5"

    command = f"python {data_training.__file__}\
    --config_file {input_data[test_case]['config']}\
    -o {outpath}\
    -i {input_data[test_case]['proton1'].parent}\
    -f {input_data[test_case]['proton1'].name}\
    --estimate_energy True\
    --regressor_dir {modelpath}"

    print(  # only with "pytest -s"
        f'''
        /n You can reproduce this test by running the following command,

        {command}
        '''
    )

    exit_status = system(command)

    # check that the script ends without crashing
    assert exit_status == 0

    # check that the produced HDF5 file is non-empty
    with tables.open_file(outpath) as file:
        assert file.get_filesize() > 0


@pytest.mark.parametrize("test_case", [
    pytest.param("PROD3B_CTA_NORTH", marks=pytest.mark.dependency(name="C1",
                                                                  depends=["g2N", "p1N"])),
    pytest.param("PROD3B_CTA_SOUTH", marks=pytest.mark.dependency(name="C2",
                                                                  depends=["g2S", "p1S"])),
])
def test_BUILD_CLASSIFICATION_MODEL_RandomForest(test_case, pipeline_testdir):
    """Launch protopipe.scripts.build_model for a Random Forest classifier."""

    infile_signal = pipeline_testdir / f"test_gamma2_noImages_{test_case}.h5"
    infile_background = pipeline_testdir / f"test_proton1_noImages_{test_case}.h5"
    outdir = pipeline_testdir / f"classification_model_{test_case}"

    config = resource_filename("protopipe", "scripts/tests/test_classifier.yaml")

    command = f"python {build_model.__file__}\
    --config_file {config}\
    --infile_signal {infile_signal}\
    --infile_background {infile_background}\
    --outdir {outdir}\
    --cameras_from_file"

    print(  # only with "pytest -s"
        f'''
        /n You can reproduce this test by running the following command,

        {command}
        '''
    )

    exit_status = system(command)
    assert exit_status == 0
