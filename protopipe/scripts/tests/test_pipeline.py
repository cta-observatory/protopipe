from pathlib import Path
from os import system
from pkg_resources import resource_filename

import tables
import pytest

from ctapipe.utils.datasets import get_dataset_path
from protopipe.scripts import (
    data_training,
    build_model,
    write_dl2,
    make_performance_EventDisplay,
)


# PROD 3B

# CONFIG FILES
config_prod3b_CTAN = resource_filename(
    "protopipe", "scripts/tests/test_config_analysis_north.yaml"
)
config_prod3b_CTAS = resource_filename(
    "protopipe", "scripts/tests/test_config_analysis_south.yaml"
)
config_AdaBoostRegressor = resource_filename(
    "protopipe", "scripts/tests/test_AdaBoostRegressor.yaml"
)
config_RandomForestRegressor = resource_filename(
    "protopipe", "scripts/tests/test_RandomForestRegressor.yaml"
)
config_RandomForestClassifier = resource_filename(
    "protopipe", "scripts/tests/test_RandomForestClassifier.yaml"
)
config_DL3_ED_prod3b = resource_filename(
    "protopipe", "scripts/tests/test_performance_ED_prod3b.yaml"
)

# TEST FILES

URL_TEST_DATA = "http://cccta-dataserver.in2p3.fr/data/protopipe/testData/"
URL_PROD3B_CTAN = f"{URL_TEST_DATA}/prod3_laPalma_baseline_Az180_Zd20"
URL_PROD3B_CTAS = f"{URL_TEST_DATA}/prod3_Paranal_baseline_Az180_Zd20"

input_data = {
    "PROD3B_CTA_NORTH": {
        "config": config_prod3b_CTAN,
        "gamma1": get_dataset_path("gamma1.simtel.gz", url=f"{URL_PROD3B_CTAN}"),
        "gamma2": get_dataset_path("gamma2.simtel.gz", url=f"{URL_PROD3B_CTAN}"),
        "gamma3": get_dataset_path("gamma3.simtel.gz", url=f"{URL_PROD3B_CTAN}"),
        "proton1": get_dataset_path("proton1.simtel.gz", url=f"{URL_PROD3B_CTAN}"),
        "proton2": get_dataset_path("proton2.simtel.gz", url=f"{URL_PROD3B_CTAN}"),
        "electron1": get_dataset_path("electron1.simtel.gz", url=f"{URL_PROD3B_CTAN}"),
    },
    "PROD3B_CTA_SOUTH": {
        "config": config_prod3b_CTAS,
        "gamma1": get_dataset_path("gamma1.simtel.gz", url=f"{URL_PROD3B_CTAS}"),
        "gamma2": get_dataset_path("gamma2.simtel.gz", url=f"{URL_PROD3B_CTAS}"),
        "gamma3": get_dataset_path("gamma3.simtel.gz", url=f"{URL_PROD3B_CTAS}"),
        "proton1": get_dataset_path("proton1.simtel.gz", url=f"{URL_PROD3B_CTAS}"),
        "proton2": get_dataset_path("proton2.simtel.gz", url=f"{URL_PROD3B_CTAS}"),
        "electron1": get_dataset_path("electron1.simtel.gz", url=f"{URL_PROD3B_CTAS}"),
    },
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
        f"""
        You can reproduce this test by running the following command,

        {command}
        """
    )

    exit_status = system(command)

    # check that the script ends without crashing
    assert exit_status == 0

    # check that the produced HDF5 file is non-empty
    with tables.open_file(outpath) as file:
        assert file.get_filesize() > 0


@pytest.mark.parametrize(
    "test_case",
    [
        pytest.param("PROD3B_CTA_NORTH", marks=pytest.mark.dependency(name="g1N")),
        pytest.param("PROD3B_CTA_SOUTH", marks=pytest.mark.dependency(name="g1S")),
    ],
)
def test_GET_GAMMAS_FOR_ENERGY_MODEL(test_case, pipeline_testdir):

    outpath = pipeline_testdir / f"test_gamma1_noImages_{test_case}.h5"

    command = f"python {data_training.__file__}\
    --config_file {input_data[test_case]['config']}\
    -o {outpath}\
    -i {input_data[test_case]['gamma1'].parent}\
    -f {input_data[test_case]['gamma1'].name}"

    print(  # only with "pytest -s"
        f"""
        You can reproduce this test by running the following command,

        {command}
        """
    )

    exit_status = system(command)

    # check that the script ends without crashing
    assert exit_status == 0

    # sanity checks on the produced HDF5 file
    with tables.open_file(outpath) as file:
        assert file.get_filesize() > 0
        assert file.root._v_attrs["status"] == "complete"


@pytest.mark.parametrize(
    "test_case",
    [
        pytest.param(
            "PROD3B_CTA_NORTH",
            marks=pytest.mark.dependency(name="EN_1", depends=["g1N"]),
        ),
        pytest.param(
            "PROD3B_CTA_SOUTH",
            marks=pytest.mark.dependency(name="ES_1", depends=["g1S"]),
        ),
    ],
)
def test_BUILD_ENERGY_MODEL_AdaBoost_DecisionTreeRegressor(test_case, pipeline_testdir):
    """Launch protopipe.scripts.build_model for a AdaBoostRegressor based on DecisionTreeRegressor."""

    infile = pipeline_testdir / f"test_gamma1_noImages_{test_case}.h5"
    outdir = pipeline_testdir / f"energy_model_{test_case}"

    command = f"python {build_model.__file__}\
    --config_file {config_AdaBoostRegressor}\
    --infile_signal {infile}\
    --outdir {outdir}\
    --cameras_from_file"

    print(  # only with "pytest -s"
        f"""
        You can reproduce this test by running the following command,

        {command}
        """
    )

    exit_status = system(command)
    assert exit_status == 0


@pytest.mark.parametrize(
    "test_case",
    [
        pytest.param(
            "PROD3B_CTA_NORTH",
            marks=pytest.mark.dependency(name="EN_2", depends=["g1N"]),
        ),
        pytest.param(
            "PROD3B_CTA_SOUTH",
            marks=pytest.mark.dependency(name="ES_2", depends=["g1S"]),
        ),
    ],
)
def test_BUILD_ENERGY_MODEL_RandomForestRegressor(test_case, pipeline_testdir):
    """Launch protopipe.scripts.build_model for a RandomForestRegressor."""

    infile = pipeline_testdir / f"test_gamma1_noImages_{test_case}.h5"
    outdir = pipeline_testdir / f"energy_model_{test_case}"

    command = f"python {build_model.__file__}\
    --config_file {config_RandomForestRegressor}\
    --infile_signal {infile}\
    --outdir {outdir}\
    --cameras_from_file"

    print(  # only with "pytest -s"
        f"""
        You can reproduce this test by running the following command,

        {command}
        """
    )

    exit_status = system(command)
    assert exit_status == 0


@pytest.mark.parametrize(
    "test_case",
    [
        pytest.param(
            "PROD3B_CTA_NORTH",
            marks=pytest.mark.dependency(name="g2N", depends=["EN_2"]),
        ),
        pytest.param(
            "PROD3B_CTA_SOUTH",
            marks=pytest.mark.dependency(name="g2S", depends=["ES_2"]),
        ),
    ],
)
def test_GET_GAMMAS_FOR_CLASSIFICATION_MODEL(test_case, pipeline_testdir):

    modelpath = pipeline_testdir / f"energy_model_{test_case}"
    outpath = pipeline_testdir / f"test_gamma2_noImages_{test_case}.h5"

    command = f"python {data_training.__file__}\
    --config_file {input_data[test_case]['config']}\
    -o {outpath}\
    -i {input_data[test_case]['gamma2'].parent}\
    -f {input_data[test_case]['gamma2'].name}\
    --estimate_energy True\
    --regressor_config {config_RandomForestRegressor}\
    --regressor_dir {modelpath}"

    print(  # only with "pytest -s"
        f"""
        You can reproduce this test by running the following command,

        {command}
        """
    )

    exit_status = system(command)

    # check that the script ends without crashing
    assert exit_status == 0

    # check that the produced HDF5 file is non-empty
    with tables.open_file(outpath) as file:
        assert file.get_filesize() > 0


@pytest.mark.parametrize(
    "test_case",
    [
        pytest.param(
            "PROD3B_CTA_NORTH",
            marks=pytest.mark.dependency(name="p1N", depends=["EN_2"]),
        ),
        pytest.param(
            "PROD3B_CTA_SOUTH",
            marks=pytest.mark.dependency(name="p1S", depends=["ES_2"]),
        ),
    ],
)
def test_GET_PROTONS_FOR_CLASSIFICATION_MODEL(test_case, pipeline_testdir):

    modelpath = pipeline_testdir / f"energy_model_{test_case}"
    outpath = pipeline_testdir / f"test_proton1_noImages_{test_case}.h5"

    command = f"python {data_training.__file__}\
    --config_file {input_data[test_case]['config']}\
    -o {outpath}\
    -i {input_data[test_case]['proton1'].parent}\
    -f {input_data[test_case]['proton1'].name}\
    --estimate_energy True\
    --regressor_config {config_RandomForestRegressor}\
    --regressor_dir {modelpath}"

    print(  # only with "pytest -s"
        f"""
        You can reproduce this test by running the following command,

        {command}
        """
    )

    exit_status = system(command)

    # check that the script ends without crashing
    assert exit_status == 0

    # check that the produced HDF5 file is non-empty
    with tables.open_file(outpath) as file:
        assert file.get_filesize() > 0


@pytest.mark.parametrize(
    "test_case",
    [
        pytest.param(
            "PROD3B_CTA_NORTH",
            marks=pytest.mark.dependency(name="C1", depends=["g2N", "p1N"]),
        ),
        pytest.param(
            "PROD3B_CTA_SOUTH",
            marks=pytest.mark.dependency(name="C2", depends=["g2S", "p1S"]),
        ),
    ],
)
def test_BUILD_CLASSIFICATION_MODEL_RandomForestClassifier(test_case, pipeline_testdir):
    """Launch protopipe.scripts.build_model for a Random Forest classifier."""

    infile_signal = pipeline_testdir / f"test_gamma2_noImages_{test_case}.h5"
    infile_background = pipeline_testdir / f"test_proton1_noImages_{test_case}.h5"
    outdir = pipeline_testdir / f"classification_model_{test_case}"

    command = f"python {build_model.__file__}\
    --config_file {config_RandomForestClassifier}\
    --infile_signal {infile_signal}\
    --infile_background {infile_background}\
    --outdir {outdir}\
    --cameras_from_file"

    print(  # only with "pytest -s"
        f"""
        You can reproduce this test by running the following command,

        {command}
        """
    )

    exit_status = system(command)
    assert exit_status == 0


@pytest.mark.parametrize(
    "test_case",
    [
        pytest.param(
            "PROD3B_CTA_NORTH", marks=pytest.mark.dependency(name="g3N", depends=["C1"])
        ),
        pytest.param(
            "PROD3B_CTA_SOUTH", marks=pytest.mark.dependency(name="g3S", depends=["C2"])
        ),
    ],
)
def test_GET_DL2_GAMMAS(test_case, pipeline_testdir):

    regressor_path = pipeline_testdir / f"energy_model_{test_case}"
    classifier_path = pipeline_testdir / f"classification_model_{test_case}"
    outpath = pipeline_testdir / f"test_DL2_tail_gamma_noImages_{test_case}.h5"

    command = f"python {write_dl2.__file__}\
    --config_file {input_data[test_case]['config']}\
    -o {outpath}\
    -i {input_data[test_case]['gamma3'].parent}\
    -f {input_data[test_case]['gamma3'].name}\
    --regressor_config {config_RandomForestRegressor}\
    --regressor_dir {regressor_path}\
    --classifier_config {config_RandomForestClassifier}\
    --classifier_dir {classifier_path}"

    print(  # only with "pytest -s"
        f"""
        You can reproduce this test by running the following command,

        {command}
        """
    )

    exit_status = system(command)

    # check that the script ends without crashing
    assert exit_status == 0

    # sanity checks on the produced HDF5 file
    with tables.open_file(outpath) as file:
        assert file.get_filesize() > 0
        assert file.root._v_attrs["status"] == "complete"


@pytest.mark.parametrize(
    "test_case",
    [
        pytest.param(
            "PROD3B_CTA_NORTH", marks=pytest.mark.dependency(name="p2N", depends=["C1"])
        ),
        pytest.param(
            "PROD3B_CTA_SOUTH", marks=pytest.mark.dependency(name="p2S", depends=["C2"])
        ),
    ],
)
def test_GET_DL2_PROTONS(test_case, pipeline_testdir):

    regressor_path = pipeline_testdir / f"energy_model_{test_case}"
    classifier_path = pipeline_testdir / f"classification_model_{test_case}"
    outpath = pipeline_testdir / f"test_DL2_tail_proton_noImages_{test_case}.h5"

    command = f"python {write_dl2.__file__}\
    --config_file {input_data[test_case]['config']}\
    -o {outpath}\
    -i {input_data[test_case]['proton2'].parent}\
    -f {input_data[test_case]['proton2'].name}\
    --regressor_config {config_RandomForestRegressor}\
    --regressor_dir {regressor_path}\
    --classifier_config {config_RandomForestClassifier}\
    --classifier_dir {classifier_path}"

    print(  # only with "pytest -s"
        f"""
        You can reproduce this test by running the following command,

        {command}
        """
    )

    exit_status = system(command)

    # check that the script ends without crashing
    assert exit_status == 0

    # sanity checks on the produced HDF5 file
    with tables.open_file(outpath) as file:
        assert file.get_filesize() > 0
        assert file.root._v_attrs["status"] == "complete"


@pytest.mark.parametrize(
    "test_case",
    [
        pytest.param(
            "PROD3B_CTA_NORTH", marks=pytest.mark.dependency(name="elN", depends=["C1"])
        ),
        pytest.param(
            "PROD3B_CTA_SOUTH", marks=pytest.mark.dependency(name="elS", depends=["C2"])
        ),
    ],
)
def test_GET_DL2_ELECTRONS(test_case, pipeline_testdir):

    regressor_path = pipeline_testdir / f"energy_model_{test_case}"
    classifier_path = pipeline_testdir / f"classification_model_{test_case}"
    outpath = pipeline_testdir / f"test_DL2_tail_electron_noImages_{test_case}.h5"

    command = f"python {write_dl2.__file__}\
    --config_file {input_data[test_case]['config']}\
    -o {outpath}\
    -i {input_data[test_case]['electron1'].parent}\
    -f {input_data[test_case]['electron1'].name}\
    --regressor_config {config_RandomForestRegressor}\
    --regressor_dir {regressor_path}\
    --classifier_config {config_RandomForestClassifier}\
    --classifier_dir {classifier_path}"

    print(  # only with "pytest -s"
        f"""
        You can reproduce this test by running the following command,

        {command}
        """
    )

    exit_status = system(command)

    # check that the script ends without crashing
    assert exit_status == 0

    # sanity checks on the produced HDF5 file
    with tables.open_file(outpath) as file:
        assert file.get_filesize() > 0
        assert file.root._v_attrs["status"] == "complete"


@pytest.mark.parametrize(
    "test_case",
    [
        pytest.param(
            "PROD3B_CTA_NORTH",
            marks=pytest.mark.dependency(name="DL3N", depends=["g3N", "p2N", "elN"]),
        ),
        pytest.param(
            "PROD3B_CTA_SOUTH",
            marks=pytest.mark.dependency(name="DL3S", depends=["g3S", "p2S", "elS"]),
        ),
    ],
)
def test_GET_DL3_ED_prod3b(test_case, pipeline_testdir):

    template_input_file = f"test_DL2_{{}}_{{}}_noImages_{test_case}.h5"

    command = f"python {make_performance_EventDisplay.__file__}\
    --config_file {config_DL3_ED_prod3b}\
    --indir {pipeline_testdir}\
    --outdir_path {pipeline_testdir}\
    --out_file_name 'test_DL3_{test_case}'\
    --template_input_file {template_input_file}"

    print(  # only with "pytest -s"
        f"""
        You can reproduce this test by running the following command,

        {command}
        """
    )

    exit_status = system(command)

    # check that the script ends without crashing
    assert exit_status == 0

    # check that the output file exists and it is not empty
    path = Path(pipeline_testdir) / f"test_DL3_{test_case}.fits.gz"
    assert path.exists() and (path.stat().st_size > 0)

    from astropy.io import fits

    with fits.open(path) as hdul:
        assert len(hdul) == 19  # check that all HDUs are there
        for hdu in hdul[1:]:
            assert hdu.size > 0  # check presence of data
