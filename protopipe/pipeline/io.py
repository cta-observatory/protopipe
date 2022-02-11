"""Utility functions mainly used in benchmarking notebooks."""

from pathlib import Path
import yaml
import pickle
import gzip

from astropy.table import Table
import tables
import pandas
import joblib


def load_config(name):
    """Load a YAML configuration file.

    Parameters
    ----------
    name: str or pathlib.Path

    Returns
    -------
    cfg: object
        Python object (usually a dictionary).
    """
    try:
        with open(name, "r") as stream:
            cfg = yaml.load(stream, Loader=yaml.FullLoader)
    except FileNotFoundError as e:
        raise e

    return cfg


def get_camera_names(input_directory=None, file_name=None):
    """Read the names of the cameras.

    Parameters
    ==========
    input_directory : str or pathlib.Path
        Path of the input file.
    file_name : str
        Name of the input file.

    Returns
    =======
    camera_names : list(str)
        Table names as a list.
    """
    if (input_directory is None) or (file_name is None):
        print("ERROR: check input")

    input_file = Path(input_directory) / file_name

    with tables.open_file(input_file, "r") as f:
        camera_names = [cam.name for cam in f.root]

    return camera_names


def read_protopipe_TRAINING_per_tel_type(
    input_directory=None, file_name=None, camera_names=None
):
    """Read a TRAINING file and extract the data per telescope type.

    Parameters
    ==========
    input_directory : str or pathlib.Path
        Path of the input directory where the TRAINING file is located.
    file_name : str
        Name of the input TRAINING file.

    Returns
    =======
    dataFrames : dict(pandas.DataFrame)
        Dictionary of tables per camera.
    """
    if (input_directory is None) or (file_name is None):
        print("ERROR: check input")
    if camera_names is None:
        print("ERROR: no cameras specified")
    # load DL1 images
    input_file = Path(input_directory) / file_name
    dataFrames = {}
    for camera in camera_names:
        dataFrames[camera] = pandas.read_hdf(input_file, f"/{camera}")
    return dataFrames


def read_TRAINING_per_tel_type_with_images(
    input_directory=None, input_filename=None, camera_names=None
):
    """Read a TRAINING file and extract the data per telescope type.

    Parameters
    ==========
    input_directory : str or pathlib.Path
        Path of the input directory where the TRAINING file is located.
    file_name : str
        Name of the input TRAINING file.
    camera_names: list
        List of camera names corresponding to the table names in the file.

    Returns
    =======
    table : dict(astropy.Table)
        Dictionary of astropy tables per camera.
    """
    input_file = Path(input_directory) / input_filename

    table = {}

    with tables.open_file(input_file, mode="r") as h5file:
        for camera in camera_names:
            table[camera] = Table()
            for key in h5file.get_node(f"/{camera}").colnames:
                table[camera][key] = h5file.get_node(f"/{camera}").col(key)

    return table


def load_models(path, cam_id_list):
    """Load the pickled dictionary of model from disk
    and fill the model dictionary.

    Parameters
    ----------
    path : string
        The path where the pre-trained, pickled models are
        stored. `path` is assumed to contain a `{cam_id}` keyword
        to be replaced by each camera identifier in `cam_id_list`
        (or at least a naked `{}`).
    cam_id_list : list
        List of camera identifiers like telescope ID or camera ID
        and the assumed distinguishing feature in the filenames of
        the various pickled regressors.

    Returns
    -------
    model_dict: dict
        Dictionary with `cam_id` as keys and pickled models as values.
    """

    model_dict = {}
    for key in cam_id_list:
        try:
            model_dict[key] = joblib.load(path.format(cam_id=key))
        except IndexError:
            model_dict[key] = joblib.load(path.format(key))

    return model_dict


def load_obj(name):
    """Load object in binary"""
    with gzip.open(name, "rb") as f:
        return pickle.load(f)


def save_obj(obj, name):
    """Save object in binary"""
    with gzip.open(name, "wb") as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
