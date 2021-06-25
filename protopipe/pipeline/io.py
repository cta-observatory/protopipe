"""Utility functions mainly used in benchmarking notebooks."""

import tables
import pandas
from astropy.table import Table


def get_camera_names(input_directory=None,
                     file_name=None):
    """Read the names of the cameras.

    Parameters
    ==========
    input_directory : pathlib.Path
        Full path of the input file.
    file_name : str
        Name of the input file.

    Returns
    =======
    camera_names : list(str)
        Table names as a list.
    """
    if (input_directory is None) or (file_name is None):
        print("ERROR: check input")

    input_file = input_directory / file_name

    with tables.open_file(input_file, 'r') as f:
        camera_names = [cam.name for cam in f.root]

    return camera_names


def read_protopipe_TRAINING_per_tel_type(input_directory=None,
                                         file_name=None,
                                         camera_names=None):
    """Read a TRAINING file and extract the data per telescope type.

    Parameters
    ==========
    input_directory : pathlib.Path
        Full path of the input directory where the TRAINING file is located.
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
    input_file = input_directory / file_name
    dataFrames = {}
    for camera in camera_names:
        dataFrames[camera] = pandas.read_hdf(input_file, f"/{camera}")
    return dataFrames


def read_TRAINING_per_tel_type_with_images(input_directory=None,
                                           input_filename=None,
                                           camera_names=None):
    """Read a TRAINING file and extract the data per telescope type.

    Parameters
    ==========
    input_directory : pathlib.Path
        Full path of the input directory where the TRAINING file is located.
    file_name : str
        Name of the input TRAINING file.
    camera_names: list
        List of camera names corresponding to the table names in the file.

    Returns
    =======
    table : dict(astropy.Table)
        Dictionary of astropy tables per camera.
    """
    input_file = input_directory / input_filename

    table = {}

    with tables.open_file(input_file, mode='r') as h5file:
        for camera in camera_names:
            table[camera] = Table()
            for key in h5file.get_node(f"/{camera}").colnames:
                table[camera][key] = h5file.get_node(f"/{camera}").col(key)

    return table
