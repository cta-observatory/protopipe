import argparse
import math
from typing import List, Union

import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
import os.path as path

from ctapipe.io import EventSource
from ctapipe.instrument import TelescopeDescription
from ctapipe.coordinates import TelescopeFrame


class bcolors:
    """Color definitions for standard and debug printing."""

    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    BOLDWARNING = "\033[1m\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    BOLDGREEN = "\033[1m\033[92m"
    REVERSED = "\033[7m"
    PURPLE = "\033[95m"


def save_fig(outdir, name, fig=None):
    """Save a figure in multiple formats."""
    for ext in ["pdf", "png"]:
        if fig:
            fig.savefig(path.join(outdir, f"{name}.{ext}"))
        else:
            plt.savefig(path.join(outdir, f"{name}.{ext}"))
    return None


class SignalHandler:
    """handles ctrl+c signals; set up via
    `signal_handler = SignalHandler()
    `signal.signal(signal.SIGINT, signal_handler)`
    # or for two step interupt:
    `signal.signal(signal.SIGINT, signal_handler.stop_drawing)`
    """

    def __init__(self):
        self.stop = False
        self.draw = True

    def __call__(self, signal, frame):
        if self.stop:
            print("you pressed Ctrl+C again -- exiting NOW")
            exit(-1)
        print("you pressed Ctrl+C!")
        print("exiting after current event")
        self.stop = True

    def stop_drawing(self, signal, frame):
        if self.stop:
            print("you pressed Ctrl+C again -- exiting NOW")
            exit(-1)

        if self.draw:
            print("you pressed Ctrl+C!")
            print("turn off drawing")
            self.draw = False
        else:
            print("you pressed Ctrl+C!")
            print("exiting after current event")
            self.stop = True


def str2bool(v):
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def make_argparser():
    from os.path import expandvars

    parser = argparse.ArgumentParser(description="")

    # Add configuration file
    parser.add_argument("--config_file", type=str, required=True)
    parser.add_argument("-o", "--outfile", type=str, required=True)
    parser.add_argument(
        "-m",
        "--max_events",
        type=int,
        default=None,
        help="maximum number of events considered per file",
    )
    parser.add_argument(
        "-i",
        "--indir",
        type=str,
        required=True,
        help="Input folder",
    )
    parser.add_argument(
        "-f",
        "--infile_list",
        type=str,
        required=True,
        nargs="*",
        help="give a specific list of files to run on",
    )

    parser.add_argument(
        "--cam_ids",
        type=str,
        default=["LSTCam", "NectarCam"],
        nargs="*",
        help="give the specific list of camera types to run on",
    )

    parser.add_argument(
        "--wave_dir",
        type=str,
        default=None,
        help="directory where to find mr_filter. " "if not set look in $PATH",
    )
    parser.add_argument(
        "--wave_temp_dir",
        type=str,
        default="/dev/shm/",
        help="directory where mr_filter to store the temporary fits" " files",
    )

    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--wave",
        dest="mode",
        action="store_const",
        const="wave",
        default="tail",
        help="if set, use wavelet cleaning -- default",
    )
    mode_group.add_argument(
        "--tail",
        dest="mode",
        action="store_const",
        const="tail",
        help="if set, use tail cleaning, otherwise wavelets",
    )

    return parser


def final_array_to_use(
    original_array, subarray_selection: Union[str, List[int]], subarrays=None
):
    """Infer IDs of telescopes and cameras with equivalent focal lengths.

    This is an helper function for utils.prod3b_array.

    Parameters
    ----------
    subarrays : `dict` [`str`, `list` [`int`]], optional
        Dictionary of subarray names linked to lists of tel_ids
        automatically extracted by telescope type.
        If set, it will extract tel_ids from there, otherwise from the custom
        list given by 'array'.
    original_array : `ctapipe.instrument.SubarrayDescription`
        Full simulated array from the first event.
    subarray_selection : {`str`, `list` [`int`]}
        Custom list of telescope IDs that the user wants to use or name of
        specific subarray.

    Returns
    -------
    tel_ids : `list` [`int`]
        List of telescope IDs to use in a format readable by ctapipe.
    cams_and_foclens : `dict` [`str`, `float`]
        Dictionary containing the IDs of the involved cameras as inferred from
        the involved telescopes IDs, together with the equivalent focal lengths
        of the telescopes.
        The camera IDs will feed both the estimators and the image cleaning.
        The equivalent focal lengths will be used to calculate the radius of
        the camera on which cut for truncated images.
    subarray : ctapipe.instrument.SubarrayDescription
        Complete subarray information of the final array/subarray selected.

    """
    if subarrays:
        tel_ids = subarrays[subarray_selection]
        selected_subarray = original_array.select_subarray(
            tel_ids, name="selected_subarray"
        )
    else:
        selected_subarray = original_array.select_subarray(
            subarray_selection, name="selected_subarray"
        )
        tel_ids = selected_subarray.tel_ids
    tel_types = selected_subarray.telescope_types
    cams_and_foclens = {
        tel_types[i]
        .camera.camera_name: tel_types[i]
        .optics.equivalent_focal_length.value
        for i in range(len(tel_types))
    }
    return set(tel_ids), cams_and_foclens, selected_subarray


def prod5N_array(file_name, site: str, subarray_selection: Union[str, List[int]]):
    """Return tel IDs and involved cameras from configuration and simtel file.

    Alpha configurations refer to the subarrays selected in [1]_.

    Parameters
    ----------
    file_name : `str`
        Name of the first file of the list of files given by the user.
    subarray_selection : {`str`, `list` [`int`]}
        Name or list if telescope IDs which identifies the subarray to extract.
    site : `str`
        Can be only "north" or "south".

    Returns
    -------
    tel_ids : `list` [`int`]
        List of telescope IDs to use in a format readable by ctapipe.
    cameras : `list` [`str`]
        List of camera types inferred from tel_ids.
        This will feed both the estimators and the image cleaning.

    References
    ----------
    .. [1] Cherenkov Telescope Array Observatory, & Cherenkov Telescope Array
       Consortium. (2021). CTAO Instrument Response Functions - prod5 version
       v0.1 (v0.1) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.5499840

    """

    with EventSource(input_url=file_name, max_events=1) as source:
        original_array = source.subarray

    tot_num_tels = original_array.num_tels

    # prod5N_alpha_north
    # CTA North Advanced Threshold Layout D25
    # Prod5 North
    # 4 LSTs and 9 MSTs (NectarCam type)

    # prod5N_alpha_south
    # CTA South
    # Prod5 layout S-M6C5-14MSTs37SSTs-MSTF
    # 0 LSTs and 14 MSTs (FlashCam type), 37 SSTs

    site_to_subarray_name = {
        "north": ["prod5N_alpha_north"],
        "south": ["prod5N_alpha_south", "prod5N_alpha_south_NectarCam"],
    }

    name_to_tel_ids = {
        "full_array": original_array.tel_ids,
        "prod5N_alpha_north": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 19, 35],
        "prod5N_alpha_south": [
            5,
            6,
            8,
            10,
            11,
            12,
            13,
            14,
            126,
            16,
            125,
            20,
            24,
            26,
            30,
            33,
            34,
            35,
            36,
            37,
            38,
            39,
            40,
            41,
            42,
            43,
            44,
            45,
            46,
            47,
            48,
            49,
            50,
            51,
            52,
            53,
            133,
            59,
            61,
            66,
            67,
            68,
            69,
            70,
            71,
            72,
            73,
            143,
            144,
            145,
            146,
        ],
        "prod5N_alpha_south_NectarCam": [
            30,
            33,
            34,
            35,
            36,
            37,
            38,
            39,
            40,
            41,
            42,
            43,
            44,
            45,
            46,
            47,
            48,
            49,
            50,
            51,
            52,
            53,
            59,
            61,
            66,
            67,
            68,
            69,
            70,
            71,
            72,
            73,
            100,
            101,
            103,
            105,
            106,
            107,
            108,
            109,
            111,
            115,
            119,
            121,
            128,
            129,
            133,
            143,
            144,
            145,
            146,
        ],
    }

    # Add subarrays by telescope type for the full original array
    # and extract the same from the alpha congiguration subarrays
    for tel_type in original_array.telescope_types:
        name_to_tel_ids[f"full_array_{tel_type}"] = original_array.get_tel_ids_for_type(
            tel_type
        )
        name_to_tel_ids[f"prod5N_alpha_north_subarray_{tel_type}"] = set(
            name_to_tel_ids[f"full_array_{tel_type}"]
        ).intersection(name_to_tel_ids["prod5N_alpha_north"])
        name_to_tel_ids[f"prod5N_alpha_south_subarray_{tel_type}"] = set(
            name_to_tel_ids[f"full_array_{tel_type}"]
        ).intersection(name_to_tel_ids["prod5N_alpha_south"])

    # Check if the user is using the correct file for the site declared in
    # the configuration
    if site.lower() == "north" and (tot_num_tels > 84):
        raise ValueError("\033[91m Input simtel file and site are uncorrelated!\033[0m")
    if site.lower() == "south" and (tot_num_tels < 180):
        raise ValueError("\033[91m infile and site uncorrelated! \033[0m")

    # Validate the subarray selection in case it is a string
    if type(subarray_selection) is str:
        if subarray_selection not in name_to_tel_ids:
            raise ValueError(
                f"""\033[91m {subarray_selection} is not a supported subarray_selection for this production.
                Possible choices are:
                 - use a custom list of IDs,
                 - add a new subarray definition to protopipe.utils.prod5N_array.
                """
            )
        elif subarray_selection not in site_to_subarray_name[site]:
            raise ValueError(
                f"""\033[91m {subarray_selection} is not a supported subarray_selection for the {site} site.
                """
            )
        else:
            print(
                f"\033[94m Extracting telescope IDs for {subarray_selection}...\033[0m"
            )
            return final_array_to_use(
                original_array, subarray_selection, name_to_tel_ids
            )
    else:  # subarray_selection is a list of telescope IDs
        print(
            f"\033[94m Extracting telescope IDs list = {subarray_selection}...\033[0m"
        )
        return final_array_to_use(original_array, subarray_selection)

    return final_array_to_use(original_array, subarray_selection)


def prod3b_array(file_name, site, array):
    """Return tel IDs and involved cameras from configuration and simtel file.

    The initial check (and the too-high cyclomatic complexity) will disappear
    with the advent of the final array layouts.

    Parameters
    ----------
    file_name : `str`
        Name of the first file of the list of files given by the user.
    array : {`str`, `list` [`int`]}
        Name of the subarray or - if not supported - a custom list of telescope
        IDs that the user wants to use
    site : `str`
        Can be only "north" or "south".
        Currently relevant only for baseline simulations.
        For non-baseline simulations only custom lists of IDs matter.

    Returns
    -------
    tel_ids : `list` [`int`]
        List of telescope IDs to use in a format readable by ctapipe.
    cameras : `list` [`str`]
        List of camera types inferred from tel_ids.
        This will feed both the estimators and the image cleaning.

    """
    with EventSource(input_url=file_name, max_events=1) as source:
        original_array = source.subarray  # get simulated array

    # Dictionaries of subarray names for BASELINE simulations
    subarrays_N = {  # La Palma has only 2 cameras
        "subarray_LSTs": original_array.get_tel_ids_for_type("LST_LST_LSTCam"),
        "subarray_MSTs": original_array.get_tel_ids_for_type("MST_MST_NectarCam"),
        "full_array": original_array.tel_ids,
    }
    subarrays_S = {  # Paranal has only 3 cameras
        "subarray_LSTs": original_array.get_tel_ids_for_type("LST_LST_LSTCam"),
        "subarray_MSTs": original_array.get_tel_ids_for_type("MST_MST_FlashCam"),
        "subarray_SSTs": original_array.get_tel_ids_for_type("SST_GCT_CHEC"),
        "full_array": original_array.tel_ids,
    }

    if site.lower() == "north":
        if original_array.num_tels > 19:  # this means non-baseline simulation..
            if (
                original_array.num_tels > 125  # Paranal non-baseline
                or original_array.num_tels == 99  # Paranal baseline
                or original_array.num_tels == 98  # gamma_test_large
            ):
                raise ValueError(
                    "\033[91m ERROR: infile and site uncorrelated! \033[0m"
                )
            if (type(array) is str) and (array != "full_array"):
                raise ValueError(
                    "\033[91m ERROR: Only 'full_array' supported for this production.\n\
                     Please, use that or define a custom array with a list of tel_ids.  \033[0m"
                )
            elif array == "full_array":
                return final_array_to_use(original_array, array, subarrays_N)
            elif (
                type(array) == list
            ):  # ..for which only custom lists are currently supported
                return final_array_to_use(original_array, array)
            else:
                raise ValueError(
                    f"\033[91m ERROR: array {array} not supported. \033[0m"
                )
        else:  # this is a baseline simulation
            if type(array) == str:
                if array not in subarrays_N.keys():
                    raise ValueError(
                        "\033[91m ERROR: requested missing camera from simtel file. \033[0m"
                    )
                else:
                    return final_array_to_use(original_array, array, subarrays_N)
            elif type(array) == list:
                if any((tel_id < 1 or tel_id > 19) for tel_id in array):
                    raise ValueError(
                        "\033[91m ERROR: non-existent telescope ID. \033[0m"
                    )
                return final_array_to_use(original_array, array)
            else:
                raise ValueError(
                    f"\033[91m ERROR: array {array} not supported. \033[0m"
                )
    elif site.lower() == "south":
        if original_array.num_tels > 99:  # this means non-baseline simulation..
            if original_array.num_tels < 126:
                raise ValueError(
                    "\033[91m ERROR: infile and site uncorrelated! \033[0m"
                )
            if type(array) == str and array != "full_array":
                raise ValueError(
                    "\033[91m ERROR: Only 'full_array' supported for this production.\n\
                     Please, use that or define a custom array with a list of tel_ids. \033[0m"
                )
            if array == "full_array":
                return final_array_to_use(original_array, array, subarrays_S)
            elif (
                type(array) == list
            ):  # ..for which only custom lists are currently supported
                return final_array_to_use(original_array, array)
            else:
                raise ValueError(
                    f"\033[91m ERROR: Array {array} not supported. \033[0m"
                )
        else:  # this is a baseline simulation
            if original_array.num_tels == 19:
                raise ValueError(
                    "\033[91m ERROR: infile and site uncorrelated! \033[0m"
                )
            if type(array) == str:
                if array not in subarrays_S.keys():
                    raise ValueError(
                        "\033[91m ERROR: requested missing camera from simtel file. \033[0m"
                    )
                else:
                    if original_array.num_tels == 98:  # this is gamma_test_large
                        subarrays_S[
                            "subarray_SSTs"
                        ] = original_array.get_tel_ids_for_type(
                            "SST_ASTRI_ASTRICam"  # in this file SSTs are ASTRI
                        )
                    return final_array_to_use(original_array, array, subarrays_S)
            elif type(array) == list:
                if any((tel_id < 1 or tel_id > 99) for tel_id in array):
                    raise ValueError(
                        "\033[91m ERROR: non-existent telescope ID. \033[0m"
                    )
                return final_array_to_use(original_array, array)
            else:
                raise ValueError(
                    f"\033[91m ERROR: Array {array} not supported. \033[0m"
                )


def effective_focal_lengths(camera_name):
    """Provide the effective focal length for the required camera.

    Data comes from Konrad's table.

    Parameters
    ----------
    camera_name : str
        Name of the camera from ctapipe.instrument.CameraDescription

    Returns
    -------
    eff_foc_len : float
        Effective focal length in meters

    """

    effective_focal_lengths = {
        "LSTCam": 29.30565 * u.m,
        "NectarCam": 16.44505 * u.m,
        "FlashCam": 16.44505 * u.m,
        "ASTRICam": 2.15191 * u.m,
        "SCTCam": 5.58310 * u.m,
        "CHEC": 2.30913 * u.m,
        "DigiCam": 5.69705 * u.m,
    }

    eff_foc_len = effective_focal_lengths[camera_name]

    return eff_foc_len


def camera_radius(camid_to_efl, cam_id="all"):
    """Get camera radii.

    Inspired from pywi-cta CTAMarsCriteria, CTA Mars like preselection cuts.
    This should be replaced by a function in ctapipe getting the radius either
    from  the pixel poisitions or from an external database

    Notes
    -----
    average_camera_radius_meters = math.tan(math.radians(average_camera_radius_degree)) * foclen
    The average camera radius values are, in degrees :
    - LST: 2.31
    - Nectar: 4.05
    - Flash: 3.95
    - SST-1M: 4.56
    - GCT-CHEC-S: 3.93
    - ASTRI: 4.67

    """

    average_camera_radii_deg = {
        "ASTRICam": 4.67,
        "CHEC": 3.93,
        "DigiCam": 4.56,
        "FlashCam": 3.95,
        "NectarCam": 4.05,
        "LSTCam": 2.31,
        "SCTCam": 4.0,  # dummy value
    }

    if cam_id in camid_to_efl.keys():
        foclen_meters = camid_to_efl[cam_id]
        average_camera_radius_meters = (
            math.tan(math.radians(average_camera_radii_deg[cam_id])) * foclen_meters
        )
    elif cam_id == "all":
        print("Available camera radii in meters:")
        for cam_id in camid_to_efl.keys():
            print(f"* {cam_id} : {camera_radius(camid_to_efl, cam_id)}")
        average_camera_radius_meters = 0
    else:
        raise ValueError("Unknown camid", cam_id)

    return average_camera_radius_meters


def get_cameras_radii(subarray, frame=TelescopeFrame(), ctamars=False):
    """Get the radius of a camera using ctapipe.

    Parameters
    ----------
    camera_name: str
        Identifier for the camera.
    frame: astropy.coordinates.baseframe.BaseCoordinateFrame
        Coordinate frame in which to get the radius of the camera
    ctamars: bool
        If True return the hard-coded values from CTAMARS (default: False)

    Returns
    -------
    camera_radii: dict
        Dictionary with camera names as keys and their radius as value
    """
    if ctamars:
        average_camera_radii_deg = {
            "ASTRICam": 4.67,
            "CHEC": 3.93,
            "DigiCam": 4.56,
            "FlashCam": 3.95,
            "NectarCam": 4.05,
            "LSTCam": 2.31,
            "SCTCam": 4.0,
        }
        return average_camera_radii_deg

    camera_radii = {}
    for tel in subarray.telescope_types:
        cam_name = tel.camera.camera_name
        geom = tel.camera.geometry
        new_geom = geom.transform_to(frame)
        camera_radii[cam_name] = new_geom.guess_radius()
    return camera_radii
