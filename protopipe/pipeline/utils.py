import yaml
import argparse

import matplotlib.pyplot as plt
import os.path as path

from ctapipe.io import event_source


def save_fig(outdir, name, fig=None):
    """Save a figure in multiple formats."""
    for ext in ["pdf", "png"]:
        if fig:
            fig.savefig(path.join(outdir, f"{name}.{ext}"))
        else:
            plt.savefig(path.join(outdir, f"{name}.{ext}"))
    return None


def load_config(name):
    """Load YAML configuration file."""
    try:
        with open(name, "r") as stream:
            cfg = yaml.load(stream, Loader=yaml.FullLoader)
    except FileNotFoundError as e:
        print(e)
        raise
    return cfg


class SignalHandler:
    """ handles ctrl+c signals; set up via
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
        "-i", "--indir", type=str, default=expandvars("$CTA_DATA/Prod3b/Paranal")
    )
    parser.add_argument(
        "-f",
        "--infile_list",
        type=str,
        default="",
        nargs="*",
        help="give a specific list of files to run on",
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


def from_list(sim_array, array):
    """Return tel IDs and involved cameras from a custom list of tel IDs.

    Helper function for utils.check_input.
    """
    subarray = sim_array.select_subarray("", array)
    tel_ids = subarray.tel_ids
    cameras = [cam.cam_id for cam in subarray.camera_types]
    return set(tel_ids), cameras


def prod3b_array(fileName, site, array):
    """Return tel IDs and involved cameras from configuration and simtel file.

    The initial check (and the too-high cyclomatic complexity) will disappear
    with the advent of the final array layouts.
    Currently not very performant: it is necessary get at least the first event
    of the simtel file to read the simulated array information.

    Parameters
    ----------
    first_fileName : str
        Name of the first file of the list of files given by the user.
    array : str or list
        Name of the subarray or - if not supported - a custom list of telescope
        IDs that the user wants to use
    site : str
        Can be only "north" or "south".
        Currently relevant only for baseline simulations.
        For non-baseline simulations only custom lists of IDs matter.

    Returns
    -------
    tel_ids : list
        List of telescope IDs that the user wants to use
    cameras : list
        List of camera types inferred from tel_ids.
        This will feed both the estimators and the image cleaning.

    """
    source = event_source(input_url=fileName, max_events=1)

    for event in source:  # get only first event
        pass

    sim_array = event.inst.subarray  # get simulated array

    # Dictionaries of subarray names for baseline simulations
    subarrays_N = {  # La Palma has only 2 cameras
        "subarray_LSTs": sim_array.get_tel_ids_for_type("LST_LST_LSTCam"),
        "subarray_MSTs": sim_array.get_tel_ids_for_type("MST_MST_NectarCam"),
        "full_array": sim_array.tel_ids,
    }
    subarrays_S = {  # Paranal has only 3 cameras
        "subarray_LSTs": sim_array.get_tel_ids_for_type("LST_LST_LSTCam"),
        "subarray_MSTs": sim_array.get_tel_ids_for_type("MST_MST_FlashCam"),
        "subarray_SSTs": sim_array.get_tel_ids_for_type("SST_GCT_CHEC"),
        "full_array": sim_array.tel_ids,
    }

    if site.lower() == "north":
        if sim_array.num_tels > 19:  # this means non-baseline simulation..
            if sim_array.num_tels > 125 or sim_array.num_tels == 99:
                raise ValueError("ERROR: infile and site uncorrelated!")
            if type(array) == str:
                raise ValueError(
                    "Subarray names not supported for this production.\
                     Please, define a costum array."
                )
            elif (
                type(array) == list
            ):  # ..for which only custom lists are currently supported
                return from_list(sim_array, array)
            else:
                raise ValueError(f"array {array} not supported")
        else:  # this is a baseline simulation
            if type(array) == str:
                if array not in subarrays_N.keys():
                    raise ValueError(
                        "ERROR: requested missing camera from simtel file."
                    )
                tel_ids = subarrays_N[array]
                subarray = sim_array.select_subarray("", tel_ids)
                cameras = [cam.cam_id for cam in subarray.camera_types]
                return set(tel_ids), cameras
            elif type(array) == list:
                if any((tel_id < 1 or tel_id > 19) for tel_id in array):
                    raise ValueError("ERROR: non-existent telescope ID.")
                return from_list(sim_array, array)
            else:
                raise ValueError(f"array {array} not supported")
    elif site.lower() == "south":
        if sim_array.num_tels > 99:  # this means non-baseline simulation..
            if sim_array.num_tels < 126:
                raise ValueError("ERROR: infile and site uncorrelated!")
            if type(array) == str:
                raise ValueError(
                    "Subarray names not supported for this production.\
                     Please, define a costum array."
                )
            elif (
                type(array) == list
            ):  # ..for which only custom lists are currently supported
                return from_list(sim_array, array)
            else:
                raise ValueError(f"array {array} not supported")
        else:  # this is a baseline simulation
            if sim_array.num_tels == 19:
                raise ValueError("ERROR: infile and site uncorrelated!")
            if type(array) == str:
                if array not in subarrays_S.keys():
                    raise ValueError(
                        "ERROR: requested missing camera from simtel file."
                    )
                tel_ids = subarrays_S[array]
                subarray = sim_array.select_subarray("", tel_ids)
                cameras = [cam.cam_id for cam in subarray.camera_types]
                return set(tel_ids), cameras
            elif type(array) == list:
                if any((tel_id < 1 or tel_id > 99) for tel_id in array):
                    raise ValueError("ERROR: non-existent telescope ID.")
                return from_list(sim_array, array)
            else:
                raise ValueError(f"array {array} not supported")
