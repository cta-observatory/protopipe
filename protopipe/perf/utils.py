import argparse
import logging

import numpy as np
import pandas as pd
import pickle
import gzip

from astropy.table import QTable
import astropy.units as u

from pyirf.simulations import SimulatedEventsInfo


def initialize_script_arguments():
    """Initialize the parser of any protopipe.scripts.make_performance_XXX script.

    Returns
    -------
    args : argparse.Namespace
        Populated argparse namespace.
    """

    parser = argparse.ArgumentParser(description="Make performance files")
    parser.add_argument("--config_file", type=str, required=True, help="")

    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--wave",
        dest="mode",
        action="store_const",
        const="wave",
        default="tail",
        help="if set, use wavelet cleaning",
    )
    mode_group.add_argument(
        "--tail",
        dest="mode",
        action="store_const",
        const="tail",
        help="if set, use tail cleaning (default)",
    )
    parser.add_argument(
        "-i",
        "--indir_parent",
        type=str,
        default=None,
        help="Directory containing the required DL2 input files (default: read from config file)",
    )
    parser.add_argument(
        "--indir_gamma",
        type=str,
        default=None,
        help="Sub-directory containing the required gamma DL2 input files (default: read from config file)",
    )
    parser.add_argument(
        "--indir_proton",
        type=str,
        default=None,
        help="Sub-directory containing the required proton DL2 input files (default: read from config file)",
    )
    parser.add_argument(
        "--indir_electron",
        type=str,
        default=None,
        help="Sub-directory containing the required electron DL2 input files (default: read from config file)",
    )
    parser.add_argument(
        "--template_input_file",
        type=str,
        default=None,
        help="template for the filename of DL2 files (default: read from config file)",
    )
    parser.add_argument(
        "--outdir_path",
        type=str,
        default=None,
        help="Output directory for DL3 file (default: read from config file)",
    )
    parser.add_argument(
        "--out_file_name",
        type=str,
        default=None,
        help="Desired name for DL3 file (default: built from config file)",
    )

    args = parser.parse_args()

    return args


def percentiles(values, bin_values, bin_edges, percentile):
    # Seems complicated for vector defined as [inf, inf, .., inf]
    percentiles_binned = np.squeeze(
        np.full((len(bin_edges) - 1, len(values.shape)), np.inf)
    )
    err_percentiles_binned = np.squeeze(
        np.full((len(bin_edges) - 1, len(values.shape)), np.inf)
    )
    for i, (bin_l, bin_h) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
        try:
            print(i)
            print(bin_l)
            print(bin_h)
            distribution = values[(bin_values > bin_l) & (bin_values < bin_h)]
            percentiles_binned[i] = np.percentile(distribution, percentile)
            print(percentiles_binned[i])
            err_percentiles_binned[i] = percentiles_binned[i] / np.sqrt(
                len(distribution)
            )
        except IndexError:
            pass
    return percentiles_binned.T, err_percentiles_binned.T


def plot_hist(ax, data, edges, norm=False, yerr=False, hist_kwargs={}, error_kw={}):
    """Utility function to plot histogram"""
    weights = np.ones_like(data)
    if norm is True:
        weights = weights / float(np.sum(data))
    if yerr is True:
        yerr = np.sqrt(data) * weights
    else:
        yerr = np.zeros(len(data))

    centers = 0.5 * (edges[1:] + edges[:-1])
    width = edges[1:] - edges[:-1]
    ax.bar(
        centers,
        data * weights,
        width=width,
        yerr=yerr,
        error_kw=error_kw,
        **hist_kwargs,
    )

    return ax


def save_obj(obj, name):
    """Save object in binary"""
    with gzip.open(name, "wb") as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    """Load object in binary"""
    with gzip.open(name, "rb") as f:
        return pickle.load(f)


def read_DL2_pyirf(infile, run_header):
    """
    Read a DL2 HDF5 protopipe file and adapt them to pyirf format.

    Parameters
    ----------
    infile: str or pathlib.Path
        Path to the input fits file
    run_header: dict
        Dictionary with info about simulated particle informations

    Returns
    -------
    events: astropy.QTable
        Astropy Table object containing the reconstructed events information.
    simulated_events: ``~pyirf.simulations.SimulatedEventsInfo``
    """
    log = logging.getLogger("pyirf")
    log.debug(f"Reading {infile}")
    df = pd.read_hdf(infile, "/reco_events")

    events = QTable(
        [
            list(df["obs_id"]),
            list(df["event_id"]),
            list(df["true_energy"]) * u.TeV,
            list(df["reco_energy"]) * u.TeV,
            list(df["gammaness"]),
            list(df["NTels_reco"]),
            list(df["reco_alt"]) * u.deg,
            list(df["reco_az"]) * u.deg,
            list(df["true_alt"]) * u.deg,
            list(df["true_az"]) * u.deg,
            list(df["pointing_alt"]) * u.deg,
            list(df["pointing_az"]) * u.deg,
            list(df["success"]),
        ],
        names=(
            "obs_id",
            "event_id",
            "true_energy",
            "reco_energy",
            "gh_score",
            "multiplicity",
            "reco_alt",
            "reco_az",
            "true_alt",
            "true_az",
            "pointing_alt",
            "pointing_az",
            "success",
        ),
    )

    # Select only DL2 events marked as fully reconstructed
    mask = events["success"]
    events = events[mask]

    n_runs = len(set(events["obs_id"]))
    log.info(f"Estimated number of runs from obs ids: {n_runs}")

    n_showers = n_runs * run_header["num_use"] * run_header["num_showers"]
    log.debug(f"Number of events from n_runs and run header: {n_showers}")

    sim_info = SimulatedEventsInfo(
        n_showers=n_showers,
        energy_min=u.Quantity(run_header["e_min"], u.TeV),
        energy_max=u.Quantity(run_header["e_max"], u.TeV),
        max_impact=u.Quantity(run_header["gen_radius"], u.m),
        spectral_index=run_header["gen_gamma"],
        viewcone=u.Quantity(run_header["diff_cone"], u.deg),
    )
    return events, sim_info
