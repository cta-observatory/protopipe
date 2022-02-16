"""Temporary development code in-between pyirf releases."""

import numpy as np
from astropy.table import Table
import astropy.units as u
from scipy.stats import norm

from pyirf.binning import calculate_bin_indices
from pyirf.benchmarks.energy_bias_resolution import inter_quantile_distance


def energy_bias_resolution(
    events,
    energy_bins,
    energy_type="true",
    bias_function=np.median,
    resolution_function=inter_quantile_distance,
):
    """
    Calculate bias and energy resolution.

    Parameters
    ----------
    events: astropy.table.QTable
        Astropy Table object containing the reconstructed events information.
    energy_bins: numpy.ndarray(dtype=float, ndim=1)
        Bin edges in energy.
    energy_type: str
        Either "true" or "reco" energy.
        Default is "true".
    bias_function: callable
        Function used to calculate the energy bias
    resolution_function: callable
        Function used to calculate the energy resolution

    Returns
    -------
    result : astropy.table.Table
        Table containing the energy bias and resolution
        per each bin in true energy.
    """

    # create a table to make use of groupby operations
    table = Table(events[["true_energy", "reco_energy"]])
    table["rel_error"] = (events["reco_energy"] / events["true_energy"]) - 1

    table["bin_index"] = calculate_bin_indices(
        table[f"{energy_type}_energy"].quantity, energy_bins
    )
    n_bins = len(energy_bins) - 1
    mask = (table["bin_index"] >= 0) & (table["bin_index"] < n_bins)

    result = Table()
    result[f"{energy_type}_energy_low"] = energy_bins[:-1]
    result[f"{energy_type}_energy_high"] = energy_bins[1:]
    result[f"{energy_type}_energy_center"] = 0.5 * (energy_bins[:-1] + energy_bins[1:])

    result["bias"] = np.nan
    result["resolution"] = np.nan

    if not len(events):
        # if we get an empty input (no selected events available)
        # we return the table filled with NaNs
        return result

    # use groupby operations to calculate the percentile in each bin
    by_bin = table[mask].group_by("bin_index")

    index = by_bin.groups.keys["bin_index"]
    result["bias"][index] = by_bin["rel_error"].groups.aggregate(bias_function)
    result["resolution"][index] = by_bin["rel_error"].groups.aggregate(
        resolution_function
    )
    return result


def angular_resolution(events, energy_bins, energy_type="true"):
    """
    Calculate the angular resolution.

    This implementation corresponds to the 68% containment of the angular
    distance distribution.

    Parameters
    ----------
    events : astropy.table.QTable
        Astropy Table object containing the reconstructed events information.
    energy_bins: numpy.ndarray(dtype=float, ndim=1)
        Bin edges in energy.
    energy_type: str
        Either "true" or "reco" energy.
        Default is "true".

    Returns
    -------
    result : astropy.table.Table
        Table containing the 68% containment of the angular distance
        distribution per each reconstructed energy bin.
    """

    ONE_SIGMA_QUANTILE = norm.cdf(1) - norm.cdf(-1)

    # create a table to make use of groupby operations
    table = Table(events[[f"{energy_type}_energy", "theta"]])

    table["bin_index"] = calculate_bin_indices(
        table[f"{energy_type}_energy"].quantity, energy_bins
    )

    n_bins = len(energy_bins) - 1
    mask = (table["bin_index"] >= 0) & (table["bin_index"] < n_bins)

    result = Table()
    result[f"{energy_type}_energy_low"] = energy_bins[:-1]
    result[f"{energy_type}_energy_high"] = energy_bins[1:]
    result[f"{energy_type}_energy_center"] = 0.5 * (energy_bins[:-1] + energy_bins[1:])

    result["angular_resolution"] = np.nan * u.deg

    # use groupby operations to calculate the percentile in each bin
    by_bin = table[mask].group_by("bin_index")

    index = by_bin.groups.keys["bin_index"]
    result["angular_resolution"][index] = by_bin["theta"].groups.aggregate(
        lambda x: np.quantile(x, ONE_SIGMA_QUANTILE)
    )
    return result
