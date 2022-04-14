"""
This module contains functions to perform varius operation over data.

The contents of the module can be used separately,
but they are also used used by the benchmarking notebooks.

Notes
-----
This module has been created essentially to clean-up/refactor the
notebooks.
The implementation of functions and classes is far from perfect and
we should really try to synchronize in some way with ctaplot/ctabenchmarks.
"""

__all__ = [
    "compute_weight_BTEL1010",
    "add_BTEL1010_weigths_to_data",
    "average_bias_of_charge_resolution",
    "calculate_RMS_around_1",
    "prepare_requirements",
    "compute_resolution",
    "compute_bias",
    "get_evt_subarray_model_output",
    "sum_of_squares",
    "OnlineBinnedStats",
    "create_lookup_function",
    "compute_psf",
    "load_tel_id",
]

from pathlib import Path

import astropy.units as u
import numpy as np
from astropy.table import Column, Table
from scipy.interpolate import RectBivariateSpline
from scipy.stats import binned_statistic

try:
    from ctapipe.io import read_table
except ImportError:
    from ctapipe.io.astropy_helpers import h5_table_to_astropy as read_table


def compute_weight_BTEL1010(true_energy, simtel_spectral_slope=-2.0):
    """Compute the weight from requirement B-TEL-1010-Intensity-Resolution.

    Parameters
    ----------
    true_energy: array_like
    simtel_spectral_slope: float
        Spectral slope from the simulation.
    """
    target_slope = -2.62  # spectral slope from B-TEL-1010
    spec_slope = simtel_spectral_slope
    # each pixel of the same image (row of data table) needs the same weight
    weight = np.power(true_energy / 200.0, target_slope - spec_slope)
    return weight


def add_BTEL1010_weigths_to_data(data, subarray=None):
    """Update data table with weights from requirement B-TEL-1010-Intensity-Resolution."""
    # and add B-TEL-1010 weights
    for tel_type in sorted(
        subarray.telescope_types, key=lambda t: -t.optics.equivalent_focal_length
    ):
        true_energies = data[str(tel_type)]["true_energy"].to(u.GeV)
        w = compute_weight_BTEL1010(true_energies)
        n_pixels = tel_type.camera.geometry.n_pixels
        weights = Column([np.repeat(w[i], n_pixels) for i in range(len(w))])
        # each pixel gets its weight
        data[str(tel_type)]["weights_B-TEL-1010"] = weights
    return data


def average_bias_of_charge_resolution(
    x_bin_edges, y_bin_edges, hist, min_phe=50, max_phe=500
):
    """Calculate the average bias of charge resolution.

    Default limits are in true photoelectrons and chosen to be safely
    away from saturation and NSB noise.

    Parameters
    ----------
    x_bin_edges : 1D array
        Bin edges in true photoelectrons.
    y_bin_edges : 1D array
        Bin edges in reconstructed/true photoelectrons.
    hist : 2D array
        The full histogram of reconstructed/true against true photoelectrons.

    Returns
    -------
    bias : float
        Average bias of charge resolution from 50 to 500 true photoelectrons.

    """
    min_edge_index = np.digitize(np.log10(min_phe), x_bin_edges) - 1
    max_edge_index = np.digitize(np.log10(max_phe), x_bin_edges)

    proj = np.zeros(600)
    for i in range(min_edge_index, max_edge_index + 1):
        proj = proj + hist[i]

    y_bin_centers = 0.5 * (y_bin_edges[1:] + y_bin_edges[:-1])

    bias = 1.0 / np.average(y_bin_centers, weights=proj)

    return bias


def calculate_RMS_around_1(values, weights):
    """Root Mean Square around 1 as proposed from comparison with CTA-MARS.

    The input values are vertical slices of the 2D histogram showing the bias-corrected charge resolution.

    Parameters
    ----------
    values : 1D array
        Values in reconstructed / true photoelectrons corrected for average bias.
    weights : 1D array
        Counts in a cell from the weigthed histogram.

    Returns
    -------
    rms : float
        Root Mean Square of around 1 for a vertical slice.

    """
    average = np.average(values, weights=weights)
    variance = np.average((values - average) ** 2, weights=weights)
    standard_deviation = np.sqrt(variance)
    a = np.power(standard_deviation, 2)
    b = np.power(average - 1, 2)
    rms = np.sqrt(a + b)
    return rms


def prepare_requirements(input_directory, site, obs_time):
    """Prepare requirements data as a dictionary.

    Parameters
    ----------
    input_directory : str or pathlib.Path
        Directory where the requirements files are stored.
    site: str
        Site identifier as in the name of the requirement files
    obs_time: int
        Observation time as in the name of the requirement files

    Returns
    -------
    requirements: dict
        Extracted requirements organized by benchmark.

    """
    requirements_input_filenames = {
        "sens": f"/{site}-{obs_time}.dat",
        "AngRes": f"/{site}-{obs_time}-AngRes.dat",
        "ERes": f"/{site}-{obs_time}-ERes.dat",
    }
    requirements = {}

    for key in requirements_input_filenames.keys():
        requirements[key] = Table.read(
            Path(input_directory) / requirements_input_filenames[key], format="ascii"
        )
    requirements["sens"].add_column(
        Column(data=(10 ** requirements["sens"]["col1"]), name="ENERGY")
    )
    requirements["sens"].add_column(
        Column(data=requirements["sens"]["col2"], name="SENSITIVITY")
    )

    return requirements


def compute_resolution(
    x_bin_edges, reco, true, mask=None, statistic=lambda x: np.percentile(np.abs(x), 68)
):
    """Compute a resolution as a binned statistic."""
    resolution = binned_statistic(
        np.log10(true[mask]),
        reco / true[mask] - 1,
        statistic=statistic,
        bins=x_bin_edges,
    )
    return resolution


def compute_bias(x_bin_edges, reco, true, mask=None, statistic="median"):
    """Compute bias as a binned statistic."""
    bias = binned_statistic(
        np.log10(true[mask]),
        reco[mask] / true[mask] - 1,
        statistic=statistic,
        bins=x_bin_edges,
    )
    return bias


def get_evt_subarray_model_output(
    data,
    weight_name="reco_energy_tel_weigth",
    keep_cols=["reco_energy"],
    model_output_name="reco_energy_tel",
    model_output_name_evt="reco_energy",
):
    """
    Returns DataStore with keepcols + score/target columns of model at the
    level-subarray-event.

    Parameters
    ----------
    data: `~pandas.DataFrame`
        Data frame
    weight_name: `str`
        Variable name in data frame to weight events with
    keep_cols: `list`, optional
        List of variables to keep in resulting data frame
    model_output_name: `str`, optional
        Name of model output (image level)
    model_output_name_evt: `str`, optional
        Name of averaged model output (shower level)

    Returns
    -------
    data: `~pandas.DataFrame`
        Data frame
    """

    keep_cols += [model_output_name]
    keep_cols += [weight_name]
    new_data = data[keep_cols].copy(deep=True)

    new_data[model_output_name_evt] = np.zeros(len(new_data))
    new_data.set_index(["tel_id"], append=True, inplace=True)

    new_data[model_output_name_evt] = new_data.groupby(["obs_id", "event_id"]).apply(
        lambda g: np.average(g[model_output_name], weights=g[weight_name])
    )

    # Remove columns
    if (
        model_output_name != "reco_energy_tel"
    ):  # we want to keep the telescope-wise energy (might keep also gammaness in the future)
        new_data = new_data.drop(columns=[model_output_name])

    # Remove duplicates
    new_data = new_data[~new_data.index.duplicated(keep="first")]

    return new_data


def sum_of_squares(x):
    x = np.asanyarray(x)
    if len(x) == 0:
        return 0
    mean = x.mean()
    return np.sum((x - mean) ** 2)


class OnlineBinnedStats:
    """Class to dynamically compute one-dimensional binned statistics.

    Parameters
    ----------
    bin_edges: array-like
        Values which define the edges of the bins to use.

    Notes
    -----
    This is an implementation of the Welford's online algorithm
    (see [1]_ and references therein).

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm
    """

    def __init__(self, bin_edges):
        self.bin_edges = bin_edges
        self.n_bins = len(bin_edges) - 1
        self.n = np.zeros(self.n_bins)
        self._mean = np.zeros(self.n_bins)
        self._m2 = np.zeros(self.n_bins)

    def update(self, x, values):
        """Update the mean and (estimated) variance of the sequence."""
        n = binned_statistic(x, values, "count", self.bin_edges).statistic
        mean = binned_statistic(x, values, "mean", bins=self.bin_edges).statistic
        m2 = binned_statistic(x, values, sum_of_squares, bins=self.bin_edges).statistic

        # empty bins are nan, but we need 0
        empty = n == 0
        mean[empty] = 0
        m2[empty] = 0

        n_total = self.n + n
        delta = self._mean - mean
        v = n_total > 0  # to avoid dividing by 0 and remove more NaNs
        self._mean[v] = (self.n[v] * self._mean[v] + n[v] * mean[v]) / n_total[v]
        self._m2[v] += m2[v] + delta[v] ** 2 * self.n[v] * n[v] / n_total[v]
        self.n = n_total

    @property
    def mean(self):
        """Compute the mean for bins with at least 1 count."""
        mean = np.full(self.n_bins, np.nan)
        valid = self.n > 0
        mean[valid] = self._mean[valid]
        return mean

    @property
    def std(self):
        """Compute the standard deviation for bins with at least 1 count."""
        std = np.full(self.n_bins, np.nan)
        valid = self.n > 1
        std[valid] = np.sqrt(self._m2[valid] / (self.n[valid] - 1))
        return std

    @property
    def bin_centers(self):
        """Compute the center for each bin."""
        return 0.5 * (self.bin_edges[:-1] + self.bin_edges[1:])

    @property
    def bin_width(self):
        """Compute the width of all bins."""
        return np.diff(self.bin_edges)


def create_lookup_function(binned_stat):
    """
    Returns a function f(x,y) that evaluates the lookup at a point.
    """
    cx = 0.5 * (binned_stat.x_edge[0:-1] + binned_stat.x_edge[1:])
    cy = 0.5 * (binned_stat.y_edge[0:-1] + binned_stat.y_edge[1:])
    z = binned_stat.statistic
    z[~np.isfinite(z)] = 0  # make sure there are no infs or nans
    interpolator = RectBivariateSpline(x=cx, y=cy, z=z, kx=1, ky=1, s=0)
    return lambda x, y: interpolator.ev(x, y)  # go back to TeV and evaluate


def compute_psf(data, ebins, radius):
    nbin = len(ebins) - 1
    psf = np.zeros(nbin)
    psf_err = np.zeros(nbin)
    for idx in range(nbin):
        emin = ebins[idx]
        emax = ebins[idx + 1]
        sel = data.loc[
            (data["true_energy"] >= emin) & (data["true_energy"] < emax), ["xi"]
        ]
        if len(sel) != 0:
            psf[idx] = np.percentile(sel["xi"], radius)
            psf_err[idx] = psf[idx] / np.sqrt(len(sel))
        else:
            psf[idx] = 0.0
            psf_err[idx] = 0.0
    return psf, psf_err


def load_tel_id(file_name=None, tel_id=None):
    """Load R0 and R1 waveforms for 1 telescope."""

    if file_name is None:
        raise ValueError("input information is undefined")
    else:
        r0_waveforms = read_table(file_name, f"/r0/event/telescope/tel_{tel_id:03d}")
        r1_waveforms = read_table(file_name, f"/r1/event/telescope/tel_{tel_id:03d}")
        return r0_waveforms, r1_waveforms
