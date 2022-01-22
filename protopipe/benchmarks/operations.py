from astropy.table import Table, Column
import astropy.units as u
from scipy.stats import binned_statistic
import numpy as np

__all__ = ["compute_weight_BTEL1010",
           "average_bias_of_charge_resolution",
           "calculate_RMS_around_1",
           "prepare_requirements"]


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
    weight = np.power(true_energy / 200., target_slope - spec_slope)
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


def average_bias_of_charge_resolution(x_bin_edges,
                                      y_bin_edges,
                                      hist,
                                      min_phe=50,
                                      max_phe=500):
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

    bias = 1. / np.average(y_bin_centers, weights=proj)

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
    variance = np.average((values - average)**2, weights=weights)
    standard_deviation = np.sqrt(variance)
    a = np.power(standard_deviation, 2)
    b = np.power(average - 1, 2)
    rms = np.sqrt(a + b)
    return rms


def prepare_requirements(input_directory, site, obs_time):
    requirements_input_filenames = {"sens": f'/{site}-{obs_time}.dat',
                                    "AngRes": f'/{site}-{obs_time}-AngRes.dat',
                                    "ERes": f'/{site}-{obs_time}-ERes.dat'}
    requirements = {}

    for key in requirements_input_filenames.keys():
        requirements[key] = Table.read(
            input_directory + requirements_input_filenames[key], format='ascii')
    requirements['sens'].add_column(
        Column(data=(10**requirements['sens']['col1']), name='ENERGY'))
    requirements['sens'].add_column(
        Column(data=requirements['sens']['col2'], name='SENSITIVITY'))

    return requirements


def compute_resolution(x_bin_edges, reco, true, mask=None,
                       statistic=lambda x: np.percentile(np.abs(x), 68)):
    """Compute a resolution as a binned statistic."""
    resolution = binned_statistic(np.log10(true[mask]),
                                  reco / true[mask] - 1,
                                  statistic=statistic,
                                  bins=x_bin_edges)
    return resolution


def compute_bias(x_bin_edges, reco, true, mask=None, statistic="median"):
    """Compute bias as a binned statistic."""
    bias = binned_statistic(np.log10(true[mask]),
                            reco[mask] / true[mask] - 1,
                            statistic=statistic,
                            bins=x_bin_edges)
    return bias
