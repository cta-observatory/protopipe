import logging

import numpy as np
import pandas as pd
import pickle
import gzip

from astropy.table import QTable
import astropy.units as u

from pyirf.simulations import SimulatedEventsInfo

def percentiles(values, bin_values, bin_edges, percentile):
    # Seems complicated for vector defined as [inf, inf, .., inf]
    percentiles_binned = np.squeeze(np.full((len(bin_edges) - 1, len(values.shape)), np.inf))
    err_percentiles_binned = np.squeeze(np.full((len(bin_edges) - 1, len(values.shape)), np.inf))
    for i, (bin_l, bin_h) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
        try:
            print(i)
            print(bin_l)
            print(bin_h)
            distribution = values[(bin_values > bin_l) & (bin_values < bin_h)]
            percentiles_binned[i] = np.percentile(distribution, percentile)
            print(percentiles_binned[i])
            err_percentiles_binned[i] = percentiles_binned[i] / np.sqrt(len(distribution))
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
    ax.bar(centers, data * weights, width=width, yerr=yerr, error_kw=error_kw, **hist_kwargs)

    return ax


def save_obj(obj, name):
    """Save object in binary"""
    with gzip.open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    """Load object in binary"""
    with gzip.open(name, 'rb') as f:
        return pickle.load(f)
    
    
def read_DL2_pyirf(infile, run_header):
    """
    Read a DL2 HDF5 protopipe file and adapt them to pyirf format 
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

    events = QTable([list(df['obs_id']),
                     list(df['event_id']),
                     list(df['true_energy']) * u.TeV, 
                     list(df['reco_energy']) * u.TeV,
                     list(df['gammaness']),
                     list(df['NTels_reco']),
                     list(df['reco_alt']) * u.deg,
                     list(df['reco_az']) * u.deg,
                     list(df['true_alt']) * u.deg,
                     list(df['true_az']) * u.deg,
                     list(df['pointing_alt']) * u.deg,
                     list(df['pointing_az']) * u.deg,                   
                    ],
                    names=('obs_id',
                           'event_id',
                           'true_energy', 
                           'reco_energy', 
                           'gh_score',
                           'multiplicity',
                           'reco_alt',
                           'reco_az',
                           'true_alt',
                           'true_az',
                           'pointing_alt',
                           'pointing_az',   
                          ),
                   )
    
    n_runs = len(set(events['obs_id']))
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


def li_ma_significance(n_on, n_off, alpha=0.2):
    """
    Calculate the Li & Ma significance.
    Formula (17) doi.org/10.1086/161295
    This functions returns 0 significance when n_on < alpha * n_off
    instead of the negative sensitivities that would result from naively
    evaluating the formula.
    Parameters
    ----------
    n_on: integer or array like
        Number of events for the on observations
    n_off: integer of array like
        Number of events for the off observations
    alpha: float
        Ratio between the on region and the off region size or obstime.
    Returns
    -------
    s_lima: float or array
        The calculated significance
    """

    scalar = np.array(n_on, copy=False).shape == tuple()

    n_on = np.array(n_on, copy=False, ndmin=1)
    n_off = np.array(n_off, copy=False, ndmin=1)

    with np.errstate(divide="ignore", invalid="ignore"):
        p_on = n_on / (n_on + n_off)
        p_off = n_off / (n_on + n_off)

        t1 = n_on * np.log(((1 + alpha) / alpha) * p_on)
        t2 = n_off * np.log((1 + alpha) * p_off)

        # lim x+->0  (x log(x)) = 0
        t1[n_on == 0] = 0
        t2[n_off == 0] = 0

        ts = t1 + t2

        significance = np.sqrt(ts * 2)

    significance[n_on < (alpha * n_off)] = 0

    if scalar:
        return significance[0]

    return significance