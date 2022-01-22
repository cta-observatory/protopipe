import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.stats import binned_statistic, norm


__all__ = ["plot_sensitivity_from_pyirf"]


LOWER_SIGMA_QUANTILE, UPPER_SIGMA_QUANTILE = norm().cdf([-1, 1])


def plot_DL1a_reco_true_correlation(ax,
                                    reco,
                                    true,
                                    nbins_x=400,
                                    nbins_y=400,
                                    mask=None,
                                    weights=None,
                                    title=None,
                                    tel_type=None):

    ax = plt.gca() if ax is None else ax

    if title:
        ax.set_title(title)
    ax.set_xlabel("log10(true #p.e)")
    ax.set_ylabel("log10(reco #p.e)")

    # Use the un-weighted histogram to count the real number of events
    h_no_weights = plt.hist2d(np.log10(true),
                              np.log10(reco),
                              bins=[nbins_x, nbins_y],
                              range=[[-7., 5.], [-7., 5.]],
                              norm=LogNorm())

    # This histogram has the weights applied,
    # which chages the number of entries
    # This is also what is plot
    h = plt.hist2d(np.log10(true),
                   np.log10(reco),
                   bins=[nbins_x, nbins_y],
                   range=[[-7., 5.], [-7., 5.]],
                   norm=LogNorm(),
                   cmap=plt.cm.rainbow,
                   weights=weights)

    ax.plot([0, 4], [0, 4], color="black")  # line showing perfect correlation
    ax.minorticks_on()
    ax.set_xticks(ticks=np.arange(-1, 5, 0.5))
    ax.set_xticklabels(["", ""] + [str(i) for i in np.arange(0, 5, 0.5)])
    ax.set_xlim(-0.2, 4.2)
    ax.set_ylim(-4., 4.)
    plt.colorbar(h[3],
                 ax=plt.gca()
                 )
    ax.grid(visible=True)

    #fig.savefig(f"./plots/calibration_recoPhesVsTruePhes_{tel_type}_protopipe_{analysis_name}.png")

    # Print some debug/benchmarking information
    print(
        f"Total number of entries in the plot of {tel_type} (before weighting) = {h_no_weights[0].sum()}")

    return ax


def plot_resolution(ax, x, resolution, label=None, fmt="bo"):
    """Plot a resolution.

    Parameters
    ----------
    ax: matplotlib.axes.Axes
    x: array-like
        Bin centers"""
    ax.plot(x, resolution, fmt, label=label)

    ax.hlines(0.0, ax.get_xlim()[0], ax.get_xlim()[
              1], ls="--", color="ideal case")

    ax.grid(which="both", axis="both", visible=True)
    ax.xlabel('log10(true #phe)')
    ax.ylabel('charge resolution')
    ax.legend(loc="best")
    ax.ylim(-0.2, 1.5)
    ax.xlim(0., 4.5)

    return ax


def plot_bias(ax, x, bias, **opt):
    """Plot a bias.

    Parameters
    ----------
    ax: matplotlib.axes.Axes
    x: array-like
        Bin centers

    Returns
    -------
    ax: matplotlib.axes.Axes
    """
    ax.plot(x, bias, **opt)
    ax.hlines(0.0, ax.get_xlim()[0], ax.get_xlim()[1], ls="--", label="no bias")

    ax.grid(which="both", axis="both", visible=True)
    ax.set_xlabel('log10(true #phe)')
    ax.set_ylabel('charge bias as mean(reco/true - 1)')
    ax.set_ylim(-1.25, 1.0)
    ax.set_xlim(0., 4.5)

    return ax


def get_single_pixels_spectrum(x,
                               bins,
                               total_entries,
                               xrange,
                               **kwargs):
    """Log-Log data for single pixels spectrum."""

    # make histogram
    hist, xbins = np.histogram(np.log10(x[x > 0]), bins=bins, range=xrange)
    # plot cumulative histogram
    # each bin is divided by the total number of entries
    plt.semilogy(xbins[:-1], hist[::-1].cumsum()[::-1] / total_entries,
                 **kwargs)

    x_values = 0.5 * (xbins[:-1] + xbins[1:])
    y_values = hist[::-1].cumsum()[::-1] / total_entries

    return x_values, y_values


def plot_sensitivity_from_pyirf(ax,
                                data,
                                energy_type="reco",
                                y_unit=u.Unit('erg cm-2 s-1'),
                                label=None,
                                color=None):
    """Produce a sensitivity plot.

    Input is expected to come from pyirf.
    """

    ax = plt.gca() if ax is None else ax

    x = data[f'{energy_type}_energy_center']
    width = (data[f'{energy_type}_energy_high']
             - data[f'{energy_type}_energy_low'])
    y = (x**2 * data['flux_sensitivity'])
    ax.errorbar(
        x.to_value(u.TeV),
        y.to_value(y_unit),
        xerr=width.to_value(u.TeV) / 2,
        ls='',
        label=label,
        color=color
    )


def plot_binned_median(x, y, bins=None, yerr=True, ax=None, **kwargs):
    """
    Plot median of y in each bin in x with the central 1-sigma interval as errorbars.
    """
    ax = ax or plt.gca()

    valid = np.isfinite(y)

    median, edges, _ = binned_statistic(
        x[valid], y[valid], statistic="median", bins=bins
    )

    if yerr is True:
        q_low, _, _ = binned_statistic(
            x[valid],
            y[valid],
            statistic=lambda vals: np.quantile(vals, LOWER_SIGMA_QUANTILE),
            bins=edges,
        )
        q_high, _, _ = binned_statistic(
            x[valid],
            y[valid],
            statistic=lambda vals: np.quantile(vals, UPPER_SIGMA_QUANTILE),
            bins=edges,
        )
        yerr = [median - q_low, q_high - median]
    else:
        yerr = None

    center = 0.5 * (edges[1:] + edges[:-1])
    width = np.diff(edges)

    return ax.errorbar(
        center,
        median,
        xerr=width / 2,
        yerr=yerr,
        **kwargs,
    )


def plot_binned_mean(x, y, bins=None, yerr=True, ax=None, **kwargs):
    """
    Plot mean of y in each bin in x with standard deviation as errorbars.
    """
    ax = ax or plt.gca()

    valid = np.isfinite(y)

    mean, edges, _ = binned_statistic(x[valid], y[valid], statistic="mean", bins=bins)

    if yerr is True:
        yerr, _, _ = binned_statistic(
            x[valid],
            y[valid],
            statistic=lambda vals: np.quantile(vals, LOWER_SIGMA_QUANTILE),
            bins=edges,
        )
    else:
        yerr = None

    center = 0.5 * (edges[1:] + edges[:-1])
    width = np.diff(edges)

    return ax.errorbar(
        center,
        mean,
        xerr=width / 2,
        yerr=yerr,
        **kwargs,
    )
