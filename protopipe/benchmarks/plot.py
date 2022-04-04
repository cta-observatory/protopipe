"""
This module contains functions and classes to make plots.

The contents of the module can be used separately,
but they are also used used by the benchmarking notebooks.

Notes
-----
This module has been created essentially to clean-up/refactor the
notebooks.
Most of the code came with the original version of the notebook,
so it is quite old, some parts have been modernized.
The implementation of functions and classes is far from perfect and
we should really try to synchronize in some way with ctaplot/ctabenchmarks.
"""
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from astropy.table import QTable
from matplotlib.colors import LogNorm
from pyirf.utils import cone_solid_angle
from scipy.optimize import curve_fit
from scipy.stats import binned_statistic, norm
from sklearn.metrics import accuracy_score, auc, roc_curve

LOWER_SIGMA_QUANTILE, UPPER_SIGMA_QUANTILE = norm().cdf([-1, 1])

__all__ = [
    "plot_profile",
    "plot_DL1a_reco_true_correlation",
    "plot_resolution",
    "plot_bias",
    "get_single_pixels_spectrum",
    "plot_sensitivity_from_pyirf",
    "plot_binned_mean",
    "plot_binned_median",
    "plot_hist",
    "plot_distributions",
    "plot_roc_curve",
    "plot_evt_roc_curve_variation",
    "plot_psf",
    "plot_background_rate",
    "BoostedDecisionTreeDiagnostic",
    "ModelDiagnostic",
    "RegressorDiagnostic",
    "ClassifierDiagnostic",
]


def plot_profile(ax, data, xcol, ycol, n_xbin, x_range, logx=False, **kwargs):
    """Plot a profiled histogram.

    Parameters
    ----------
    ax: `matplotlib.axes.Axes`
        Empty or existing axes
    data: `pandas.DataFrame``
        A dataframe with at least 2 columns
    xcol: str
        Name of the column to use as X coordinate
    ycol: str
        Name of the column to use as Y coordinate
    n_xbin: int
        Number of bins in the X axis
    x_range: list
        Lower and upper limit of the X data to use

    Returns
    -------
    ax: `matplotlib.axes.Axes`
        Axes filled by the plot
    """
    color = kwargs.get("color", "red")
    label = kwargs.get("label", "")
    fill = kwargs.get("fill", False)
    alpha = kwargs.get("alpha", 1)
    xlabel = kwargs.get("xlabel", "")
    ylabel = kwargs.get("ylabel", "")
    xlim = kwargs.get("xlim", None)
    ms = kwargs.get("ms", 8)

    if logx is False:
        bin_edges = np.linspace(x_range[0], x_range[-1], n_xbin, True)
        bin_center = 0.5 * (bin_edges[1:] + bin_edges[:-1])
        bin_width = bin_edges[1:] - bin_edges[:-1]
    else:
        bin_edges = np.logspace(
            np.log10(x_range[0]), np.log10(x_range[-1]), n_xbin, True
        )
        bin_center = np.sqrt(bin_edges[1:] * bin_edges[:-1])
        bin_width = bin_edges[1:] - bin_edges[:-1]

    y = []
    yerr = []
    for idx in range(len(bin_center)):
        counts = data[
            (data[xcol] > bin_edges[idx]) & (data[xcol] <= bin_edges[idx + 1])
        ][ycol]
        y.append(counts.mean())
        yerr.append(counts.std() / np.sqrt(len(counts)))

    ax.errorbar(
        x=bin_center,
        y=y,
        xerr=bin_width / 2.0,
        yerr=yerr,
        label=label,
        fmt="o",
        color=color,
        ms=ms,
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if logx is True:
        ax.set_xscale("log")
    ax.legend(loc="upper right", framealpha=1, fontsize="medium")

    return ax


def plot_DL1a_reco_true_correlation(
    ax,
    reco,
    true,
    nbins_x=400,
    nbins_y=400,
    mask=None,
    weights=None,
    title=None,
    tel_type=None,
):
    """Plot the correlation between reconstructed and true number of photoelectrons.

    Parameters
    ----------
    ax: `matplotlib.axes.Axes`
        Empty or existing axes
    reco: ndarray
        1D array containing the reconstructed charges obtained by flatteing the 2D array (n_samples, n_pixels)
    true: ndarray
        1D array containing the true charges obtained by flatteing the 2D array (n_samples, n_pixels)
    nbins_x: int
        Number of bins for the X axis (reconstructed charges)
    nbins_y: int
        Number of bins for the Y axis (true charges)
    mask: ndarray
        A boolean mask (default: None)
    weights: ndarray
        An array of the same shape of reco and true to weight the charges (default: None)
    title: str
        Title of the plot (default: None)
    tel_type: str
        String representation of a ctapipe.instrument.TelescopeDescription instance

    Returns
    -------
    ax: `matplotlib.axes.Axes`
        Axes filled by the plot
    """

    ax = plt.gca() if ax is None else ax

    if title:
        ax.set_title(title)
    ax.set_xlabel("log10(true #p.e)")
    ax.set_ylabel("log10(reco #p.e)")

    # Use the un-weighted histogram to count the real number of events
    h_no_weights = plt.hist2d(
        np.log10(true),
        np.log10(reco),
        bins=[nbins_x, nbins_y],
        range=[[-7.0, 5.0], [-7.0, 5.0]],
        norm=LogNorm(),
    )

    # This histogram has the weights applied,
    # which chages the number of entries
    # This is also what is plot
    h = plt.hist2d(
        np.log10(true),
        np.log10(reco),
        bins=[nbins_x, nbins_y],
        range=[[-7.0, 5.0], [-7.0, 5.0]],
        norm=LogNorm(),
        cmap=plt.cm.rainbow,
        weights=weights,
    )

    ax.plot([0, 4], [0, 4], color="black")  # line showing perfect correlation
    ax.minorticks_on()
    ax.set_xticks(ticks=np.arange(-1, 5, 0.5))
    ax.set_xticklabels(["", ""] + [str(i) for i in np.arange(0, 5, 0.5)])
    ax.set_xlim(-0.2, 4.2)
    ax.set_ylim(-4.0, 4.0)
    plt.colorbar(h[3], ax=plt.gca())
    ax.grid(visible=True)

    # fig.savefig(f"./plots/calibration_recoPhesVsTruePhes_{tel_type}_protopipe_{analysis_name}.png")

    # Print some debug/benchmarking information
    print(
        f"Total number of entries in the plot of {tel_type} (before weighting) = {h_no_weights[0].sum()}"
    )

    return ax


def plot_resolution(ax, x, resolution, label=None, fmt="bo"):
    """Plot a resolution.

    Parameters
    ----------
    ax: matplotlib.axes.Axes
    x: array-like
        Bin centers"""
    ax.plot(x, resolution, fmt, label=label)

    ax.hlines(0.0, ax.get_xlim()[0], ax.get_xlim()[1], ls="--", color="ideal case")

    ax.grid(which="both", axis="both", visible=True)
    ax.xlabel("log10(true #phe)")
    ax.ylabel("charge resolution")
    ax.legend(loc="best")
    ax.ylim(-0.2, 1.5)
    ax.xlim(0.0, 4.5)

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
    ax.set_xlabel("log10(true #phe)")
    ax.set_ylabel("charge bias as mean(reco/true - 1)")
    ax.set_ylim(-1.25, 1.0)
    ax.set_xlim(0.0, 4.5)

    return ax


def get_single_pixels_spectrum(x, bins, total_entries, xrange, **kwargs):
    """Log-Log data for single pixels spectrum."""

    # make histogram
    hist, xbins = np.histogram(np.log10(x[x > 0]), bins=bins, range=xrange)
    # plot cumulative histogram
    # each bin is divided by the total number of entries
    plt.semilogy(xbins[:-1], hist[::-1].cumsum()[::-1] / total_entries, **kwargs)

    x_values = 0.5 * (xbins[:-1] + xbins[1:])
    y_values = hist[::-1].cumsum()[::-1] / total_entries

    return x_values, y_values


def plot_sensitivity_from_pyirf(
    ax, data, energy_type="reco", y_unit=u.Unit("erg cm-2 s-1"), label=None, color=None
):
    """Produce a sensitivity plot.

    Input is expected to come from pyirf.
    """

    ax = plt.gca() if ax is None else ax

    x = data[f"{energy_type}_energy_center"]
    width = data[f"{energy_type}_energy_high"] - data[f"{energy_type}_energy_low"]
    y = x ** 2 * data["flux_sensitivity"]
    ax.errorbar(
        x.to_value(u.TeV),
        y.to_value(y_unit),
        xerr=width.to_value(u.TeV) / 2,
        ls="",
        label=label,
        color=color,
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

    return ax.errorbar(center, median, xerr=width / 2, yerr=yerr, **kwargs)


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

    return ax.errorbar(center, mean, xerr=width / 2, yerr=yerr, **kwargs)


def plot_hist(
    ax, data, nbin, limit, norm=False, yerr=False, hist_kwargs={}, error_kw={}
):
    """Utility function to plot histogram"""
    bin_edges = np.linspace(limit[0], limit[-1], nbin + 1, True)
    y, tmp = np.histogram(data, bins=bin_edges)
    weights = np.ones_like(y)
    if norm is True:
        weights = weights / float(np.sum(y))
    if yerr is True:
        yerr = np.sqrt(y) * weights
    else:
        yerr = np.zeros(len(y))

    centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    width = bin_edges[1:] - bin_edges[:-1]
    ax.bar(
        centers, y * weights, width=width, yerr=yerr, error_kw=error_kw, **hist_kwargs
    )

    return ax


def plot_distributions(
    suptitle,
    feature_list,
    data_list,
    nbin=30,
    hist_kwargs_list={},
    error_kw_list={},
    ncols=2,
):
    """Plot feature distributions for several data set. Returns list of axes."""
    n_feature = len(feature_list)
    nrows = (
        int(n_feature / ncols)
        if n_feature % ncols == 0
        else round((n_feature + 1) / ncols)
    )

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5 * ncols, 5 * nrows))
    plt.suptitle(suptitle)
    if nrows == 1 and ncols == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for i, colname in enumerate(feature_list):
        ax = axes[i]

        for j, data in enumerate(data_list):
            if colname in [
                "hillas_intensity",
                "h_max",
                "impact_dist",
            ]:  # automatically log these quantities for clarity
                # Range for binning
                range_min = min([np.log10(data[colname]).min() for data in data_list])
                range_max = max([np.log10(data[colname]).max() for data in data_list])
                myrange = [range_min, range_max]

                ax = plot_hist(
                    ax=ax,
                    data=np.log10(data[colname]),
                    nbin=nbin,
                    limit=myrange,
                    norm=True,
                    yerr=True,
                    hist_kwargs=hist_kwargs_list[j],
                    error_kw=error_kw_list[j],
                )
                ax.set_xlabel(f"log10({colname})")
            else:
                range_min = min([data[colname].min() for data in data_list])
                range_max = max([data[colname].max() for data in data_list])
                myrange = [range_min, range_max]

                ax = plot_hist(
                    ax=ax,
                    data=data[colname],
                    nbin=nbin,
                    limit=myrange,
                    norm=True,
                    yerr=True,
                    hist_kwargs=hist_kwargs_list[j],
                    error_kw=error_kw_list[j],
                )

                ax.set_xlabel(colname)

        ax.set_ylabel("Arbitrary units")
        ax.legend()
        ax.grid()

    return fig, axes


def plot_roc_curve(ax, model_output, y, **kwargs):
    """Plot ROC curve for a given set of model outputs and labels"""
    fpr, tpr, _ = roc_curve(y_score=model_output, y_true=y)
    roc_auc = auc(fpr, tpr)
    label = "{} (area={:.2f})".format(kwargs.pop("label"), roc_auc)  # Remove label
    ax.plot(fpr, tpr, label=label, **kwargs)
    return ax


def plot_evt_roc_curve_variation(ax, data_test, cut_list, model_output_name):
    """

    Parameters
    ----------
    ax: `~matplotlib.axes.Axes`
        Axis
    data_test: `~pd.DataFrame`
        Test data
    cut_list: `list`
         Cut list

    Returns
    -------
    ax:  `~matplotlib.axes.Axes`
        Axis
    """
    color = 1.0
    step_color = 1.0 / (len(cut_list))
    for i, cut in enumerate(cut_list):
        c = color - (i + 1) * step_color

        data = data_test.query(cut)
        if len(data) == 0:
            continue

        opt = dict(
            color=str(c), lw=2, label="{}".format(cut.replace("reco_energy", "E"))
        )
        plot_roc_curve(ax, data[model_output_name], data["label"], **opt)
    ax.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")

    return ax


def plot_psf(ax, x, y, err, **kwargs):

    label = kwargs.get("label", "")
    xlabel = kwargs.get("xlabel", "")
    xlim = kwargs.get("xlim", None)

    ax.errorbar(x, y, yerr=err, fmt="o", label=label)
    ax.set_ylabel("PSF (68% containment)")
    ax.set_xlabel("True energy [TeV]")
    if xlim is not None:
        ax.set_xlim(xlim)
    return ax


def plot_background_rate(input_file, ax, label, color):

    color = color if color else None

    rad_max = QTable.read(input_file, hdu="RAD_MAX")[0]
    bg_rate = QTable.read(input_file, hdu="BACKGROUND")[0]

    reco_bins = np.append(bg_rate["ENERG_LO"], bg_rate["ENERG_HI"][-1])

    # first fov bin, [0, 1] deg
    fov_bin = 0
    rate_bin = bg_rate["BKG"].T[:, fov_bin]

    # interpolate theta cut for given e reco bin
    e_center_bg = 0.5 * (bg_rate["ENERG_LO"] + bg_rate["ENERG_HI"])
    e_center_theta = 0.5 * (rad_max["ENERG_LO"] + rad_max["ENERG_HI"])
    theta_cut = np.interp(e_center_bg, e_center_theta, rad_max["RAD_MAX"].T[:, 0])

    # undo normalization
    rate_bin *= cone_solid_angle(theta_cut)
    rate_bin *= np.diff(reco_bins)
    ax.errorbar(
        0.5 * (bg_rate["ENERG_LO"] + bg_rate["ENERG_HI"]).to_value(u.TeV)[1:-1],
        rate_bin.to_value(1 / u.s)[1:-1],
        xerr=np.diff(reco_bins).to_value(u.TeV)[1:-1] / 2,
        ls="",
        label=label,
        color=color,
    )


class BoostedDecisionTreeDiagnostic(object):
    """
    Class producing diagnostic plot for the BDT method
    """

    @classmethod
    def plot_error_rate(cls, ax, model, data_scikit, **kwargs):
        """Diagnostic plot showing error rate as a function of the specialisation"""
        import matplotlib.pyplot as plt

        if ax is None:
            ax = plt.gca()

        test_errors = []
        for test_predict in model.staged_predict(data_scikit["X_test"]):
            test_errors.append(
                1.0 - accuracy_score(test_predict, data_scikit["y_test"])
            )

        ntrees = len(model)
        ax.plot(range(1, ntrees + 1), test_errors, **kwargs)
        ax.set_xlabel("Number of Trees")
        ax.set_ylabel("Error rate")
        ax.grid()
        plt.tight_layout()
        return ax

    @classmethod
    def plot_tree_error_rate(cls, ax, model, **kwargs):
        """Diagnostic plot showing tree error rate"""
        import matplotlib.pyplot as plt

        if ax is None:
            ax = plt.gca()

        ntrees = len(model)
        estimator_errors = model.estimator_errors_[:ntrees]

        ntrees = len(model)
        ax.plot(range(1, ntrees + 1), estimator_errors, **kwargs)
        ax.set_xlabel("Tree number")
        ax.set_ylabel("Error rate / tree")
        ax.grid()
        plt.tight_layout()
        return ax


class ModelDiagnostic(object):
    """
    Base class for model diagnostics.

    Parameters
    ----------
    model: `~sklearn.base.BaseEstimator`
        Best model
    feature_name_list: list
        List of the features used to buil the model
    target_name: str
        Name of the target (e.g. score, gamaness, energy, etc.)
    """

    def __init__(self, model, feature_name_list, target_name):
        self.model = model
        self.feature_name_list = feature_name_list
        self.target_name = target_name

    def plot_feature_importance(self, ax, **kwargs):
        """
        Plot importance of features

        Parameters
        ----------
        ax: `~matplotlib.axes.Axes`
            Axis
        """
        if ax is None:
            import matplotlib.pyplot as plt

            ax = plt.gca()

        importance = self.model.feature_importances_
        importance, feature_labels = zip(
            *sorted(zip(importance, self.feature_name_list), reverse=True)
        )

        bin_edges = np.arange(0, len(importance) + 1)
        bin_width = bin_edges[1:] - bin_edges[:-1] - 0.1

        ax.bar(bin_edges[:-1], importance, width=bin_width, **kwargs)
        ax.set_xticks(np.arange(0, len(importance)))
        ax.set_xticklabels(feature_labels, rotation=75)

        return ax

    def plot_features(
        self,
        suptitle,
        data_list,
        nbin=30,
        hist_kwargs_list={},
        error_kw_list={},
        ncols=2,
    ):
        """
        Plot model features for different data set (e.g. training and test samples).

        Parameters
        ----------
        data_list: list
            List of data
        nbin: int
            Number of bin
        hist_kwargs_list: dict
            Dictionary with histogram options
        error_kw_list: dict
            Dictionary with error bar options
        ncols: int
            Number of columns
        """
        return plot_distributions(
            suptitle,
            self.feature_name_list,
            data_list,
            nbin,
            hist_kwargs_list,
            error_kw_list,
            ncols,
        )

    def add_image_model_output(self):
        raise NotImplementedError("Please Implement this method")


class RegressorDiagnostic(ModelDiagnostic):
    """
    Class to plot several diagnostic plots for regression.

    Parameters
    ----------
    model: sklearn.base.BaseEstimator
        Scikit model
    feature_name_list: str
        List of features
    target_name: str
        Name of target (e.g. `mc_energy`)
    data_train: `~pandas.DataFrame`
        Data frame
    data_test: `~pandas.DataFrame`
        Data frame
    """

    def __init__(
        self,
        model,
        feature_name_list,
        target_name,
        is_target_log,
        data_train,
        data_test,
        output_name,
        estimation_weight,
    ):
        super().__init__(model, feature_name_list, target_name)

        self.data_train = data_train
        self.data_test = data_test
        self.is_target_log = is_target_log
        self.feature_name_list = feature_name_list

        self.target_estimation_name = self.target_name
        self.estimation_weight = estimation_weight

        self.output_name = output_name
        self.output_name_img = output_name + "_tel"
        self.output_weight_img = output_name + "_tel" + "_weight"

        # Compute and add target estimation
        self.data_train = self.add_image_model_output(self.data_train)
        self.data_test = self.add_image_model_output(self.data_test)

    @staticmethod
    def plot_resolution_distribution(
        ax, y_true, y_reco, nbin=100, fit_range=[-3, 3], fit_kwargs={}, hist_kwargs={}
    ):
        """
        Compute bias and resolution with a gaussian fit
        and return a plot with the fit results and the migration distribution.
        """

        def gauss(x, ampl, mean, std):
            return ampl * np.exp(-0.5 * ((x - mean) / std) ** 2)

        if ax is None:
            ax = plt.gca()

        migration = (y_reco - y_true) / y_true
        bin_edges = np.linspace(fit_range[0], fit_range[-1], nbin + 1, True)
        y, tmp = np.histogram(migration, bins=bin_edges)
        x = (bin_edges[:-1] + bin_edges[1:]) / 2

        try:
            param, cov = curve_fit(gauss, x, y)
        except:
            param = [-1, -1, -1]
            cov = [[]]
            # print('Not enough stat ? (#evts={})'.format(len(y_true)))

        plot_hist(
            ax=ax,
            data=migration,
            nbin=nbin,
            yerr=False,
            norm=False,
            limit=fit_range,
            hist_kwargs=hist_kwargs,
        )

        ax.plot(x, gauss(x, param[0], param[1], param[2]), **fit_kwargs)

        return ax, param, cov

    def add_image_model_output(self, data):

        features_values = data[self.feature_name_list].to_numpy()

        if self.estimation_weight == "CTAMARS":
            # Get an array of trees
            predictions_trees = np.array(
                [tree.predict(features_values) for tree in self.model.estimators_]
            )
            v = np.mean(predictions_trees, axis=0)
            w = np.std(predictions_trees, axis=0)
            if self.is_target_log:
                data[self.output_name_img] = 10 ** v
                data[self.output_weight_img] = 10 ** w
            else:
                data[self.output_name_img] = v
                data[self.output_weight_img] = w
        else:
            data.eval(
                f"{self.output_weight_img} = {self.estimation_weight}", inplace=True
            )
            v = self.model.predict(features_values)
            if self.is_target_log:
                data[self.output_name_img] = 10 ** v
            else:
                data[self.output_name_img] = v

        return data


class ClassifierDiagnostic(ModelDiagnostic):
    """
    Class to plot several diagnostic plot for classification.

    Assume that positives and negatives are respectively labeled as 1 and 0.

    Parameters
    ----------
    model: sklearn.base.BaseEstimator
        Scikit model
    feature_name_list: list
        List of features
    model_output_name: str
        Name of output
    is_output_proba: bool
        If false, `decision_function` will be called, otherwise, predict_proba.
        In the last case we only consider the probability for signal event
    """

    def __init__(
        self,
        model,
        feature_name_list,
        target_name,
        data_train,
        data_test,
        model_output_name="score",
        is_output_proba=False,
    ):
        super().__init__(model, feature_name_list, target_name)

        self.data_train = data_train
        self.data_test = data_test
        self.model_output_name = model_output_name
        self.is_output_proba = is_output_proba

        # Compute and add model output
        self.data_train = self.add_image_model_output(
            self.data_train, col_name=self.model_output_name
        )
        self.data_test = self.add_image_model_output(
            self.data_test, col_name=self.model_output_name
        )

    def add_image_model_output(self, data, col_name):
        """Add model output column"""
        if self.is_output_proba is False:
            data[col_name] = self.model.decision_function(data[self.feature_name_list])
        else:  # Interested in signal probability
            data[col_name] = self.model.predict_proba(
                data[self.feature_name_list].to_numpy()
            )[:, 1]
        return data

    def plot_image_model_output_distribution(
        self,
        title="",
        cut=None,
        nbin=30,
        hist_kwargs_list=[
            {
                "edgecolor": "blue",
                "color": "blue",
                "label": "Gamma training sample",
                "alpha": 0.2,
                "fill": True,
                "ls": "-",
                "lw": 2,
            },
            {
                "edgecolor": "blue",
                "color": "blue",
                "label": "Gamma test sample",
                "alpha": 1,
                "fill": False,
                "ls": "--",
                "lw": 2,
            },
            {
                "edgecolor": "red",
                "color": "red",
                "label": "Proton training sample",
                "alpha": 0.2,
                "fill": True,
                "ls": "-",
                "lw": 2,
            },
            {
                "edgecolor": "red",
                "color": "red",
                "label": "Proton test sample",
                "alpha": 1,
                "fill": False,
                "ls": "--",
                "lw": 2,
            },
        ],
        error_kw_list=[
            dict(ecolor="blue", lw=2, capsize=3, capthick=2, alpha=0.2),
            dict(ecolor="blue", lw=2, capsize=3, capthick=2, alpha=1),
            dict(ecolor="red", lw=2, capsize=3, capthick=2, alpha=0.2),
            dict(ecolor="red", lw=2, capsize=3, capthick=2, alpha=1),
        ],
    ):
        """Plot output distribution. Need more output column"""
        if cut is not None:
            data_test = self.data_test.query(cut)
            data_train = self.data_train.query(cut)
        else:
            data_test = self.data_test
            data_train = self.data_train

        return plot_distributions(
            title,
            [self.model_output_name],
            [
                data_train.query("label==1"),
                data_test.query("label==1"),
                data_train.query("label==0"),
                data_test.query("label==0"),
            ],
            nbin,
            hist_kwargs_list,
            error_kw_list,
            1,
        )
