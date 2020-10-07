import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import accuracy_score

from .utils import plot_hist, plot_distributions, plot_roc_curve

__all__ = [
    "ModelDiagnostic",
    "RegressorDiagnostic",
    "ClassifierDiagnostic",
    "BoostedDecisionTreeDiagnostic",
]


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
        ax.set_xticks(np.arange(0, len(importance) + 1))
        ax.set_xticklabels(feature_labels, rotation=75)

        return ax

    def plot_features(
        self, data_list, nbin=30, hist_kwargs_list={}, error_kw_list={}, ncols=2
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
    Class to plot several diagnostic plot for regression

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
        self, model, feature_name_list, target_name, data_train, data_test, output_name
    ):
        super().__init__(model, feature_name_list, target_name)

        self.data_train = data_train
        self.data_test = data_test

        self.target_estimation_name = self.target_name

        self.output_name = output_name
        self.output_name_img = output_name + "_img"

        # Compute and add target estimation
        self.data_train = self.add_image_model_output(
            self.data_train, col_name=self.output_name_img
        )
        self.data_test = self.add_image_model_output(
            self.data_test, col_name=self.output_name_img
        )

    @staticmethod
    def plot_resolution_distribution(
        ax, y_true, y_reco, nbin=100, fit_range=[-3, 3], fit_kwargs={}, hist_kwargs={}
    ):
        """
        Compute bias and resolution with a gaussian fit
        and returns a plot with the fit results and the migration distribution
        """

        def gauss(x, ampl, mean, std):
            return ampl * np.exp(-0.5 * ((x - mean) / std) ** 2)

        if ax is None:
            import matplotlib.pyplot as plt

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
            print("Not enough stat ? (#evts={})".format(len(y_true)))

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

    def add_image_model_output(self, data, col_name):
        data[col_name] = self.model.predict(data[self.feature_name_list])
        return data


class ClassifierDiagnostic(ModelDiagnostic):
    """
    Class to plot several diagnostic plot for classification. Assume that positives and
    negatives are respectively labeled as 1 and 0.

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
            data[col_name] = self.model.predict_proba(data[self.feature_name_list])[
                :, 1
            ]
        return data

    def plot_image_model_output_distribution(
        self,
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

    # def plot_evt_model_output_distribution(self,
    #                                 data_train,
    #                                 data_test,
    #                                 nbin=30,
    #                                 hist_kwargs_list=[
    #             {'edgecolor': 'blue', 'color': 'blue', 'label': 'Gamma training sample',
    #             'alpha': 0.2, 'fill': True, 'ls': '-', 'lw': 2},
    #             {'edgecolor': 'blue', 'color': 'blue', 'label': 'Gamma test sample',
    #             'alpha': 1, 'fill': False, 'ls': '--', 'lw': 2},
    #             {'edgecolor': 'red', 'color': 'red', 'label': 'Proton training sample',
    #             'alpha': 0.2, 'fill': True, 'ls': '-', 'lw': 2},
    #             {'edgecolor': 'red', 'color': 'red', 'label': 'Proton test sample',
    #             'alpha': 1, 'fill': False, 'ls': '--', 'lw': 2}
    #         ],
    #                                 error_kw_list=[
    #             dict(ecolor='blue', lw=2, capsize=3, capthick=2, alpha=0.2),
    #             dict(ecolor='blue', lw=2, capsize=3, capthick=2, alpha=1),
    #             dict(ecolor='red', lw=2, capsize=3, capthick=2, alpha=0.2),
    #             dict(ecolor='red', lw=2, capsize=3, capthick=2, alpha=1)
    # ]):
    #     return plot_distributions(
    #         [self.model_output_name],
    #         [data_train.query('label==1'), data_test.query('label==1'),
    #          data_train.query('label==0'), data_test.query('label==0')],
    #         nbin,
    #         hist_kwargs_list,
    #         error_kw_list,
    #         1
    #     )
    #
    # def plot_evt_model_output_distribution_variation(self, data_train, data_test, cut_list,
    #                                                  nbin=30, ncols=2, hist_kwargs_list={},
    #                                                  error_kw_list={}):
    #     """
    #     Plot model output distribution for several data set for a list of cut
    #
    #     Parameters
    #     ----------
    #     data_train: `~pandas.DataFrame`
    #         Data frame for training sample
    #     data_test:`~pandas.DataFrame`
    #         Data frame for test sample
    #     cut_list: list
    #         List of cuts
    #     nbin: int
    #         Number of bins
    #     ncols: int
    #         Number of column to display variation
    #     hist_kwargs_list: list
    #         list of kwargs
    #     error_kw_list:
    #         list of error_dict
    #
    #     Returns
    #     -------
    #     fig: `matplotlib.figure.Figure`
    #         Figure object
    #     axes: list
    #         List of `~matplotlib.axes.Axes`
    #     """
    #     import matplotlib.pyplot as plt
    #     n_feature = len(cut_list)
    #     nrows = int(n_feature / ncols) if n_feature % ncols == 0 else int(
    #         (n_feature + 1) / ncols)
    #     fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5 * ncols, 3 * nrows))
    #     if nrows == 1 and ncols == 1:
    #         axes = [axes]
    #     else:
    #         axes = axes.flatten()
    #
    #     data_list = [data_train.query('label==1'), data_test.query('label==1'),
    #                  data_train.query('label==0'), data_test.query('label==0')]
    #
    #     for i, colname in enumerate(cut_list):
    #         ax = axes[i]
    #
    #         # Range for binning
    #         range_min = np.array([data.query(cut_list[i])[self.model_output_name].min() for data in data_list])
    #         range_min = range_min[np.where(np.isfinite(range_min))[0]]
    #         range_min = min(range_min)
    #
    #         range_max = np.array([data.query(cut_list[i])[self.model_output_name].max() for data in data_list])
    #         range_max = range_max[np.where(np.isfinite(range_max))[0]]
    #         range_max = min(range_max)
    #
    #         # myrange = [range_min, range_max]
    #         myrange = [0,1]
    #
    #         for j, data in enumerate(data_list):
    #             if len(data) == 0:
    #                 continue
    #
    #             ax = plot_hist(
    #                 ax=ax, data=data.query(cut_list[i])[self.model_output_name],
    #                 nbin=nbin, limit=myrange,
    #                 norm=True, yerr=True,
    #                 hist_kwargs=hist_kwargs_list[j],
    #                 error_kw=error_kw_list[j]
    #             )
    #
    #         ax.set_xlim(myrange)
    #         ax.set_xlabel(self.model_output_name)
    #         ax.set_ylabel('Arbitrary units')
    #         ax.legend(loc='best', fontsize='x-small')
    #         ax.set_title(cut_list[i])
    #         ax.grid()
    #     plt.tight_layout()
    #
    #     return fig, axes
    #
    # def plot_evt_roc_curve_variation(self, ax, data_test, cut_list):
    #     """
    #
    #     Parameters
    #     ----------
    #     ax: `~matplotlib.axes.Axes`
    #         Axis
    #     data_test: `~pd.DataFrame`
    #         Test data
    #     cut_list: `list`
    #         Cut list
    #
    #     Returns
    #     -------
    #     ax:  `~matplotlib.axes.Axes`
    #         Axis
    #     """
    #     color = 1.
    #     step_color = 1. / (len(cut_list))
    #     for i, cut in enumerate(cut_list):
    #         c = color - (i + 1) * step_color
    #
    #         data = data_test.query(cut)
    #         if len(data) == 0:
    #             continue
    #
    #         opt = dict(color=str(c), lw=2, label='{}'.format(cut.replace('reco_energy', 'E')))
    #         plot_roc_curve(ax, data[self.model_output_name], data['label'], **opt)
    #     ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    #
    #     return ax


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


# class RandomForestDiagnostic(object):
#     """
#     Class producing diagnostic plot for the RF method
#     """
#
#     @classmethod
#     def plot_error_rate
