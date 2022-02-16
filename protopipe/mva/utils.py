import numpy as np
import pandas as pd
from sklearn.metrics import auc, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


def prepare_data(ds, derived_features, cuts, select_data=True, label=None):
    """Add custom variables to the input data and optionally select it.

    Parameters
    ----------
    ds : pandas.DataFrame
        Input data not yet selected.
    derived_features: dict
        Dictionary of more complex featuresread from the configuration file.
    cuts: str
        Fiducial cuts from protopipe.mva.utils.make_cut_list
    select_data: bool
        If True apply cuts to the final dataframe.
    label: str
        Name of the classifier target label if any.

    Returns
    -------
    ds : pandas.DataFrame
        Input data integrated with new variables and optionally selected for
        the fiducial cuts.
    """

    # This is always useful
    ds["log10_true_energy"] = np.log10(ds["true_energy"])

    if label is not None:  # only for classification
        ds["label"] = np.full(len(ds), label)

        # This is needed because our reference analysis uses energy as
        # feature for classification
        # We should propably support a more elastic choice in the future.
        if not all(
            i in derived_features
            for i in ["log10_reco_energy", "log10_reco_energy_tel"]
        ):
            raise ValueError(
                "log10_reco_energy and log10_reco_energy_tel need to be model features."
            )

    # Compute derived features and add them to the dataframe
    for feature_name, feature_expression in derived_features.items():
        ds.eval(f"{feature_name} = {feature_expression}", inplace=True)

    if select_data:
        ds = ds.query(cuts)

    return ds


def make_cut_list(cuts):
    cut_list = ""
    for idx, cut in enumerate(cuts):
        cut_list += cut
        if idx != len(cuts) - 1:
            cut_list += " and "
    return cut_list


def split_train_test(survived_images, train_fraction, feature_name_list, target_name):
    """Split the data selected for cuts in train and test samples.

    If the estimator is a classifier, data is split in a stratified fashion,
    using this as the class labels.

    Parameters
    ----------
    survived_images: `~pandas.DataFrame`
        Images that survived the selection cuts.
    train_fraction: `float`
        Fraction of data to be used for training.
    feature_name_list: `list`
        List of variables to use for training the model.
    target_name: `str`
        Variable against which to train.

    Returns
    -------
    X_train: `~pandas.DataFrame`
        Data frame
    X_test: `~pandas.DataFrame`
        Data frame
    y_train: `~pandas.DataFrame`
        Data frame
    y_test: `~pandas.DataFrame`
        Data frame
    data_train: `~pandas.DataFrame`
        Training data indexed by observation ID and event ID.
    data_test: `~pandas.DataFrame`
        Test data indexed by observation ID and event ID.
    """

    # If the estimator is a classifier, data is split in a stratified fashion,
    # using this as the class labels
    labels = None
    if target_name == "label":
        labels = survived_images[target_name]

    if train_fraction != 1.0:
        data_train, data_test = train_test_split(
            survived_images,
            train_size=train_fraction,
            random_state=0,
            shuffle=True,
            stratify=labels,
        )
        y_train = data_train[target_name]
        X_train = data_train[feature_name_list]

        y_test = data_test[target_name]
        X_test = data_test[feature_name_list]

        data_train = data_train.set_index(["obs_id", "event_id"])
        data_test = data_test.set_index(["obs_id", "event_id"])
    else:
        # if the user wants to use the whole input dataset
        # there is not 'test' data, though we shuffle anyway
        data_train = survived_images
        shuffle(data_train, random_state=0, n_samples=None)
        y_train = data_train[target_name]
        X_train = data_train[feature_name_list]

        data_test = None
        y_test = None
        X_test = None

    return X_train, X_test, y_train, y_test, data_train, data_test


def get_evt_subarray_model_output(
    data,
    weight_name=None,
    keep_cols=["reco_energy"],
    model_output_name="score_img",
    model_output_name_evt="score",
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
        Name of averaged model output (event level)
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
    new_data = new_data.drop(columns=[model_output_name])

    # Remove duplicates
    new_data = new_data[~new_data.index.duplicated(keep="first")]

    return new_data


def get_evt_model_output(
    data_dict,
    weight_name=None,
    reco_energy_label="reco_energy",
    model_output_name="score_img",
    model_output_name_evt="score",
):
    """
    Returns DataStore with reco energy + score/target columns of model at the level-event.

    Parameters
    ----------
    data: `~pandas.DataFrame`
        Data frame with at least (obs_id, evt_id), image score/proba and label
    weight_name: `str`
        Variable name in data frame to weight events with
    reco_energy_label: `list`, optional
        List of variables to keep in resulting data frame
    model_output_name: `str`, optional
        Name of model output (image level)
    model_output_name: `str`, optional
        Name of averaged model output (event level)

    Returns
    -------
    data: `~pandas.DataFrame`
        Data frame

    Warnings
    --------
    Need optimisation, too much loops.
    """

    # Get list of observations
    obs_ids = np.array([])
    for cam_id in data_dict:
        obs_ids = np.hstack((obs_ids, data_dict[cam_id].index.get_level_values(0)))
    obs_ids = np.unique(obs_ids)

    obs_id_list = []
    evt_id_list = []
    model_output_list = []
    reco_energy_list = []
    label_list = []

    # Loop on observation Id
    for obs_id in obs_ids:

        # Get joint list of events
        evt_ids = np.array([])
        for cam_id in data_dict:
            evt_ids = np.hstack(
                (evt_ids, data_dict[cam_id].loc[(obs_id,)].index.get_level_values(0))
            )
        evt_ids = np.unique(evt_ids)

        for evt_id in evt_ids:
            output = np.array([])
            weight = np.array([])
            for cam_id in data_dict:
                try:  # Stack camera information
                    data = data_dict[cam_id].xs(obs_id).xs(evt_id)
                    output = np.hstack((output, data[model_output_name]))
                    weight = np.hstack((weight, data[weight_name]))
                    reco_energy = np.hstack((np.array([]), data[reco_energy_label]))[
                        0
                    ]  # single entry is enough
                    label = np.hstack((np.array([]), data["label"]))[
                        0
                    ]  # single entry is enough
                except:
                    pass  # No information for this type of camera
            obs_id_list.append(obs_id)
            evt_id_list.append(evt_id)
            model_output_list.append(np.sum((output * weight)) / np.sum(weight))
            reco_energy_list.append(reco_energy)
            label_list.append(label)

    data = {
        "obs_id": obs_id_list,
        "event_id": evt_id_list,
        model_output_name_evt: model_output_list,
        reco_energy_label: reco_energy_list,
        "label": label_list,
    }

    return pd.DataFrame(data=data)


def plot_roc_curve(ax, model_output, y, **kwargs):
    """Plot ROC curve for a given set of model outputs and labels"""
    fpr, tpr, _ = roc_curve(y_score=model_output, y_true=y)
    roc_auc = auc(fpr, tpr)
    label = "{} (area={:.2f})".format(kwargs.pop("label"), roc_auc)  # Remove label
    ax.plot(fpr, tpr, label=label, **kwargs)
    return ax


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
    feature_list, data_list, nbin=30, hist_kwargs_list={}, error_kw_list={}, ncols=2
):
    """Plot feature distributions for several data set. Returns list of axes."""
    import matplotlib.pyplot as plt

    n_feature = len(feature_list)
    nrows = (
        int(n_feature / ncols)
        if n_feature % ncols == 0
        else int((n_feature + 1) / ncols)
    )
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5 * ncols, 5 * nrows))
    if nrows == 1 and ncols == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for i, colname in enumerate(feature_list):
        ax = axes[i]

        # Range for binning
        range_min = min([data[colname].min() for data in data_list])
        range_max = max([data[colname].max() for data in data_list])
        myrange = [range_min, range_max]

        for j, data in enumerate(data_list):
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
        ax.legend(loc="upper left")
        ax.grid()
    plt.tight_layout()

    return fig, axes


def plot_profile(ax, data, xcol, ycol, nbin, limit, hist_kwargs={}):
    """Plot profile of a distribution"""
    bin_edges = np.linspace(limit[0], limit[-1], nbin + 1, True)
    bin_center = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    bin_width = bin_edges[1:] - bin_edges[:-1]

    y = []
    yerr = []
    for idx in range(len(bin_center)):
        counts = data[
            (data[xcol] > bin_edges[idx]) & (data[xcol] <= bin_edges[idx + 1])
        ][ycol]
        y.append(counts.mean())
        yerr.append(counts.std() / np.sqrt(len(counts)))

    print(y)
    print(yerr)

    ax.errorbar(x=bin_center, y=y, xerr=bin_width / 2.0, yerr=yerr, **hist_kwargs)

    return ax
