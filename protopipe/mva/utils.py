import numpy as np
import pandas as pd
import pickle
import gzip
from sklearn.metrics import auc, roc_curve


def save_obj(obj, name):
    """Save object in binary"""
    with gzip.open(name, "wb") as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    """Load object in binary"""
    with gzip.open(name, "rb") as f:
        return pickle.load(f)


def prepare_data(ds, cuts, label=None):
    """Add variables in data frame"""
    ds["log10_hillas_intensity"] = np.log10(
        ds["hillas_intensity_reco"]
    )  # THIS SHOULDN'T BE HARDCODED!!!
    ds["log10_impact_dist"] = np.log10(ds["impact_dist"])
    ds["log10_true_energy"] = np.log10(ds["true_energy"])
    try:  # for classification
        ds["log10_reco_energy"] = np.log10(ds["reco_energy"])
        ds["log10_reco_energy_tel"] = np.log10(ds["reco_energy_tel"])
        ds["label"] = np.full(len(ds), label)
    except:
        pass

    ds = ds.query(cuts)

    return ds


def make_cut_list(cuts):
    cut_list = ""
    for idx, cut in enumerate(cuts):
        cut_list += cut
        if idx != len(cuts) - 1:
            cut_list += " and "
    return cut_list


def split_train_test(ds, train_fraction, feature_name_list, target_name):
    # Get the run id corresponding to 70% of the statistics
    # (and sort to avoid troubles...)
    obs_ids = np.sort(pd.unique(ds["obs_id"]))
    max_train_obs_idx = int(train_fraction * len(obs_ids))
    run_max_train = obs_ids[max_train_obs_idx]

    # Split the data for training
    data_train = ds.query("obs_id < {}".format(run_max_train))
    y_train = data_train[target_name]
    X_train = data_train[feature_name_list]
    data_train = data_train.set_index(["obs_id", "event_id"])

    # Split the data for training
    data_test = ds.query("obs_id >= {}".format(run_max_train))
    y_test = data_test[target_name]
    X_test = data_test[feature_name_list]
    data_test = data_test.set_index(["obs_id", "event_id"])

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
    --------
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
    --------
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
