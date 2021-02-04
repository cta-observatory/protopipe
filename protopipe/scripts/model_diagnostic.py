#!/usr/bin/env python

import os
import pandas as pd
import numpy as np
import argparse
from os import path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from astropy.table import Table, Column

import joblib

from protopipe.pipeline.utils import load_config, save_fig

from protopipe.mva import (
    RegressorDiagnostic,
    ClassifierDiagnostic,
    BoostedDecisionTreeDiagnostic,
)
from protopipe.mva.utils import (
    load_obj,
    get_evt_subarray_model_output,
    plot_roc_curve,
    plot_hist,
)


def main():
    # Read arguments
    parser = argparse.ArgumentParser(description="Make diagnostic plot")
    parser.add_argument("--config_file", type=str, required=True)
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
        help="if set, use tail cleaning, otherwise wavelets",
    )
    args = parser.parse_args()

    # Read configuration file
    cfg = load_config(args.config_file)

    model_type = cfg["General"]["model_type"]

    # Import parameters
    indir = cfg["General"]["outdir"]

    cam_ids = cfg["General"]["cam_id_list"]

    # Model
    method_name = cfg["Method"]["name"]
    target_name = cfg["Method"]["target_name"]
    if model_type in "classifier":
        use_proba = cfg["Method"]["use_proba"]

    # Diagnostic
    nbins = cfg["Diagnostic"]["energy"]["nbins"]
    energy_edges = np.logspace(
        np.log10(cfg["Diagnostic"]["energy"]["min"]),
        np.log10(cfg["Diagnostic"]["energy"]["max"]),
        nbins + 1,
        True,
    )

    # Will be further used to get model output of events
    diagnostic = dict()

    for idx, cam_id in enumerate(cam_ids):
        print("### Model diagnostic for {}".format(cam_id))

        # Load data
        data_scikit = load_obj(
            path.join(
                indir,
                "data_scikit_{}_{}_{}_{}.pkl.gz".format(
                    model_type, method_name, args.mode, cam_id
                ),
            )
        )
        data_train = pd.read_pickle(
            path.join(
                indir,
                "data_train_{}_{}_{}_{}.pkl.gz".format(
                    model_type, method_name, args.mode, cam_id
                ),
            )
        )
        data_test = pd.read_pickle(
            path.join(
                indir,
                "data_test_{}_{}_{}_{}.pkl.gz".format(
                    model_type, method_name, args.mode, cam_id
                ),
            )
        )

        # Load model
        outname = "{}_{}_{}_{}.pkl.gz".format(
            model_type, args.mode, cam_id, method_name
        )
        model = joblib.load(path.join(indir, outname))

        outdir = os.path.join(
            indir,
            "diagnostic_{}_{}_{}_{}".format(model_type, method_name, args.mode, cam_id),
        )
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        if model_type in "regressor":
            diagnostic[cam_id] = RegressorDiagnostic(
                model=model,
                feature_name_list=cfg["FeatureList"],
                target_name=target_name,
                data_train=data_train,
                data_test=data_test,
                output_name="reco_energy",
            )
        elif model_type in "classifier":

            if use_proba is True:
                ouput_model_name = "gammaness"
            else:
                ouput_model_name = "score"

            diagnostic[cam_id] = ClassifierDiagnostic(
                model=model,
                feature_name_list=cfg["FeatureList"],
                target_name=target_name,
                data_train=data_train,
                data_test=data_test,
                model_output_name=ouput_model_name,
                is_output_proba=use_proba,
            )

        # Image-level diagnostic - feature importance
        plt.figure(figsize=(5, 5))
        ax = plt.gca()
        ax = diagnostic[cam_id].plot_feature_importance(
            ax,
            **{
                "alpha": 0.7,
                "edgecolor": "black",
                "linewidth": 2,
                "color": "darkgreen",
            },
        )
        ax.set_ylabel("Feature importance")
        ax.grid()
        plt.title(cam_id)
        plt.tight_layout()
        save_fig(outdir, "feature_importances")

        # Diagnostic for regressor
        if model_type in "regressor":

            # Image-level diagnostic[cam_id] - features
            fig, axes = diagnostic[cam_id].plot_features(
                data_list=[data_train, data_test],
                nbin=30,
                hist_kwargs_list=[
                    {
                        "edgecolor": "blue",
                        "color": "blue",
                        "label": "Gamma training",
                        "alpha": 0.2,
                        "fill": True,
                        "ls": "-",
                        "lw": 2,
                    },
                    {
                        "edgecolor": "blue",
                        "color": "blue",
                        "label": "Gamma test",
                        "alpha": 1,
                        "fill": False,
                        "ls": "--",
                        "lw": 2,
                    },
                ],
                error_kw_list=[
                    dict(ecolor="blue", lw=2, capsize=2, capthick=2, alpha=0.2),
                    dict(ecolor="blue", lw=2, capsize=2, capthick=2, alpha=0.2),
                ],
                ncols=3,
            )
            plt.title(cam_id)
            fig.tight_layout()
            save_fig(outdir, "features", fig=fig)

            # Compute averaged energy
            print("Process test sample...")
            data_test_evt = get_evt_subarray_model_output(
                data_test,
                weight_name="hillas_intensity_reco",
                keep_cols=["tel_id", "true_energy"],
                model_output_name="reco_energy_img",
                model_output_name_evt="reco_energy",
            )

            ncols = 5
            nrows = (
                int(nbins / ncols) if nbins % ncols == 0 else int((nbins + 1) / ncols)
            )
            if nrows == 0:
                nrows = 1
                ncols = 1
            fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5 * 5, 10))
            try:
                axes = axes.flatten()
            except:
                axes = [axes]

            bias = []
            resolution = []
            energy_centres = []

            for ibin in range(len(energy_edges) - 1):
                ax = axes[ibin]

                data = data_test_evt.query(
                    "true_energy >= {} and true_energy < {}".format(
                        energy_edges[ibin], energy_edges[ibin + 1]
                    )
                )
                print("Estimate energy for {} evts".format(len(data)))

                er = data["reco_energy"]
                emc = data["true_energy"]

                opt_hist = {
                    "edgecolor": "black",
                    "color": "darkgreen",
                    "label": "data",
                    "alpha": 0.7,
                    "fill": True,
                }
                opt_fit = {"c": "red", "lw": 2, "label": "Best fit"}
                ax, fit_param, cov = diagnostic[cam_id].plot_resolution_distribution(
                    ax=ax,
                    y_true=emc,
                    y_reco=er,
                    nbin=50,
                    fit_range=[-2, 2],
                    hist_kwargs=opt_hist,
                    fit_kwargs=opt_fit,
                )
                if fit_param[2] < 0:  # negative value are allowed for the fit
                    fit_param[2] *= -1

                label = "[{:.2f},{:.2f}] TeV\n#Evts={}\nmean={:.2f}\nstd={:.2f}".format(
                    energy_edges[ibin],
                    energy_edges[ibin + 1],
                    len(er),
                    fit_param[1],
                    fit_param[2],
                )

                ax.set_ylabel("# Evts")
                ax.set_xlabel("(ereco-emc) / emc")
                ax.set_xlim([-2, 2])
                ax.grid()

                evt_patch = mpatches.Patch(color="white", label=label)
                data_patch = mpatches.Patch(color="blue", label="data")
                fit_patch = mpatches.Patch(color="red", label="best fit")
                ax.legend(loc="best", handles=[evt_patch, data_patch, fit_patch])
                plt.tight_layout()

                print(
                    " Fit results: ({:.3f},{:.3f} TeV)".format(
                        energy_edges[ibin], energy_edges[ibin + 1]
                    )
                )

                try:
                    print(" - A    : {:.3f} +/- {:.3f}".format(fit_param[0], cov[0][0]))
                    print(" - mean : {:.3f} +/- {:.3f}".format(fit_param[1], cov[1][1]))
                    print(" - std  : {:.3f} +/- {:.3f}".format(fit_param[2], cov[2][2]))
                except:
                    print(" ==> Problem with fit, no covariance...".format())
                    continue

                bias.append(fit_param[1])
                resolution.append(fit_param[2])
                energy_centres.append(
                    (energy_edges[ibin] + energy_edges[ibin + 1]) / 2.0
                )

            save_fig(outdir, "migration_distribution", fig=fig)

            plt.figure(figsize=(5, 5))
            ax = plt.gca()
            ax.plot(
                energy_centres,
                resolution,
                marker="s",
                color="darkorange",
                label="Resolution",
            )
            ax.plot(energy_centres, bias, marker="s", color="darkgreen", label="Bias")
            ax.set_xlabel("True energy [TeV]")
            ax.set_ylabel("Energy resolution")
            ax.set_xscale("log")
            ax.grid()
            ax.legend()
            ax.set_ylim([-0.2, 1.2])
            plt.title(cam_id)
            plt.tight_layout()
            save_fig(outdir, "energy_resolution")

            # Write results
            t = Table()
            t["ENERGY"] = Column(
                energy_centres, unit="TeV", description="Energy centers"
            )
            t["BIAS"] = Column(bias, unit="", description="Bias from gauusian fit")
            t["RESOL"] = Column(
                bias, unit="", description="Resolution from gauusian fit"
            )
            t.write(
                os.path.join(outdir, "energy_resolution.fits"),
                format="fits",
                overwrite=True,
            )

        elif model_type in "classifier":

            # Image-level diagnostic - features
            fig, axes = diagnostic[cam_id].plot_features(
                data_list=[
                    data_train.query("label==1"),
                    data_test.query("label==1"),
                    data_train.query("label==0"),
                    data_test.query("label==0"),
                ],
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
                ncols=3,
            )
            plt.title(cam_id)
            fig.tight_layout()
            save_fig(outdir, "features", fig=fig)

            if method_name in "AdaBoostClassifier":
                # Image-level diagnostic - method
                plt.figure(figsize=(5, 5))
                ax = plt.gca()
                opt = {"color": "darkgreen", "ls": "-", "lw": 2}
                BoostedDecisionTreeDiagnostic.plot_error_rate(
                    ax, model, data_scikit, **opt
                )
                plt.title(cam_id)
                plt.tight_layout()
                save_fig(path, outdir, "bdt_diagnostic_error_rate")

                plt.figure(figsize=(5, 5))
                ax = plt.gca()
                BoostedDecisionTreeDiagnostic.plot_tree_error_rate(ax, model, **opt)
                plt.title(cam_id)
                plt.tight_layout()
                save_fig(path, outdir, "bdt_diagnostic_tree_error_rate")

            # Image-level diagnostic - model output
            fig, ax = diagnostic[cam_id].plot_image_model_output_distribution(nbin=50)
            ax[0].set_xlim([0, 1])
            plt.title(cam_id)
            fig.tight_layout()
            save_fig(outdir, "image_distribution", fig=fig)

            # Image-level diagnostic - ROC curve on train and test samples
            plt.figure(figsize=(5, 5))
            ax = plt.gca()
            plot_roc_curve(
                ax,
                diagnostic[cam_id].data_train[diagnostic[cam_id].model_output_name],
                diagnostic[cam_id].data_train["label"],
                **dict(color="darkgreen", lw=2, label="Training sample"),
            )
            plot_roc_curve(
                ax,
                data_test[diagnostic[cam_id].model_output_name],
                diagnostic[cam_id].data_test["label"],
                **dict(color="darkorange", lw=2, label="Test sample"),
            )
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
            ax.legend(loc="lower right")
            plt.title(cam_id)
            plt.tight_layout()
            save_fig(outdir, "image_roc_curve")

            # Parameters for energy variation
            cut_list = [
                "reco_energy >= {:.2f} and reco_energy <= {:.2f}".format(
                    energy_edges[i], energy_edges[i + 1]
                )
                for i in range(len(energy_edges) - 1)
            ]

            hist_kwargs_list = [
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
            ]

            error_kw_list = [
                dict(ecolor="blue", lw=2, capsize=3, capthick=2, alpha=0.2),
                dict(ecolor="blue", lw=2, capsize=3, capthick=2, alpha=1),
                dict(ecolor="red", lw=2, capsize=3, capthick=2, alpha=0.2),
                dict(ecolor="red", lw=2, capsize=3, capthick=2, alpha=1),
            ]

            # Image-level diagnostic - model output distribution variation
            n_feature = len(cut_list)
            ncols = 2
            nrows = (
                int(n_feature / ncols)
                if n_feature % ncols == 0
                else int((n_feature + 1) / ncols)
            )
            fig, axes = plt.subplots(
                nrows=nrows, ncols=ncols, figsize=(5 * ncols, 3 * nrows)
            )
            if nrows == 1 and ncols == 1:
                axes = [axes]
            else:
                axes = axes.flatten()

            data_list = [
                data_train.query("label==1"),
                data_test.query("label==1"),
                data_train.query("label==0"),
                data_test.query("label==0"),
            ]

            for i, colname in enumerate(cut_list):
                ax = axes[i]

                # Range for binning
                the_range = [0, 1]

                for j, data in enumerate(data_list):
                    if len(data) == 0:
                        continue

                    ax = plot_hist(
                        ax=ax,
                        data=data.query(cut_list[i])[ouput_model_name],
                        nbin=30,
                        limit=the_range,
                        norm=True,
                        yerr=True,
                        hist_kwargs=hist_kwargs_list[j],
                        error_kw=error_kw_list[j],
                    )

                ax.set_xlim(the_range)
                ax.set_xlabel(ouput_model_name)
                ax.set_ylabel("Arbitrary units")
                ax.legend(loc="best", fontsize="x-small")
                ax.set_title(cut_list[i])
                ax.grid()
            fig.tight_layout()
            save_fig(outdir, "image_distribution_variation", fig=fig)

            # Image-level diagnostic - ROC curve variation on test sample
            plt.figure(figsize=(5, 5))
            ax = plt.gca()

            color = 1.0
            step_color = 1.0 / (len(cut_list))
            for i, cut in enumerate(cut_list):
                c = color - (i + 1) * step_color

                data = data_test.query(cut)
                if len(data) == 0:
                    continue

                opt = dict(
                    color=str(c),
                    lw=2,
                    label="{}".format(cut.replace("reco_energy", "E")),
                )
                plot_roc_curve(ax, data[ouput_model_name], data["label"], **opt)
            ax.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
            ax.set_title(cam_id)
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.legend(loc="lower right", fontsize="x-small")
            plt.tight_layout()
            save_fig(outdir, "image_roc_curve_variation")


if __name__ == "__main__":
    main()
