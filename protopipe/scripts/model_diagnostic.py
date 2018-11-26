#!/usr/bin/env python

import os
import pandas as pd
import numpy as np
import argparse
from os import path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from astropy.table import Table, Column

from sklearn.externals import joblib

from protopipe.pipeline.utils import load_config

from protopipe.mva import (RegressorDiagnostic, ClassifierDiagnostic,
                           BoostedDecisionTreeDiagnostic)
from protopipe.mva.utils import load_obj, get_evt_model_output


def main():
    # Read arguments
    parser = argparse.ArgumentParser(description='Make diagnostic plot')
    parser.add_argument('--config_file', type=str, required=True)
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument('--wave', dest="mode", action='store_const',
                            const="wave", default="tail",
                            help="if set, use wavelet cleaning")
    mode_group.add_argument('--tail', dest="mode", action='store_const',
                            const="tail",
                            help="if set, use tail cleaning, otherwise wavelets")
    args = parser.parse_args()

    # Read configuration file
    cfg = load_config(args.config_file)

    model_type = cfg['General']['model_type']

    # Import parameters
    indir = cfg['General']['outdir']

    cam_ids = cfg['General']['cam_id_list']

    # Model
    method_name = cfg['Method']['name']
    target_name = cfg['Method']['target_name']

    # Diagnostic
    nbins = cfg['Diagnostic']['energy']['nbins']
    energy_edges = np.logspace(
        np.log10(cfg['Diagnostic']['energy']['min']),
        np.log10(cfg['Diagnostic']['energy']['max']),
        nbins + 1,
        True
    )

    for idx, cam_id in enumerate(cam_ids):
        # Load data
        data_scikit = load_obj(
            path.join(indir, 'data_scikit_{}_{}_{}.pkl.gz'.format(model_type, args.mode, cam_id))
        )
        data_train = pd.read_pickle(
            path.join(indir, 'data_train_{}_{}_{}.pkl.gz'.format(model_type, args.mode, cam_id))
        )
        data_test = pd.read_pickle(
            path.join(indir, 'data_test_{}_{}_{}.pkl.gz'.format(model_type, args.mode, cam_id))
        )

        # Load model
        outname = '{}_{}_{}_{}.pkl.gz'.format(model_type, args.mode, cam_id, method_name)
        model = joblib.load(path.join(indir, outname))

        outdir = os.path.join(indir, 'diagnostic_{}_{}_{}'.format(model_type, args.mode, cam_id))
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        if model_type in 'regressor':
            diagnostic = RegressorDiagnostic(
                model=model,
                feature_name_list=cfg['FeatureList'],
                target_name=target_name,
                data_train=data_train,
                data_test=data_test,
                output_name='reco_energy'
            )
        elif model_type in 'classifier':
            diagnostic = ClassifierDiagnostic(
                model=model,
                feature_name_list=cfg['FeatureList'],
                target_name=target_name,
                data_train=data_train,
                data_test=data_test
            )

        # Image-level diagnostic - feature importance
        plt.figure(figsize=(5, 5))
        ax = plt.gca()
        ax = diagnostic.plot_feature_importance(ax, **{'alpha': 0.7, 'edgecolor': 'black',
                                                       'linewidth': 2,
                                                       'color': 'darkgreen'})
        ax.set_ylabel('Feature importance')
        ax.grid()
        plt.title(cam_id)
        plt.tight_layout()
        plt.savefig(path.join(outdir, 'feature_importances.pdf'))

        # Diagnostic for regressor
        if model_type in 'regressor':

            # Image-level diagnostic - features
            fig, axes = diagnostic.plot_features(
                data_list=[data_train, data_test],
                nbin=30,
                hist_kwargs_list=[
                    {'edgecolor': 'blue', 'color': 'blue', 'label': 'Gamma training',
                     'alpha': 0.2, 'fill': True, 'ls': '-', 'lw': 2},
                    {'edgecolor': 'blue', 'color': 'blue', 'label': 'Gamma test',
                     'alpha': 1,
                     'fill': False, 'ls': '--', 'lw': 2}],
                error_kw_list=[
                    dict(ecolor='blue', lw=2, capsize=2, capthick=2, alpha=0.2),
                    dict(ecolor='blue', lw=2, capsize=2, capthick=2, alpha=0.2)],
                ncols=3
            )
            plt.title(cam_id)
            fig.tight_layout()
            fig.savefig(path.join(outdir, 'features.pdf'))

            #from IPython import embed
            #embed()

            # Compute averaged energy
            print('Process test sample...')
            data_test_evt = get_evt_model_output(
                data_test, weight_name='sum_signal_cam',
                keep_cols=['mc_energy'], model_output_name='reco_energy_img',
                model_output_name_evt='reco_energy'
            )

            ncols = 5
            nrows = int(nbins / ncols) if nbins % ncols == 0 else int((nbins + 1) / ncols)
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

                data = data_test_evt.query('mc_energy >= {} and mc_energy < {}'.format(
                    energy_edges[ibin],
                    energy_edges[ibin + 1]
                ))
                print('Estimate energy for {} evts'.format(len(data)))

                er = data['reco_energy']
                emc = data['mc_energy']

                opt_hist = {'edgecolor': 'black', 'color': 'darkgreen',
                            'label': 'data', 'alpha': 0.7, 'fill': True}
                opt_fit = {'c': 'red', 'lw': 2, 'label': 'Best fit'}
                ax, fit_param, cov = diagnostic.plot_resolution_distribution(
                    ax=ax,
                    y_true=emc,
                    y_reco=er,
                    nbin=50,
                    fit_range=[-2, 2],
                    hist_kwargs=opt_hist,
                    fit_kwargs=opt_fit
                )
                if fit_param[2] < 0:  # negative value are allowed for the fit
                    fit_param[2] *= -1


                label = '[{:.2f},{:.2f}] TeV\n#Evts={}\nmean={:.2f}\nstd={:.2f}'.format(
                    energy_edges[ibin],
                    energy_edges[ibin + 1],
                    len(er), fit_param[1], fit_param[2]
                )

                ax.set_ylabel('# Evts')
                ax.set_xlabel('(ereco-emc) / emc')
                ax.set_xlim([-2, 2])
                ax.grid()

                evt_patch = mpatches.Patch(color='white', label=label)
                data_patch = mpatches.Patch(color='blue', label='data')
                fit_patch = mpatches.Patch(color='red', label='best fit')
                ax.legend(loc='best', handles=[evt_patch, data_patch, fit_patch])
                plt.tight_layout()

                print(' Fit results: ({:.3f},{:.3f} TeV)'.format(
                    energy_edges[ibin],
                    energy_edges[ibin + 1])
                )

                try:
                    print(' - A    : {:.3f} +/- {:.3f}'.format(fit_param[0], cov[0][0]))
                    print(' - mean : {:.3f} +/- {:.3f}'.format(fit_param[1], cov[1][1]))
                    print(' - std  : {:.3f} +/- {:.3f}'.format(fit_param[2], cov[2][2]))
                except:
                    print(' ==> Problem with fit, no covariance...'.format())
                    continue

                bias.append(fit_param[1])
                resolution.append(fit_param[2])
                energy_centres.append((energy_edges[ibin] + energy_edges[ibin + 1]) / 2.)

            plt.savefig(path.join(outdir, 'migration_distribution.pdf'))

            plt.figure(figsize=(5, 5))
            ax = plt.gca()
            ax.plot(energy_centres, resolution, marker='s', color='darkorange', label='Resolution')
            ax.plot(energy_centres, bias, marker='s', color='darkgreen', label='Bias')
            ax.set_xlabel('True energy [TeV]')
            ax.set_ylabel('Energy resolution')
            ax.set_xscale('log')
            ax.grid()
            ax.legend()
            plt.title(cam_id)
            plt.tight_layout()
            plt.savefig(path.join(outdir, 'energy_resolution.pdf'))

            # Write results
            t = Table()
            t['ENERGY'] = Column(energy_centres, unit='TeV', description='Energy centers')
            t['BIAS'] = Column(bias, unit='', description='Bias from gauusian fit')
            t['RESOL'] = Column(bias, unit='', description='Resolution from gauusian fit')
            t.write(os.path.join(outdir, 'energy_resolution.fits'), format='fits', overwrite=True)


        elif model_type in 'classifier':

            # Image-level diagnostic - features
            fig, axes = diagnostic.plot_features(
                data_list=[data_train.query('label==1'), data_test.query('label==1'),
                           data_train.query('label==0'), data_test.query('label==0')],
                nbin=30,
                hist_kwargs_list=[
                    {'edgecolor': 'blue', 'color': 'blue', 'label': 'Gamma training sample',
                     'alpha': 0.2, 'fill': True, 'ls': '-', 'lw': 2},
                    {'edgecolor': 'blue', 'color': 'blue', 'label': 'Gamma test sample',
                     'alpha': 1, 'fill': False, 'ls': '--', 'lw': 2},
                    {'edgecolor': 'red', 'color': 'red', 'label': 'Proton training sample',
                     'alpha': 0.2, 'fill': True, 'ls': '-', 'lw': 2},
                    {'edgecolor': 'red', 'color': 'red', 'label': 'Proton test sample',
                     'alpha': 1, 'fill': False, 'ls': '--', 'lw': 2}
                ],
                error_kw_list=[
                    dict(ecolor='blue', lw=2, capsize=3, capthick=2, alpha=0.2),
                    dict(ecolor='blue', lw=2, capsize=3, capthick=2, alpha=1),
                    dict(ecolor='red', lw=2, capsize=3, capthick=2, alpha=0.2),
                    dict(ecolor='red', lw=2, capsize=3, capthick=2, alpha=1)
                ],
                ncols=3
            )
            plt.title(cam_id)
            fig.tight_layout()
            fig.savefig(path.join(outdir, 'features.pdf'))

            # Image-level diagnostic - method
            plt.figure(figsize=(5,5))
            ax = plt.gca()
            opt = {'color': 'darkgreen', 'ls': '-', 'lw': 2}
            BoostedDecisionTreeDiagnostic.plot_error_rate(ax, model, data_scikit, **opt)
            plt.title(cam_id)
            plt.tight_layout()
            plt.savefig(os.path.join(outdir, 'bdt_diagnostic_error_rate.pdf'))

            plt.figure(figsize=(5,5))
            ax = plt.gca()
            BoostedDecisionTreeDiagnostic.plot_tree_error_rate(ax, model, **opt)
            plt.title(cam_id)
            plt.tight_layout()
            plt.savefig(os.path.join(outdir, 'bdt_diagnostic_tree_error_rate.pdf'))

            # Image-level diagnostic - score
            fig, ax = diagnostic.plot_image_score_distribution(nbin=50)
            plt.title(cam_id)
            fig.tight_layout()
            fig.savefig(os.path.join(outdir, 'image_score_distribution.pdf'))

            # Image-level diagnostic - ROC curve
            plt.figure(figsize=(5,5))
            ax = plt.gca()
            diagnostic.plot_roc_curve(ax, **dict(color='darkorange', lw=2))
            ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            ax.legend(loc='lower right')
            plt.title(cam_id)
            plt.tight_layout()
            plt.savefig(os.path.join(outdir, 'image_roc_curve.pdf'))

            # Compute average score - event-wise
            print('Process training sample...')
            data_train_evt = get_evt_model_output(
                data_train, weight_name='sum_signal_cam',
                keep_cols=['label', 'reco_energy'], model_output_name='score_img',
                model_output_name_evt='score'
            )
            print('Process test sample...')
            data_test_evt = get_evt_model_output(
                data_test, weight_name='sum_signal_cam',
                keep_cols=['label', 'reco_energy'], model_output_name='score_img',
                model_output_name_evt='score'
            )

            # Event-level diagnostic - score distribution
            fig, ax = diagnostic.plot_evt_score_distribution(data_train=data_train_evt,
                                                             data_test=data_test_evt,
                                                             nbin=50)
            plt.title(cam_id)
            fig.tight_layout()
            fig.savefig(os.path.join(outdir, 'evt_score_distribution.pdf'))

            # Event-level diagnostic - score distribution variation
            cut_list = ['reco_energy >= {:.2f} and reco_energy <= {:.2f}'.format(
                energy_edges[i],
                energy_edges[i+1]
            ) for i in range(len(energy_edges) - 1)]

            fig, ax = diagnostic.plot_evt_score_distribution_variation(
                data_train_evt,
                data_test_evt,
                cut_list,
                nbin=50,
                ncols=2
            )
            for x in ax:
                x.set_xlim([-0.5, 0.5])
            fig.tight_layout()
            fig.savefig(path.join(outdir, 'evt_score_distribution_energy.pdf'))

            # Event-level diagnostic - ROC curve variation
            plt.figure(figsize=(5,5))
            ax = plt.gca()
            diagnostic.plot_evt_roc_curve_variation(ax, data_test_evt, cut_list)
            ax.set_title(cam_id)
            plt.tight_layout()
            plt.savefig(os.path.join(outdir, 'evt_roc_curve_variation.pdf'))


if __name__ == '__main__':
    main()

