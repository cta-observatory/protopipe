#!/usr/bin/env python

import os
import pandas as pd
import argparse
from os import path
from sklearn.ensemble import (
    AdaBoostRegressor,
    AdaBoostClassifier,
    RandomForestClassifier,
)
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
import joblib
from sklearn.metrics import classification_report
from sklearn.calibration import CalibratedClassifierCV

from protopipe.pipeline.utils import load_config, get_camera_names

from protopipe.mva import TrainModel
from protopipe.mva.utils import make_cut_list, prepare_data, save_obj


def main():

    # Read arguments
    parser = argparse.ArgumentParser(
        description="Build model for regression/classification"
    )
    parser.add_argument("--config_file", type=str, required=True)
    parser.add_argument(
        "--max_events",
        type=int,
        default=None,
        help="maximum number of events for training",
    )
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

    # These last CL arguments can overwrite the values from the config

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--cameras_from_config',
                       action='store_true',
                       help="Get cameras configuration file (Priority 1)",)
    group.add_argument('--cameras_from_file',
                       action='store_true',
                       help="Get cameras from input file (Priority 2)",)
    group.add_argument('--cam_id_list',
                       type=str,
                       default=None,
                       help="Select cameras like 'LSTCam CHEC' (Priority 3)",)

    parser.add_argument(
        "-i",
        "--indir",
        type=str,
        default=None,
        help="Directory containing the required input file(s)"
    )
    parser.add_argument(
        "--infile_signal",
        type=str,
        default=None,
        help="SIGNAL file (default: read from config file)",
    )
    parser.add_argument(
        "--infile_background",
        type=str,
        default=None,
        help="BACKGROUND file (default: read from config file)",
    )
    parser.add_argument("-o", "--outdir", type=str, default=None)

    args = parser.parse_args()

    # Read configuration file
    cfg = load_config(args.config_file)

    # Type of model (regressor or classifier)
    model_type = cfg["General"]["model_type"]

    # Import parameters
    if args.indir is None:
        data_dir = cfg["General"]["data_dir"]
    else:
        data_dir = args.indir

    if args.outdir is None:
        outdir = cfg["General"]["outdir"]
    else:
        outdir = args.outdir
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    table_name_template = cfg["General"]["table_name_template"]

    # List of features
    feature_list = cfg["FeatureList"]

    # Optimisation parameters
    method_name = cfg["Method"]["name"]
    tuned_parameters = [cfg["Method"]["tuned_parameters"]]
    scoring = cfg["Method"]["scoring"]
    cv = cfg["Method"]["cv"]

    # Split fraction
    train_fraction = cfg["Split"]["train_fraction"]

    if model_type in "regressor":

        if args.infile_signal is None:
            data_file = cfg["General"]["data_file"].format(args.mode)
        else:
            data_file = args.infile_signal

        filename = path.join(data_dir, data_file)

        if args.cameras_from_config:
            cam_ids = cfg["General"]["cam_id_list"]
        elif args.cameras_from_file:
            cam_ids = get_camera_names(filename)
        else:
            cam_ids = args.cam_id_list.split()

        table_name = [table_name_template + cam_id for cam_id in cam_ids]

        # List of cuts
        cuts = make_cut_list(cfg["SigFiducialCuts"])
        init_model = AdaBoostRegressor(DecisionTreeRegressor(max_depth=None))

        # Name of target
        target_name = cfg["Method"]["target_name"]

    elif model_type in "classifier":

        # read signal file from either config file or CLI
        if args.infile_signal is None:
            data_sig_file = cfg["General"]["data_sig_file"].format(args.mode)
        else:
            data_sig_file = args.infile_signal

        # read background file from either config file or CLI
        if args.infile_background is None:
            data_bkg_file = cfg["General"]["data_bkg_file"].format(args.mode)
        else:
            data_bkg_file = args.infile_background

        filename_sig = path.join(data_dir, data_sig_file)
        filename_bkg = path.join(data_dir, data_bkg_file)

        if args.cameras_from_config:
            print("TAKING CAMERAS FROM CONFIG")
            cam_ids = cfg["General"]["cam_id_list"]
        elif args.cameras_from_file:
            print("TAKING CAMERAS FROM TRAINING FILE")
            # in the same analysis all particle types are analyzed in the
            # same way so we can just use gammas
            cam_ids = get_camera_names(filename_sig)
        else:
            print("TAKING CAMERAS FROM CLI")
            cam_ids = args.cam_id_lists.split()

        table_name = [table_name_template + cam_id for cam_id in cam_ids]

        # List of cuts
        sig_cuts = make_cut_list(cfg["SigFiducialCuts"])
        bkg_cuts = make_cut_list(cfg["BkgFiducialCuts"])

        # Model
        if method_name in "AdaBoostClassifier":
            init_model = AdaBoostClassifier(DecisionTreeClassifier(max_depth=4))
        elif method_name in "RandomForestClassifier":
            init_model = RandomForestClassifier(
                n_estimators=500,
                max_depth=None,
                min_samples_split=0.05,
                max_features="sqrt",
                bootstrap=True,
                random_state=None,
                criterion="gini",
                class_weight="balanced_subsample",  # Tree-wise re-weighting
            )

            # Name of target
            target_name = cfg["Method"]["target_name"]

        use_same_number_of_sig_and_bkg_for_training = cfg["Split"][
            "use_same_number_of_sig_and_bkg_for_training"
        ]

    print("### Using {} for model construction".format(method_name))

    print(f"LIST OF CAMERAS TO USE = {cam_ids}")

    models = dict()
    for idx, cam_id in enumerate(cam_ids):

        print("### Building model for {}".format(cam_id))

        if model_type in "regressor":
            # Load data
            data = pd.read_hdf(filename, table_name[idx], mode="r")
            data = prepare_data(ds=data, cuts=cuts)[0:args.max_events]

            print(f"Going to split {len(data)} SIGNAL images...")

            # Init model factory
            factory = TrainModel(
                case=model_type, target_name=target_name, feature_name_list=feature_list
            )

            # Split data
            factory.split_data(data_sig=data, train_fraction=train_fraction)
            print("Training sample: sig {}".format(len(factory.data_train)))
            print("Test sample: sig {}".format(len(factory.data_test)))
        elif model_type in "classifier":
            # Load data
            data_sig = pd.read_hdf(filename_sig, table_name[idx], mode="r")
            data_bkg = pd.read_hdf(filename_bkg, table_name[idx], mode="r")

            # Add label
            data_sig = prepare_data(ds=data_sig, label=1, cuts=sig_cuts)
            data_bkg = prepare_data(ds=data_bkg, label=0, cuts=bkg_cuts)

            if args.max_events:
                data_sig = data_sig[0:(args.max_events - 1)]
                data_bkg = data_bkg[0:(args.max_events - 1)]

            print(f"Going to split {len(data_sig)} SIGNAL images and {len(data_bkg)} BACKGROUND images")

            # Init model factory
            factory = TrainModel(
                case=model_type, target_name=target_name, feature_name_list=feature_list
            )

            # Split data
            factory.split_data(
                data_sig=data_sig,
                data_bkg=data_bkg,
                train_fraction=train_fraction,
                force_same_nsig_nbkg=use_same_number_of_sig_and_bkg_for_training,
            )

            print(
                "Training sample: sig {} and bkg {}".format(
                    len(factory.data_train.query("label==1")),
                    len(factory.data_train.query("label==0")),
                )
            )
            print(
                "Test sample: sig {} and bkg {}".format(
                    len(factory.data_test.query("label==1")),
                    len(factory.data_test.query("label==0")),
                )
            )

        # Build model
        best_model = factory.get_optimal_model(
            init_model, tuned_parameters, scoring=scoring, cv=cv
        )

        if model_type in "classifier":
            # print report
            if model_type in "classifier":
                print(
                    classification_report(
                        factory.data_scikit["y_test"],
                        best_model.predict(factory.data_scikit["X_test"]),
                    )
                )

            # Calibrate model if necessary on test data
            if cfg["Method"]["calibrate_output"] is True:
                print("==> Calibrate classifier...")

                best_model = CalibratedClassifierCV(
                    best_model, method="sigmoid", cv="prefit"
                )

                best_model.fit(
                    factory.data_scikit["X_test"], factory.data_scikit["y_test"]
                )

        # save model
        models[cam_id] = best_model
        outname = "{}_{}_{}_{}.pkl.gz".format(
            model_type, args.mode, cam_id, method_name
        )
        joblib.dump(best_model, path.join(outdir, outname))

        # save data
        save_obj(
            factory.data_scikit,
            path.join(
                outdir,
                "data_scikit_{}_{}_{}_{}.pkl.gz".format(
                    model_type, method_name, args.mode, cam_id
                ),
            ),
        )
        factory.data_train.to_pickle(
            path.join(
                outdir,
                "data_train_{}_{}_{}_{}.pkl.gz".format(
                    model_type, method_name, args.mode, cam_id
                ),
            )
        )
        factory.data_test.to_pickle(
            path.join(
                outdir,
                "data_test_{}_{}_{}_{}.pkl.gz".format(
                    model_type, method_name, args.mode, cam_id
                ),
            )
        )


if __name__ == "__main__":
    main()
