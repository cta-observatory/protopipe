import os
from os import path
import importlib

import pandas as pd

from sklearn.metrics import classification_report
from sklearn.calibration import CalibratedClassifierCV

from protopipe.pipeline.io import load_config, get_camera_names
from protopipe.mva import TrainModel
from protopipe.mva.io import initialize_script_arguments, save_output
from protopipe.mva.utils import (
    make_cut_list,
    prepare_data
)


def main():

    # INITIALIZE CLI arguments
    args = initialize_script_arguments()

    # LOAD CONFIGURATION FILE
    cfg = load_config(args.config_file)

    # INPUT CONFIGURATION

    # Import parameters
    if args.indir_signal is None:
        data_dir_signal = cfg["General"]["data_dir_signal"]
    else:
        data_dir_signal = args.indir_signal

    if args.outdir is None:
        outdir = cfg["General"]["outdir"]
    else:
        outdir = args.outdir
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # Get file containing gammas (signal)
    if args.infile_signal is None:
        data_sig_file = cfg["General"]["data_sig_file"].format(args.mode)
    else:
        data_sig_file = args.infile_signal

    filename_sig = path.join(data_dir_signal, data_sig_file)

    print(f"INPUT SIGNAL FILE PATH= {filename_sig}")

    # Cameras to use
    if args.cameras_from_config:
        print("GETTING CAMERAS FROM CONFIGURATION FILE")
        cam_ids = cfg["General"]["cam_id_list"]
    elif args.cameras_from_file:
        print("GETTING CAMERAS FROM SIGNAL TRAINING FILE")
        # in the same analysis all particle types are analyzed in the
        # same way so we can just use signal
        cam_ids = get_camera_names(data_dir_signal, data_sig_file)
    else:
        print("GETTING CAMERAS FROM CLI")
        cam_ids = args.cam_id_list.split()

    # The names of the tables inside the HDF5 file are the camera's names
    table_name = [cam_id for cam_id in cam_ids]

    # Dataset split train-test fraction
    train_fraction = cfg["Split"]["train_fraction"]
    # Name of target quantity
    target_name = cfg["Method"]["target_name"]

    # Get list of features
    features_basic = cfg["FeatureList"]["Basic"]
    features_derived = cfg["FeatureList"]["Derived"]
    feature_list = features_basic + list(features_derived)
    print("Going to use the following features to train the model:")
    print(feature_list)
    # sort features_to_use alphabetically to ensure order
    # preservation with model.predict in protopipe.scripts
    feature_list = sorted(feature_list)

    # GridSearchCV
    use_GridSearchCV = cfg["GridSearchCV"]["use"]
    scoring = cfg["GridSearchCV"]["scoring"]
    cv = cfg["GridSearchCV"]["cv"]
    refit = cfg["GridSearchCV"]["refit"]
    grid_search_verbose = cfg["GridSearchCV"]["verbose"]
    grid_search_njobs = cfg["GridSearchCV"]["njobs"]

    # Hyper-parameters of the main model
    tuned_parameters = cfg["Method"]["tuned_parameters"]

    # Initialize the model dynamically

    # There always at least one (main) model to initialize
    model_to_use = cfg['Method']['name']
    module_name = '.'.join(model_to_use.split('.', 2)[:-1])
    class_name = model_to_use.split('.')[-1]
    module = importlib.import_module(module_name)  # sklearn.XXX
    model = getattr(module, class_name)

    # Check for any base estimator if main model is a meta-estimator
    if "base_estimator" in cfg['Method']:
        base_estimator_cfg = cfg['Method']['base_estimator']
        base_estimator_name = base_estimator_cfg['name']
        base_estimator_pars = base_estimator_cfg['parameters']
        base_estimator_module_name = '.'.join(base_estimator_name.split('.', 2)[:-1])
        base_estimator_class_name = base_estimator_name.split('.')[-1]
        base_estimator_module = importlib.import_module(
            base_estimator_module_name)  # sklearn.XXX
        base_estimator_model = getattr(base_estimator_module, base_estimator_class_name)
        initialized_base_estimator = base_estimator_model(**base_estimator_pars)
        print(f"...based on {base_estimator_module_name}.{base_estimator_class_name}")
        initialized_model = model(base_estimator=initialized_base_estimator,
                                  **cfg['Method']['tuned_parameters'])
    else:
        initialized_model = model(**cfg['Method']['tuned_parameters'])

    # Map model types to the models supported by the script
    model_types = {"regressor": ["RandomForestRegressor",
                                 "AdaBoostRegressor"],
                   "classifier": ["RandomForestClassifier"]}

    if class_name in model_types["regressor"]:

        try:
            log_10_target = cfg["Method"]["log_10_target"]
        except KeyError:
            log_10_target = True

        if log_10_target:
            target_name = f"log10_{target_name}"

        # Get the selection cuts
        cuts = make_cut_list(cfg["SigFiducialCuts"])

    elif class_name in model_types["classifier"]:

        if args.indir_background is None:
            data_dir_background = cfg["General"]["data_dir_background"]
        else:
            data_dir_background = args.indir_background

        # read background file from either config file or CLI
        if args.infile_background is None:
            data_bkg_file = cfg["General"]["data_bkg_file"].format(args.mode)
        else:
            data_bkg_file = args.infile_background

        filename_bkg = path.join(data_dir_background, data_bkg_file)

        # Get the selection cuts
        sig_cuts = make_cut_list(cfg["SigFiducialCuts"])
        bkg_cuts = make_cut_list(cfg["BkgFiducialCuts"])

        use_same_number_of_sig_and_bkg_for_training = cfg["Split"][
            "use_same_number_of_sig_and_bkg_for_training"
        ]

    else:
        raise ValueError("ERROR: not a supported model")

    print(f"Using {module_name}.{class_name} for model construction")

    print(f"LIST OF CAMERAS TO USE = {cam_ids}")

    models = dict()
    for idx, cam_id in enumerate(cam_ids):

        print("### Building model for {}".format(cam_id))

        if class_name in model_types["regressor"]:

            # Load data
            data_sig = pd.read_hdf(filename_sig, table_name[idx], mode="r")
            # Add any derived feature and apply fiducial cuts
            data_sig = prepare_data(ds=data_sig,
                                    derived_features=features_derived,
                                    select_data=True,
                                    cuts=cuts)

            if args.max_events:
                data_sig = data_sig[0:args.max_events]

            print(f"Going to split {len(data_sig)} SIGNAL images...")

            # Initialize the model
            factory = TrainModel(
                case="regressor",
                target_name=target_name,
                feature_name_list=feature_list
            )

            # Split the TRAINING dataset in a train and test sub-datasets
            # Useful to test the models before using them for DL2 production
            factory.split_data(data_sig=data_sig, train_fraction=train_fraction)
            print("Training sample: sig {}".format(len(factory.data_train)))
            if factory.data_test is not None:
                print("Test sample: sig {}".format(len(factory.data_test)))

        else:  # if it's not a regressor it's a classifier

            # Load data
            data_sig = pd.read_hdf(filename_sig, table_name[idx], mode="r")
            data_bkg = pd.read_hdf(filename_bkg, table_name[idx], mode="r")

            # Add label
            data_sig = prepare_data(ds=data_sig,
                                    label=1,
                                    cuts=sig_cuts,
                                    select_data=True,
                                    derived_features=features_derived)
            data_bkg = prepare_data(ds=data_bkg,
                                    label=0,
                                    cuts=bkg_cuts,
                                    select_data=True,
                                    derived_features=features_derived)

            if args.max_events:
                data_sig = data_sig[0:args.max_events]
                data_bkg = data_bkg[0:args.max_events]

            print(
                f"Going to split {len(data_sig)} SIGNAL images and {len(data_bkg)} BACKGROUND images")

            # Initialize the model
            factory = TrainModel(
                case="classifier", target_name=target_name, feature_name_list=feature_list
            )

            # Split the TRAINING dataset in a train and test sub-datasets
            # Useful to test the models before using them for DL2 production
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

        if use_GridSearchCV:
            print("Going to perform exhaustive cross-validated grid-search over"
                  " specified parameter values...")
            # Apply optimization of the hyper-parameters via grid search
            # and return best model
            best_model = factory.get_optimal_model(
                initialized_model, tuned_parameters, scoring=scoring, cv=cv,
                refit=refit, verbose=grid_search_verbose, njobs=grid_search_njobs)
        else:  # otherwise use directly the initial model
            best_model = initialized_model

            # Fit the chosen model on the train data
            best_model.fit(
                factory.data_scikit["X_train"],
                factory.data_scikit["y_train"],
                sample_weight=factory.data_scikit["w_train"],
            )

        if class_name in model_types["classifier"]:

            print(
                classification_report(
                    factory.data_scikit["y_test"],
                    best_model.predict(factory.data_scikit["X_test"]),
                )
            )

            # Calibrate model if necessary on test data (GridSearchCV)
            if use_GridSearchCV and cfg["Method"]["calibrate_output"]:
                print("==> Calibrate classifier...")

                best_model = CalibratedClassifierCV(
                    best_model, method="sigmoid", cv="prefit"
                )

                best_model.fit(
                    factory.data_scikit["X_test"], factory.data_scikit["y_test"]
                )

        save_output(models,
                    cam_id,
                    factory,
                    best_model,
                    model_types,
                    class_name,
                    outdir)


if __name__ == "__main__":
    main()
