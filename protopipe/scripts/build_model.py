#!/usr/bin/env python

import os
from os import path
import importlib

import pandas as pd
from sklearn.ensemble import (
    AdaBoostRegressor,
    AdaBoostClassifier,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.calibration import CalibratedClassifierCV

from protopipe.pipeline.utils import load_config, get_camera_names
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

    # Get file containing gammas (signal)
    if args.infile_signal is None:
        data_sig_file = cfg["General"]["data_file"].format(args.mode)
    else:
        data_sig_file = args.infile_signal

    filename_sig = path.join(data_dir, data_sig_file)

    # Cameras to use
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

    # The names of the tables inside the HDF5 file are the camera's names
    table_name = [cam_id for cam_id in cam_ids]

    # Dataset split train-test fraction
    train_fraction = cfg["Split"]["train_fraction"]
    # Name of target quantity
    target_name = cfg["Method"]["target_name"]

    # List of features
    feature_list = cfg["FeatureList"]

    # GridSearchCV
    use_GridSearchCV = cfg["GridSearchCV"]["use"]
    scoring = cfg["GridSearchCV"]["scoring"]
    cv = cfg["GridSearchCV"]["cv"]

    # This is a list of dictionaries to generalize for GridSearchCV
    # The initial model will be always initialized with the first set of
    # tuned_parameters
    tuned_parameters = [cfg["Method"]["tuned_parameters"]]

    # Initialize the model dynamically
    model_to_use = cfg['Method']['name']
    module_name = '.'.join(model_to_use.split('.', 2)[:-1])
    class_name = model_to_use.split('.')[-1]
    module = importlib.import_module(module_name)  # sklearn.XXX
    model = getattr(module, class_name)
    initialized_model = model(**cfg['Method']['tuned_parameters'])

    # Map model types to the models supported by the script
    model_type = {"regressor": ["RandomForestRegressor",
                                "AdaBoostRegressor"],
                  "classifier": ["RandomForestClassifier",
                                 "AdaBoostClassifier"]}

    # Hyper-parameters based on type of model

    # if method_name in ["RandomForestRegressor", "RandomForestClassifier"]:
    #     n_estimators = tuned_parameters[0]["n_estimators"]
    #     criterion = tuned_parameters[0]["criterion"]
    #     max_depth = None if tuned_parameters[0]["max_depth"] == "None" else tuned_parameters[0]["max_depth"]
    #     min_samples_split = tuned_parameters[0]["min_samples_split"]
    #     min_samples_leaf = tuned_parameters[0]["min_samples_leaf"]
    #     min_weight_fraction_leaf = tuned_parameters[0]["min_weight_fraction_leaf"]
    #     max_features = tuned_parameters[0]["max_features"]
    #     max_leaf_nodes = None if tuned_parameters[0]["max_leaf_nodes"] == "None" else tuned_parameters[0]["max_leaf_nodes"]
    #     min_impurity_decrease = tuned_parameters[0]["min_impurity_decrease"]
    #     bootstrap = False if tuned_parameters[0]["bootstrap"] == "False" else True
    #     oob_score = False if tuned_parameters[0]["oob_score"] == "False" else True
    #     n_jobs = None if tuned_parameters[0]["n_jobs"] == "None" else tuned_parameters[0]["n_jobs"]
    #     random_state = None if tuned_parameters[0]["random_state"] == "None" else tuned_parameters[0]["random_state"]
    #     verbose = tuned_parameters[0]["verbose"]
    #     warm_start = False if tuned_parameters[0]["warm_start"] == "False" else True
    #     ccp_alpha = tuned_parameters[0]["ccp_alpha"]
    #     max_samples = None if tuned_parameters[0]["max_samples"] == "None" else tuned_parameters[0]["max_samples"]
    # 
    # if method_name == 'AdaBoostRegressor':
    #     base_estimator = 'None'  # (aka DecisionTreeRegressor)
    #     n_estimators = cfg["Method"]["n_estimators"]
    #     learning_rate = cfg["Method"]["learning_rate"]
    #     random_state = None if cfg["Method"]["random_state"] == "None" else cfg["Method"]["random_state"]
    # 
    # if method_name == 'RandomForestClassifier':
    #     class_weight = cfg["Method"]["class_weight"]

    if class_name in model_type["regressor"]:

        # Get the selection cuts
        cuts = make_cut_list(cfg["SigFiducialCuts"])

        # # # Initialize the model
        # if method_name in "AdaBoostRegressor":
        #     init_model = AdaBoostRegressor(
        #         base_estimator=DecisionTreeRegressor(max_depth=None)
        #         )
        # if method_name in "RandomForestRegressor":
        #     init_model = RandomForestRegressor(
        #         n_estimators=n_estimators,
        #         criterion=criterion,
        #         max_depth=max_depth,
        #         min_samples_split=min_samples_split,
        #         min_samples_leaf=min_samples_leaf,
        #         min_weight_fraction_leaf=min_weight_fraction_leaf,
        #         max_features=max_features,
        #         max_leaf_nodes=max_leaf_nodes,
        #         min_impurity_decrease=min_impurity_decrease,
        #         bootstrap=bootstrap,
        #         oob_score=oob_score,
        #         n_jobs=n_jobs,
        #         random_state=random_state,
        #         verbose=verbose,
        #         warm_start=warm_start,
        #         ccp_alpha=ccp_alpha,
        #         max_samples=max_samples
        #     )
        # else:
        #     print("ERROR: we support only AdaBoostRegressor and RandomForestRegressor at the moment!")
        #     exit()

    elif class_name in model_type["classifier"]:

        # # read signal file from either config file or CLI
        # if args.infile_signal is None:
        #     data_sig_file = cfg["General"]["data_sig_file"].format(args.mode)
        # else:
        #     data_sig_file = args.infile_signal

        # read background file from either config file or CLI
        if args.infile_background is None:
            data_bkg_file = cfg["General"]["data_bkg_file"].format(args.mode)
        else:
            data_bkg_file = args.infile_background

        # filename_sig = path.join(data_dir, data_sig_file)
        filename_bkg = path.join(data_dir, data_bkg_file)

        # table_name = [table_name_template + cam_id for cam_id in cam_ids]

        # Get the selection cuts
        sig_cuts = make_cut_list(cfg["SigFiducialCuts"])
        bkg_cuts = make_cut_list(cfg["BkgFiducialCuts"])

        # # Initialize the the base estimator
        # if method_name in "AdaBoostClassifier":
        #     init_model = AdaBoostClassifier(DecisionTreeClassifier(max_depth=4))
        # elif method_name in "RandomForestClassifier":
        #     init_model = RandomForestClassifier(
        #         n_estimators=n_estimators,
        #         criterion=criterion,
        #         max_depth=max_depth,
        #         min_samples_split=min_samples_split,
        #         min_samples_leaf=min_samples_leaf,
        #         min_weight_fraction_leaf=min_weight_fraction_leaf,
        #         max_features=max_features,
        #         max_leaf_nodes=max_leaf_nodes,
        #         min_impurity_decrease=min_impurity_decrease,
        #         bootstrap=bootstrap,
        #         oob_score=oob_score,
        #         n_jobs=n_jobs,
        #         random_state=random_state,
        #         verbose=verbose,
        #         warm_start=warm_start,
        #         class_weight=class_weight,
        #         ccp_alpha=ccp_alpha,
        #         max_samples=max_samples
        #     )

            # # Name of target
            # target_name = cfg["Method"]["target_name"]

        use_same_number_of_sig_and_bkg_for_training = cfg["Split"][
            "use_same_number_of_sig_and_bkg_for_training"
        ]

    print("### Using {} for model construction".format(model_to_use))

    print(f"LIST OF CAMERAS TO USE = {cam_ids}")

    models = dict()
    for idx, cam_id in enumerate(cam_ids):

        print("### Building model for {}".format(cam_id))

        if class_name in model_type["regressor"]:

            # Load data
            data = pd.read_hdf(filename_sig, table_name[idx], mode="r")
            data = prepare_data(ds=data, cuts=cuts)[0:args.max_events]

            print(f"Going to split {len(data)} SIGNAL images...")

            # Initialize the model
            factory = TrainModel(
                case="regressor",
                target_name=target_name,
                feature_name_list=feature_list
            )

            # Split the TRAINING dataset in a train and test sub-datasets
            # Useful to test the models before using them for DL2 production
            factory.split_data(data_sig=data, train_fraction=train_fraction)
            print("Training sample: sig {}".format(len(factory.data_train)))
            print("Test sample: sig {}".format(len(factory.data_test)))

        else:  # if it's not a regressor it's a classifier

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
            # Apply optimization of the hyper-parameters via grid search
            # and return best model
            best_model = factory.get_optimal_model(
                initialized_model, tuned_parameters, scoring=scoring, cv=cv
            )
        else:  # otherwise use directly the initial model
            best_model = initialized_model

            # Fit the chosen model on the train data
            best_model.fit(
                factory.data_scikit["X_train"],
                factory.data_scikit["y_train"],
                sample_weight=factory.data_scikit["w_train"],
            )

        if class_name in model_type["classifier"]:

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
                    model_type,
                    class_name,
                    outdir)


if __name__ == "__main__":
    main()
