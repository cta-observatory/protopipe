#!/usr/bin/env python

import os
import pandas as pd
from os import path
from sklearn.ensemble import (
    AdaBoostRegressor,
    AdaBoostClassifier,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.calibration import CalibratedClassifierCV

from protopipe.pipeline.utils import load_config

from protopipe.mva import TrainModel
from protopipe.mva.io import initialize_script_arguments, save_output
from protopipe.mva.utils import (
    make_cut_list,
    prepare_data
)


def main():

    args = initialize_script_arguments()

    # LOAD CONFIGURATION FILE
    cfg = load_config(args.config_file)

    # INPUT CONFIGURATION

    # I/O settings
    data_dir = cfg["General"]["data_dir"]
    outdir = cfg["General"]["outdir"]
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    # Types of cameras
    cam_ids = cfg["General"]["cam_id_list"]
    table_name = [cam_id for cam_id in cam_ids]
    # Type of model (regression or classification)
    model_type = cfg["General"]["model_type"]
    # Dataset split train-test fraction
    train_fraction = cfg["Split"]["train_fraction"]
    # List of features
    feature_list = cfg["FeatureList"]
    method_name = cfg["Method"]["name"]

    # GridSearchCV
    use_GridSearchCV = cfg["GridSearchCV"]["use"]
    scoring = cfg["GridSearchCV"]["scoring"]
    cv = cfg["GridSearchCV"]["cv"]

    # This is a list of dictionaries to generalize for GridSearchCV
    # The initial model will be always initialized with the first set of
    # tuned_parameters
    tuned_parameters = [cfg["Method"]["tuned_parameters"]]

    # Hyper-parameters based on type of model

    if method_name in ["RandomForestRegressor", "RandomForestClassifier"]:
        n_estimators = tuned_parameters[0]["n_estimators"]
        criterion = tuned_parameters[0]["criterion"]
        max_depth = None if tuned_parameters[0]["max_depth"] == "None" else tuned_parameters[0]["max_depth"]
        min_samples_split = tuned_parameters[0]["min_samples_split"]
        min_samples_leaf = tuned_parameters[0]["min_samples_leaf"]
        min_weight_fraction_leaf = tuned_parameters[0]["min_weight_fraction_leaf"]
        max_features = tuned_parameters[0]["max_features"]
        max_leaf_nodes = None if tuned_parameters[0]["max_leaf_nodes"] == "None" else tuned_parameters[0]["max_leaf_nodes"]
        min_impurity_decrease = tuned_parameters[0]["min_impurity_decrease"]
        bootstrap = False if tuned_parameters[0]["bootstrap"] == "False" else True
        oob_score = False if tuned_parameters[0]["oob_score"] == "False" else True
        n_jobs = None if tuned_parameters[0]["n_jobs"] == "None" else tuned_parameters[0]["n_jobs"]
        random_state = None if tuned_parameters[0]["random_state"] == "None" else tuned_parameters[0]["random_state"]
        verbose = tuned_parameters[0]["verbose"]
        warm_start = False if tuned_parameters[0]["warm_start"] == "False" else True
        ccp_alpha = tuned_parameters[0]["ccp_alpha"]
        max_samples = None if tuned_parameters[0]["max_samples"] == "None" else tuned_parameters[0]["max_samples"]

    if method_name == 'AdaBoostRegressor':
        base_estimator = 'None'  # (aka DecisionTreeRegressor)
        n_estimators = cfg["Method"]["n_estimators"]
        learning_rate = cfg["Method"]["learning_rate"]
        random_state = None if cfg["Method"]["random_state"] == "None" else cfg["Method"]["random_state"]

    if method_name == 'RandomForestClassifier':
        class_weight = cfg["Method"]["class_weight"]

    if model_type == "regressor":
        data_file = cfg["General"]["data_file"]
        filename = path.join(data_dir, data_file)

        # Get the selection cuts
        cuts = make_cut_list(cfg["SigFiducialCuts"])

        # # Initialize the model
        if method_name in "AdaBoostRegressor":
            init_model = AdaBoostRegressor(
                base_estimator=DecisionTreeRegressor(max_depth=None)
                )
        if method_name in "RandomForestRegressor":
            init_model = RandomForestRegressor(
                n_estimators=n_estimators,
                criterion=criterion,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                min_weight_fraction_leaf=min_weight_fraction_leaf,
                max_features=max_features,
                max_leaf_nodes=max_leaf_nodes,
                min_impurity_decrease=min_impurity_decrease,
                bootstrap=bootstrap,
                oob_score=oob_score,
                n_jobs=n_jobs,
                random_state=random_state,
                verbose=verbose,
                warm_start=warm_start,
                ccp_alpha=ccp_alpha,
                max_samples=max_samples
            )
        else:
            print("ERROR: we support only AdaBoostRegressor and RandomForestRegressor at the moment!")
            exit()

        # Name of target quantity
        target_name = cfg["Method"]["target_name"]

    elif model_type in "classifier":
        data_sig_file = cfg["General"]["data_sig_file"].format(args.mode)
        data_bkg_file = cfg["General"]["data_bkg_file"].format(args.mode)
        filename_sig = path.join(data_dir, data_sig_file)
        filename_bkg = path.join(data_dir, data_bkg_file)

        # Get the selection cuts
        sig_cuts = make_cut_list(cfg["SigFiducialCuts"])
        bkg_cuts = make_cut_list(cfg["BkgFiducialCuts"])

        # Initialize the the base estimator
        if method_name in "AdaBoostClassifier":
            init_model = AdaBoostClassifier(DecisionTreeClassifier(max_depth=4))
        elif method_name in "RandomForestClassifier":
            init_model = RandomForestClassifier(
                n_estimators=n_estimators,
                criterion=criterion,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                min_weight_fraction_leaf=min_weight_fraction_leaf,
                max_features=max_features,
                max_leaf_nodes=max_leaf_nodes,
                min_impurity_decrease=min_impurity_decrease,
                bootstrap=bootstrap,
                oob_score=oob_score,
                n_jobs=n_jobs,
                random_state=random_state,
                verbose=verbose,
                warm_start=warm_start,
                class_weight=class_weight,
                ccp_alpha=ccp_alpha,
                max_samples=max_samples
            )

            # Name of target
            target_name = cfg["Method"]["target_name"]

        use_same_number_of_sig_and_bkg_for_training = cfg["Split"][
            "use_same_number_of_sig_and_bkg_for_training"
        ]

    print("### Using {} for model construction".format(method_name))

    models = dict()
    for idx, cam_id in enumerate(cam_ids):

        print("### Building model for {}".format(cam_id))

        if model_type in "regressor":

            # Load data
            data = pd.read_hdf(filename, table_name[idx], mode="r")
            data = prepare_data(ds=data, cuts=cuts)[0: args.max_events]

            # Initialize the model
            factory = TrainModel(
                case=model_type,
                target_name=target_name,
                feature_name_list=feature_list
            )

            # Split the TRAINING dataset in a train and test sub-datasets
            # Useful to test the models before using them for DL2 production
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

            data_sig = data_sig[0: args.max_events]
            data_bkg = data_bkg[0: args.max_events]

            # Initialize the model
            factory = TrainModel(
                case=model_type, target_name=target_name, feature_name_list=feature_list
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

        # Apply optimization of the hyper-parameters via grid search if enabled
        if use_GridSearchCV == "True":
            best_model = factory.get_optimal_model(
                init_model, tuned_parameters, scoring=scoring, cv=cv
            )
        else:  # otherwise use directly the initial model
            best_model = init_model

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
                    factory.data_scikit["X_test"],
                    factory.data_scikit["y_test"]
                )

        save_output(models,
                    cam_id,
                    factory,
                    best_model,
                    model_type,
                    method_name,
                    outdir)


if __name__ == "__main__":
    main()
