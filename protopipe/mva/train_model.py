import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from .utils import split_train_test

__all__ = ["TrainModel"]


class TrainModel(object):
    """
    Train classification or regressor model.

    Parameters
    ----------
    case: str
        Possibilities are regressor or classifier
    feature_name_list: list
        List of features
    target_name: str, optional
        Regression target
    """

    def __init__(self, case, feature_name_list, target_name=None):
        self.case = case

        self.target_name = target_name
        self.feature_name_list = feature_name_list

        self.data_test = None
        self.data_train = None
        self.data_scikit = None

    def split_data(
        self, data_sig, train_fraction, data_bkg=None, force_same_nsig_nbkg=False
    ):
        """
        Load and split data to build train/test samples.

        Parameters
        ----------
        data_sig: `~pandas.DataFrame`
            Data frame
        train_fraction: float
            Fraction of events to build the training sample
        data_bkg: `~pandas.DataFrame`
            Data frame
        force_same_nsig_nbkg: bool
            If true, the same number of signal and bkg events will be used
            to build a classifier
        """

        if self.case == "regressor":
            (
                X_train,
                X_test,
                y_train,
                y_test,
                self.data_train,
                self.data_test,
            ) = split_train_test(
                survived_images=data_sig,
                train_fraction=train_fraction,
                feature_name_list=self.feature_name_list,
                target_name=self.target_name,
            )
            weight = np.ones(len(self.data_train))
            weight_train = weight / sum(weight)

        else:
            (
                X_train_sig,
                X_test_sig,
                y_train_sig,
                y_test_sig,
                data_train_sig,
                data_test_sig,
            ) = split_train_test(
                survived_images=data_sig,
                train_fraction=train_fraction,
                feature_name_list=self.feature_name_list,
                target_name=self.target_name,
            )

            (
                X_train_bkg,
                X_test_bkg,
                y_train_bkg,
                y_test_bkg,
                data_train_bkg,
                data_test_bkg,
            ) = split_train_test(
                survived_images=data_bkg,
                train_fraction=train_fraction,
                feature_name_list=self.feature_name_list,
                target_name=self.target_name,
            )

            max_events = None

            if force_same_nsig_nbkg is True:
                if len(X_train_bkg) <= len(X_train_sig):
                    max_events = len(X_train_bkg)
                else:
                    max_events = len(X_train_sig)
            X_train = X_train_sig[0:max_events].append(X_train_bkg[0:max_events])
            y_train = y_train_sig[0:max_events].append(y_train_bkg[0:max_events])
            self.data_train = data_train_sig[0:max_events].append(
                data_train_bkg[0:max_events]
            )

            if force_same_nsig_nbkg is True:
                if len(X_test_bkg) <= len(X_test_sig):
                    max_events = len(X_test_bkg)
                else:
                    max_events = len(X_test_sig)

            try:
                X_test = X_test_sig[0:max_events].append(X_test_bkg[0:max_events])
                y_test = y_test_sig[0:max_events].append(y_test_bkg[0:max_events])
                self.data_test = data_test_sig[0:max_events].append(
                    data_test_bkg[0:max_events]
                )
            except TypeError as e:
                if str(e) != "'NoneType' object is unsubscriptable":
                    raise
                else:
                    X_test = None
                    y_test = None
                    self.data_test = None

            weight = np.ones(len(X_train))
            weight_train = weight / sum(weight)

        if X_test is not None:
            self.data_scikit = {
                "X_train": X_train.values,
                "X_test": X_test.values,
                "y_train": y_train.values,
                "y_test": y_test.values,
                "w_train": weight_train,
            }
        else:
            self.data_scikit = {
                "X_train": X_train.values,
                "X_test": None,
                "y_train": y_train.values,
                "y_test": None,
                "w_train": weight_train,
            }

    def get_optimal_model(self, init_model, tuned_parameters, scoring, cv, refit=True, verbose=2, njobs=1):
        """
        Get optimal hyperparameters for an estimator and return the best model.

        The best parameters are obtained by performing an exhaustive search
        over specified parameter values.

        Parameters
        ----------
        init_model: `~sklearn.base.BaseEstimator`
            Model to optimise
        tuned_parameters: dict
            Contains parameter names and ranges to optimise on
        scoring: str
            Estimator
        cv: int
            number of split for x-validation
        refit: bool, str, or callable, default=False
            Refit the estimator using the best found parameters on the whole dataset.
        verbose: int
            Controls the verbosity: the higher, the more messages.
            >1 : the computation time for each fold and parameter candidate is displayed
            >2 : the score is also displayed
            >3 : the fold and candidate parameter indexes are also displayed together with the starting time of the computation
        njobs: int
            Number of jobs to run in parallel. -1 means using all processors.

        Returns
        -------
        best_estimator: `~sklearn.base.BaseEstimator`
            Best model
        """
        model = GridSearchCV(init_model,
                             tuned_parameters,
                             scoring=scoring,
                             cv=cv,
                             refit=refit,
                             verbose=verbose)
        model.fit(
            self.data_scikit["X_train"],
            self.data_scikit["y_train"],
            sample_weight=self.data_scikit["w_train"],
        )

        print("Best parameters set found on development set:")
        for key in model.best_params_.keys():
            print(" - {}: {}".format(key, model.best_params_[key]))

        print("Grid scores on development set:")
        means = model.cv_results_["mean_test_score"]
        stds = model.cv_results_["std_test_score"]
        if verbose > 2:
            for mean, std, params in zip(means, stds, model.cv_results_["params"]):
                print(" - {:.3f}+/-{:.3f} for {}".format(mean, std * 2, params))

        grid_search_cv_results = pd.DataFrame(model.cv_results_)
        if verbose > 3:
            print(grid_search_cv_results)

        best_estimator = model.best_estimator_

        return best_estimator
