"""Input functions for a model initilization."""

import argparse
import joblib
from os import path

from protopipe.mva.utils import save_obj


def initialize_script_arguments():
    """Initialize the parser of protopipe.scripts.build_model.

    Returns
    -------
    args : argparse.Namespace
        Populated argparse namespace.
    """

    parser = argparse.ArgumentParser(
        description="Build model for regression/classification"
    )
    parser.add_argument("--config_file", type=str, required=True)

    parser.add_argument(
        "--max_events",
        type=int,
        default=-1,
        help="maximum number of events to use",
    )

    args = parser.parse_args()

    return args

def save_output(models,
                cam_id,
                factory,
                best_model,
                model_type,
                method_name,
                outdir):
    """Save model and data used to produce it per camera-type."""

    models[cam_id] = best_model
    outname = "{}_{}_{}.pkl.gz".format(
        model_type, cam_id, method_name
    )
    joblib.dump(best_model, path.join(outdir, outname))

    # SAVE DATA
    save_obj(
        factory.data_scikit,
        path.join(
            outdir,
            "data_scikit_{}_{}_{}.pkl.gz".format(
                model_type, method_name, cam_id
            ),
        ),
    )
    factory.data_train.to_pickle(
        path.join(
            outdir,
            "data_train_{}_{}_{}.pkl.gz".format(
                model_type, method_name, cam_id
            ),
        )
    )
    factory.data_test.to_pickle(
        path.join(
            outdir,
            "data_test_{}_{}_{}.pkl.gz".format(
                model_type, method_name, cam_id
            ),
        )
    )
