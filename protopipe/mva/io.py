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
        default=None,
        help="maximum number of events to use",
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
