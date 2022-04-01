"""Process simtel data for the training of the estimator models."""

import numpy as np
import astropy.units as u
from astropy.coordinates.angle_utilities import angular_separation
from sys import exit as sys_exit
from glob import glob
import signal
import tables as tb
import pandas as pd
from tqdm import tqdm

from ctapipe.utils import CutFlow
from protopipe.pipeline.temp import MySimTelEventSource

from protopipe.pipeline import EventPreparer
from protopipe.pipeline.io import load_config, load_models
from protopipe.pipeline.utils import (
    make_argparser,
    prod5N_array,
    prod3b_array,
    str2bool,
    SignalHandler,
    bcolors,
)


def main():

    # Argument parser
    parser = make_argparser()

    parser.add_argument(
        "--debug", action="store_true", help="Print debugging information",
    )

    parser.add_argument(
        "--show_progress_bar",
        action="store_true",
        help="Show information about execution progress",
    )

    parser.add_argument(
        "--save_images", action="store_true", help="Save also all images",
    )

    parser.add_argument(
        "--estimate_energy",
        type=str2bool,
        default=False,
        help="Estimate the events' energy with a regressor from\
         protopipe.scripts.build_model",
    )
    parser.add_argument(
        "--regressor_dir", type=str, default="./", help="regressors directory"
    )
    parser.add_argument(
        "--regressor_config",
        type=str,
        default=None,
        help="Configuration file used to produce regressor model",
    )

    args = parser.parse_args()

    # Read configuration file
    cfg = load_config(args.config_file)

    # Check that the user specify site, array and production
    try:
        site = cfg["General"]["site"]
        array = cfg["General"]["array"]
        production = cfg["General"]["production"]
        assert all(len(x) > 0 for x in [site, array, production])
    except (KeyError, AssertionError):
        raise ValueError(
            bcolors.FAIL
            + """At least one of 'site', 'array' and 'production' 
            are not properly defined in the analysis configuration
            file."""
            + bcolors.ENDC
        )
        sys_exit(-1)

    if args.infile_list:
        filenamelist = []
        for f in args.infile_list:
            filenamelist += glob("{}/{}".format(args.indir, f))
        filenamelist.sort()
    else:
        raise ValueError("don't know which input to use...")

    if not filenamelist:
        print("no files found; check indir: {}".format(args.indir))
        sys_exit(-1)
    else:
        print("found {} files".format(len(filenamelist)))

    # Get the IDs of the involved telescopes and associated cameras together
    # with the equivalent focal lengths from the first event
    if production == "Prod5N":
        allowed_tels, cams_and_foclens, subarray = prod5N_array(
            filenamelist[0], site, array
        )
    elif production == "Prod3b":
        allowed_tels, cams_and_foclens, subarray = prod3b_array(
            filenamelist[0], site, array
        )
    else:
        raise ValueError(bcolors.FAIL + "Unsupported production." + bcolors.ENDC)
        sys_exit(-1)

    # keeping track of events and where they were rejected
    evt_cutflow = CutFlow("EventCutFlow")
    img_cutflow = CutFlow("ImageCutFlow")

    preper = EventPreparer(
        config=cfg,
        subarray=subarray,
        cams_and_foclens=cams_and_foclens,
        mode=args.mode,
        event_cutflow=evt_cutflow,
        image_cutflow=img_cutflow,
    )

    # catch ctr-c signal to exit current loop and still display results
    signal_handler = SignalHandler()
    signal.signal(signal.SIGINT, signal_handler)

    # Regressor information
    regressor_method = cfg["EnergyRegressor"]["method_name"]
    try:
        estimation_weight = cfg["EnergyRegressor"]["estimation_weight"]
    except KeyError:
        estimation_weight = "CTAMARS"

    # wrapper for the scikit-learn regressor
    if args.estimate_energy is True:

        # Read configuration file
        regressor_config = load_config(args.regressor_config)
        log_10_target = regressor_config["Method"]["log_10_target"]

        regressor_files = args.regressor_dir + "/regressor_{cam_id}_{regressor}.pkl.gz"
        reg_file = regressor_files.format(
            **{
                "mode": args.mode,
                "wave_args": "mixed",  # ToDo, control
                "regressor": regressor_method,
                "cam_id": "{cam_id}",
            }
        )

        regressors = load_models(reg_file, cam_id_list=cams_and_foclens.keys())

    # COLUMN DESCRIPTOR AS DICTIONARY
    # Column descriptor for the file containing output training data."""
    DataTrainingOutput = dict(
        # ======================================================================
        # ARRAY
        obs_id=tb.Int16Col(dflt=1, pos=0),
        event_id=tb.Int32Col(dflt=1, pos=1),
        tel_id=tb.Int16Col(dflt=1, pos=2),
        N_LST=tb.Int16Col(dflt=1, pos=3),
        N_MST=tb.Int16Col(dflt=1, pos=4),
        N_SST=tb.Int16Col(dflt=1, pos=5),
        n_tel_reco=tb.FloatCol(dflt=1, pos=6),
        n_tel_discri=tb.FloatCol(dflt=1, pos=7),
        # ======================================================================
        # DL1
        hillas_intensity_reco=tb.Float32Col(dflt=1, pos=8),
        hillas_intensity=tb.Float32Col(dflt=1, pos=9),
        hillas_x_reco=tb.Float32Col(dflt=1, pos=10),
        hillas_y_reco=tb.Float32Col(dflt=1, pos=11),
        hillas_x=tb.Float32Col(dflt=1, pos=12),
        hillas_y=tb.Float32Col(dflt=1, pos=13),
        hillas_r_reco=tb.Float32Col(dflt=1, pos=14),
        hillas_r=tb.Float32Col(dflt=1, pos=15),
        hillas_phi_reco=tb.Float32Col(dflt=1, pos=16),
        hillas_phi=tb.Float32Col(dflt=1, pos=17),
        hillas_length_reco=tb.Float32Col(dflt=1, pos=18),
        hillas_length=tb.Float32Col(dflt=1, pos=19),
        hillas_width_reco=tb.Float32Col(dflt=1, pos=20),
        hillas_width=tb.Float32Col(dflt=1, pos=21),
        hillas_psi_reco=tb.Float32Col(dflt=1, pos=22),
        hillas_psi=tb.Float32Col(dflt=1, pos=23),
        hillas_skewness_reco=tb.Float32Col(dflt=1, pos=24),
        hillas_skewness=tb.Float32Col(dflt=1, pos=25),
        hillas_kurtosis=tb.Float32Col(dflt=1, pos=26),
        hillas_kurtosis_reco=tb.Float32Col(dflt=1, pos=27),
        leakage_intensity_width_1_reco=tb.Float32Col(dflt=np.nan, pos=28),
        leakage_intensity_width_2_reco=tb.Float32Col(dflt=np.nan, pos=29),
        leakage_intensity_width_1=tb.Float32Col(dflt=np.nan, pos=30),
        leakage_intensity_width_2=tb.Float32Col(dflt=np.nan, pos=31),
        concentration_cog=tb.Float32Col(dflt=np.nan, pos=32),
        concentration_core=tb.Float32Col(dflt=np.nan, pos=33),
        concentration_pixel=tb.Float32Col(dflt=np.nan, pos=34),
        # The following are missing from current ctapipe DL1 output
        # Not sure if it's worth to add them
        hillas_ellipticity_reco=tb.FloatCol(dflt=1, pos=35),
        hillas_ellipticity=tb.FloatCol(dflt=1, pos=36),
        max_signal_cam=tb.Float32Col(dflt=1, pos=37),
        pixels=tb.Int16Col(dflt=-1, pos=38),
        clusters=tb.Int16Col(dflt=-1, pos=39),
        # ======================================================================
        # DL2 - DIRECTION RECONSTRUCTION
        impact_dist=tb.Float32Col(dflt=1, pos=40),
        h_max=tb.Float32Col(dflt=1, pos=41),
        alt=tb.Float32Col(dflt=np.nan, pos=42),
        az=tb.Float32Col(dflt=np.nan, pos=43),
        err_est_pos=tb.Float32Col(dflt=1, pos=44),
        err_est_dir=tb.Float32Col(dflt=1, pos=45),
        xi=tb.Float32Col(dflt=np.nan, pos=46),
        offset=tb.Float32Col(dflt=np.nan, pos=47),
        mc_core_x=tb.FloatCol(dflt=1, pos=48),
        mc_core_y=tb.FloatCol(dflt=1, pos=49),
        reco_core_x=tb.FloatCol(dflt=1, pos=50),
        reco_core_y=tb.FloatCol(dflt=1, pos=51),
        mc_h_first_int=tb.FloatCol(dflt=1, pos=52),
        mc_x_max=tb.Float32Col(dflt=np.nan, pos=53),
        is_valid=tb.BoolCol(dflt=False, pos=54),
        good_image=tb.Int16Col(dflt=1, pos=55),
        true_az=tb.Float32Col(dflt=np.nan, pos=56),
        true_alt=tb.Float32Col(dflt=np.nan, pos=57),
        pointing_az=tb.Float32Col(dflt=np.nan, pos=58),
        pointing_alt=tb.Float32Col(dflt=np.nan, pos=59),
        # ======================================================================
        # DL2 - ENERGY ESTIMATION
        true_energy=tb.FloatCol(dflt=np.nan, pos=60),
        reco_energy=tb.FloatCol(dflt=np.nan, pos=61),
        reco_energy_tel=tb.Float32Col(dflt=np.nan, pos=62),
        # ======================================================================
        # DL1 IMAGES
        # this is optional data saved by the user
        # since these data declarations require to know how many pixels
        # each saved image will have,
        # we add them later on, right before creating the table
        # We list them here for reference
        # true_image=tb.Float32Col(shape=(1855), pos=56),
        # reco_image=tb.Float32Col(shape=(1855), pos=57),
        # cleaning_mask_reco=tb.BoolCol(shape=(1855), pos=58),  # not in ctapipe
        # =======================================================================
        #    TEMP
        N_reco_LST=tb.Int16Col(dflt=-1, pos=63),
        N_reco_MST=tb.Int16Col(dflt=-1, pos=64),
        N_reco_SST=tb.Int16Col(dflt=-1, pos=64),
        image_extraction=tb.Int16Col(dflt=-1, pos=65),
    )

    outfile = tb.open_file(args.outfile, mode="w")
    outfile.root._v_attrs["status"] = "incomplete"
    outTable = {}
    outData = {}

    # Configuration options for SimTelEventSource
    # Readout window integration correction
    try:
        calib_scale = cfg["Calibration"]["calib_scale"]
    except KeyError:
        # defaults for no calibscale applied
        calib_scale = 1.0

    for i, filename in enumerate(filenamelist):

        print("file: {} filename = {}".format(i, filename))

        source = MySimTelEventSource(
            input_url=filename,
            calib_scale=calib_scale,
            allowed_tels=allowed_tels,
            max_events=args.max_events,
        )

        # loop that cleans and parametrises the images and performs the
        # reconstruction for each event
        for (
            event,
            reco_image,
            cleaning_mask_reco,
            cleaning_mask_clusters,
            true_image,
            n_pixel_dict,
            hillas_dict,
            hillas_dict_reco,
            leakage_dict,
            concentration_dict,
            n_tels,
            n_tels_reco,
            max_signals,
            n_cluster_dict,
            reco_result,
            impact_dict,
            good_event,
            good_for_reco,
            image_extraction_status,
        ) in tqdm(
            preper.prepare_event(
                source, save_images=args.save_images, debug=args.debug
            ),
            desc=source.__class__.__name__,
            total=source.max_events,
            unit="event",
            disable=not args.show_progress_bar,
        ):
            # True direction
            true_az = event.simulation.shower.az
            true_alt = event.simulation.shower.alt

            # Array pointing in AltAz frame
            pointing_az = event.pointing.array_azimuth
            pointing_alt = event.pointing.array_altitude

            if good_event:

                xi = angular_separation(
                    event.simulation.shower.az,
                    event.simulation.shower.alt,
                    reco_result.az,
                    reco_result.alt,
                )

                offset = angular_separation(
                    event.pointing.array_azimuth,
                    event.pointing.array_altitude,
                    reco_result.az,
                    reco_result.alt,
                )

                # Impact parameter
                reco_core_x = reco_result.core_x
                reco_core_y = reco_result.core_y

                # Height of shower maximum
                h_max = reco_result.h_max
                # Todo add conversion in number of radiation length,
                # need an atmosphere profile

                is_valid = True

            else:  # something went wrong and the shower's reconstruction failed

                xi = np.nan * u.deg
                offset = np.nan * u.deg
                reco_core_x = np.nan * u.m
                reco_core_y = np.nan * u.m
                h_max = np.nan * u.m
                reco_result.alt = np.nan * u.deg
                reco_result.az = np.nan * u.deg
                is_valid = False

            reco_energy = np.nan
            reco_energy_tel = dict()

            # Not optimal at all, two loop on tel!!!
            # For energy estimation
            # Estimate energy only if the shower was reconstructed
            if (args.estimate_energy is True) and is_valid:
                weight_tel = np.zeros(len(hillas_dict.keys()))
                weight_statistic_tel = np.zeros(len(hillas_dict.keys()))
                energy_tel = np.zeros(len(hillas_dict.keys()))

                for idx, tel_id in enumerate(hillas_dict.keys()):

                    # use only images that survived cleaning and
                    # parametrization
                    if not good_for_reco[tel_id]:
                        # bad images will get an undetermined energy
                        # this is a per-telescope energy
                        # NOT the estimated energy for the shower
                        reco_energy_tel[tel_id] = np.nan
                        continue

                    cam_id = source.subarray.tel[tel_id].camera.camera_name
                    moments = hillas_dict[tel_id]
                    model = regressors[cam_id]

                    ############################################################
                    #                  GET FEATURES
                    ############################################################

                    # Read feature list from model configutation file
                    features_basic = regressor_config["FeatureList"]["Basic"]
                    features_derived = regressor_config["FeatureList"]["Derived"]
                    features = features_basic + list(features_derived)

                    # Create a pandas Dataframe with basic quantities
                    # This is needed in order to connect the I/O system of the
                    # model inputs to the in-memory computation of this script
                    data = pd.DataFrame(
                        {
                            "hillas_intensity": [moments.intensity],
                            "hillas_width": [moments.width.to("deg").value],
                            "hillas_length": [moments.length.to("deg").value],
                            "hillas_x": [moments.x.to("deg").value],
                            "hillas_y": [moments.y.to("deg").value],
                            "hillas_phi": [moments.phi.to("deg").value],
                            "hillas_r": [moments.r.to("deg").value],
                            "leakage_intensity_width_1_reco": [
                                leakage_dict[tel_id]["leak1_reco"]
                            ],
                            "leakage_intensity_width_2_reco": [
                                leakage_dict[tel_id]["leak2_reco"]
                            ],
                            "leakage_intensity_width_1": [
                                leakage_dict[tel_id]["leak1"]
                            ],
                            "leakage_intensity_width_2": [
                                leakage_dict[tel_id]["leak2"]
                            ],
                            "concentration_cog": [
                                concentration_dict[tel_id]["concentration_cog"]
                            ],
                            "concentration_core": [
                                concentration_dict[tel_id]["concentration_core"]
                            ],
                            "concentration_pixel": [
                                concentration_dict[tel_id]["concentration_pixel"]
                            ],
                            "az": [reco_result.az.to("deg").value],
                            "alt": [reco_result.alt.to("deg").value],
                            "h_max": [h_max.value],
                            "impact_dist": [impact_dict[tel_id].to("m").value],
                        }
                    )

                    # Compute derived features and add them to the dataframe
                    for key, expression in features_derived.items():
                        data.eval(f"{key} = {expression}", inplace=True)

                    # features_img = np.array(
                    #     [
                    #         np.log10(moments.intensity),
                    #         np.log10(impact_dict[tel_id].value),
                    #         moments.width.value,
                    #         moments.length.value,
                    #         h_max.value,
                    #     ]
                    # )

                    # sort features_to_use alphabetically to ensure order
                    # preservation with model.fit in protopipe.mva
                    features = sorted(features)

                    # Select the values for the full set of features
                    features_values = data[features].to_numpy()

                    ############################################################

                    if estimation_weight == "CTAMARS":
                        # Get an array of trees
                        predictions_trees = np.array(
                            [
                                tree.predict(features_values)
                                for tree in model.estimators_
                            ]
                        )
                        energy_tel[idx] = np.mean(predictions_trees, axis=0)
                        weight_statistic_tel[idx] = np.std(predictions_trees, axis=0)
                    else:
                        data.eval(
                            f"estimation_weight = {estimation_weight}", inplace=True
                        )
                        energy_tel[idx] = model.predict(features_values)
                        weight_tel[idx] = data["estimation_weight"]

                    if log_10_target:
                        energy_tel[idx] = 10 ** energy_tel[idx]
                        weight_tel[idx] = 10 ** weight_tel[idx]
                        weight_statistic_tel[idx] = 10 ** weight_statistic_tel[idx]

                    if estimation_weight == "CTAMARS":
                        # in CTAMARS the average is done after converting
                        # energy and weight to linear energy scale
                        weight_tel[idx] = 1 / (weight_statistic_tel[idx] ** 2)

                    reco_energy_tel[tel_id] = energy_tel[idx]

                reco_energy = np.sum(weight_tel * energy_tel) / sum(weight_tel)
            else:
                for idx, tel_id in enumerate(hillas_dict.keys()):
                    reco_energy_tel[tel_id] = np.nan

            for idx, tel_id in enumerate(hillas_dict.keys()):
                cam_id = source.subarray.tel[tel_id].camera.camera_name

                if cam_id not in outData:

                    if args.save_images is True:
                        # we define and save images content here, to make it
                        # adaptive to different cameras

                        n_pixels = source.subarray.tel[tel_id].camera.geometry.n_pixels
                        DataTrainingOutput["true_image"] = tb.Float32Col(
                            shape=(n_pixels), pos=56
                        )
                        DataTrainingOutput["reco_image"] = tb.Float32Col(
                            shape=(n_pixels), pos=57
                        )
                        DataTrainingOutput["cleaning_mask_reco"] = tb.BoolCol(
                            shape=(n_pixels), pos=58
                        )  # not in ctapipe
                        DataTrainingOutput["cleaning_mask_clusters"] = tb.BoolCol(
                            shape=(n_pixels), pos=58
                        )  # not in ctapipe

                    outTable[cam_id] = outfile.create_table(
                        "/", cam_id, DataTrainingOutput,
                    )
                    outData[cam_id] = outTable[cam_id].row

                moments = hillas_dict[tel_id]
                ellipticity = moments.width / moments.length

                # Write to file also the Hillas parameters that have been used
                # to calculate reco_results

                moments_reco = hillas_dict_reco[tel_id]
                ellipticity_reco = moments_reco.width / moments_reco.length

                outData[cam_id]["good_image"] = good_for_reco[tel_id]
                outData[cam_id]["is_valid"] = is_valid
                outData[cam_id]["impact_dist"] = impact_dict[tel_id].to("m").value
                outData[cam_id]["max_signal_cam"] = max_signals[tel_id]
                outData[cam_id]["hillas_intensity"] = moments.intensity

                outData[cam_id]["N_LST"] = np.sum(
                    [v for k, v in n_tels.items() if k.startswith("LST")]
                )
                outData[cam_id]["N_MST"] = np.sum(
                    [v for k, v in n_tels.items() if k.startswith("MST")]
                )
                outData[cam_id]["N_SST"] = np.sum(
                    [v for k, v in n_tels.items() if k.startswith("SST")]
                )
                outData[cam_id]["N_reco_LST"] = np.sum(
                    [v for k, v in n_tels_reco.items() if k.startswith("LST")]
                )
                outData[cam_id]["N_reco_MST"] = np.sum(
                    [v for k, v in n_tels_reco.items() if k.startswith("MST")]
                )
                outData[cam_id]["N_reco_SST"] = np.sum(
                    [v for k, v in n_tels_reco.items() if k.startswith("SST")]
                )

                outData[cam_id]["hillas_width"] = moments.width.to("deg").value
                outData[cam_id]["hillas_length"] = moments.length.to("deg").value
                outData[cam_id]["hillas_psi"] = moments.psi.to("deg").value
                outData[cam_id]["hillas_skewness"] = moments.skewness
                outData[cam_id]["hillas_kurtosis"] = moments.kurtosis
                outData[cam_id]["h_max"] = h_max.to("m").value
                outData[cam_id]["err_est_pos"] = np.nan
                outData[cam_id]["err_est_dir"] = np.nan
                outData[cam_id]["true_energy"] = event.simulation.shower.energy.to(
                    "TeV"
                ).value
                outData[cam_id]["true_az"] = true_az.to("deg").value
                outData[cam_id]["true_alt"] = true_alt.to("deg").value
                outData[cam_id]["pointing_az"] = pointing_az.to("deg").value
                outData[cam_id]["pointing_alt"] = pointing_alt.to("deg").value
                outData[cam_id]["hillas_x"] = moments.x.to("deg").value
                outData[cam_id]["hillas_y"] = moments.y.to("deg").value
                outData[cam_id]["hillas_phi"] = moments.phi.to("deg").value
                outData[cam_id]["hillas_r"] = moments.r.to("deg").value

                outData[cam_id]["pixels"] = n_pixel_dict[tel_id]
                outData[cam_id]["obs_id"] = event.index.obs_id
                outData[cam_id]["event_id"] = event.index.event_id
                outData[cam_id]["tel_id"] = tel_id
                outData[cam_id]["xi"] = xi.to("deg").value
                outData[cam_id]["reco_energy"] = reco_energy
                outData[cam_id]["hillas_ellipticity"] = ellipticity.value
                outData[cam_id]["clusters"] = n_cluster_dict[tel_id]
                outData[cam_id]["n_tel_discri"] = n_tels["GOOD images"]
                outData[cam_id]["mc_core_x"] = event.simulation.shower.core_x.to(
                    "m"
                ).value
                outData[cam_id]["mc_core_y"] = event.simulation.shower.core_y.to(
                    "m"
                ).value
                outData[cam_id]["reco_core_x"] = reco_core_x.to("m").value
                outData[cam_id]["reco_core_y"] = reco_core_y.to("m").value
                outData[cam_id][
                    "mc_h_first_int"
                ] = event.simulation.shower.h_first_int.to("m").value
                outData[cam_id]["offset"] = offset.to("deg").value
                outData[cam_id][
                    "mc_x_max"
                ] = event.simulation.shower.x_max.value  # g / cm2
                outData[cam_id]["alt"] = reco_result.alt.to("deg").value
                outData[cam_id]["az"] = reco_result.az.to("deg").value
                outData[cam_id]["reco_energy_tel"] = reco_energy_tel[tel_id]
                # Variables from hillas_dist_reco
                outData[cam_id]["n_tel_reco"] = n_tels["GOOD images"]
                outData[cam_id]["hillas_x_reco"] = moments_reco.x.to("deg").value
                outData[cam_id]["hillas_y_reco"] = moments_reco.y.to("deg").value
                outData[cam_id]["hillas_phi_reco"] = moments_reco.phi.to("deg").value
                outData[cam_id]["hillas_ellipticity_reco"] = ellipticity_reco.value
                outData[cam_id]["hillas_r_reco"] = moments_reco.r.to("deg").value
                outData[cam_id]["hillas_skewness_reco"] = moments_reco.skewness
                outData[cam_id]["hillas_kurtosis_reco"] = moments_reco.kurtosis
                outData[cam_id]["hillas_width_reco"] = moments_reco.width.to(
                    "deg"
                ).value
                outData[cam_id]["hillas_length_reco"] = moments_reco.length.to(
                    "deg"
                ).value
                outData[cam_id]["hillas_psi_reco"] = moments_reco.psi.to("deg").value
                outData[cam_id]["hillas_intensity_reco"] = moments_reco.intensity
                outData[cam_id]["leakage_intensity_width_1_reco"] = leakage_dict[
                    tel_id
                ]["leak1_reco"]
                outData[cam_id]["leakage_intensity_width_2_reco"] = leakage_dict[
                    tel_id
                ]["leak2_reco"]
                outData[cam_id]["leakage_intensity_width_1"] = leakage_dict[tel_id][
                    "leak1"
                ]
                outData[cam_id]["leakage_intensity_width_2"] = leakage_dict[tel_id][
                    "leak2"
                ]
                outData[cam_id]["concentration_cog"] = concentration_dict[tel_id][
                    "concentration_cog"
                ]
                outData[cam_id]["concentration_core"] = concentration_dict[tel_id][
                    "concentration_core"
                ]
                outData[cam_id]["concentration_pixel"] = concentration_dict[tel_id][
                    "concentration_pixel"
                ]
                outData[cam_id]["image_extraction"] = image_extraction_status[tel_id]

                # =======================
                # IMAGES INFORMATION
                # =======================

                if args.save_images is True:
                    # we define and save images content here, to make it
                    # adaptive to different cameras

                    outData[cam_id]["true_image"] = true_image[tel_id]
                    outData[cam_id]["reco_image"] = reco_image[tel_id]
                    outData[cam_id]["cleaning_mask_reco"] = cleaning_mask_reco[tel_id]
                    outData[cam_id]["cleaning_mask_clusters"] = cleaning_mask_clusters[
                        tel_id
                    ]
                # =======================

                outData[cam_id].append()

            if signal_handler.stop:
                break
        if signal_handler.stop:
            break
    # make sure that all the events are properly stored
    for table in outTable.values():
        table.flush()

    print(
        bcolors.BOLD
        + "\n\n==================================================\n"
        + "Statistical summary of processed events and images\n"
        + "==================================================\n"
        # + bcolors.ENDC
    )

    evt_cutflow()

    # Catch specific cases
    triggered_events = evt_cutflow.cuts["min2Tels trig"][1]
    reconstructed_events = evt_cutflow.cuts["min2Tels reco"][1]

    if triggered_events == 0:
        print(
            "\033[93mWARNING: No events have been triggered"
            " by the selected telescopes! \033[0m"
        )
    else:
        print("\n")
        img_cutflow()
        if reconstructed_events == 0:
            print(
                "\033[93m WARNING: None of the triggered events have been "
                "properly reconstructed by the selected telescopes!\n"
                "DL1 file will be empty! \033[0m"
            )
        print(bcolors.ENDC)

    # Conclude by writing some metadata
    outfile.root._v_attrs["status"] = "complete"
    outfile.root._v_attrs["num_showers"] = source.simulation_config.num_showers
    outfile.root._v_attrs["shower_reuse"] = source.simulation_config.shower_reuse

    outfile.close()

    print("Job done!")


if __name__ == "__main__":
    main()
