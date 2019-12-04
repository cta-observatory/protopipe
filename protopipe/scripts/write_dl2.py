#!/usr/bin/env python

from sys import exit
import numpy as np
from glob import glob
import signal
from astropy.coordinates.angle_utilities import angular_separation
import tables as tb

# ctapipe
# from ctapipe.io import EventSourceFactory
from ctapipe.io import event_source
from ctapipe.utils.CutFlow import CutFlow
from ctapipe.reco.energy_regressor import EnergyRegressor
from ctapipe.reco.event_classifier import EventClassifier

# Utilities
from protopipe.pipeline import EventPreparer
from protopipe.pipeline.utils import (
    make_argparser,
    prod3b_array,
    str2bool,
    load_config,
    SignalHandler,
)

# from memory_profiler import profile

# @profile
def main():

    # Argument parser
    parser = make_argparser()
    parser.add_argument("--regressor_dir", default="./", help="regressors directory")
    parser.add_argument("--classifier_dir", default="./", help="regressors directory")
    parser.add_argument(
        "--force_tailcut_for_extended_cleaning",
        type=str2bool,
        default=False,
        help="For tailcut cleaning for energy/score estimation",
    )
    parser.add_argument(
        "--save_images",
        action="store_true",
        help="Save images in images.h5 (one file testing)",
    )
    args = parser.parse_args()

    # Read configuration file
    cfg = load_config(args.config_file)

    # Read site layout
    site = cfg["General"]["site"]
    array = cfg["General"]["array"]

    # Add force_tailcut_for_extended_cleaning in configuration
    cfg["General"][
        "force_tailcut_for_extended_cleaning"
    ] = args.force_tailcut_for_extended_cleaning
    cfg["General"]["force_mode"] = "tail"
    force_mode = args.mode
    if cfg["General"]["force_tailcut_for_extended_cleaning"] is True:
        force_mode = "tail"
    print("force_mode={}".format(force_mode))
    print("mode={}".format(args.mode))

    if args.infile_list:
        filenamelist = []
        for f in args.infile_list:
            filenamelist += glob("{}/{}".format(args.indir, f))
        filenamelist.sort()

    if not filenamelist:
        print("no files found; check indir: {}".format(args.indir))
        exit(-1)

    # Get telescope IDs and involved camera types from the first event
    allowed_tels, cameras = prod3b_array(filenamelist[0], site, array)

    # keeping track of events and where they were rejected
    evt_cutflow = CutFlow("EventCutFlow")
    img_cutflow = CutFlow("ImageCutFlow")

    # Event preparer
    preper = EventPreparer(
        config=cfg,
        cameras=cameras,
        mode=args.mode,
        event_cutflow=evt_cutflow,
        image_cutflow=img_cutflow,
    )

    # Regressor and classifier methods
    regressor_method = cfg["EnergyRegressor"]["method_name"]
    classifier_method = cfg["GammaHadronClassifier"]["method_name"]
    use_proba_for_classifier = cfg["GammaHadronClassifier"]["use_proba"]

    if regressor_method in ["None", "none", None]:
        use_regressor = False
    else:
        use_regressor = True

    if classifier_method in ["None", "none", None]:
        use_classifier = False
    else:
        use_classifier = True

    # Classifiers
    if use_classifier:
        classifier_files = (
            args.classifier_dir + "/classifier_{mode}_{cam_id}_{classifier}.pkl.gz"
        )
        clf_file = classifier_files.format(
            **{
                "mode": force_mode,
                "wave_args": "mixed",
                "classifier": classifier_method,
                "cam_id": "{cam_id}",
            }
        )
        classifier = EventClassifier.load(clf_file, cam_id_list=cameras)

    # Regressors
    if use_regressor:
        regressor_files = (
            args.regressor_dir + "/regressor_{mode}_{cam_id}_{regressor}.pkl.gz"
        )
        reg_file = regressor_files.format(
            **{
                "mode": force_mode,
                "wave_args": "mixed",
                "regressor": regressor_method,
                "cam_id": "{cam_id}",
            }
        )
        regressor = EnergyRegressor.load(reg_file, cam_id_list=cameras)

    # catch ctr-c signal to exit current loop and still display results
    signal_handler = SignalHandler()
    signal.signal(signal.SIGINT, signal_handler)

    # Declaration of the column descriptor for the (possible) images file
    class StoredImages(tb.IsDescription):
        event_id = tb.Int32Col(dflt=1, pos=0)
        tel_id = tb.Int16Col(dflt=1, pos=1)
        dl1_phe_image = tb.Float32Col(shape=(1855), pos=2)
        mc_phe_image = tb.Float32Col(shape=(1855), pos=3)

    # this class defines the reconstruction parameters to keep track of
    class RecoEvent(tb.IsDescription):
        obs_id = tb.Int16Col(dflt=-1, pos=0)
        event_id = tb.Int32Col(dflt=-1, pos=1)
        NTels_trig = tb.Int16Col(dflt=0, pos=2)
        NTels_reco = tb.Int16Col(dflt=0, pos=3)
        NTels_reco_lst = tb.Int16Col(dflt=0, pos=4)
        NTels_reco_mst = tb.Int16Col(dflt=0, pos=5)
        NTels_reco_sst = tb.Int16Col(dflt=0, pos=6)
        mc_energy = tb.Float32Col(dflt=np.nan, pos=7)
        reco_energy = tb.Float32Col(dflt=np.nan, pos=8)
        reco_alt = tb.Float32Col(dflt=np.nan, pos=9)
        reco_az = tb.Float32Col(dflt=np.nan, pos=10)
        offset = tb.Float32Col(dflt=np.nan, pos=11)
        xi = tb.Float32Col(dflt=np.nan, pos=12)
        ErrEstPos = tb.Float32Col(dflt=np.nan, pos=13)
        ErrEstDir = tb.Float32Col(dflt=np.nan, pos=14)
        gammaness = tb.Float32Col(dflt=np.nan, pos=15)
        success = tb.BoolCol(dflt=False, pos=16)
        score = tb.Float32Col(dflt=np.nan, pos=17)
        h_max = tb.Float32Col(dflt=np.nan, pos=18)
        reco_core_x = tb.Float32Col(dflt=np.nan, pos=19)
        reco_core_y = tb.Float32Col(dflt=np.nan, pos=20)
        mc_core_x = tb.Float32Col(dflt=np.nan, pos=21)
        mc_core_y = tb.Float32Col(dflt=np.nan, pos=22)

    reco_outfile = tb.open_file(
        mode="w",
        # if no outfile name is given (i.e. don't to write the event list to disk),
        # need specify two "driver" arguments
        **(
            {"filename": args.outfile}
            if args.outfile
            else {
                "filename": "no_outfile.h5",
                "driver": "H5FD_CORE",
                "driver_core_backing_store": False,
            }
        )
    )

    reco_table = reco_outfile.create_table("/", "reco_events", RecoEvent)
    reco_event = reco_table.row

    # Create the images file only if the user want to store the images
    if args.save_images is True:
        images_outfile = tb.open_file("images.h5", mode="w")
        images_table = {}
        images_phe = {}

    for i, filename in enumerate(filenamelist):

        source = event_source(
            input_url=filename, allowed_tels=allowed_tels, max_events=args.max_events
        )
        # loop that cleans and parametrises the images and performs the reconstruction
        for (
            event,
            dl1_phe_image,
            mc_phe_image,
            n_pixel_dict,
            hillas_dict,
            hillas_dict_reco,
            n_tels,
            tot_signal,
            max_signals,
            n_cluster_dict,
            reco_result,
            impact_dict,
        ) in preper.prepare_event(source):

            # Angular quantities
            run_array_direction = event.mcheader.run_array_direction

            # Angular separation between true and reco direction
            xi = angular_separation(
                event.mc.az, event.mc.alt, reco_result.az, reco_result.alt
            )

            # Angular separation bewteen the center of the camera and the reco direction.
            offset = angular_separation(
                run_array_direction[0],  # az
                run_array_direction[1],  # alt
                reco_result.az,
                reco_result.alt,
            )

            # Height of shower maximum
            h_max = reco_result.h_max

            if hillas_dict is not None:

                # Estimate particle energy
                if use_regressor is True:
                    energy_tel = np.zeros(len(hillas_dict.keys()))
                    weight_tel = np.zeros(len(hillas_dict.keys()))

                    for idx, tel_id in enumerate(hillas_dict.keys()):
                        cam_id = event.inst.subarray.tel[tel_id].camera.cam_id
                        moments = hillas_dict[tel_id]
                        model = regressor.model_dict[cam_id]

                        # Features to be fed in the regressor
                        features_img = np.array(
                            [
                                np.log10(moments.intensity),
                                np.log10(impact_dict[tel_id].value),
                                moments.width.value,
                                moments.length.value,
                                h_max.value,
                            ]
                        )

                        energy_tel[idx] = model.predict([features_img])
                        weight_tel[idx] = moments.intensity

                    reco_energy = np.sum(weight_tel * energy_tel) / sum(weight_tel)
                else:
                    reco_energy = np.nan

                # Estimate particle score/gammaness
                if use_classifier is True:
                    score_tel = np.zeros(len(hillas_dict.keys()))
                    gammaness_tel = np.zeros(len(hillas_dict.keys()))
                    weight_tel = np.zeros(len(hillas_dict.keys()))

                    for idx, tel_id in enumerate(hillas_dict.keys()):
                        cam_id = event.inst.subarray.tel[tel_id].camera.cam_id
                        moments = hillas_dict[tel_id]
                        model = classifier.model_dict[cam_id]
                        # Features to be fed in the classifier
                        features_img = np.array(
                            [
                                np.log10(reco_energy),
                                moments.width.value,
                                moments.length.value,
                                moments.skewness,
                                moments.kurtosis,
                                h_max.value,
                            ]
                        )
                        # Output of classifier according to type of classifier
                        if use_proba_for_classifier is False:
                            score_tel[idx] = model.decision_function([features_img])
                        else:
                            gammaness_tel[idx] = model.predict_proba([features_img])[
                                :, 1
                            ]
                        # Should test other weighting strategy (e.g. power of charge, impact, etc.)
                        # For now, weighting a la Mars
                        weight_tel[idx] = np.sqrt(moments.intensity)

                    # Weight the final decision/proba
                    if use_proba_for_classifier is True:
                        gammaness = np.sum(weight_tel * gammaness_tel) / sum(weight_tel)
                    else:
                        score = np.sum(weight_tel * score_tel) / sum(weight_tel)
                else:
                    score = np.nan
                    gammaness = np.nan

                # Regardless if energy or gammaness is estimated, if the user
                # wants to save the images of the run we do it here
                # (Probably not the most efficient way, but for one file is ok)
                if args.save_images is True:
                    for idx, tel_id in enumerate(hillas_dict.keys()):
                        cam_id = event.inst.subarray.tel[tel_id].camera.cam_id
                        if cam_id not in images_phe:
                            images_table[cam_id] = images_outfile.create_table(
                                "/", "_".join(["images", cam_id]), StoredImages
                            )
                            images_phe[cam_id] = images_table[cam_id].row

                shower = event.mc
                mc_core_x = shower.core_x
                mc_core_y = shower.core_y

                reco_core_x = reco_result.core_x
                reco_core_y = reco_result.core_y

                alt, az = reco_result.alt, reco_result.az

                # Fill table's attributes
                reco_event["NTels_trig"] = len(event.dl0.tels_with_data)
                reco_event["NTels_reco"] = len(hillas_dict)
                reco_event["NTels_reco_lst"] = n_tels["LST_LST_LSTCam"]
                reco_event["NTels_reco_mst"] = (
                    n_tels["MST_MST_NectarCam"]
                    + n_tels["MST_MST_FlashCam"]
                    + n_tels["SCT_SCT_SCTCam"]
                )
                reco_event["NTels_reco_sst"] = (
                    n_tels["SST_1M_DigiCam"]
                    + n_tels["SST_ASTRI_ASTRICam"]
                    + n_tels["SST_GCT_CHEC"]
                )
                reco_event["reco_energy"] = reco_energy
                reco_event["reco_alt"] = alt.to("deg").value
                reco_event["reco_az"] = az.to("deg").value
                reco_event["offset"] = offset.to("deg").value
                reco_event["xi"] = xi.to("deg").value
                reco_event["h_max"] = h_max.to("m").value
                reco_event["reco_core_x"] = reco_core_x.to("m").value
                reco_event["reco_core_y"] = reco_core_y.to("m").value
                reco_event["mc_core_x"] = mc_core_x.to("m").value
                reco_event["mc_core_y"] = mc_core_y.to("m").value
                if use_proba_for_classifier is True:
                    reco_event["gammaness"] = gammaness
                else:
                    reco_event["score"] = score
                reco_event["success"] = True
                reco_event["ErrEstPos"] = np.nan
                reco_event["ErrEstDir"] = np.nan
            else:
                reco_event["success"] = False

            # save basic event infos
            reco_event["mc_energy"] = event.mc.energy.to("TeV").value
            reco_event["event_id"] = event.r1.event_id
            reco_event["obs_id"] = event.r1.obs_id

            if args.save_images is True:
                images_phe[cam_id]["event_id"] = event.r0.event_id
                images_phe[cam_id]["tel_id"] = tel_id
                images_phe[cam_id]["dl1_phe_image"] = dl1_phe_image
                images_phe[cam_id]["mc_phe_image"] = mc_phe_image

                images_phe[cam_id].append()

            # Fill table
            reco_table.flush()
            reco_event.append()

            if signal_handler.stop:
                break
        if signal_handler.stop:
            break

    # make sure everything gets written out nicely
    reco_table.flush()

    if args.save_images is True:
        for table in images_table.values():
            table.flush()

    # Add in meta-data's table?
    try:
        print()
        evt_cutflow()
        print()
        img_cutflow()

    except ZeroDivisionError:
        pass

    print("Job done!")


if __name__ == "__main__":
    main()
