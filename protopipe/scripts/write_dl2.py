#!/usr/bin/env python

from sys import exit
import numpy as np
from glob import glob
import signal
from astropy.coordinates.angle_utilities import angular_separation
import tables as tb
import astropy.units as u

# ctapipe
# from ctapipe.io import EventSourceFactory
from ctapipe.io import event_source
from ctapipe.utils.CutFlow import CutFlow
from ctapipe.reco.energy_regressor import EnergyRegressor
from ctapipe.reco.event_classifier import EventClassifier

# Utilities
from protopipe.pipeline import EventPreparer
from protopipe.pipeline.utils import (
    bcolors,
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

    parser.add_argument(
        "--debug", action="store_true", help="Print debugging information",
    )

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

    try:  # If the user didn't specify a site and/or and array...
        site = cfg["General"]["site"]
        array = cfg["General"]["array"]
    except KeyError:  # ...raise an error and exit.
        print(
            bcolors.FAIL
            + "ERROR: make sure that both 'site' and 'array' are "
            + "specified in the analysis configuration file!"
            + bcolors.ENDC
        )
        exit()

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

    # Get the IDs of the involved telescopes and associated cameras together
    # with the equivalent focal lengths from the first event
    allowed_tels, cams_and_foclens, subarray = prod3b_array(
        filenamelist[0], site, array
    )

    # keeping track of events and where they were rejected
    evt_cutflow = CutFlow("EventCutFlow")
    img_cutflow = CutFlow("ImageCutFlow")

    # Event preparer
    preper = EventPreparer(
        config=cfg,
        subarray=subarray,
        cams_and_foclens=cams_and_foclens,
        mode=args.mode,
        event_cutflow=evt_cutflow,
        image_cutflow=img_cutflow,
    )

    # Regressor and classifier methods
    regressor_method = cfg["EnergyRegressor"]["method_name"]
    classifier_method = cfg["GammaHadronClassifier"]["method_name"]
    use_proba_for_classifier = cfg["GammaHadronClassifier"]["use_proba"]

    if regressor_method in ["None", "none", None]:
        print(
            bcolors.OKBLUE
            + "The energy of the event will NOT be estimated."
            + bcolors.ENDC
        )
        use_regressor = False
    else:
        use_regressor = True

    if classifier_method in ["None", "none", None]:
        if args.debug:
            print(
                bcolors.OKBLUE
                + "The particle type of the event will NOT be estimated."
                + bcolors.ENDC
            )
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
        classifier = EventClassifier.load(clf_file, cam_id_list=cams_and_foclens.keys())
        if args.debug:
            print(
                bcolors.OKBLUE
                + "The particle type of the event will be estimated"
                + " using the models stored in"
                + f" {args.classifier_dir}\n"
                + bcolors.ENDC
            )

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
        regressor = EnergyRegressor.load(reg_file, cam_id_list=cams_and_foclens.keys())
        if args.debug:
            print(
                bcolors.OKBLUE
                + "The energy of the event will be estimated"
                + " using the models stored in"
                + f" {args.regressor_dir}\n"
                + bcolors.ENDC
            )

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
        pointing_az = tb.Float32Col(dflt=np.nan, pos=7)
        pointing_alt = tb.Float32Col(dflt=np.nan, pos=8)
        true_az = tb.Float32Col(dflt=np.nan, pos=9)
        true_alt = tb.Float32Col(dflt=np.nan, pos=10)
        true_energy = tb.Float32Col(dflt=np.nan, pos=11)
        reco_energy = tb.Float32Col(dflt=np.nan, pos=12)
        reco_alt = tb.Float32Col(dflt=np.nan, pos=13)
        reco_az = tb.Float32Col(dflt=np.nan, pos=14)
        offset = tb.Float32Col(dflt=np.nan, pos=15)
        xi = tb.Float32Col(dflt=np.nan, pos=16)
        ErrEstPos = tb.Float32Col(dflt=np.nan, pos=17)
        ErrEstDir = tb.Float32Col(dflt=np.nan, pos=18)
        gammaness = tb.Float32Col(dflt=np.nan, pos=19)
        success = tb.BoolCol(dflt=False, pos=20)
        score = tb.Float32Col(dflt=np.nan, pos=21)
        h_max = tb.Float32Col(dflt=np.nan, pos=22)
        reco_core_x = tb.Float32Col(dflt=np.nan, pos=23)
        reco_core_y = tb.Float32Col(dflt=np.nan, pos=24)
        true_core_x = tb.Float32Col(dflt=np.nan, pos=25)
        true_core_y = tb.Float32Col(dflt=np.nan, pos=26)
        is_valid=tb.BoolCol(dflt=False, pos=27)

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
            reco_image,
            cleaning_mask_reco,
            cleaning_mask_clusters,
            true_image,
            n_pixel_dict,
            hillas_dict,
            hillas_dict_reco,
            leakage_dict,
            n_tels,
            max_signals,
            n_cluster_dict,
            reco_result,
            impact_dict,
            good_event,
            good_for_reco,
        ) in preper.prepare_event(
            source, save_images=args.save_images, debug=args.debug
        ):

            # True energy
            true_energy = event.mc.energy.value
            
            # True direction
            true_az = event.mc.az
            true_alt = event.mc.alt
            
            # Angular quantities
            run_array_direction = event.mcheader.run_array_direction
            pointing_az, pointing_alt = run_array_direction[0], run_array_direction[1]

            if good_event:  # aka it has been successfully reconstructed

                # Angular separation between
                # - true direction
                # - reconstruted direction
                xi = angular_separation(
                    event.mc.az, event.mc.alt, reco_result.az, reco_result.alt
                )

                # Angular separation between
                # - center of the array's FoV
                # - reconstructed direction
                offset = angular_separation(
                    pointing_az,
                    pointing_alt,
                    reco_result.az,
                    reco_result.alt,
                )

                # Reconstructed height of shower maximum
                h_max = reco_result.h_max

                # Reconstructed position of the shower's core on the ground
                reco_core_x = reco_result.core_x
                reco_core_y = reco_result.core_y

                # Reconstructed direction of the shower's in the sky
                alt, az = reco_result.alt, reco_result.az

                # Successfully reconstructed shower
                is_valid = True

            else:  # no successful reconstruction assign dummy values

                xi = np.nan * u.deg
                offset = np.nan * u.deg
                reco_core_x = np.nan * u.m
                reco_core_y = np.nan * u.m
                h_max = np.nan * u.m
                alt = np.nan * u.deg
                az = np.nan * u.deg
                is_valid = False
                reco_energy = np.nan
                score = np.nan
                gammaness = np.nan
                reco_event["success"] = False

            # Estimate particle energy
            if use_regressor and is_valid:
                energy_tel = np.zeros(len(hillas_dict.keys()))
                energy_tel_classifier = {}
                weight_tel = np.zeros(len(hillas_dict.keys()))

                for idx, tel_id in enumerate(hillas_dict.keys()):

                    cam_id = source.subarray.tel[tel_id].camera.camera_name
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

                    if good_for_reco[tel_id] == 1:
                        energy_tel[idx] = model.predict([features_img])
                    else:
                        energy_tel[idx] = np.nan

                    weight_tel[idx] = moments.intensity

                    # Record the values regardless of the validity
                    # We don't use this now, but it should be recorded
                    energy_tel_classifier[tel_id] = energy_tel[idx]

                # Use only images with valid estimated energies to calculate
                # the average
                energy_tel_selected = energy_tel[~np.isnan(energy_tel)]
                weight_tel_selected = weight_tel[~np.isnan(energy_tel)]

                # Try getting the average weighted energy of the shower
                # If no image had a valid estimated energy record it as nan
                if len(energy_tel_selected) == 0:
                    reco_energy = np.nan
                    energy_estimated = False
                else:
                    reco_energy = np.sum(
                        weight_tel_selected * energy_tel_selected
                    ) / sum(weight_tel_selected)
                    energy_estimated = True
            else:
                reco_energy = np.nan
                energy_estimated = False

            # Estimate particle score/gammaness
            if use_classifier and is_valid:
                score_tel = np.zeros(len(hillas_dict.keys()))
                gammaness_tel = np.zeros(len(hillas_dict.keys()))
                weight_tel = np.zeros(len(hillas_dict.keys()))

                for idx, tel_id in enumerate(hillas_dict.keys()):
                    cam_id = source.subarray.tel[tel_id].camera.camera_name
                    moments = hillas_dict[tel_id]
                    model = classifier.model_dict[cam_id]
                    # Features to be fed in the classifier
                    # this should be read in some way from
                    # the classifier configuration file!!!!!

                    features_img = np.array(
                        [
                            np.log10(reco_energy),
                            np.log10(energy_tel_classifier[tel_id]),
                            np.log10(moments.intensity),
                            moments.width.value,
                            moments.length.value,
                            h_max.value,
                            impact_dict[tel_id].value,
                        ]
                    )

                    # Here we check for valid telescope-wise energies
                    # Because it means that it's a good image
                    # WARNING: currently we should REQUIRE to estimate both
                    # energy AND particle type
                    if not np.isnan(energy_tel_classifier[tel_id]):
                        # Output of classifier according to type of classifier
                        if use_proba_for_classifier is False:
                            score_tel[idx] = model.decision_function([features_img])
                        else:
                            gammaness_tel[idx] = model.predict_proba([features_img])[
                                :, 1
                            ]
                        weight_tel[idx] = np.sqrt(moments.intensity)
                    else:
                        # WARNING:
                        # this is true only because we use telescope-wise
                        # energies as a feature of the model!!!
                        score_tel[idx] = np.nan
                        gammaness_tel[idx] = np.nan

                # Use only images with valid estimated energies to calculate
                # the average
                if use_proba_for_classifier is False:
                    score_tel_selected = score_tel[~np.isnan(score_tel)]
                    weight_tel_selected = weight_tel[~np.isnan(score_tel)]
                else:
                    gammaness_tel_selected = gammaness_tel[~np.isnan(gammaness_tel)]
                    weight_tel_selected = weight_tel[~np.isnan(gammaness_tel)]

                # Try getting the average weighted score or gammaness
                # If no image had a valid estimated energy record it as nan
                if len(weight_tel_selected) > 0:

                    # Weight the final decision/proba
                    if use_proba_for_classifier is True:
                        gammaness = np.sum(
                            weight_tel_selected * gammaness_tel_selected
                        ) / sum(weight_tel_selected)
                    else:
                        score = np.sum(weight_tel_selected * score_tel_selected) / sum(
                            weight_tel_selected
                        )

                    particle_type_estimated = True

                else:

                    score = np.nan
                    gammaness = np.nan
                    particle_type_estimated = False

            else:
                score = np.nan
                gammaness = np.nan
                particle_type_estimated = False

            if energy_estimated and particle_type_estimated:
                reco_event["success"] = True
            else:
                if args.debug:
                    print(
                        bcolors.WARNING
                        + f"energy_estimated = {energy_estimated}\n"
                        + f"particle_type_estimated = {particle_type_estimated}\n"
                        + bcolors.ENDC
                    )
                reco_event["success"] = False

            # If the user wants to save the images of the run
            if args.save_images is True:
                for idx, tel_id in enumerate(hillas_dict.keys()):
                    cam_id = event.inst.subarray.tel[tel_id].camera.cam_id
                    if cam_id not in images_phe:
                        images_table[cam_id] = images_outfile.create_table(
                            "/", "_".join(["images", cam_id]), StoredImages
                        )
                        images_phe[cam_id] = images_table[cam_id].row

                images_phe[cam_id]["event_id"] = event.r0.event_id
                images_phe[cam_id]["tel_id"] = tel_id
                images_phe[cam_id]["reco_image"] = reco_image[tel_id]
                images_phe[cam_id]["true_image"] = true_image[tel_id]
                images_phe[cam_id]["cleaning_mask_reco"] = cleaning_mask_reco[tel_id]
                images_phe[cam_id]["cleaning_mask_clusters"] = cleaning_mask_clusters[
                    tel_id
                ]

                images_phe[cam_id].append()

            # Now we start recording the data to file
            reco_event["event_id"] = event.index.event_id
            reco_event["obs_id"] = event.index.obs_id
            reco_event["NTels_trig"] = len(event.dl0.tels_with_data)
            reco_event["NTels_reco"] = len(hillas_dict)
            reco_event["NTels_reco_lst"] = n_tels["LST_LST_LSTCam"]
            reco_event["NTels_reco_mst"] = (
                n_tels["MST_MST_NectarCam"]
                + n_tels["MST_MST_FlashCam"]
                + n_tels["MST_SCT_SCTCam"]
            )
            reco_event["NTels_reco_sst"] = (
                n_tels["SST_1M_DigiCam"]
                + n_tels["SST_ASTRI_ASTRICam"]
                + n_tels["SST_GCT_CHEC"]
            )
            reco_event["pointing_az"] = pointing_az.to("deg").value
            reco_event["pointing_alt"] = pointing_alt.to("deg").value
            reco_event["reco_energy"] = reco_energy
            reco_event["reco_alt"] = alt.to("deg").value
            reco_event["reco_az"] = az.to("deg").value
            reco_event["offset"] = offset.to("deg").value
            reco_event["xi"] = xi.to("deg").value
            reco_event["h_max"] = h_max.to("m").value
            reco_event["reco_core_x"] = reco_core_x.to("m").value
            reco_event["reco_core_y"] = reco_core_y.to("m").value
            reco_event["is_valid"] = is_valid

            if use_proba_for_classifier is True:
                reco_event["gammaness"] = gammaness
            else:
                reco_event["score"] = score
            reco_event["ErrEstPos"] = np.nan
            reco_event["ErrEstDir"] = np.nan

            # Simulated information
            shower = event.mc
            mc_core_x = shower.core_x
            mc_core_y = shower.core_y
            reco_event["true_energy"] = event.mc.energy.to("TeV").value
            reco_event["true_az"] = true_az.to("deg").value
            reco_event["true_alt"] = true_alt.to("deg").value
            reco_event["true_core_x"] = mc_core_x.to("m").value
            reco_event["true_core_y"] = mc_core_y.to("m").value

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
