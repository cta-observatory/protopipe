#!/usr/bin/env python
from astropy.coordinates.angle_utilities import angular_separation
from sys import exit
from glob import glob
import signal
import tables as tb

from ctapipe.utils.CutFlow import CutFlow
from ctapipe.io import event_source
from ctapipe.reco.energy_regressor import *

from protopipe.pipeline import EventPreparer
from protopipe.pipeline.utils import (
    make_argparser,
    prod3b_tel_ids,
    str2bool,
    load_config,
    SignalHandler,
)


def main():

    # Argument parser
    parser = make_argparser()
    parser.add_argument(
        "--estimate_energy",
        type=str2bool,
        default=False,
        help="Make estimation of energy",
    )
    parser.add_argument(
        "--regressor_dir", type=str, default="./", help="regressors directory"
    )
    args = parser.parse_args()

    # Read configuration file
    cfg = load_config(args.config_file)

    # Read site layout
    site = cfg["General"]["site"]
    array = cfg["General"]["array"]

    if args.infile_list:
        filenamelist = []
        for f in args.infile_list:
            filenamelist += glob("{}/{}".format(args.indir, f))
        filenamelist.sort()
    else:
        raise ValueError("don't know which input to use...")

    if not filenamelist:
        print("no files found; check indir: {}".format(args.indir))
        exit(-1)
    else:
        print("found {} files".format(len(filenamelist)))

    # keeping track of events and where they were rejected
    evt_cutflow = CutFlow("EventCutFlow")
    img_cutflow = CutFlow("ImageCutFlow")

    preper = EventPreparer(
        config=cfg, mode=args.mode, event_cutflow=evt_cutflow, image_cutflow=img_cutflow
    )

    # catch ctr-c signal to exit current loop and still display results
    signal_handler = SignalHandler()
    signal.signal(signal.SIGINT, signal_handler)

    # Regressor method
    regressor_method = cfg["EnergyRegressor"]["method_name"]

    # wrapper for the scikit-learn regressor
    if args.estimate_energy is True:
        # regressor_files = args.regressor_dir + "/regressor_{mode}_{cam_id}_{regressor}.pkl.gz"
        regressor_files = (
            args.regressor_dir + "/regressor_{mode}_{cam_id}_{regressor}.pkl.gz"
        )
        reg_file = regressor_files.format(
            **{
                "mode": args.mode,
                "wave_args": "mixed",  # ToDo, control
                "regressor": regressor_method,
                "cam_id": "{cam_id}",
            }
        )

        # from IPython import embed
        # embed()

        regressor = EnergyRegressor.load(reg_file, cam_id_list=args.cam_ids)

    class EventFeatures(tb.IsDescription):
        impact_dist = tb.Float32Col(dflt=1, pos=0)
        sum_signal_evt = tb.Float32Col(dflt=1, pos=1)
        max_signal_cam = tb.Float32Col(dflt=1, pos=2)
        sum_signal_cam = tb.Float32Col(dflt=1, pos=3)
        N_LST = tb.Int16Col(dflt=1, pos=4)
        N_MST = tb.Int16Col(dflt=1, pos=5)
        N_SST = tb.Int16Col(dflt=1, pos=6)
        width = tb.Float32Col(dflt=1, pos=7)
        length = tb.Float32Col(dflt=1, pos=8)
        skewness = tb.Float32Col(dflt=1, pos=9)
        kurtosis = tb.Float32Col(dflt=1, pos=10)
        h_max = tb.Float32Col(dflt=1, pos=11)
        err_est_pos = tb.Float32Col(dflt=1, pos=12)
        err_est_dir = tb.Float32Col(dflt=1, pos=13)
        mc_energy = tb.FloatCol(dflt=1, pos=14)
        local_distance = tb.Float32Col(dflt=1, pos=15)
        n_pixel = tb.Int16Col(dflt=1, pos=16)
        n_cluster = tb.Int16Col(dflt=-1, pos=17)
        obs_id = tb.Int16Col(dflt=1, pos=18)
        event_id = tb.Int32Col(dflt=1, pos=19)
        tel_id = tb.Int16Col(dflt=1, pos=20)
        xi = tb.Float32Col(dflt=np.nan, pos=21)
        reco_energy = tb.FloatCol(dflt=np.nan, pos=22)
        ellipticity = tb.FloatCol(dflt=1, pos=23)
        n_tel_reco = tb.FloatCol(dflt=1, pos=24)
        n_tel_discri = tb.FloatCol(dflt=1, pos=25)
        mc_core_x = tb.FloatCol(dflt=1, pos=26)
        mc_core_y = tb.FloatCol(dflt=1, pos=27)
        reco_core_x = tb.FloatCol(dflt=1, pos=28)
        reco_core_y = tb.FloatCol(dflt=1, pos=29)
        mc_h_first_int = tb.FloatCol(dflt=1, pos=30)
        offset = tb.Float32Col(dflt=np.nan, pos=31)
        mc_x_max = tb.Float32Col(dflt=np.nan, pos=31)
        alt = tb.Float32Col(dflt=np.nan, pos=33)
        az = tb.Float32Col(dflt=np.nan, pos=34)
        reco_energy_tel = tb.Float32Col(dflt=np.nan, pos=35)
        # from hillas_reco
        ellipticity_reco = tb.FloatCol(dflt=1, pos=36)
        local_distance_reco = tb.Float32Col(dflt=1, pos=37)
        skewness_reco = tb.Float32Col(dflt=1, pos=38)
        kurtosis_reco = tb.Float32Col(dflt=1, pos=39)
        width_reco = tb.Float32Col(dflt=1, pos=40)
        length_reco = tb.Float32Col(dflt=1, pos=41)
        psi = tb.Float32Col(dflt=1, pos=42)
        psi_reco = tb.Float32Col(dflt=1, pos=43)
        sum_signal_cam_reco = tb.Float32Col(dflt=1, pos=44)

    feature_outfile = tb.open_file(args.outfile, mode="w")
    feature_table = {}
    feature_events = {}

    # Telescopes in analysis
    allowed_tels = set(prod3b_tel_ids(array, site=site))

    for i, filename in enumerate(filenamelist):

        print("file: {} filename = {}".format(i, filename))

        source = event_source(
            input_url=filename, allowed_tels=allowed_tels, max_events=args.max_events
        )

        # loop that cleans and parametrises the images and performs the
        # reconstruction for each event
        for (
            event,
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

            xi = angular_separation(
                event.mc.az, event.mc.alt, reco_result.az, reco_result.alt
            )

            offset = angular_separation(
                run_array_direction[0],  # az
                run_array_direction[1],  # alt
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

            reco_energy = np.nan
            reco_energy_tel = dict()

            # Not optimal at all, two loop on tel!!!
            # For energy estimation
            if args.estimate_energy is True:
                weight_tel = np.zeros(len(hillas_dict.keys()))
                energy_tel = np.zeros(len(hillas_dict.keys()))

                for idx, tel_id in enumerate(hillas_dict.keys()):
                    cam_id = event.inst.subarray.tel[tel_id].camera.cam_id
                    moments = hillas_dict[tel_id]
                    model = regressor.model_dict[cam_id]

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
                    reco_energy_tel[tel_id] = energy_tel[idx]

                reco_energy = np.sum(weight_tel * energy_tel) / sum(weight_tel)
            else:
                for idx, tel_id in enumerate(hillas_dict.keys()):
                    reco_energy_tel[tel_id] = np.nan

            for idx, tel_id in enumerate(hillas_dict.keys()):
                cam_id = event.inst.subarray.tel[tel_id].camera.cam_id

                if cam_id not in feature_events:
                    feature_table[cam_id] = feature_outfile.create_table(
                        "/", "_".join(["feature_events", cam_id]), EventFeatures
                    )
                    feature_events[cam_id] = feature_table[cam_id].row

                moments = hillas_dict[tel_id]
                ellipticity = moments.width / moments.length

                # Write to file also the Hillas parameters that have been used
                # to calculate reco_results

                moments_reco = hillas_dict_reco[tel_id]
                ellipticity_reco = moments_reco.width / moments_reco.length

                feature_events[cam_id]["impact_dist"] = (
                    impact_dict[tel_id].to("m").value
                )
                feature_events[cam_id]["sum_signal_evt"] = tot_signal
                feature_events[cam_id]["max_signal_cam"] = max_signals[tel_id]
                feature_events[cam_id]["sum_signal_cam"] = moments.intensity
                feature_events[cam_id]["N_LST"] = n_tels["LST_LST_LSTCam"]
                feature_events[cam_id]["N_MST"] = n_tels["MST_MST_NectarCam"]
                feature_events[cam_id]["N_SST"] = n_tels["SST"]
                feature_events[cam_id]["width"] = moments.width.to("m").value
                feature_events[cam_id]["length"] = moments.length.to("m").value
                feature_events[cam_id]["psi"] = moments.psi.to("deg").value
                feature_events[cam_id]["skewness"] = moments.skewness
                feature_events[cam_id]["kurtosis"] = moments.kurtosis
                feature_events[cam_id]["h_max"] = h_max.to("m").value
                feature_events[cam_id]["err_est_pos"] = np.nan
                feature_events[cam_id]["err_est_dir"] = np.nan
                feature_events[cam_id]["mc_energy"] = event.mc.energy.to("TeV").value
                feature_events[cam_id]["local_distance"] = moments.r.to("m").value
                feature_events[cam_id]["n_pixel"] = n_pixel_dict[tel_id]
                feature_events[cam_id]["obs_id"] = event.r0.obs_id
                feature_events[cam_id]["event_id"] = event.r0.event_id
                feature_events[cam_id]["tel_id"] = tel_id
                feature_events[cam_id]["xi"] = xi.to("deg").value
                feature_events[cam_id]["reco_energy"] = reco_energy
                feature_events[cam_id]["ellipticity"] = ellipticity.value
                feature_events[cam_id]["n_cluster"] = n_cluster_dict[tel_id]
                feature_events[cam_id]["n_tel_reco"] = n_tels["reco"]
                feature_events[cam_id]["n_tel_discri"] = n_tels["discri"]
                feature_events[cam_id]["mc_core_x"] = event.mc.core_x.to("m").value
                feature_events[cam_id]["mc_core_y"] = event.mc.core_y.to("m").value
                feature_events[cam_id]["reco_core_x"] = reco_core_x.to("m").value
                feature_events[cam_id]["reco_core_y"] = reco_core_y.to("m").value
                feature_events[cam_id]["mc_h_first_int"] = event.mc.h_first_int.to(
                    "m"
                ).value
                feature_events[cam_id]["offset"] = offset.to("deg").value
                feature_events[cam_id]["mc_x_max"] = event.mc.x_max.value  # g / cm2
                feature_events[cam_id]["alt"] = reco_result.alt.to("deg").value
                feature_events[cam_id]["az"] = reco_result.az.to("deg").value
                feature_events[cam_id]["reco_energy_tel"] = reco_energy_tel[tel_id]
                # Variables from hillas_dist_reco
                feature_events[cam_id]["ellipticity_reco"] = ellipticity_reco.value
                feature_events[cam_id]["local_distance_reco"] = moments_reco.r.to(
                    "m"
                ).value
                feature_events[cam_id]["skewness_reco"] = moments_reco.skewness
                feature_events[cam_id]["kurtosis_reco"] = moments_reco.kurtosis
                feature_events[cam_id]["width_reco"] = moments_reco.width.to("m").value
                feature_events[cam_id]["length_reco"] = moments_reco.length.to(
                    "m"
                ).value
                feature_events[cam_id]["psi_reco"] = moments_reco.psi.to("deg").value
                feature_events[cam_id]["sum_signal_cam_reco"] = moments_reco.intensity

                feature_events[cam_id].append()

            if signal_handler.stop:
                break
        if signal_handler.stop:
            break
    # make sure that all the events are properly stored
    for table in feature_table.values():
        table.flush()

    img_cutflow()
    evt_cutflow()


if __name__ == "__main__":
    main()
