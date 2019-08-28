import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
import warnings
from traitlets.config import Config
from collections import namedtuple, OrderedDict

# CTAPIPE utilities
from ctapipe.calib import CameraCalibrator
from ctapipe.image.extractor import LocalPeakWindowSum
from ctapipe.image import hillas
from ctapipe.utils.CutFlow import CutFlow
from ctapipe.coordinates import GroundFrame

# from ctapipe.image.hillas import hillas_parameters_5 as hillas_parameters
from ctapipe.image.hillas import hillas_parameters
from ctapipe.reco.HillasReconstructor import HillasReconstructor

# monkey patch the camera calibrator to do NO integration correction
# import ctapipe


# def null_integration_correction_func(
#     n_chan, pulse_shape, refstep, time_slice, window_width, window_shift
# ):
#     return np.ones(n_chan)


# apply the patch
# ctapipe.calib.camera.dl1.integration_correction = null_integration_correction_func

# PiWy utilities
try:
    from pywicta.io import geometry_converter
    from pywicta.io.images import simtel_event_to_images
    from pywi.processing.filtering import pixel_clusters
    from pywi.processing.filtering.pixel_clusters import filter_pixels_clusters
except ImportError as e:
    print("pywicta package could not be imported")
    print("wavelet cleaning will not work")
    print(e)

# Pipeline utilities
from .image_cleaning import ImageCleaner

__all__ = ["EventPreparer"]

PreparedEvent = namedtuple(
    "PreparedEvent",
    [
        "event",
        "n_pixel_dict",
        "hillas_dict",
        "hillas_dict_reco",
        "n_tels",
        "tot_signal",
        "max_signals",
        "n_cluster_dict",
        "reco_result",
        "impact_dict",
    ],
)


def stub(event):
    return PreparedEvent(
        event=event,
        n_pixel_dict=None,
        hillas_dict=None,
        hillas_dict_reco=None,
        n_tels=None,
        tot_signal=None,
        max_signals=None,
        n_cluster_dict=None,
        reco_result=None,
        impact_dict=None,
    )


class EventPreparer:
    """Class which loop on events and returns results stored in container.

    The Class has several purposes. First of all, it prepares the images of the
    event that will be further use for reconstruction by applying calibration,
    cleaning and selection. Then, it reconstructs the geometry of the event and
    then returns image (e.g. Hillas parameters)and event information
    (e.g. results of the reconstruction).

    Parameters
    ----------
    config: dict
        Configuration with analysis parameters
    mode: str
        Mode of the reconstruction, e.g. tail or wave
    event_cutflow: ctapipe.utils.CutFlow
        Statistic of events processed
    image_cutflow: ctapipe.utils.CutFlow
        Statistic of images processed

    Returns: dict
        Dictionnary of results
    """

    def __init__(self, config, mode, event_cutflow=None, image_cutflow=None):
        # Cleaning for reconstruction
        self.cleaner_reco = ImageCleaner(  # for reconstruction
            config=config["ImageCleaning"]["biggest"], mode=mode
        )

        # Cleaning for energy/score estimation
        # Add possibility to force energy/score cleaning with tailcut analysis
        force_mode = mode
        try:
            if config["General"]["force_tailcut_for_extended_cleaning"] is True:
                force_mode = config["General"]["force_mode"]
                print("> Activate force-mode for cleaning!!!!")
        except:
            pass  # force_mode = mode

        self.cleaner_extended = ImageCleaner(  # for energy/score estimation
            config=config["ImageCleaning"]["extended"], mode=force_mode
        )

        # Image book keeping
        self.image_cutflow = image_cutflow or CutFlow("ImageCutFlow")

        # Add quality cuts on images
        charge_bounds = config["ImageSelection"]["charge"]
        npix_bounds = config["ImageSelection"]["pixel"]
        ellipticity_bounds = config["ImageSelection"]["ellipticity"]
        nominal_distance_bounds = config["ImageSelection"]["nominal_distance"]

        self.camera_radius = {
            "LSTCam": 1.126,
            "NectarCam": 1.126,
        }  # Average between max(xpix) and max(ypix), in meter

        self.image_cutflow.set_cuts(
            OrderedDict(
                [
                    ("noCuts", None),
                    ("min pixel", lambda s: np.count_nonzero(s) < npix_bounds[0]),
                    ("min charge", lambda x: x < charge_bounds[0]),
                    # ("poor moments", lambda m: m.width <= 0 or m.length <= 0 or np.isnan(m.width) or np.isnan(m.length)),
                    # TBC, maybe we loose events without nan conditions
                    ("poor moments", lambda m: m.width <= 0 or m.length <= 0),
                    (
                        "bad ellipticity",
                        lambda m: (m.width / m.length) < ellipticity_bounds[0]
                        or (m.width / m.length) > ellipticity_bounds[-1],
                    ),
                    # ("close to the edge", lambda m, cam_id: m.r.value > (nominal_distance_bounds[-1] * 1.12949101073069946))
                    # in meter
                    (
                        "close to the edge",
                        lambda m, cam_id: m.r.value
                        > (nominal_distance_bounds[-1] * self.camera_radius[cam_id]),
                    ),  # in meter
                ]
            )
        )

        cfg = Config()
        cfg["ChargeExtractorFactory"]["window_width"] = 5
        cfg["ChargeExtractorFactory"]["window_shift"] = 2
        extractor = LocalPeakWindowSum(config=cfg)

        self.calib = CameraCalibrator(
            config=cfg,
            image_extractor=extractor,
            # eventsource=source,
            # tool=None,
        )

        # Reconstruction
        self.shower_reco = HillasReconstructor()

        # Event book keeping
        self.event_cutflow = event_cutflow or CutFlow("EventCutFlow")

        # Add cuts on events
        min_ntel = config["Reconstruction"]["min_tel"]
        self.event_cutflow.set_cuts(
            OrderedDict(
                [
                    ("noCuts", None),
                    ("min2Tels trig", lambda x: x < min_ntel),
                    ("min2Tels reco", lambda x: x < min_ntel),
                    ("direction nan", lambda x: x.is_valid == False),
                ]
            )
        )

    def prepare_event(self, source, return_stub=False):

        # configuration for the camera calibrator
        # modifies the integration window to be more like in MARS
        # JLK, only for LST!!!!
        # Option for integration correction is done above

        for event in source:

            self.event_cutflow.count("noCuts")

            if self.event_cutflow.cut("min2Tels trig", len(event.dl0.tels_with_data)):
                if return_stub:
                    yield stub(event)
                else:
                    continue

            # TEST : print correction factor for event
            for tel in event.dl0.tels_with_data:
                print(
                    f"Correction factor for telescope #{tel} : {self.calib._get_correction(event, tel)}"
                )
            # calibrate the event
            self.calib(event)

            # telescope loop
            tot_signal = 0
            max_signals = {}
            n_pixel_dict = {}
            hillas_dict_reco = {}  # for geometry
            hillas_dict = {}  # for discrimination
            n_tels = {
                "tot": len(event.dl0.tels_with_data),
                "LST_LST_LSTCam": 0,
                "MST_MST_NectarCam": 0,
                "SST": 0,  # will change later wheh working on Paranal
            }
            n_cluster_dict = {}
            impact_dict_reco = {}  # impact distance measured in tilt system

            point_azimuth_dict = {}
            point_altitude_dict = {}

            # To compute impact parameter in tilt system
            run_array_direction = event.mcheader.run_array_direction
            az, alt = run_array_direction[0], run_array_direction[1]

            ground_frame = GroundFrame()

            for tel_id in event.dl0.tels_with_data:
                self.image_cutflow.count("noCuts")

                camera = event.inst.subarray.tel[tel_id].camera

                # count the current telescope according to its size
                tel_type = str(event.inst.subarray.tel[tel_id])
                # JLK, N telescopes before cut selection are not really interesting for
                # discrimination, too much fluctuations
                # n_tels[tel_type] += 1

                # the camera image as a 1D array and stuff needed for calibration
                # Choose gain according to pywicta's procedure

                # image_1d = simtel_event_to_images(
                #    event=event, tel_id=tel_id, ctapipe_format=True
                # )
                # pmt_signal = image_1d.input_image  # calibrated image
                pmt_signal = event.dl1.tel[tel_id].image

                # clean the image
                try:
                    with warnings.catch_warnings():
                        # Image with biggest cluster (reco cleaning)
                        image_biggest = self.cleaner_reco.clean_image(
                            pmt_signal, camera
                        )
                        image_biggest2d = geometry_converter.image_1d_to_2d(
                            image_biggest, camera.cam_id
                        )
                        image_biggest2d = filter_pixels_clusters(image_biggest2d)
                        image_biggest = geometry_converter.image_2d_to_1d(
                            image_biggest2d, camera.cam_id
                        )

                        # Image for score/energy estimation (with clusters)
                        image_extended = self.cleaner_extended.clean_image(
                            pmt_signal, camera
                        )
                except FileNotFoundError as e:  # JLK, WHAT?
                    print(e)
                    continue

                # Apply some selection
                if self.image_cutflow.cut("min pixel", image_biggest):
                    continue

                if self.image_cutflow.cut("min charge", np.sum(image_biggest)):
                    continue

                # For cluster counts
                image_2d = geometry_converter.image_1d_to_2d(
                    image_extended, camera.cam_id
                )
                n_cluster_dict[tel_id] = pixel_clusters.number_of_pixels_clusters(
                    array=image_2d, threshold=0
                )

                # could this go into `hillas_parameters` ...?
                max_signals[tel_id] = np.max(image_extended)

                # do the hillas reconstruction of the images
                # QUESTION should this change in numpy behaviour be done here
                # or within `hillas_parameters` itself?
                # JLK: make selection on biggest cluster
                with np.errstate(invalid="raise", divide="raise"):
                    try:

                        moments_reco = hillas_parameters(
                            camera, image_biggest
                        )  # for geometry (eg direction)
                        moments = hillas_parameters(
                            camera, image_extended
                        )  # for discrimination and energy reconstruction

                        # if width and/or length are zero (e.g. when there is only only one
                        # pixel or when all  pixel are exactly in one row), the
                        # parametrisation won't be very useful: skip
                        if self.image_cutflow.cut("poor moments", moments_reco):
                            # print('poor moments')
                            continue

                        if self.image_cutflow.cut(
                            "close to the edge", moments_reco, camera.cam_id
                        ):
                            # print('close to the edge')
                            continue

                        if self.image_cutflow.cut("bad ellipticity", moments_reco):
                            # print('bad ellipticity: w={}, l={}'.format(moments_reco.width, moments_reco.length))
                            continue

                    except (FloatingPointError, hillas.HillasParameterizationError):
                        continue

                point_azimuth_dict[tel_id] = event.mc.tel[tel_id].azimuth_raw * u.rad
                point_altitude_dict[tel_id] = event.mc.tel[tel_id].altitude_raw * u.rad

                n_tels[tel_type] += 1
                hillas_dict[tel_id] = moments
                hillas_dict_reco[tel_id] = moments_reco
                n_pixel_dict[tel_id] = len(np.where(image_extended > 0)[0])
                tot_signal += moments.intensity

            n_tels["reco"] = len(hillas_dict_reco)
            n_tels["discri"] = len(hillas_dict)
            if self.event_cutflow.cut("min2Tels reco", n_tels["reco"]):
                if return_stub:
                    yield stub(event)
                else:
                    continue

            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")

                    # Reconstruction results
                    reco_result = self.shower_reco.predict(
                        hillas_dict_reco,
                        event.inst,
                        SkyCoord(alt=alt, az=az, frame="altaz"),
                        {
                            tel_id: SkyCoord(
                                alt=point_altitude_dict[tel_id],
                                az=point_azimuth_dict[tel_id],
                                frame="altaz",
                            )
                            for tel_id in event.dl0.tels_with_data
                        },
                    )

                    # reco_result = self.shower_reco.predict(
                    #     hillas_dict_reco,
                    #     event.inst,
                    #     point_altitude_dict,  # this needs to be changed using ctapipe07!
                    #     point_azimuth_dict,  # this needs to be changed using ctapipe07!
                    # )

                    # shower_sys = TiltedGroundFrame(pointing_direction=HorizonFrame(
                    #     az=reco_result.az,
                    #     alt=reco_result.alt
                    # ))

                    # Impact parameter for energy estimation (/ tel)
                    subarray = event.inst.subarray
                    for tel_id in hillas_dict.keys():

                        pos = subarray.positions[tel_id]

                        tel_ground = SkyCoord(
                            pos[0], pos[1], pos[2], frame=ground_frame
                        )
                        # tel_tilt = tel_ground.transform_to(shower_sys)

                        core_ground = SkyCoord(
                            reco_result.core_x,
                            reco_result.core_y,
                            0 * u.m,
                            frame=ground_frame,
                        )
                        # core_tilt = core_ground.transform_to(shower_sys)

                        # Should be better handled (tilted frame)
                        impact_dict_reco[tel_id] = np.sqrt(
                            (core_ground.x - tel_ground.x) ** 2
                            + (core_ground.y - tel_ground.y) ** 2
                        )

            except Exception as e:
                print("exception in reconstruction:", e)
                raise
                if return_stub:
                    yield stub(event)
                else:
                    continue

            if self.event_cutflow.cut("direction nan", reco_result):
                if return_stub:
                    yield stub(event)
                else:
                    continue

            yield PreparedEvent(
                event=event,
                n_pixel_dict=n_pixel_dict,
                hillas_dict=hillas_dict,
                hillas_dict_reco=hillas_dict_reco,
                n_tels=n_tels,
                tot_signal=tot_signal,
                max_signals=max_signals,
                n_cluster_dict=n_cluster_dict,
                reco_result=reco_result,
                impact_dict=impact_dict_reco,
            )
