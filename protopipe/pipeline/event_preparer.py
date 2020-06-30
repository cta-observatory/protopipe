"""Calibrate, clean the image, and reconstruct the direction of an event."""
import math
import numpy as np

from astropy import units as u
from astropy.coordinates import SkyCoord
import warnings
from traitlets.config import Config
from collections import namedtuple, OrderedDict

# CTAPIPE utilities
from ctapipe.calib import CameraCalibrator
from ctapipe.calib.camera.gainselection import GainSelector
from ctapipe.image.extractor import TwoPassWindowSum
from ctapipe.image import hillas, leakage, number_of_islands, largest_island
from ctapipe.utils.CutFlow import CutFlow
from ctapipe.coordinates import GroundFrame

# from ctapipe.image.timing_parameters import timing_parameters
from ctapipe.image.hillas import hillas_parameters
from ctapipe.reco.HillasReconstructor import HillasReconstructor

# Pipeline utilities
from .image_cleaning import ImageCleaner

# PiWy utilities
try:
    from pywicta.io import geometry_converter

    # from pywicta.io.images import simtel_event_to_images
    from pywi.processing.filtering import pixel_clusters
    from pywi.processing.filtering.pixel_clusters import filter_pixels_clusters
except ImportError as e:
    print("pywicta not imported-->(no wavelet cleaning)")
    print(e)

__all__ = ["EventPreparer"]

PreparedEvent = namedtuple(
    "PreparedEvent",
    [
        "event",
        "dl1_phe_image",
        "dl1_phe_image_mask_reco",
        "dl1_phe_image_1stPass",
        "calibration_status",
        "mc_phe_image",
        "n_pixel_dict",
        "hillas_dict",
        "hillas_dict_reco",
        "leakage_dict",
        "n_tels",
        "tot_signal",
        "max_signals",
        "n_cluster_dict",
        "reco_result",
        "impact_dict",
    ],
)


def camera_radius(camid_to_efl, cam_id="all"):
    """
    Inspired from pywi-cta CTAMarsCriteria, CTA Mars like preselection cuts.
    This should be replaced by a function in ctapipe getting the radius either
    from  the pixel poisitions or from an external database
    Note
    ----
    average_camera_radius_meters = math.tan(math.radians(average_camera_radius_degree)) * foclen
    The average camera radius values are, in degrees :
    - LST: 2.31
    - Nectar: 4.05
    - Flash: 3.95
    - SST-1M: 4.56
    - GCT-CHEC-S: 3.93
    - ASTRI: 4.67

    """

    average_camera_radii_deg = {
        "ASTRICam": 4.67,
        "CHEC": 3.93,
        "DigiCam": 4.56,
        "FlashCam": 3.95,
        "NectarCam": 4.05,
        "LSTCam": 2.31,
        "SCTCam": 4.0,  # dummy value
    }

    if cam_id in camid_to_efl.keys():
        foclen_meters = camid_to_efl[cam_id]
        average_camera_radius_meters = (
            math.tan(math.radians(average_camera_radii_deg[cam_id])) * foclen_meters
        )
    elif cam_id == "all":
        print("Available camera radii in meters:")
        for cam_id in camid_to_efl.keys():
            print(f"* {cam_id} : ", camera_radius(camid_to_efl, cam_id))
        average_camera_radius_meters = 0
    else:
        raise ValueError("Unknown camid", cam_id)

    return average_camera_radius_meters


def stub(event):
    """Default container for images that did not survive cleaning."""
    return PreparedEvent(
        event=event,
        dl1_phe_image=None,  # container for the calibrated image in phe
        dl1_phe_image_mask_reco=None,  # container for the reco cleaning mask
        mc_phe_image=None,  # container for the simulated image in phe
        n_pixel_dict=None,
        hillas_dict=None,
        hillas_dict_reco=None,
        leakage_dict=None,
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

    def __init__(
        self,
        config,
        subarray,
        cams_and_foclens,
        mode,
        event_cutflow=None,
        image_cutflow=None,
        debug=False,
    ):
        """Initiliaze an EventPreparer object."""
        # Cleaning for reconstruction
        self.cleaner_reco = ImageCleaner(  # for reconstruction
            config=config["ImageCleaning"]["biggest"],
            cameras=cams_and_foclens.keys(),
            mode=mode,
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
            config=config["ImageCleaning"]["extended"],
            cameras=cams_and_foclens.keys(),
            mode=force_mode,
        )

        # Image book keeping
        self.image_cutflow = image_cutflow or CutFlow("ImageCutFlow")

        # Add quality cuts on images
        charge_bounds = config["ImageSelection"]["charge"]
        npix_bounds = config["ImageSelection"]["pixel"]
        ellipticity_bounds = config["ImageSelection"]["ellipticity"]
        nominal_distance_bounds = config["ImageSelection"]["nominal_distance"]

        if debug:
            camera_radius(
                cams_and_foclens, "all"
            )  # Display all registered camera radii

        self.camera_radius = {
            cam_id: camera_radius(cams_and_foclens, cam_id)
            for cam_id in cams_and_foclens.keys()
        }

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

        # Configuration for the camera calibrator

        cfg = Config()

        extractor = TwoPassWindowSum(config=cfg, subarray=subarray)
        # Get the name of the image extractor in order to adapt some options
        # specific to TwoPassWindowSum later on
        self.extractorName = list(extractor.get_current_config().items())[0][0]

        self.calib = CameraCalibrator(
            config=cfg, image_extractor=extractor, subarray=subarray,
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
                    ("direction nan", lambda x: x.is_valid is False),
                ]
            )
        )

    def prepare_event(self, source, return_stub=False, save_images=False, debug=False):
        """
        Calibrate, clean and reconstruct the direction of an event.

        Parameters
        ----------
        source : ctapipe.io.EventSource
            A container of selected showers from a simtel file.
        return_stub : bool
            If True, yield also images from events that won't be reconstructed.
            This feature is not currently available.
        save_images : bool
            If True, save photoelectron images from reconstructed events.
        debug : bool
            If True, print some debugging information (to be expanded).

        Yields
        ------
        PreparedEvent: dict
            Dictionary containing event-image information to be written.

        """
        ievt = 0
        for event in source:

            # Display event counts
            if debug:
                ievt += 1
                if (ievt < 10) or (ievt % 10 == 0):
                    print(ievt)

            self.event_cutflow.count("noCuts")

            if self.event_cutflow.cut("min2Tels trig", len(event.dl0.tels_with_data)):
                if return_stub:
                    yield stub(event)
                else:
                    continue

            self.calib(event)  # Calibrate the event

            # telescope loop
            tot_signal = 0
            dl1_phe_image = {}
            dl1_phe_image_mask_reco = {}
            dl1_phe_image_1stPass = {}
            calibration_status = {}
            mc_phe_image = {}
            max_signals = {}
            n_pixel_dict = {}
            hillas_dict_reco = {}  # for direction reconstruction
            hillas_dict = {}  # for discrimination
            leakage_dict = {}
            n_tels = {
                "tot": len(event.dl0.tels_with_data),
                "LST_LST_LSTCam": 0,
                "MST_MST_NectarCam": 0,
                "MST_MST_FlashCam": 0,
                "MST_SCT_SCTCam": 0,
                "SST_1M_DigiCam": 0,
                "SST_ASTRI_ASTRICam": 0,
                "SST_GCT_CHEC": 0,
            }
            n_cluster_dict = {}
            impact_dict_reco = {}  # impact distance measured in tilt system

            point_azimuth_dict = {}
            point_altitude_dict = {}

            # Compute impact parameter in tilt system
            run_array_direction = event.mcheader.run_array_direction
            az, alt = run_array_direction[0], run_array_direction[1]

            ground_frame = GroundFrame()

            for tel_id in event.dl0.tels_with_data:
                self.image_cutflow.count("noCuts")

                camera = source.subarray.tel[tel_id].camera.geometry

                # count the current telescope according to its size
                tel_type = str(source.subarray.tel[tel_id])

                # use ctapipe's functionality to get the calibrated image
                pmt_signal = event.dl1.tel[tel_id].image

                # Save the calibrated image after the gain has been chosen
                # automatically by ctapipe, together with the simulated one
                if save_images is True:
                    dl1_phe_image[tel_id] = pmt_signal
                    dl1_phe_image_1stPass[tel_id] = None
                    mc_phe_image[tel_id] = event.mc.tel[tel_id].true_image

                if self.cleaner_reco.mode == "tail":  # tail uses only ctapipe

                    # Cleaning used for direction reconstruction
                    image_biggest, mask_reco = self.cleaner_reco.clean_image(
                        pmt_signal, camera
                    )

                    # calculate the leakage (before filtering)
                    leakages = {}  # this is needed by both cleanings
                    # The check on SIZE shouldn't be here, but for the moment
                    # I prefer to sacrifice fancincess
                    if np.sum(image_biggest[mask_reco]) != 0.0:
                        leakage_biggest = leakage(camera, image_biggest, mask_reco)
                        leakages["leak1_reco"] = leakage_biggest["intensity_width_1"]
                        leakages["leak2_reco"] = leakage_biggest["intensity_width_2"]
                    else:
                        leakages["leak1_reco"] = 0.0
                        leakages["leak2_reco"] = 0.0

                    # find all islands using this cleaning
                    num_islands, labels = number_of_islands(camera, mask_reco)

                    if num_islands == 1:  # if only ONE islands is left ...
                        # ...use directly the old mask and reduce dimensions
                        # to make Hillas parametrization faster
                        camera_biggest = camera[mask_reco]
                        image_biggest = image_biggest[mask_reco]
                        if save_images is True:
                            dl1_phe_image_mask_reco[tel_id] = mask_reco

                    elif num_islands > 1:  # if more islands survived..
                        # ...find the biggest one
                        mask_biggest = largest_island(labels)
                        # and also reduce dimensions
                        camera_biggest = camera[mask_biggest]
                        image_biggest = image_biggest[mask_biggest]
                        if save_images is True:
                            dl1_phe_image_mask_reco[tel_id] = mask_biggest

                    else:  # if no islands survived use old camera and image
                        camera_biggest = camera

                    # Cleaning used for score/energy estimation
                    image_extended, mask_extended = self.cleaner_extended.clean_image(
                        pmt_signal, camera
                    )

                    # calculate the leakage (before filtering)
                    # this part is not well coded, but for the moment it works
                    if np.sum(image_extended[mask_extended]) != 0.0:
                        leakage_extended = leakage(
                            camera, image_extended, mask_extended
                        )
                        leakages["leak1"] = leakage_extended["intensity_width_1"]
                        leakages["leak2"] = leakage_extended["intensity_width_2"]
                    else:
                        leakages["leak1"] = 0.0
                        leakages["leak2"] = 0.0

                    # find all islands with this cleaning
                    # we will also register how many have been found
                    n_cluster_dict[tel_id], labels = number_of_islands(
                        camera, mask_extended
                    )

                    # NOTE: the next check shouldn't be necessary if we keep
                    # all the isolated pixel clusters, but for now the
                    # two cleanings are set the same in analysis.yml because
                    # the performance of the extended one has never been really
                    # studied in model estimation.
                    # (This is a nice way to ask for volunteers :P)

                    # if some islands survived

                    if n_cluster_dict[tel_id] > 0:
                        # keep all of them and reduce dimensions
                        camera_extended = camera[mask_extended]
                        image_extended = image_extended[mask_extended]
                    else:  # otherwise continue with the old camera and image
                        camera_extended = camera

                    # could this go into `hillas_parameters` ...?
                    # this is basically the charge of ALL islands
                    # not calculated later by the Hillas parametrization!
                    max_signals[tel_id] = np.max(image_extended)

                else:  # for wavelets we stick to old pywi-cta code
                    try:  # "try except FileNotFoundError" not clear to me, but for now it stays...
                        with warnings.catch_warnings():
                            # Image with biggest cluster (reco cleaning)
                            image_biggest, mask_reco = self.cleaner_reco.clean_image(
                                pmt_signal, camera
                            )
                            image_biggest2d = geometry_converter.image_1d_to_2d(
                                image_biggest, camera.camera_name
                            )
                            image_biggest2d = filter_pixels_clusters(image_biggest2d)
                            image_biggest = geometry_converter.image_2d_to_1d(
                                image_biggest2d, camera.camera_name
                            )

                            # Image for score/energy estimation (with clusters)
                            (
                                image_extended,
                                mask_extended,
                            ) = self.cleaner_extended.clean_image(pmt_signal, camera)

                            # This last part was outside the pywi-cta block
                            # before, but is indeed part of it because it uses
                            # pywi-cta functions in the "extended" case

                            # For cluster counts
                            image_2d = geometry_converter.image_1d_to_2d(
                                image_extended, camera.camera_name
                            )
                            n_cluster_dict[
                                tel_id
                            ] = pixel_clusters.number_of_pixels_clusters(
                                array=image_2d, threshold=0
                            )
                            # could this go into `hillas_parameters` ...?
                            max_signals[tel_id] = np.max(image_extended)

                    except FileNotFoundError as e:  # JLK, WHAT?
                        print(e)
                        continue

                # ==============================================================

                # Apply some selection
                if self.image_cutflow.cut("min pixel", image_biggest):
                    continue

                if self.image_cutflow.cut("min charge", np.sum(image_biggest)):
                    continue

                # do the hillas reconstruction of the images
                # QUESTION should this change in numpy behaviour be done here
                # or within `hillas_parameters` itself?
                # JLK: make selection on biggest cluster
                with np.errstate(invalid="raise", divide="raise"):
                    try:

                        moments_reco = hillas_parameters(
                            camera_biggest, image_biggest
                        )  # for geometry (eg direction)
                        moments = hillas_parameters(
                            camera_extended, image_extended
                        )  # for discrimination and energy reconstruction

                        # if width and/or length are zero (e.g. when there is
                        # only only one pixel or when all  pixel are exactly
                        # in one row), the parametrisation
                        # won't be very useful: skip
                        if self.image_cutflow.cut("poor moments", moments_reco):
                            continue

                        if self.image_cutflow.cut(
                            "close to the edge", moments_reco, camera.camera_name
                        ):
                            continue

                        if self.image_cutflow.cut("bad ellipticity", moments_reco):
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
                leakage_dict[tel_id] = leakages

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
                        source.subarray,
                        SkyCoord(alt=alt, az=az, frame="altaz"),
                        {
                            tel_id: SkyCoord(
                                alt=point_altitude_dict[tel_id],
                                az=point_azimuth_dict[tel_id],
                                frame="altaz",
                            )  # cycle only on tels which still have an image
                            for tel_id in point_altitude_dict.keys()
                        },
                    )

                    # Impact parameter for energy estimation (/ tel)
                    subarray = source.subarray
                    for tel_id in hillas_dict.keys():

                        pos = subarray.positions[tel_id]

                        tel_ground = SkyCoord(
                            pos[0], pos[1], pos[2], frame=ground_frame
                        )

                        core_ground = SkyCoord(
                            reco_result.core_x,
                            reco_result.core_y,
                            0 * u.m,
                            frame=ground_frame,
                        )

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
                dl1_phe_image=dl1_phe_image,
                dl1_phe_image_mask_reco=dl1_phe_image_mask_reco,
                dl1_phe_image_1stPass=dl1_phe_image_1stPass,
                calibration_status=calibration_status,
                mc_phe_image=mc_phe_image,
                n_pixel_dict=n_pixel_dict,
                hillas_dict=hillas_dict,
                hillas_dict_reco=hillas_dict_reco,
                leakage_dict=leakage_dict,
                n_tels=n_tels,
                tot_signal=tot_signal,
                max_signals=max_signals,
                n_cluster_dict=n_cluster_dict,
                reco_result=reco_result,
                impact_dict=impact_dict_reco,
            )
