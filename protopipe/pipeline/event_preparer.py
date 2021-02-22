"""Calibrate, clean the image, and reconstruct the direction of an event."""
import numpy as np

from astropy import units as u
from astropy.coordinates import SkyCoord
import warnings
from traitlets.config import Config
from collections import namedtuple, OrderedDict

# CTAPIPE utilities
from ctapipe.containers import ReconstructedShowerContainer
from ctapipe.calib import CameraCalibrator
from ctapipe.image.extractor import TwoPassWindowSum
from ctapipe.image import leakage, number_of_islands, largest_island
from ctapipe.utils.CutFlow import CutFlow
from ctapipe.coordinates import GroundFrame, TelescopeFrame, CameraFrame

# from ctapipe.image.timing_parameters import timing_parameters
from ctapipe.image.hillas import hillas_parameters, HillasParameterizationError
from ctapipe.reco.reco_algorithms import (
    TooFewTelescopesException,
    InvalidWidthException,
)

# Pipeline utilities
from .image_cleaning import ImageCleaner
from .utils import bcolors, effective_focal_lengths, camera_radius, CTAMARS_radii
from .temp import (
    HillasParametersTelescopeFrameContainer,
    MyCameraGeometry,
    MyHillasReconstructor,
)

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
        "dl1_phe_image_mask_clusters",
        "mc_phe_image",
        "n_pixel_dict",
        "hillas_dict",
        "hillas_dict_reco",
        "leakage_dict",
        "n_tels",
        "max_signals",
        "n_cluster_dict",
        "reco_result",
        "impact_dict",
        "good_event",
        "good_for_reco",
    ],
)


def stub(
    event,
    true_image,
    image,
    cleaning_mask_reco,
    cleaning_mask_clusters,
    good_for_reco,
    hillas_dict,
    hillas_dict_reco,
    n_tels,
    leakage_dict,
):
    """Default container for images that did not survive cleaning."""
    return PreparedEvent(
        event=event,
        dl1_phe_image=image,  # container for the calibrated image in phe
        dl1_phe_image_mask_reco=cleaning_mask_reco,  # container for the reco cleaning mask
        dl1_phe_image_mask_clusters=cleaning_mask_clusters,
        mc_phe_image=true_image,  # container for the simulated image in phe
        n_pixel_dict=dict.fromkeys(hillas_dict_reco.keys(), 0),
        good_for_reco=good_for_reco,
        hillas_dict=hillas_dict,
        hillas_dict_reco=hillas_dict_reco,
        leakage_dict=leakage_dict,
        n_tels=n_tels,
        max_signals=dict.fromkeys(hillas_dict_reco.keys(), np.nan),  # no charge
        n_cluster_dict=dict.fromkeys(hillas_dict_reco.keys(), 0),  # no clusters
        reco_result=ReconstructedShowerContainer(),  # defaults to nans
        impact_dict=dict.fromkeys(
            hillas_dict_reco, np.nan * u.m
        ),  # undefined impact parameter
        good_event=False,
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

        # Radii in meters from CTA-MARS
        # self.camera_radius = {
        #     cam_id: camera_radius(cams_and_foclens, cam_id)
        #     for cam_id in cams_and_foclens.keys()
        # }

        # Radii in degrees from CTA-MARS
        self.camera_radius = {
            cam_id: CTAMARS_radii(cam_id) for cam_id in cams_and_foclens.keys()
        }

        self.image_cutflow.set_cuts(
            OrderedDict(
                [
                    ("noCuts", None),
                    ("min pixel", lambda s: np.count_nonzero(s) < npix_bounds[0]),
                    ("min charge", lambda x: x < charge_bounds[0]),
                    (
                        "poor moments",
                        lambda m: m.width <= 0
                        or m.length <= 0
                        or np.isnan(m.width)
                        or np.isnan(m.length),
                    ),
                    (
                        "bad ellipticity",
                        lambda m: (m.width / m.length) < ellipticity_bounds[0]
                        or (m.width / m.length) > ellipticity_bounds[-1],
                    ),
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
        self.shower_reco = MyHillasReconstructor()

        # Event book keeping
        self.event_cutflow = event_cutflow or CutFlow("EventCutFlow")

        # Add cuts on events
        self.min_ntel = config["Reconstruction"]["min_tel"]
        self.event_cutflow.set_cuts(
            OrderedDict(
                [
                    ("noCuts", None),
                    ("min2Tels trig", lambda x: x < self.min_ntel),
                    ("min2Tels reco", lambda x: x < self.min_ntel),
                    ("direction nan", lambda x: x.is_valid is False),
                ]
            )
        )

    def prepare_event(self, source, return_stub=True, save_images=False, debug=False):
        """
        Calibrate, clean and reconstruct the direction of an event.

        Parameters
        ----------
        source : ctapipe.io.EventSource
            A container of selected showers from a simtel file.
        geom_cam_tel: dict
            Dictionary of of MyCameraGeometry objects for each camera in the file
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

        # =============================================================
        #                TRANSFORMED CAMERA GEOMETRIES
        # =============================================================

        # These are the camera geometries were the Hillas parametrization will
        # be performed.
        # They are transformed to TelescopeFrame using the effective focal
        # lengths

        # These geometries could be used also to performe the image cleaning,
        # but for the moment we still do that in the CameraFrame

        geom_cam_tel = {}
        for camera in source.subarray.camera_types:

            # Original geometry of each camera
            geom = camera.geometry
            # Same but with focal length as an attribute
            # This is planned to disappear and be imported by ctapipe
            focal_length = effective_focal_lengths(camera.camera_name)

            geom_cam_tel[camera.camera_name] = MyCameraGeometry(
                camera_name=camera.camera_name,
                pix_type=geom.pix_type,
                pix_id=geom.pix_id,
                pix_x=geom.pix_x,
                pix_y=geom.pix_y,
                pix_area=geom.pix_area,
                cam_rotation=geom.cam_rotation,
                pix_rotation=geom.pix_rotation,
                frame=CameraFrame(focal_length=focal_length),
            ).transform_to(TelescopeFrame())

        # =============================================================

        ievt = 0
        for event in source:

            # Display event counts
            if debug:
                print(
                    bcolors.BOLD
                    + f"EVENT #{event.count}, EVENT_ID #{event.index.event_id}"
                    + bcolors.ENDC
                )
                print(
                    bcolors.BOLD
                    + f"has triggered telescopes {event.r0.tels_with_data}"
                    + bcolors.ENDC
                )
                ievt += 1
                # if (ievt < 10) or (ievt % 10 == 0):
                #     print(ievt)

            self.event_cutflow.count("noCuts")

            if self.event_cutflow.cut("min2Tels trig", len(event.dl0.tels_with_data)):
                if return_stub:
                    print(
                        bcolors.WARNING
                        + f"WARNING : < {self.min_tel} triggered telescopes!"
                        + bcolors.ENDC
                    )
                    # we show this, but we proceed to analyze it

            # =============================================================
            #                CALIBRATION
            # =============================================================

            if debug:
                print(
                    bcolors.OKBLUE
                    + "Extracting all calibrated images..."
                    + bcolors.ENDC
                )
            self.calib(event)  # Calibrate the event

            # =============================================================
            #                BEGINNING OF LOOP OVER TELESCOPES
            # =============================================================

            dl1_phe_image = {}
            dl1_phe_image_mask_reco = {}
            dl1_phe_image_mask_clusters = {}
            mc_phe_image = {}
            max_signals = {}
            n_pixel_dict = {}
            hillas_dict_reco = {}  # for direction reconstruction
            hillas_dict = {}  # for discrimination
            leakage_dict = {}
            n_tels = {
                "Triggered": len(event.dl0.tels_with_data),
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

            good_for_reco = {}  # 1 = success, 0 = fail

            # Compute impact parameter in tilt system
            run_array_direction = event.mcheader.run_array_direction
            az, alt = run_array_direction[0], run_array_direction[1]

            ground_frame = GroundFrame()

            for tel_id in event.dl0.tels_with_data:

                point_azimuth_dict[tel_id] = event.mc.tel[tel_id].azimuth_raw * u.rad
                point_altitude_dict[tel_id] = event.mc.tel[tel_id].altitude_raw * u.rad

                if debug:
                    print(
                        bcolors.OKBLUE
                        + f"Working on telescope #{tel_id}..."
                        + bcolors.ENDC
                    )

                self.image_cutflow.count("noCuts")

                camera = source.subarray.tel[tel_id].camera.geometry

                # count the current telescope according to its type
                tel_type = str(source.subarray.tel[tel_id])
                n_tels[tel_type] += 1

                # use ctapipe's functionality to get the calibrated image
                pmt_signal = event.dl1.tel[tel_id].image

                # If required...
                if save_images is True:
                    # Save the simulated and reconstructed image of the event
                    dl1_phe_image[tel_id] = pmt_signal
                    mc_phe_image[tel_id] = event.mc.tel[tel_id].true_image

                # We now ASSUME that the event will be good
                good_for_reco[tel_id] = 1
                # later we change to 0 if any condition is NOT satisfied

                if self.cleaner_reco.mode == "tail":  # tail uses only ctapipe

                    # Cleaning used for direction reconstruction
                    image_biggest, mask_reco = self.cleaner_reco.clean_image(
                        pmt_signal, camera
                    )

                    # calculate the leakage (before filtering)
                    leakages = {}  # this is needed by both cleanings
                    # The check on SIZE shouldn't be here, but for the moment
                    # I prefer to sacrifice elegancy...
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
                        dl1_phe_image_mask_reco[tel_id] = mask_reco

                    # Cleaning used for score/energy estimation
                    image_extended, mask_extended = self.cleaner_extended.clean_image(
                        pmt_signal, camera
                    )
                    dl1_phe_image_mask_clusters[tel_id] = mask_extended

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

                # =============================================================
                #                PRELIMINARY IMAGE SELECTION
                # =============================================================

                cleaned_image_is_good = True  # we assume this

                # Apply some selection
                if self.image_cutflow.cut("min pixel", image_biggest):
                    if debug:
                        print(
                            bcolors.WARNING
                            + "WARNING : not enough pixels!"
                            + bcolors.ENDC
                        )
                    good_for_reco[tel_id] = 0  # we record it as BAD
                    cleaned_image_is_good = False

                if self.image_cutflow.cut("min charge", np.sum(image_biggest)):
                    if debug:
                        print(
                            bcolors.WARNING
                            + "WARNING : not enough charge!"
                            + bcolors.ENDC
                        )
                    good_for_reco[tel_id] = 0  # we record it as BAD
                    cleaned_image_is_good = False

                if debug and (not cleaned_image_is_good):  # BAD image quality
                    print(
                        bcolors.WARNING
                        + "WARNING : The cleaned image didn't pass"
                        + " preliminary cuts.\n"
                        + "An attempt to parametrize it will be made,"
                        + " but the image will NOT be used for"
                        + " direction reconstruction."
                        + bcolors.ENDC
                    )

                # =============================================================
                #                   IMAGE PARAMETRIZATION
                # =============================================================

                with np.errstate(invalid="raise", divide="raise"):
                    try:

                        # Filter the cameras in TelescopeFrame with the same
                        # cleaning masks
                        camera_biggest_tel = geom_cam_tel[camera.camera_name][
                            camera_biggest.pix_id
                        ]
                        camera_extended_tel = geom_cam_tel[camera.camera_name][
                            camera_extended.pix_id
                        ]

                        # Parametrize the image in the TelescopeFrame
                        moments_reco = hillas_parameters(
                            camera_biggest_tel, image_biggest
                        )  # for geometry (eg direction)
                        moments = hillas_parameters(
                            camera_extended_tel, image_extended
                        )  # for discrimination and energy reconstruction

                        if debug:
                            print("Image parameters from main cluster cleaning:")
                            print(moments_reco)

                            print("Image parameters from all-clusters cleaning:")
                            print(moments)

                        # ===================================================
                        #             PARAMETRIZED IMAGE SELECTION
                        # ===================================================

                        # if width and/or length are zero (e.g. when there is
                        # only only one pixel or when all  pixel are exactly
                        # in one row), the parametrisation
                        # won't be very useful: skip
                        if self.image_cutflow.cut("poor moments", moments_reco):
                            if debug:
                                print(
                                    bcolors.WARNING
                                    + "WARNING : poor moments!"
                                    + bcolors.ENDC
                                )
                            good_for_reco[tel_id] = 0  # we record it as BAD

                        if self.image_cutflow.cut(
                            "close to the edge", moments_reco, camera.camera_name
                        ):
                            if debug:
                                print(
                                    bcolors.WARNING
                                    + "WARNING : out of containment radius!\n"
                                    + f"Camera radius = {self.camera_radius[camera.camera_name]}\n"
                                    + f"COG radius = {moments_reco.r}"
                                    + bcolors.ENDC
                                )

                            good_for_reco[tel_id] = 0

                        if self.image_cutflow.cut("bad ellipticity", moments_reco):
                            if debug:
                                print(
                                    bcolors.WARNING
                                    + "WARNING : bad ellipticity"
                                    + bcolors.ENDC
                                )
                            good_for_reco[tel_id] = 0

                        if debug and good_for_reco[tel_id] == 1:
                            print(
                                bcolors.OKGREEN
                                + "Image survived and correctly parametrized."
                                # + "\nIt will be used for direction reconstruction!"
                                + bcolors.ENDC
                            )
                        elif debug and good_for_reco[tel_id] == 0:
                            print(
                                bcolors.WARNING
                                + "Image not survived or "
                                + "not good enough for parametrization."
                                # + "\nIt will be NOT used for direction reconstruction, "
                                # + "BUT it's information will be recorded."
                                + bcolors.ENDC
                            )

                        hillas_dict[tel_id] = moments
                        hillas_dict_reco[tel_id] = moments_reco
                        n_pixel_dict[tel_id] = len(np.where(image_extended > 0)[0])
                        leakage_dict[tel_id] = leakages

                    except (
                        FloatingPointError,
                        HillasParameterizationError,
                        ValueError,
                    ) as e:
                        if debug:
                            print(
                                bcolors.FAIL
                                + "Parametrization error: "
                                + f"{e}\n"
                                + "Dummy parameters recorded."
                                + bcolors.ENDC
                            )
                        good_for_reco[tel_id] = 0
                        hillas_dict[tel_id] = HillasParametersTelescopeFrameContainer()
                        hillas_dict_reco[
                            tel_id
                        ] = HillasParametersTelescopeFrameContainer()
                        n_pixel_dict[tel_id] = len(np.where(image_extended > 0)[0])
                        leakage_dict[tel_id] = leakages

                # END OF THE CYCLE OVER THE TELESCOPES

            # =============================================================
            #                   DIRECTION RECONSTRUCTION
            # =============================================================

            # n_tels["reco"] = len(hillas_dict_reco)
            # n_tels["discri"] = len(hillas_dict)

            # convert dictionary in numpy array to get a "mask"
            images_status = np.asarray(list(good_for_reco.values()))
            # record how many images will used for reconstruction
            n_tels["GOOD images"] = len(np.extract(images_status == 1, images_status))
            n_tels["BAD images"] = n_tels["Triggered"] - n_tels["GOOD images"]

            if self.event_cutflow.cut("min2Tels reco", n_tels["GOOD images"]):
                if debug:
                    print(
                        bcolors.FAIL
                        + f"WARNING: < {self.min_ntel} ({n_tels['GOOD images']}) images remaining!"
                        + "\nWARNING : direction reconstruction is not possible!"
                        + bcolors.ENDC
                    )

                # create a dummy container for direction reconstruction
                reco_result = ReconstructedShowerContainer()

                if return_stub:  # if saving all events (default)
                    if debug:
                        print(bcolors.OKBLUE + "Recording event..." + bcolors.ENDC)
                        print(
                            bcolors.WARNING
                            + "WARNING: This is event shall NOT be used further along the pipeline."
                            + bcolors.ENDC
                        )
                    yield stub(  # record event with dummy info
                        event,
                        mc_phe_image,
                        dl1_phe_image,
                        dl1_phe_image_mask_reco,
                        dl1_phe_image_mask_clusters,
                        good_for_reco,
                        hillas_dict,
                        hillas_dict_reco,
                        n_tels,
                        leakage_dict,
                    )
                    continue
                else:
                    continue

            if debug:
                print(
                    bcolors.OKBLUE
                    + "Starting direction reconstruction..."
                    + bcolors.ENDC
                )

            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")

                    # use only the successfully parametrized images
                    # to reconstruct the direction of this event
                    successfull_hillas = np.where(images_status == 1)[0]
                    all_images = np.asarray(list(good_for_reco.keys()))
                    good_images = set(all_images[successfull_hillas])
                    good_hillas_dict_reco = {
                        k: v for k, v in hillas_dict_reco.items() if k in good_images
                    }

                    if debug:
                        print(
                            bcolors.PURPLE
                            + f"{len(good_hillas_dict_reco)} images will be "
                            + "used to recover the shower's direction..."
                            + bcolors.ENDC
                        )

                    # Reconstruction results
                    reco_result = self.shower_reco.predict(
                        good_hillas_dict_reco,
                        source.subarray,
                        SkyCoord(alt=alt, az=az, frame="altaz"),
                        None,  # use the array direction
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

            except (Exception, TooFewTelescopesException, InvalidWidthException) as e:
                if debug:
                    print("exception in reconstruction:", e)
                raise
                if return_stub:
                    if debug:
                        print(
                            bcolors.FAIL
                            + "Shower could NOT be correctly reconstructed! "
                            + "Recording event..."
                            + "WARNING: This is event shall NOT be used further along the pipeline."
                            + bcolors.ENDC
                        )

                    yield stub(  # record event with dummy info
                        event,
                        mc_phe_image,
                        dl1_phe_image,
                        dl1_phe_image_mask_reco,
                        dl1_phe_image_mask_clusters,
                        good_for_reco,
                        hillas_dict,
                        hillas_dict_reco,
                        n_tels,
                        leakage_dict,
                    )
                else:
                    continue

            if self.event_cutflow.cut("direction nan", reco_result):
                if debug:
                    print(
                        bcolors.WARNING + "WARNING: undefined direction!" + bcolors.ENDC
                    )
                if return_stub:
                    if debug:
                        print(
                            bcolors.FAIL
                            + "Shower could NOT be correctly reconstructed! "
                            + "Recording event..."
                            + "WARNING: This is event shall NOT be used further along the pipeline."
                            + bcolors.ENDC
                        )

                    yield stub(  # record event with dummy info
                        event,
                        mc_phe_image,
                        dl1_phe_image,
                        dl1_phe_image_mask_reco,
                        dl1_phe_image_mask_clusters,
                        good_for_reco,
                        hillas_dict,
                        hillas_dict_reco,
                        n_tels,
                        leakage_dict,
                    )
                else:
                    continue

            if debug:
                print(
                    bcolors.BOLDGREEN
                    + "Shower correctly reconstructed! "
                    + "Recording event..."
                    + bcolors.ENDC
                )

            yield PreparedEvent(
                event=event,
                dl1_phe_image=dl1_phe_image,
                dl1_phe_image_mask_reco=dl1_phe_image_mask_reco,
                dl1_phe_image_mask_clusters=dl1_phe_image_mask_clusters,
                mc_phe_image=mc_phe_image,
                n_pixel_dict=n_pixel_dict,
                hillas_dict=hillas_dict,
                hillas_dict_reco=hillas_dict_reco,
                leakage_dict=leakage_dict,
                n_tels=n_tels,
                max_signals=max_signals,
                n_cluster_dict=n_cluster_dict,
                reco_result=reco_result,
                impact_dict=impact_dict_reco,
                good_event=True,
                good_for_reco=good_for_reco,
            )
