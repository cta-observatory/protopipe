"""Calibrate, clean the image, and reconstruct the direction of an event."""
import numpy as np

from astropy import units as u
from astropy.coordinates import SkyCoord
import warnings
from collections import namedtuple, OrderedDict

# CTAPIPE utilities
from ctapipe.instrument import CameraGeometry
from ctapipe.containers import ReconstructedShowerContainer
from ctapipe.image import (
    leakage_parameters,
    number_of_islands,
    largest_island,
    concentration_parameters,
)
from ctapipe.utils import CutFlow
from ctapipe.coordinates import (
    GroundFrame,
    TelescopeFrame,
    CameraFrame,
    TiltedGroundFrame,
    MissingFrameAttributeWarning,
)

from ctapipe.image.hillas import hillas_parameters, HillasParameterizationError
from ctapipe.reco.reco_algorithms import (
    TooFewTelescopesException,
    InvalidWidthException,
)

# Pipeline utilities
from .image_cleaning import ImageCleaner
from .utils import (
    bcolors,
    effective_focal_lengths,
    camera_radius,
    get_cameras_radii,
)
from .temp import (
    MyCameraCalibrator,
    TwoPassWindowSum,
    HillasParametersTelescopeFrameContainer,
    HillasReconstructor,
)

# PiWy utilities
try:
    from pywicta.io import geometry_converter
    from pywi.processing.filtering import pixel_clusters
    from pywi.processing.filtering.pixel_clusters import filter_pixels_clusters

    wavelets_error = False
except ImportError:
    wavelets_error = True

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
        "concentration_dict",
        "n_tels",
        "n_tels_reco",
        "max_signals",
        "n_cluster_dict",
        "reco_result",
        "impact_dict",
        "good_event",
        "good_for_reco",
        "image_extraction_status",
    ],
)


def stub(
    event,
    true_image,
    image,
    n_pixel_dict,
    cleaning_mask_reco,
    cleaning_mask_clusters,
    good_for_reco,
    hillas_dict,
    hillas_dict_reco,
    n_tels,
    n_tels_reco,
    leakage_dict,
    concentration_dict,
    passed,
):
    """Default container for images that did not survive cleaning."""
    return PreparedEvent(
        event=event,
        dl1_phe_image=image,  # container for the calibrated image in phe
        dl1_phe_image_mask_reco=cleaning_mask_reco,  # container for the reco cleaning mask
        dl1_phe_image_mask_clusters=cleaning_mask_clusters,
        mc_phe_image=true_image,  # container for the simulated image in phe
        n_pixel_dict=n_pixel_dict,
        good_for_reco=good_for_reco,
        hillas_dict=hillas_dict,
        hillas_dict_reco=hillas_dict_reco,
        leakage_dict=leakage_dict,
        concentration_dict=concentration_dict,
        n_tels=n_tels,
        n_tels_reco=n_tels_reco,
        max_signals=dict.fromkeys(hillas_dict_reco.keys(), np.nan),  # no charge
        n_cluster_dict=dict.fromkeys(hillas_dict_reco.keys(), 0),  # no clusters
        reco_result=ReconstructedShowerContainer(),  # defaults to nans
        impact_dict=dict.fromkeys(
            hillas_dict_reco, np.nan * u.m
        ),  # undefined impact parameter
        good_event=False,
        image_extraction_status=passed,
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

        if mode == "wave" and wavelets_error:
            raise ImportError(
                "WARNING: 'pywicta' and/or 'pywi' packages could"
                "not be imported - wavelet cleaning will not work"
            )

        # Readout window integration correction
        try:
            self.apply_integration_correction = config["Calibration"][
                "apply_integration_correction"
            ]
        except KeyError:
            # defaults to enabled
            self.apply_integration_correction = True

        # Shifts in peak time and waveforms applied by the calibrator
        try:
            self.apply_peak_time_shift = config["Calibration"]["apply_peak_time_shift"]
            self.apply_waveform_time_shift = config["Calibration"][
                "apply_waveform_time_shift"
            ]
        except KeyError:
            # defaults to disabled
            self.apply_peak_time_shift = False
            self.apply_waveform_time_shift = False

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
        except KeyError:
            pass  # force_mode = mode

        self.cleaner_extended = ImageCleaner(  # for energy/score estimation
            config=config["ImageCleaning"]["extended"],
            cameras=cams_and_foclens.keys(),
            mode=force_mode,
        )

        # Image book keeping
        self.image_cutflow = image_cutflow or CutFlow("ImageCutFlow")

        # Add quality cuts on images
        try:
            self.image_selection_source = config["ImageSelection"]["source"]
            charge_bounds = config["ImageSelection"]["charge"]
            npix_bounds = config["ImageSelection"]["pixel"]
            ellipticity_bounds = config["ImageSelection"]["ellipticity"]
            nominal_distance_bounds = config["ImageSelection"]["nominal_distance"]
        except KeyError:
            # defaults for a CTAMARS-like analysis
            self.image_selection_source = "extended"
            charge_bounds = [50.0, 1.0e10]
            npix_bounds = [3, 1e10]
            ellipticity_bounds = [0.1, 0.6]
            nominal_distance_bounds = [0.0, 0.8]

        if debug:
            camera_radius(
                cams_and_foclens, "all"
            )  # Display all registered camera radii

        # Get available cameras radii in degrees using ctapipe...
        cameras_radii = get_cameras_radii(
            subarray, frame=TelescopeFrame(), ctamars=False
        )
        self.camera_radius = {
            cam_id: cameras_radii[cam_id].value for cam_id in cams_and_foclens.keys()
        }

        self.image_cutflow.set_cuts(
            OrderedDict(
                [
                    ("noCuts", None),
                    ("bad image extraction", lambda p: p == 0),
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
                    ),
                ]
            )
        )

        extractor = TwoPassWindowSum(
            subarray=subarray,
            apply_integration_correction=self.apply_integration_correction,
        )
        # Get the name of the image extractor in order to adapt some options
        # specific to TwoPassWindowSum later on
        self.extractorName = list(extractor.get_current_config().items())[0][0]

        self.calib = MyCameraCalibrator(
            image_extractor=extractor,
            subarray=subarray,
            apply_peak_time_shift=self.apply_peak_time_shift,
            apply_waveform_time_shift=self.apply_waveform_time_shift,
        )

        # Reconstruction
        self.shower_reco = HillasReconstructor(subarray)

        # Event book keeping
        self.event_cutflow = event_cutflow or CutFlow("EventCutFlow")

        # Add cuts on events
        self.min_ntel = config["Reconstruction"]["min_tel"]
        self.min_ntel_LST = 2  # LST stereo trigger
        try:
            self.LST_stereo = config["Reconstruction"]["LST_stereo"]
        except KeyError:
            print(
                bcolors.WARNING
                + "WARNING: the 'LST_stereo' setting has not been specified in the analysis configuration file."
                + "\t It has been set to 'True'!"
                + bcolors.ENDC
            )
            self.LST_stereo = True
        self.event_cutflow.set_cuts(
            OrderedDict(
                [
                    ("noCuts", None),
                    ("min2Tels trig", lambda x: x < self.min_ntel),
                    (
                        "no-LST-stereo + <2 other types",
                        lambda x, y: (x < self.min_ntel_LST) and (y < 2),
                    ),
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
            Dictionary of MyCameraGeometry objects for each camera in the file
        return_stub : bool
            If True, yield also images from events that won't be reconstructed.
            This is required for DL1 benchmarking.
        save_images : bool
            If True, save photoelectron images from reconstructed events.
        debug : bool
            If True, print debugging information.

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

            geom_cam_tel[camera.camera_name] = CameraGeometry(
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
                    + f"has triggered telescopes {event.r1.tel.keys()}"
                    + bcolors.ENDC
                )
                ievt += 1
                # if (ievt < 10) or (ievt % 10 == 0):
                #     print(ievt)

            self.event_cutflow.count("noCuts")

            # LST stereo condition
            # whenever there is only 1 LST in an event, we remove that telescope
            # if the remaining telescopes are less than min_tel we remove the event

            lst_tel_ids = set(source.subarray.get_tel_ids_for_type("LST_LST_LSTCam"))
            triggered_LSTs = set(event.r0.tel.keys()).intersection(lst_tel_ids)
            n_triggered_LSTs = len(triggered_LSTs)
            n_triggered_non_LSTs = len(event.r0.tel.keys()) - n_triggered_LSTs

            bad_LST_stereo = False
            if self.LST_stereo and self.event_cutflow.cut(
                "no-LST-stereo + <2 other types", n_triggered_LSTs, n_triggered_non_LSTs
            ):
                bad_LST_stereo = True
                # we proceed to analyze the event up to
                # DL1a/b for the associated benchmarks
                if debug:
                    print(
                        bcolors.WARNING
                        + "WARNING: LST_stereo is set to 'True'\n"
                        + f"This event has < {self.min_ntel_LST} triggered LSTs\n"
                        + "and < 2 triggered telescopes from other telescope types.\n"
                        + "The event will be processed up to DL1b."
                        + bcolors.ENDC
                    )

            # this checks for < 2 triggered telescopes of ANY type
            if self.event_cutflow.cut("min2Tels trig", len(event.r1.tel.keys())):
                if return_stub and debug:
                    print(
                        bcolors.WARNING
                        + f"WARNING : < {self.min_ntel} triggered telescopes!"
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
            passed = self.calib(event)  # Calibrate the event

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
            concentration_dict = {}
            n_tels = {
                "Triggered": len(event.r1.tel.keys()),
                "LST_LST_LSTCam": 0,
                "MST_MST_NectarCam": 0,
                "MST_MST_FlashCam": 0,
                "MST_SCT_SCTCam": 0,
                "SST_1M_DigiCam": 0,
                "SST_ASTRI_ASTRICam": 0,
                "SST_GCT_CHEC": 0,
                "SST_ASTRI_CHEC": 0,
            }
            n_tels_reco = {
                "LST_LST_LSTCam": 0,
                "MST_MST_NectarCam": 0,
                "MST_MST_FlashCam": 0,
                "MST_SCT_SCTCam": 0,
                "SST_1M_DigiCam": 0,
                "SST_ASTRI_ASTRICam": 0,
                "SST_GCT_CHEC": 0,
                "SST_ASTRI_CHEC": 0,
            }
            n_cluster_dict = {}
            impact_dict_reco = {}  # impact distance measured in tilt system

            point_azimuth_dict = {}
            point_altitude_dict = {}

            good_for_reco = {}  # 1 = success, 0 = fail

            # filter warnings for missing obs time. this is needed because MC data has no obs time
            warnings.filterwarnings(
                action="ignore", category=MissingFrameAttributeWarning
            )

            # Array pointing in AltAz frame
            az = event.pointing.array_azimuth
            alt = event.pointing.array_altitude
            array_pointing = SkyCoord(az=az, alt=alt, frame="altaz")

            # Actual telescope pointings
            telescope_pointings = {
                tel_id: SkyCoord(
                    alt=event.pointing.tel[tel_id].altitude,
                    az=event.pointing.tel[tel_id].azimuth,
                    frame="altaz",
                )
                for tel_id in event.pointing.tel.keys()
            }

            ground_frame = GroundFrame()

            tilted_frame = TiltedGroundFrame(pointing_direction=array_pointing)

            for tel_id in event.r1.tel.keys():

                point_azimuth_dict[tel_id] = event.pointing.tel[tel_id].azimuth
                point_altitude_dict[tel_id] = event.pointing.tel[tel_id].altitude

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

                # We now ASSUME that the event will be good at all levels
                extracted_image_is_good = True
                cleaned_image_is_good = True
                good_for_reco[tel_id] = 1
                # later we change to 0 at the first condition NOT satisfied

                # Apply some selection over image extraction
                if self.image_cutflow.cut("bad image extraction", passed[tel_id]):
                    if debug:
                        print(
                            bcolors.WARNING
                            + "WARNING : bad image extraction!"
                            + "WARNING : image processed and recorded, but NOT used for DL2."
                            + bcolors.ENDC
                        )
                    # we set immediately this image as BAD at all levels
                    # for now (and for simplicity) we will process it anyway
                    # any other failing condition will just confirm this
                    extracted_image_is_good = False
                    cleaned_image_is_good = False
                    good_for_reco[tel_id] = 0

                # use ctapipe's functionality to get the calibrated image
                # and scale the reconstructed values if required
                pmt_signal = event.dl1.tel[tel_id].image

                # If required...
                if save_images is True:
                    # Save the simulated and reconstructed image of the event
                    dl1_phe_image[tel_id] = pmt_signal
                    mc_phe_image[tel_id] = event.simulation.tel[tel_id].true_image

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
                        leakage_biggest = leakage_parameters(
                            camera, image_biggest, mask_reco
                        )
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
                        leakage_extended = leakage_parameters(
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

                    except FileNotFoundError as e:
                        print(e)
                        continue

                # =============================================================
                #                PRELIMINARY IMAGE SELECTION
                # =============================================================

                if self.image_selection_source == "extended":
                    cleaned_image_to_use = image_extended
                elif self.image_selection_source == "biggest":
                    cleaned_image_to_use = image_biggest
                else:
                    raise ValueError(
                        "Only supported cleanings are 'biggest' or 'extended'."
                    )

                # Apply some selection
                if self.image_cutflow.cut("min pixel", cleaned_image_to_use):
                    if debug:
                        print(
                            bcolors.WARNING
                            + "WARNING : not enough pixels!"
                            + bcolors.ENDC
                        )
                    good_for_reco[tel_id] = 0  # we record it as BAD
                    cleaned_image_is_good = False

                if self.image_cutflow.cut("min charge", np.sum(cleaned_image_to_use)):
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

                        # Add concentration parameters
                        concentrations = {}
                        concentrations_extended = concentration_parameters(
                            camera_extended_tel, image_extended, moments
                        )
                        concentrations["concentration_cog"] = concentrations_extended[
                            "cog"
                        ]
                        concentrations["concentration_core"] = concentrations_extended[
                            "core"
                        ]
                        concentrations["concentration_pixel"] = concentrations_extended[
                            "pixel"
                        ]

                        # ===================================================
                        #             PARAMETRIZED IMAGE SELECTION
                        # ===================================================
                        if self.image_selection_source == "extended":
                            moments_to_use = moments
                        else:
                            moments_to_use = moments_reco

                        # if width and/or length are zero (e.g. when there is
                        # only only one pixel or when all  pixel are exactly
                        # in one row), the parametrisation
                        # won't be very useful: skip
                        if self.image_cutflow.cut("poor moments", moments_to_use):
                            if debug:
                                print(
                                    bcolors.WARNING
                                    + "WARNING : poor moments!"
                                    + bcolors.ENDC
                                )
                            good_for_reco[tel_id] = 0  # we record it as BAD

                        if self.image_cutflow.cut(
                            "close to the edge", moments_to_use, camera.camera_name
                        ):
                            if debug:
                                print(
                                    bcolors.WARNING
                                    + "WARNING : out of containment radius!\n"
                                    + f"Camera radius = {self.camera_radius[camera.camera_name]}\n"
                                    + f"COG radius = {moments_to_use.r}"
                                    + bcolors.ENDC
                                )

                            good_for_reco[tel_id] = 0

                        if self.image_cutflow.cut("bad ellipticity", moments_to_use):
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
                                + bcolors.ENDC
                            )
                        elif debug and good_for_reco[tel_id] == 0:
                            print(
                                bcolors.WARNING
                                + "Image not survived or "
                                + "not good enough for parametrization."
                                + "\nIt will be NOT used for direction reconstruction, "
                                + "BUT it's information will be recorded."
                                + bcolors.ENDC
                            )

                        hillas_dict[tel_id] = moments
                        hillas_dict_reco[tel_id] = moments_reco
                        n_pixel_dict[tel_id] = np.count_nonzero(image_extended)
                        leakage_dict[tel_id] = leakages
                        concentration_dict[tel_id] = concentrations

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
                        n_pixel_dict[tel_id] = np.count_nonzero(image_extended)
                        leakage_dict[tel_id] = leakages

                        concentrations = {}
                        concentrations["concentration_cog"] = np.nan
                        concentrations["concentration_core"] = np.nan
                        concentrations["concentration_pixel"] = np.nan
                        concentration_dict[tel_id] = concentrations

                if good_for_reco[tel_id] == 1:
                    n_tels_reco[tel_type] += 1

                # END OF THE CYCLE OVER THE TELESCOPES

            # =============================================================
            #                   DIRECTION RECONSTRUCTION
            # =============================================================

            if bad_LST_stereo:
                if debug:
                    print(
                        bcolors.WARNING
                        + "WARNING: This event was triggered with 1 LST image and <2 images from other telescope types."
                        + "\nWARNING : direction reconstruction will not be performed."
                        + bcolors.ENDC
                    )

                # Set all the involved images as NOT good for recosntruction
                # even though they might have been
                # but this is because of the LST stereo trigger....
                for tel_id in event.r0.tel.keys():
                    if good_for_reco[tel_id]:
                        good_for_reco[tel_id] = 0
                        n_tels_reco[str(source.subarray.tel[tel_id])] -= 1
                # and set the number of good and bad images accordingly
                n_tels["GOOD images"] = 0
                n_tels["BAD images"] = n_tels["Triggered"] - n_tels["GOOD images"]

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
                        n_pixel_dict,
                        dl1_phe_image_mask_reco,
                        dl1_phe_image_mask_clusters,
                        good_for_reco,
                        hillas_dict,
                        hillas_dict_reco,
                        n_tels,
                        n_tels_reco,
                        leakage_dict,
                        concentration_dict,
                        passed,
                    )
                    continue
                else:
                    continue

            # Now in case the only triggered telescopes were
            # - < self.min_ntel_LST LST,
            # - >=2 any other telescope type,
            # we remove the single-LST image and continue reconstruction with
            # the images from the other telescope types
            if (
                self.LST_stereo
                and (n_triggered_LSTs < self.min_ntel_LST)
                and (n_triggered_LSTs != 0)
                and (n_triggered_non_LSTs >= 2)
            ):
                if debug:
                    print(
                        bcolors.WARNING
                        + f"WARNING: LST stereo trigger condition is active.\n"
                        + f"This event triggered < {self.min_ntel_LST} LSTs "
                        + f"and {n_triggered_non_LSTs} images from other telescope types.\n"
                        + bcolors.ENDC
                    )
                for tel_id in triggered_LSTs:  # in case we test for min_ntel_LST>2
                    if good_for_reco[tel_id]:
                        # we don't use it for reconstruction
                        good_for_reco[tel_id] = 0
                        n_tels_reco[str(source.subarray.tel[tel_id])] -= 1
                        if debug:
                            print(
                                bcolors.WARNING
                                + f"WARNING: LST image #{tel_id} removed, even though it passed quality cuts."
                                + bcolors.ENDC
                            )
                # TODO: book-keeping of this kind of events doesn't seem easy

            # convert dictionary in numpy array to get a "mask"
            images_status = np.asarray(list(good_for_reco.values()))
            # record how many images will be used for reconstruction
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
                        n_pixel_dict,
                        dl1_phe_image_mask_reco,
                        dl1_phe_image_mask_clusters,
                        good_for_reco,
                        hillas_dict,
                        hillas_dict_reco,
                        n_tels,
                        n_tels_reco,
                        leakage_dict,
                        concentration_dict,
                        passed,
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

                    if self.image_selection_source == "extended":
                        hillas_dict_to_use = hillas_dict
                    else:
                        hillas_dict_to_use = hillas_dict_reco

                    # use only the successfully parametrized images
                    # to reconstruct the direction of this event
                    successfull_hillas = np.where(images_status == 1)[0]
                    all_images = np.asarray(list(good_for_reco.keys()))
                    good_images = set(all_images[successfull_hillas])
                    good_hillas_dict = {
                        k: v for k, v in hillas_dict_to_use.items() if k in good_images
                    }

                    if debug:
                        print(
                            bcolors.PURPLE
                            + f"{len(good_hillas_dict)} images "
                            + f"(from telescopes #{list(good_hillas_dict.keys())}) will be "
                            + "used to recover the shower's direction..."
                            + bcolors.ENDC
                        )

                    reco_result = self.shower_reco._predict(
                        event,
                        good_hillas_dict,
                        source.subarray,
                        array_pointing,
                        telescope_pointings,
                    )

                    # Impact parameter for telescope-wise energy estimation
                    for tel_id in hillas_dict_to_use.keys():

                        pos = source.subarray.positions[tel_id]

                        tel_ground = SkyCoord(
                            pos[0], pos[1], pos[2], frame=ground_frame
                        )

                        core_ground = SkyCoord(
                            reco_result.core_x,
                            reco_result.core_y,
                            0 * u.m,
                            frame=ground_frame,
                        )

                        # Go back to the tilted frame

                        # this should be the same...
                        tel_tilted = tel_ground.transform_to(tilted_frame)

                        # but this not
                        core_tilted = core_ground.transform_to(tilted_frame)

                        impact_dict_reco[tel_id] = np.sqrt(
                            (core_tilted.x - tel_tilted.x) ** 2
                            + (core_tilted.y - tel_tilted.y) ** 2
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
                        n_pixel_dict,
                        dl1_phe_image_mask_reco,
                        dl1_phe_image_mask_clusters,
                        good_for_reco,
                        hillas_dict,
                        hillas_dict_reco,
                        n_tels,
                        n_tels_reco,
                        leakage_dict,
                        concentration_dict,
                        passed,
                    )
                    continue
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
                        n_pixel_dict,
                        dl1_phe_image_mask_reco,
                        dl1_phe_image_mask_clusters,
                        good_for_reco,
                        hillas_dict,
                        hillas_dict_reco,
                        n_tels,
                        n_tels_reco,
                        leakage_dict,
                        concentration_dict,
                        passed,
                    )
                    continue
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
                concentration_dict=concentration_dict,
                n_tels=n_tels,
                n_tels_reco=n_tels_reco,
                max_signals=max_signals,
                n_cluster_dict=n_cluster_dict,
                reco_result=reco_result,
                impact_dict=impact_dict_reco,
                good_event=True,
                good_for_reco=good_for_reco,
                image_extraction_status=passed,
            )
