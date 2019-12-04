import math
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
        "mc_phe_image",
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

# ==============================================================================
#               THIS PART WILL DISAPPEAR WITH NEXT CTAPIPE RELEASE
# ==============================================================================

from scipy.sparse.csgraph import connected_components

def camera_radius(cam_id=None):
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
    
    ThS - Nov. 2019
    """

    if cam_id == "ASTRICam":
        average_camera_radius_degree = 4.67
        foclen_meters = 2.15
    elif cam_id == "CHEC":
        average_camera_radius_degree = 3.93
        foclen_meters = 2.283
    elif cam_id == "DigiCam":
        average_camera_radius_degree = 4.56
        foclen_meters = 5.59
    elif cam_id == "FlashCam":
        average_camera_radius_degree = 3.95
        foclen_meters = 16.0
    elif cam_id == "NectarCam":
        average_camera_radius_degree = 4.05
        foclen_meters = 16.0
    elif cam_id == "LSTCam":
        average_camera_radius_degree = 2.31
        foclen_meters = 28.0
    elif cam_id == "all":
        print("Available camera radii")
        print("   * LST           : ",camera_radius("LSTCam"))
        print("   * MST - Nectar  : ",camera_radius("NectarCam"))
        print("   * MST - Flash   : ",camera_radius("FlashCam"))
        print("   * SST - ASTRI   : ",camera_radius("ASTRICam"))
        print("   * SST - CHEC    : ",camera_radius("CHEC"))
        print("   * SST - DigiCam : ",camera_radius("DigiCam"))       
        average_camera_radius_degree = 0
        foclen_meters = 0
    else: 
        raise ValueError('Unknown camid', cam_id)

    average_camera_radius_meters = math.tan(math.radians(average_camera_radius_degree)) * foclen_meters
    return average_camera_radius_meters

# This function is already in 0.7.0, but in introducing "largest_island"
# a small change has been done that requires it to be explicitly put here.
def number_of_islands(geom, mask):
    """
    Search a given pixel mask for connected clusters.
    This can be used to seperate between gamma and hadronic showers.

    Parameters
    ----------
    geom: `~ctapipe.instrument.CameraGeometry`
        Camera geometry information
    mask: ndarray
        input mask (array of booleans)

    Returns
    -------
    num_islands: int
        Total number of clusters
    island_labels: ndarray
        Contains cluster membership of each pixel.
        Dimesion equals input mask.
        Entries range from 0 (not in the pixel mask) to num_islands.
    """
    # compress sparse neighbor matrix
    neighbor_matrix_compressed = geom.neighbor_matrix_sparse[mask][:, mask]
    # pixels in no cluster have label == 0
    island_labels = np.zeros(geom.n_pixels, dtype="int32")

    num_islands, island_labels_compressed = connected_components(
        neighbor_matrix_compressed, directed=False
    )

    # count clusters from 1 onwards
    island_labels[mask] = island_labels_compressed + 1

    return num_islands, island_labels


def largest_island(islands_labels):
    """Find the biggest island and filter it from the image.

    This function takes a list of islands in an image and isolates the largest one
    for later parametrization.

    Parameters
    ----------

    islands_labels : array
        Flattened array containing a list of labelled islands from a cleaned image.
        Second returned value of the function 'number_of_islands'.

    Returns
    -------

    islands_labels : array
        A boolean mask created from the input labels and filtered for the largest island.

    """
    return islands_labels == np.argmax(np.bincount(islands_labels[islands_labels > 0]))


# ==============================================================================


def stub(event):
    return PreparedEvent(
        event=event,
        dl1_phe_image=None,  # container for the calibrated image in phe
        mc_phe_image=None,  # container for the simulated image in phe
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
    (e.g. results of the reconstruction).    #--------------------------------------------------------------------------


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

    def __init__(self, config, mode, event_cutflow=None, image_cutflow=None, debug=False):
        """Initiliaze an EventPreparer object."""
        # Cleaning for reconstruction
        self.cleaner_reco = ImageCleaner(  # for reconstruction
            config=config["ImageCleaning"]["biggest"], mode=mode)

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
        
        if (debug): camera_radius("all") # Display all registered camera radii
        
        self.camera_radius = {
            "LSTCam": camera_radius("LSTCam"), # was 1.126,
            "NectarCam": camera_radius("NectarCam"), # was 1.126,
            "FlashCam": camera_radius("FlashCam"),
            "ASTRICam": camera_radius("ASTRICam"),
            "CHEC": camera_radius("CHEC"),
            "DigiCam": camera_radius("DigiCam")

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

        # configuration for the camera calibrator
        # modifies the integration window to be more like in MARS
        # JLK, only for LST!!!!
        cfg = Config()
        cfg["ChargeExtractorFactory"]["window_width"] = 5
        cfg["ChargeExtractorFactory"]["window_shift"] = 2
        extractor = LocalPeakWindowSum(config=cfg)

        self.calib = CameraCalibrator(config=cfg, image_extractor=extractor)

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
    
    def prepare_event(self, source, return_stub=False, save_images=False, debug=False):
        """ 
        Loop over evenst
        (doc to be completed)
        """
        ievt = 0
        for event in source:
            
            # Display event counts
            ievt+=1
            if (debug):
                if (ievt< 10) or (ievt%10==0) : print(ievt)
        
            self.event_cutflow.count("noCuts")

            if self.event_cutflow.cut("min2Tels trig", len(event.dl0.tels_with_data)):
                if return_stub:
                    yield stub(event)
                else:
                    continue

            self.calib(event)

            # telescope loop
            tot_signal = 0
            dl1_phe_image = None
            mc_phe_image = None
            max_signals = {}
            n_pixel_dict = {}
            hillas_dict_reco = {}  # for direction reconstruction
            hillas_dict = {}  # for discrimination
            n_tels = {
                "tot": len(event.dl0.tels_with_data),
                "LST_LST_LSTCam": 0,
                "MST_MST_NectarCam": 0,
                "SST": 0,  # add later correct names when testing on Paranal
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

                camera = event.inst.subarray.tel[tel_id].camera

                # count the current telescope according to its size
                tel_type = str(event.inst.subarray.tel[tel_id])

                # use ctapipe's functionality to get the calibrated image
                pmt_signal = event.dl1.tel[tel_id].image

                # Save the calibrated image after the gain has been chosen
                # automatically by ctapipe, together with the simulated one
                if save_images is True:
                    dl1_phe_image = pmt_signal
                    mc_phe_image = event.mc.tel[tel_id].photo_electron_image

                if self.cleaner_reco.mode == "tail":  # tail uses only ctapipe

                    # Cleaning used for direction reconstruction
                    image_biggest, mask_reco = self.cleaner_reco.clean_image(
                        pmt_signal, camera
                    )
                    # find all islands using this cleaning
                    num_islands, labels = number_of_islands(camera, mask_reco)

                    if num_islands == 1:  # if only ONE islands is left ...
                        # ...use directly the old mask and reduce dimensions
                        # to make Hillas parametrization faster
                        camera_biggest = camera[mask_reco]
                        image_biggest = image_biggest[mask_reco]
                    elif num_islands > 1:  # if more islands survived..
                        # ...find the biggest one
                        mask_biggest = largest_island(labels)
                        # and also reduce dimensions
                        camera_biggest = camera[mask_biggest]
                        image_biggest = image_biggest[mask_biggest]
                    else:  # if no islands survived use old camera and image
                        camera_biggest = camera

                    # Cleaning used for score/energy estimation
                    image_extended, mask_extended = self.cleaner_extended.clean_image(
                        pmt_signal, camera
                    )

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
                    
                    if num_islands > 0:
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
                                image_biggest, camera.cam_id
                            )
                            image_biggest2d = filter_pixels_clusters(image_biggest2d)
                            image_biggest = geometry_converter.image_2d_to_1d(
                                image_biggest2d, camera.cam_id
                            )

                            # Image for score/energy estimation (with clusters)
                            image_extended, mask_extended = self.cleaner_extended.clean_image(
                                pmt_signal, camera
                            )

                            # This last part was outside the pywi-cta block
                            # before, but is indeed part of it because it uses
                            # pywi-cta functions in the "extended" case

                            # For cluster counts
                            image_2d = geometry_converter.image_1d_to_2d(
                                image_extended, camera.cam_id
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
                            "close to the edge", moments_reco, camera.cam_id
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
                            )  # cycle only on tels which still have an image
                            for tel_id in point_altitude_dict.keys()
                        },
                    )

                    # Impact parameter for energy estimation (/ tel)
                    subarray = event.inst.subarray
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
                mc_phe_image=mc_phe_image,
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
