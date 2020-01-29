"""Calibrate, clean the image, and reconstruct the direction of an event."""
from abc import abstractmethod
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
from ctapipe.image.extractor import LocalPeakWindowSum
from ctapipe.image import hillas
from ctapipe.image.cleaning import tailcuts_clean
from ctapipe.utils.CutFlow import CutFlow
from ctapipe.coordinates import GroundFrame
from ctapipe.image.extractor import (
    ImageExtractor,
    sum_samples_around_peak,
    extract_pulse_time_around_peak,
)

from ctapipe.image.timing_parameters import timing_parameters
from ctapipe.image.hillas import hillas_parameters, camera_to_shower_coordinates
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

    ThS, MP - Nov. 2019
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


def slide_window(waveform, window_width):
    """Smooth a pixel's waveform with a kernel of certain size via convolution.

    Parameters
    ----------
    waveforms : array_like
        DL0-level waveforms of one event.
        Shape: (n_samples)
    window_width : int
        Size of the smoothing kernel.

    Returns
    -------
    sum : array_like
        Array containing the sums for each of the kernel positions.
        Shape: (n_samples - window_width - 1)

    """
    sums = np.convolve(waveform, np.ones(window_width, dtype=int), "valid")
    return sums


class MyImageExtractor(ImageExtractor):
    """
    Overwrite ImageExtractor to use subarray for telescope information.
    """

    def __init__(self, config=None, parent=None, subarray=None, **kwargs):
        """
        Base component to handle the extraction of charge and pulse time
        from an image cube (waveforms).

        Parameters
        ----------
        config : traitlets.loader.Config
            Configuration specified by config file or cmdline arguments.
            Used to set traitlet values.
            Set to None if no configuration to pass.
        tool : ctapipe.core.Tool or None
            Tool executable that is calling this component.
            Passes the correct logger to the component.
            Set to None if no Tool to pass.
        subarray: ctapipe.instrument.SubarrayDescription
            Description of the subarray
        kwargs
        """
        super().__init__(config=config, parent=parent, **kwargs)
        self.subarray = subarray
        for trait in list(self.class_traits()):
            try:
                getattr(self, trait).attach_subarray(subarray)
            except (AttributeError, TypeError):
                pass

    @abstractmethod
    def __call__(self, waveforms, telid=None):  # added telid to the call
        """
        Call the relevant functions to fully extract the charge and time
        for the particular extractor.

        Parameters
        ----------
        waveforms : ndarray
            Waveforms stored in a numpy array of shape
            (n_pix, n_samples).

        Returns
        -------
        charge : ndarray
            Extracted charge.
            Shape: (n_pix)
        pulse_time : ndarray
            Floating point pulse time in each pixel.
            Shape: (n_pix)
        """


class MyCameraCalibrator(CameraCalibrator):
    """Create a child class of CameraCalibrator."""

    def _calibrate_dl0(self, event, telid):
        """Override class method to perform gain selection at R0 instead of R1.

        Then select R1 waveforms using R0-selected gain channels.
        """
        waveforms_r0 = event.r0.tel[telid].waveform
        _, selected_gain_channel = self.gain_selector(waveforms_r0)

        waveforms_r1 = event.r1.tel[telid].waveform
        if self._check_r1_empty(waveforms_r1):
            return

        # Use R0-selected gain channels to select R1 waveforms
        _, n_pixels, _ = waveforms_r1.shape
        waveforms_gs = waveforms_r1[selected_gain_channel, np.arange(n_pixels)]
        if selected_gain_channel is not None:
            event.r1.tel[telid].selected_gain_channel = selected_gain_channel
        else:
            if event.r1.tel[telid].selected_gain_channel is None:
                raise ValueError(
                    "EventSource is loading pre-gainselected waveforms "
                    "without filling the selected_gain_channel container"
                )

        reduced_waveforms = self.data_volume_reducer(waveforms_gs)
        event.dl0.tel[telid].waveform = reduced_waveforms

    def _calibrate_dl1(self, event, telid):
        waveforms = event.dl0.tel[telid].waveform
        if self._check_dl0_empty(waveforms):
            return
        n_pixels, n_samples = waveforms.shape
        if n_samples == 1:
            # To handle ASTRI and dst
            # TODO: Improved handling of ASTRI and dst
            #   - dst with custom EventSource?
            #   - Read into dl1 container directly?
            #   - Don't do anything if dl1 container already filled
            #   - Update on SST review decision
            corrected_charge = waveforms[..., 0]
            pulse_time = np.zeros(n_pixels)
        else:
            # TODO: pass camera to ImageExtractor.__init__
            if self.image_extractor.requires_neighbors():
                camera = event.inst.subarray.tel[telid].camera
                self.image_extractor.neighbors = camera.neighbor_matrix_where
            charge, pulse_time = self.image_extractor(waveforms, telid)

            # Apply integration correction
            # TODO: Remove integration correction
            correction = self._get_correction(event, telid)
            corrected_charge = charge * correction

        event.dl1.tel[telid].image = corrected_charge
        event.dl1.tel[telid].pulse_time = pulse_time


class TwoPassWindowSum(MyImageExtractor):  # later change to ImageExtractor
    """Extract image by integrating the waveform a second time using a
     time-gradient linear fit.

    This is in particular the CTA-MARS version.
    Procedure:
    1) find waveform peak from maximum sum of three consecutive samples;
       add 1 sample on each side and integrate charge in 5-sample window;
       time is obtained as a charge-weighted average of the sample numbers for
       the 5 integrated samples;
       No information from neighboouring pixels is used.
    2) preliminary image cleaning via simple tailcut with minimum number
       of core neighbours set at 1,
    3) only the biggest cluster of pixels ("main island") is kept.
    4) Parametrize following Hillas approach only if the resulting image has 3
       or more pixels (already in protopipe but after "real" image cleaning).
    5) Do a linear fit of pulse time vs. distance along major image axis (in
       CTA-MARS the ROOT "robust" fit option is used).
    6) For all pixels except the core ones in the preliminary image, integrate
       the waveform once more, in a fixed window of 5 samples set at the time
       "predicted" by the linear time fit.
       If the predicted time for a pixel leads to a window outside the readout
       window, then integrate also the last (or first) 5 samples.
    7) The result is an image with core pixels calibrated with a 1st pass and
       non-core pixels re-calibrated with a 2nd pass
    8) Clean the resulting calibrated image again with the a double-boundary
       tailcut cleaning. (NOT PART OF IMAGE EXTRACTOR)

    Dr. Michele Peresano, 2019

    """

    def __call__(self, waveforms, telid=None):
        """
        Call this ImageExtractor.

        Parameters
        ----------
        waveforms : array of size (N_pixels, N_samples)
            DL0-level waveforms of one event.

        Returns
        -------
        charge : array_like
            Integrated charge per pixel.
            Shape: (n_pix, n_channels)
        pulse_time : array_like
            Samples in which the waveform peak has been recognized.
            Shape: (n_pix)

        """

        # STEP 1

        # Starting from DL0, the channel is already selected (if more than one)
        # event.dl0.tel[tel_id].waveform object has shape (N_pixels, N_samples)
        # For each pixel, we slide a 3-samples window through the whole
        # waveform, summing each time the ADC counts contained within it.

        window1_width = 3  # could become a configurable
        sums = np.apply_along_axis(slide_window, 1, waveforms, window1_width)
        # 'sums' has still the same shape of 'waveforms'

        # For each pixel, in each of the (N_samples - 2) positions, we check if
        # the window encountered the maximum number of ADC counts and its index
        # will correspond to the first sample (start) of the 3-samples window
        startWindows = np.apply_along_axis(np.argmax, 1, sums)  # (N_pixels)

        # Build 'width' and 'shift' arrays that adapt on the position of the
        # window along each waveform
        window_1_at_start = startWindows == 1
        window_1_at_end = startWindows == (waveforms.shape[1] - window1_width)

        # Since we have to add 1 sample on each side, window_shift will always
        # be (-)1, while window_width will always be first_window_width + 2
        window_widths_2 = np.full_like(startWindows, window1_width + 2)
        window_shifts = np.full_like(startWindows, 1)

        # BUT, if the resulting 5-samples window falls outside of the readout
        # window then we always take the first (or last) 5 samples
        window_widths_2[window_1_at_start] = window1_width + 2
        window_widths_2[window_1_at_end] = window1_width
        window_shifts[window_1_at_start] = 0
        window_shifts[window_1_at_end] = -2

        # 'peak_index' has no sense in this case, it's simply the start of
        # the 3-samples window
        preliminary_charges = sum_samples_around_peak(
            waveforms, startWindows, window_widths_2, window_shifts
        )
        preliminary_pulse_times = extract_pulse_time_around_peak(
            waveforms, startWindows, window_widths_2, window_shifts
        )

        first_dl1_image = preliminary_charges

        # STEP 2

        # set thresholds for core-pixels depending on telescope type
        # boundary thresholds will be half of core thresholds
        # WARNING: in dev version these values should be
        # read from a configuration file
        subarray = self.subarray
        if subarray.tel[telid].type == "LST":
            core_th = 6  # core threshold (should be ok!)
        if subarray.tel[telid].type == "MST":
            core_th = 8  # core threshold (should be ok!)
        if subarray.tel[telid].type == "SST":
            core_th = 4  # core threshold (dummy value)

        # Preliminary image cleaning with simple two-level tail-cut
        camera = self.subarray.tel[telid].camera
        mask_1 = tailcuts_clean(
            camera,
            first_dl1_image,
            picture_thresh=core_th,
            boundary_thresh=core_th / 2,
            keep_isolated_pixels=False,
            min_number_picture_neighbors=1,
        )
        image_1 = first_dl1_image
        image_1[~mask_1] = 0

        # STEP 3

        # # find all islands using this cleaning
        num_islands, labels = number_of_islands(camera, mask_1)
        if num_islands == 0:
            image_2 = image_1  # no islands = image unchanged
        else:
            # ...find the biggest one
            mask_biggest = largest_island(labels)
            image_2 = image_1
            image_2[~mask_biggest] = 0

        # Indexes of pixels that will need the 2nd pass
        nonCore_pixels_ids = np.where(image_2 < core_th)[0]
        nonCore_pixels_mask = image_2 < core_th
        # print(f"There are {len(non_core_pix_ids)} non-core pixels.")

        # STEP 4

        # check if the resulting image has 3 or more pixels
        if np.count_nonzero(image_2) < 3:
            charge = image_2  # main cluster image
            pulse_time = preliminary_pulse_times  # 1st pass pulse times
        else:
            hillas = hillas_parameters(camera, image_2)

            # STEP 5

            # linear fit of pulse time vs. distance along major image axis
            timing = timing_parameters(camera, image_2, preliminary_pulse_times, hillas)

            long, trans = camera_to_shower_coordinates(
                camera.pix_x, camera.pix_y, hillas.x, hillas.y, hillas.psi
            )

            # for LSTCam and NectarCam sample = ns, but not for other cameras
            # Here treated as sample, but think about a general method!
            predicted_pulse_times = (
                timing.slope * long[nonCore_pixels_ids] + timing.intercept
            )
            predicted_peaks = np.zeros(len(predicted_pulse_times))
            np.rint(predicted_pulse_times.value, predicted_peaks)
            predicted_peaks = predicted_peaks.astype(np.int64)

            # set sample to 1 if predicted time is < 1
            predicted_peaks[predicted_peaks < 1] = 1
            # set sample to max if predicted time is > max
            predicted_peaks[predicted_peaks > waveforms.shape[1]] = waveforms.shape[1]

            # STEP 6

            # select only the waveforms correspondent to non-core pixels
            nonCore_waveforms = waveforms[nonCore_pixels_ids]

            # Build 'width' and 'shift' arrays that adapt on the position of
            # the window along each waveform

            # Since we have to add 1 sample on each side, window_shift should
            # be (-)1, while window_width will always be first_window_width + 2
            window_widths = np.full_like(predicted_peaks, 5, dtype=np.int64)
            window_shifts = np.full_like(predicted_peaks, 2, dtype=np.int64)

            # BUT, if the resulting 5-samples window falls outside of the
            # readout window then we always take the first (or last) 5 samples
            window_widths[predicted_peaks == 1] = 5
            window_shifts[predicted_peaks == 1] = 0
            window_widths[predicted_peaks == waveforms.shape[1]] = 5
            window_shifts[predicted_peaks == waveforms.shape[1]] = 5

            # re-calibrate non-core pixels using the fixed 5-samples window
            charge_noCore = sum_samples_around_peak(
                nonCore_waveforms, predicted_peaks, window_widths, window_shifts
            )
            pulse_times_noCore = extract_pulse_time_around_peak(
                nonCore_waveforms, predicted_peaks, window_widths, window_shifts
            )

            # STEP 7

            # combine core and non-core information in the final output
            charge = image_2  # core + non-core pixels
            charge[nonCore_pixels_mask] = charge_noCore  # non-core pixels
            pulse_time = preliminary_pulse_times  # core + non-core pixels
            pulse_time[nonCore_pixels_mask] = pulse_times_noCore  # non-core pixels

        return charge, pulse_time
        # to make one of the plots from Abelardo I need both passes information
        # this would require to modify too much current ctapipe...
        # return charge, pulse_time, first_dl1_image, preliminary_pulse_times


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

        # configuration for the camera calibrator
        # modifies the integration window to be more like in MARS
        # JLK, only for LST!!!!
        cfg = Config()
        # cfg["ChargeExtractorFactory"]["window_width"] = 5
        # cfg["ChargeExtractorFactory"]["window_shift"] = 2
        cfg["ThresholdGainSelector"]["threshold"] = 4000.0  # 40 @ R1!
        extractor = TwoPassWindowSum(config=cfg, subarray=subarray)
        gain_selector = GainSelector.from_name("ThresholdGainSelector", config=cfg)

        self.calib = MyCameraCalibrator(
            config=cfg, gain_selector=gain_selector, image_extractor=extractor
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

    def prepare_event(self, source, return_stub=False, save_images=False, debug=False):
        """
        Loop over events
        (doc to be completed)
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
