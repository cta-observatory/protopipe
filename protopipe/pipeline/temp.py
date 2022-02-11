"""Temporary definitions of classes and methods.

Modifications of ctapipe code needed for the development of protopipe.

Tests are performed with protopipe and, if positive, pushed to ctapipe for
possible futher modifications.
As newer releases of ctapipe are imported in protopipe, the content of this
file will change until eventually it will disappear.

"""

import logging
import warnings
from itertools import combinations
from functools import lru_cache
from typing import Tuple

import numpy as np
from numpy import nan
from astropy import units as u
from astropy.coordinates import SkyCoord, AltAz, cartesian_to_spherical
from traitlets import Bool
from scipy.ndimage.filters import convolve1d

from ctapipe.io import SimTelEventSource
from ctapipe.containers import (
    ArrayEventContainer,
    SimulatedCameraContainer,
    SimulatedEventContainer,
    EventType,
)
from ctapipe.core import Container, Field
from ctapipe.core.traits import Float, FloatTelescopeParameter, BoolTelescopeParameter
from ctapipe.calib import CameraCalibrator
from ctapipe.calib.camera.calibrator import shift_waveforms
from ctapipe.image import (
    extract_around_peak,
    integration_correction,
    ImageExtractor,
    tailcuts_clean,
    number_of_islands,
    brightest_island,
    hillas_parameters,
    HillasParameterizationError,
    timing_parameters,
    camera_to_shower_coordinates,
)
from ctapipe.reco.hillas_reconstructor import (
    HillasPlane,
    normalise,
    angle,
    line_line_intersection_3d,
)
from ctapipe.reco.reco_algorithms import (
    Reconstructor,
    InvalidWidthException,
    TooFewTelescopesException,
)
from ctapipe.containers import (
    HillasParametersContainer,
    TimingParametersContainer,
    LeakageContainer,
    MorphologyContainer,
    ConcentrationContainer,
    IntensityStatisticsContainer,
    PeakTimeStatisticsContainer,
    ReconstructedShowerContainer,
)

from ctapipe.coordinates import (
    CameraFrame,
    TelescopeFrame,
    GroundFrame,
    TiltedGroundFrame,
    project_to_ground,
    MissingFrameAttributeWarning,
)

logger = logging.getLogger(__name__)


def apply_simtel_r1_calibration(
    r0_waveforms, pedestal, dc_to_pe, gain_selector, calib_scale=1.0, calib_shift=0.0
):
    """
    Perform the R1 calibration for R0 simtel waveforms. This includes:
        - Gain selection
        - Pedestal subtraction
        - Conversion of samples into units proportional to photoelectrons
          (If the full signal in the waveform was integrated, then the resulting
          value would be in photoelectrons.)
          (Also applies flat-fielding)
    Parameters
    ----------
    r0_waveforms : ndarray
        Raw ADC waveforms from a simtel file. All gain channels available.
        Shape: (n_channels, n_pixels, n_samples)
    pedestal : ndarray
        Pedestal stored in the simtel file for each gain channel
        Shape: (n_channels, n_pixels)
    dc_to_pe : ndarray
        Conversion factor between R0 waveform samples and ~p.e., stored in the
        simtel file for each gain channel
        Shape: (n_channels, n_pixels)
    gain_selector : ctapipe.calib.camera.gainselection.GainSelector
    calib_scale : float
        Extra global scale factor for calibration.
        Conversion factor to transform the integrated charges
        (in ADC counts) into number of photoelectrons on top of dc_to_pe.
        Defaults to no scaling.
    calib_shift: float
        Shift the resulting R1 output in p.e. for simulating miscalibration.
        Defaults to no shift.
    Returns
    -------
    r1_waveforms : ndarray
        Calibrated waveforms
        Shape: (n_pixels, n_samples)
    selected_gain_channel : ndarray
        The gain channel selected for each pixel
        Shape: (n_pixels)
    """
    n_channels, n_pixels, n_samples = r0_waveforms.shape
    ped = pedestal[..., np.newaxis]
    DC_to_PHE = dc_to_pe[..., np.newaxis]
    gain = DC_to_PHE * calib_scale
    r1_waveforms = (r0_waveforms - ped) * gain + calib_shift
    if n_channels == 1:
        selected_gain_channel = np.zeros(n_pixels, dtype=np.int8)
        r1_waveforms = r1_waveforms[0]
    else:
        selected_gain_channel = gain_selector(r0_waveforms)
        r1_waveforms = r1_waveforms[selected_gain_channel, np.arange(n_pixels)]
    return r1_waveforms, selected_gain_channel


class MySimTelEventSource(SimTelEventSource):
    """ Read events from a SimTelArray data file (in EventIO format)."""

    calib_scale = Float(
        default_value=1.0,
        help=(
            "Factor to transform ADC counts into number of photoelectrons."
            " Corrects the DC_to_PHE factor."
        ),
    ).tag(config=True)

    calib_shift = Float(
        default_value=0.0,
        help=(
            "Factor to shift the R1 photoelectron samples. "
            "Can be used to simulate mis-calibration."
        ),
    ).tag(config=True)

    def _generate_events(self):
        data = ArrayEventContainer()
        data.simulation = SimulatedEventContainer()
        data.meta["origin"] = "hessio"
        data.meta["input_url"] = self.input_url
        data.meta["max_events"] = self.max_events

        self._fill_array_pointing(data)

        for counter, array_event in enumerate(self.file_):

            event_id = array_event.get("event_id", -1)
            obs_id = self.file_.header["run"]
            data.count = counter
            data.index.obs_id = obs_id
            data.index.event_id = event_id

            self._fill_trigger_info(data, array_event)

            if data.trigger.event_type == EventType.SUBARRAY:
                self._fill_simulated_event_information(data, array_event)

            # this should be done in a nicer way to not re-allocate the
            # data each time (right now it's just deleted and garbage
            # collected)
            data.r0.tel.clear()
            data.r1.tel.clear()
            data.dl0.tel.clear()
            data.dl1.tel.clear()
            data.pointing.tel.clear()
            data.simulation.tel.clear()

            telescope_events = array_event["telescope_events"]
            tracking_positions = array_event["tracking_positions"]

            for tel_id, telescope_event in telescope_events.items():
                adc_samples = telescope_event.get("adc_samples")
                if adc_samples is None:
                    adc_samples = telescope_event["adc_sums"][:, :, np.newaxis]

                n_gains, n_pixels, n_samples = adc_samples.shape
                true_image = (
                    array_event.get("photoelectrons", {})
                    .get(tel_id - 1, {})
                    .get("photoelectrons", None)
                )

                data.simulation.tel[tel_id] = SimulatedCameraContainer(
                    true_image=true_image
                )

                data.pointing.tel[tel_id] = self._fill_event_pointing(
                    tracking_positions[tel_id]
                )

                r0 = data.r0.tel[tel_id]
                r1 = data.r1.tel[tel_id]
                r0.waveform = adc_samples

                cam_mon = array_event["camera_monitorings"][tel_id]
                pedestal = cam_mon["pedestal"] / cam_mon["n_ped_slices"]
                dc_to_pe = array_event["laser_calibrations"][tel_id]["calib"]

                # fill dc_to_pe and pedestal_per_sample info into monitoring
                # container
                mon = data.mon.tel[tel_id]
                mon.calibration.dc_to_pe = dc_to_pe
                mon.calibration.pedestal_per_sample = pedestal

                r1.waveform, r1.selected_gain_channel = apply_simtel_r1_calibration(
                    adc_samples,
                    pedestal,
                    dc_to_pe,
                    self.gain_selector,
                    self.calib_scale,
                    self.calib_shift,
                )

                # get time_shift from laser calibration
                time_calib = array_event["laser_calibrations"][tel_id]["tm_calib"]
                pix_index = np.arange(n_pixels)

                dl1_calib = data.calibration.tel[tel_id].dl1
                dl1_calib.time_shift = time_calib[r1.selected_gain_channel, pix_index]

            yield data


class MyCameraCalibrator(CameraCalibrator):
    def _calibrate_dl1(self, event, telid):
        waveforms = event.dl0.tel[telid].waveform
        selected_gain_channel = event.dl0.tel[telid].selected_gain_channel
        dl1_calib = event.calibration.tel[telid].dl1

        if self._check_dl0_empty(waveforms):
            return

        selected_gain_channel = event.r1.tel[telid].selected_gain_channel
        time_shift = event.calibration.tel[telid].dl1.time_shift
        readout = self.subarray.tel[telid].camera.readout
        n_pixels, n_samples = waveforms.shape

        # subtract any remaining pedestal before extraction
        if dl1_calib.pedestal_offset is not None:
            # this copies intentionally, we don't want to modify the dl0 data
            # waveforms have shape (n_pixel, n_samples), pedestals (n_pixels, )
            waveforms = waveforms - dl1_calib.pedestal_offset[:, np.newaxis]

        if n_samples == 1:
            # To handle ASTRI and dst
            # TODO: Improved handling of ASTRI and dst
            #   - dst with custom EventSource?
            #   - Read into dl1 container directly?
            #   - Don't do anything if dl1 container already filled
            #   - Update on SST review decision
            charge = waveforms[..., 0].astype(np.float32)
            peak_time = np.zeros(n_pixels, dtype=np.float32)
        else:

            # shift waveforms if time_shift calibration is available
            if time_shift is not None:
                if self.apply_waveform_time_shift.tel[telid]:
                    sampling_rate = readout.sampling_rate.to_value(u.GHz)
                    time_shift_samples = time_shift * sampling_rate
                    waveforms, remaining_shift = shift_waveforms(
                        waveforms, time_shift_samples
                    )
                    remaining_shift /= sampling_rate
                else:
                    remaining_shift = time_shift

            extractor = self.image_extractors[self.image_extractor_type.tel[telid]]
            this_extractor_name = extractor.__class__.__name__
            if this_extractor_name == "TwoPassWindowSum":
                charge, peak_time, passed = extractor(
                    waveforms, telid=telid, selected_gain_channel=selected_gain_channel
                )
            else:
                charge, peak_time = extractor(
                    waveforms, telid=telid, selected_gain_channel=selected_gain_channel
                )

            # correct non-integer remainder of the shift if given
            if self.apply_peak_time_shift.tel[telid] and time_shift is not None:
                peak_time -= remaining_shift

        # Calibrate extracted charge
        charge *= dl1_calib.relative_factor / dl1_calib.absolute_factor

        event.dl1.tel[telid].image = charge
        event.dl1.tel[telid].peak_time = peak_time
        if this_extractor_name == "TwoPassWindowSum":
            return passed
        else:
            return 1

    def __call__(self, event):
        """
        Perform the full camera calibration from R1 to DL1. Any calibration
        relating to data levels before the data level the file is read into
        will be skipped.

        Parameters
        ----------
        event : container
            A `~ctapipe.containers.ArrayEventContainer` event container
        """
        # TODO: How to handle different calibrations depending on telid?
        tel = event.r1.tel or event.dl0.tel or event.dl1.tel
        passed = {}
        for telid in tel.keys():
            self._calibrate_dl0(event, telid)
            passed[telid] = self._calibrate_dl1(event, telid)

        return passed


class TwoPassWindowSum(ImageExtractor):
    """Extractor based on [1]_ which integrates the waveform a second time using
    a time-gradient linear fit. This is in particular the version implemented
    in the CTA-MARS analysis pipeline [2]_.

    Notes
    -----

    #. slide a 3-samples window through the waveform, finding max counts sum;
       the range of the sliding is the one allowing extension from 3 to 5;
       add 1 sample on each side and integrate charge in the 5-sample window;
       time is obtained as a charge-weighted average of the sample numbers;
       No information from neighboouring pixels is used.
    #. Preliminary image cleaning via simple tailcut with minimum number
       of core neighbours set at 1,
    #. Only the brightest cluster of pixels is kept.
    #. Parametrize following Hillas approach only if the resulting image has 3
       or more pixels.
    #. Do a linear fit of pulse time vs. distance along major image axis
       (CTA-MARS uses ROOT "robust" fit option,
       aka Least Trimmed Squares, to get rid of far outliers - this should
       be implemented in 'timing_parameters', e.g scipy.stats.siegelslopes).
    #. For all pixels except the core ones in the main island, integrate
       the waveform once more, in a fixed window of 5 samples set at the time
       "predicted" by the linear time fit.
       If the predicted time for a pixel leads to a window outside the readout
       window, then integrate the last (or first) 5 samples.
    #. The result is an image with main-island core pixels calibrated with a
       1st pass and non-core pixels re-calibrated with a 2nd pass.

    References
    ----------
    .. [1] J. Holder et al., Astroparticle Physics, 25, 6, 391 (2006)
    .. [2] https://forge.in2p3.fr/projects/step-by-step-reference-mars-analysis/wiki

    """

    # Get thresholds for core-pixels depending on telescope type.
    # WARNING: default values are not yet optimized
    core_threshold = FloatTelescopeParameter(
        default_value=[
            ("type", "*", 6.0),
            ("type", "LST*", 6.0),
            ("type", "MST*", 8.0),
            ("type", "SST*", 4.0),
        ],
        help="Picture threshold for internal tail-cuts pass",
    ).tag(config=True)

    disable_second_pass = Bool(
        default_value=False,
        help="only run the first pass of the extractor, for debugging purposes",
    ).tag(config=True)

    apply_integration_correction = BoolTelescopeParameter(
        default_value=True, help="Apply the integration window correction"
    ).tag(config=True)

    @lru_cache(maxsize=4096)
    def _calculate_correction(self, telid, width, shift):
        """Obtain the correction for the integration window specified for each
        pixel.

        The TwoPassWindowSum image extractor applies potentially different
        parameters for the integration window to each pixel, depending on the
        position of the peak. It has been decided to apply gain selection
        directly here. For basic definitions look at the documentation of
        `integration_correction`.

        Parameters
        ----------
        telid : int
            Index of the telescope in use.
        width : int
            Width of the integration window in samples
        shift : int
            Window shift to the left of the pulse peak in samples

        Returns
        -------
        correction : ndarray
            Value of the pixel-wise gain-selected integration correction.

        """
        readout = self.subarray.tel[telid].camera.readout
        # Calculate correction of first pixel for both channels
        return integration_correction(
            readout.reference_pulse_shape,
            readout.reference_pulse_sample_width.to_value("ns"),
            (1 / readout.sampling_rate).to_value("ns"),
            width,
            shift,
        )

    def _apply_first_pass(
        self, waveforms, telid
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Execute step 1.

        Parameters
        ----------
        waveforms : array of size (N_pixels, N_samples)
            DL0-level waveforms of one event.
        telid : int
            Index of the telescope.

        Returns
        -------
        charge : array_like
            Integrated charge per pixel.
            Shape: (n_pix)
        pulse_time : array_like
            Samples in which the waveform peak has been recognized.
            Shape: (n_pix)
        """
        # STEP 1

        # Starting from DL0, the channel is already selected (if more than one)
        # event.dl0.tel[tel_id].waveform object has shape (N_pixels, N_samples)

        # For each pixel, we slide a 3-samples window through the
        # waveform summing each time the ADC counts contained within it.

        peak_search_window_width = 3
        sums = convolve1d(
            waveforms, np.ones(peak_search_window_width), axis=1, mode="nearest"
        )
        # 'sums' has now still shape of (N_pixels, N_samples)
        # each element is the center-sample of each 3-samples sliding window

        # For each pixel, we check where the peak search window encountered
        # the maximum number of ADC counts.
        # We want to stop before the edge of the readout window in order to
        # later extend the search window to a 1+3+1 integration window.
        # Since in 'sums' the peak index corresponds to the center of the
        # search window, we shift it on the right by 2 samples so to get the
        # correspondent sample index in each waveform.
        peak_index = np.argmax(sums[:, 2:-2], axis=1) + 2
        # Now peak_index has the shape of (N_pixels).

        # The final 5-samples window will be 1+3+1, centered on the 3-samples
        # window in which the highest amount of ADC counts has been found
        window_width = peak_search_window_width + 2
        window_shift = 2

        # this function is applied to all pixels together
        charge_1stpass, pulse_time_1stpass = extract_around_peak(
            waveforms,
            peak_index,
            window_width,
            window_shift,
            self.sampling_rate_ghz[telid],
        )

        # Get integration correction factors
        if self.apply_integration_correction.tel[telid]:
            correction = self._calculate_correction(telid, window_width, window_shift)
        else:
            correction = np.ones(waveforms.shape[0])

        return charge_1stpass, pulse_time_1stpass, correction

    def _apply_second_pass(
        self,
        waveforms,
        telid,
        selected_gain_channel,
        charge_1stpass_uncorrected,
        pulse_time_1stpass,
        correction,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Follow steps from 2 to 7.

        Parameters
        ----------
        waveforms : array of shape (N_pixels, N_samples)
            DL0-level waveforms of one event.
        telid : int
            Index of the telescope.
        selected_gain_channel: array of shape (N_channels, N_pixels)
            Array containing the index of the selected gain channel for each
            pixel (0 for low gain, 1 for high gain).
        charge_1stpass_uncorrected : array of shape N_pixels
            Pixel charges reconstructed with the 1st pass, but not corrected.
        pulse_time_1stpass : array of shape N_pixels
            Pixel-wise pulse times reconstructed with the 1st pass.
        correction: array of shape N_pixels
            Charge correction from 1st pass.

        Returns
        -------
        charge : array_like
            Integrated charge per pixel.
            Note that in the case of a very bright full-camera image this can
            coincide the 1st pass information.
            Also in the case of very dim images the 1st pass will be recycled,
            but in this case the resulting image should be discarded
            from further analysis.
            Shape: (n_pix)
        pulse_time : array_like
            Samples in which the waveform peak has been recognized.
            Same specifications as above.
            Shape: (n_pix)
        """
        # STEP 2

        # Apply correction to 1st pass charges
        charge_1stpass = charge_1stpass_uncorrected * correction[selected_gain_channel]

        # Set thresholds for core-pixels depending on telescope
        core_th = self.core_threshold.tel[telid]
        # Boundary thresholds will be half of core thresholds.

        # Preliminary image cleaning with simple two-level tail-cut
        camera_geometry = self.subarray.tel[telid].camera.geometry
        mask_clean = tailcuts_clean(
            camera_geometry,
            charge_1stpass,
            picture_thresh=core_th,
            boundary_thresh=core_th / 2,
            keep_isolated_pixels=False,
            min_number_picture_neighbors=1,
        )

        # STEP 3

        # find all islands using this cleaning
        num_islands, labels = number_of_islands(camera_geometry, mask_clean)

        if num_islands > 0:
            # ...find the brightest one
            mask_brightest_island = brightest_island(
                num_islands, labels, charge_1stpass
            )
        else:
            mask_brightest_island = mask_clean

        # for all pixels except the core ones in the main island of the
        # preliminary image, the waveform will be integrated once more (2nd pass)

        mask_2nd_pass = ~mask_brightest_island | (
            mask_brightest_island & (charge_1stpass < core_th)
        )

        # STEP 4

        # if the resulting image has less then 3 pixels
        if np.count_nonzero(mask_brightest_island) < 3:
            # we return the 1st pass information
            passed = 0
            return charge_1stpass, pulse_time_1stpass, passed

        # otherwise we proceed by parametrizing the image
        camera_geometry_brightest = camera_geometry[mask_brightest_island]
        charge_brightest = charge_1stpass[mask_brightest_island]
        try:
            hillas = hillas_parameters(camera_geometry_brightest, charge_brightest)
        except (FloatingPointError, HillasParameterizationError, ValueError):
            # we return the 1st pass information
            passed = 0
            return charge_1stpass, pulse_time_1stpass, passed

        # STEP 5

        # linear fit of pulse time vs. distance along major image axis
        # using only the main island surviving the preliminary
        # image cleaning
        timing = timing_parameters(
            geom=camera_geometry_brightest,
            image=charge_brightest,
            peak_time=pulse_time_1stpass[mask_brightest_island],
            hillas_parameters=hillas,
        )

        # If the fit returns nan
        if np.isnan(timing.slope):
            passed = 0
            return charge_1stpass, pulse_time_1stpass, passed

        # get projected distances along main image axis
        longitude, _ = camera_to_shower_coordinates(
            camera_geometry.pix_x, camera_geometry.pix_y, hillas.x, hillas.y, hillas.psi
        )

        # get the predicted times as a linear relation
        predicted_pulse_times = (
            timing.slope * longitude[mask_2nd_pass] + timing.intercept
        )

        # Convert time in ns to sample index using the sampling rate from
        # the readout.
        # Approximate the value obtained to nearest integer, then cast to
        # int64 otherwise 'extract_around_peak' complains.

        predicted_peaks = np.rint(
            predicted_pulse_times.value * self.sampling_rate_ghz[telid]
        ).astype(np.int64)

        # Due to the fit these peak indexes can lead to an integration window
        # outside the readout window, so in the next step we check for this.

        # STEP 6

        # select all pixels except the core ones in the main island
        waveforms_to_repass = waveforms[mask_2nd_pass]

        # Build 'width' and 'shift' arrays that adapt on the position of the
        # window along each waveform

        # As before we will integrate the charge in a 5-sample window centered
        # on the peak
        window_width_default = 5
        window_shift_default = 2

        # first we find where the integration window edges WOULD BE
        integration_windows_start = predicted_peaks - window_shift_default
        integration_windows_end = integration_windows_start + window_width_default

        # then we define 2 possible edge cases
        # the predicted integration window falls before the readout window
        integration_before_readout = integration_windows_start < 0
        # or after
        integration_after_readout = integration_windows_end > (
            waveforms_to_repass.shape[1] - 1
        )

        # If the resulting 5-samples window falls before the readout
        # window we take the first 5 samples
        window_width_before = 5
        window_shift_before = 0

        # If the resulting 5-samples window falls after the readout
        # window we take the last 5 samples
        window_width_after = 6
        window_shift_after = 4

        # put all values of widths and shifts for 2nd pass pixels together
        window_widths = np.full(waveforms_to_repass.shape[0], window_width_default)
        window_widths[integration_before_readout] = window_width_before
        window_widths[integration_after_readout] = window_width_after
        window_shifts = np.full(waveforms_to_repass.shape[0], window_shift_default)
        window_shifts[integration_before_readout] = window_shift_before
        window_shifts[integration_after_readout] = window_shift_after

        # Now we have to (re)define the pathological predicted times for which
        # - either the peak itself falls outside of the readout window
        # - or is within the first or last 2 samples (so that at least 1 sample
        # of the integration window is outside of the readout window)
        # We place them at the first or last sample, so the special window
        # widhts and shifts that we defined earlier put the integration window
        # for these 2 cases either in the first 5 samples or the last

        # set sample to 0 (beginning of the waveform)
        # if predicted time falls before
        # but also if it's so near the edge that the integration window falls
        # outside
        predicted_peaks[predicted_peaks < 2] = 0

        # set sample to max-1 (first sample has index 0)
        # if predicted time falls after
        predicted_peaks[predicted_peaks > (waveforms_to_repass.shape[1] - 3)] = (
            waveforms_to_repass.shape[1] - 1
        )

        # re-calibrate 2nd pass pixels using the fixed 5-samples window
        reintegrated_charge, reestimated_pulse_times = extract_around_peak(
            waveforms_to_repass,
            predicted_peaks,
            window_widths,
            window_shifts,
            self.sampling_rate_ghz[telid],
        )

        if self.apply_integration_correction.tel[telid]:
            # Modify integration correction factors only for non-core pixels
            # now we compute 3 corrections for the default, before, and after cases:
            correction = self._calculate_correction(
                telid, window_width_default, window_shift_default
            )[selected_gain_channel][mask_2nd_pass]

            correction_before = self._calculate_correction(
                telid, window_width_before, window_shift_before
            )[selected_gain_channel][mask_2nd_pass]

            correction_after = self._calculate_correction(
                telid, window_width_after, window_shift_after
            )[selected_gain_channel][mask_2nd_pass]

            correction[integration_before_readout] = correction_before[
                integration_before_readout
            ]
            correction[integration_after_readout] = correction_after[
                integration_after_readout
            ]

            reintegrated_charge *= correction

        # STEP 7

        # Combine in the final output with,
        # - core pixels from the main cluster
        # - rest of the pixels which have been passed a second time

        # Start from a copy of the 1st pass charge
        charge_2ndpass = charge_1stpass.copy()
        # Overwrite the charges of pixels marked for second pass
        # leaving untouched the core pixels of the main island
        # from the preliminary (cleaned) image
        charge_2ndpass[mask_2nd_pass] = reintegrated_charge

        # Same approach for the pulse times
        pulse_time_2ndpass = pulse_time_1stpass.copy()
        pulse_time_2ndpass[mask_2nd_pass] = reestimated_pulse_times

        passed = 1

        return charge_2ndpass, pulse_time_2ndpass, passed

    def __call__(self, waveforms, telid, selected_gain_channel):
        """
        Call this ImageExtractor.

        Parameters
        ----------
        waveforms : array of shape (N_pixels, N_samples)
            DL0-level waveforms of one event.
        telid : int
            Index of the telescope.
        selected_gain_channel: array of shape (N_channels, N_pixels)
            Array containing the index of the selected gain channel for each
            pixel (0 for low gain, 1 for high gain).

        Returns
        -------
        charge : array_like
            Integrated charge per pixel.
            Shape: (n_pix)
        pulse_time : array_like
            Samples in which the waveform peak has been recognized.
            Shape: (n_pix)
        """

        charge1, pulse_time1, correction1 = self._apply_first_pass(waveforms, telid)

        # FIXME: properly make sure that output is 32Bit instead of downcasting here
        if self.disable_second_pass:
            passed = 1
            return (
                (charge1 * correction1[selected_gain_channel]).astype("float32"),
                pulse_time1.astype("float32"),
                passed,
            )

        charge2, pulse_time2, passed = self._apply_second_pass(
            waveforms, telid, selected_gain_channel, charge1, pulse_time1, correction1
        )
        # FIXME: properly make sure that output is 32Bit instead of downcasting here
        return charge2.astype("float32"), pulse_time2.astype("float32"), passed


class HillasParametersTelescopeFrameContainer(Container):
    container_prefix = "hillas"

    intensity = Field(nan, "total intensity (size)")

    x = Field(nan * u.deg, "centroid x coordinate", unit=u.deg)
    y = Field(nan * u.deg, "centroid x coordinate", unit=u.deg)
    r = Field(nan * u.deg, "radial coordinate of centroid", unit=u.deg)
    phi = Field(nan * u.deg, "polar coordinate of centroid", unit=u.deg)

    length = Field(nan * u.deg, "standard deviation along the major-axis", unit=u.deg)
    width = Field(nan * u.deg, "standard spread along the minor-axis", unit=u.deg)
    psi = Field(nan * u.deg, "rotation angle of ellipse", unit=u.deg)

    skewness = Field(nan, "measure of the asymmetry")
    kurtosis = Field(nan, "measure of the tailedness")


class CoreParametersContainer(Container):
    """Telescope-wise shower's direction in the Tilted/Ground Frame"""

    container_prefix = "core"
    psi = Field(nan * u.deg, "Image direction in the Tilted/Ground Frame", unit="deg")


class ImageParametersContainer(Container):
    """ Collection of image parameters """

    container_prefix = "params"
    hillas = Field(HillasParametersContainer(), "Hillas Parameters")
    timing = Field(TimingParametersContainer(), "Timing Parameters")
    leakage = Field(LeakageContainer(), "Leakage Parameters")
    concentration = Field(ConcentrationContainer(), "Concentration Parameters")
    morphology = Field(MorphologyContainer(), "Image Morphology Parameters")
    intensity_statistics = Field(
        IntensityStatisticsContainer(), "Intensity image statistics"
    )
    peak_time_statistics = Field(
        PeakTimeStatisticsContainer(), "Peak time image statistics"
    )
    core = Field(
        CoreParametersContainer(), "Image direction in the Tilted/Ground Frame"
    )


class HillasReconstructor(Reconstructor):
    """
    class that reconstructs the direction of an atmospheric shower
    using a simple hillas parametrisation of the camera images it
    provides a direction estimate in two steps and an estimate for the
    shower's impact position on the ground.
    so far, it does neither provide an energy estimator nor an
    uncertainty on the reconstructed parameters
    """

    def __init__(self, subarray, config=None, parent=None, **kwargs):
        super().__init__(config=config, parent=parent, **kwargs)
        self.subarray = subarray

        self._cam_radius_m = {
            cam: cam.geometry.guess_radius() for cam in subarray.camera_types
        }

        telframe = TelescopeFrame()
        self._cam_radius_deg = {}
        for cam, radius_m in self._cam_radius_m.items():
            point_cam = SkyCoord(0 * u.m, radius_m, frame=cam.geometry.frame)
            point_tel = point_cam.transform_to(telframe)
            self._cam_radius_deg[cam] = point_tel.fov_lon

    def __call__(self, event):
        """
        Perform the full shower geometry reconstruction on the input event.
        Parameters
        ----------
        event : container
            `ctapipe.containers.ArrayEventContainer`
        """

        # Read only valid HillasContainers
        hillas_dict = {
            tel_id: dl1.parameters.hillas
            for tel_id, dl1 in event.dl1.tel.items()
            if np.isfinite(dl1.parameters.hillas.intensity)
        }

        # Due to tracking the pointing of the array will never be a constant
        array_pointing = SkyCoord(
            az=event.pointing.array_azimuth,
            alt=event.pointing.array_altitude,
            frame=AltAz(),
        )

        telescope_pointings = {
            tel_id: SkyCoord(
                alt=event.pointing.tel[tel_id].altitude,
                az=event.pointing.tel[tel_id].azimuth,
                frame=AltAz(),
            )
            for tel_id in event.dl1.tel.keys()
        }

        try:
            result = self._predict(
                event, hillas_dict, self.subarray, array_pointing, telescope_pointings
            )
        except (TooFewTelescopesException, InvalidWidthException):
            result = ReconstructedShowerContainer()

        return result

    def _predict(
        self, event, hillas_dict, subarray, array_pointing, telescopes_pointings
    ):
        """
        The function you want to call for the reconstruction of the
        event. It takes care of setting up the event and consecutively
        calls the functions for the direction and core position
        reconstruction.  Shower parameters not reconstructed by this
        class are set to np.nan
        Parameters
        -----------
        hillas_dict: dict
            dictionary with telescope IDs as key and
            HillasParametersContainer instances as values
        inst : ctapipe.io.InstrumentContainer
            instrumental description
        array_pointing: SkyCoord[AltAz]
            pointing direction of the array
        telescopes_pointings: dict[SkyCoord[AltAz]]
            dictionary of pointing direction per each telescope
        Raises
        ------
        TooFewTelescopesException
            if len(hillas_dict) < 2
        InvalidWidthException
            if any width is np.nan or 0
        """

        # filter warnings for missing obs time. this is needed because MC data has no obs time
        warnings.filterwarnings(action="ignore", category=MissingFrameAttributeWarning)

        # Here we perform some basic quality checks BEFORE applying reconstruction
        # This should be substituted by a DL1 QualityQuery specific to this
        # reconstructor

        # stereoscopy needs at least two telescopes
        if len(hillas_dict) < 2:
            raise TooFewTelescopesException(
                "need at least two telescopes, have {}".format(len(hillas_dict))
            )
        # check for np.nan or 0 width's as these screw up weights
        if any([np.isnan(hillas_dict[tel]["width"].value) for tel in hillas_dict]):
            raise InvalidWidthException(
                "A HillasContainer contains an ellipse of width==np.nan"
            )
        if any([hillas_dict[tel]["width"].value == 0 for tel in hillas_dict]):
            raise InvalidWidthException(
                "A HillasContainer contains an ellipse of width==0"
            )

        hillas_planes, psi_core_dict = self.initialize_hillas_planes(
            hillas_dict, subarray, telescopes_pointings, array_pointing
        )

        # algebraic direction estimate
        direction, err_est_dir = self.estimate_direction(hillas_planes)

        # array pointing is needed to define the tilted frame
        core_pos = self.estimate_core_position(
            event, hillas_dict, array_pointing, psi_core_dict, hillas_planes
        )

        # container class for reconstructed showers
        _, lat, lon = cartesian_to_spherical(*direction)

        # estimate max height of shower
        h_max = self.estimate_h_max(hillas_planes)

        # astropy's coordinates system rotates counter-clockwise.
        # Apparently we assume it to be clockwise.
        # that's why lon get's a sign
        result = ReconstructedShowerContainer(
            alt=lat,
            az=-lon,
            core_x=core_pos[0],
            core_y=core_pos[1],
            tel_ids=[h for h in hillas_dict.keys()],
            average_intensity=np.mean([h.intensity for h in hillas_dict.values()]),
            is_valid=True,
            alt_uncert=err_est_dir,
            az_uncert=err_est_dir,
            h_max=h_max,
        )

        return result

    def initialize_hillas_planes(
        self, hillas_dict, subarray, telescopes_pointings, array_pointing
    ):
        """
        Creates a dictionary of `.HillasPlane` from a dictionary of
        hillas parameters
        Parameters
        ----------
        hillas_dict : dictionary
            dictionary of hillas moments
        subarray : ctapipe.instrument.SubarrayDescription
            subarray information
        telescopes_pointings: dictionary
            dictionary of pointing direction per each telescope
        array_pointing: SkyCoord[AltAz]
            pointing direction of the array
        Notes
        -----
        The part of the algorithm taking into account divergent pointing
        mode and the correction to the psi angle is explained in [gasparetto]_
        section 7.1.4.
        """

        hillas_planes = {}

        # dictionary to store the telescope-wise image directions
        # to be projected on the ground and corrected in case of mispointing
        corrected_angle_dict = {}

        k = next(iter(telescopes_pointings))
        horizon_frame = telescopes_pointings[k].frame
        for tel_id, moments in hillas_dict.items():

            camera = self.subarray.tel[tel_id].camera

            pointing = SkyCoord(
                alt=telescopes_pointings[tel_id].alt,
                az=telescopes_pointings[tel_id].az,
                frame=horizon_frame,
            )

            focal_length = subarray.tel[tel_id].optics.equivalent_focal_length

            if moments.x.unit.is_equivalent(u.m):  # Image parameters are in CameraFrame

                # we just need any point on the main shower axis a bit away from the cog
                p2_x = moments.x + 0.1 * self._cam_radius_m[camera] * np.cos(
                    moments.psi
                )
                p2_y = moments.y + 0.1 * self._cam_radius_m[camera] * np.sin(
                    moments.psi
                )

                camera_frame = CameraFrame(
                    focal_length=focal_length, telescope_pointing=pointing
                )

                cog_coord = SkyCoord(x=moments.x, y=moments.y, frame=camera_frame)
                p2_coord = SkyCoord(x=p2_x, y=p2_y, frame=camera_frame)

            else:  # Image parameters are already in TelescopeFrame

                # we just need any point on the main shower axis a bit away from the cog
                p2_delta_alt = moments.y + 0.1 * self._cam_radius_deg[camera] * np.sin(
                    moments.psi
                )
                p2_delta_az = moments.x + 0.1 * self._cam_radius_deg[camera] * np.cos(
                    moments.psi
                )

                telescope_frame = TelescopeFrame(telescope_pointing=pointing)

                cog_coord = SkyCoord(
                    fov_lon=moments.x, fov_lat=moments.y, frame=telescope_frame
                )
                p2_coord = SkyCoord(
                    fov_lon=p2_delta_az, fov_lat=p2_delta_alt, frame=telescope_frame
                )

            # Coordinates in the sky
            cog_coord = cog_coord.transform_to(horizon_frame)
            p2_coord = p2_coord.transform_to(horizon_frame)

            # re-project from sky to a "fake"-parallel-pointing telescope
            # then recalculate the psi angle in order to be able to project
            # it on the ground
            # This is done to bypass divergent pointing or mispointing

            camera_frame_parallel = CameraFrame(
                focal_length=focal_length, telescope_pointing=array_pointing
            )
            cog_sky_to_parallel = cog_coord.transform_to(camera_frame_parallel)
            p2_sky_to_parallel = p2_coord.transform_to(camera_frame_parallel)
            angle_psi_corr = np.arctan2(
                cog_sky_to_parallel.y - p2_sky_to_parallel.y,
                cog_sky_to_parallel.x - p2_sky_to_parallel.x,
            )

            corrected_angle_dict[tel_id] = angle_psi_corr

            circle = HillasPlane(
                p1=cog_coord,
                p2=p2_coord,
                telescope_position=subarray.positions[tel_id],
                weight=moments.intensity * (moments.length / moments.width),
            )
            hillas_planes[tel_id] = circle

        return hillas_planes, corrected_angle_dict

    def estimate_direction(self, hillas_planes):
        """calculates the origin of the gamma as the weighted average
        direction of the intersections of all hillas planes
        Returns
        -------
        gamma : shape (3) numpy array
            direction of origin of the reconstructed shower as a 3D vector
        crossings : shape (n,3) list
            an error estimate
        """

        crossings = []
        for perm in combinations(hillas_planes.values(), 2):
            n1, n2 = perm[0].norm, perm[1].norm
            # cross product automatically weighs in the angle between
            # the two vectors: narrower angles have less impact,
            # perpendicular vectors have the most
            crossing = np.cross(n1, n2)

            # two great circles cross each other twice (one would be
            # the origin, the other one the direction of the gamma) it
            # doesn't matter which we pick but it should at least be
            # consistent: make sure to always take the "upper" solution
            if crossing[2] < 0:
                crossing *= -1
            crossings.append(crossing * perm[0].weight * perm[1].weight)

        result = normalise(np.sum(crossings, axis=0))
        off_angles = [angle(result, cross) for cross in crossings] * u.rad

        err_est_dir = np.average(
            off_angles, weights=[len(cross) for cross in crossings]
        )

        return result, err_est_dir

    def estimate_core_position(
        self, event, hillas_dict, array_pointing, corrected_angle_dict, hillas_planes
    ):
        """
        Estimate the core position by intersection the major ellipse lines of each telescope.
        Parameters
        -----------
        hillas_dict: dict[HillasContainer]
            dictionary of hillas moments
        array_pointing: SkyCoord[HorizonFrame]
            Pointing direction of the array
        Returns
        -----------
        core_x: u.Quantity
            estimated x position of impact
        core_y: u.Quantity
            estimated y position of impact
        Notes
        -----
        The part of the algorithm taking into account divergent pointing
        mode and the usage of a corrected psi angle is explained in [gasparetto]_
        section 7.1.4.
        """

        # Since psi has been recalculated in the fake CameraFrame
        # it doesn't need any further corrections because it is now independent
        # of both pointing and cleaning/parametrization frame.
        # This angle will be used to visualize the telescope-wise directions of
        # the shower core the ground.
        psi_core = corrected_angle_dict

        # Record these values
        for tel_id in hillas_dict.keys():
            event.dl1.tel[tel_id].parameters = ImageParametersContainer()
            event.dl1.tel[tel_id].parameters.core = CoreParametersContainer()
            event.dl1.tel[tel_id].parameters.core.psi = psi_core[tel_id]

        # Transform them for numpy
        psi = u.Quantity(list(psi_core.values()))

        # Estimate the position of the shower's core
        # from the TiltedFram to the GroundFrame

        z = np.zeros(len(psi))
        uvw_vectors = np.column_stack([np.cos(psi).value, np.sin(psi).value, z])

        tilted_frame = TiltedGroundFrame(pointing_direction=array_pointing)
        ground_frame = GroundFrame()

        positions = [
            (
                SkyCoord(*plane.pos, frame=ground_frame)
                .transform_to(tilted_frame)
                .cartesian.xyz
            )
            for plane in hillas_planes.values()
        ]

        core_position = line_line_intersection_3d(uvw_vectors, positions)

        core_pos_tilted = SkyCoord(
            x=core_position[0] * u.m, y=core_position[1] * u.m, frame=tilted_frame
        )

        core_pos = project_to_ground(core_pos_tilted)

        return core_pos.x, core_pos.y

    def estimate_h_max(self, hillas_planes):
        """
        Estimate the max height by intersecting the lines of the cog directions of each telescope.
        Returns
        -----------
        astropy.unit.Quantity
            the estimated max height
        """
        uvw_vectors = np.array([plane.a for plane in hillas_planes.values()])
        positions = [plane.pos for plane in hillas_planes.values()]

        # not sure if its better to return the length of the vector of the z component
        return np.linalg.norm(line_line_intersection_3d(uvw_vectors, positions)) * u.m
