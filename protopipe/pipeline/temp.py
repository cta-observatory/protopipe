"""Temporary definitions of classes and methods.

In this file we store spin-offs of ctapipe's API components which are needed
for the development of protopipe.

Normally tests are performed with protopipe and, if positive, later pushed to
ctapipe for possible futher modifications.
As newer releases of ctapipe are imported in protopipe, the content of this
file will change.

"""

import logging

import numpy as np
from numpy import nan
import astropy.units as u
from astropy.coordinates import SkyCoord

from ctapipe.io import SimTelEventSource
from ctapipe.containers import (ArrayEventContainer,
                                SimulatedCameraContainer,
                                SimulatedEventContainer,
                                EventType)
from ctapipe.core import Container, Field
from ctapipe.core.traits import Float
from ctapipe.coordinates import CameraFrame, TelescopeFrame
from ctapipe.instrument import CameraGeometry
from ctapipe.reco.hillas_reconstructor import HillasReconstructor
from ctapipe.reco.hillas_reconstructor import HillasPlane

logger = logging.getLogger(__name__)


def apply_simtel_r1_calibration(r0_waveforms,
                                pedestal,
                                dc_to_pe,
                                gain_selector,
                                calib_scale=1.0,
                                calib_shift=0.0):
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
        )
    ).tag(config=True)

    calib_shift = Float(
        default_value=0.0,
        help=(
            "Factor to shift the R1 photoelectron samples. "
            "Can be used to simulate mis-calibration."
        )
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
                    self.calib_shift
                )

                # get time_shift from laser calibration
                time_calib = array_event["laser_calibrations"][tel_id]["tm_calib"]
                pix_index = np.arange(n_pixels)

                dl1_calib = data.calibration.tel[tel_id].dl1
                dl1_calib.time_shift = time_calib[r1.selected_gain_channel, pix_index]

            yield data


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


class MyCameraGeometry(CameraGeometry):

    """Modifications inspired from PR 1191."""

    def transform_to(self, frame):
        """
        Transform the pixel coordinates stored in this geometry
        and the pixel and camera rotations to another camera coordinate frame.

        Parameters
        ----------
        frame: ctapipe.coordinates.CameraFrame
            The coordinate frame to transform to.
        """

        coord = SkyCoord(x=self.pix_x, y=self.pix_y, frame=self.frame)
        trans = coord.transform_to(frame)

        # also transform the unit vectors, to get rotation / mirroring
        uv = SkyCoord(x=[1, 0], y=[0, 1], unit=self.pix_x.unit, frame=self.frame)
        uv_trans = uv.transform_to(frame)

        try:
            rot = np.arctan2(uv_trans[0].y, uv_trans[1].y)
            det = np.linalg.det([uv_trans.x.value, uv_trans.y.value])
        except AttributeError:
            rot = np.arctan2(uv_trans[0].fov_lat, uv_trans[1].fov_lat)
            det = np.linalg.det([uv_trans.fov_lon.value, uv_trans.fov_lat.value])

        cam_rotation = rot + det * self.cam_rotation
        pix_rotation = rot + det * self.pix_rotation

        try:
            return CameraGeometry(
                cam_id=self.cam_id,
                pix_id=self.pix_id,
                pix_x=trans.x,
                pix_y=trans.y,
                pix_area=self.pix_area,
                pix_type=self.pix_type,
                sampling_rate=self.sampling_rate,
                pix_rotation=pix_rotation,
                cam_rotation=cam_rotation,
                neighbors=None,
                apply_derotation=False,
                frame=frame,
            )
        except AttributeError:
            return CameraGeometry(
                camera_name=self.camera_name,
                pix_id=self.pix_id,
                pix_x=trans.fov_lat,
                pix_y=trans.fov_lon,
                pix_area=CameraGeometry.guess_pixel_area(
                    trans.fov_lat, trans.fov_lon, self.pix_type
                ),
                pix_type=self.pix_type,
                pix_rotation=pix_rotation,
                cam_rotation=cam_rotation,
                neighbors=None,
                apply_derotation=False,
                frame=frame,
            )


class MyHillasReconstructor(HillasReconstructor):

    """Child class of MyHillasReconstructor with optional input frames."""

    def initialize_hillas_planes(
        self, hillas_dict, subarray, telescopes_pointings, array_pointing
    ):
        """
        Creates a dictionary of :class:`.HillasPlane` from a dictionary of
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
        """

        self.hillas_planes = {}
        k = next(iter(telescopes_pointings))
        horizon_frame = telescopes_pointings[k].frame
        for tel_id, moments in hillas_dict.items():

            pointing = SkyCoord(
                alt=telescopes_pointings[tel_id].alt,
                az=telescopes_pointings[tel_id].az,
                frame=horizon_frame,
            )

            if moments.x.unit == u.Unit("m"):  # Image parameters are in CameraFrame

                # we just need any point on the main shower axis a bit away from the cog
                p2_x = moments.x + 0.1 * u.m * np.cos(moments.psi)
                p2_y = moments.y + 0.1 * u.m * np.sin(moments.psi)
                focal_length = subarray.tel[tel_id].optics.equivalent_focal_length

                camera_frame = CameraFrame(
                    focal_length=focal_length, telescope_pointing=pointing
                )

                cog_coord = SkyCoord(x=moments.x, y=moments.y, frame=camera_frame,)
                p2_coord = SkyCoord(x=p2_x, y=p2_y, frame=camera_frame)

                # ============
                # DIVERGENT

                # re-project from sky to a "fake"-parallel-pointing telescope
                # then recalculate the psi angle

                # WARNING: this part will need to be reproduced accordingly in the TelescopeFrame case!

                if self.divergent_mode:
                    camera_frame_parallel = CameraFrame(
                        focal_length=focal_length, telescope_pointing=array_pointing
                    )
                    cog_sky_to_parallel = cog_coord.transform_to(camera_frame_parallel)
                    p2_sky_to_parallel = p2_coord.transform_to(camera_frame_parallel)
                    angle_psi_corr = np.arctan2(
                        cog_sky_to_parallel.y - p2_sky_to_parallel.y,
                        cog_sky_to_parallel.x - p2_sky_to_parallel.x,
                    )
                    self.corrected_angle_dict[tel_id] = angle_psi_corr

                # ============

            else:  # Image parameters are already in TelescopeFrame

                # we just need any point on the main shower axis a bit away from the cog
                p2_x = moments.x + 0.1 * u.deg * np.cos(moments.psi)
                p2_y = moments.y + 0.1 * u.deg * np.sin(moments.psi)

                telescope_frame = TelescopeFrame(telescope_pointing=pointing)

                cog_coord = SkyCoord(
                    fov_lon=moments.x, fov_lat=moments.y, frame=telescope_frame,
                )
                p2_coord = SkyCoord(fov_lon=p2_x, fov_lat=p2_y, frame=telescope_frame)

            cog_coord = cog_coord.transform_to(horizon_frame)
            p2_coord = p2_coord.transform_to(horizon_frame)

            circle = HillasPlane(
                p1=cog_coord,
                p2=p2_coord,
                telescope_position=subarray.positions[tel_id],
                weight=moments.intensity * (moments.length / moments.width),
            )
            self.hillas_planes[tel_id] = circle
