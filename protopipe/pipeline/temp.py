"""Temporary definitions of classes and methods.

In this file we store spin-offs of ctapipe's API components which are needed
for the development of protopipe.

Normally tests are performed with protopipe and, if positive, later pushed to
ctapipe for possible futher modifications.
As newer releases of ctapipe are imported in protopipe, the content of this
file will change.

"""

import logging
import warnings

import numpy as np
from numpy import nan
from astropy import units as u

from ctapipe.io import SimTelEventSource
from ctapipe.containers import (ArrayEventContainer,
                                SimulatedCameraContainer,
                                SimulatedEventContainer,
                                EventType)
from ctapipe.core import Container, Field
from ctapipe.core.traits import Float
from ctapipe.reco.hillas_reconstructor import HillasPlane, normalise, angle, line_line_intersection_3d
from ctapipe.reco.reco_algorithms import (
    Reconstructor,
    InvalidWidthException,
    TooFewTelescopesException,
)
from ctapipe.containers import (HillasParametersContainer,
                                TimingParametersContainer,
                                LeakageContainer,
                                MorphologyContainer,
                                ConcentrationContainer,
                                IntensityStatisticsContainer,
                                PeakTimeStatisticsContainer,
                                ReconstructedShowerContainer)
from itertools import combinations

from ctapipe.coordinates import (
    CameraFrame,
    TelescopeFrame,
    GroundFrame,
    TiltedGroundFrame,
    project_to_ground,
    MissingFrameAttributeWarning,
)
from astropy.coordinates import (
    SkyCoord,
    AltAz,
    spherical_to_cartesian,
    cartesian_to_spherical,
)

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