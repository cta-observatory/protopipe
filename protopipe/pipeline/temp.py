"""Temporary definitions of classes and methods.

In this file we store spin-offs of ctapipe's API components which are needed
for the development of protopipe.

Normally tests are performed with protopipe and, if positive, later pushed to
ctapipe for possible futher modifications.
As newer releases of ctapipe are imported in protopipe, the content of this
file will change.

"""

from ctapipe.instrument import CameraGeometry


class MyCameraGeometry(CameraGeometry):

    """Child class of the usual CameraGeometry with modifications taken from PR #1191"""

    def __init__(
        self,
        camera_name,
        pix_id,
        pix_x,
        pix_y,
        pix_area,
        pix_type,
        pix_rotation="0d",
        cam_rotation="0d",
        neighbors=None,
        apply_derotation=True,
        frame=None,
        equivalent_focal_length=None,  # this is new
    ):

        if pix_x.ndim != 1 or pix_y.ndim != 1:
            raise ValueError(
                f"Pixel coordinates must be 1 dimensional, got {pix_x.ndim}"
            )

        assert len(pix_x) == len(pix_y), "pix_x and pix_y must have same length"
        self.n_pixels = len(pix_x)
        self.camera_name = camera_name
        self.pix_id = pix_id
        self.pix_x = pix_x
        self.pix_y = pix_y
        self.pix_area = pix_area
        self.pix_type = pix_type
        self.pix_rotation = Angle(pix_rotation)
        self.cam_rotation = Angle(cam_rotation)
        self._neighbors = neighbors
        self.frame = frame
        self.equivalent_focal_length = equivalent_focal_length  # this is new

        if neighbors is not None:
            if isinstance(neighbors, list):
                lil = lil_matrix((self.n_pixels, self.n_pixels), dtype=bool)
                for pix_id, neighbors in enumerate(neighbors):
                    lil[pix_id, neighbors] = True
                self._neighbors = lil.tocsr()
            else:
                self._neighbors = csr_matrix(neighbors)

        if self.pix_area is None:
            self.pix_area = self.guess_pixel_area(pix_x, pix_y, pix_type)

        if apply_derotation:
            # todo: this should probably not be done, but need to fix
            # GeometryConverter and reco algorithms if we change it.
            self.rotate(cam_rotation)

        # cache border pixel mask per instance
        self.border_cache = {}

    def transform_to(self, frame):
        """
        Transform the pixel coordinates stored in this geometry
        and the pixel and camera rotations to another camera coordinate frame.

        Parameters
        ----------
        frame: ctapipe.coordinates.CameraFrame
            The coordinate frame to transform to.
        """
        if self.frame is None:
            if self.equivalent_focal_length is not None:
                self.frame = CameraFrame(focal_length=self.equivalent_focal_length)
            else:
                self.frame = CameraFrame()

        coord = SkyCoord(x=self.pix_x, y=self.pix_y, frame=self.frame)
        trans = coord.transform_to(frame)

        # also transform the unit vectors, to get rotation / mirroring
        uv = SkyCoord(x=[1, 0], y=[0, 1], unit=self.pix_x.unit, frame=self.frame)
        uv_trans = uv.transform_to(frame)

        rot = np.arctan2(uv_trans[0].fov_lat, uv_trans[1].fov_lat)
        det = np.linalg.det([uv_trans.fov_lon.value, uv_trans.fov_lat.value])

        # try:
        #    rot = np.arctan2(uv_trans[0].fov_lat, uv_trans[1].fov_lat)
        #    det = np.linalg.det([uv_trans.fov_lat.value, uv_trans.fov_lat.value])
        # except AttributeError:
        #    rot = np.arctan2(uv_trans[0].fov_lat, uv_trans[1].fov_lat)
        #    det = np.linalg.det([uv_trans.x.value, uv_trans.fov_lat.value])

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
