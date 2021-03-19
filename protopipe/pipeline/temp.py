"""Temporary definitions of classes and methods.

In this file we store spin-offs of ctapipe's API components which are needed
for the development of protopipe.

Normally tests are performed with protopipe and, if positive, later pushed to
ctapipe for possible futher modifications.
As newer releases of ctapipe are imported in protopipe, the content of this
file will change.

"""

import os
import time
import logging
from pathlib import Path

import requests
from tqdm import tqdm
from urllib.parse import urlparse
import numpy as np
from numpy import nan
import astropy.units as u
from astropy.coordinates import SkyCoord
from requests.exceptions import HTTPError

from ctapipe.core import Container, Field
from ctapipe.coordinates import CameraFrame, TelescopeFrame
from ctapipe.instrument import CameraGeometry
from ctapipe.reco import HillasReconstructor
from ctapipe.reco.HillasReconstructor import HillasPlane

logger = logging.getLogger(__name__)


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


try:
    import ctapipe_resources

    has_resources = True
except ImportError:
    has_resources = False


def download_file(url, path, auth=None, chunk_size=10240, progress=False):
    """
    Download a file. Will write to ``path + '.part'`` while downloading
    and rename after successful download to the final name.
    Parameters
    ----------
    url: str or url
        The URL to download
    path: pathlib.Path or str
        Where to store the downloaded data.
    auth: None or tuple of (username, password) or a request.AuthBase instance.
    chunk_size: int
        Chunk size for writing the data file, 10 kB by default.
    """
    logger.info(f"Downloading {url} to {path}")
    name = urlparse(url).path.split("/")[-1]
    path = Path(path)

    with requests.get(url, stream=True, auth=auth, timeout=5) as r:
        # make sure the request is successful
        r.raise_for_status()

        total = float(r.headers.get("Content-Length", float("inf")))
        pbar = tqdm(
            total=total,
            disable=not progress,
            unit="B",
            unit_scale=True,
            desc=f"Downloading {name}",
        )

        try:
            # open a .part file to avoid creating
            # a broken file at the intended location
            part_file = path.with_suffix(path.suffix + ".part")

            part_file.parent.mkdir(parents=True, exist_ok=True)
            with part_file.open("wb") as f:
                for chunk in r.iter_content(chunk_size=chunk_size):
                    f.write(chunk)
                    pbar.update(len(chunk))
        except:  # we really want to catch everythin here
            # cleanup part file if something goes wrong
            if part_file.is_file():
                part_file.unlink()
            raise

    # when successful, move to intended location
    part_file.rename(path)


def get_cache_path(url, cache_name="ctapipe", env_override="CTAPIPE_CACHE"):
    if os.getenv(env_override):
        base = Path(os.environ["CTAPIPE_CACHE"])
    else:
        base = Path(os.environ["HOME"]) / ".cache" / cache_name

    url = urlparse(url)

    path = os.path.join(url.netloc.rstrip("/"), url.path.lstrip("/"))
    path = base / path
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def download_file_cached(
    name,
    cache_name="ctapipe",
    auth=None,
    env_prefix="CTAPIPE_DATA_",
    default_url="http://cccta-dataserver.in2p3.fr/data/",
    progress=False,
):
    """
    Downloads a file from a dataserver and caches the result locally
    in ``$HOME/.cache/<cache_name>``.
    If the file is found in the cache, no new download is performed.
    Parameters
    ----------
    name: str or pathlib.Path
        the name of the file, relative to the data server url
    cache_name: str
        What name to use for the cache directory
    env_prefix: str
        Prefix for the environt variables used for overriding the URL,
        and providing username and password in case authentication is required.
    auth: True, None or tuple of (username, password)
        Authentication data for the request. Will be passed to ``requests.get``.
        If ``True``, read username and password for the request from
        the env variables ``env_prefix + 'USER'`` and ``env_prefix + PASSWORD``
    default_url: str
        The default url from which to download ``name``, can be overriden
        by setting the env variable ``env_prefix + URL``
    Returns
    -------
    path: pathlib.Path
        the full path to the downloaded data.
    """
    logger.debug(f"File {name} is not available in cache, downloading.")

    base_url = os.environ.get(env_prefix + "URL", default_url).rstrip("/")
    url = base_url + "/" + str(name).lstrip("/")

    path = get_cache_path(url, cache_name=cache_name)
    part_file = path.with_suffix(path.suffix + ".part")

    if part_file.is_file():
        logger.warning("Another download for this file is already running, waiting.")
        while part_file.is_file():
            time.sleep(1)

    # if we already dowloaded the file, just use it
    if path.is_file():
        logger.debug(f"File {name} is available in cache.")
        return path

    if auth is True:
        try:
            auth = (
                os.environ[env_prefix + "USER"],
                os.environ[env_prefix + "PASSWORD"],
            )
        except KeyError:
            raise KeyError(
                f'You need to set the env variables "{env_prefix}USER"'
                f' and "{env_prefix}PASSWORD" to download test files.'
            ) from None

    download_file(url=url, path=path, auth=auth, progress=progress)
    return path


DEFAULT_URL = "http://cccta-dataserver.in2p3.fr/data/ctapipe-extra/v0.3.3/"
def get_dataset_path(filename, url=DEFAULT_URL):
    """
    Returns the full file path to an auxiliary dataset needed by
    ctapipe, given the dataset's full name (filename with no directory).
    This will first search for the file in directories listed in
    tne environment variable CTAPIPE_SVC_PATH (if set), and if not found,
    will look in the ctapipe_resources module
    (if installed with the ctapipe-extra package), which contains the defaults.
    Parameters
    ----------
    filename: str
        name of dataset to fetch
    Returns
    -------
    string with full path to the given dataset
    """

    searchpath = os.getenv("CTAPIPE_SVC_PATH")

    if searchpath:
        filepath = find_in_path(filename=filename, searchpath=searchpath, url=url)

        if filepath:
            return filepath

    if has_resources and (url is DEFAULT_URL):
        logger.debug(
            "Resource '{}' not found in CTAPIPE_SVC_PATH, looking in "
            "ctapipe_resources...".format(filename)
        )

        return Path(ctapipe_resources.get(filename))

    # last, try downloading the data
    try:
        return download_file_cached(filename, default_url=url, progress=True)
    except HTTPError as e:
        # let 404 raise the FileNotFoundError instead of HTTPError
        if e.response.status_code != 404:
            raise

    raise FileNotFoundError(
        f"Couldn't find resource: '{filename}',"
        " You might want to install ctapipe_resources"
    )