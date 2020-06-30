import numpy as np
from ctapipe.image.cleaning import mars_cleaning_1st_pass

try:
    from pywicta.denoising.wavelets_mrtransform import WaveletTransform
    from pywicta.denoising import cdf
    from pywicta.denoising.inverse_transform_sampling import EmpiricalDistribution
    from pywicta.io import geometry_converter
    from pywi.processing.filtering.pixel_clusters import filter_pixels_clusters
    from pywi.processing.filtering import pixel_clusters
except ImportError as e:
    print("pywicta package could not be imported")
    print("wavelet cleaning will not work")
    print(e)

__all__ = ["ImageCleaner"]


class ImageCleaner(object):
    """Class applying image cleaning. It can handle wavelet or tail-cuts-based methods.

    Parameters
    ----------
    config: dict
        Configuration file with sections corresponding to wave or tail
    mode: str
        Model corresponding to `wave` or `tail`
    """

    def __init__(self, config, cameras, mode="tail"):

        self.config = config
        self.mode = mode
        self.initialise_clean_opt(cameras)

    def initialise_clean_opt(self, cameras):
        """Initialise cleaner according to the different camera type"""

        self.cleaners = {}  # Function for cleaning for corresp. camera
        self.clean_opts = {}  # Options for cleaners for corresp. camera
        opt = self.config[self.mode]

        if self.mode in "tail":

            # Initialise cleaner per camera type
            for type in opt["thresholds"]:  # cycle through all the cameras...
                cam_id = list(type.keys())[0]
                if cam_id in cameras:
                    # ...but initialize the cleaner only for the once in use
                    cuts = list(type.values())[0]

                    self.clean_opts[cam_id] = {
                        "picture_thresh": cuts[0],
                        "boundary_thresh": cuts[1],
                        "keep_isolated_pixels": opt["keep_isolated_pixels"],
                        "min_number_picture_neighbors": opt[
                            "min_number_picture_neighbors"
                        ],
                    }

                    # Select the CTA-MARS 1st pass as cleaning algorithm
                    self.cleaners[
                        cam_id
                    ] = lambda img, geom, opt: mars_cleaning_1st_pass(
                        image=img, geom=geom, **opt
                    )

        elif self.mode in "wave":

            # Initialise cleaner per camera type
            for cam_id in opt["options"]:
                opt_type = opt["options"][cam_id]

                # WARNING WARNING WARNING
                # camera models for noise injection
                # Need to be updated if calib/signal integration is changed!!!!
                self.noise_model = {
                    "ASTRICam": EmpiricalDistribution(cdf.ASTRI_CDF_FILE),
                    "DigiCam": EmpiricalDistribution(cdf.DIGICAM_CDF_FILE),
                    "FlashCam": EmpiricalDistribution(cdf.FLASHCAM_CDF_FILE),
                    "NectarCam": EmpiricalDistribution(cdf.NECTARCAM_CDF_FILE),
                    "LSTCam": EmpiricalDistribution(cdf.LSTCAM_CDF_FILE),
                    "CHEC": EmpiricalDistribution(cdf.DIGICAM_CDF_FILE),
                }

                self.clean_opts[cam_id] = {
                    "type_of_filtering": opt_type["type_of_filtering"],
                    "filter_thresholds": opt_type["filter_thresholds"],
                    "last_scale_treatment": opt_type["last_scale_treatment"],
                    "kill_isolated_pixels": opt_type["kill_isolated_pixels"],
                    "detect_only_positive_structures": opt_type[
                        "detect_only_positive_structures"
                    ],
                    "tmp_files_directory": opt["tmp_files_directory"],
                    "clusters_threshold": opt_type["clusters_threshold"],
                    "noise_distribution": self.noise_model[cam_id],
                }

                self.cleaners[cam_id] = lambda img, opt: WaveletTransform().clean_image(
                    input_image=img, **opt
                )

    def clean_image(self, img, geom):
        """Clean image according to configuration

        Parameters
        ----------
        img: array
            Calibrated image
        geom: `~ctapipe.XXX`
            Camera geometry

        Returns
        -------
        new_img: `np.array`
            Cleaned image
        mask: `np.array`
            Boolean array which corresponds to the cleaned pixels
        """

        new_img = np.copy(img)
        cam_id = geom.camera_name

        if self.mode in "tail":
            mask = self.cleaners[cam_id](new_img, geom, self.clean_opts[cam_id])
            new_img[~mask] = 0
            # this is still the full camera, but the non-surviving pixels are
            # set to 0: this is not a problem for the number_of_islands
            # ctapipe function used later

            return new_img, mask

        if self.mode in "wave":
            image_2d = geometry_converter.image_1d_to_2d(new_img, cam_id)
            image_2d = self.cleaners[cam_id](image_2d, self.clean_opts[cam_id])
            new_img = geometry_converter.image_2d_to_1d(image_2d, cam_id)

        return new_img, mask
