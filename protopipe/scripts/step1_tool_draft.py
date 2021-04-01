#!/usr/bin/env python
"""Process simtel data for the training of an estimator model.

This pipeline tool can produce data files in HDF5 file format with the
following information:

- simulated information (if available)
- DL1a (optional), reconstructed calibrated images and peak times
- DL1b (optional), image parameters and cleaning masks
- DL2a (requires DL1b), shower geometry reconstruction

For documentation and examples, please visit the webpage
https://cta-observatory.github.io/protopipe/scripts/data_training.html"""

import sys

from tqdm.autonotebook import tqdm

from ctapipe.calib.camera import CameraCalibrator, GainSelector
from ctapipe.core import Tool
from ctapipe.core.traits import Bool, List, classes_with_traits
from ctapipe.image import ImageCleaner, ImageProcessor
from ctapipe.image.extractor import ImageExtractor
from ctapipe.io import DataLevel, DL1Writer, EventSource, SimTelEventSource
from ctapipe.io.dl1writer import DL1_DATA_MODEL_VERSION

from protopipe.pipeline.temp import (
    HillasParametersTelescopeFrameContainer,
    MyCameraGeometry,
    MyHillasReconstructor,
)


class Step1(Tool):
    """
    Process data from lower-data levels up to DL1, including both image
    extraction and optinally image parameterization
    """

    name = "ctapipe-processor"
    description = __doc__ + f" This currently writes {DL1_DATA_MODEL_VERSION} DL1 data"
    examples = """
    To process data with all default values:
    > protopipe-STEP1 --input events.simtel.gz --output events.dl1.h5 --progress
    Or use an external configuration file, where you can specify all options:
    > protopipe-STEP1 --config step1.json --progress
    The config file should be in JSON or python format (see traitlets docs). For an
    example, see ctapipe/examples/stage1_config.json in the main code repo.
    """

    progress_bar = Bool(help="show progress bar during processing").tag(config=True)

    aliases = {
        "input": "EventSource.input_url",
        "output": "DL1Writer.output_path",
        "allowed-tels": "EventSource.allowed_tels",
        "max-events": "EventSource.max_events",
        "image-cleaner-type": "ImageProcessor.image_cleaner_type",
    }

    flags = {
        "compute-image-parameters": (
            {"DataWriter": {"write_image_parameters": True}},
            "Compute reconstructed image parameters",
        ),
        "compute-shower-parameters": (
            {"DataWriter": {"write_shower_parameters": True}},
            "Compute reconstructed shower geometry",
        ),
        "write-images": (
            {"DataWriter": {"write_images": True}},
            "store DL1/Event/Telescope images in output",
        ),
        "write-image-parameters": (
            {"DataWriter": {"write_image_parameters": True}},
            "store DL1/Event/Telescope parameters in output",
        ),
        "write-shower-parameters": (
            {"DataWriter": {"write_shower_parameters": True}},
            "store DL2/Event/Telescope parameters in output",
        ),
        "write-index-tables": (
            {"DataWriter": {"write_index_tables": True}},
            "generate PyTables index tables for the parameter and image datasets",
        ),
        "overwrite": (
            {"DataWriter": {"overwrite": True}},
            "Overwrite output file if it exists",
        ),
        "progress": (
            {"Stage1Tool": {"progress_bar": True}},
            "show a progress bar during event processing",
        ),
    }

    classes = (
        [CameraCalibrator, DL1Writer, ImageProcessor, MyHillasReconstructor]
        + classes_with_traits(EventSource)
        + classes_with_traits(ImageCleaner)
        + classes_with_traits(ImageExtractor)
        + classes_with_traits(GainSelector)
    )

    def setup(self):

        # setup components:
        self.event_source = EventSource(parent=self)
        compatible_datalevels = [DataLevel.R1, DataLevel.DL0, DataLevel.DL1_IMAGES]
        if not self.event_source.has_any_datalevel(compatible_datalevels):
            self.log.critical(
                f"{self.name} needs the EventSource to provide "
                f"either R1 or DL0 or DL1A data"
                f", {self.event_source} provides only {self.event_source.datalevels}"
            )
            sys.exit(1)

        self.calibrate = CameraCalibrator(
            parent=self, subarray=self.event_source.subarray
        )
        self.process_images = ImageProcessor(
            subarray=self.event_source.subarray,
            is_simulation=self.event_source.is_simulation,
            parent=self,
        )
        self.write_dl1 = DL1Writer(event_source=self.event_source, parent=self)

        # warn if max_events prevents writing the histograms
        if (
            isinstance(self.event_source, SimTelEventSource)
            and self.event_source.max_events
            and self.event_source.max_events > 0
        ):
            self.log.warning(
                "No Simulated shower distributions will be written because "
                "EventSource.max_events is set to a non-zero number (and therefore "
                "shower distributions read from the input Simulation file are invalid)."
            )

    def _write_processing_statistics(self):
        """ write out the event selection stats, etc. """
        # NOTE: don't remove this, not part of DL1Writer
        image_stats = self.process_images.check_image.to_table(functions=True)
        image_stats.write(
            self.write_dl1.output_path,
            path="/dl1/service/image_statistics",
            append=True,
            serialize_meta=True,
        )

    def start(self):
        self.event_source.subarray.info(printer=self.log.info)
        for event in tqdm(
            self.event_source,
            desc=self.event_source.__class__.__name__,
            total=self.event_source.max_events,
            unit="ev",
            disable=not self.progress_bar,
        ):

            self.log.log(9, "Processessing event_id=%s", event.index.event_id)
            self.calibrate(event)
            if self.write_dl1.write_parameters:
                self.process_images(event)
            self.write_dl1(event)

    def finish(self):
        self.write_dl1.write_simulation_histograms(self.event_source)
        self.write_dl1.finish()
        self._write_processing_statistics()


def main():
    """ run the tool"""
    tool = Step1()
    tool.run()


if __name__ == "__main__":
    main()
