.. _pipeline:

pipeline
========

Introduction
------------

``protopipe.pipeline`` contains classes that are used in scripts to produce

- tables with images information (DL1), typically for g/h classifier and energy regressor,
- tables with event information, typically used for performance estimation (DL2).

Two classes from the sub-module are used to process the events:

- ``EventPreparer``, which loops on events and to provide event parameters
  (e.g. impact parameter) and image parameters (e.g. Hillas parameters),
- ``ImageCleaner``, cleans the images according to different options.

Details
-------

.. warning::

  The version of *ctapipe* used by *protopipe* is always the last stable version
  packaged on the Anaconda framework.
  This means that some of the more cutting-edge code needs to be hard-coded.
  This code should always be stored in ``protopipe.pipeline.temp`` and
  disappear at each newer release of *ctapipe*.

The following is a description of the *default* algorithms and settings, chosen
to mimic the CTA-MARS pipeline.

Management of events, triggers and images
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- for each event ALL images from ALL telescopes are processed up to DL1b included,

- to define what happens *after* DL1b, the ``Reconstruction`` section of the ``analysis.yaml`` configuration
  file accepts two options,

  - ``LST_stereo``, trigger requirement (at least 2 LST images - ``True`` by default)
  - ``min_tel``, minimum number of telescopes from ANY telescope type to consider events,

In case ``LST_stereo`` is set to ``True``, events with,

- 1 LST trigger and less than 2 triggers from any other telescope type will not
  be processed after DL1b
- 1 LST trigger and at least 2 triggers from any other telescope type will get processed
  at DL2a **without** the single-LST image and provided the remaining
  images are still at least 2 after the image quality cuts have been taken into
  consideration.

Calibration
^^^^^^^^^^^

The current calibration is performed using:

* automatic gain channel selection (when more than one) above 4000 ADC counts at RO level,
* charge and pulse times extraction via ``ctapipe.image.extractors.TwoPassWindowSum``
* correction for the integration window.

.. figure:: ./double-pass-image-extraction.png
  :width: 800
  :alt: Explanation of ``ctapipe.image.extractors.TwoPassWindowSum``

  Explanation of ``ctapipe.image.extractors.TwoPassWindowSum``.

.. note::

  The photoelectron units used later for cleaning the images are those **not**
  corrected for the average bias. Said this, thanks to the integration correction
  this effect is now negligible.
  For details on how these values are obtained, please refer to the calibration
  benchmarks (:ref:`beforepushing`).

Imaging
^^^^^^^

**Cleaning** is performed using ``ctapipe.image.cleaning.mars_cleaning_1st_pass``,
but the settings are user-dependent.

.. note::
  To replicate CTA-MARS behaviour it is sufficient to fix a minimum number of 1
  core neighbour while using the cleaning thresholds shown above.

**Selection** is performed by the following requirements:

* at least 50 phe (still biased units),
* image's center of gravity (COG) within 80% of camera radius (radii stored in ``protopipe.pipeline.utils``),
* ellipticity between 0.1 and 0.6,
* at least 3 surviving pixels.

**Parametrization** is performed by ``ctapipe.image.hillas.hillas_parameters``
in the ``ctapipe.coordinates.TelescopeFrame`` using the effective focal lenghts
values stored in ``protopipe.pipeline.utils``.

Direction reconstruction
^^^^^^^^^^^^^^^^^^^^^^^^

Performed via ``ctapipe.reco.HillasReconstructor`` with a minimum number of 2 surviving images
per event.

The camera corrections correspond to those of *ctapipe* and are performed inside
the reconstructor.

Reference/API
-------------

.. automodapi:: protopipe.pipeline
    :no-inheritance-diagram:
    :include-all-objects:
    :skip: Path, Table, EventSource
