.. _pipeline:

pipeline
========

Introduction
------------

`protopipe.pipeline` contains classes that are used in scripts
producing tables with images information (DL1), typically for g/h classifier and
energy regressor, and tables with event information, typically used for
performance estimation (DL2).

Two classes from the sub-module are used to process the events. The EventPreparer class, which
goal is to loop on events and to provide event parameters (e.g. impact parameter) and
image parameters (e.g. Hillas parameters). The second class, ImageCleaner,
is dedicated to clean the images according to different options
(tail cut and/or wavelet).

Details
-------

.. warning::

  The ctapipe version used in protopipe is always the last stable version
  packaged on the Anaconda framework.
  This means that some of the code needed to be hard-coded in order to
  implement newer features. This code will disappear at each newer release of
  ctapipe.

The following is a description of the *default* algorithms and settings.

Calibration
^^^^^^^^^^^

The current calibration is performed using:

* automatic high gain selection at threshold of 4000 counts in low gain at RO level,
* charge and pulse times extraction via ``ctapipe.image.extractors.TwoPassWindowSum``
* no integration correction

The resulting **optimized cleaning thresholds** for LSTCam and NectarCam
when requiring 99.7% rejection of the "noise" (0 true phes) are
(4.2, 2.1) for LSTCam and (4., 2.) for NectarCam.

.. note::

  These phe units are not corrected for the average bias but are the ones
  effectively used through the pipeline at the moment (also CTA-MARS).
  For details on how these values are obtained, plese refer to the calibration
  benchmarks (:ref:`beforepushing`)

Imaging
^^^^^^^

**Cleaning** is performed using ``ctapipe.image.cleaning.mars_cleaning_1st_pass``,
but the settings are user-dependent.

.. note::
  To replicate CTA-MARS behaviour it is sufficient to fix a minimum number of 1
  core neighbour while using the cleaning thresholds shown above.

**Selection** is performed by the following requirements:

* at least 50 phe (still biased units),
* image's center of gravity (COG) within 80% of camera radius,
* ellipticity between 0.1 and 0.6,
* at least 3 surviving pixels.

**Parametrization** is performed by ``ctapipe.image.hillas.hillas_parameters``.

Direction reconstruction
^^^^^^^^^^^^^^^^^^^^^^^^

Performed via ``ctapipe.reco.HillasReconstructor`` with a minimum number of 2 surviving images
per event.

The camera corrections correspond to those of *ctapipe* and are performed inside
the reconstructor.

Proposals for improvements and/or fixes
---------------------------------------

.. note::

  This section will be moved to the repository as a issues.
  Any further update will appear there.


* The EventPreparer class is a bit messy: it should return the event and one container
  with several results (hillas parameters, reconstructed shower, etc.). In addition
  some things are hard-coded , e.g. for now calibration is done in the same way
  (not a problem since only LSTCam and NectarCam have been considered until now),
  camera radius is also hard-coded for LST and MST, and computation of the impact
  parameters in the frame of the shower system should be better implemented.

Reference/API
-------------

.. automodapi:: protopipe.pipeline
    :no-inheritance-diagram:
    :include-all-objects:
    :skip: event_source
