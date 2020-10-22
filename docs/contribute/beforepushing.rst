.. _beforepushing:

Code and performance checks
===========================

Tests and unit-tests
--------------------

Being *protopipe* based on *ctapipe*, all the tools imported from the latter
have been already tested and approved (remember that *protopipe* uses always the
latest version of *ctapipe* which has been released in the Anaconda framework).

.. warning::
  This is not true for,

  - hard-coded parts that had to be modified from ctapipe,
  - protopipe functions themselves (which will eventually migrate to ctapipe)

  Regarding the first point: given the difference in versions between the
  imported ctapipe and its development version, sometimes it's possible that, in
  order to code a new feature, this has to be pull-requested to ctapipe and at
  the same time hardcoded in protopipe, until the new version of ctapipe is released.

For the moment there is only one test, that is more than a unit-test.

This test is in charge to detect if changes in
`protopipe.pipeline.event_preparer` or `protopipe.scripts.write_dl1` produce any
fatal behaviour or crash.

It uses the `gamma_test_large` file of ctapipe (a Prod2 CTA South array composed of
LSTCam, FlashCam and ASTRICam with about ~100 simulated showers).
It is expected that an HDF5 file named `test_dl1.h5` is produced and is non-empty.

The test can be executed directly from the main folder of protopipe by launching
`pytest`. It is also automatically triggered by the continuous integration
everytime a new pull-request is pushed to the repository, and its correct
execution is a mandatory condition for merging.

.. warning::
  For the moment, in this test the shower images are not saved, so the
  correct production of the `images.h5` file is not automatically tested.

Benchmarks
----------

.. toctree::
   :hidden:

   benchmarks/benchmarks_DL1_calibration.ipynb

*protopipe* contains a folder named *benchmarks* (not a module)
hosting the notebooks used for benchmarking.

Their contents follow the development triggered by the ongoing
comparison between protopipe and CTA-MARS (see
`this issue <https://github.com/cta-observatory/protopipe/issues/24>`__ and
references therein for a summary).

For this reason, the
`MC sample <https://forge.in2p3.fr/attachments/download/63177/CTA-N_from_South.zip>`)
to be used for these benchmarks **needs** to be the same.

The benchmarks are organised in 3 folders,

- DL1

  * `calibration <benchmarks/benchmarks_DL1_calibration.ipynb>`__ | *benchmarks_DL1_calibration.ipynb*
  * `image cleaning <benchmarks/benchmarks_DL1_image-cleaning.ipynb>`__ | *benchmarks_DL1_image-cleaning.ipynb*

- DL2

  * `direction reconstruction <benchmarks/benchmarks_DL2_direction-reconstruction.ipynb>`__ | *benchmarks_DL2_direction-reconstruction.ipynb*
  * `energy estimation <benchmarks/benchmarks_DL2_energy-estimation.ipynb>`__ | *benchmarks_DL2_energy-estimation.ipynb*
  * `particle classification <benchmarks/benchmarks_DL2_particle_classification.ipynb>`__ | *benchmarks_DL2_particle_classification.ipynb*

- DL3

  * `Point Spread Function <benchmarks/benchmarks_DL3_PSF.ipynb>`__ | *benchmarks_DL3_PSF.ipynb*
  * `Instrument Response Functions and sensitivity <benchmarks/benchmarks_DL3_IRFs_and_sensitivity.ipynb>`__ | *benchmarks_DL3_IRFs_and_sensitivity.ipynb*

The DL3 folder contains the associate reference data:

- the CTA requirements,
- the ASWG performance (to be updated).

.. note::
   This part of protopipe is not meant to be kept here in the end, in order to
   avoid divergences with
   `ctaplot <https://github.com/cta-observatory/ctaplot>`__ and
   `cta-benchmarks <https://github.com/cta-observatory/cta-benchmarks>`__.
   In particular, the contents of the DL3 folder would be likely the same, so
   probably we will not do that here at all.

   At the time of this editing (early March 2020), the only limitation to this
   approach are the missing new DL1/DL2 data formats from *ctapipe*, to be
   implemented soon (see this
   `issue <https://github.com/cta-observatory/protopipe/issues/25>`__
   for news about this).

.. note::
   The storage of static versions of the benchmarks in this documentation is temporary.
   It it is planned to run such benchmarks in the
   future via an external resource which will deal automatically with the
   pipeline code, the reference files and related output.

   For the moment the purpose of use these tools in the current setup is to help
   early developers and testers to check if their changes improve or degrade
   previous performances.

   Please, find `here <https://indico.cta-observatory.org/event/1995/contributions/19991/attachments/15559/19825/CTAC_Lugano_2019_Peresano.pdf>`__
   the **latest complete performance**.
