.. _changelog:

.. _@HealthyPear: https://github.com/HealthyPear
.. _@gaia-verna: https://github.com/gaia-verna
.. _@kosack: https://github.com/kosack
.. _@tstolarczyk: https://github.com/tstolarczyk
.. _@vuillaut: https://github.com/vuillaut
.. _@adonini: https://github.com/adonini

Changelog
=========

.. _protopipe_0p5_release:

**0.5.0** (TBD)
---------------

. . .

.. _protopipe_0p4p4post1_release:

`0.4.0.post1 <https://github.com/cta-observatory/protopipe/releases/tag/v0.4.0.post1>`__ (Mar 5th, 2021)
---------------------------------------------------------------------------------------------------------

Summary
+++++++

This is a post-release that takes care of project maintenance, so it doesn't change the performance of the code.

Contributors
++++++++++++

- Michele Peresano (`@HealthyPear`_)

Changes from previous release
+++++++++++++++++++++++++++++

Pull-requests that contain changes belonging to multiple classes are repeated.

🐛 Bug Fixes
^^^^^^^^^^^^

- Fix zenodo configuration file and add LICENSE file (:pr:`106`) `@HealthyPear`_

🧰 Maintenance
^^^^^^^^^^^^^^

- Update CHANGELOG (:pr:`108`) `@HealthyPear`_
- Fix zenodo configuration file and add LICENSE file (:pr:`106`) `@HealthyPear`_
- Prepare first upload to PyPI (:pr:`107`) `@HealthyPear`_

.. _protopipe_0p4_release:

`0.4.0 <https://github.com/cta-observatory/protopipe/releases/tag/v0.4.0>`__ (Feb 22th, 2021)
---------------------------------------------------------------------------------------------

Summary
+++++++

This release brings many improvements of which the most relevant are summarised
here depending on their scope within the pipeline workflow.

Performance-wise, ``protopipe`` caught up with the ``EventDisplay`` and ``CTAMARS``
historical pipelines starting from about 500 GeV onwards.
Below this threshold, even if compatible with requirements, the sensitivity
diverges. The cause seems to be a low-energy effect delimited to
the steps before model training.

- All pipeline

  - upgrade to the API of ``ctapipe 0.9.1``
  - documentation also on ``readthedocs`` and link to ``Zenodo``
  - Continuous Integration is now performed on ``GitHub``
  - New benchmarks have been added
  - Reference analysis and benchmarks results have been updated

- Data training

  - calibration benchmarks need only ``ctapipe-stage1-process`` 
  - ``write_dl1`` has become ``data_training``
  - DL1 parameters and (optionally) images are merged in a single file
  - DL1 parameters names as in ``ctapipe`` and they are in degrees (``TelescopeFrame``)
  - scale correction with the effective focal length
  - fixed bugs and wrong behaviors

- Modeling and DL2 production

  - fixed bugs and wrong behaviors
  - Added missing features to get closer to ``CTAMARS``

- DL3

  - the performance step is now based on the `pyirf <https://cta-observatory.github.io/pyirf/>`_ library
  - performance results are stored `here <https://forge.in2p3.fr/projects/benchmarks-reference-analysis/wiki/Protopipe_performance_data>`_

Contributors
++++++++++++

- Michele Peresano (`@HealthyPear`_)
- Gaia Verna (`@gaia-verna`_)
- Alice Donini (`@adonini`_)

Changes from previous release
+++++++++++++++++++++++++++++

Pull-requests that contain changes belonging to multiple classes are repeated.

🚀 General features
^^^^^^^^^^^^^^^^^^^

- Performance using Pyirf (:pr:`83`) `@gaia-verna`_ & `@adonini`_
- Towards using Pyirf (:pr:`79`) `@gaia-verna`_ & `@adonini`_
- Upgrade of DL2 production (:pr:`77`) `@HealthyPear`_
- Upgrade calibration benchmarks (:pr:`59`) `@HealthyPear`_
- Upgrade of data training (:pr:`58`) `@HealthyPear`_

🐛 Bug Fixes
^^^^^^^^^^^^

- Fix calibration benchmarking settings (:pr:`100`) `@HealthyPear`_
- Fix plot of simulated signal and noise of 2nd pass image extraction (:pr:`99`) `@HealthyPear`_
- Upgrade of DL2 production (:pr:`77`) `@HealthyPear`_
- Upgrade of data training (:pr:`58`) `@HealthyPear`_

🧰 Maintenance
^^^^^^^^^^^^^^

- Fix zenodo configuration file and add LICENSE file (:pr:`106`) `@HealthyPear`_
- Update documentation + general maintenance (:pr:`62`) `@HealthyPear`_
- Use mamba to create virtual enviroment for the CI (:pr:`101`) `@HealthyPear`_
- Upgrade all other notebooks and their docs version (:pr:`76`) `@HealthyPear`_
- Upgrade calibration benchmarks (:pr:`59`) `@HealthyPear`_
- Upgrade of data training (:pr:`58`) `@HealthyPear`_
- Enable CI from GitHub actions (:pr:`84`) `@HealthyPear`_



.. _protopipe_0p3_release:

`0.3.0 <https://github.com/cta-observatory/protopipe/releases/tag/v0.3.0>`__ (Nov 9th, 2020)
--------------------------------------------------------------------------------------------

Summary
+++++++

- early improvements related to the DL1 comparison against the CTAMARS pipeline
- improvements to basic maintenance
- a more consistent approach for full-scale analyses
- bug fixes

Contributors
++++++++++++

- Michele Peresano (`@HealthyPear`_)
- Thierry Stolarczyk (`@tstolarczyk`_)
- Gaia Verna (`@gaia-verna`_)
- Karl Kosack (`@kosack`_)
- Thomas Vuillaume (`@vuillaut`_)

Changes from previous release
+++++++++++++++++++++++++++++

🚀 General features
^^^^^^^^^^^^^^^^^^^

- Add missing variables in write\_dl2 (:pr:`66`) `@HealthyPear`_
- Add missing dl1 parameters (:pr:`41`) `@HealthyPear`_
- Updates on notebooks (:pr:`47`) `@HealthyPear`_
- New plots for calibration benchmarking (:pr:`43`) `@HealthyPear`_
- Double-pass image extractor (:pr:`48`) `@HealthyPear`_
- Notebooks for low-level benchmarking (:pr:`42`) `@HealthyPear`_
- Improved handling of sites, arrays and cameras for all Prod3b simtel productions (:pr:`33`) `@HealthyPear`_
- Change gain selection (:pr:`35`) `@HealthyPear`_
- Changes for adding Cameras beyond LSTCam and NectarCam  (:pr:`29`) `@tstolarczyk`_

🌐 GRID support
^^^^^^^^^^^^^^^

- Update configuration files (:pr:`74`) `@HealthyPear`_
- Update documentation for GRID support (:pr:`54`) `@HealthyPear`_
- Rollback for GRID support (:pr:`52`) `@HealthyPear`_

🐛 Bug Fixes
^^^^^^^^^^^^  

- Bugfix in Release Drafter workflow file (:pr:`71`) `@HealthyPear`_
- Convert pointing values to float64 at reading time (:pr:`68`) `@HealthyPear`_
- Rollback for GRID support (:pr:`52`) `@HealthyPear`_
- Fix recording of DL1 image and record reconstruction cleaning mask (:pr:`46`) `@gaia-verna`_
- consistent definition of angular separation to the source with config (:pr:`39`) `@vuillaut`_
- Update write\_dl1.py (:pr:`30`) `@tstolarczyk`_

🧰 Maintenance
^^^^^^^^^^^^^^

- Update benchmarks and documentation (:pr:`75`) `@HealthyPear`_
- Bugfix in Release Drafter workflow file (:pr:`71`) `@HealthyPear`_
- Add release drafter (:pr:`67`) `@HealthyPear`_
- Add benchmark notebooks for medium and late stages (:pr:`55`) `@HealthyPear`_
- Update documentation for GRID support (:pr:`54`) `@HealthyPear`_
- Updated documentation (:pr:`50`) `@HealthyPear`_
- Implementation of a first unit test (DL1) (:pr:`34`) `@HealthyPear`_
- Updated documentation (Closes #23) (:pr:`32`) `@HealthyPear`_
- Added Travis CI configuration file (:pr:`18`) `@HealthyPear`_
- Update README.md (:pr:`28`) `@tstolarczyk`_
- Added versioning to init.py and setup.py using the manual approach. (:pr:`20`) `@HealthyPear`_
- Update README.md (:pr:`21`) `@tstolarczyk`_


.. _gammapy_0p2p1_release:

`0.2.1 <https://github.com/cta-observatory/protopipe/releases/tag/v0.2.1>`__ (Oct 28th, 2019)
---------------------------------------------------------------------------------------------

Summary
+++++++

- Released Oct 28, 2019
- 1 contributor
- 1 pull requests

**Description**

The ctapipe-based cleaning algorithm for the biggest cluster was crashing in
case of cleaned images with no surviving pixel clusters.

**Contributors:**

In alphabetical order by first name:

- Michele Peresano

Pull Requests
+++++++++++++

- (:pr:`16`) Bugfix: Closes #15 (Michele Peresano)

`0.2.0 <https://github.com/cta-observatory/protopipe/releases/tag/v0.2.0>`__ (Oct 24th, 2019)
---------------------------------------------------------------------------------------------

Summary
+++++++

- Released Oct 24, 2019
- 3 contributor(s)
- 7 pull requests

**Description**

*protopipe* 0.2 now fully supports the stable release of *ctapipe* 0.7.0.

The main improvements involve the calibration process
(high gain selected by default),
the direction reconstruction and new camera-type labels.

Code based on *pywi*/*pywi-cta* libraries, relevant for wavelet-based image
cleaning, has been removed in favor of *ctapipe* or made completely optional
where needed. Wavelet cleaning is still optional but will need those two
libraries to be additionally installed. Tailcut-based cleaning is now faster.

The README has been improved with installation, basic use, and developer instructions.
Dependencies are listed in ``protopipe_environment.yaml`` and have been simplified.

The auxiliary scripts ``merge_tables.py`` and ``merge.sh`` have been added to allow merging of DL1 and DL2 HDF5 tables.

The ``mars_cleaning_1st_pass`` method is now imported from _ctapipe_.
Novel code using the largest cluster of survived pixels
(``number_of_islands`` and ``largest_island`` methods in the
``event_preparer`` module) has been hardcoded in _protopipe_ and will
disappear with the next release of _ctapipe_.

Model estimators now load the camera types directly from the ``analysis .yaml`` configuration file.

**Contributors:**

In alphabetical order by first name:

- Alice Donini
- Michele Peresano
- Thierry Stolarczyk

Pull Requests
+++++++++++++

This list is incomplete. Small improvements and bug fixes are not listed here.

The complete list is found `here <https://github.com/gammapy/gammapy/pulls?q=is%3Apr+milestone%3A0.16+is%3Aclosed>`__.

- (:pr:`9`) Update image cleaning and make wavelet-based algorithms independent
- (:pr:`8`) Import CTA-MARS 1st pass cleaning from ctapipe

`0.1.1 <https://github.com/cta-observatory/protopipe/releases/tag/v0.1.1>`__ (Oct 1st, 2019)
--------------------------------------------------------------------------------------------

Summary
+++++++

- Released Oct 1, 2019
- X contributor(s)
- X pull request(s)

**Description**

The ``write_dl1`` and ``write_dl2`` tools can now save an additional file
through the flag ``--save-images`` when applied to a single run.
This file will contain the original and calibrated (after gain selection)
photoelectron images per event.
A new method ``save_fig`` has been introduced in the ``utils`` module,
so that ``model_diagnostic`` can save images also in PNG format.
Additional docstrings and PEP8 formatting have been added throughout the code.

**Contributors:**

In alphabetical order by first name:

- ...

Pull Requests
+++++++++++++

The development of *protopipe* on GitHub started out directly in the master branch,
so there are no pull request we can list here.

`0.1.0 <https://github.com/cta-observatory/protopipe/releases/tag/v0.1.0>`__ (Sep 23th, 2019)
---------------------------------------------------------------------------------------------

Summary
+++++++

- Released Sep 23, 2019
- 6 contributor(s)
- 1 pull request(s)

**Description**

First version of *protopipe* to be publicly release on GitHub.
This version is based on ctapipe 0.6.2 (conda package stable version).
Its performance has been shown in a
`presentation <https://indico.cta-observatory.org/event/1995/contributions/19991/attachments/15559/19825/CTAC_Lugano_2019_Peresano.pdf>`__
at the CTAC meeting in Lugano 2019.

**Contributors:**

In alphabetical order by first name:

- David Landriu
- Julien Lefacheur
- Karl Kosack
- Michele Peresano
- Thomas Vuillaume
- Tino Michael

Pull Requests
+++++++++++++

- (:pr:`2`) Custom arrays, example configs and aux scripts (M.Peresano)
