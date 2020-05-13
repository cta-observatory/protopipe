.. X.Y.Z (Mon DD, YYYY)
.. -------------------
..
.. Summary
.. +++++++
..
.. - Released Mon DD, YYYY
.. - X contributor(s)
.. - X pull requests
..
.. **Description**
..
.. bla bla.
..
.. **Contributors:**
..
.. In alphabetical order by first name:
..
.. - ...
..
.. Pull Requests
.. +++++++++++++
..
.. - [#X] Title (Author)

.. _protopipe_0p3_release:

0.3 (Unreleased)
-----------------

Planned for early summer 2020.

.. _gammapy_0p2p1_release:

`0.2.1 <https://github.com/cta-observatory/protopipe/releases/tag/v0.2.1>`__ (Oct 28, 2019)
-------------------------------------------------------------------------------------------

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

- [#16] Bugfix: Closes #15 (Michele Peresano)

`0.2.0 <https://github.com/cta-observatory/protopipe/releases/tag/v0.2.0>`__ (Oct 24, 2019)
-------------------------------------------------------------------------------------------

Summary
+++++++

- Released Oct 24, 2019
- 3 contributor(s)
- 7 pull requests

**Description**

*protopipe* 0.2 now fully supports the stable release of *ctapipe* 0.7.0. The main improvements involve the calibration process (high gain selected by default), the direction reconstruction and new camera-type labels.

Code based on *pywi*/*pywi-cta* libraries, relevant for wavelet-based image cleaning, has been removed in favor of *ctapipe* or made completely optional where needed. Wavelet cleaning is still optional but will need those two libraries to be additionally installed. Tailcut-based cleaning is now faster.

The README has been improved with installation, basic use, and developer instructions. Dependencies are listed in `protopipe_environment.yaml` and have been simplified.

The auxiliary scripts `merge_tables.py` and `merge.sh` have been added to allow merging of DL1 and DL2 HDF5 tables.

The `mars_cleaning_1st_pass` method is now imported from _ctapipe_. Novel code using the largest cluster of survived pixels (`number_of_islands` and `largest_island` methods in the `event_preparer` module) has been hardcoded in _protopipe_ and will disappear with the next release of _ctapipe_.

Model estimators now load the camera types directly from the `analysis .yaml` configuration file.

**Contributors:**

In alphabetical order by first name:

- Alice Donini
- Michele Peresano
- Thierry Stolarczyk

Pull Requests
+++++++++++++

This list is incomplete. Small improvements and bug fixes are not listed here.
The complete list is found `here <https://github.com/gammapy/gammapy/pulls?q=is%3Apr+milestone%3A0.16+is%3Aclosed>`__.

- [#9] Update image cleaning and make wavelet-based algorithms independent
- [#8] Import CTA-MARS 1st pass cleaning from ctapipe

`0.1.1 <https://github.com/cta-observatory/protopipe/releases/tag/v0.1.1>`__ (Oct 1, 2019)
------------------------------------------------------------------------------------------

Summary
+++++++

- Released Oct 1, 2019
- X contributor(s)
- X pull request(s)

**Description**

The `write_dl1` and `write_dl2` tools can now save an additional file through the flag `--save-images` when applied to a single run. This file will contain the original and calibrated (after gain selection) photoelectron images per event.

A new method `save_fig` has been introduced in the `utils` module, so that `model_diagnostic` can save images also in PNG format.

Additional docstrings and PEP8 formatting have been added throughout the code.

**Contributors:**

In alphabetical order by first name:

- ...

Pull Requests
+++++++++++++

The development of *protopipe* on GitHub started out directly in the master branch,
so there are no pull request we can list here.

`0.1.0 <https://github.com/cta-observatory/protopipe/releases/tag/v0.1.0>`__ (Sep 23, 2019)
-------------------------------------------------------------------------------------------

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

- [#2] Custom arrays, example configs and aux scripts (M.Peresano)
