<ins>Author:</ins> Dr. Michele Peresano  
<ins>Affiliation:</ins> CEA-Saclay / Irfu / DAp

#### Contributors in alphabetical order

- Alice Donini
- Karl Kosack
- Thierry Stolarczyk
- Michele Peresano
- Thomas Vuillaume

# Releases

## [0.2](https://github.com/cta-observatory/protopipe/compare/v0.1.1...0.2) (Oct 23rd, 2019)

_protopipe_ 0.2 now fully supports the stable release of _ctapipe_ 0.7.0. The main improvements involve the calibration process (high gain selected by default), the direction reconstruction and new camera-type labels.

Code based on _pywi_/_pywi-cta_ libraries, relevant for wavelet-based image cleaning, has been removed in favor of _ctapipe_ or made completely optional where needed. Wavelet cleaning is still optional but will need those two libraries to be additionally installed. Tailcut-based cleaning is now faster.

The README has been improved with installation, basic use, and developer instructions. Dependencies are listed in `protopipe_environment.yaml` and have been simplified.

The auxiliary scripts `merge_tables.py` and `merge.sh` have been added to allow merging of DL1 and DL2 HDF5 tables.

The `mars_cleaning_1st_pass` method is now imported from _ctapipe_. Novel code using the largest cluster of survived pixels (`number_of_islands` and `largest_island` methods in the `event_preparer` module) has been hardcoded in _protopipe_ and will disappear with the next release of _ctapipe_.

Model estimators now load the camera types directly from the `analysis .yaml` configuration file.

### Pull Requests:

This is an incomplete list: only notable entries are listed.

- [[#13](https://github.com/cta-observatory/protopipe/pull/13)] Bugfix in `save_fig` method used by `model_diagnostic.py`
- [[#10](https://github.com/cta-observatory/protopipe/pull/10)] Update installation process
- [[#9](https://github.com/cta-observatory/protopipe/pull/9)] Update image cleaning and make wavelet-based algorithms independent
- [[#8](https://github.com/cta-observatory/protopipe/pull/8)] Import CTA-MARS 1st pass cleaning from ctapipe

## [0.1.1](https://github.com/cta-observatory/protopipe/compare/v0.1...0.1.1) (Oct 1st, 2019)

The `write_dl1` and `write_dl2` tools can now save an additional file through the flag `--save-images` when applied to a single run. This file will contain the original and calibrated (after gain selection) photoelectron images per event.

A new method `save_fig` has been introduced in the `utils` module, so that `model_diagnostic` can save images also in PNG format.

Additional docstrings and PEP8 formatting have been added throughout the code.

## 0.1 (Sep 23rd, 2019)

This version of protopipe is based on ctapipe 0.6.2 (conda package stable version).
Its performance has been [shown](https://indico.cta-observatory.org/event/1995/contributions/19991/attachments/15559/19825/CTAC_Lugano_2019_Peresano.pdf) at the CTAC meeting in Lugano 2019.
