<ins>Author:</ins> Dr. Michele Peresano  
<ins>Affiliation:</ins> CEA-Saclay / Irfu

# What's new

## October 4th, 2019

### General changes

 - Fixed environment file containing specific macos-dependent package builds, which were preventing a smooth installation on a Linux machine,
 - Updated README.

## October 2nd, 2019

### Disclaimer(s)

 - For the moment, the master branch host code almost completely compliant with ctapipe 0.7.
 - It is currently been tested
   - Expect potentially frequent changes.
   - Use it and fell free to comment / communicate any strange or wrong behaviour
 - Expect a more organized approach to the development with reviewed pull-requests.

### General changes

- Almost complete support for ctapipe 0.7
  - 1D to 2D conversion of DL1 images is still done using pywi-cta
- PEP8 compliant formatting through the whole code
- Removed some stale commented code
- README.rst has been updated "Requirements" and a new "Basic use" instruction section
- Added auxiliary scripts which allow to merge DL1 and DL2 HDF5 tables

### Specific changes

* scripts/write_dl1.py
  - now only one DL1 calibrated image is (optionally) saved, since ctapipe 0.7 choses the gain autonomously
  - Updated LST and MST camera type keys
* scripts/write_dl2.py
  - as in write_dl1.py, only one DL1 calibrated image is (optionally) saved
  - Estimator files now loaded using camera types read directly from analysis configuration file
* pipeline/event_preparer.py
  - updated calibration process with support for ctapipe 0.7
  - updated direction reconstruction with support for ctapipe 0.7
  - dumped the use of pywi-cta in obtaining the calibrated image in favor of ctapipe
  - Updated LST and MST camera type keys

# Previous releases

<ins>protopipe 0.1.1:</ins>

* scripts/write_dl1.py
  - added optional saving of images of a run in separate file for test purposes
* scripts/write_dl2.py
  - added optional saving of images of a run in separate file for test purposes
* pipeline/event_preparer.py
  - added support for the optional storage of DL1 images
* scripts/model_diagnostic.py
  - added import of new pipeline/utils.py function "save_fig" and applied it
  - PEP8 formatting via "black" package
* pipeline/utils.py
  - added function "save_fig" to save figures in multiple formats
  - added docstring to "load_config"
