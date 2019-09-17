<ins>Author:</ins> Dr. Michele Peresano  
<ins>Affiliation:</ins> CEA-Saclay / Irfu

This is a local, development document.
It is not meant to be shown in the final release, but rather a tool for any future developer of protopipe.
All changes are in reverse chronological order.

<ins>Changes made from protopipe 0.1 to 0.2:</ins>

* 2019-08-28 : Updated to new predict method
  - in the newer ctapipe's version 'predict' works with a dictionary of astropipe's SkyCoord(inates)
* 2019-08-28 : Get pmt_signal using ctapipe and not pywi-cta
* 2019-08-28 : Updated teltype handling for telescope counting in eventpreparer
* 2019-08-27 : Updated LST and MST keys in event_preparer.py
* 2019-08-27 : Updated LST and MST keys in write_dl1.py
* 2019-08-26 : Fixed and simplified calibration; few formatting changes.
 - CameraCalibrator moved to \__init\__ function
 - the instance call and the name of the integration method had to changed according to new CameraCalibrator in ctapipe

<ins>protopipe 0.1.1:</ins>

13/09/19

* pipeline/utils.py
  - added function "save_fig" to save figures in multiple formats
  - added docstring to "load_config"
* scripts/model_diagnostic.py
  - added import of new pipeline/utils.py function "save_fig" and applied it
  - PEP8 formatting via "black" package
