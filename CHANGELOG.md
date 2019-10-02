<ins>Author:</ins> Dr. Michele Peresano  
<ins>Affiliation:</ins> CEA-Saclay / Irfu

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
