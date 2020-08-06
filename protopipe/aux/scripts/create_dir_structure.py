"""Ctapipe/protopipe analysis directory structure.

Author: Dr. Michele Peresano
Affilitation: CEA-Saclay/Irfu

"""

import os
import sys

# FUNCTIONS


def makedir(name):
    """Create folder if non-existent and output OS error if any."""
    if not os.path.exists(name):
        try:
            os.mkdir(name)
        except OSError:
            print("Creation of the directory {} failed".format(name))
        else:
            print("Successfully created the directory {}".format(name))
    return None


# MAIN

# Input can be included in a better way later
# along the rest of the protopipe configuration

# read name of working directory as 1st argument
# create it if not existent
wd = sys.argv[1]
makedir(wd)

# read name of the analysis as 2nd argument
analysisName = sys.argv[2]
# Create analysis parent folder
analysis = os.path.join(wd, analysisName)
makedir(analysis)

subdirectories = {
    "configs": [],
    "data": ["DL0", "DL1", "DL2", "DL3"],
    "estimators": ["energy_regressor", "gamma_hadron_classifier"],
    "performance": [],  # here no subdirectories, make_performance.py will do it
}

for d in subdirectories:
    subdir = os.path.join(analysis, d)
    makedir(subdir)
    for dd in subdirectories[d]:
        subsubdir = os.path.join(subdir, dd)
        makedir(subsubdir)
        if dd == "DL1":
            makedir(os.path.join(subsubdir, "for_classification"))
            makedir(os.path.join(subsubdir, "for_energy_estimation"))

print("Directory structure ready for protopipe analysis on DIRAC.")
