#!/usr/bin/env bash

# Author: Dr. Michele Peresano
# Affiliation: CEA-Saclay/Irfu

# This script allows you to launch multiple
# performance estimations associated to
# multiple choices of observation times.

# Some color codes
RED='\033[0;31m'
BLUE='\033[0;34m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

MODE="tail"

CONFIG_PATH=$1 # argument given to script

# Call 'make_performance.py' for each observation time
# $PROTOPIPE is the folder contaning protopipe
for time in "100 s" "30 min" "5 h" "50 h"; do
  printf "${BLUE}Launching performance estimation for $time...${NC}\n"
  python $PROTOPIPE/protopipe/scripts/make_performance.py \
  --config_file=$CONFIG_PATH/performance.yaml \
  --obs_time="$time" \
  --$MODE
done

printf "${GREEN}All performances have been estimated. \
 Please check the notebooks!${NC}\n"
