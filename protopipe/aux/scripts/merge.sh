#!/usr/bin/env bash

PROTOPIPE= # fill with the path of your local protopipe repository

# User variables
TYPE="dl1"  # Here dl1 or dl2
MODE="tail"  # Here, tail or wave
PARTICLE="gamma"  # This a list: gamma, proton, electron

# For Python's script
INPUT_DIR="" # fill with folder containing the HDF5 files you want to merge
OUTPUT_DIR=$INPUT_DIR

# Merge files
for part in $PARTICLE; do
    echo "Merging $part..."
    OUTPUT_FILE="$OUTPUT_DIR/${TYPE}_${MODE}_${part}_merged.h5"
    TEMPLATE_FILE_NAME="${TYPE}_${MODE}_${part}"

    echo $OUTPUT_DIR
    echo $TEMPLATE_FILE_NAME
    echo $OUTPUT_FILE

    python $PROTOPIPE/aux/scripts/merge_tables.py --indir=$INPUT_DIR --template_file_name=$TEMPLATE_FILE_NAME --outfile=$OUTPUT_FILE
done
