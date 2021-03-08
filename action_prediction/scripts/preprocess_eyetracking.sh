#!/bin/bash

## Command(s) to run:
# export PYTHONPATH=${PYTHONPATH}:/Users/maedbhking/Documents/bdd-driveratt
# export PYTHONPATH=${PYTHONPATH}:/Users/maedbhking/Documents/bdd-driveratt/eye_tracking/lib/pupil/pupil_src/shared_modules
# export PYTHONPATH=${PYTHONPATH}:/Users/maedbhking/.local/bin

# navigate to eyetracking dir
data_dir=/Users/maedbhking/Documents/cerebellum_learning_connect/data/BIDS_dir/sourcedata/behavioral_sessions/eyetracking/
cd /Users/maedbhking/Documents/cerebellum_learning_connect/data/BIDS_dir/sourcedata/behavioral_sessions/eyetracking/

# preprocess the data
python /Users/maedbhking/Documents/cerebellum_learning_connect/learning_connect/scripts/preprocess_eye.py --data_dir=${data_dir}