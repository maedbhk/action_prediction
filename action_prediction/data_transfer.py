import os

def behavior_from_savio():
    os.system("rsync -avrz maedbhking@dtn.brc.berkeley.edu:/global/scratch/maedbhking/projects/cerebellum_learning_connect/data/BIDS_dir/derivatives/behavior/group_behavior.csv /Users/maedbhking/Documents/action_prediction/data/behavior/")


def eyetracking_from_savio():
    os.system("rsync -avrz maedbhking@dtn.brc.berkeley.edu:/global/scratch/maedbhking/projects/cerebellum_learning_connect/data/BIDS_dir/sourcedata/behavioral_sessions/eyetracking/*.csv /Users/maedbhking/Documents/action_prediction/data/eyetracking/")
