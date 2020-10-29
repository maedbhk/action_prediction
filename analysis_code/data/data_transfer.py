import os

def behavior_from_savio():
    # transfer behavior data from savio)
    os.makedirs('/Users/shannonlee/Documents/social_eye/data/behavioral_sessions')
    os.system("rsync -avrz maedbhking@dtn.brc.berkeley.edu:/global/scratch/maedbhking/projects/cerebellum_learning_connect/data/BIDS_dir/sourcedata/behavioral_sessions/behavior  /Users/shannonlee/Documents/social_eye/data/behavioral_sessions")
