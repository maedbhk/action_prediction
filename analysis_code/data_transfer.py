import os

def data_from_savio():
    # transfer behavior data from savio) 
    os.system("rsync -avrz maedbhking@dtn.brc.berkeley.edu:/global/scratch/maedbhking/projects/cerebellum_learning_connect/data/BIDS_dir/derivatives/behavior  /Users/maedbhking/Documents/social_eye/data/derivatives/")
    os.system("rsync -avrz maedbhking@dtn.brc.berkeley.edu:/global/scratch/maedbhking/projects/cerebellum_learning_connect/data/BIDS_dir/derivatives/eyetracking  /Users/maedbhking/Documents/social_eye/data/derivatives/")