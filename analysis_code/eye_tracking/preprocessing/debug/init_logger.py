#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# LOGGING INIT

import time
import os, sys
import logging

# get a logger
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

# we want two handlers
# one for the sys.stdout (console)
# one for writing into the logfile

if len(logger.handlers) != 2:

    # create handlers
    logfile = os.path.join('/home/whitney/Teresa/etcomp/log_files',
                           str('temp_' + time.strftime("%Y_%m_%d-%H-%M-%S") + '.log'))

    # final logfile name
    logfile = os.path.join('/home/whitney/Teresa/etcomp/log_files',
                           str('log_preprocess_' + time.strftime("%Y_%m_%d-%H-%M-%S") + '.log'))

    # delete file if it already exists
    try:
        os.remove(logfile)
    except FileNotFoundError:
        pass

    # define handlers
    logging_file = logging.FileHandler(filename=logfile)
    logging_cons = logging.StreamHandler(sys.stdout)

    # set handler level
    logging_file.setLevel(logging.WARNING)
    logging_cons.setLevel(logging.DEBUG)

    # create a logging format
    formatter = logging.Formatter("%(asctime)s - %(name)-65s - %(levelname)-8s - %(message)s", "%Y-%m-%d %H:%M:%S")
    logging_file.setFormatter(formatter)
    logging_cons.setFormatter(formatter)

    # add the handlers to the logger
    logger.addHandler(logging_file)
    logger.addHandler(logging_cons)

else:
    print('Logger Already initialized? Found more than two handles')

# To close
# [h.close() for h in logger.handlers]
# [logger.removeHandler(h) for h in logger.handlers]

