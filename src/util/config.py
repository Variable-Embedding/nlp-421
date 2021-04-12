"""Runs system configurations
"""
from src.util import constants
from src.util.constants import *

import logging
import sys

from datetime import datetime
import os
from os.path import join


def run_configuration():
    """Runs basic configuration for the workflow.
    """
    for folder in ALL_FOLDERS:
        if not os.path.exists(folder):
            os.makedirs(folder)

    logger = logging.getLogger("pipeline").getChild("configuration")
    format = "%(asctime)s:%(name)s:%(levelname)s:%(message)s"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    log_filename = "{}.log".format(timestamp)
    filename = join(constants.LOGGING_PATH, log_filename)

    log_formatter = logging.Formatter(format)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(log_formatter)

    logging.basicConfig(filename=filename, format=format, level=logging.INFO)
    logging.getLogger().addHandler(stream_handler)



    logger.info("Logging configurations finished.")
