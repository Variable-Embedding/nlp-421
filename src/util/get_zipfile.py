import requests, zipfile, io
import os
import logging

from src.util.constants import *


def get_zipfile(url, unzip_path):

    directory = os.listdir(unzip_path)

    if len(directory) > 0:
        logging.info(f'Directory Not Empty {unzip_path}.')
        logging.info(f'Current files {directory}')
        logging.info(f'Skipping download from URL: {url} - to force download, delete or rename files in this directory.')
    else:
        logging.info(f'Starting download from {url}.')
        r = requests.get(url)
        z = zipfile.ZipFile(io.BytesIO(r.content))
        z.extractall(unzip_path)
        directory = os.listdir(unzip_path)
        logging.info(f'Downloaded {directory} to {unzip_path}')

    logging.info('get_zipfile() Complete')

    return True
