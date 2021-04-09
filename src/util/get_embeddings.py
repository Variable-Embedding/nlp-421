import requests, zipfile, io
import os
import logging
from pathlib import Path
from src.util.constants import *
import pickle
import time
import numpy as np


def get_embeddings(url, unzip_path):

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

        if any([".pickle" in i for i in directory]):
            logging.info(f'Pickle files detected, skipping conversion from .txt to .pickle format.')
            pass
        else:
            logging.info(f'Converting .txt embedding files to .pickle objects for faster IO.')
            write_pickle(directory, unzip_path)

    logging.info('get_embeddings() Complete')

    return True


def write_pickle(directory, unzip_path):
    for text_file in directory:
        text_file_path = os.sep.join([unzip_path, text_file])
        embedding_dict = parse_embedding_txt(text_file_path)

        pickle_file = f'{Path(text_file).stem}.pickle'
        pickle_file_path = os.sep.join([unzip_path, pickle_file])

        f = open(pickle_file_path, "wb")
        pickle.dump(embedding_dict, f, protocol=2)
        f.close()
        # test read pickle
        parse_embedding_pickle(pickle_file_path)

        logging.info(f'Wrote embedding pickle to {pickle_file_path}.')


def parse_embedding_pickle(embedding_file_path):
    """
    :param embedding_file_path: string, full path to the embedding txt file
    :return: a dictionary of embeddings k: word (string), v: embedding vector of float32s (numpy array)
    """
    if ".pickle" in embedding_file_path:
        start_time = time.time()

        with open(embedding_file_path, "rb") as handle:
            embedding_dict = pickle.load(handle)

        end_time = time.time()
        logging.info(f'PICKLE Read Time is {end_time - start_time}')
    else:
        logging.info(f'Did not read file, please indicate a file ending in .pickle.')

    return embedding_dict


def parse_embedding_txt(embedding_file_path):
    """
    :param embedding_file_path: string, full path to the embedding txt file
    :return: a dictionary of embeddings k: word (string), v: embedding vector of float32s (numpy array)
    """
    logging.info(f'Parsing word embeddings from {embedding_file_path}')
    embedding_dict = {}

    start_time = time.time()
    with open(embedding_file_path, 'r', encoding="utf-8") as f:
        for line in f:
            word, vector = read_line(line)
            embedding_dict.update({word: vector})
    end_time = time.time()
    logging.info(f'TEXT Read Time is {end_time-start_time}')

    return embedding_dict


def read_line(line):
    values = line.split()
    word = values[0]
    vector = np.asarray(values[1:], "float32")

    return tuple((word, vector))
