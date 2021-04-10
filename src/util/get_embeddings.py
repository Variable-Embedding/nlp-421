import requests, zipfile, io
import os
import logging
from pathlib import Path
from src.util.constants import *
import pickle
import time
import numpy as np


# TODO: Complete this function
# def prep_nn_embeddings(word2idx, target_vocab):
#     """
#     Consume pre-trained embedding with vocab of target corpus, return embedding layer for training.
#
#     :returns : an embedding space t hat has words from the target vocabulary, if they exists, or
#     initialize random embedding for new words from the target corpus
#
#     source: https://medium.com/@martinpella/how-to-use-pre-trained-word-embeddings-in-pytorch-71ca59249f76
#     """
#     matrix_len = len(target_vocab)
#     weights_matrix = np.zeros((matrix_len, 50))
#     words_found = 0
#
#     for i, word in enumerate(target_vocab):
#         try:
#             weights_matrix[i] = word2idx[word]
#             words_found += 1
#         except KeyError:
#             weights_matrix[i] = np.random.normal(scale=0.6, size=(emb_dim,))

def get_embeddings(glove_embeddings):
    """
    Return pre-trained embedding data to the neural network pipeline.

    :param glove_embeddings: A dict of GloVe embeddings
    :return: Return a 3-Tuple of GloVe embeddings :
             word2idx, dict, {word:int}
             idx2word, dict, {int:word}
             vectors, list, [numpy array]
    """
    # Initialize data structs
    words = []
    idx = 0
    word2idx = {}
    idx2word = {}
    # TODO: vectors will need to become array/torch object.
    vectors = []

    for word, vector in glove_embeddings.items():
        words.append(word)
        word2idx[word] = idx
        idx2word[idx] = word
        vectors.append(vector)
        idx += 1

    return word2idx, idx2word, vectors


def download_embeddings(url, unzip_path):
    """
    Get and process GloVe embeddings.

    :param url: The target URL, see constants.py for urls to GloVe downloads.
    :param unzip_path: The full path to unzip GloVe downloads to.
    :return: bool, return True if everything runs
    """

    directory = os.listdir(unzip_path)

    if len(directory) > 0:
        logging.info(f'Directory Not Empty {unzip_path}.')
        logging.info(f'Current files {directory}')
        logging.info(f'Skipping download from URL: {url} - to force download, delete or rename files in this directory.')

        write_pickle(directory, unzip_path)

    else:
        logging.info(f'Starting download from {url}.')
        r = requests.get(url)
        z = zipfile.ZipFile(io.BytesIO(r.content))
        z.extractall(unzip_path)
        directory = os.listdir(unzip_path)
        logging.info(f'Downloaded {directory} to {unzip_path}')

        write_pickle(directory, unzip_path)

    logging.info('download_embeddings() Complete')

    return True


def write_pickle(directory, unzip_path):
    """
    Write dictionaries to pickled files if the provided directory does not already have pickle files.

    :param directory: A dictionary of word embeddings.
    :param unzip_path: Full path to file locations.
    :return: None, writes data to disk.
    """

    if any([".pickle" in i for i in directory]):
        logging.info(f'Pickle files detected, skipping conversion from .txt to .pickle format.')

    else:
        logging.info(f'Converting .txt embedding files to .pickle objects for faster IO.')

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
    Read pickled embedding files from disk.

    :param embedding_file_path: string, full path to the embedding txt file
    :return: a dictionary of embeddings k: word (string), v: embedding vector of float32s (numpy array)
    """
    logging.info(f'Reading embedding file from {embedding_file_path}.')
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
    Read text embedding files from disk.

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
    """
    Read lines in a text file from embeddings.

    :param line: Each line of a text open object.
    :return: 2-tuple of word (string) and numpy array (vector).
    """

    values = line.split()
    # the first element is assumed to be the word
    word = values[0]

    # catch cases where first n strings are repeating as the word
    try:
        # the rest of list is usually the vector but sometimes it is not
        vector = np.asarray(values[1:], "float32")
    except ValueError:
        word = return_repeating_word(values)
        vector = np.asarray(values, "float32")

    return tuple((word, vector))


def return_repeating_word(values):
    """
    A helper function for read_line(). Return repeating chars as a single word.

    :param values: values, a line of embedding text data
    :return: A string of repeating characters. Manipulate values input with pop(0)
    """
    word = []
    first_char = values.pop(0)
    word.append(first_char)

    while values:
        curr_char = values.pop(0)
        if curr_char == first_char:
            word.append(curr_char)
        else:
            break

    word = ''.join(map(str, word))

    return word




