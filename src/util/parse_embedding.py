import numpy as np
import logging
from src.util.constants import *


def parse_embedding(embedding_filename):
    logging.info(f'Parsing word embeddings from {embedding_filename}')
    embeddings_dict = {}

    with open(embedding_filename, 'r', encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], "float32")
            embeddings_dict[word] = vector

    return embeddings_dict
