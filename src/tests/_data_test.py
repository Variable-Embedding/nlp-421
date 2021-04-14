
from src.util.get_embeddings import parse_embedding_pickle

from src.util.constants import *

import os
import random


class DataTest:
    """
    An embedding dictionary for testing from project workflow, else, return the pytorch glove vectors.
    """

    def __init__(self):

        def _find_embedding_pickle(curr_dir):
            """
            A helper function to find any embedding pickle file for testing embedding functions.

            :param curr_dir: Full path, preferably from constants.py
            :return: An embedding dictionary for testing from project workflow, else, return False.
            """
            directory = os.listdir(curr_dir)
            if len(directory) == 0:
                return False
            else:
                pickle_files = None
                while directory:
                    curr_ls = directory.pop()
                    next_dir = os.sep.join([EMBEDDING_FOLDER, curr_ls])
                    next_ls = os.listdir(next_dir)
                    pickle_files = [i for i in next_ls if ".pickle" in i]

                    if pickle_files:
                        file_name = random.choice(pickle_files)
                        file_path = os.sep.join([next_dir, file_name])
                        return file_path

                if pickle_files is None:
                    return False

        self.embedding_file = _find_embedding_pickle(EMBEDDING_FOLDER)
        self.embedding_dim = 300 if "300d" in self.embedding_file \
            else 200 if "200d" in self.embedding_file \
            else 100 if "100d" in self.embedding_file \
            else 50 if "50d" in self.embedding_file else 25

    def embeddings_dict(self):
        if self.embedding_file:
            return parse_embedding_pickle(self.embedding_file)
        else:
            return False

    def compare_embedding_dicts(self
                              , embedding_file_1="glove_common_crawl"
                              , embedding_file_2='torch_glove_840B'
                                ):
        embedding_1_path = os.sep.join([EMBEDDING_FOLDER, embedding_file_1])
        embedding_2_path = os.sep.join([EMBEDDING_FOLDER, embedding_file_2])

        embedding_1_pickle, embedding_2_pickle = None, None

        if os.path.exists(embedding_1_path) and os.path.exists(embedding_2_path):

            curr_ls = os.listdir(embedding_1_path)
            pickle_files = [i for i in curr_ls if ".pickle" in i]
            if pickle_files:
                file_name = random.choice(pickle_files)
                file_path = os.sep.join([embedding_1_path, file_name])

                embedding_1_pickle = parse_embedding_pickle(file_path)

            curr_ls = os.listdir(embedding_2_path)
            pickle_files = [i for i in curr_ls if ".pickle" in i]
            if pickle_files:
                file_name = random.choice(pickle_files)
                file_path = os.sep.join([embedding_2_path, file_name])

                embedding_2_pickle = parse_embedding_pickle(file_path)

            if embedding_1_pickle and embedding_2_pickle:
                return embedding_1_pickle, embedding_2_pickle

        else:

            return False

