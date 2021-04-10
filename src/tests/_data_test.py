
from src.util.get_embeddings import parse_embedding_pickle
from src.util.constants import *

import os
import random


class DataTest:

    def __init__(self):

        def _find_embedding_pickle(curr_dir):
            """
            A helper function to find any embedding pickle file for testing embedding functions.

            :param curr_dir: Full path, preferably from constants.py
            :return: An embedding dictionary for testing.
            """
            directory = os.listdir(curr_dir)
            if len(directory) == 0:
                return
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

    def embeddings_dict(self):
        x = parse_embedding_pickle(self.embedding_file)
        return x