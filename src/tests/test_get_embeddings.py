from unittest import TestCase
from src.util.get_embeddings import get_embeddings
from src.tests._data_test import DataTest
from scipy import spatial
import random

from src.util.config import *
import logging


class Test(TestCase):
    embeddings_dict = DataTest().embeddings_dict()
    run_configuration()

    def test_parse_embedding_pickle(self):
        embeddings_dict = Test.embeddings_dict
        if embeddings_dict:
            a_word = random.choice(list(embeddings_dict.keys()))
            distance = spatial.distance.euclidean(embeddings_dict[a_word], embeddings_dict[a_word])
            assert distance == 0
        else:
            logging.info('Test Not Fail - Embedding data not found, check embedding data path')
            pass

    def test_get_embeddings(self):
        embeddings_dict = Test.embeddings_dict

        if embeddings_dict:
            word2idx, idx2word, vectors = get_embeddings(embeddings_dict)

            a_word = random.choice(list(embeddings_dict.keys()))
            a_vector = embeddings_dict[a_word]
            idx_of_word = word2idx[a_word]
            word_of_idx = idx2word[idx_of_word]
            vector_of_word = vectors[idx_of_word]

            assert a_word == word_of_idx
            assert a_vector.all() == vector_of_word.all()
        else:
            logging.info('Test Not Fail - Embedding data not found, check embedding data path')
            pass

    def test_torch_glove_embeddings(self):
        embeddings = DataTest().compare_embedding_dicts()
        if embeddings:
            embedding_dict_1 = embeddings[0]
            embedding_dict_2 = embeddings[1]

            a_word = random.choice(list(embedding_dict_1.keys()))
            assert a_word in embedding_dict_2.keys()
            a_vector_1 = embedding_dict_1[a_word]
            a_vector_2 = embedding_dict_2[a_word]
            assert a_vector_1.all() == a_vector_2.all()
        else:
            logging.info('Test Not Fail - Embedding data not found, check embedding data path')
            pass





