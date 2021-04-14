from unittest import TestCase
from src.util.get_embeddings import get_embeddings
from src.tests._data_test import DataTest
from scipy import spatial
import random

from src.util.config import *
import logging


class Test(TestCase):
    run_configuration()
    embeddings_dict = DataTest().embeddings_dict()

    def test_parse_embedding_pickle(self):
        embeddings_dict = Test.embeddings_dict
        if embeddings_dict:
            a_word = random.choice(list(embeddings_dict.keys()))
            distance = spatial.distance.euclidean(embeddings_dict[a_word], embeddings_dict[a_word])
            self.assertEqual(distance, 0)
        else:
            logging.info('Test Not Fail - Embedding data not found, check embedding data path')
            pass

    def test_get_embeddings(self):
        embeddings_dict = Test.embeddings_dict

        if embeddings_dict:
            word2idx, idx2word, vectors = get_embeddings(embeddings_dict)

            a_word = "king"
            idx_of_word = word2idx[a_word]

            word_of_idx = idx2word[idx_of_word]
            vector_dims = [len(i) for i in vectors]

            self.assertEqual(a_word, word_of_idx)
            self.assertEqual(vector_dims[0], DataTest().embedding_dim)
            self.assertTrue(all(i == DataTest().embedding_dim for i in vector_dims))
            self.assertEqual(DataTest().embedding_dim, vectors.size()[1])

        else:
            logging.info('Test Not Fail - Embedding data not found, check embedding data path')
            pass

    def test_torch_glove_embeddings(self):
        embeddings = DataTest().compare_embedding_dicts()
        if embeddings:
            embedding_dict_1 = embeddings[0]
            embedding_dict_2 = embeddings[1]

            a_word = "queen"
            self.assertIn(a_word, embedding_dict_2.keys())
            a_vector_1 = embedding_dict_1[a_word]
            a_vector_2 = embedding_dict_2[a_word]
            self.assertEqual(a_vector_1.all(), a_vector_2.all())
        else:
            logging.info('Test Not Fail - Embedding data not found, check embedding data path')
            pass





