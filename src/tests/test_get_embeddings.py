from unittest import TestCase
from src.util.get_embeddings import parse_embedding_pickle, get_embeddings
from src.tests._data_test import DataTest
from scipy import spatial
import random


class Test(TestCase):
    embeddings_dict = DataTest().embeddings_dict()

    def test_parse_embedding_pickle(self):
        embeddings_dict = Test.embeddings_dict
        a_word = random.choice(list(embeddings_dict.keys()))
        distance = spatial.distance.euclidean(embeddings_dict[a_word], embeddings_dict[a_word])
        assert distance == 0

    def test_get_embeddings(self):
        embeddings_dict = Test.embeddings_dict
        word2idx, idx2word, vectors = get_embeddings(embeddings_dict)

        a_word = random.choice(list(embeddings_dict.keys()))
        a_vector = embeddings_dict[a_word]
        idx_of_word = word2idx[a_word]
        word_of_idx = idx2word[idx_of_word]
        vector_of_word = vectors[idx_of_word]

        assert a_word == word_of_idx
        assert a_vector.all() == vector_of_word.all()



