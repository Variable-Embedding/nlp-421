from unittest import TestCase

from src.util.get_embeddings import random_embedding_vector
from src.models.rnn_model import Model
from src.stages.stage_prep_corpus import stage_prep_corpus
from src.stages.stage_prep_embedding import stage_prep_embedding
import torch
import torch.nn as nn


class TestModel(TestCase):
    target_vocab = ["<book_start>", "<title>", "<author>", "the"]
    matrix_len = len(target_vocab)
    embedding_dim = 300

    def test_rnn_model(self):
        embedding_layer = torch.zeros((TestModel.matrix_len, TestModel.embedding_dim))

        for i, word in enumerate(TestModel.target_vocab):
            embedding_layer[i] = torch.from_numpy(random_embedding_vector(embedding_dim=TestModel.embedding_dim))

        embeddings = {"embedding_layer": embedding_layer
                    , "dictionary_size": TestModel.matrix_len
                    , "embedding_size": TestModel.embedding_dim}

        embedding_layer = embeddings["embedding_layer"]
        dictionary_size = embeddings["dictionary_size"]
        embedding_size = embeddings["embedding_size"]

        model = Model(dictionary_size=dictionary_size
                      , embedding_layer=embedding_layer
                      , embedding_size=embedding_size)

        self.assertEqual(model.embedding_size, embeddings["embedding_size"])

    def test_rnn_model_dropout_default(self):
        number_of_layers = 1
        dropout_probability = .5
        expected_dropout = 1
        expected_dropout_probability = nn.Dropout(expected_dropout)

        embedding_layer = torch.zeros((TestModel.matrix_len, TestModel.embedding_dim))

        for i, word in enumerate(TestModel.target_vocab):
            embedding_layer[i] = torch.from_numpy(random_embedding_vector(embedding_dim=TestModel.embedding_dim))

        embedding_layer = embedding_layer
        dictionary_size = embedding_layer.size()[0]
        embedding_size = embedding_layer.size()[1]

        model = Model(dictionary_size=dictionary_size
                      , embedding_layer=embedding_layer
                      , embedding_size=embedding_size
                      , number_of_layers=number_of_layers
                      , dropout_probability=dropout_probability)

        self.assertEqual(model.dropout.p, expected_dropout_probability.p)









