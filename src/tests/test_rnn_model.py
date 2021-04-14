from unittest import TestCase

from src.util.get_embeddings import random_embedding_vector
from src.models.rnn_model import Model
from src.stages.stage_prep_corpus import stage_prep_corpus
from src.stages.stage_prep_embedding import stage_prep_embedding
import torch
import torch.nn as nn


class TestModel(TestCase):
    target_vocab = ["<book_start>", "<title>", "<author>", "the"]

    def test_stage_rnn_model(self):

        matrix_len = len(TestModel.target_vocab)
        embedding_dim = 300

        vectors = torch.zeros((matrix_len, embedding_dim))

        for i, word in enumerate(TestModel.target_vocab):
            vectors[i] = torch.from_numpy(random_embedding_vector(embedding_dim=embedding_dim))

        embeddings = {"emb_layer": vectors
                    , "num_embeddings": matrix_len
                    , "embedding_dim": embedding_dim}

        embedding_layer = embeddings["emb_layer"]
        dictionary_size = embeddings["num_embeddings"]
        embedding_size = embeddings["embedding_dim"]

        model = Model(dictionary_size=dictionary_size
                    , embedding_layer=embedding_layer
                    , embedding_size=embedding_size)

        self.assertEqual(model.embedding_size, embeddings["embedding_dim"])

    def test_stage_rnn_model_dropout_default(self):
        number_of_layers = 1
        dropout_probability = .5
        test_dropout_probability = nn.Dropout(1)

        matrix_len = len(TestModel.target_vocab)
        embedding_dim = 300

        vectors = torch.zeros((matrix_len, embedding_dim))

        for i, word in enumerate(TestModel.target_vocab):
            vectors[i] = torch.from_numpy(random_embedding_vector(embedding_dim=embedding_dim))

        embedding_layer = vectors
        dictionary_size = vectors.size()[0]
        embedding_size = vectors.size()[1]

        model = Model(dictionary_size=dictionary_size
                      , embedding_layer=embedding_layer
                      , embedding_size=embedding_size
                      , number_of_layers=number_of_layers
                      , dropout_probability=dropout_probability)

        self.assertEqual(model.dropout.p, test_dropout_probability.p)









