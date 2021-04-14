from unittest import TestCase

from src.util.get_embeddings import random_embedding_vector
from src.models.rnn_model import Model
from src.stages.stage_prep_corpus import stage_prep_corpus
from src.stages.stage_prep_embedding import stage_prep_embedding
import torch
import torch.nn as nn


class TestModel(TestCase):

    def test_stage_rnn_model(self):
        target_vocab, _ = stage_prep_corpus(is_gutenberg=True)

        embeddings = stage_prep_embedding(embedding_type="glove_common_crawl"
                                          , target_vocab=target_vocab)

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

        target_vocab = ["<book_start>", "<title>", "<author>", "the"]

        matrix_len = len(target_vocab)
        embedding_dim = 300

        vectors = torch.zeros((matrix_len, embedding_dim))

        for i, word in enumerate(target_vocab):
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









