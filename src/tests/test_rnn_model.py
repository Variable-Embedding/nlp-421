from unittest import TestCase

from src.models.rnn_model import Model
from src.stages.stage_prep_corpus import stage_prep_corpus
from src.stages.stage_prep_embedding import stage_prep_embedding


class TestModel(TestCase):

    def test_stage_prep_embedding(self):
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




