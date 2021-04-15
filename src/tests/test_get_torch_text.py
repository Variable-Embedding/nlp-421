from unittest import TestCase

import random
from src.runners.run_dataprep import run_dataprep


class Test(TestCase):
    def test_get_torch_text(self):
        nn_data = run_dataprep(embedding_type="glove_common_crawl", corpus_type="WikiText2")

        a_word = "king"

        data_sets = nn_data.keys()

        # 'word2idx', 'vectors', 'target_vocab', 'vocabulary', 'corpus'

        check_vectors = []

        for data_set in data_sets:
            data = nn_data[data_set]
            word2idx = data["word2idx"]
            a_word_idx = word2idx[a_word]
            target_vocab = data["target_vocab"]
            vectors = data["vectors"]

            self.assertEqual(a_word, target_vocab[a_word_idx])
            check_vectors.append(vectors[a_word_idx])

        #TODO check to amke sure vectors are the same for a given word across corpra

