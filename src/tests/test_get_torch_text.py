from unittest import TestCase

import random
from src.runners.run_dataprep import run_dataprep


class Test(TestCase):
    def test_get_torch_text(self):
        nn_data = run_dataprep(embedding_type="glove_common_crawl", corpus_type="WikiText2")

        data_sets = nn_data.keys()

        global_corpus = []
        for data_set in data_sets:
            data = nn_data[data_set]
            target_vocab = data["target_vocab"]
            global_corpus.append(target_vocab)

        common_corpus = list(set(set(global_corpus[0])
                                 .intersection(set(global_corpus[1])))
                             .intersection(set(global_corpus[2])))

        a_common_token = random.choice(common_corpus)

        vectors = []

        for data_set in data_sets:
            data = nn_data[data_set]
            word2idx = data["word2idx"]
            a_common_tokens_idx = word2idx[a_common_token]
            target_vocab = data["target_vocab"]
            vector = data["vectors"]
            a_common_tokens_vector = vector[a_common_tokens_idx]

            vectors.append(a_common_tokens_vector)

            self.assertEqual(a_common_token, target_vocab[a_common_tokens_idx])

        self.assertEqual(vectors[0].all(), vectors[1].all())
        self.assertEqual(vectors[1].all(), vectors[2].all())




        #TODO check to amke sure vectors are the same for a given word across corpra

