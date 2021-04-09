from unittest import TestCase
from src.util.get_embeddings import parse_embedding_pickle
from src.util.constants import *

import os
from scipy import spatial


class Test(TestCase):
    def test_get_embeddings(self):
        pickle_name = 'glove.twitter.27B.25d.pickle'

        embedding_file = os.sep.join([WORKFLOW_ROOT, 'data', 'embeddings', 'glove_twitter', pickle_name])
        embeddings_dict = parse_embedding_pickle(embedding_file)

        result = spatial.distance.euclidean(embeddings_dict['king'], embeddings_dict['king'])
        assert result == 0
