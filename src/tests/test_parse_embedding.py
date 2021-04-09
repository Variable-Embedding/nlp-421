from unittest import TestCase
from src.util.parse_embedding import parse_embedding
from src.util.constants import *

import os
from scipy import spatial

class Test(TestCase):
    def test_parse_embedding(self):
        embedding_file = os.sep.join([WORKFLOW_ROOT, 'data', 'embeddings', 'glove_twitter', 'glove.twitter.27B.25d.txt'])
        embeddings_dict = parse_embedding(embedding_file)
        result = spatial.distance.euclidean(embeddings_dict['king'], embeddings_dict['king'])

        assert result == 0
