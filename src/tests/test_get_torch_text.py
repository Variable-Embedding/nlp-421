from unittest import TestCase

import random
from src.util.get_torch_text import  get_torch_text


class Test(TestCase):
    def test_get_torch_text(self):
        vocabulary = get_torch_text("WikiText2")
        for k, v in vocabulary.items():
            a_word = random.choice(v.itos)
            a_index = v.stoi[a_word]
            self.assertEqual(a_word, v.itos[a_index])


