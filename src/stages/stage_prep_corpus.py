from src.util.get_gutenberg import get_gutenberg, simple_vocabulary
from src.util.get_torch_text import get_torch_text
from src.util.constants import *
import logging
import string
import time


def stage_prep_corpus(corpus_type):
    """
    Manage all functions to process a target corpus, return a vocabulary for now.
    :return: A list of words representing all the unique words in a corpus, i.e. the vocabulary.
    """

    if corpus_type == "WikiText2":

        vocabulary, corpra = get_torch_text(corpus_type=corpus_type)

        return vocabulary, corpra
