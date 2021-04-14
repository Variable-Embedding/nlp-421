from src.util.get_gutenberg import get_gutenberg, simple_vocabulary
from src.util.get_torch_text import get_torch_text
from src.util.constants import *
import logging
import string
import time


def stage_prep_corpus(corpus_type='gutenberg'):
    """
    Manage all functions to process a target corpus, return a vocabulary for now.
    :return: A list of words representing all the unique words in a corpus, i.e. the vocabulary.
    """

    if corpus_type == 'gutenberg':
        book_urls = [
                     "https://www.gutenberg.org/files/1342/1342-0.txt"
                   , "https://www.gutenberg.org/files/11/11-0.txt"
                   , "https://www.gutenberg.org/files/98/98-0.txt"
                   , "https://www.gutenberg.org/files/65053/65053-0.txt"
                   , "https://www.gutenberg.org/files/5200/5200-0.txt"
                   , "https://www.gutenberg.org/files/174/174-0.txt"
                   , "https://www.gutenberg.org/files/1952/1952-0.txt"
                   , "https://www.gutenberg.org/files/1400/1400-0.txt"
                   , "https://www.gutenberg.org/files/46/46-0.txt"
                   , "https://www.gutenberg.org/files/863/863-0.txt"
                   , "https://www.gutenberg.org/files/4517/4517-0.txt"
                   , "https://www.gutenberg.org/files/41/41-0.txt"
                   , "https://www.gutenberg.org/files/1399/1399-0.txt"
                     ]

        vocabulary = []
        corpra = []

        book_counter = 0

        for idx, book_url in enumerate(book_urls):
            book = get_gutenberg(book_url)
            book_vocab, book_corpus = simple_vocabulary(book)
            book_vocab.sort()
            vocabulary.extend(book_vocab)
            corpra.extend(book_corpus)
            book_counter += 1

        # return only the unique instances of words in the corpus
        vocabulary = list(set(vocabulary))
        logging.info(f'Corpus returning {book_counter} books, with vocab size of {len(vocabulary)}, a total of {len(corpra)} tokens.')
        return vocabulary, corpra

    elif corpus_type == "WikiText2":

        vocabulary = get_torch_text(corpus_type=corpus_type)

        breakpoint()

        return vocabulary, None
