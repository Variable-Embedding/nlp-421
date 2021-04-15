
from src.util.constants import *
import logging

import torch
from torchtext.datasets.wikitext2 import WikiText2
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.utils import get_tokenizer
from collections import Counter
from torchtext.vocab import Vocab
from torchtext.experimental.vocab import build_vocab_from_text_file
from torchtext.experimental.transforms import basic_english_normalize


def get_torch_text(corpus_type):
    """
    A convenience wrapper to get corpra for experimentation with torch text.
    :param corpus_type: string, required. Any of available torch text dataset types.
    :return:

    source: https://github.com/pytorch/text/tree/master/torchtext
    """
    torch_text_name = 'wikitext-2' if corpus_type == "WikiText2" else None
    torch_text_path = os.sep.join([CORPRA_FOLDER, torch_text_name])

    if corpus_type == "WikiText2":
        logging.info(f'Using TorchText to get {corpus_type} at {CORPRA_FOLDER}')

        if torch_text_name and os.path.exists(torch_text_path):
            vocabulary = read_torch_vocab(torch_text_path, corpus_type)
            return vocabulary

        else:

            logging.info(f'Did not find existing .token files. Downloading and returning corpra.')
            WikiText2(root=CORPRA_FOLDER)

            vocabulary = read_torch_vocab(torch_text_path, corpus_type)
            return vocabulary
    else:

        return


def read_torch_vocab(torch_text_path, corpus_type):
    """Leveraging torch text experimental functions.

    :param torch_text_path:
    :param corpus_type:
    :return:

    source: https://github.com/pytorch/text/blob/master/torchtext/experimental/vocab.py
    """

    files = os.listdir(torch_text_path)
    if all([".tokens" in i for i in files]):
        logging.info(f'Found existing {corpus_type} .token files.\n'
                     f'\tReturning copra from disk instead of downloading them.\n'
                     f'\tTo force new download, delete or rename these files:\n'
                     f'\t{files}')

        tokenizer = get_tokenizer('basic_english')
        vocabulary = {}

        for file in files:
            file_path = os.sep.join([torch_text_path, file])
            counter = Counter()
            f = open(file_path, 'r')

            for line in f:
                counter.update(tokenizer(line))

            v = Vocab(counter, min_freq=1)
            key = 'train' if '.train.' in file else 'test' if '.test.' in file else 'valid'
            vocabulary.update({key: v})
            f.close()

        logging.info(f'Completed parsing vocab for {corpus_type}.')
        for k, v in vocabulary.items():
            logging.info(f'Dataset {k}: with vocabulary of length: {len(v)}.')

        return vocabulary



