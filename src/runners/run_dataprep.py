from src.stages.stage_prep_corpus import stage_prep_corpus
from src.stages.stage_prep_embedding import stage_prep_embedding
from src.util.get_embeddings import prep_corpus_embeddings
from src.util.constants import *
import logging


def run_dataprep(embedding_type, corpus_type, target_glove=None):
    """
    :returns: a dictionary of data keyed by phases: 'train', 'valid', 'test'
           {
            # word2idx -> dict of token to idx
            "word2idx": word2idx
            # vectors -> torch tensor embedding layer
             , "vectors": vectors
            # target_vocab -> list of tokens
             , "target_vocab": target_vocab
            # vocabulary -> torchtext Vocab object
             , "vocabulary": vocab
            # corpus -> the numeric representation of corpus, in original corpus sequence
             , "corpus": corpus
             }
    """
    # pre-trained embedding -> {"word2idx": word2idx, "idx2word": idx2word, "vectors": vectors}
    embeddings = stage_prep_embedding(embedding_type=embedding_type, target_glove=target_glove)
    vocabulary, corpra = stage_prep_corpus(corpus_type)
    # data_sets -> 'train', 'valid', 'test'
    data_sets = vocabulary.keys()

    nn_data = {}

    for data_set in data_sets:
        vocab = vocabulary[data_set]
        corpus = corpra[data_set]
        vectors = embeddings["vectors"]
        # torchtext.vocab.Vocab.stoi -> dict of token string to numeric id
        word2idx = vocab.stoi
        # torchtext.vocab.Vocab.itos -> list of token string indexed by numeric id
        target_vocab = vocab.itos

        x = {"word2idx": word2idx
             , "vectors": vectors
             , "target_vocab": target_vocab}

        word2idx, idx2word, vectors = prep_corpus_embeddings(**x)

        y = {
            # word2idx -> dict of token to idx
            "word2idx": word2idx
            # torch tensor embedding layer
             , "vectors": vectors
            # target_vocab -> list of tokens
             , "target_vocab": target_vocab
            # vocabulary -> torchtext Vocab object
             , "vocabulary": vocab
            # the numeric representation of corpus, in original sequence
             , "corpus": corpus
             }

        nn_data.update({data_set: y})

    logging.info(f'Stage Data Prep Complete, returning nn_data dictionary containing:\n')

    for k, v in nn_data.items():
        logging.info(f'\tDataset {k}: with data: {v.keys()}')

    return nn_data





