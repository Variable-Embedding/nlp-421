from src.stages.stage_prep_corpus import stage_prep_corpus
from src.stages.stage_prep_embedding import stage_prep_embedding
from src.util.constants import *
import logging


def run_dataprep(embedding_type, is_gutenberg=True, target_glove=None):
    """
    A runner to prepare data for nlp pipeline, consolidate stages and helper functions

    :param embedding_type: string, a key name for a type of GloVe embedding to download and prep
        embedding_type is one of: "glove_twitter", "glove_common_crawl", or "840B" which is the same as glove_common_crawl
        If "glove_common_crawl" or "840B" are selected, then the target glove is defaulted to "glove.840B.300d.pickle"
        and embedding size is default to 300 (based on glove params).
    :param target_glove: the target pre-trained word embeddings to use
    :param is_gutenberg: default to true, a toy dataset
    :return nn_embedding data and the target corpra:

    all data required to start nlp-pipeline as a dictionary

    :nn_embedding:    {"emb_layer": emb_layer
                      , "num_embeddings": num_embeddings
                      , "embedding_dim": embedding_dim
                      , "word2idx": word2idx
                      , "idx2word": idx2word
                      , "vectors": vectors
                        }

    :target_corpra: a list of tokens representing the toy dataset of books
    """
    # should be some function that returns the target vocabulary
    target_vocab, target_corpra = stage_prep_corpus(is_gutenberg=is_gutenberg)

    # prepares dictionaries of {word : embedding vectors}
    embeddings = stage_prep_embedding(embedding_type=embedding_type
                                      , target_glove=target_glove
                                      , target_vocab=target_vocab)

    logging.info(f'Stage Data Prep Complete, returning embedding dictionaries and corpra\n')

    return embeddings, target_corpra





