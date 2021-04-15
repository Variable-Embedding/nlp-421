from src.stages.stage_prep_corpus import stage_prep_corpus
from src.stages.stage_prep_embedding import stage_prep_embedding
from src.util.get_embeddings import prep_corpus_embeddings
from src.util.constants import *
import logging


def run_dataprep(embedding_type, corpus_type, target_glove=None):
    """
    """
    # pre-trained embedding -> {"word2idx": word2idx, "idx2word": idx2word, "vectors": vectors}
    embeddings = stage_prep_embedding(embedding_type=embedding_type, target_glove=target_glove)
    vocabulary = stage_prep_corpus(corpus_type)

    corpus_embeddings = {"word2idx": vocabulary['train'].stoi
                         , "vectors": embeddings['vectors']
                         , "target_vocab": vocabulary['train'].itos}

    nn_data = prep_corpus_embeddings(**corpus_embeddings)
    nn_data.update({'vocabulary': vocabulary})

    logging.info(f'Stage Data Prep Complete, returning nn_data:\n')
    for k, v in nn_data.items():
        logging.info(f'\tDictionary key: {k}, type: {type(v)}, with len: {len(v)}')

    return nn_data





