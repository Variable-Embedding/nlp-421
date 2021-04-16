import logging
from src.util.get_embeddings import download_embeddings\
    , parse_embedding_pickle\
    , get_embeddings\
    , get_torch_glove

from src.util.constants import *


def stage_prep_embedding(embedding_type, target_glove=None):
    """Download and prepare GloVe embeddings for modeling.

    :param embedding_type: string, a key name for a type of GloVe embedding to download and prep
           embedding_type is one of:
                "glove_twitter"
                "glove_common_crawl"
                or "840B" which is the same as glove_common_crawl
    :param target_glove: string, optional. If passed, returns embedding dictionaries of the target GloVe.
    :return: dict of embedding -> {"word2idx": word2idx, "idx2word": idx2word, "vectors": vectors}
    """
    # make directory for specified embedding data

    if embedding_type == "840B":
        embedding_folder = os.sep.join([EMBEDDING_FOLDER, f'torch_glove_{embedding_type}'])
    else:
        embedding_folder = os.sep.join([EMBEDDING_FOLDER, embedding_type])

    if not os.path.exists(embedding_folder):
        os.makedirs(embedding_folder)

    if embedding_type == 'glove_common_crawl' or embedding_type == "840B" and target_glove is None:
        target_glove = "glove.840B.300d.pickle"

    def status_log(x):
        logging.info(f'Data Prep - Embedding is {x}.')
        if target_glove is not None:
            logging.info(f'Data Prep - '
                         f'Target Glove Embedding is {target_glove}.')
        else:
            logging.info('Data Prep - '
                         'Target Glove Embedding not specified, only downloading and preparing embedding data.'
                         )
            logging.info('Data Prep - '
                         'To return embedding data, specify a target embedding filename.')

    url = None
    has_embedding = False

    if embedding_type == "glove_twitter":
        status_log(embedding_type)
        url = GLOVE_TWITTER

    elif embedding_type == 'glove_common_crawl':
        status_log(embedding_type)
        url = GLOVE_COMMON_CRAWL_840B

    elif embedding_type == "840B":
        # this should be the same thing as glove_common_crawl, just packaged inside torch nlp method
        status_log(embedding_type)
        has_embedding = get_torch_glove(torch_glove_type=embedding_type)

    # has_embedding: bool, if True, download_embeddings worked
    if url is not None:
        has_embedding = download_embeddings(url, unzip_path=embedding_folder)

    if has_embedding and target_glove:
        glove_target = os.sep.join([embedding_folder, target_glove])
        glove_embeddings = parse_embedding_pickle(glove_target)
        word2idx, idx2word, vectors = get_embeddings(glove_embeddings)

        embeddings = {"word2idx": word2idx, "idx2word": idx2word, "vectors": vectors}

        return embeddings






