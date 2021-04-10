import logging
from src.util.get_embeddings import download_embeddings, parse_embedding_pickle, get_embeddings
from src.util.constants import *


def stage_prep_embedding(embedding_type, target_glove=None):
    """
    Download and prepare GloVe embeddings for modeling.

    :param embedding_type: string, a key name for a type of GloVe embedding to download and prep
        embedding_type is one of: "glove_twitter", "glove_common_crawl"
    :param target_glove: string, optional. If passed, returns embedding dictionaries of the target GloVe.
    :return: default None
    """
    # make directory for specified embedding data
    embedding_folder = os.sep.join([EMBEDDING_FOLDER, embedding_type])
    if not os.path.exists(embedding_folder):
        os.makedirs(embedding_folder)

    def status_log(x):
        logging.info(f'Data Prep - Embedding is {x}.')
        if target_glove is not None:
            logging.info(f'Data Prep - Target Glove Embedding is {target_glove}.')
        else:
            logging.info('Data Prep - '
                         'Target Glove Embedding not specified, only downloading and preparing embedding data.'
                         )
            logging.info('Data Prep - To return embedding data, specify a target embedding filename.')

    if embedding_type == "glove_twitter":
        status_log(embedding_type)
        url = GLOVE_TWITTER
    elif embedding_type == 'glove_common_crawl':
        status_log(embedding_type)
        url = GLOVE_COMMON_CRAWL_840B
    else:
        # add elif statements here for other embedding types
        url = None

    # has_embedding: bool, if True, download_embeddings worked
    if url is not None:
        has_embedding = download_embeddings(url, unzip_path=embedding_folder)
    else:
        has_embedding = False

    if has_embedding and target_glove:
        glove_target = os.sep.join([embedding_folder, target_glove])
        glove_embeddings = parse_embedding_pickle(glove_target)
        embeddings = get_embeddings(glove_embeddings)
        return embeddings







