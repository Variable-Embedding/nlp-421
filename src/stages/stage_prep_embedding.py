import logging
from src.util.get_embeddings import get_embeddings
from src.util.constants import *


def stage_prep_embedding(embedding_type):
    if embedding_type == "glove_twitter":
        logging.info('Data Prep - Embedding is GloVe Twitter.')
        url = GLOVE_TWITTER
        embedding_folder = os.sep.join([EMBEDDING_FOLDER, embedding_type])
        if not os.path.exists(embedding_folder):
            os.makedirs(embedding_folder)
    else:
        # add elif statements here for other embedding types
        url = None
        embedding_folder = None

    # has_embedding: bool, if True, get_embeddings worked
    if url is not None:
        has_embedding = get_embeddings(url, unzip_path=embedding_folder)
    else:
        has_embedding = False





