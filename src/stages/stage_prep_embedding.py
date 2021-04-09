import logging
from src.util.get_zipfile import get_zipfile
from src.util.parse_embedding import parse_embedding
from src.util.constants import *


def stage_prep_embedding(embedding_type, embedding_version):
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

    if url is not None:
        has_embedding = get_zipfile(url, unzip_path=embedding_folder)
    else:
        has_embedding = False

    embedding_file_path = os.sep.join([embedding_folder, embedding_version])

    if has_embedding and os.path.exists(embedding_file_path):
        embeddings_dict = parse_embedding(embedding_file_path)

        return embeddings_dict



