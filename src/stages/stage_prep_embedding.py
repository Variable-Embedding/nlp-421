import logging
from src.util.get_embeddings import get_embeddings, parse_embedding_pickle
from src.util.constants import *


def stage_prep_embedding(embedding_type, target_glove=None):
    """
    Download and prepare GloVe embeddings for modeling.

    :param embedding_type: string, a key name for a type of GloVe embedding to download and prep
    :param target_glove: string, optional. If passed, returns embedding dictionaries of the target GloVe.
    :return: default None
    """
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

    if has_embedding and target_glove:
        glove_target = os.sep.join([embedding_folder, target_glove])
        embeddings = parse_embedding_pickle(glove_target)

        #TODO: configure embedding objects to work with nn
        words = []
        idx = 0
        word2idx = {}
        idx2word = {}
        vectors = []

        for word, vector in embeddings.items():
            words.append(word)
            word2idx[word] = idx
            idx2word[idx] = word
            vectors.append(vector)
            idx += 1

        #TODO: Integerate embedding objects with LSTM RNN





