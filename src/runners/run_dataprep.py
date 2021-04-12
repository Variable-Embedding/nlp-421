from src.stages.stage_prep_corpus import stage_prep_corpus
from src.stages.stage_prep_embedding import stage_prep_embedding


def run_dataprep(embedding_type, target_glove=None):
    """
    A runner to prepare data for nlp pipeline, consolidate stages and helper functions

    :param embedding_type: string, a key name for a type of GloVe embedding to download and prep
        embedding_type is one of: "glove_twitter", "glove_common_crawl", or "840B" which is the same as glove_common_crawl
        If "glove_common_crawl" or "840B" are selected, then the target glove is defaulted to "glove.840B.300d.pickle"
        and embedding size is default to 300 (based on glove params).
    :param target_glove:
    :return: all data required to start nlp-pipeline.
    """
    # should be some function that returns the target vocabulary
    target_vocab = stage_prep_corpus(is_gutenberg=True)

    # prepares dictionaries of {word : embedding vectors}
    embeddings = stage_prep_embedding(embedding_type=embedding_type
                                      , target_glove=target_glove
                                      , target_vocab=target_vocab)

    return embeddings





