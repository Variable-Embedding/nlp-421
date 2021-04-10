
from src.stages.stage_prep_embedding import stage_prep_embedding


def run_dataprep(embedding_type, target_glove=None):
    # prepares dictionaries of {word : embedding vectors}
    embeddings = stage_prep_embedding(embedding_type=embedding_type, target_glove=target_glove)

    return embeddings





