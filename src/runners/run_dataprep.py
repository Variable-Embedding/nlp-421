
from src.stages.stage_prep_embedding import stage_prep_embedding

def run_dataprep(embedding_type, embedding_version):
    # this is embeddings dict: key is word, value is embedding
    embeddings_dict = stage_prep_embedding(embedding_type=embedding_type, embedding_version=embedding_version)





