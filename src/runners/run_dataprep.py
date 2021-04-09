
from src.stages.stage_prep_embedding import stage_prep_embedding

def run_dataprep(embedding_type):
    # this is embeddings dict: key is word, value is embedding
    stage_prep_embedding(embedding_type=embedding_type)





