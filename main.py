from src.runners.run_dataprep import run_dataprep
from src.util.config import run_configuration
import logging


if __name__ == '__main__':
    run_configuration()

    logging.info('Launched main.py')
    # embeddings = run_dataprep(embedding_type="glove_twitter", target_glove="glove.twitter.27B.200d.pickle")
    embeddings = run_dataprep(embedding_type="glove_common_crawl")

    #TODO: consume embeddings into a nn


