from src.runners.run_dataprep import run_dataprep
from src.util.config import run_configuration
import logging


if __name__ == '__main__':
    run_configuration()
    logging.info('Launched main.py')

    # our custom functions to prep toy dataset and glove embeddings
    embeddings, corpra = run_dataprep(embedding_type="glove_common_crawl")

    ### TESTING OTHER TYPES OF EMBEDDINGS BELOW ###

    # the same thing as above but with torchnlp taking care of the download
    # embeddings = run_dataprep(embedding_type="840B")





