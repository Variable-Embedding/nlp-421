from src.runners.run_dataprep import run_dataprep
from src.util.config import run_configuration
import logging

from src.util.get_gutenberg import get_gutenberg


if __name__ == '__main__':
    run_configuration()

    logging.info('Launched main.py')
    # testing a few different ways to get GloVe embeddings
    # twitter glove
    embeddings = run_dataprep(embedding_type="glove_twitter", target_glove="glove.twitter.27B.200d.pickle")

    # the next two should be the equal to each other

    # our custom functions to download and prep glove
    # embeddings = run_dataprep(embedding_type="glove_common_crawl")
    # the same thing as above but with torchnlp taking care of the download
    # embeddings = run_dataprep(embedding_type="840B")

    # experimenting with a simple corpus, a book, from project gutenberg
    # get_gutenberg()
