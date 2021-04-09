from src.runners.run_dataprep import run_dataprep
from src.util.config import run_configuration
import logging


if __name__ == '__main__':
    run_configuration()

    logging.info('Launched main.py')
    run_dataprep(embedding_type="glove_twitter", embedding_version="glove.twitter.27B.25d.txt")


