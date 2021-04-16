from src.runners.run_dataprep import run_dataprep
from src.runners.run_rnn_experiment import run_rnn_experiment
from src.util.config import run_configuration
import logging


if __name__ == '__main__':
    run_configuration()
    logging.info('Launched main.py')

    # our custom functions to prep toy dataset and glove embeddings
    nn_data = run_dataprep(embedding_type="glove_common_crawl", corpus_type="WikiText2")
    result = run_rnn_experiment(**nn_data)






