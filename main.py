from src.runners.run_dataprep import run_dataprep
from src.runners.run_rnn_experiment import run_rnn_experiment
from src.util.config import run_configuration
import logging
import torch

if __name__ == '__main__':
    run_configuration()
    logging.info('Launched main.py')

    # data pipeline to return GloVe emebddings and WikiText corpus for nn training
    nn_data = run_dataprep(embedding_type="glove_common_crawl", corpus_type="WikiText2")

    # for performance testing, run with and without gpu, if available
    if torch.cuda.is_available():
        device = torch.device("cuda")
        # run the experiment with and without multiprocessing and with/without gpu/cpu
        run_rnn_experiment(**nn_data, enable_mp=False, device="gpu")
        run_rnn_experiment(**nn_data, enable_mp=True, device="gpu")
        run_rnn_experiment(**nn_data, enable_mp=False, device="cpu")
        run_rnn_experiment(**nn_data, enable_mp=True, device="cpu")
    else:
        # run the experiment with and without multiprocessing if just CPU
        run_rnn_experiment(**nn_data, enable_mp=False)
        run_rnn_experiment(**nn_data, enable_mp=True)








