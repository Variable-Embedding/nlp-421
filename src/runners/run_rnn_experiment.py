
from src.models.rnn_model import Model


def run_rnn_experiment(**kwargs):
    """
    """

    stages = kwargs.keys()

    model = Model(**kwargs['train'])

    # TODO: Build out training workflow
    return True
