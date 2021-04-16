
from src.models.rnn_model import Model
from src.stages.stage_train_rnn_model import stage_train_rnn_model
import logging
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


def run_rnn_experiment(**nn_data):
    """
    """

    stages = nn_data.keys()

    model = Model(**nn_data['train'])

    train_losses, valid_losses = train_model(model=model
                                             , train_tokens=nn_data['train']["corpus"]
                                             , valid_tokens=nn_data['valid']["corpus"]
                                             )

    # TODO: Build out training workflow
    return True


def train_model(model
                , train_tokens
                , valid_tokens=None
                , number_of_epochs=1
                , learning_rate=1
                , learning_rate_decay=1):

    model.to(model.device)
    step_size = model.sequence_step_size

    num_iters = len(train_tokens) // model.batch_size // model.sequence_step_size
    num_iters += len(train_tokens) // model.batch_size // model.sequence_length

    training_losses = []
    validation_losses = []

    num_parameters = sum([np.prod(p.size()) for p in model.parameters()])
    logging.info("Number of model parameters: {}".format(num_parameters))

    for epoch in range(number_of_epochs):
        t_losses = []
        model.train()
        # implement dataloader here
        batch_data(train_tokens, model)

    return None, None


def batch_data(tokens
               , model
               , batch_size=None
               , sequence_length=None
               , sequence_step_size=None
               , shuffle=False
               ):
    """Helper function to batch the data.

    Args:
        tokens: the data to batch.
        model: the model to batch for.
        batch_size: the batch size, if None will use model.batch_size.
        sequence_step_size: the sequence step size.
        shuffle: Whether to shuffle the order of sequences.

    Returns:
        Iterator for batched data.
    """
    print('poo')
    if batch_size is None:
        batch_size = model.batch_size
    if sequence_length is None:
        sequence_length = model.sequence_length
    if sequence_step_size is None:
        sequence_step_size = model.sequence_step_size

    data = torch.tensor(tokens, dtype=torch.int64)
    words_per_batch = data.size(0) // batch_size
    data = data[:words_per_batch * batch_size]
    data = data.view(batch_size, -1)

    sequence_start_list = list(range(0, data.size(1) - sequence_length - 1, sequence_step_size))
    if shuffle:
        np.random.shuffle(sequence_start_list)

    for sequence_start in sequence_start_list:
        sequence_end = sequence_start + sequence_length
        prefix = data[:,sequence_start:sequence_end].transpose(1, 0).to(model.device)
        target = data[:,sequence_start + 1:sequence_end + 1].transpose(1, 0).to(model.device)
        yield prefix, target
        del prefix
        del target
