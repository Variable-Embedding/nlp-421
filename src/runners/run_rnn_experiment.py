
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

    return None, None

