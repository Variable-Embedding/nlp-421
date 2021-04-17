
from src.models.rnn_model import Model
from src.stages.stage_train_rnn_model import stage_train_rnn_model
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

def run_rnn_experiment(epochs=1, **nn_data):
    """
    """
    stages = nn_data.keys()
    logging.info(f'Preparing experiment for {stages}.')

    model = Model(**nn_data['train'])

    train_data = nn_data['train']["tokens"]
    valid_data = nn_data['valid']["tokens"]
    test_data = nn_data['test']["tokens"]

    report_model_parameters(model, train_data)

    train_set = LanguageModelSequence(model=model, data=train_data)
    train_dataloader = DataLoader(train_set, batch_size=model.batch_size, shuffle=False)

    valid_set = LanguageModelSequence(model=model, data=valid_data)
    valid_dataloader = DataLoader(valid_set, batch_size=model.batch_size, shuffle=False)
    test_set = LanguageModelSequence(model=model, data=test_data)
    test_dataloader = DataLoader(test_set, batch_size=model.batch_size, shuffle=False)

    logging.info('Starting Training')
    train_perplexity = []
    total_epochs = tqdm(range(epochs), desc="Training Progress")
    for epoch in total_epochs:
        epoch_loss = train_epoch(model=model, epoch=epoch, train_dataloader=train_dataloader)
        train_perplexity.append(np.exp(np.mean(epoch_loss)))

    return True


def train_epoch(model, train_dataloader, epoch, learning_rate=1, learning_rate_decay=1):
    model.train()


    # states = generate_initial_states(model)

    epoch_loss = []
    epoch_progress = tqdm(train_dataloader, desc=f'EPOCH: {epoch}', position=0, leave=True)

    for idx, (x, y) in enumerate(epoch_progress):
        model.init_hidden()
        model.zero_grad()
        output = model(x)
        output = output.to(model.device)
        loss = loss_function(output, y)

        batch_loss = loss.item() / model.batch_size
        epoch_progress.set_description('EPOCH: {} - Loss: {:.2f}'.format(epoch, batch_loss))
        epoch_progress.refresh()
        epoch_loss.append(batch_loss)
        loss.backward()

        with torch.no_grad():
            norm = nn.utils.clip_grad_norm_(model.parameters(), model.max_norm)
            for param in model.parameters():
                lr = learning_rate * (learning_rate_decay ** epoch)
                param -= lr * param.grad

    return epoch_loss


def generate_initial_states(model, batch_size=None):
    """Helper function to generate initial state needed for the model.forward function

    Args:
        model: model for which the states are initialized.
        batch_size: the batch size for states. If None will use model.batch_size.

    Returns:
        A list of tuples of torch.array.
    """
    if batch_size is None:
        batch_size = model.batch_size

    return (torch.zeros(model.num_layers, batch_size, model.embedding_size, device=model.device),
            torch.zeros(model.num_layers, batch_size, model.embedding_size, device=model.device))

def detach_states(states):
    """Helper function for detaching the states.

    Args:
        states: states to detach.

    Returns:
        List of detached states.
    """
    h, c = states
    return (h.detach(), c.detach())

class LanguageModelSequence(Dataset):
    def __init__(self, model, data, sequence_length=None, sequence_step_size=None):

        self.data = data

        self.sequence_length = model.sequence_length if sequence_length is None else sequence_length
        self.sequence_step_size = model.sequence_step_size if sequence_step_size is None else sequence_step_size

        self.model=model

    def __getitem__(self, index):
        x = self.data[index: index + self.sequence_length]

        y = self.data[index + self.sequence_length: index + self.sequence_length + self.sequence_step_size]

        return torch.tensor(x, dtype=torch.long).to(self.model.device), torch.tensor(y, dtype=torch.long).to(self.model.device)

    def __len__(self):
        return len(self.data)


def report_model_parameters(model, train_data):

    num_iters = len(train_data) // model.batch_size // model.sequence_step_size
    num_iters += len(train_data) // model.batch_size // model.sequence_length

    num_parameters = sum([np.prod(p.size()) for p in model.parameters()])
    logging.info("Number of model parameters: {}".format(num_parameters))


def loss_function(output, target):
    """Loss function for the model.

    Args:
        output: the output of the model.
        target: the expected tokens.

    Returns:
        Loss.
    """
    return F.cross_entropy(output.reshape(-1, output.size(2)), target.reshape(-1)) * target.size(1)
