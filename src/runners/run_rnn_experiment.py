
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

    num_iters = report_model_parameters(model, train_data)

    # TODO: fix dataloader and dataset class, use the code below with those objects
    # train_set = LanguageModelSequence(model=model, data=train_data)
    # train_dataloader = DataLoader(train_set, batch_size=model.batch_size, shuffle=False)
    # valid_set = LanguageModelSequence(model=model, data=valid_data)
    # valid_dataloader = DataLoader(valid_set, batch_size=model.batch_size, shuffle=False)
    # test_set = LanguageModelSequence(model=model, data=test_data)
    # test_dataloader = DataLoader(test_set, batch_size=model.batch_size, shuffle=False)

    logging.info('Starting Training')
    train_perplexity = []
    total_epochs = tqdm(range(epochs), desc="Training Progress")
    for epoch in total_epochs:
        epoch_loss = train_epoch(model=model, curr_epoch=epoch, total_epochs=epochs, tokens=train_data, num_iters=num_iters)
        train_perplexity.append(np.exp(np.mean(epoch_loss)))

    return True


def train_epoch(model, curr_epoch, total_epochs, num_iters, learning_rate=1, learning_rate_decay=1, tokens=None, train_dataloader=None, display_frequency=0.02):
    model.train()
    epoch_counter = curr_epoch+1
    epoch_loss = []
    # epoch_progress = tqdm(train_dataloader, desc=f'EPOCH: {epoch}', position=0, leave=True)

    display_interval = int(num_iters * display_frequency)
    logging.info(f'Updating Statistics every {display_interval} iterations.')

    epoch_progress = tqdm(batch_data(tokens=tokens, model=model)
                          , desc=f'EPOCH: {epoch_counter}', position=0, leave=True, total=num_iters * total_epochs)

    for idx, (x, y) in enumerate(epoch_progress):
        model.init_hidden()
        model.zero_grad()
        output = model(x)
        output = output.to(model.device)
        loss = loss_function(output, y)
        batch_loss = loss.item() / model.batch_size

        if idx == 0:
            curr_perplexity = np.exp(batch_loss)
            epoch_progress.set_description('EPOCH: {} - Start Perplexity: {:.2f} - Start Loss: {:.2f}'.format(epoch_counter, curr_perplexity, batch_loss))
            epoch_progress.refresh()

        if idx > 0 and idx % display_interval == 0:
            curr_perplexity = np.exp(batch_loss)
            epoch_progress.set_description('EPOCH: {} - Curr  Perplexity: {:.2f} - Loss: {:.2f}'.format(epoch_counter, curr_perplexity, batch_loss))
            epoch_progress.refresh()

        epoch_loss.append(batch_loss)
        loss.backward()

        with torch.no_grad():
            norm = nn.utils.clip_grad_norm_(model.parameters(), model.max_norm)
            for param in model.parameters():
                lr = learning_rate * (learning_rate_decay ** curr_epoch)
                param -= lr * param.grad

    return epoch_loss


def batch_data(tokens, model, batch_size=None, sequence_length=None, sequence_step_size=None, shuffle=False):
    """Helper function to batch the data.

    Args:
        tokens: the data to batch.
        model: the model to batch for.
        batch_size: the batch size, if None will use model.batch_size.
        sequence_step_size: the sequence step size.
        sequence_length: length of token sequence
        shuffle: Whether to shuffle the order of sequences.

    Returns:
        Iterator for batched data.
    """
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


class LanguageModelSequence(Dataset):
    # FIXME: Need to account for batch size when returning sequences, use batch_data() func for now.
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

    return num_iters


def loss_function(output, target):
    """Loss function for the model.

    Args:
        output: the output of the model.
        target: the expected tokens.

    Returns:
        Loss.
    """
    return F.cross_entropy(output.reshape(-1, output.size(2)), target.reshape(-1)) * target.size(1)
